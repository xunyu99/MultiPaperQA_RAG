"""sessions / messages 两张表：多轮会话（Step 8c）。

设计上的几个决定：

1. **和论文数据完全解耦**。`sessions` 里没有指向 `papers` 的外键 —— 删一篇论文
   不该让历史会话跟着消失（对话记录是用户资产）。范围字段里存的是**记下来的 id**，
   查到就用，查不到（论文被下架了）就退回全库。
2. **消息靠 `order_index` 排序，不靠时间戳**。同一秒内写入 user + assistant 两条是
   常态，按时间排会抖；`(session_id, order_index)` 上有唯一约束，顺序是确定的。
   读历史时也**只取 user/assistant 成对的**（半截的 assistant 不会出现，见下面的
   `append_exchange`）。
3. **一轮问答是一个事务**。要么 user + assistant 两条都在，要么都不在 ——
   不然前端刷新会看到"有问没答"的空轮次。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from app.db.repositories._util import loads_json

TABLE = "sessions"
MESSAGE_TABLE = "messages"

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# 会话标题从第一个问题里截这么长（再长列表里也显示不下）
TITLE_LIMIT = 30


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    return f"s-{uuid.uuid4().hex[:16]}"


def make_title(question: str) -> str:
    """标题 = 问题的第一行前 30 字。不调 LLM 总结 —— 省一次调用，且够用。"""
    line = (question or "").strip().splitlines()[0] if question else ""
    clean = " ".join(line.split())
    return clean[:TITLE_LIMIT] if clean else "新会话"


def create(
    conn: sqlite3.Connection,
    *,
    session_id: str | None = None,
    title: str = "",
    paper_id: str | None = None,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    sid = session_id or new_session_id()
    now = _now()
    conn.execute(
        """
        INSERT INTO sessions (session_id, title, scope_paper_id, scope_paper_ids,
                              message_count, created_at, updated_at)
        VALUES (?, ?, ?, ?, 0, ?, ?)
        """,
        (sid, title, paper_id, _dump_ids(paper_ids), now, now),
    )
    return get(conn, sid)  # type: ignore[return-value]


def get(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    return _normalize(row) if row else None


def list_recent(
    conn: sqlite3.Connection,
    limit: int = 50,
    *,
    include_empty: bool = False,
) -> list[dict[str, Any]]:
    """历史会话，最近使用的在前。

    **默认不返回一条消息都没有的会话**：前端点"新建会话"时并不立刻落库（懒创建），
    但接口/调试里可能留下空壳；空会话出现在列表里只会让人以为"我建过一堆没用的东西"。
    """
    where = "" if include_empty else "WHERE message_count > 0"
    rows = conn.execute(
        f"SELECT * FROM sessions {where} ORDER BY updated_at DESC, session_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_normalize(row) for row in rows]


def rename(conn: sqlite3.Connection, session_id: str, title: str) -> None:
    conn.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
        (title[:TITLE_LIMIT], _now(), session_id),
    )


def set_scope(
    conn: sqlite3.Connection,
    session_id: str,
    paper_id: str | None,
    paper_ids: list[str] | None,
) -> None:
    """更新 sticky scope。**不改 updated_at** —— 它只跟"说过话"有关，跟改设置无关。"""
    conn.execute(
        "UPDATE sessions SET scope_paper_id = ?, scope_paper_ids = ? WHERE session_id = ?",
        (paper_id, _dump_ids(paper_ids), session_id),
    )


def delete(conn: sqlite3.Connection, session_id: str) -> int:
    """删会话。消息靠外键 ON DELETE CASCADE 一起清掉。"""
    return conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,)).rowcount


def append_exchange(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    question: str,
    answer: str,
    citations: dict[str, Any] | None = None,
) -> None:
    """把一轮问答写进去（user + assistant 两条，同一个事务由调用方包）。"""
    row = conn.execute(
        "SELECT COALESCE(MAX(order_index), 0) AS last FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    start = int(row["last"] or 0)
    now = _now()
    conn.executemany(
        """
        INSERT INTO messages (message_id, session_id, role, content, citations,
                              created_at, order_index)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (f"m-{uuid.uuid4().hex[:16]}", session_id, ROLE_USER, question, None, now, start + 1),
            (
                f"m-{uuid.uuid4().hex[:16]}",
                session_id,
                ROLE_ASSISTANT,
                answer,
                json.dumps(citations or {}, ensure_ascii=False),
                now,
                start + 2,
            ),
        ],
    )
    conn.execute(
        """
        UPDATE sessions
        SET message_count = message_count + 2,
            updated_at = ?,
            title = CASE WHEN title = '' THEN ? ELSE title END
        WHERE session_id = ?
        """,
        (now, make_title(question), session_id),
    )


def list_messages(conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY order_index",
        (session_id,),
    ).fetchall()
    return [
        {
            "message_id": row["message_id"],
            "role": row["role"],
            "content": row["content"],
            "citations": loads_json(row["citations"]) or {},
            "created_at": row["created_at"],
            "order_index": row["order_index"],
        }
        for row in rows
    ]


# 读历史时多取几条做缓冲：空/失败的助手消息会被丢掉，不多取就可能凑不满 rounds 轮
_HISTORY_BUFFER = 4


def history(conn: sqlite3.Connection, session_id: str, rounds: int = 2) -> list[dict[str, str]]:
    """给"指代改写"用的最近几轮，**倒序取再翻回来**（要的是最近几轮，不是最早的）。

    注意 `rounds` 是**轮数**不是消息条数：一轮 = 用户一条 + 助手一条，所以内部按
    `rounds * 2` 取消息。窗口大小的唯一来源是 `rewrite.HISTORY_ROUNDS`，
    调用方把那个常量传进来，别在这儿另写一个数。

    **空内容、以及标了生成失败的助手消息会被丢掉**。实测库里 9 条助手消息有 6 条
    是空的（流式一个 token 都没吐时把空串落了库），它们提供不了任何锚点，留着只会
    白占窗口 —— 所以上面多取 `_HISTORY_BUFFER` 条，丢掉之后仍能凑满 rounds 轮。
    """
    if rounds <= 0:
        return []
    rows = conn.execute(
        "SELECT role, content, citations FROM messages WHERE session_id = ?"
        " ORDER BY order_index DESC LIMIT ?",
        (session_id, rounds * 2 + _HISTORY_BUFFER),
    ).fetchall()
    kept = [row for row in rows if _usable(row)]
    kept.reverse()  # 上面是倒序取的，翻回时间顺序
    # 按"最近 rounds 个用户提问"切窗口，而不是按消息条数：过滤掉空的助手回复之后，
    # 按条数切会切出以助手回复开头的半截轮次（一条没有问题的孤立回答）。
    asked = [index for index, row in enumerate(kept) if row["role"] == ROLE_USER]
    if len(asked) > rounds:
        kept = kept[asked[-rounds] :]
    return [{"role": row["role"], "content": row["content"]} for row in kept]


def _usable(row: sqlite3.Row) -> bool:
    """这条消息能不能进改写历史。

    空白的、标了生成失败的、以及**闲聊轮**都不能：闲聊的助手回复（"你好！我是……"）
    没有任何论文话题可指代，塞进历史只会让下一轮的"它呢？"去找闲聊里的指代对象。
    """
    if not (row["content"] or "").strip():
        return False
    if row["role"] != ROLE_ASSISTANT:
        return True
    try:
        meta = loads_json(row["citations"])
    except ValueError:  # 手工写坏的 citations 不该让整轮问答挂掉
        return True
    return not (isinstance(meta, dict) and (meta.get("_failed") or meta.get("_chitchat")))


def _dump_ids(paper_ids: list[str] | None) -> str | None:
    return json.dumps(paper_ids, ensure_ascii=False) if paper_ids else None


def _normalize(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["scope_paper_ids"] = loads_json(data.get("scope_paper_ids")) or []
    return data
