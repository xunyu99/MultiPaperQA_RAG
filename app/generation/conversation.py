"""多轮会话的编排层：改写 -> 定范围 -> 检索 -> 落库。

**为什么单独一层**：接口（`/chat`）和 CLI 脚本都要这套流程，而且它有好几个
必须按顺序做的判断（见下面的 `SCOPE` 链）。写进接口的 async 函数里就没人测得到了。

**范围解析链（sticky scope）** —— 按顺序问四个问题，第一个答上来的为准：

1. 前端显式给了范围（用户点了 chips）→ 用它，`source=explicit`
2. 改写后的问句里**字面**提到了论文名 → 用它，`source=question`
3. 这个会话上一轮用过什么范围 → 继承，`source=sticky`
4. 都没有 → 全库，`source=all`

第 3 条就是"sticky"：用户第一轮说"pe-clip 的方法"，第二轮问"它的消融实验呢"，
第二轮问句里没有论文名，但**不该退回全库** —— 那会让追问答到别的论文上。

注意第 2 条排在 sticky 前面：用户**改口**问另一篇时，问句本身是最可靠的信号。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.db.connection import transaction
from app.db.repositories import papers as papers_repo
from app.db.repositories import sessions as sessions_repo
from app.generation.intent import QueryIntent, parse_intent
from app.generation.rewrite import HISTORY_ROUNDS
from app.providers.llm import LLMClient
from app.retrieval.asset_target import resolve_targets
from app.retrieval.scope import resolve_scope

SCOPE_EXPLICIT = "explicit"
SCOPE_QUESTION = "question"
SCOPE_STICKY = "sticky"
SCOPE_ALL = "all"


@dataclass
class Turn:
    """一轮提问的准备结果。检索要用 `standalone` 和 `scope`，展示要用原始 `question`。"""

    question: str
    standalone: str
    scope: list[str]
    scope_source: str
    history: list[dict[str, str]] = field(default_factory=list)
    session_id: str | None = None
    # 入口小模型的结构化判断（见 intent.parse_intent）
    is_chitchat: bool = False
    # 资产定点注入：问句点名"表 3"时定位到的 chunk_id。空 = 让 retriever 用正则兜底
    injected: list[str] = field(default_factory=list)

    @property
    def rewritten(self) -> bool:
        return self.standalone != self.question


def prepare_turn(
    conn: sqlite3.Connection,
    question: str,
    *,
    session_id: str | None = None,
    paper_id: str | None = None,
    paper_ids: list[str] | None = None,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    enable_rewrite: bool = True,
) -> Turn:
    """把"用户这一句"变成"可以拿去检索的东西"。"""
    settings = settings or get_settings()
    session = sessions_repo.get(conn, session_id) if session_id else None
    # 窗口大小只有一个来源（rewrite.HISTORY_ROUNDS），这里别另写一个数
    history = (
        sessions_repo.history(conn, session_id, rounds=HISTORY_ROUNDS) if session_id else []
    )

    # 一次小模型调用拿全部入口理解：改写 + 闲聊判定 + 资产编号
    intent = parse_intent(
        question, history, llm=llm, settings=settings, enabled=enable_rewrite
    )
    standalone = intent.rewritten_query or question

    sticky = _sticky_scope(session) if session else []
    scope, source = resolve_scope_chain(
        conn,
        standalone,
        original=question,
        explicit_paper_id=paper_id,
        explicit_paper_ids=paper_ids,
        sticky=sticky,
    )

    # 把这一轮定下来的范围记回会话，供下一轮继承。**只有真正说到话才记**：
    # set_scope 不碰 updated_at，所以不会把会话"顶"到列表最前面。
    if session_id:
        with transaction(conn):
            sessions_repo.set_scope(
                conn,
                session_id,
                scope[0] if len(scope) == 1 else None,
                scope if len(scope) > 1 else None,
            )

    injected = _resolve_injected(conn, scope, intent)

    return Turn(
        question=question,
        standalone=standalone,
        scope=scope,
        scope_source=source,
        history=history,
        session_id=session_id,
        is_chitchat=intent.is_chitchat,
        injected=injected,
    )


def _resolve_injected(
    conn: sqlite3.Connection, scope: list[str], intent: QueryIntent
) -> list[str]:
    """结构化输出里的资产编号 → 本体所在 chunk。

    模型没点名（或点了个库里没有的编号）就返回空 —— 调用方把空当成"没注入"，
    让 retriever 的正则再兜一次底（"表 3"这种写法正则很准，模型只是更懂"第三张表"）。
    """
    if not intent.asset_type or not intent.asset_number:
        return []
    return resolve_targets(conn, scope, [(intent.asset_type, intent.asset_number)])


def resolve_scope_chain(
    conn: sqlite3.Connection,
    standalone: str,
    *,
    original: str | None = None,
    explicit_paper_id: str | None = None,
    explicit_paper_ids: list[str] | None = None,
    sticky: list[str] | None = None,
) -> tuple[list[str], str]:
    """范围链：**问句里点名的论文 > 面板勾选 > 上一轮的范围 > 全库**。

    第 1 条排在最前是刻意的：用户都写出论文名了，说明这一轮他改主意了，
    面板上剩的旧勾选不该把它压回去（前端收到 `evidence.scope` 后会把勾选一起换掉）。

    `explicit_paper_ids=[]`（**显式空列表**）表示"取消勾选 / 点了全库"—— 那是一个**决定**，
    不是"没给"，所以不允许 sticky 再把它粘回来。只有 `None`（没给）才轮到 sticky。

    `original` 传用户原话：**问句点名要在改写后和原话上都找一遍**。改写偶尔会把论文名
    "美化"掉（提示词要求保持原意，但模型不是 100% 听话），而用户已经说出口的名字
    比面板上的旧勾选可信。
    """
    known = [dict(row) for row in papers_repo.list_all(conn)]

    for text in (standalone, original or ""):
        if not text:
            continue
        hit = resolve_scope(text, known)
        if hit:
            return hit, SCOPE_QUESTION

    if explicit_paper_ids is not None or explicit_paper_id:
        if not explicit_paper_ids and not explicit_paper_id:
            return [], SCOPE_ALL  # 显式全库
        raw = list(explicit_paper_ids) if explicit_paper_ids else [str(explicit_paper_id)]
        return _normalize(raw, SCOPE_EXPLICIT, conn)

    if sticky:
        # 粘住的范围里可能有已经被下架的论文，过滤掉再决定
        alive = {paper["paper_id"] for paper in known}
        kept = [paper_id for paper_id in sticky if paper_id in alive]
        if kept:
            return _normalize(kept, SCOPE_STICKY, conn)
    return [], SCOPE_ALL


def _normalize(scope: list[str], source: str, conn: sqlite3.Connection) -> tuple[list[str], str]:
    """**"选满全部论文" 归一化成全库。**

    前端有个"全选"按钮，会把每篇都勾上。要是就这么传给检索层，5 篇会走多篇配额
    （每篇 `PER_PAPER_K` 条）一次吐 15 条证据 —— 而用户的意思是"不限定"，
    那就该走单次全库查询、按 `TOP_K` 取。两者在**检索行为**上不是一回事，
    所以在这里统一：范围覆盖了库里所有论文（且不止一篇）→ 当成全库。

    只对多篇生效：库里只有一篇时，"全部"就是那一篇，单篇和全库的取数逻辑本来就一样。
    """
    known_ids = {row["paper_id"] for row in papers_repo.list_all(conn)}
    if len(known_ids) > 1 and scope and set(scope) >= known_ids:
        return [], SCOPE_ALL
    return scope, source


def save_turn(
    conn: sqlite3.Connection,
    turn: Turn,
    *,
    answer: str,
    citations: dict[str, Any],
    cards: list[dict[str, Any]] | None = None,
) -> None:
    """一轮问答落库。user + assistant 两条在一个事务里（见 sessions 模块第 3 条）。

    `cards` 是**当时展示给用户的证据卡快照**（已截断）。为什么要存：刷新页面后
    历史消息如果只剩一句回答，引用就断了 —— 用户点 `[E2]` 找不到任何东西。
    它同时是一份"当时到底拿什么回答的"记录，比事后重新检索更可信。
    """
    if not turn.session_id:
        return
    with transaction(conn):
        sessions_repo.append_exchange(
            conn,
            turn.session_id,
            question=turn.question,
            answer=answer,
            citations={
                **(citations or {}),
                # 顺手把这一轮实际用的检索问句和范围记下来：
                # 以后排查"为什么答到别的论文上"时，这两个字段是关键证据
                "_standalone": turn.standalone,
                "_scope": turn.scope,
                "_scope_source": turn.scope_source,
                # 入口小模型的判断也留一份：事后能看出"是模型判成闲聊了"还是"检索没找到"
                "_chitchat": turn.is_chitchat,
                "_injected": turn.injected,
                "_cards": cards or [],
            },
        )


def _sticky_scope(session: dict[str, Any] | None) -> list[str]:
    if not session:
        return []
    ids = session.get("scope_paper_ids") or []
    if ids:
        return list(ids)
    single = session.get("scope_paper_id")
    return [single] if single else []
