"""关键词通道：SQLite FTS5 + trigram 分词器。

和向量通道**并列**的第二路召回，不是替代品。两路各自召回之后合并去重（可选 RRF）
再 rerank，见 retriever 的注释。

**为什么用 FTS5 而不是自己算 BM25**：`chunks.index_text` 本来就在 SQLite 里
（向量那边只是它的一份拷贝，存在 Chroma）。FTS5 跟 chunks 同库同进程、**能进同一个
事务** —— 索引和 chunks 永远一致，不需要额外的同步逻辑，也不需要多一个存储。

**分词器为什么是 trigram（不是默认的 unicode61）**，这条是实测出来的：

- 默认的 `unicode61` 对中文**完全无效**：一整句中文会被当成**一个** token，
  查"微表情"永远 0 命中（实测连 6 个字的整句也查不到）。
- trigram 按 3 字符滑窗建索引，中英通吃："PE-CLIP"、"3.1"、中文词都能查；
  英文还天然带子串匹配（查 "recognit" 能命中 "recognition"，不需要词干化）。
- 代价有两条，都是硬边界：**短于 3 个字符的词查不到**（"F1"、"AI"、"识别"
  会静默返回 0 条，不报错）；索引比 unicode61 大几倍（这个量级无所谓）。
  编号类查询（"表 3"）本来就不走这条路 —— 那是 `assets.label_norm` 定向的活。

**查询必须 sanitize**：用户问句里的引号、括号、问号会让 FTS5 直接 `syntax error`
（实测被这个打断过）。`to_match_query` 把问句切成词、每词单独加引号、用 OR 连接。
**不用 AND** —— 这条路是候选生成，召回优先，精度交给后面的 rerank；用 AND 会把
召回打到接近 0。
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TABLE = "chunks_fts"

# 词元：英文/数字串（允许 . _ + -，方法名里常见：PE-CLIP、GPT-4o、3.1），
# 或者一段连续中文。
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]*|[\u4e00-\u9fff]+")

# trigram 的硬下限：短于 3 个字符的词元一个都匹配不到。
# 提前丢掉，而不是发一个必然为空的查询。
MIN_TOKEN = 3


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: str
    similarity: float  # = -bm25()，**越大越相关**（跟向量通道同一个方向）
    paper_id: str | None = None


def to_match_query(query: str) -> str | None:
    """自然语言问句 → FTS5 的 MATCH 表达式。返回 None = 这次没有可查的词。

    每个词单独加引号有两个作用：挡住词里的 `* ( ) ^ :` 这些 FTS5 保留字符；
    trigram 下引号表示"这段字符要连着出现"，正是我们想要的短语语义。

    **整串也会作为一个短语子句**：用户从 PDF 里复制的往往就是一个术语短语，
    实测 `adapter tuning` 整串只出现在 1 个 chunk，而按词 OR 会命中 49 个 ——
    不挂这个子句，"短语唯一"这个最大的优势就被自己抹掉了。FTS5 的 OR 会把
    "短语和单词都命中"的行排在前面（它命中的子句更多），所以不用额外排序。
    长问句的整串短语通常命中 0 条，等于白挂一个子句，无害。
    """
    raw = " ".join((query or "").split())
    tokens = [token for token in _TOKEN.findall(raw) if len(token) >= MIN_TOKEN]
    if not tokens:
        return None
    clauses = [f'"{token}"' for token in tokens]
    phrase = raw.replace('"', " ").strip()
    if len(tokens) > 1 and phrase:
        clauses.insert(0, f'"{phrase}"')
    return " OR ".join(clauses)


# ----------------------------------------------------------------------
# 写入（跟 chunks 在同一个事务里调用）
# ----------------------------------------------------------------------
def delete_paper(conn: sqlite3.Connection, paper_id: str) -> int:
    """删掉这篇论文的 FTS 行。重索引时先删后写，幂等的关键。"""
    cursor = conn.execute(f"DELETE FROM {TABLE} WHERE paper_id = ?", (paper_id,))
    return cursor.rowcount


def replace_paper(
    conn: sqlite3.Connection,
    paper_id: str,
    items: Sequence[Mapping[str, Any]],
) -> int:
    """按论文重建 FTS 索引，返回写入条数。

    `items` 跟 `vector_index.add_chunks` 收的是**同一个东西**
    （`indexer._vector_items` 的产物）：{chunk_id, index_text, metadata}。
    两边共用一份，别在这儿另拼一次文本 —— 那样迟早不一致。
    """
    delete_paper(conn, paper_id)
    rows = []
    for item in items:
        text = (item.get("index_text") or "").strip()
        if not text:
            continue
        meta = item.get("metadata") or {}
        rows.append(
            (
                item["chunk_id"],
                text,
                meta.get("paper_id") or paper_id,
                meta.get("section_type"),
                meta.get("chunk_type"),
            )
        )
    if rows:
        conn.executemany(
            f"INSERT INTO {TABLE} (chunk_id, index_text, paper_id, section_type, chunk_type)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


# ----------------------------------------------------------------------
# 查询
# ----------------------------------------------------------------------
def search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 5,
    *,
    paper_ids: Sequence[str] | None = None,
    section_type: str | None = None,
) -> list[KeywordHit]:
    """关键词召回。返回按相关度排序的 hit，**越大越相关**。

    `paper_ids` / `section_type` 在 FTS 表里是冗余列，直接进 WHERE ——
    先过滤再排序，比 MATCH 完再回 chunks 表 JOIN 快。
    """
    match = to_match_query(query)
    if match is None:
        return []

    sql = [f"SELECT chunk_id, paper_id, bm25({TABLE}) AS score FROM {TABLE} WHERE {TABLE} MATCH ?"]
    params: list[Any] = [match]
    if paper_ids:
        placeholders = ", ".join("?" for _ in paper_ids)
        sql.append(f"AND paper_id IN ({placeholders})")
        params.extend(paper_ids)
    if section_type:
        sql.append("AND section_type = ?")
        params.append(section_type)
    # bm25() 返回的是**负数**，越小越相关（跟向量的距离一个方向），所以升序
    sql.append("ORDER BY score LIMIT ?")
    params.append(k)

    return [
        KeywordHit(
            chunk_id=row["chunk_id"],
            similarity=-float(row["score"]),
            paper_id=row["paper_id"],
        )
        for row in conn.execute(" ".join(sql), params)
    ]


def count(conn: sqlite3.Connection, paper_id: str | None = None) -> int:
    """FTS 里的行数。用来跟 `chunks` 对账（升级后没重建索引会明显对不上）。"""
    if paper_id:
        row = conn.execute(
            f"SELECT count(*) AS n FROM {TABLE} WHERE paper_id = ?", (paper_id,)
        ).fetchone()
    else:
        row = conn.execute(f"SELECT count(*) AS n FROM {TABLE}").fetchone()
    return int(row["n"])
