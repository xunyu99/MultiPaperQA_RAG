"""sections 表：章节树。

v1 里这张表不参与召回，只服务三个消费方（详见 PLAN §2）：
切分边界、`index_text` 的章节路径、证据卡的定位串。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

from app.db.connection import count_rows
from app.db.repositories._util import as_batch, upsert_rows

TABLE = "sections"

COLUMNS = (
    "section_id",
    "paper_id",
    "parent_id",
    "level",
    "order_index",
    "title",
    "section_type",
    "page_start",
    "page_end",
)

# section_type 的归一化取值。认不出来的标题一律落 other，不要留空 ——
# 空值会让以后"按类型定向"的查询悄悄漏掉章节。
SECTION_TYPES = (
    "abstract",
    "intro",
    "related",
    "method",
    "experiment",
    "conclusion",
    "reference",
    "other",
)

RowInput = Mapping[str, Any] | Iterable[Mapping[str, Any]]


def upsert(conn: sqlite3.Connection, rows: RowInput) -> int:
    return upsert_rows(conn, TABLE, as_batch(rows), "section_id", COLUMNS)


def get(conn: sqlite3.Connection, section_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM sections WHERE section_id = ?", (section_id,)).fetchone()


def list_by_paper(conn: sqlite3.Connection, paper_id: str) -> list[sqlite3.Row]:
    """按论文内顺序返回，直接可以打印成缩进树。"""
    return conn.execute(
        "SELECT * FROM sections WHERE paper_id = ? ORDER BY order_index",
        (paper_id,),
    ).fetchall()


def list_by_type(conn: sqlite3.Connection, paper_id: str, section_type: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM sections WHERE paper_id = ? AND section_type = ? ORDER BY order_index",
        (paper_id, section_type),
    ).fetchall()


def delete_by_paper(conn: sqlite3.Connection, paper_id: str) -> int:
    """重新解析一篇论文时，先把旧的章节树清掉再重建。"""
    return conn.execute("DELETE FROM sections WHERE paper_id = ?", (paper_id,)).rowcount


def count(conn: sqlite3.Connection, paper_id: str | None = None) -> int:
    if paper_id is None:
        return count_rows(conn, TABLE)
    return count_rows(conn, TABLE, "paper_id = ?", (paper_id,))
