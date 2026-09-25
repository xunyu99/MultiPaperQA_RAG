"""chunks 表：检索与生成单元。

`content` 给生成看，`index_text` 进向量，两者刻意不等 —— 见 PLAN §2 铁律 2。
`block_ids` 是 JSON 数组，保留"这个 chunk 由哪些原子块拼成"，重切分和调试都要用。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

from app.db.connection import count_rows
from app.db.repositories._util import as_batch, dumps_json, upsert_rows

TABLE = "chunks"

COLUMNS = (
    "chunk_id",
    "paper_id",
    "order_index",
    "content",
    "index_text",
    "token_count",
    "block_ids",
    "page_start",
    "page_end",
    "chunk_type",
)

_JSON_FIELDS = ("block_ids",)

RowInput = Mapping[str, Any] | Iterable[Mapping[str, Any]]


def _normalize(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if not any(field in row for field in _JSON_FIELDS):
        return row
    fixed = dict(row)
    for field in _JSON_FIELDS:
        if field in fixed:
            fixed[field] = dumps_json(fixed[field])
    return fixed


def upsert(conn: sqlite3.Connection, rows: RowInput) -> int:
    batch = [_normalize(row) for row in as_batch(rows)]
    return upsert_rows(conn, TABLE, batch, "chunk_id", COLUMNS)


def get(conn: sqlite3.Connection, chunk_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()


def list_by_paper(conn: sqlite3.Connection, paper_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE paper_id = ? ORDER BY order_index",
        (paper_id,),
    ).fetchall()


def get_many(conn: sqlite3.Connection, chunk_ids: Iterable[str]) -> list[sqlite3.Row]:
    """按 id 批量取回，顺序不保证 —— 检索链路的"回到事实源"这一步用它。"""
    ids = list(chunk_ids)
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    return conn.execute(
        f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
        ids,
    ).fetchall()


def delete_by_paper(conn: sqlite3.Connection, paper_id: str) -> int:
    return conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,)).rowcount


def count(conn: sqlite3.Connection, paper_id: str | None = None) -> int:
    if paper_id is None:
        return count_rows(conn, TABLE)
    return count_rows(conn, TABLE, "paper_id = ?", (paper_id,))
