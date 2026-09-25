"""blocks 表：原子块，永不切分的最小单位。

公式 / 表格 / 图片各自占一块，本体（LaTeX / HTML / 图片路径）存在这里，
`assets` 表另存一份用于挂载到 chunk。`bbox` 是 [x0, y0, x1, y1]，
入库前转成 JSON 字符串。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

from app.db.connection import count_rows
from app.db.repositories._util import as_batch, dumps_json, upsert_rows

TABLE = "blocks"

COLUMNS = (
    "block_id",
    "paper_id",
    "section_id",
    "order_index",
    "block_type",
    "heading_level",
    "text",
    "latex",
    "html",
    "caption",
    "page_idx",
    "bbox",
    "image_path",
)

_JSON_FIELDS = ("bbox",)

RowInput = Mapping[str, Any] | Iterable[Mapping[str, Any]]


def _normalize(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """把结构化字段转成 JSON 字符串，其余原样。"""
    if not any(field in row for field in _JSON_FIELDS):
        return row
    fixed = dict(row)
    for field in _JSON_FIELDS:
        if field in fixed:
            fixed[field] = dumps_json(fixed[field])
    return fixed


def upsert(conn: sqlite3.Connection, rows: RowInput) -> int:
    batch = [_normalize(row) for row in as_batch(rows)]
    return upsert_rows(conn, TABLE, batch, "block_id", COLUMNS)


def get(conn: sqlite3.Connection, block_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM blocks WHERE block_id = ?", (block_id,)).fetchone()


def list_by_paper(conn: sqlite3.Connection, paper_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM blocks WHERE paper_id = ? ORDER BY order_index",
        (paper_id,),
    ).fetchall()


def list_by_section(conn: sqlite3.Connection, section_id: str) -> list[sqlite3.Row]:
    """切分时按章节取块，所以这里必须按 order_index 排好，顺序即原文顺序。"""
    return conn.execute(
        "SELECT * FROM blocks WHERE section_id = ? ORDER BY order_index",
        (section_id,),
    ).fetchall()


def get_many(conn: sqlite3.Connection, block_ids: Iterable[str]) -> list[sqlite3.Row]:
    """按 id 批量取回。**顺序不保证**，调用方要自己按需要的顺序排。"""
    ids = list(block_ids)
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    return conn.execute(
        f"SELECT * FROM blocks WHERE block_id IN ({placeholders})",
        ids,
    ).fetchall()


def type_counts(conn: sqlite3.Connection, paper_id: str) -> dict[str, int]:
    """块类型分布。Step 2 的验收要打印它，用来肉眼判断解析是否正常。"""
    rows = conn.execute(
        "SELECT block_type, count(*) AS n FROM blocks WHERE paper_id = ? GROUP BY block_type",
        (paper_id,),
    ).fetchall()
    return {row["block_type"]: int(row["n"]) for row in rows}


def delete_by_paper(conn: sqlite3.Connection, paper_id: str) -> int:
    return conn.execute("DELETE FROM blocks WHERE paper_id = ?", (paper_id,)).rowcount


def count(conn: sqlite3.Connection, paper_id: str | None = None) -> int:
    if paper_id is None:
        return count_rows(conn, TABLE)
    return count_rows(conn, TABLE, "paper_id = ?", (paper_id,))
