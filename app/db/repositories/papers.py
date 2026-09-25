"""papers 表：论文元信息。

写入走 `upsert`（主键冲突就更新），不提供 `insert` —— 重复入库同一篇论文
必须是安全的，详见 `_util.upsert_rows` 的注释。

注意两件事：

1. **必须带上所有 NOT NULL 且没有默认值的列**（`title` 就是）。SQLite 的
   `INSERT ... ON CONFLICT DO UPDATE` 是**先试着插、再处理唯一冲突**，
   NOT NULL 检查发生在冲突处理之前 —— 所以只传 paper_id + index_version 会
   直接报 `NOT NULL constraint failed`，根本走不到 UPDATE 分支。
2. 给全之后，**没给的列保持原值**，不会被冲成 NULL（这是 ON CONFLICT DO UPDATE
   相对 INSERT OR REPLACE 的关键差别）。

所以要"只改一个字段"时，用下面的 `set_status` / `set_indexed` 这类专用函数，
不要拿 upsert 当部分更新用。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

from app.db.connection import count_rows
from app.db.repositories._util import as_batch, upsert_rows

TABLE = "papers"

COLUMNS = (
    "paper_id",
    "title",
    "citation",
    "pdf_path",
    "mineru_dir",
    "content_hash",
    "parse_version",
    "index_version",
    "status",
    "created_at",
)

# 状态流转：pending（已登记未解析）→ parsed（结构化块已入库）→ indexed（向量已写好）
# 任何一步抛异常就置 failed，下次重跑从该步继续。
STATUS_PENDING = "pending"
STATUS_PARSED = "parsed"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"

RowInput = Mapping[str, Any] | Iterable[Mapping[str, Any]]


def upsert(conn: sqlite3.Connection, rows: RowInput) -> int:
    return upsert_rows(conn, TABLE, as_batch(rows), "paper_id", COLUMNS)


def get(conn: sqlite3.Connection, paper_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()


def list_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM papers ORDER BY created_at, paper_id").fetchall()


def set_status(conn: sqlite3.Connection, paper_id: str, status: str) -> None:
    conn.execute("UPDATE papers SET status = ? WHERE paper_id = ?", (status, paper_id))


def set_indexed(conn: sqlite3.Connection, paper_id: str, index_version: str) -> None:
    """切分入库后标记：记下本次用的切分指纹，状态置 indexed。"""
    conn.execute(
        "UPDATE papers SET index_version = ?, status = ? WHERE paper_id = ?",
        (index_version, STATUS_INDEXED, paper_id),
    )


def delete(conn: sqlite3.Connection, paper_id: str) -> int:
    """删论文。子表靠外键 ON DELETE CASCADE 一起清掉，不用手写级联。"""
    return conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,)).rowcount


def count(conn: sqlite3.Connection) -> int:
    return count_rows(conn, TABLE)
