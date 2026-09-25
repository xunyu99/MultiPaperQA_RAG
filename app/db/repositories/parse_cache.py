"""parse_cache 表：解析产物复用。

这张表存在的唯一理由是**省 MinerU 的钱和时间**：一篇 PDF 从上传到解析完要几分钟，
重复上传不该再花一次。判断依据是两件事一起看：

    content_hash  —— PDF 字节没变（同一份文件）
    parse_version —— 解析配置没变（MinerU 模型 / 语言 / OCR / 开关 / converter 版本）

两个都命中就能直接用本地 `mineru_dir` 里的 content_list.json 重建 blocks。

它**不挂外键**，删论文时刻意保留 —— 解析产物是"资产"，不是"这篇论文的从属数据"。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

from app.db.connection import count_rows
from app.db.repositories._util import as_batch, upsert_rows

TABLE = "parse_cache"

COLUMNS = (
    "content_hash",
    "mineru_version",
    "paper_id",
    "mineru_dir",
    "created_at",
)

# 主键是复合的：同一份文件在不同解析配置下算两条缓存
KEY = ("content_hash", "mineru_version")

RowInput = Mapping[str, Any] | Iterable[Mapping[str, Any]]


def upsert(conn: sqlite3.Connection, rows: RowInput) -> int:
    return upsert_rows(conn, TABLE, as_batch(rows), KEY, COLUMNS)


def get(conn: sqlite3.Connection, content_hash: str, mineru_version: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM parse_cache WHERE content_hash = ? AND mineru_version = ?",
        (content_hash, mineru_version),
    ).fetchone()


def list_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM parse_cache ORDER BY created_at DESC").fetchall()


def delete(conn: sqlite3.Connection, content_hash: str, mineru_version: str) -> int:
    """删缓存行。**注意它不会删磁盘上的 mineru_dir** —— 那要手动清。"""
    return conn.execute(
        "DELETE FROM parse_cache WHERE content_hash = ? AND mineru_version = ?",
        (content_hash, mineru_version),
    ).rowcount


def count(conn: sqlite3.Connection) -> int:
    return count_rows(conn, TABLE)
