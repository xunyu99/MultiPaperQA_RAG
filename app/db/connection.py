"""SQLite 连接层：唯一创建连接、初始化 schema 的地方。

约定（这几条是刻意的，改动前先想清楚）：

1. 每连接都开 `foreign_keys` —— SQLite 默认是关的，不开外键约束不会生效。
2. 开 WAL：并发读不阻塞写，单机反复入库时不用等锁。
3. `row_factory = sqlite3.Row`，取字段一律 `row["paper_id"]`，不用下标。
4. **读 schema.sql 必须显式 `encoding="utf-8"`** —— Windows 默认用 GBK 解码，
   文件里的中文注释会直接 `UnicodeDecodeError`（见 PLAN §7 踩坑 1/2）。

用法：
    from app.db.connection import connect, init_db

    conn = connect()        # 默认 storage/app.db
    init_db(conn)           # 建表，幂等
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.config import get_settings

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# 建库后应该存在的表（验收用）
EXPECTED_TABLES = (
    "assets",
    "blocks",
    "chunk_assets",
    "chunks",
    "chunks_fts",
    "messages",
    "papers",
    "parse_cache",
    "sessions",
    "sections",
)


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """打开连接并设好 pragma。传 `":memory:"` 可建内存库（测试用）。"""
    if db_path == ":memory:":
        target = ":memory:"
    else:
        path = Path(db_path or get_settings().db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = str(path)

    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """按 schema.sql 建表 + 建索引，幂等，随便调。

    **先补列再执行 schema.sql**：schema.sql 里 `CREATE INDEX ... (label_norm)`
    这种引用新列的语句，在"表已存在但还没这列"的老库上会先于补列执行而报错
    （实测 `no such column: label_norm`）。新库上 _migrate 找不到表会自动跳过。
    """
    _migrate(conn)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


# 列级升级：`CREATE TABLE IF NOT EXISTS` 对已存在的表什么都不做，所以 schema.sql
# 里新增/删掉的列不会自动反映到老库上（下面 verify_schema 只能**发现**问题，不能修）。
# 改 schema 的列就往这里追加一条，init_db 时缺的补上、多余的删掉。
_ADDED_COLUMNS = (
    ("assets", "label_norm", "TEXT"),
)
# 删列要 SQLite >= 3.35（本仓库实测 3.45）。纯多余的列才往这里放 —— 被索引
# 或约束引用的列删不掉。
_DROPPED_COLUMNS = (
    # 资产检索文本改成现算（assets.index_text_of），表里这份快照没有任何读点
    ("assets", "index_text"),
)


def _migrate(conn: sqlite3.Connection) -> list[str]:
    """把老库补到当前 schema，返回实际执行的 ALTER（空列表 = 本来就齐）。"""
    applied: list[str] = []
    for table, column, decl in _ADDED_COLUMNS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue  # 表还没建（schema.sql 会建）
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            applied.append(f"{table}.{column}")
    for table, column in _DROPPED_COLUMNS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column in existing:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
            applied.append(f"-{table}.{column}")
    return applied


# 每个探针查一遍"建表时最容易漏掉的列"。
# 为什么需要它：`CREATE TABLE IF NOT EXISTS` 对**已存在**的表什么都不做，
# 所以改了 schema.sql 之后老库不会自动升级，会一路跑到"no such column"才炸。
_SCHEMA_PROBES = (
    "SELECT paper_id, content_hash, parse_version, index_version FROM papers LIMIT 1",
    "SELECT section_id, section_type, page_start, page_end FROM sections LIMIT 1",
    "SELECT block_id, heading_level, block_type, caption, image_path FROM blocks LIMIT 1",
    "SELECT chunk_id, index_text, block_ids, chunk_type FROM chunks LIMIT 1",
    "SELECT chunk_id, index_text, paper_id, section_type FROM chunks_fts LIMIT 1",
    "SELECT asset_id, asset_type, raw_content, caption, label_norm FROM assets LIMIT 1",
    "SELECT chunk_id, asset_id, relation FROM chunk_assets LIMIT 1",
    "SELECT content_hash, mineru_version, paper_id, mineru_dir FROM parse_cache LIMIT 1",
    "SELECT session_id, title, scope_paper_id, scope_paper_ids, message_count FROM sessions LIMIT 1",
    "SELECT message_id, session_id, role, content, citations, order_index FROM messages LIMIT 1",
)


def verify_schema(conn: sqlite3.Connection) -> list[str]:
    """返回 schema 与当前代码不匹配的地方（空列表 = 没问题）。"""
    problems: list[str] = []
    for sql in _SCHEMA_PROBES:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError as exc:
            problems.append(f"{sql}  →  {exc}")
    return problems


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """一个事务块：正常提交，出错回滚。入库这种多表写入全部走它。"""
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def query_all(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def query_one(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
) -> sqlite3.Row | None:
    return conn.execute(sql, params).fetchone()


def table_names(conn: sqlite3.Connection) -> list[str]:
    """库里的业务表。

    排除两类：`sqlite_` 开头的内部表，以及 FTS5 为虚拟表自动建的影子表
    （`chunks_fts_data` / `_idx` / `_docsize` / `_config` 这些 —— 它们在
    sqlite_master 里也是 type='table'，但不是业务表）。
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'chunks_fts_%' "
        "ORDER BY name"
    ).fetchall()
    return [row["name"] for row in rows]


def count_rows(conn: sqlite3.Connection, table: str, where: str = "", params: Sequence[Any] = ()) -> int:
    sql = f"SELECT count(*) AS n FROM {table}"
    if where:
        sql += f" WHERE {where}"
    row = conn.execute(sql, params).fetchone()
    return int(row["n"])
