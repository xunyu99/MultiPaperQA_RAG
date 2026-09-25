"""assets 表 + chunk_assets 绑定：表格 / 公式 / 图片。

两类数据分开看：

1. 资产本体（`assets`）：表格 HTML、公式 LaTeX、原图路径。**只作为附件进生成
   上下文，不进向量**。
2. 绑定（`chunk_assets`）：一个 chunk 可以挂多个资产，一个资产也可以被多个
   chunk 引用（跨页表格合并后就是这种情况）。`relation='primary'` 表示资产
   本体就落在那个 chunk 里。

v1 的检索是 chunk 级的，资产靠绑定关系被"连带带出"。BACKLOG 里的资产解析器
（「表 3」「第三个公式」）会直接查这张表。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

from app.db.connection import count_rows
from app.db.repositories._util import as_batch, dumps_json, upsert_rows

TABLE = "assets"

COLUMNS = (
    "asset_id",
    "block_id",
    "paper_id",
    "asset_type",
    "page_idx",
    "bbox",
    "raw_content",
    "image_path",
    "caption",
    "label_norm",
)

# 只有三种。MinerU 的 image 块如果只是插图没正文价值，不要写进来。
ASSET_TABLE = "table"
ASSET_FORMULA = "formula"
ASSET_FIGURE = "figure"
ASSET_TYPES = (ASSET_TABLE, ASSET_FORMULA, ASSET_FIGURE)

_JSON_FIELDS = ("bbox",)

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
    return upsert_rows(conn, TABLE, batch, "asset_id", COLUMNS)


def get(conn: sqlite3.Connection, asset_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()


def get_many(conn: sqlite3.Connection, asset_ids: Iterable[str]) -> list[sqlite3.Row]:
    """按 id 批量取，顺序不保证。空输入直接返回，不发 SQL。"""
    ids = list(asset_ids)
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    return conn.execute(
        f"SELECT * FROM assets WHERE asset_id IN ({placeholders})",
        ids,
    ).fetchall()


def list_by_paper(
    conn: sqlite3.Connection,
    paper_id: str,
    asset_type: str | None = None,
) -> list[sqlite3.Row]:
    """按论文取资产，可选按类型过滤（table / formula / figure）。"""
    if asset_type is None:
        return conn.execute(
            "SELECT * FROM assets WHERE paper_id = ? ORDER BY page_idx, asset_id",
            (paper_id,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM assets WHERE paper_id = ? AND asset_type = ? ORDER BY page_idx, asset_id",
        (paper_id, asset_type),
    ).fetchall()


def type_counts(conn: sqlite3.Connection, paper_id: str) -> dict[str, int]:
    rows = conn.execute(
        "SELECT asset_type, count(*) AS n FROM assets WHERE paper_id = ? GROUP BY asset_type",
        (paper_id,),
    ).fetchall()
    return {row["asset_type"]: int(row["n"]) for row in rows}


def list_by_block(conn: sqlite3.Connection, block_id: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM assets WHERE block_id = ?", (block_id,)).fetchall()


def delete_by_paper(conn: sqlite3.Connection, paper_id: str) -> int:
    """重新切分一篇论文前先清掉它的旧资产（chunk_assets 靠外键级联）。"""
    return conn.execute("DELETE FROM assets WHERE paper_id = ?", (paper_id,)).rowcount


# ----------------------------------------------------------------------
# chunk_assets 绑定
# ----------------------------------------------------------------------
def link_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    asset_ids: Iterable[str],
    relation: str = "primary",
) -> int:
    """把资产挂到 chunk 上，重复挂同一条不会报错（关系会被更新）。"""
    ids = list(asset_ids)
    if not ids:
        return 0
    conn.executemany(
        "INSERT INTO chunk_assets (chunk_id, asset_id, relation) VALUES (?, ?, ?) "
        "ON CONFLICT (chunk_id, asset_id) DO UPDATE SET relation = excluded.relation",
        [(chunk_id, asset_id, relation) for asset_id in ids],
    )
    return len(ids)


def unlink_chunk(conn: sqlite3.Connection, chunk_id: str, asset_ids: Iterable[str] | None = None) -> int:
    """解绑。重切分一个 chunk 前先调它，避免留孤儿绑定。"""
    if asset_ids is None:
        return conn.execute("DELETE FROM chunk_assets WHERE chunk_id = ?", (chunk_id,)).rowcount
    ids = list(asset_ids)
    if not ids:
        return 0
    placeholders = ", ".join("?" for _ in ids)
    return conn.execute(
        f"DELETE FROM chunk_assets WHERE chunk_id = ? AND asset_id IN ({placeholders})",
        [chunk_id, *ids],
    ).rowcount


def assets_of_chunk(conn: sqlite3.Connection, chunk_id: str) -> list[sqlite3.Row]:
    """取一个 chunk 挂着的资产本体（带 relation），生成上下文用这个。"""
    return conn.execute(
        "SELECT a.*, ca.relation FROM chunk_assets ca "
        "JOIN assets a ON a.asset_id = ca.asset_id "
        "WHERE ca.chunk_id = ? "
        "ORDER BY a.page_idx, a.asset_id",
        (chunk_id,),
    ).fetchall()


def chunks_of_asset(conn: sqlite3.Connection, asset_id: str) -> list[sqlite3.Row]:
    """反查：一个资产被哪些 chunk 引用。做资产解析器（BACKLOG B4）时用它定位。"""
    return conn.execute(
        "SELECT c.*, ca.relation FROM chunk_assets ca "
        "JOIN chunks c ON c.chunk_id = ca.chunk_id "
        "WHERE ca.asset_id = ? "
        "ORDER BY c.order_index",
        (asset_id,),
    ).fetchall()


def count(conn: sqlite3.Connection) -> int:
    return count_rows(conn, TABLE)


def binding_count(conn: sqlite3.Connection) -> int:
    return count_rows(conn, "chunk_assets")
