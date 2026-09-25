"""Step 1 验收：repositories 的增删查，以及 schema 上的约束真的生效。

跑：
    .venv\\Scripts\\python.exe -m pytest

这里只用内存库，不碰 storage/app.db。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest

from app.db.connection import EXPECTED_TABLES, connect, init_db, table_names, verify_schema
from app.db.repositories import assets, blocks, chunks, papers, parse_cache, sections
from app.db.repositories._util import loads_json


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    connection = connect(":memory:")
    init_db(connection)
    yield connection
    connection.close()


# ----------------------------------------------------------------------
# 假数据：全部走同一个 helper，保证同一批写入的字段完全一致
# （upsert_rows 要求同一批 rows 的键一致，不一致会明确报错）
# ----------------------------------------------------------------------
def _paper(**kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "paper_id": "p1",
        "title": "论文标题",
        "citation": None,
        "pdf_path": None,
        "mineru_dir": None,
        "status": papers.STATUS_PENDING,
    }
    row.update(kw)
    return row


def _section(**kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "section_id": "s1",
        "paper_id": "p1",
        "parent_id": None,
        "level": 1,
        "order_index": 0,
        "title": "Abstract",
        "section_type": "abstract",
        "page_start": 1,
        "page_end": 1,
    }
    row.update(kw)
    return row


def _block(**kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "block_id": "b1",
        "paper_id": "p1",
        "section_id": "s1",
        "order_index": 0,
        "block_type": "text",
        "text": None,
        "latex": None,
        "html": None,
        "caption": None,
        "page_idx": None,
        "bbox": None,
        "image_path": None,
    }
    row.update(kw)
    return row


def _chunk(**kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "chunk_id": "c1",
        "paper_id": "p1",
        "order_index": 0,
        "content": "正文",
        "index_text": "[论文标题 > Abstract] 正文",
        "token_count": 10,
        "block_ids": ["b1"],
        "page_start": 1,
        "page_end": 1,
        "chunk_type": "text",
    }
    row.update(kw)
    return row


def _asset(**kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "asset_id": "a1",
        "block_id": "b2",
        "paper_id": "p1",
        "asset_type": assets.ASSET_TABLE,
        "page_idx": 5,
        "bbox": None,
        "raw_content": None,
        "image_path": None,
        "caption": None,
        "label_norm": None,
    }
    row.update(kw)
    return row


def _seed(conn: sqlite3.Connection) -> None:
    """3 篇论文，覆盖 text / table / equation 三种块和 table / formula 两种资产。"""
    papers.upsert(
        conn,
        [
            _paper(paper_id="p1", title="论文A", citation="张三 · 2024", status="parsed"),
            _paper(paper_id="p2", title="论文B", citation="李四 · 2023", status="parsed"),
            _paper(paper_id="p3", title="论文C", citation="王五 · 2022"),
        ],
    )
    sections.upsert(
        conn,
        [
            _section(section_id="s1", paper_id="p1", order_index=0, title="Abstract"),
            _section(
                section_id="s2",
                paper_id="p1",
                order_index=1,
                title="3 Method",
                section_type="method",
            ),
            _section(
                section_id="s3",
                paper_id="p2",
                order_index=0,
                title="4 Experiments",
                section_type="experiment",
            ),
        ],
    )
    blocks.upsert(
        conn,
        [
            _block(block_id="b1", paper_id="p1", section_id="s1", order_index=0),
            _block(
                block_id="b2",
                paper_id="p1",
                section_id="s2",
                order_index=1,
                block_type="table",
                caption="Table 1: 主实验结果",
                html="<table><tr><td>0.85</td></tr></table>",
                page_idx=5,
                bbox=[10, 20, 30, 40],
            ),
            _block(
                block_id="b3",
                paper_id="p1",
                section_id="s2",
                order_index=2,
                block_type="equation",
                latex=r"E = mc^2",
                page_idx=6,
            ),
            _block(
                block_id="b4",
                paper_id="p2",
                section_id="s3",
                order_index=0,
                text="我们在 ImageNet 上做了实验",
                page_idx=3,
            ),
        ],
    )
    chunks.upsert(
        conn,
        [
            _chunk(chunk_id="c1", paper_id="p1", order_index=0, block_ids=["b1"]),
            _chunk(
                chunk_id="c2",
                paper_id="p1",
                order_index=1,
                content="Table 1: 主实验结果",
                index_text="[论文A > 3 Method] Table 1: 主实验结果",
                block_ids=["b2", "b3"],
                page_start=5,
                page_end=6,
                chunk_type="table",
            ),
            _chunk(
                chunk_id="c3",
                paper_id="p2",
                order_index=0,
                content="我们在 ImageNet 上做了实验",
                index_text="[论文B > 4 Experiments] 我们在 ImageNet 上做了实验",
                block_ids=["b4"],
                page_start=3,
                page_end=3,
            ),
        ],
    )
    assets.upsert(
        conn,
        [
            _asset(
                asset_id="a1",
                block_id="b2",
                asset_type=assets.ASSET_TABLE,
                raw_content="<table><tr><td>0.85</td></tr></table>",
                caption="Table 1: 主实验结果",
                label_norm="table:1",
            ),
            _asset(
                asset_id="a2",
                block_id="b3",
                asset_type=assets.ASSET_FORMULA,
                page_idx=6,
                raw_content=r"E = mc^2",
            ),
        ],
    )
    assets.link_chunk(conn, "c2", ["a1", "a2"])


# ----------------------------------------------------------------------
# schema
# ----------------------------------------------------------------------
def test_schema_tables(conn: sqlite3.Connection) -> None:
    assert sorted(table_names(conn)) == sorted(EXPECTED_TABLES)


def test_init_db_is_idempotent(conn: sqlite3.Connection) -> None:
    init_db(conn)
    assert sorted(table_names(conn)) == sorted(EXPECTED_TABLES)


def test_init_db_adds_new_columns_to_an_old_db(tmp_path: Any) -> None:
    """老库缺 `assets.label_norm` 时，init_db 要补列，而不是等到运行时 `no such column`。

    `CREATE TABLE IF NOT EXISTS` 对已存在的表什么都不做，所以光改 schema.sql
    老库是不会升级的 —— 这就是 _migrate 存在的理由。
    """
    db_path = tmp_path / "old.db"
    connection = connect(str(db_path))
    connection.executescript(
        # 老版本的 assets 表：列齐，就是没有后来加上的 label_norm
        "CREATE TABLE assets ("
        " asset_id TEXT PRIMARY KEY, block_id TEXT NOT NULL, paper_id TEXT NOT NULL,"
        " asset_type TEXT NOT NULL, page_idx INTEGER, bbox TEXT, raw_content TEXT,"
        " image_path TEXT, caption TEXT, index_text TEXT);"
    )
    connection.commit()
    assert verify_schema(connection), "老表缺列，探针应该报出来"

    init_db(connection)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(assets)")}
    assert "label_norm" in columns
    assert "index_text" not in columns, "多余的列要一并删掉，别让 schema 和库各说各话"
    assert verify_schema(connection) == []
    connection.close()


# ----------------------------------------------------------------------
# 三种查询（Step 1 验收要求：按 paper_id / section_type / asset_type 各查出结果）
# ----------------------------------------------------------------------
def test_query_by_paper_id(conn: sqlite3.Connection) -> None:
    _seed(conn)

    assert papers.count(conn) == 3
    assert chunks.count(conn, "p1") == 2
    assert [row["chunk_id"] for row in chunks.list_by_paper(conn, "p1")] == ["c1", "c2"]
    assert blocks.count(conn, "p2") == 1

    # 论文 A 的 chunk 里不该出现论文 B 的任何东西
    assert all(row["paper_id"] == "p1" for row in chunks.list_by_paper(conn, "p1"))


def test_query_by_section_type(conn: sqlite3.Connection) -> None:
    _seed(conn)

    methods = sections.list_by_type(conn, "p1", "method")
    assert [row["section_id"] for row in methods] == ["s2"]
    assert sections.list_by_type(conn, "p1", "experiment") == []
    assert [row["section_id"] for row in sections.list_by_type(conn, "p2", "experiment")] == ["s3"]

    # 章节树按 order_index 返回，可以直接打印
    assert [row["title"] for row in sections.list_by_paper(conn, "p1")] == [
        "Abstract",
        "3 Method",
    ]


def test_query_by_asset_type(conn: sqlite3.Connection) -> None:
    _seed(conn)

    # 按 page_idx 排序：a1 在第 5 页，a2 在第 6 页
    assert [row["asset_id"] for row in assets.list_by_paper(conn, "p1")] == ["a1", "a2"]
    only_tables = assets.list_by_paper(conn, "p1", assets.ASSET_TABLE)
    assert [row["asset_id"] for row in only_tables] == ["a1"]
    assert assets.type_counts(conn, "p1") == {"table": 1, "formula": 1}
    assert assets.list_by_paper(conn, "p3") == []


def test_chunk_asset_binding(conn: sqlite3.Connection) -> None:
    _seed(conn)

    attached = assets.assets_of_chunk(conn, "c2")
    assert [row["asset_id"] for row in attached] == ["a1", "a2"]
    assert {row["relation"] for row in attached} == {"primary"}

    assert [row["chunk_id"] for row in assets.chunks_of_asset(conn, "a1")] == ["c2"]
    assert assets.binding_count(conn) == 2

    # 重复挂同一条不报错，也不产生第二行
    assets.link_chunk(conn, "c2", ["a1"])
    assert assets.binding_count(conn) == 2


def test_json_fields_roundtrip(conn: sqlite3.Connection) -> None:
    _seed(conn)

    block = blocks.get(conn, "b2")
    assert block is not None
    assert loads_json(block["bbox"]) == [10, 20, 30, 40]

    chunk = chunks.get(conn, "c2")
    assert chunk is not None
    assert loads_json(chunk["block_ids"]) == ["b2", "b3"]

    assert blocks.get(conn, "b1")["bbox"] is None


def test_block_type_counts(conn: sqlite3.Connection) -> None:
    _seed(conn)
    assert blocks.type_counts(conn, "p1") == {"text": 1, "table": 1, "equation": 1}


def test_get_many(conn: sqlite3.Connection) -> None:
    _seed(conn)
    assert {row["chunk_id"] for row in chunks.get_many(conn, ["c1", "c3"])} == {"c1", "c3"}
    assert chunks.get_many(conn, []) == []


# ----------------------------------------------------------------------
# 约束
# ----------------------------------------------------------------------
def test_reupsert_keeps_children_and_untouched_columns(conn: sqlite3.Connection) -> None:
    """重新 upsert 论文不能顺手把子表删掉。

    这是 `_util.upsert_rows` 刻意不用 INSERT OR REPLACE 的原因：
    REPLACE 是先删后插，会顺着 ON DELETE CASCADE 清空 sections / chunks / assets。
    """
    _seed(conn)
    before = papers.get(conn, "p1")
    assert before is not None

    papers.upsert(conn, {"paper_id": "p1", "title": "论文A（改名）"})

    after = papers.get(conn, "p1")
    assert after is not None
    assert after["title"] == "论文A（改名）"
    assert after["citation"] == "张三 · 2024"  # 没给的列保持原值
    assert after["status"] == "parsed"
    assert after["created_at"] == before["created_at"]

    assert sections.count(conn, "p1") == 2
    assert chunks.count(conn, "p1") == 2
    assert assets.count(conn) == 2
    assert assets.binding_count(conn) == 2


def test_delete_paper_cascades(conn: sqlite3.Connection) -> None:
    _seed(conn)

    # 外键没开的话这里不会级联，所以这个测试同时验证 PRAGMA foreign_keys
    assert papers.delete(conn, "p1") == 1

    assert sections.count(conn, "p1") == 0
    assert blocks.count(conn, "p1") == 0
    assert chunks.count(conn, "p1") == 0
    assert assets.list_by_paper(conn, "p1") == []
    assert assets.binding_count(conn) == 0

    # 论文 B 的东西必须还在
    assert chunks.count(conn, "p2") == 1
    assert blocks.count(conn, "p2") == 1


def test_delete_by_paper_for_reindex(conn: sqlite3.Connection) -> None:
    _seed(conn)

    assert chunks.delete_by_paper(conn, "p1") == 2
    assert blocks.delete_by_paper(conn, "p1") == 3
    assert sections.delete_by_paper(conn, "p1") == 2
    assert chunks.count(conn, "p1") == 0
    assert chunks.count(conn, "p2") == 1


def test_order_index_unique_per_paper(conn: sqlite3.Connection) -> None:
    _seed(conn)

    with pytest.raises(sqlite3.IntegrityError):
        blocks.upsert(
            conn,
            _block(block_id="b9", paper_id="p1", section_id="s1", order_index=0),
        )


def test_unknown_field_is_rejected(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="不认识的字段"):
        papers.upsert(conn, {**_paper(), "typo_field": 1})


def test_mixed_keys_in_one_batch_is_rejected(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="字段必须完全一致"):
        papers.upsert(
            conn,
            [
                {"paper_id": "p1", "title": "A"},
                {"paper_id": "p2", "title": "B", "authors": "某人"},
            ],
        )


def test_set_status(conn: sqlite3.Connection) -> None:
    _seed(conn)
    papers.set_status(conn, "p1", papers.STATUS_INDEXED)
    assert papers.get(conn, "p1")["status"] == papers.STATUS_INDEXED


def test_delete_returns_zero_for_missing(conn: sqlite3.Connection) -> None:
    assert papers.delete(conn, "不存在") == 0
    assert papers.get(conn, "不存在") is None


# ----------------------------------------------------------------------
# parse_cache：解析产物复用（删论文时它必须活下来）
# ----------------------------------------------------------------------
def _cache_row(**kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "content_hash": "hash-a",
        "mineru_version": "mineru:vlm|lang:ch|ocr:0|formula:1|table:1",
        "paper_id": "p1",
        "mineru_dir": "storage/mineru/p1",
    }
    row.update(kw)
    return row


def test_parse_cache_composite_key(conn: sqlite3.Connection) -> None:
    parse_cache.upsert(conn, _cache_row())
    # 同一份 PDF、不同解析配置 → 两条独立缓存
    parse_cache.upsert(conn, _cache_row(mineru_version="mineru:pipeline|lang:ch|ocr:0|formula:1|table:1"))
    assert parse_cache.count(conn) == 2

    hit = parse_cache.get(conn, "hash-a", "mineru:vlm|lang:ch|ocr:0|formula:1|table:1")
    assert hit is not None
    assert hit["mineru_dir"] == "storage/mineru/p1"
    assert parse_cache.get(conn, "hash-a", "不存在的配置") is None


def test_parse_cache_upsert_is_idempotent(conn: sqlite3.Connection) -> None:
    parse_cache.upsert(conn, _cache_row())
    parse_cache.upsert(conn, _cache_row(mineru_dir="storage/mineru/moved"))
    assert parse_cache.count(conn) == 1
    assert parse_cache.get(conn, "hash-a", "mineru:vlm|lang:ch|ocr:0|formula:1|table:1")["mineru_dir"] == (
        "storage/mineru/moved"
    )


def test_parse_cache_survives_paper_deletion(conn: sqlite3.Connection) -> None:
    """删论文要留着解析产物记录 —— 下次传同一份 PDF 就不必再花 MinerU 的钱。"""
    _seed(conn)
    parse_cache.upsert(conn, _cache_row(paper_id="p1"))

    assert papers.delete(conn, "p1") == 1

    assert papers.get(conn, "p1") is None
    assert parse_cache.count(conn) == 1
    assert parse_cache.get(conn, "hash-a", "mineru:vlm|lang:ch|ocr:0|formula:1|table:1")["paper_id"] == "p1"


def test_parse_cache_delete(conn: sqlite3.Connection) -> None:
    parse_cache.upsert(conn, _cache_row())
    assert parse_cache.delete(conn, "hash-a", "mineru:vlm|lang:ch|ocr:0|formula:1|table:1") == 1
    assert parse_cache.count(conn) == 0
