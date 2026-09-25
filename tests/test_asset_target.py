"""资产定点注入：问句里点名"表 3"时，把它本体所在的 chunk 直接顶到前面。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.db.connection import connect, init_db, transaction
from app.db.repositories import assets as assets_repo
from app.db.repositories import blocks as blocks_repo
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import papers as papers_repo
from app.retrieval import asset_target, keyword_index
from app.retrieval.retriever import retrieve
from app.retrieval.vector_index import ChunkVectorIndex


class _Embedder(Embeddings):
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.mapping[text]


# ---------------------------------------------------------------- 解析
def test_parse_targets_reads_explicit_numbers() -> None:
    assert asset_target.parse_targets("PE-CLIP 的表 3 里写了什么") == [("table", "3")]
    assert asset_target.parse_targets("Table 2 和 Figure 1 说明了什么") == [
        ("table", "2"),
        ("figure", "1"),
    ]
    assert asset_target.parse_targets("表 VIII 的结果") == [("table", "VIII")]


def test_parse_targets_ignores_words_containing_the_label() -> None:
    """跟正文提及共用一套正则：不认识的写法就返回空，绝不硬猜。"""
    assert asset_target.parse_targets("第三张表说明了什么") == []
    assert asset_target.parse_targets("a notable 8.68% improvement") == []


# ---------------------------------------------------------------- 定位
def _setup(tmp_path: Path):
    conn = connect(":memory:")
    init_db(conn)
    with transaction(conn):
        papers_repo.upsert(conn, {"paper_id": "p1", "title": "P1"})
        blocks_repo.upsert(
            conn,
            [
                {"block_id": "p1:b1", "paper_id": "p1", "section_id": None, "order_index": 1,
                 "block_type": "text", "heading_level": None, "text": "正文", "latex": None,
                 "html": None, "caption": None, "page_idx": 1, "bbox": None, "image_path": None},
                {"block_id": "p1:b2", "paper_id": "p1", "section_id": None, "order_index": 2,
                 "block_type": "table", "heading_level": None, "text": None, "latex": None,
                 "html": "<table><tr><td>x</td></tr></table>", "caption": "Table 1: 结果",
                 "page_idx": 2, "bbox": None, "image_path": None},
            ],
        )
        chunks_repo.upsert(
            conn,
            [
                {"chunk_id": "p1:c0001", "paper_id": "p1", "order_index": 1, "content": "正文一",
                 "index_text": "正文一", "block_ids": ["p1:b1"], "page_start": 1, "page_end": 1,
                 "chunk_type": "text"},
                {"chunk_id": "p1:c0002", "paper_id": "p1", "order_index": 2,
                 "content": "表格块 [TABLE_REF asset_id=p1:table0001]", "index_text": "Table 1: 结果",
                 "block_ids": ["p1:b2"], "page_start": 2, "page_end": 2, "chunk_type": "table"},
            ],
        )
        assets_repo.upsert(
            conn,
            [{"asset_id": "p1:table0001", "block_id": "p1:b2", "paper_id": "p1",
              "asset_type": "table", "caption": "Table 1: 结果", "label_norm": "table:1",
              "raw_content": "<table><tr><td>x</td></tr></table>", "page_idx": 2}],
        )
        keyword_index.replace_paper(conn, "p1", [{"chunk_id": "p1:c0001", "index_text": "正文一",
                                                  "metadata": {"paper_id": "p1"}}])
    settings = Settings(storage_dir=tmp_path, rerank_enabled=False, top_k=2)
    # 向量侧让 c0001 明显更近 —— 不注入的话 c0002 排在后面
    index = ChunkVectorIndex(
        settings,
        embedder=_Embedder(
            {
                "正文一": [1.0, 0.0],
                "Table 1: 结果": [0.2, 1.0],
                "表 1 里写了什么": [1.0, 0.0],
                "随便问点什么": [1.0, 0.0],
            }
        ),
        collection_name="asset_probe",
    )
    index.add_chunks(
        [
            {"chunk_id": row["chunk_id"], "index_text": row["index_text"],
             "metadata": {"paper_id": row["paper_id"]}}
            for row in conn.execute("SELECT * FROM chunks")
        ]
    )
    return conn, settings, index


def test_resolve_targets_finds_the_chunk_holding_the_block(tmp_path: Path) -> None:
    conn, _, _ = _setup(tmp_path)
    assert asset_target.resolve_targets(conn, ["p1"], [("table", "1")]) == ["p1:c0002"]
    # 编号不存在 / 没有编号的资产 → 空
    assert asset_target.resolve_targets(conn, ["p1"], [("table", "9")]) == []


def test_resolve_targets_refuses_ambiguous_scope(tmp_path: Path) -> None:
    """label_norm 只在论文内唯一，多篇/全库下"表 1"是歧义 —— 一律不定向。"""
    conn, _, _ = _setup(tmp_path)
    assert asset_target.resolve_targets(conn, [], [("table", "1")]) == []
    assert asset_target.resolve_targets(conn, ["p1", "p2"], [("table", "1")]) == []


# ---------------------------------------------------------------- 注入
def test_pinned_chunk_comes_first_and_is_not_duplicated(tmp_path: Path) -> None:
    conn, settings, index = _setup(tmp_path)
    stats: dict = {}
    items = retrieve(
        conn, "表 1 里写了什么", index=index, settings=settings,
        paper_ids=["p1"], stats=stats,
    )

    assert items[0].chunk_id == "p1:c0002", "点名的那张表要排第一，而不是让相似度决定"
    assert [item.chunk_id for item in items].count("p1:c0002") == 1, "不能重复占名额"
    assert stats["rerank"]["pinned"] == ["p1:c0002"]


def test_without_pinning_the_vector_order_wins(tmp_path: Path) -> None:
    """关掉注入（auto_asset=False）时，顺序回到纯向量：c0001 在前。"""
    conn, settings, index = _setup(tmp_path)
    items = retrieve(
        conn, "表 1 里写了什么", index=index, settings=settings,
        paper_ids=["p1"], auto_asset=False,
    )
    assert items[0].chunk_id == "p1:c0001"


def test_explicit_injection_overrides_the_regex(tmp_path: Path) -> None:
    """前置小 LLM 以后从 `injected` 进来 —— 显式给了就不再自己解析问句。"""
    conn, settings, index = _setup(tmp_path)
    items = retrieve(
        conn, "随便问点什么", index=index, settings=settings,
        paper_ids=["p1"], injected=["p1:c0002"],
    )
    assert items[0].chunk_id == "p1:c0002"
