"""同章节去重（`section_cap`）的验收：真 SQLite + 真 Chroma，假向量，不联网。

要守住三件事：

1. `section_cap=0` 时结果和 v1 逐条一致（关掉就是没加过）；
2. 开了以后，同一章节不会占满名额，别的章节能顶上来；
3. **绝不会因此少给证据** —— 章节不够多样时用重复补回来。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.db.connection import connect, init_db, transaction
from app.db.repositories import blocks as blocks_repo
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import papers as papers_repo
from app.db.repositories import sections as sections_repo
from app.retrieval.retriever import retrieve
from app.retrieval.vector_index import ChunkVectorIndex


class ScriptedEmbedder(Embeddings):
    """按文本查表返回预设向量 —— 排名完全可控。"""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.mapping[text]


# 相似度严格递减：最近 > 次近 > 第三 > 最远
_VECTORS = {
    "查询": [1.0, 0.0],
    "最近": [1.0, 0.05],
    "次近": [1.0, 0.10],
    "第三": [1.0, 0.20],
    "最远": [0.5, 0.5],
}


def _block(**kw) -> dict:
    row = {
        "block_id": "", "paper_id": "p1", "section_id": None, "order_index": 0,
        "block_type": "text", "heading_level": None, "text": None, "latex": None,
        "html": None, "caption": None, "page_idx": None, "bbox": None, "image_path": None,
    }
    row.update(kw)
    return row


def _seed(conn: sqlite3.Connection) -> None:
    """p1 的 method 章节切了 3 个 chunk（会手拉手占名额），intro 只有 1 个。"""
    with transaction(conn):
        papers_repo.upsert(conn, {"paper_id": "p1", "title": "论文一"})
        sections_repo.upsert(
            conn,
            [
                {"section_id": "p1:s1", "paper_id": "p1", "parent_id": None, "level": 1,
                 "order_index": 1, "title": "1 Introduction", "section_type": "intro"},
                {"section_id": "p1:s2", "paper_id": "p1", "parent_id": None, "level": 1,
                 "order_index": 2, "title": "2 Method", "section_type": "method"},
            ],
        )
        blocks_repo.upsert(
            conn,
            [
                _block(block_id="p1:b1", section_id="p1:s2", order_index=1, text="方法一"),
                _block(block_id="p1:b2", section_id="p1:s2", order_index=2, text="方法二"),
                _block(block_id="p1:b3", section_id="p1:s2", order_index=3, text="方法三"),
                _block(block_id="p1:b4", section_id="p1:s1", order_index=4, text="引言"),
            ],
        )
        chunks_repo.upsert(
            conn,
            [
                {"chunk_id": "p1:c0001", "paper_id": "p1", "order_index": 1, "content": "一",
                 "index_text": "最近", "block_ids": ["p1:b1"], "page_start": 1, "page_end": 1,
                 "chunk_type": "text"},
                {"chunk_id": "p1:c0002", "paper_id": "p1", "order_index": 2, "content": "二",
                 "index_text": "次近", "block_ids": ["p1:b2"], "page_start": 1, "page_end": 1,
                 "chunk_type": "text"},
                {"chunk_id": "p1:c0003", "paper_id": "p1", "order_index": 3, "content": "三",
                 "index_text": "第三", "block_ids": ["p1:b3"], "page_start": 1, "page_end": 1,
                 "chunk_type": "text"},
                {"chunk_id": "p1:c0004", "paper_id": "p1", "order_index": 4, "content": "四",
                 "index_text": "最远", "block_ids": ["p1:b4"], "page_start": 1, "page_end": 1,
                 "chunk_type": "text"},
            ],
        )


def _setup(tmp_path: Path):
    conn = connect(":memory:")
    init_db(conn)
    _seed(conn)
    # 关掉 rerank：这里测的是同章节去重，重排会改变顺序把断言打乱（见 test_rerank.py）
    settings = Settings(storage_dir=tmp_path, rerank_enabled=False)
    index = ChunkVectorIndex(settings, embedder=ScriptedEmbedder(_VECTORS), collection_name="dedupe")
    index.add_chunks(
        [
            {"chunk_id": row["chunk_id"], "index_text": row["index_text"],
             "metadata": {"chunk_id": row["chunk_id"], "paper_id": row["paper_id"],
                          "page_start": row["page_start"]}}
            for row in chunks_repo.list_by_paper(conn, "p1")
        ]
    )
    return conn, settings, index


def test_cap_off_is_the_old_behaviour(tmp_path: Path) -> None:
    """关掉去重 = v1 行为：长章节的两条照样手拉手进榜。"""
    conn, settings, index = _setup(tmp_path)
    try:
        items = retrieve(conn, "查询", k=2, index=index, settings=settings, section_cap=0)
        assert [item.chunk_id for item in items] == ["p1:c0001", "p1:c0002"]
        assert {item.section_title for item in items} == {"2 Method"}
    finally:
        conn.close()


def test_cap_lets_another_section_in(tmp_path: Path) -> None:
    """开了以后第二名换成别的章节 —— 这正是 cross_1 想要的位移。"""
    conn, settings, index = _setup(tmp_path)
    try:
        items = retrieve(conn, "查询", k=2, index=index, settings=settings, section_cap=1)
        assert [item.chunk_id for item in items] == ["p1:c0001", "p1:c0004"]
        assert len({item.section_title for item in items}) == 2
    finally:
        conn.close()


def test_cap_never_returns_fewer_than_k(tmp_path: Path) -> None:
    """章节不够多样时用重复补回来：宁可重复，也不能少给证据。"""
    conn, settings, index = _setup(tmp_path)
    try:
        items = retrieve(conn, "查询", k=3, index=index, settings=settings, section_cap=1)
        assert len(items) == 3
        assert [item.chunk_id for item in items] == ["p1:c0001", "p1:c0004", "p1:c0002"]
    finally:
        conn.close()


def test_cap_is_reported_in_stats(tmp_path: Path) -> None:
    """诊断信息要能看见"谁被去重挤掉了"，不然出问题没法归因。"""
    conn, settings, index = _setup(tmp_path)
    try:
        stats: dict = {}
        retrieve(conn, "查询", k=2, index=index, settings=settings, section_cap=1, stats=stats)
        assert stats["section_cap"] == 1
        assert "p1:c0002" in stats["duplicates"]
    finally:
        conn.close()


def test_default_is_off(tmp_path: Path) -> None:
    """默认不开启：没测出好处之前，它不该悄悄改变现有行为。"""
    conn, settings, index = _setup(tmp_path)
    try:
        assert settings.section_cap == 0
        items = retrieve(conn, "查询", k=2, index=index, settings=settings)
        assert [item.chunk_id for item in items] == ["p1:c0001", "p1:c0002"]
    finally:
        conn.close()
