"""Step 5 验收：检索链路（真 SQLite + 真 Chroma，假向量，不联网）。"""

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
from app.db.repositories import sections as sections_repo
from app.retrieval.retriever import retrieve
from app.retrieval.vector_index import ChunkVectorIndex


class ScriptedEmbedder(Embeddings):
    """**按文本查表**返回预设向量 —— 测试里就能精确控制排名，不靠运气。"""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.mapping[text]


_VECTORS = {
    "查询": [1.0, 0.0],
    "最近": [1.0, 0.1],
    "次近": [0.7, 0.7],
    "最远": [0.0, 1.0],
}


def _block(**kw) -> dict:
    """同一批写入的字段必须完全一致（upsert_rows 的硬要求），所以给全字段。"""
    row = {
        "block_id": "", "paper_id": "p1", "section_id": None, "order_index": 0,
        "block_type": "text", "heading_level": None, "text": None, "latex": None,
        "html": None, "caption": None, "page_idx": None, "bbox": None, "image_path": None,
    }
    row.update(kw)
    return row


def _seed(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        papers_repo.upsert(conn, {"paper_id": "p1", "title": "测试论文", "citation": "Zhang et al."})
        sections_repo.upsert(
            conn,
            [
                {"section_id": "p1:s1", "paper_id": "p1", "parent_id": None, "level": 1,
                 "order_index": 1, "title": "1 Introduction", "section_type": "intro"},
                {"section_id": "p1:s2", "paper_id": "p1", "parent_id": None, "level": 1,
                 "order_index": 2, "title": "2 Method", "section_type": "method"},
            ],
        )
        # 注意插入顺序：c0001 在前。SQL 的 IN 查询通常按 rowid 返回，
        # 所以代码不重排的话结果会是 c0001, c0002, c0003 —— 而向量库的排名正好是反的。
        blocks_repo.upsert(
            conn,
            [
                _block(block_id="p1:b1", section_id="p1:s1", order_index=1,
                       text="正文一", bbox=[1.0, 1.0, 2.0, 2.0], page_idx=1),
                _block(block_id="p1:b2", section_id="p1:s2", order_index=2,
                       text="正文二", bbox=[1.0, 3.0, 2.0, 4.0], page_idx=2),
                _block(block_id="p1:b3", section_id="p1:s2", order_index=3, block_type="table",
                       html="<table><tr><td>x</td></tr></table>", caption="Table 1: 结果",
                       bbox=[1.0, 5.0, 2.0, 6.0], page_idx=3),
            ],
        )
        chunks_repo.upsert(
            conn,
            [
                {"chunk_id": "p1:c0001", "paper_id": "p1", "order_index": 1, "content": "最远的块",
                 "index_text": "最远", "block_ids": ["p1:b1"], "page_start": 1, "page_end": 1,
                 "chunk_type": "text"},
                {"chunk_id": "p1:c0002", "paper_id": "p1", "order_index": 2, "content": "次近的块",
                 "index_text": "次近", "block_ids": ["p1:b2"], "page_start": 2, "page_end": 2,
                 "chunk_type": "text"},
                {"chunk_id": "p1:c0003", "paper_id": "p1", "order_index": 3,
                 "content": "最近的块 [TABLE_REF asset_id=p1:table0001]",
                 "index_text": "最近", "block_ids": ["p1:b3"], "page_start": 3, "page_end": 3,
                 "chunk_type": "table"},
            ],
        )
        assets_repo.upsert(
            conn,
            [{"asset_id": "p1:table0001", "block_id": "p1:b3", "paper_id": "p1",
              "asset_type": "table", "caption": "Table 1: 结果",
              "raw_content": "<table><tr><td>x</td></tr></table>", "page_idx": 3}],
        )
        assets_repo.link_chunk(conn, "p1:c0003", ["p1:table0001"])


def _setup(tmp_path: Path):
    conn = connect(":memory:")
    init_db(conn)
    _seed(conn)
    # 关掉 rerank：这些用例断言的是**召回顺序**，重排会把它盖掉；
    # 而且不关就会真的去打百炼的接口（慢、还花钱）。rerank 自己的用例见 test_rerank.py
    settings = Settings(storage_dir=tmp_path, rerank_enabled=False)
    index = ChunkVectorIndex(settings, embedder=ScriptedEmbedder(_VECTORS), collection_name="probe")
    index.add_chunks(
        [
            {"chunk_id": row["chunk_id"], "index_text": row["index_text"],
             "metadata": {"chunk_id": row["chunk_id"], "paper_id": row["paper_id"],
                          "page_start": row["page_start"]}}
            for row in chunks_repo.list_by_paper(conn, "p1")
        ]
    )
    return conn, settings, index


def test_result_order_follows_the_vector_store(tmp_path: Path) -> None:
    """**最容易踩的坑**：SQL 的 IN 返回顺序跟向量库排名无关，必须按后者重排。"""
    conn, settings, index = _setup(tmp_path)
    try:
        items = retrieve(conn, "查询", k=3, index=index, settings=settings)
        assert [item.chunk_id for item in items] == ["p1:c0003", "p1:c0002", "p1:c0001"]
        assert items[0].similarity > items[-1].similarity
    finally:
        conn.close()


def test_section_and_paper_are_attached(tmp_path: Path) -> None:
    conn, settings, index = _setup(tmp_path)
    try:
        items = retrieve(conn, "查询", k=3, index=index, settings=settings)
        first = items[0]
        assert first.paper["title"] == "测试论文"
        assert first.section_title == "2 Method"
        assert first.section_type == "method"
        assert first.section_order == 2
    finally:
        conn.close()


def test_assets_and_trace_are_attached(tmp_path: Path) -> None:
    conn, settings, index = _setup(tmp_path)
    try:
        items = retrieve(conn, "查询", k=1, index=index, settings=settings)
        first = items[0]
        assert [asset["asset_id"] for asset in first.assets] == ["p1:table0001"]
        assert first.trace[0]["block_id"] == "p1:b3"
        assert first.trace[0]["bbox"] == [1.0, 5.0, 2.0, 6.0]
        assert first.trace[0]["page_idx"] == 3
    finally:
        conn.close()


def test_mention_only_asset_is_not_attached(tmp_path: Path) -> None:
    """正文**只是提到**资产的块，不带本体（PLAN §6.2「资产携带」规则 3）。

    交叉引用在入库时确实挂到了 `chunk_assets`（这里手工模拟那一步），但 `retrieve` 走的是
    `expand_assets`，它**只认占位符** —— 所以这个块拿不到那张表。
    这条是"只提到的不带"的根据：谁哪天把 `_enrich` 改成"绑了就带"，一个综述段落
    就会顺带拖进一堆图表的本体，这个用例会先红。
    """
    conn, settings, index = _setup(tmp_path)
    try:
        with transaction(conn):
            chunks_repo.upsert(
                conn,
                [{"chunk_id": "p1:c0004", "paper_id": "p1", "order_index": 4,
                  "content": "如 Table 1 所示，我们的方法更好。",  # 提到，但没有占位符
                  "index_text": "最近", "block_ids": ["p1:b2"], "page_start": 2,
                  "page_end": 2, "chunk_type": "text"}],
            )
            # 交叉引用：本体在 c0003 的那张表，也挂到 c0004 上（入库时就是这么干的）
            assets_repo.link_chunk(conn, "p1:c0004", ["p1:table0001"])
        index.add_chunks(
            [{"chunk_id": "p1:c0004", "index_text": "最近",
              "metadata": {"chunk_id": "p1:c0004", "paper_id": "p1", "page_start": 2}}]
        )

        items = retrieve(conn, "查询", k=5, index=index, settings=settings)
        by_id = {item.chunk_id: item for item in items}

        # 本体块（正文有占位符）带着资产
        assert [a["asset_id"] for a in by_id["p1:c0003"].assets] == ["p1:table0001"]
        # 只是提到的块不带 —— 哪怕 chunk_assets 里挂着
        assert by_id["p1:c0004"].assets == []
    finally:
        conn.close()


def test_orphan_vector_is_dropped(tmp_path: Path) -> None:
    """向量在、SQLite 里没有（论文被删过）→ 丢弃，不能传到证据卡。"""
    conn, settings, index = _setup(tmp_path)
    try:
        index.add_chunks(
            [{"chunk_id": "p1:ghost", "index_text": "查询",
              "metadata": {"chunk_id": "p1:ghost", "paper_id": "p1"}}]
        )
        items = retrieve(conn, "查询", k=5, index=index, settings=settings)
        assert all(item.chunk_id != "p1:ghost" for item in items)
    finally:
        conn.close()


def test_no_hits_returns_empty(tmp_path: Path) -> None:
    conn, settings, index = _setup(tmp_path)
    try:
        index.reset_all()
        assert retrieve(conn, "查询", k=3, index=index, settings=settings) == []
    finally:
        conn.close()
