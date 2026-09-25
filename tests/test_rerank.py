"""重排接入（真 SQLite + 真 Chroma + 假向量 + 假 reranker，不联网）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.db.connection import connect, init_db, transaction
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import assets as assets_repo
from app.db.repositories import blocks as blocks_repo
from app.db.repositories import papers as papers_repo
from app.retrieval import keyword_index
from app.retrieval.retriever import retrieve
from app.retrieval.vector_index import ChunkVectorIndex


class _Embedder(Embeddings):
    """按文本查表返回预设向量 —— 测试里精确控制向量名次。"""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.mapping[text]


class _FakeReranker:
    """按**段落原文**给分，顺便把收到的 query / documents 记下来给断言用。"""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen_query: str | None = None
        self.seen_documents: list[str] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.seen_query = query
        self.seen_documents = list(documents)
        return [self.scores.get(doc.strip(), 0.0) for doc in documents]


class _BrokenReranker:
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        raise RuntimeError("上游 429")


def _seed_paper(conn: sqlite3.Connection, paper_id: str, texts: list[str]) -> None:
    with transaction(conn):
        papers_repo.upsert(conn, {"paper_id": paper_id, "title": paper_id})
        chunks_repo.upsert(
            conn,
            [
                {
                    "chunk_id": f"{paper_id}:c{index:04d}",
                    "paper_id": paper_id,
                    "order_index": index,
                    "content": text,
                    "index_text": text,
                    "block_ids": [],
                    "page_start": 1,
                    "page_end": 1,
                    "chunk_type": "text",
                }
                for index, text in enumerate(texts, 1)
            ],
        )
        keyword_index.replace_paper(
            conn,
            paper_id,
            [
                {
                    "chunk_id": f"{paper_id}:c{index:04d}",
                    "index_text": text,
                    "metadata": {"paper_id": paper_id},
                }
                for index, text in enumerate(texts, 1)
            ],
        )


def _setup(
    tmp_path: Path,
    papers: dict[str, list[str]],
    query_vector: list[float],
    query_text: str = "改写后的问题",
):
    conn = connect(":memory:")
    init_db(conn)
    vectors: dict[str, list[float]] = {}
    for paper_id, texts in papers.items():
        _seed_paper(conn, paper_id, texts)
        for index, text in enumerate(texts, 1):
            # 越靠前的 chunk 向量越接近 query（余弦相似度递减）
            vectors[text] = [1.0, float(index)]
    vectors[query_text] = query_vector
    settings = Settings(storage_dir=tmp_path, per_paper_k=2, top_k=3)
    index = ChunkVectorIndex(
        settings, embedder=_Embedder(vectors), collection_name="rerank_probe"
    )
    index.add_chunks(
        [
            {"chunk_id": row["chunk_id"], "index_text": row["index_text"],
             "metadata": {"paper_id": row["paper_id"]}}
            for row in conn.execute("SELECT * FROM chunks")
        ]
    )
    return conn, settings, index


def test_rerank_reorders_the_candidates(tmp_path: Path) -> None:
    conn, settings, index = _setup(tmp_path, {"p1": ["最近", "次近", "最远"]}, [1.0, 0.0])
    reranker = _FakeReranker({"最远": 0.9, "最近": 0.5, "次近": 0.1})

    items = retrieve(
        conn, "改写后的问题", k=3, index=index, settings=settings, reranker=reranker
    )

    # 向量名次是 最近 > 次近 > 最远；重排把"最远"顶到第一 —— 证明重排真的生效了
    assert [item.chunk_id.split(":")[-1] for item in items] == ["c0003", "c0001", "c0002"]
    assert items[0].rank_score == 0.9


def test_keyword_only_hit_has_no_vector_similarity(tmp_path: Path) -> None:
    """关键词独有命中没有向量相似度，展示层必须是 0.0。

    `KeywordHit.similarity` 是 **-bm25**（实测能到 11.4），跟余弦不是一个量纲；
    早期版本把它当余弦塞进 `RetrievedChunk.similarity`，前端就会显示"相似度 11.407"。
    """
    conn, settings, index = _setup(
        tmp_path, {"p1": ["adapter tuning details"]}, [1.0, 0.0], query_text="adapter tuning"
    )
    # 向量侧只有别的 chunk 命中；关键词侧多出来一条
    settings = settings.model_copy(update={"rerank_enabled": False})
    with transaction(conn):
        chunks_repo.upsert(
            conn,
            [{"chunk_id": "p1:c0002", "paper_id": "p1", "order_index": 2,
              "content": "keyword only", "index_text": "adapter tuning", "block_ids": [],
              "page_start": 1, "page_end": 1, "chunk_type": "text"}],
        )
    keyword_index.replace_paper(
        conn, "p1",
        [{"chunk_id": "p1:c0002", "index_text": "adapter tuning",
          "metadata": {"paper_id": "p1"}}],
    )
    # 让向量侧只认识 c0001（不把 c0002 放进向量库）
    items = retrieve(conn, "adapter tuning", k=5, index=index, settings=settings)
    by_id = {item.chunk_id: item for item in items}
    assert "p1:c0002" in by_id, "关键词独有命中应该被召回"
    assert by_id["p1:c0002"].similarity == 0.0


def test_rerank_receives_the_rewritten_question(tmp_path: Path) -> None:
    """**rerank 吃的是改写后的问题** —— 中文原话带悬空指代，打不出分数。"""
    conn, settings, index = _setup(tmp_path, {"p1": ["段落一", "段落二"]}, [1.0, 0.0])
    reranker = _FakeReranker({})

    retrieve(conn, "改写后的问题", k=2, index=index, settings=settings, reranker=reranker)

    assert reranker.seen_query == "改写后的问题"
    assert sorted(reranker.seen_documents) == ["段落一", "段落二"]  # 送的是正文，不是 index_text


def test_rerank_text_replaces_placeholders_with_captions(tmp_path: Path) -> None:
    """占位符要换成"这是哪张表"，不能剔成空白 —— 交叉编码器看不到向量索引里的表注。

    三级兜底：caption → label_norm 显示名 → 类型词。
    """
    conn, settings, index = _setup(
        tmp_path, {"p1": ["正文 A", "正文 B", "正文 C"]}, [1.0, 0.0]
    )
    with transaction(conn):
        for index_, (chunk_id, kind, asset_id) in enumerate(
            [
                ("p1:c0001", "TABLE", "p1:table0001"),
                ("p1:c0002", "FIGURE", "p1:figure0001"),
                ("p1:c0003", "FORMULA", "p1:formula0001"),
            ],
            1,
        ):
            conn.execute(
                "UPDATE chunks SET content = ? WHERE chunk_id = ?",
                (f"正文 {index_} [{kind}_REF asset_id={asset_id}]", chunk_id),
            )
        # assets.block_id 有外键指向 blocks，得先把块建出来
        blocks_repo.upsert(
            conn,
            [
                {"block_id": f"p1:b{i}", "paper_id": "p1", "section_id": None, "order_index": i,
                 "block_type": "text", "heading_level": None, "text": None, "latex": None,
                 "html": None, "caption": None, "page_idx": None, "bbox": None, "image_path": None}
                for i in (1, 2, 3)
            ],
        )
        assets_repo.upsert(
            conn,
            [
                {"asset_id": "p1:table0001", "block_id": "p1:b1", "paper_id": "p1",
                 "asset_type": "table", "caption": "Table 1: 主实验结果", "label_norm": "table:1"},
                {"asset_id": "p1:figure0001", "block_id": "p1:b2", "paper_id": "p1",
                 "asset_type": "figure", "caption": "", "label_norm": "figure:2"},
                {"asset_id": "p1:formula0001", "block_id": "p1:b3", "paper_id": "p1",
                 "asset_type": "formula", "caption": None, "label_norm": None},
            ],
        )

    reranker = _FakeReranker({})
    retrieve(conn, "改写后的问题", k=3, index=index, settings=settings, reranker=reranker)

    joined = " ".join(reranker.seen_documents)
    assert "Table 1: 主实验结果" in joined  # 有 caption
    assert "Figure 2" in joined  # caption 空 → 用 label_norm 的显示名
    assert "Formula" in joined  # 都没有 → 退到类型词
    assert "asset_id" not in joined and "_REF" not in joined  # 占位符本身不留


def test_rerank_failure_falls_back_to_the_merge_order(tmp_path: Path) -> None:
    """重排挂了不能拖垮请求：顺序退回合并顺序（向量在前），并在 stats 里写明原因。"""
    conn, settings, index = _setup(tmp_path, {"p1": ["最近", "次近", "最远"]}, [1.0, 0.0])
    stats: dict[str, Any] = {}

    items = retrieve(
        conn, "改写后的问题", k=3, index=index, settings=settings,
        reranker=_BrokenReranker(), stats=stats,
    )

    assert [item.chunk_id.split(":")[-1] for item in items] == ["c0001", "c0002", "c0003"]
    assert stats["rerank"]["ok"] is False
    assert "429" in stats["rerank"]["error"]


def test_rerank_can_be_disabled(tmp_path: Path) -> None:
    conn, settings, index = _setup(tmp_path, {"p1": ["最近", "次近"]}, [1.0, 0.0])
    settings = settings.model_copy(update={"rerank_enabled": False})
    stats: dict[str, Any] = {}

    retrieve(
        conn, "改写后的问题", k=2, index=index, settings=settings,
        reranker=_FakeReranker({"次近": 99.0}), stats=stats,
    )

    assert stats["rerank"] == {"enabled": False}


def test_multi_paper_quota_is_applied_after_rerank(tmp_path: Path) -> None:
    """多篇：重排之后**每篇**再取 quota 条 —— 一篇不能把名额吃光。

    即使重排把所有高分都给了 p1，p2 也要保住自己的名额。
    """
    conn, settings, index = _setup(
        tmp_path, {"p1": ["p1 一", "p1 二"], "p2": ["p2 一", "p2 二"]}, [1.0, 0.0]
    )
    reranker = _FakeReranker({"p1 一": 0.9, "p1 二": 0.8, "p2 一": 0.1, "p2 二": 0.05})

    items = retrieve(
        conn, "改写后的问题", index=index, settings=settings,
        paper_ids=["p1", "p2"], reranker=reranker,
    )

    per_paper: dict[str, int] = {}
    for item in items:
        per_paper[item.paper["paper_id"]] = per_paper.get(item.paper["paper_id"], 0) + 1
    assert per_paper == {"p1": 2, "p2": 2}
    # 每篇内部按重排分排：p1 的两条里 "p1 一" 在前
    assert items[0].chunk_id.startswith("p1")
