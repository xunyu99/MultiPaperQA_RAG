"""Step 4 验收：Chroma 封装（用假向量，不联网、不花钱）。"""

from __future__ import annotations

import math
import re
import zlib
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.retrieval.vector_index import ChunkVectorIndex


class FakeEmbedder(Embeddings):
    """确定性假向量：词袋哈希 + 归一化。

    相同词越多、余弦相似度越高 —— 足够验证检索链路，且完全不联网。
    **不能用 Python 的 `hash()`**：字符串的 hash 每次进程启动都不一样（有随机盐），
    那样测试时好时坏。
    """

    dimension = 32

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @classmethod
    def _vec(cls, text: str) -> list[float]:
        vector = [0.0] * cls.dimension
        for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
            vector[zlib.crc32(token.encode()) % cls.dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def _make_index(tmp_path: Path, collection: str = "chunks") -> ChunkVectorIndex:
    settings = Settings(storage_dir=tmp_path)
    return ChunkVectorIndex(settings, embedder=FakeEmbedder(), collection_name=collection)


def _items() -> list[dict]:
    return [
        {
            "chunk_id": "p1:c0001",
            "index_text": "micro expression recognition is difficult because the movement is subtle",
            "metadata": {"chunk_id": "p1:c0001", "paper_id": "p1", "page_start": 1,
                         "section_type": "intro", "chunk_type": "text"},
        },
        {
            "chunk_id": "p1:c0002",
            "index_text": "we train the model on the training split with batch size sixteen",
            "metadata": {"chunk_id": "p1:c0002", "paper_id": "p1", "page_start": 3,
                         "section_type": "experiment", "chunk_type": "text"},
        },
        {
            "chunk_id": "p2:c0001",
            "index_text": "micro expression recognition with a multimodal large language model",
            "metadata": {"chunk_id": "p2:c0001", "paper_id": "p2", "page_start": None,
                         "section_type": "method", "chunk_type": "mixed"},
        },
    ]


def test_add_and_count(tmp_path: Path) -> None:
    index = _make_index(tmp_path)
    assert index.add_chunks(_items()) == 3
    assert index.count() == 3
    assert index.counts_by_paper() == {"p1": 2, "p2": 1}


def test_empty_text_is_skipped(tmp_path: Path) -> None:
    index = _make_index(tmp_path)
    assert index.add_chunks([{"chunk_id": "x", "index_text": "   ", "metadata": {}}]) == 0
    assert index.count() == 0


def test_none_metadata_is_dropped(tmp_path: Path) -> None:
    """Chroma 的 metadata 不收 None（page_start 经常是 None）。"""
    index = _make_index(tmp_path)
    index.add_chunks(_items())
    hit = index.search("multimodal large language model", k=1)[0]
    assert hit.chunk_id == "p2:c0001"
    assert "page_start" not in hit.metadata
    assert hit.metadata["paper_id"] == "p2"


def test_search_ranks_by_similarity(tmp_path: Path) -> None:
    index = _make_index(tmp_path)
    index.add_chunks(_items())
    hits = index.search("batch size for training", k=3)
    assert hits[0].chunk_id == "p1:c0002"
    # 相似度要递减，而且换算成了"越大越像"
    assert hits[0].similarity >= hits[-1].similarity
    assert -1.0 <= hits[0].similarity <= 1.0


def test_paper_filter_is_applied(tmp_path: Path) -> None:
    """过滤要在向量库层做，不是检索完再筛 —— 所以结果里不能出现别的论文。"""
    index = _make_index(tmp_path)
    index.add_chunks(_items())
    hits = index.search("micro expression recognition", k=3, paper_id="p1")
    assert hits, "过滤后不该一条都没有"
    assert {hit.metadata["paper_id"] for hit in hits} == {"p1"}


def test_reset_paper_only_removes_that_paper(tmp_path: Path) -> None:
    index = _make_index(tmp_path)
    index.add_chunks(_items())
    assert index.reset_paper("p1") == 2
    assert index.counts_by_paper() == {"p2": 1}
    # 再删一次是 0，不是报错（幂等）
    assert index.reset_paper("p1") == 0


def test_rewrite_same_paper_does_not_duplicate(tmp_path: Path) -> None:
    """先删后写：同一篇论文重跑索引，条数不变（PLAN Step 4 的验收）。"""
    index = _make_index(tmp_path)
    index.add_chunks(_items())
    index.reset_paper("p1")
    index.add_chunks([item for item in _items() if item["metadata"]["paper_id"] == "p1"])
    assert index.count() == 3


def test_search_is_stable_across_repeats(tmp_path: Path) -> None:
    index = _make_index(tmp_path)
    index.add_chunks(_items())
    first = [hit.chunk_id for hit in index.search("micro expression", k=3)]
    second = [hit.chunk_id for hit in index.search("micro expression", k=3)]
    assert first == second
