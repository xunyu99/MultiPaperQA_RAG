"""Chroma 向量索引的封装 —— 换 Qdrant / pgvector 只改这一个文件。

四个刻意的设计（每条都有理由，不是默认值）：

1. **距离用 cosine**。Chroma 默认是 `l2`（欧氏距离），语义向量比的是**方向**不是
   长度，用 L2 会受向量模长影响。建 collection 时显式写 `{"hnsw:space": "cosine"}`。
2. **一个 collection**，id 用 `chunk_id`，metadata 带 `paper_id` / `section_type` /
   `page_start` / `chunk_type`。Chroma 的 metadata 只收标量，所以 `bbox`、
   `block_ids` 这类结构化字段一律不进来（要它们就回 SQLite）。
3. **写入按 paper_id 先删后写**。重复入库同一篇论文必须幂等；不先删就会留下
   已经不存在于 `chunks` 表的孤儿向量。
4. **不存文本**（`add_texts` 会把文本塞进 Chroma 的 document 字段）。文本的唯一
   事实源是 SQLite 的 `chunks.index_text`，检索拿到 id 回表取 —— 两处存文本迟早不一致。

score 方向说明：Chroma 返回的是**距离**（越小越像）。本模块统一换算成
`similarity = 1 - distance` 返回，越大越像，肉眼看得懂。

为什么向量检索直接用底层 collection 的 `query()`，不走 langchain 的
`similarity_search_*`：实测 langchain-chroma 1.x 里
`similarity_search_by_vector_with_relevance_scores` **返回的其实还是原始距离**
（名字里的 relevance 有误导性，并没有做归一化）。直接用 Chroma 原生接口，
返回什么就是什么，少一层解释空间。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from app.config import Settings, get_settings
from app.providers.embedder import Embedder

COLLECTION_NAME = "chunks"


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    similarity: float
    distance: float
    metadata: dict[str, Any]


class ChunkVectorIndex:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings)
        # 不给就用 settings 里的（默认 "chunks"）—— 双 collection 对照靠它切换
        self.collection_name = collection_name or self.settings.chroma_collection
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=str(self.settings.chroma_dir),
            # 注意：不写这一行就是 l2 距离，不是 cosine —— 见模块开头第 1 条
            collection_metadata={"hnsw:space": "cosine"},
            # 关掉匿名遥测：Chroma 默认会往 posthog 发数据，本机没网会一直重试、
            # 还会在日志里刷警告。我们不需要它。
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def reset_paper(self, paper_id: str) -> int:
        """删掉这篇论文的旧向量，返回删了几条。

        Chroma 的 `delete` 只按 id 删，所以先按 metadata 查出 id 再删 ——
        这样不用碰 `_collection` 私有属性。先删后写是幂等的关键。
        """
        ids = self.ids_of_paper(paper_id)
        if ids:
            self._store.delete(ids=ids)
        return len(ids)

    def add_chunks(self, items: Sequence[dict[str, Any]]) -> int:
        """写入一批 chunk。每项形如：

            {"chunk_id": "...", "index_text": "...", "metadata": {...}}

        只 embed `index_text`（不是 `content`）—— 见 PLAN 铁律 2。
        """
        rows = [item for item in items if (item.get("index_text") or "").strip()]
        if not rows:
            return 0
        self._store.add_texts(
            texts=[row["index_text"] for row in rows],
            ids=[row["chunk_id"] for row in rows],
            metadatas=[_clean_metadata(row["metadata"]) for row in rows],
        )
        return len(rows)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 5,
        paper_id: str | None = None,
    ) -> list[SearchHit]:
        """向量检索（含 query embedding）。`paper_id` 不为空时在**向量库层**过滤。"""
        vector = self.embedder.embed_query(query)
        return self.search_by_vector(vector, k=k, paper_id=paper_id)

    def search_by_vector(
        self,
        vector: Sequence[float],
        k: int = 5,
        paper_id: str | None = None,
    ) -> list[SearchHit]:
        """用现成的向量检索 —— 跳过 embedding，纯查库。

        单独拆出来是为了把两段耗时分开量：query embedding 是**网络往返**
        （100–300ms），Chroma 查询本身是毫秒级。混在一起量，
        "单次检索 <50ms"这条验收永远看着不达标。
        """
        where = {"paper_id": paper_id} if paper_id else None
        result = self._store._collection.query(
            query_embeddings=[list(vector)],
            n_results=k,
            where=where,
            include=["metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for index, chunk_id in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                SearchHit(
                    chunk_id=str(metadata.get("chunk_id") or chunk_id),
                    similarity=1.0 - distance,
                    distance=distance,
                    metadata=metadata,
                )
            )
        return hits

    def search_timed(
        self,
        query: str,
        k: int = 5,
        paper_id: str | None = None,
    ) -> tuple[list[SearchHit], dict[str, float]]:
        """检索并返回分段耗时：`{"embed_ms": ..., "search_ms": ..., "total_ms": ...}`。"""
        started = time.perf_counter()
        vector = self.embedder.embed_query(query)
        embedded = time.perf_counter()
        hits = self.search_by_vector(vector, k=k, paper_id=paper_id)
        finished = time.perf_counter()
        return hits, {
            "embed_ms": (embedded - started) * 1000,
            "search_ms": (finished - embedded) * 1000,
            "total_ms": (finished - started) * 1000,
        }

    # ------------------------------------------------------------------
    # 统计 / 对账
    # ------------------------------------------------------------------
    def ids_of_paper(self, paper_id: str) -> list[str]:
        result = self._store.get(where={"paper_id": paper_id}, include=["metadatas"])
        return [str(item) for item in (result.get("ids") or [])]

    def all_ids(self) -> list[str]:
        result = self._store.get(include=["metadatas"])
        return [str(item) for item in (result.get("ids") or [])]

    def count(self) -> int:
        return len(self.all_ids())

    def counts_by_paper(self) -> dict[str, int]:
        result = self._store.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in result.get("metadatas") or []:
            paper_id = str((metadata or {}).get("paper_id") or "?")
            counts[paper_id] = counts.get(paper_id, 0) + 1
        return counts

    def reset_all(self) -> None:
        """清空 collection。重建索引时用，慎用。"""
        ids = self.all_ids()
        if ids:
            self._store.delete(ids=ids)


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma 的 metadata 只收标量，且不收 None。

    `page_start` 经常是 None（比如纯资产块没有页码），直接传进去会被拒。
    """
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
    return cleaned
