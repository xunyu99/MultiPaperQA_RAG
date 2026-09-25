"""向量接入层（LangChain OpenAIEmbeddings）。

DeepSeek 不提供 embedding，所以向量走百炼 text-embedding-v4，
复用 DASHSCOPE_API_KEY（和 Planner 同一个 key、不同模型）。
本地 bge-m3 留到 Phase 7 做对比实验，接口不变。
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import Settings, get_settings


class Embedder(Embeddings):
    """实现 LangChain 的 Embeddings 接口，这样能直接喂给 langchain-chroma。

    只多这一层继承：`Embeddings` 是个 ABC，定义了 `embed_documents` / `embed_query`
    两个方法，我们本来就实现了，声明成子类能让类型检查和下游库都认。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.embedding_base_url,
            dimensions=self.settings.embedding_dim,
            chunk_size=self.settings.embedding_batch_size,
            # 关键：非 OpenAI 端点必须关掉，否则会用 tiktoken 做 token 计算和补零
            check_embedding_ctx_length=False,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._client.embed_documents([t if t.strip() else " " for t in texts])
        self._validate(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._client.embed_query(text or " ")
        if len(vector) != self.settings.embedding_dim:
            raise ValueError(
                f"查询向量维度不一致：.env 配的是 {self.settings.embedding_dim}，实际 {len(vector)}。"
            )
        return vector

    def _validate(self, vectors: list[list[float]]) -> None:
        if not vectors:
            raise ValueError("embedding 返回空结果")
        dim = len(vectors[0])
        if dim != self.settings.embedding_dim:
            raise ValueError(
                f"向量维度不一致：.env 配的是 {self.settings.embedding_dim}，实际返回 {dim}。"
                "请改 EMBEDDING_DIM，否则后续检索全错。"
            )

    @property
    def dimension(self) -> int:
        return self.settings.embedding_dim
