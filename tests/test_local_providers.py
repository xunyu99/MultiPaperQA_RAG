"""本地通道的**接线**测试：不下载模型，只确认 provider 开关选对了实现。

模型本身跑得对不对由 `scripts/check_local_models.py` 验收（要下模型，不进单测）。
"""

from __future__ import annotations

from app.config import Settings
from app.providers.embedder import Embedder
from app.providers.reranker import Reranker


def _settings(**kw) -> Settings:
    base = {"dashscope_api_key": "x", "embedding_provider": "api", "rerank_provider": "dashscope"}
    base.update(kw)
    return Settings(**base)


def test_embedder_picks_local_branch_without_loading_model() -> None:
    """选 local 时不去建 OpenAI 客户端；也**不该**在构造阶段加载模型（懒加载）。"""
    embedder = Embedder(_settings(embedding_provider="local"))
    assert embedder._client.__class__.__name__ == "_LocalEmbeddings"
    assert embedder._client._model is None  # 构造完还没加载


def test_embedder_picks_api_branch_by_default() -> None:
    embedder = Embedder(_settings())
    assert embedder._client.__class__.__name__ == "OpenAIEmbeddings"


def test_reranker_picks_local_branch_without_http_client() -> None:
    reranker = Reranker(_settings(rerank_provider="local_bge"))
    assert reranker._client is None
    assert reranker._local is None


def test_reranker_picks_api_branch_by_default() -> None:
    reranker = Reranker(_settings())
    assert reranker._client is not None
