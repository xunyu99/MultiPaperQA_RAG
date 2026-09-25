"""重排接入层：交叉编码器给"问题-段落"逐对打分，把真正相关的顶上来。

**为什么要单独一层**：百炼的 text-rerank **不在 OpenAI 兼容面里**
（`/compatible-mode/v1` 下面没有它），所以不能复用 `OpenAIEmbeddings` /
`ChatOpenAI` 那套 client，得自己发 HTTP。端点、请求体、响应体三样都跟 OpenAI 的
约定不一样，全部收在这个文件里，换厂商只改这里。

**输入必须是改写后的问题**。两个理由，都是实测出来的：

1. 中文原话（"它在 DFEW 上呢？"）里带悬空指代，rerank 拿它当 query 打不出分数；
2. 就算问题自包含，有些 reranker 在"中文 query + 英文段落"上没有区分度 —— 实测本地
   `bge-reranker-base`：中文问题打英文相关段落 (-10.1884) 和无关段落 (-10.1938)
   只差 0.005，等于没打分。

**模型选型**（同一组样例，相关段落 / 无关段落 / 中文提问打英文段落）：

    qwen3.7-text-rerank   0.9888 / 0.0332 / 0.8445   ← 默认
    gte-rerank-v2         0.5026 / 0.0060 / 0.1714

qwen3.7 的分数落在可解释的 0~1 区间，相关和无关差两个数量级 —— 这意味着以后想加
"分数低于 τ 就不给模型看证据"（图里的信号③）是可行的；gte 那组压缩在 0.5/0.17
附近，阈值很难定。

调用点在 `retriever` 里，传进去的 `query` 就是改写后的那个（见 `prepare_turn`）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import httpx

from app.config import Settings, get_settings

# 单篇文档送进去的长度上限（字符）。一篇 chunk 可能上千字符，整篇送过去又慢又容易
# 触发服务端长度限制；重排看的是"这段在不在讲这个问题"，开头部分足够判断。
MAX_DOC_CHARS = 3000


class Reranker:
    """百炼 text-rerank 的薄封装。失败一律抛异常，由调用方决定怎么降级。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # 本地 ONNX/CrossEncoder 分支：不发 HTTP，所以不建 httpx client
        self._local: Any = None
        if self.settings.rerank_provider == "local_bge":
            self._client: Any = None
            return
        kwargs: dict[str, Any] = {"timeout": self.settings.rerank_timeout}
        if self.settings.dashscope_proxy:
            kwargs["proxy"] = self.settings.dashscope_proxy
        self._client = httpx.Client(**kwargs)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[float]:
        """返回与 `documents` **等长**的相关度分，越大越相关。

        服务端只返回它认为靠前的几条（`top_n`），其余位置补 0.0 —— 调用方按等长
        数组取值，不用去猜哪些没返回。
        """
        if not documents:
            return []
        if self.settings.rerank_provider == "local_bge":
            return self._rerank_local(query, documents)
        payload = {
            "model": self.settings.rerank_model,
            "input": {
                "query": query or "",
                "documents": [_clip(doc) for doc in documents],
            },
            "parameters": {
                "return_documents": False,
                "top_n": top_n or len(documents),
            },
        }
        response = self._client.post(
            self.settings.rerank_base_url,
            headers={
                "Authorization": f"Bearer {self.settings.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return _parse_scores(response.json(), len(documents))

    def _rerank_local(self, query: str, documents: Sequence[str]) -> list[float]:
        """本地交叉编码器打分，返回 0~1（**过一遍 sigmoid**）。

        为什么要 sigmoid：`bge-reranker-v2-m3` 吐的是 logits（-10 ~ +10 那种），
        而下游的 τ 阈值（"重排分低于阈值就判资料不足"）是按百炼那种 0~1 可解释分数量纲设计的。
        不归一化的话，换 provider 就等于换了量纲，阈值会直接失效。
        """
        if self._local is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - 取决于本机是否装了可选依赖
                raise RuntimeError(
                    "RERANK_PROVIDER=local_bge 需要 sentence-transformers，先装：\n"
                    "  .venv\\Scripts\\python.exe -m pip install sentence-transformers"
                ) from exc
            self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
            self._local = CrossEncoder(
                self.settings.rerank_local_model,
                cache_folder=str(self.settings.model_cache_dir),
                device="cpu",
                max_length=512,
            )
        logits = self._local.predict(
            [[query or "", _clip(doc)] for doc in documents], convert_to_numpy=True
        )
        return [1.0 / (1.0 + math.exp(-float(value))) for value in logits]


def _clip(text: str) -> str:
    value = " ".join((text or "").split())
    return value[:MAX_DOC_CHARS] if len(value) > MAX_DOC_CHARS else value


def _parse_scores(data: dict[str, Any], total: int) -> list[float]:
    """从响应里抠出 `[{index, relevance_score}]`，按 index 填回等长数组。

    三种形态都认：百炼原生的 `output.results`、顶层 `results`、以及 OpenAI 风格的
    `data`（字段名 `relevance_score` 或 `score`）—— 不同版本/网关回来的不一样，
    与其让它 500，不如都收下。
    """
    results = (data.get("output") or {}).get("results") or data.get("results") or data.get("data") or []
    scores = [0.0] * total
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if index is None or not 0 <= int(index) < total:
            continue
        value = item.get("relevance_score", item.get("score", 0.0))
        scores[int(index)] = float(value or 0.0)
    return scores
