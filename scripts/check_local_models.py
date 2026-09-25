"""本地模型验收（M1）：确认 bge-m3 / bge-reranker-v2-m3 真的能用，且**跨语言有效**。

用法：
    .venv\\Scripts\\python.exe -m scripts.check_local_models

判据不是"跑起来了"，而是**分得开**：中文提问打英文的相关段落，分数要明显高于无关段落。
这条判据来自一次真事故 —— 本地 `bge-reranker-base`（单语版）实测相关 -10.1884 / 无关 -10.1938，
只差 0.005，等于没打分，所以后来选了多语言版 `bge-reranker-v2-m3`。
"""

from __future__ import annotations

import math
import time

from app.config import get_settings
from app.providers.reranker import Reranker

# 中文提问 + 一篇英文论文里的相关段落 / 另一篇论文里的无关段落（都取自本项目语料）
QUERY = "动态人脸表情识别里，CLIP 这类视觉语言模型存在什么问题"
RELEVANT = (
    "The emergence of Vision-Language Models (VLMs) like CLIP provides appealing solutions "
    "to various vision problems including Dynamic Facial Expression Recognition (DFER). "
    "However, most of the proposed approaches face major challenges, particularly related to "
    "inefficient full fine-tuning of the encoders and the complexity of the models."
)
UNRELATED = (
    "We present a novel computer vision algorithm to annotate a large database of one million "
    "images of facial expressions of emotion in the wild. Annotations include Action Units and "
    "their intensities as well as emotion category."
)

# 阈值只要"分得开"就算过；实测值会打印出来，用来和 API 版对照
MIN_EMBEDDING_MARGIN = 0.02
MIN_RERANK_MARGIN = 0.10


def main() -> int:
    settings = get_settings()
    print(f"模型缓存目录：{settings.model_cache_dir}")
    print(f"embedding 模型：{settings.embedding_local_model}（{settings.embedding_provider=}）")
    print(f"rerank 模型   ：{settings.rerank_local_model}（{settings.rerank_provider=}）")
    print()

    ok = True
    ok &= _check_embedding(settings)
    ok &= _check_rerank(settings)

    print()
    print("结果：" + ("通过" if ok else "不通过 —— 别拿它跑评测，先查模型/依赖"))
    return 0 if ok else 1


def _check_embedding(settings) -> bool:
    from app.providers.embedder import Embedder

    started = time.perf_counter()
    embedder = Embedder(settings.model_copy(update={"embedding_provider": "local"}))
    vectors = embedder.embed_documents([QUERY, RELEVANT, UNRELATED])
    elapsed = time.perf_counter() - started
    dim = len(vectors[0])
    relevant = _cosine(vectors[0], vectors[1])
    unrelated = _cosine(vectors[0], vectors[2])
    margin = relevant - unrelated
    print(f"[embedding] 维度={dim}（配置 {settings.embedding_dim}）耗时={elapsed:.1f}s")
    print(f"            相关={relevant:.4f}  无关={unrelated:.4f}  差距={margin:.4f}")
    passed = dim == settings.embedding_dim and margin >= MIN_EMBEDDING_MARGIN
    print(f"            → {'通过' if passed else '不通过'}（门槛 {MIN_EMBEDDING_MARGIN}）")
    return passed


def _check_rerank(settings) -> bool:
    started = time.perf_counter()
    reranker = Reranker(settings.model_copy(update={"rerank_provider": "local_bge"}))
    scores = reranker.rerank(QUERY, [RELEVANT, UNRELATED])
    elapsed = time.perf_counter() - started
    margin = scores[0] - scores[1]
    print(f"[rerank]    相关={scores[0]:.4f}  无关={scores[1]:.4f}  差距={margin:.4f}  耗时={elapsed:.1f}s")
    passed = all(0.0 <= score <= 1.0 for score in scores) and margin >= MIN_RERANK_MARGIN
    print(f"            → {'通过' if passed else '不通过'}（门槛 {MIN_RERANK_MARGIN}，且分数必须在 0~1）")
    return passed


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
