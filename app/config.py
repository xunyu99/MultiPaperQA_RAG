"""全局配置：唯一从 .env 读数的地方。

规则：业务代码不允许直接 os.getenv，一律 get_settings().xxx，
这样换模型/换 key 只改 .env，不碰业务代码。

两条模型通道，各用各的 key / base_url / 模型：
  - 规划通道（DashScope / Qwen）：结构化路由、查询改写。顺带提供 embedding。
  - 生成通道（DeepSeek）        ：正文回答 + 看图。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ProviderConfig(BaseModel):
    """一个 OpenAI 兼容端点的连接参数。"""

    api_key: str = ""
    base_url: str = ""
    proxy: str | None = None
    timeout: float = 90.0
    max_retries: int = 2
    # 厂商私有参数，原样塞进请求体。例：{"thinking": {"type": "disabled"}}
    extra_body: dict[str, Any] = {}


class Settings(BaseSettings):
    """环境变量名不区分大小写，字段名小写即可自动对上，无需 alias。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 规划通道：DashScope / Qwen（同时提供 embedding）----------
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_proxy: str | None = None
    planner_model: str = "qwen-flash"

    # ---------- 生成通道：DeepSeek ----------
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_proxy: str | None = None
    llm_timeout: float = 90.0
    llm_max_retries: int = 2
    # 关思考模式等厂商私有参数，写成一整块传进去。
    # 正确写法先用 scripts/check_thinking.py 探测，再填到 .env。
    # 例：LLM_EXTRA_BODY={"thinking": {"type": "disabled"}}
    llm_extra_body: dict[str, Any] = {}

    # ---------- 向量模型（复用 DASHSCOPE_API_KEY）----------
    embedding_provider: Literal["api", "local"] = "api"
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024
    embedding_batch_size: int = 8
    embedding_local_model: str = "BAAI/bge-m3"

    # ---------- 重排（Step 12 才用）----------
    rerank_enabled: bool = True
    # "dashscope" = 百炼 text-rerank（默认，免下载）；"local_bge" = 本地 ONNX，见 BACKLOG
    rerank_provider: str = "dashscope"
    # 实测对比（相关段落 / 无关段落 / 中文提问打英文段落）：
    #   qwen3.7-text-rerank  0.9888 / 0.0332 / 0.8445   ← 分数量纲可解释，选它
    #   gte-rerank-v2        0.5026 / 0.0060 / 0.1714
    # 注意 qwen3-reranker-* 这些名字会返回 "Model not exist"，OpenAI 兼容面的
    # /rerank 是 404 —— 只有下面这个原生端点 + qwen3.7-text-rerank 可用。
    rerank_model: str = "qwen3.7-text-rerank"
    # 本地重排模型（RERANK_PROVIDER=local_bge 时用）。选多语言版而不是单语 base 版：
    # 实测 bge-reranker-base 在"中文问 × 英文段落"上只差 0.005，等于没打分（见 providers/reranker.py）。
    rerank_local_model: str = "BAAI/bge-reranker-v2-m3"
    # **不是** OpenAI 兼容面的地址，百炼的 rerank 是独立服务
    rerank_base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    rerank_timeout: float = 30.0
    model_cache_dir: Path = PROJECT_ROOT / "storage" / "models"

    # ---------- MinerU 解析 ----------
    mineru_api_key: str = ""
    mineru_base_url: str = "https://mineru.net/api/v4"
    mineru_proxy: str | None = None
    mineru_model_version: str = "vlm"
    mineru_language: str = "ch"
    mineru_enable_formula: bool = True
    mineru_enable_table: bool = True
    mineru_is_ocr: bool = False
    mineru_poll_interval_seconds: int = 5
    mineru_poll_timeout_seconds: int = 900

    # ---------- 存储 ----------
    storage_dir: Path = PROJECT_ROOT / "storage"
    db_path: Path = PROJECT_ROOT / "storage" / "app.db"
    # Chroma collection 名。**双 collection 对照**用：API embedding 用 "chunks"，
    # 本地 bge-m3 另起一个（如 "chunks_bge_m3"）—— 换 embedding 必须换集合，
    # 否则会把现有向量覆盖掉，且不可逆（EVAL_PLAN §4.3）。
    # 走环境变量就能切：CHROMA_COLLECTION=chunks_bge_m3 EMBEDDING_PROVIDER=local
    chroma_collection: str = "chunks"

    # ---------- 切分 ----------
    chunk_target_tokens: int = 800
    chunk_max_tokens: int = 1200
    chunk_min_tokens: int = 200
    # chunk 之间不重叠。经典 RAG 的 10-20% overlap 针对"按字符硬切"，
    # 我们按块切，边界天然落在段落之间，不需要缝合（见 PLAN Step 3）。
    block_overlap: int = 0
    # 只有"被切开的长正文块"内部才带 overlap，按句子取够这么多字符
    text_overlap_chars: int = 150

    # ---------- 检索 ----------
    candidate_k: int = 50
    # 单篇 / 全库返回几条。
    # 试过 3：Recall@3 = 60%，比 5 条的 80% 掉了两道题（gold 都在第 4~5 名），
    # 省下的 ~950 token 在 64k 上下文里只占 1.5% —— 不划算，退回 5。
    # 真想只喂 3 条，走 B3 rerank（召回 5 -> 重排 -> 取 3），别把召回也砍掉。
    top_k: int = 5
    # 多篇时**每篇**几条（总数 = 篇数 x 这个值）。对比题要的是"每篇都有代表"，
    # 所以给每篇固定名额，而不是按相似度全局截断（后者会让一篇吃光名额）。
    # 比单篇少是对的（一篇 5 条时，对比题有一半是同一篇的），但**不能只给 2**：
    # 实测每篇 2 条时，pe-clip 的两个名额被同一个 4.5 Visualization 全占了。
    # 3 条 = 每篇留一条容错。等 B3 rerank 上了，这里给它更宽的候选（5），重排后再裁到 3。
    per_paper_k: int = 3
    # 同一篇论文的同一个章节最多占几个 top-k 名额。0 = 关（v1 老行为）。
    # 长章节会切成好几个 chunk，它们内容相邻、向量也像，经常手拉手占满名额。
    section_cap: int = 0
    # 双路召回：**每路每篇**取几条候选（候选深度，不是最终返回条数）。
    # 单篇 6 + 6 = 12 条进融合；多篇每篇 5 + 5 = 10 条。
    # 多篇给得少一点：候选总数是 篇数 x 深度，而最终每篇只留 PER_PAPER_K 条。
    channel_k: int = 6
    channel_k_multi: int = 5
    # 关键词通道开关。做 A/B（双路 vs 纯向量）时用，评测脚本靠它切换。
    keyword_enabled: bool = True
    rrf_k: int = 60
    rerank_top_k: int = 5

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def chroma_dir(self) -> Path:
        """Chroma 的持久化目录。跟着 storage_dir 走，方便整体搬走。"""
        return self.storage_dir / "chroma"

    @property
    def planner_provider(self) -> ProviderConfig:
        """规划通道连接参数（Qwen）。"""
        return ProviderConfig(
            api_key=self.dashscope_api_key,
            base_url=self.dashscope_base_url,
            proxy=self.dashscope_proxy,
            timeout=self.llm_timeout,
            max_retries=self.llm_max_retries,
        )

    @property
    def generation_provider(self) -> ProviderConfig:
        """生成通道连接参数（DeepSeek），带关思考等私有参数。"""
        return ProviderConfig(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            proxy=self.llm_proxy,
            timeout=self.llm_timeout,
            max_retries=self.llm_max_retries,
            extra_body=self.llm_extra_body,
        )

    def ensure_dirs(self) -> None:
        """建好运行期需要的目录，幂等，随便调。"""
        for path in (
            self.storage_dir,
            self.storage_dir / "pdfs",
            self.storage_dir / "mineru",
            self.model_cache_dir,
            self.chroma_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
