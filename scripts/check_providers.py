r"""Step 1 验收：确认 key、网络、模型名真实可用。

跑法（项目根目录）：
    .\.venv\Scripts\python.exe -m scripts.check_providers

四道关卡：文本生成 / 结构化输出 / 向量 Embedding / 看图。
任何一道失败都会打印真实报错，不吞异常。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.providers.embedder import Embedder
from app.providers.llm import LLMClient

console = Console()

EXPECTED_NUMBERS = ["71", "83", "95"]
CHART_PATH = Path(__file__).resolve().parent.parent / "storage" / "check_chart.png"


class SmokePlan(BaseModel):
    """连通性自检用的最小 schema。

    故意只放两个字段：一个字符串、一个整数。
    如果模型能按这个 schema 返回，说明 function calling 通道可用。
    """

    topic: str = Field(description="用一句话概括主题")
    difficulty: int = Field(description="1-5 的整数，越大越难")


def mask(secret: str) -> str:
    if not secret:
        return "(空)"
    return f"{secret[:6]}...{secret[-4:]}" if len(secret) > 12 else "***"


def make_test_chart(path: Path) -> None:
    """画三根柱子、下面标数字，用来验证模型真的在看图而不是瞎猜。"""
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (420, 250), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:  # 老版本 Pillow 不支持 size 参数
        font = ImageFont.load_default()

    draw.text((20, 14), "Accuracy on three datasets", fill="black", font=font)
    for x, value, color in [(40, 71, "#4C78A8"), (160, 83, "#F58518"), (280, 95, "#54A24B")]:
        height = int(value / 100 * 130)
        draw.rectangle([x, 190 - height, x + 70, 190], fill=color)
        draw.text((x + 20, 200), str(value), fill="black", font=font)
    img.save(path)


def check_text(settings) -> tuple[str, str]:
    client = LLMClient(settings)
    start = time.perf_counter()
    text = client.chat("用一句话解释什么是检索增强生成（RAG）。", max_tokens=120)
    cost = time.perf_counter() - start
    return text.strip()[:60], f"{cost:.2f}s / {settings.llm_model}"


def check_structured(settings) -> tuple[str, str]:
    client = LLMClient(settings)
    start = time.perf_counter()
    plan = client.structured(
        SmokePlan,
        "把这句话转成结构化结果：论文里的表格和公式很难切分。",
    )
    cost = time.perf_counter() - start
    # plan 是 SmokePlan 对象，不是 dict —— 类型已经由 pydantic 校验过
    return str(plan), f"{cost:.2f}s / {settings.planner_model}"


def check_embedding(settings) -> tuple[str, str]:
    embedder = Embedder(settings)
    start = time.perf_counter()
    docs = embedder.embed_documents(["表格跨页会被解析成两个块", "公式必须作为整体切分"])
    query = embedder.embed_query("为什么公式不能切断")
    cost = time.perf_counter() - start
    return f"docs={len(docs)} dim={len(query)}", f"{cost:.2f}s / {settings.embedding_model}"


def check_image(settings) -> tuple[str, str]:
    make_test_chart(CHART_PATH)
    client = LLMClient(settings)
    start = time.perf_counter()
    answer = client.ask_with_image(
        "图中三根柱子下方的数字分别是多少？只按从左到右的顺序输出三个数字，用逗号分隔。",
        CHART_PATH,
    )
    cost = time.perf_counter() - start
    hits = sum(1 for n in EXPECTED_NUMBERS if n in answer)
    return f"{answer.strip()[:60]}（命中 {hits}/3）", f"{cost:.2f}s / {settings.llm_model}"


def main() -> int:
    settings = get_settings()
    settings.ensure_dirs()

    console.print("[bold]Step 1 · 连通性自检[/bold]")
    console.print(f"  规划模型 : {settings.planner_model} @ {settings.dashscope_base_url}")
    console.print(f"  规划 key : {mask(settings.dashscope_api_key)}")
    console.print(f"  生成模型 : {settings.llm_model} @ {settings.llm_base_url}")
    console.print(
        f"  生成 key : {mask(settings.llm_api_key)}   代理: {settings.llm_proxy or '(未设置)'}"
    )
    console.print(
        f"  向量模型 : {settings.embedding_model}（{settings.embedding_dim} 维）"
        f" @ {settings.embedding_base_url}"
    )
    console.print("  向量 key : 复用 DASHSCOPE_API_KEY")
    if settings.llm_extra_body:
        console.print(f"  生成通道私有参数 : {settings.llm_extra_body}")
    console.print()

    if not settings.llm_api_key:
        console.print("[red]LLM_API_KEY 是空的，先填 .env。[/red]")
        return 1
    if not settings.dashscope_api_key:
        console.print("[red]DASHSCOPE_API_KEY 是空的，Planner 和向量都用它，先填 .env。[/red]")
        return 1

    checks = [
        ("文本生成", lambda: check_text(settings)),
        ("结构化输出", lambda: check_structured(settings)),
        ("向量 Embedding", lambda: check_embedding(settings)),
        ("看图（生成模型）", lambda: check_image(settings)),
    ]

    table = Table()
    table.add_column("检查项", style="cyan", no_wrap=True)
    table.add_column("结果", style="green")
    table.add_column("详情", style="dim")

    failed = 0
    for name, fn in checks:
        try:
            result, detail = fn()
            table.add_row(name, result, detail)
        except Exception as exc:  # 不吞异常，打印真实原因
            failed += 1
            table.add_row(
                name, "[red]失败[/red]", f"[red]{type(exc).__name__}: {str(exc)[:120]}[/red]"
            )
    console.print(table)

    if failed:
        console.print(f"\n[red]{failed} 项失败[/red]，对照排查：")
        console.print("  401 / invalid key  → 对应服务的 key 不对")
        console.print("  Connection error   → 网络或代理，考虑在 .env 里设 LLM_PROXY")
        console.print("  model not found    → 模型名或开通情况")
        console.print("  维度不一致          → 改 .env 里的 EMBEDDING_DIM")
        console.print("  结构化输出失败      → 把报错贴我，看模型返回的原始 tool_calls")
        return 1

    console.print("\n[green]四项全通过，可以进 Step 2。[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
