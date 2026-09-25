r"""探测 DeepSeek 生成通道的思考模式开关。

跑法（项目根目录）：
    .\.venv\Scripts\python.exe -m scripts.check_thinking

做三件事：
  1. 列出这个 key 能用的模型名（看有没有非思考的版本）
  2. 依次尝试几种"关思考"的参数写法，每个都真实发一次请求
  3. 打印耗时、token 用量、是否返回 reasoning_content

判定：参数被接受 + 没有 reasoning_content + 耗时明显下降 → 那个写法是对的，
把它填进 .env 的 LLM_EXTRA_BODY。
"""

from __future__ import annotations

import json
import sys
import time

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from app.config import get_settings

console = Console()

PROMPT = "1+1 等于几？只回答数字。"

CANDIDATES: list[tuple[str, dict]] = [
    ("baseline（不加参数）", {}),
    ('thinking={"type":"disabled"}', {"thinking": {"type": "disabled"}}),
    ("enable_thinking=False", {"enable_thinking": False}),
    ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
    ('reasoning_effort="none"', {"reasoning_effort": "none"}),
]


def main() -> int:
    settings = get_settings()
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
    )

    console.print("[bold]1) 这个 key 能用的模型[/bold]")
    try:
        ids = [m.id for m in client.models.list().data]
        console.print("   " + ", ".join(ids))
    except Exception as exc:
        console.print(
            f"   [yellow]列模型失败（不影响后续探测）：{type(exc).__name__}: {str(exc)[:120]}[/yellow]"
        )

    console.print(f"\n[bold]2) 逐个试关思考的写法（模型 {settings.llm_model}）[/bold]")
    table = Table()
    table.add_column("写法", style="cyan")
    table.add_column("思考状态", style="green")
    table.add_column("耗时", style="dim")
    table.add_column("token 用量", style="dim")

    winner: tuple[str, dict] | None = None
    for name, extra in CANDIDATES:
        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=64,
                extra_body=extra or None,
            )
        except Exception as exc:
            table.add_row(name, f"[red]{type(exc).__name__}[/red]", "-", str(exc)[:60])
            continue

        cost = time.perf_counter() - start
        message = resp.choices[0].message
        raw = message.model_dump()
        reasoning = raw.get("reasoning_content") or ""
        usage = resp.usage

        detail = f"in={usage.prompt_tokens} out={usage.completion_tokens}"
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
        if reasoning_tokens:
            detail += f" reasoning={reasoning_tokens}"

        if reasoning:
            state = f"[yellow]思考 {len(reasoning)} 字[/yellow]"
        else:
            state = "[green]无思考[/green]"
            if winner is None:
                winner = (name, extra)

        table.add_row(name, state, f"{cost:.2f}s", detail)
        console.print(f"    -> {(message.content or '').strip()[:50]!r}")
        if not extra:
            console.print(f"    -> baseline 返回字段: {sorted(raw.keys())}")

    console.print(table)

    if winner:
        name, extra = winner
        console.print(f"\n[green]结论：{name} 可用[/green]")
        console.print("把这一行写进 .env（去掉行首的 #）：")
        console.print(f"LLM_EXTRA_BODY={json.dumps(extra, ensure_ascii=False)}")
    else:
        console.print("\n[yellow]没探测到能关掉思考的写法。[/yellow]")
        console.print("备选：看第 1 步的模型列表里有没有非思考版本，把 LLM_MODEL 换成它。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
