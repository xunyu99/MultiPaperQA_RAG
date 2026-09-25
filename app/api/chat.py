r"""`POST /chat` 的流式协议（SSE）。

**为什么是 SSE 不是 WebSocket**：这里只有服务器往客户端单向推，SSE 就是
`text/event-stream` 里的几行文本，浏览器 `EventSource` 原生支持，FastAPI 用
`StreamingResponse` 直接就能写 —— 不需要引 `sse-starlette`，也不需要握手、心跳、
重连那套。要双向（比如打断生成）再换 WebSocket。

**事件顺序（前端照着这个写就行）**：

```
event: evidence        <- 先给证据：检索结果、卡片、引用映射（前端可以立刻渲染来源）
data: {"cards": [...], "citations": {...}, "scope": [...]}

event: token           <- 回答一个字一个字来，可能有 0~N 个
data: {"text": "..."}

event: done            <- 生成结束，带上引用校验（哪些 [En] 是真用到了）
data: {"answer": "...", "citations": {...}, "latency_ms": 1234}
```

出错时发一个 `event: error`，**不抛异常** —— 流已经开了，HTTP 状态码改不了，
只能用一个事件把错误说清楚。

一条硬规矩：**每个 `data:` 行必须是单行**（SSE 靠换行分帧）。所以 JSON 一律
`ensure_ascii=True` 压成一行，中文变成 `\uXXXX`，前端 `JSON.parse` 会自动还原。

（这个 docstring 前面的 `r` 不能省：里面写了 `\uXXXX`，普通字符串会当成转义序列，
Python 直接 `SyntaxError` —— 和 Windows 路径写 `\.` 是同一个坑。）
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable


def sse_frame(event: str, data: dict[str, Any]) -> str:
    """拼一个 SSE 帧。`ensure_ascii=True` 是为了保证单行（见模块开头）。"""
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def card_payload(cards: Iterable[Any], text_limit: int | None = None) -> list[dict[str, Any]]:
    """证据卡 -> 前端能直接用的结构（**含溯源坐标**）。

    `trace` 里每项是 `{block_id, page_idx, bbox}`，前端拿 `page_idx` 定位页码、
    拿 `bbox` 画高亮框（归一化 0-1000，见 PLAN §2.4）。
    资产只给 id + 类型 + caption：**图片路径之类的不进这里**，前端要用再按 id 拿。

    `text_limit` 用来存历史快照：存进会话里的卡片要短（只够回看，不必是全文），
    而流式返回给前端的卡片给全文。
    """
    out: list[dict[str, Any]] = []
    for card in cards:
        text = card.text
        if text_limit is not None and len(text) > text_limit:
            text = text[:text_limit] + "…"
        out.append(
            {
                "label": card.label,
                "chunk_id": card.chunk_id,
                "paper_id": card.paper_id,
                "paper_title": card.paper_title,
                "section_id": card.section_id,
                "section_title": card.section_title,
                "page_start": card.page_start,
                "page_end": card.page_end,
                "similarity": round(card.similarity, 4),
                # 排序分：真正决定名次的那个分（重排分；重排没跑就是合并名次分）。
                # 卡片上显示它，不显示余弦 —— 别拿余弦冒充相关度（见 PLAN Step 9）。
                "rank_score": round(float(getattr(card, "rank_score", 0.0) or 0.0), 4),
                "text": text,
                "assets": [
                    {
                        "asset_id": asset.get("asset_id"),
                        "asset_type": asset.get("asset_type"),
                        "caption": asset.get("caption"),
                    }
                    for asset in card.assets
                ],
                "trace": card.trace,
            }
        )
    return out


def citation_payload(report: Any) -> dict[str, Any]:
    """引用校验 -> 前端结构：档位、拒答原因、哪些 `[En]` 存在、用到了哪些、有没有编造。

    `mode` 是 Step 9 的档位（`grounded` / `unknown`），`unknown_reason` 决定前端那句措辞
    （不在范围 / 库里没有 / 依据不足 / 生成失败 / 闲聊）—— 前端只认这两个字段，
    文案不写死在后端。
    """
    return {
        "cited": list(getattr(report, "cited", []) or []),
        "unknown": list(getattr(report, "unknown", []) or []),
        "insufficient": bool(getattr(report, "insufficient", False)),
        "coverage": round(float(getattr(report, "coverage", 0.0) or 0.0), 4),
        "mode": getattr(report, "mode", None),
        "unknown_reason": getattr(report, "unknown_reason", None),
        "downgraded": bool(getattr(report, "downgraded", False)),
    }


async def stream_events(
    *,
    context: dict[str, Any],
    answer_stream: AsyncIterator[str],
    finalize: Any,
) -> AsyncIterator[str]:
    """把三样东西拼成 SSE 帧流：证据 -> 增量 -> 收尾。

    抽成独立函数是为了**能在测试里不碰网络地跑**：塞一个假的 `answer_stream`
    进去，帧格式、拼接、收尾全都真的走一遍。
    """
    yield sse_frame("evidence", context)

    parts: list[str] = []
    try:
        async for piece in answer_stream:
            parts.append(piece)
            yield sse_frame("token", {"text": piece})
    except Exception as exc:  # noqa: BLE001 - 流已开，只能用事件报错
        yield sse_frame("error", {"stage": "generate", "message": f"{type(exc).__name__}: {exc}"})
        return

    answer = "".join(parts).strip()
    try:
        payload = finalize(answer)
    except Exception as exc:  # noqa: BLE001
        yield sse_frame("error", {"stage": "finalize", "message": f"{type(exc).__name__}: {exc}"})
        return
    yield sse_frame("done", payload)
