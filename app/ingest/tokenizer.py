"""token 数估算。

**这是估算，不是精确值。** 真正的 tokenizer 是模型私有的（DeepSeek / Qwen 各一套），
本机装不了也用不上 —— 切分只需要一个稳定的尺度，不需要和某个模型的 tokenizer 对齐。

估算规则：

    中日韩字符   1 字 ≈ 1 token
    其余（英文、数字、符号）  ≈ 3.6 字符 / token

这两个数在国内几个常见 tokenizer 上偏保守（估出来略多于真实值），而切分时
"宁可少装一点"是安全的方向：估多了只是 chunk 略小，估少了会撑爆上下文。
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")

CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str | None) -> int:
    """估算文本的 token 数。空文本算 0。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return int(cjk + other / CHARS_PER_TOKEN) + 1


def chars_for_tokens(tokens: int) -> int:
    """反推：这么多 token 大约能装多少字符。只用于"连句子边界都找不到"的兜底切分。"""
    return max(1, int(tokens * CHARS_PER_TOKEN))
