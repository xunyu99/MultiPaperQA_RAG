"""流式生成通道（`LLMClient.chat_stream`）—— 不联网，不调真模型。

这个文件是补出来的：原来接口层的测试用了一个"吐字符串"的假 LLM，
**恰好绕过了真正会出错的那一步**（把 `AIMessageChunk` 展平成文本）。
结果线上表现为"证据都返回了、回答是空的"，而且不报错。
所以这里必须用真的 `AIMessageChunk`，不能用字符串糊过去。
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessageChunk

from app.config import Settings
from app.providers.llm import LLMClient, flatten_content


def _collect(client: LLMClient) -> str:
    async def run() -> str:
        parts = [piece async for piece in client.chat_stream("问题", system="系统")]
        return "".join(parts)

    return asyncio.run(run())


class FakeStreamingModel:
    """只实现 `astream`，吐真的 AIMessageChunk（不是字符串）。"""

    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self.chunks = chunks

    async def astream(self, messages):  # noqa: ANN001
        for chunk in self.chunks:
            yield chunk


def test_flatten_content_accepts_a_message_object() -> None:
    """**这条就是那个 bug 的哨兵。**

    直接迭代 pydantic 的 message 对象拿到的是 `(字段名, 值)` 元组，
    展平出来是空字符串 —— 不报错，静默丢掉所有 token。
    """
    assert flatten_content(AIMessageChunk(content="你好")) == "你好"
    assert flatten_content(AIMessageChunk(content=[{"type": "text", "text": "多模态"}])) == "多模态"
    assert flatten_content("本来就是字符串") == "本来就是字符串"
    assert flatten_content([{"type": "text", "text": "块"}, "尾巴"]) == "块尾巴"


def test_chat_stream_yields_text(monkeypatch) -> None:  # noqa: ANN001
    client = LLMClient(Settings(_env_file=None))
    fake = FakeStreamingModel(
        [
            AIMessageChunk(content="这是"),
            AIMessageChunk(content=""),          # 空 chunk 要跳过，不能吐空串
            AIMessageChunk(content="答案[E1]。"),
        ]
    )
    monkeypatch.setattr(client, "_build", lambda *args, **kwargs: fake)
    assert _collect(client) == "这是答案[E1]。"


def test_chat_stream_handles_content_blocks(monkeypatch) -> None:  # noqa: ANN001
    """多模态模型按 content block 分片回来时也要能拼。"""
    client = LLMClient(Settings(_env_file=None))
    fake = FakeStreamingModel(
        [
            AIMessageChunk(content=[{"type": "text", "text": "上半句"}]),
            AIMessageChunk(content=[{"type": "text", "text": "下半句"}]),
        ]
    )
    monkeypatch.setattr(client, "_build", lambda *args, **kwargs: fake)
    assert _collect(client) == "上半句下半句"


class FlakyStreamingModel:
    """第一次一个 token 都不吐，之后正常 —— 模拟上游返回空 choices。"""

    def __init__(self, chunks: list[AIMessageChunk], empty_calls: int = 1) -> None:
        self.chunks = chunks
        self.empty_calls = empty_calls
        self.calls = 0

    async def astream(self, messages):  # noqa: ANN001
        self.calls += 1
        if self.calls <= self.empty_calls:
            return
        for chunk in self.chunks:
            yield chunk


def test_chat_stream_retries_once_when_nothing_came_back(monkeypatch) -> None:  # noqa: ANN001
    """模型一个字都没吐时重试一次 —— 这是"回答是空的"最常见的成因。"""
    client = LLMClient(Settings(_env_file=None))
    fake = FlakyStreamingModel([AIMessageChunk(content="重试后的答案")])
    monkeypatch.setattr(client, "_build", lambda *args, **kwargs: fake)
    assert _collect(client) == "重试后的答案"
    assert fake.calls == 2


def test_chat_stream_gives_up_after_one_retry(monkeypatch) -> None:  # noqa: ANN001
    """两次都空就认了 —— 上层会给兜底文案，不会把空串存进会话。"""
    client = LLMClient(Settings(_env_file=None))
    fake = FlakyStreamingModel([], empty_calls=99)
    monkeypatch.setattr(client, "_build", lambda *args, **kwargs: fake)
    assert _collect(client) == ""
    assert fake.calls == 2
