"""裁判的**接线与解析**测试：全用假模型，不联网、不花钱。

判据本身好不好使由 `scripts/check_judge.py`（自检脚本）验证 —— 那一步要联网。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.eval.judge import Judge, parse_label


class _FakeLLM:
    """假模型：记录收到的 prompt，返回预设 JSON。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return type("Reply", (), {"content": self.replies.pop(0)})()


def _judge(tmp_path: Path, replies: list[str]) -> tuple[Judge, _FakeLLM]:
    llm = _FakeLLM(replies)
    settings = Settings(dashscope_api_key="x")
    return Judge(settings=settings, llm=llm, cache_path=tmp_path / "judge_cache.json"), llm


# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------
def test_parse_label_strict_json() -> None:
    assert parse_label('{"label": 1, "reason": "ok"}') == (1, "ok")


def test_parse_label_tolerates_code_fence_and_prose() -> None:
    """模型偶尔会加一句客套话或包 ```json —— 抓第一段 {...}。"""
    raw = 'Sure!\n```json\n{"label": 0, "reason": "evidence misses the number"}\n```'
    assert parse_label(raw) == (0, "evidence misses the number")


def test_parse_label_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_label("我觉得这个答案还行")
    with pytest.raises(ValueError):
        parse_label('{"label": 2, "reason": "x"}')


# ----------------------------------------------------------------------
# 调用与缓存
# ----------------------------------------------------------------------
def test_groundedness_passes_content_to_the_prompt(tmp_path: Path) -> None:
    judge, llm = _judge(tmp_path, ['{"label": 1, "reason": "supported"}'])
    result = judge.groundedness("Q?", "EVIDENCE-TEXT", "ANSWER-TEXT", question_id="q1")
    assert (result.label, result.reason, result.cached) == (1, "supported", False)
    assert "EVIDENCE-TEXT" in llm.prompts[0] and "ANSWER-TEXT" in llm.prompts[0]


def test_correctness_passes_reference_to_the_prompt(tmp_path: Path) -> None:
    judge, llm = _judge(tmp_path, ['{"label": 0, "reason": "wrong number"}'])
    result = judge.correctness("Q?", "REFERENCE-TEXT", "ANSWER-TEXT", question_id="q1")
    assert result.label == 0
    assert "REFERENCE-TEXT" in llm.prompts[0]


def test_same_answer_is_served_from_cache(tmp_path: Path) -> None:
    """同一份答案第二次判定必须走缓存 —— M2 会反复跑同一批回答，这是省钱的关键。"""
    judge, llm = _judge(tmp_path, ['{"label": 1, "reason": "supported"}'])
    first = judge.groundedness("Q?", "E", "A", question_id="q1")
    second = judge.groundedness("Q?", "E", "A", question_id="q1")

    assert first.cached is False and second.cached is True
    assert len(llm.prompts) == 1, "第二次不该再调模型"


def test_different_answer_is_judged_again(tmp_path: Path) -> None:
    judge, llm = _judge(
        tmp_path, ['{"label": 1, "reason": "a"}', '{"label": 0, "reason": "b"}']
    )
    judge.groundedness("Q?", "E", "A1", question_id="q1")
    judge.groundedness("Q?", "E", "A2", question_id="q1")
    assert len(llm.prompts) == 2


def test_cache_survives_new_judge_instance(tmp_path: Path) -> None:
    """缓存落盘：换一个 Judge 实例（模拟重跑）也不重复花钱。"""
    cache = tmp_path / "judge_cache.json"
    settings = Settings(dashscope_api_key="x")
    first_llm = _FakeLLM(['{"label": 1, "reason": "ok"}'])
    Judge(settings=settings, llm=first_llm, cache_path=cache).groundedness("Q?", "E", "A", "q1")

    second_llm = _FakeLLM([])  # 没有可用的回复：一旦真调用就会 IndexError
    result = Judge(settings=settings, llm=second_llm, cache_path=cache).groundedness("Q?", "E", "A", "q1")
    assert result.cached is True and second_llm.prompts == []
