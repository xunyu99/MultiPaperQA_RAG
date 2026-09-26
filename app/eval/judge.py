"""LLM 裁判：groundedness（忠实性）/ correctness（正确性）—— M2。

两个判据（EVAL_PLAN §2.4）：

- **groundedness**：答案是否只依据给定证据（幻觉检测），**不需要参考答案**；
- **correctness**：答案与参考答案的**事实**是否一致，需要 `reference`。

五条刻意的决定（每条都有来由）：

1. **prompt 用英文模板**：中文提示词经控制台/管道传递时会被编码搞成乱码 —— 实测模型收到
   一串问号、回我"您输入的是一串问号"。所以模板全英文，中文内容（问题/证据/答案）作参数传入。
2. **二值 0/1 + 一句理由**：比 1–5 连续分稳（模型对"几分"没有校准），答辩也说得清分数怎么来的。
3. **判据写死在 prompt 里，专治"偏严"**：实测 `qwen3.8-max` 会把
   "证据说 DFEW 上 batch size=8，答案说 PE-CLIP 用了 8" 判成 0 —— 那是张冠李戴的误伤
   （PE-CLIP 就是在 DFEW 上训练的）。所以明确写：论文自身设置 / 数据集配置算支持；
   只有"证据里没有这个事实"或"安到别的论文头上"才判 0。拒答也不算幻觉（判 1）。
4. **只输出 JSON，但解析容错**：模型偶尔包 ```json 代码块或加一句话 → 抓第一段 `{...}`。
5. **缓存**：key = `(题号, 答案哈希, 裁判, prompt 版本, 模型)`，落 `.eval_out/judge_cache.json`。
   重跑同一份答案不重复花钱 —— M2 会反复跑同一批回答，这条是省钱的关键。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

# prompt 改了就要改版本号：缓存 key 里有它，否则旧判定会被当成本次的结论
PROMPT_VERSION = "judge_v1"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / ".eval_out" / "judge_cache.json"

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

GROUNDEDNESS_PROMPT = """You are a strict but fair fact-checker for a retrieval-augmented QA system about research papers.

Task: decide whether the ANSWER is fully supported by the EVIDENCE.

Rules:
1. Supported means: every factual claim in the ANSWER can be found in the EVIDENCE.
2. Evidence about a dataset, table or experiment of the paper being asked about DOES support an answer
   that states that paper's own setting. Example: evidence "the batch size is set to 8 for DFEW"
   supports "PE-CLIP used batch size 8", because that paper trains on DFEW.
3. Label 0 only when (a) the ANSWER states facts the EVIDENCE does not contain, or
   (b) facts are attributed to the wrong paper or method.
4. A refusal such as "insufficient information" is NOT a hallucination: label 1.
5. Do not judge style, completeness of the answer, or language.

Reply with JSON only, no code fences, no extra text:
{{"label": 1, "reason": "<one short sentence>"}}

QUESTION:
{question}

EVIDENCE:
{evidence}

ANSWER:
{answer}
"""

CORRECTNESS_PROMPT = """You are a strict but fair fact-checker for a retrieval-augmented QA system about research papers.

Task: compare the ANSWER with the REFERENCE on FACTS (numbers, names, methods, conclusions).

Rules:
1. Label 1 when the ANSWER agrees with the REFERENCE on every fact the question asks for,
   even if wording, ordering or extra correct detail differ.
2. Label 0 when a required fact is missing, wrong, or contradicts the REFERENCE.
3. A refusal such as "insufficient information" is a 0 when the REFERENCE does contain the answer.
4. Ignore style, length and language.

Reply with JSON only, no code fences, no extra text:
{{"label": 1, "reason": "<one short sentence>"}}

QUESTION:
{question}

REFERENCE:
{reference}

ANSWER:
{answer}
"""


@dataclass
class JudgeResult:
    """label：1 = 通过（有据 / 正确），0 = 不通过，None = 这次没判出来（见 reason）。"""

    label: int | None
    reason: str
    judge: str
    cached: bool = False


# 裁判是**评测里的辅助步骤**，它挂掉不能把跑了半小时的全量任务一起带走
# （2026-09-26 实测：一次 OpenAITimeoutError 让 50 题的任务在第 40 多题整体崩了，
#  dump 都没落盘）。所以：失败重试一次，再失败就记 None，从分母里排除。
JUDGE_ATTEMPTS = 2


def parse_label(raw: str) -> tuple[int, str]:
    """从模型输出里抠出 (label, reason)。容忍代码块和多余的客套话。"""
    match = _JSON_BLOCK.search(raw or "")
    if match is None:
        raise ValueError(f"裁判没有给出 JSON：{(raw or '')[:120]!r}")
    payload = json.loads(match.group(0))
    label = int(payload.get("label"))
    if label not in (0, 1):
        raise ValueError(f"label 只能是 0/1，收到 {label!r}")
    return label, str(payload.get("reason") or "").strip()


class Judge:
    """两个裁判的入口。`llm` 可注入 —— 单测用假模型，不联网、不花钱。"""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: Any | None = None,
        cache_path: Path | str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        self.model = self.settings.judge_model
        # 裁判要读整张证据卡（几千 token），planner 默认超时（90s）实测不够用 ——
        # qwen3.8-max 在长输入上会超过它，然后整批任务被打断。这里单独放宽，
        # 宁可这条判得慢，也不要它 500 掉。
        self.timeout = 240.0
        self.cache_path = Path(cache_path or DEFAULT_CACHE_PATH)
        self._cache: dict[str, dict[str, Any]] = _load_cache(self.cache_path)

    # ---- 两个判据 ----
    def groundedness(
        self, question: str, evidence: str, answer: str, question_id: str = ""
    ) -> JudgeResult:
        prompt = GROUNDEDNESS_PROMPT.format(question=question, evidence=evidence, answer=answer)
        return self._judge("groundedness", prompt, question_id, answer)

    def correctness(
        self, question: str, reference: str, answer: str, question_id: str = ""
    ) -> JudgeResult:
        prompt = CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer)
        return self._judge("correctness", prompt, question_id, answer)

    # ---- 内部 ----
    def _client(self) -> Any:
        if self._llm is None:
            # 裁判走**独立**模型（planner 通道 = 百炼），不用生成模型自评
            from app.providers.llm import LLMClient

            # planner_provider 的 timeout 取的就是 llm_timeout 这个字段（见 config.planner_provider）
            settings = self.settings.model_copy(update={"llm_timeout": self.timeout})
            self._llm = LLMClient(settings).build_planner(model=self.model)
        return self._llm

    def _judge(self, name: str, prompt: str, question_id: str, answer: str) -> JudgeResult:
        # key 取**整段 prompt** 的哈希：问题、证据、参考答案、答案全在里面。
        # 2026-09-26 修：以前用"答案哈希"，结果改了 reference 之后缓存照旧命中 ——
        # asset_1 因此拿到一条**旧 reference 时代**的判定（模型明明把表 2 的 9 行都列了，
        # 却被按"漏了初始学习率"判 0，而那句早被我删掉了）。
        # 缓存必须覆盖判官的**全部输入**，否则就是拿旧标准判新答案。
        key = _cache_key(name, self.model, PROMPT_VERSION, question_id, prompt)
        hit = self._cache.get(key)
        if hit is not None:
            return JudgeResult(
                label=int(hit["label"]), reason=str(hit.get("reason") or ""),
                judge=name, cached=True,
            )

        last_error: Exception | None = None
        for attempt in range(JUDGE_ATTEMPTS):
            try:
                reply = self._client().invoke(prompt)
                raw = str(getattr(reply, "content", reply)).strip()
                label, reason = parse_label(raw)
            except Exception as exc:  # noqa: BLE001 - 网络/解析/超时都算"这次没判出来"
                last_error = exc
                if attempt + 1 < JUDGE_ATTEMPTS:
                    time.sleep(2.0)
                continue
            self._cache[key] = {"label": label, "reason": reason, "raw": raw[:500]}
            _save_cache(self.cache_path, self._cache)
            return JudgeResult(label=label, reason=reason, judge=name)

        return JudgeResult(
            label=None,
            reason=f"judge failed: {type(last_error).__name__}: {str(last_error)[:200]}",
            judge=name,
            cached=False,
        )


def _cache_key(judge: str, model: str, version: str, question_id: str, prompt: str) -> str:
    """key 里放**完整 prompt 的哈希** —— 判官看到的东西变了就必须重判。"""
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{judge}|{model}|{version}|{question_id}|{digest}"


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # 缓存坏了不该让评测崩，重算就是
        return {}


def _save_cache(path: Path, payload: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
