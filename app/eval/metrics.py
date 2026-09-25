"""评测指标 —— 都能自动算，不需要人工标注，也不需要 LLM 裁判。

为什么不用 LLM 当裁判：要多花钱、结果还不稳定，而且答辩时说不清分数怎么来的。
这三条都能从检索结果和回答文本里机械算出来：

1. **Recall@k**：top-k 里有没有命中"预期章节"。
   为什么不标 chunk_id：太累且一改切分参数就失效。为什么不标页码：一页常有好几个 chunk，
   太容易蒙中。**标章节**刚好 —— 既稳定（章节标题来自 MinerU）又能真的反映"找对地方了吗"。
2. **引用可定位率 / 编造引用**：解析回答里的 `[En]`，反查 chunk_id 是否存在。
   编造引用（`unknown` 非空）单列，它是幻觉的硬信号，不能混进平均值里被稀释。
3. **无依据率**：三种情况之一 —— ① 编造了引用编号；② 该拒答却挂了 grounded；
   ③ 挂了 grounded 却拿不出有效引用。**降级不算无依据**：降级本身就是正确处理
   （那段正文不会给用户看），它的次数单独用 `downgraded_count` 盯，应当恒为 0。

Step 9 之后多两个量：**拒答的 reason 命不命中**（`reason_accuracy`）和
**"拒答里还带引用"的条数**（`refusal_cited`，量的是模型听不听话，目标恒为 0）。

另加一个**回答率**（非错库题里给出实质答案的比例）：它和 Recall 是两件事 ——
检索对了但答不出来，说明生成那步有问题，分开看才知道该修哪一层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.generation.answerer import MODE_GROUNDED, MODE_UNKNOWN, CitationReport
from app.retrieval.retriever import RetrievedChunk

# 超过这么多字的回答，一条引用都没有就可疑（和 answerer 里保持一致的判据）
UNCITED_LENGTH_HINT = 60


@dataclass
class Outcome:
    """一道题的完整结果。"""

    question_id: str
    qtype: str
    question: str
    expect_insufficient: bool
    gold: list[dict[str, str]]
    scope: list[str]
    hits: list[RetrievedChunk]
    answer: str
    citations: CitationReport
    latency_ms: float = 0.0
    note: str = ""
    # 跨论文题：**每一篇论文都要有代表**才算召回成功（gold 里同一篇列多个候选章节时，
    # 那些候选之间是"或"）。默认是"命中任一即可"。
    require_all_papers: bool = False
    # 这道题期望的拒答 reason（只有两道拒绝题标了）—— 用来量"超纲"和"库里没有"分没分对
    expect_reason: str | None = None
    # 模型原始输出（没被降级/剥标记处理过）。落盘诊断用：对比它和 answer 就知道
    # 这一轮到底是拒答、还是被降级了。指标不使用它。
    raw_answer: str = ""

    @property
    def insufficient(self) -> bool:
        return self.citations.insufficient

    @property
    def mode(self) -> str:
        return self.citations.mode

    @property
    def unknown_reason(self) -> str | None:
        return self.citations.unknown_reason


@dataclass
class Report:
    total: int = 0
    # None = 这次没算（比如 --retrieval-only 不调 LLM），显示成「不适用」而不是 0%
    recall_at_k: float | None = None
    citation_rate: float | None = None
    unknown_citations: int = 0
    ungrounded_rate: float | None = None
    out_of_scope_accuracy: float | None = None
    answered_rate: float | None = None
    # 声明有据却拿不出引用（被降级）的次数 —— 应当恒为 0
    downgraded_count: int = 0
    # 拒答输出里**还带着 [En]** 的条数 —— 量模型听不听话，目标恒为 0
    refusal_cited: int = 0
    # 两道拒绝题的 reason 命中率（标了 expect_reason 的题才有数）
    reason_accuracy: float | None = None
    by_type: dict[str, dict[str, float | None]] = field(default_factory=dict)
    fails: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# 单题判定
# ----------------------------------------------------------------------
def hits_gold(outcome: Outcome, k: int) -> bool | None:
    """top-k 里有没有命中预期章节。没标 gold 的题（错库题）返回 None。"""
    if not outcome.gold:
        return None
    top = outcome.hits[:k]

    def matched(target: dict[str, str]) -> bool:
        return any(
            hit.paper.get("paper_id") == target["paper_id"]
            and target["section"] in (hit.section_title or "")
            for hit in top
        )

    if not outcome.require_all_papers:
        return any(matched(target) for target in outcome.gold)

    # 跨论文对比题：**按论文分组**，每组至少命中一条 —— 否则一篇占满 top-k、
    # 另一篇一条证据都没有，也算"过"，那就考不出配额策略了。
    for paper_id in {target["paper_id"] for target in outcome.gold}:
        candidates = [target for target in outcome.gold if target["paper_id"] == paper_id]
        if not any(matched(target) for target in candidates):
            return False
    return True


def is_ungrounded(outcome: Outcome) -> bool:
    """无依据：编造引用 / 挂了 grounded 却拿不出引用 / 该拒答却挂了 grounded。

    **降级（downgraded）不算无依据** —— 那是正确处置，正文已经被换成固定文案；
    次数由 `Report.downgraded_count` 单独统计。
    """
    if outcome.citations.unknown:
        return True
    if outcome.mode == MODE_GROUNDED and not outcome.citations.cited:
        return True
    if outcome.expect_insufficient and outcome.mode != MODE_UNKNOWN:
        return True  # 该拒答却答了实质内容
    return False


# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------
def evaluate(
    outcomes: list[Outcome],
    k: int = 5,
    retrieval_only: bool = False,
) -> Report:
    """`retrieval_only=True` 时只算检索层指标 —— 其余留 None，显示成「不适用」。"""
    report = Report(total=len(outcomes))
    if not outcomes:
        return report

    report.recall_at_k = _rate(
        [hits_gold(o, k) for o in outcomes if hits_gold(o, k) is not None]
    )
    if not retrieval_only:
        report.citation_rate = _rate(
            [bool(o.citations.cited) for o in outcomes if not o.expect_insufficient]
        )
        report.unknown_citations = sum(len(o.citations.unknown) for o in outcomes)
        report.ungrounded_rate = _rate([is_ungrounded(o) for o in outcomes])
        report.out_of_scope_accuracy = _rate(
            [o.insufficient for o in outcomes if o.expect_insufficient]
        )
        report.answered_rate = _rate(
            [not o.insufficient for o in outcomes if not o.expect_insufficient]
        )
        report.downgraded_count = sum(1 for o in outcomes if o.citations.downgraded)
        report.refusal_cited = sum(
            1 for o in outcomes if o.insufficient and o.citations.cited
        )
        report.reason_accuracy = _rate(
            [o.unknown_reason == o.expect_reason for o in outcomes if o.expect_reason]
        )

    for qtype in sorted({o.qtype for o in outcomes}):
        group = [o for o in outcomes if o.qtype == qtype]
        report.by_type[qtype] = {
            "count": float(len(group)),
            "recall": _rate([hits_gold(o, k) for o in group]),
            "citation": None if retrieval_only else _rate(
                [bool(o.citations.cited) for o in group if not o.expect_insufficient]
            ),
            "ungrounded": None if retrieval_only else _rate([is_ungrounded(o) for o in group]),
            "unknown": None if retrieval_only else _rate([o.insufficient for o in group]),
        }

    for outcome in outcomes:
        bad = hits_gold(outcome, k) is False
        if not retrieval_only:
            # 需要人看一眼的题：召回失败 / 编造引用 / 该拒答却答了 / 挂了 grounded 却没引用，
            # 外加"该答的题却拒答了"—— 降级也算：那说明模型吐了一段拿不出出处的正文，
            # 虽然处置正确（正文没给用户看），但值得有人知道它发生过。
            bad = (
                bad
                or is_ungrounded(outcome)
                or (outcome.mode == MODE_UNKNOWN and not outcome.expect_insufficient)
            )
        if bad:
            report.fails.append(outcome.question_id)
    return report


def _rate(values: list[Any]) -> float | None:
    """命中率；**样本为空返回 None** —— 空集算 0% 会把"不适用"误报成"全错"。"""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(1.0 for v in values if v) / len(values)
