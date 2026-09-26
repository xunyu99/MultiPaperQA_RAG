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

import re
import unicodedata
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
    # 要点命中率用的必含事实点。每项是字符串（单一写法）或字符串列表（任一同义写法命中即算），
    # 见 EVAL_PLAN §2.1。拒答题为空。
    keypoints: list[Any] = field(default_factory=list)
    # 分层与难度：分层的指标单独出（全局题看 rerank，图表题看资产注入）
    layer: str = ""
    difficulty: str = ""
    # 参考答案：只喂给 correctness 裁判，不进机械指标（裁判在 M2 接）
    reference: str | None = None
    # LLM 裁判结果（M2）：1 = 通过，0 = 不通过，None = 这轮没判
    judge_grounded: int | None = None
    judge_correct: int | None = None
    judge_note: str = ""

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
    # 首个 gold 命中排名的倒数均值。Recall 只看"有没有进 top-k"，
    # MRR 才看得出"第一条是不是就对了" —— rerank 的价值主要在这里显形。
    mrr_at_k: float | None = None
    # 要点命中率：每题命中 keypoints 数 ÷ 该题 keypoints 总数，再按题平均。
    # 检索对了但答不出关键数字，就是生成层的问题 —— Recall 看不见这一层。
    keypoint_rate: float | None = None
    # 要点没答全的题（人工复核清单，不与 fails 混在一起）
    keypoint_misses: list[str] = field(default_factory=list)
    # 裁判层（M2）：有据率 / 正确率；缓存命中数单独记（算钱用）
    groundedness_rate: float | None = None
    correctness_rate: float | None = None
    judge_cached: int = 0
    by_type: dict[str, dict[str, float | None]] = field(default_factory=dict)
    by_layer: dict[str, dict[str, float | None]] = field(default_factory=dict)
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


def _matches(hit: Any, target: dict[str, str]) -> bool:
    return (
        hit.paper.get("paper_id") == target["paper_id"]
        and target["section"] in (hit.section_title or "")
    )


def first_gold_rank(outcome: Outcome, k: int) -> int | None:
    """第一个命中 gold 的排名（1 起）；没命中返回 None。没标 gold 的题也返回 None。

    跨论文题要求"每篇都有代表"，所以取**所有论文都到齐**的位置（各篇首个命中的最大排名）——
    只算第一篇中不算过，那正是 `require_all_papers` 要考的东西。
    """
    if not outcome.gold:
        return None
    top = outcome.hits[:k]

    if not outcome.require_all_papers:
        for rank, hit in enumerate(top, 1):
            if any(_matches(hit, target) for target in outcome.gold):
                return rank
        return None

    arrived: dict[str, int] = {}
    for rank, hit in enumerate(top, 1):
        for target in outcome.gold:
            if _matches(hit, target):
                arrived.setdefault(target["paper_id"], rank)
    needed = {target["paper_id"] for target in outcome.gold}
    if not needed <= set(arrived):
        return None
    return max(arrived[paper_id] for paper_id in needed)


def _normalize_for_match(text: str) -> str:
    """匹配前的归一化：NFKC（全角→半角）+ casefold + 空白折叠。

    故意**不做同义词替换** —— 数字的写法差异（`2e-4` / `2×10^-4`）交给 keypoints 的
    备选写法列表去覆盖，而不是写成一条越来越聪明的正则（那种规则最后没人敢改）。
    """
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def keypoint_score(outcome: Outcome) -> tuple[int, int] | None:
    """(命中条数, 总条数)；这道题没标 keypoints 返回 None。"""
    keypoints = outcome.keypoints or []
    if not keypoints:
        return None
    answer = _normalize_for_match(outcome.answer)
    hit = 0
    for item in keypoints:
        alternatives = item if isinstance(item, (list, tuple)) else [item]
        if any(
            _normalize_for_match(str(alt)) in answer
            for alt in alternatives
            if str(alt).strip()
        ):
            hit += 1
    return hit, len(keypoints)


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
    # MRR：**没命中的题按 0 计**（分母是"所有标了 gold 的题"），否则"少算几道"会把分数抬高
    golded = [o for o in outcomes if o.gold]
    if golded:
        reciprocal = [
            1.0 / rank if (rank := first_gold_rank(o, k)) else 0.0 for o in golded
        ]
        report.mrr_at_k = sum(reciprocal) / len(reciprocal)
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
        # 要点命中率：每题的**比例**再取平均 —— 这样 3 个要点全错和 3 个错 2 个不会同权
        scores = [score for o in outcomes if (score := keypoint_score(o)) is not None]
        if scores:
            report.keypoint_rate = sum(hit / total for hit, total in scores) / len(scores)
            report.keypoint_misses = [
                o.question_id
                for o in outcomes
                if (score := keypoint_score(o)) is not None and score[0] < score[1]
            ]
        report.groundedness_rate = _rate(
            [o.judge_grounded == 1 for o in outcomes if o.judge_grounded is not None]
        )
        report.correctness_rate = _rate(
            [o.judge_correct == 1 for o in outcomes if o.judge_correct is not None]
        )
        report.judge_cached = sum(1 for o in outcomes if o.judge_note == "cached")

    for qtype in sorted({o.qtype for o in outcomes}):
        group = [o for o in outcomes if o.qtype == qtype]
        report.by_type[qtype] = {
            "count": float(len(group)),
            "recall": _rate([hits_gold(o, k) for o in group]),
            "mrr": _mrr(group, k),
            "citation": None if retrieval_only else _rate(
                [bool(o.citations.cited) for o in group if not o.expect_insufficient]
            ),
            "ungrounded": None if retrieval_only else _rate([is_ungrounded(o) for o in group]),
            "unknown": None if retrieval_only else _rate([o.insufficient for o in group]),
            "keypoints": None if retrieval_only else _keypoint_rate(group),
        }

    # 分层汇总：同一个组件在不同题型上的作用完全不同（全局题看 rerank、图表题看资产注入）
    for layer in sorted({o.layer for o in outcomes if o.layer}):
        group = [o for o in outcomes if o.layer == layer]
        report.by_layer[layer] = {
            "count": float(len(group)),
            "recall": _rate([hits_gold(o, k) for o in group]),
            "mrr": _mrr(group, k),
            "citation": None if retrieval_only else _rate(
                [bool(o.citations.cited) for o in group if not o.expect_insufficient]
            ),
            "keypoints": None if retrieval_only else _keypoint_rate(group),
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


def _mrr(outcomes: list[Outcome], k: int) -> float | None:
    golded = [o for o in outcomes if o.gold]
    if not golded:
        return None
    return sum(
        1.0 / rank if (rank := first_gold_rank(o, k)) else 0.0 for o in golded
    ) / len(golded)


def _keypoint_rate(outcomes: list[Outcome]) -> float | None:
    scores = [score for o in outcomes if (score := keypoint_score(o)) is not None]
    if not scores:
        return None
    return sum(hit / total for hit, total in scores) / len(scores)
