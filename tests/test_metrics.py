"""Step 7 验收：三个指标算得对（纯函数，不碰库不联网）。"""

from __future__ import annotations

import pytest

from app.eval.metrics import (
    Outcome,
    evaluate,
    first_gold_rank,
    hits_gold,
    is_ungrounded,
    keypoint_score,
)
from app.generation.answerer import DOWNGRADE_ANSWER, CitationReport
from app.retrieval.retriever import RetrievedChunk


def _hit(paper_id: str, section: str, similarity: float = 0.6) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{paper_id}:c0001",
        similarity=similarity,
        rank=0,
        content="正文",
        raw_content="正文",
        paper={"paper_id": paper_id, "title": f"论文 {paper_id}"},
        chunk_order=1,
        page_start=1,
        page_end=1,
        chunk_type="text",
        token_count=10,
        section_id=f"{paper_id}:s1",
        section_title=section,
        section_order=1,
    )


def _outcome(
    qid: str = "q1",
    *,
    qtype: str = "detail",
    hits: list[RetrievedChunk] | None = None,
    gold: list[dict] | None = None,
    answer: str = "论文提出 X [E1]。",
    cited: list[str] | None = None,
    unknown: list[str] | None = None,
    mode: str = "grounded",
    reason: str | None = None,
    expect_insufficient: bool = False,
    expect_reason: str | None = None,
    require_all_papers: bool = False,
    keypoints: list | None = None,
    layer: str = "",
) -> Outcome:
    return Outcome(
        question_id=qid,
        qtype=qtype,
        question="问题",
        expect_insufficient=expect_insufficient,
        gold=gold if gold is not None else [{"paper_id": "p1", "section": "4.2 Implementation Setup"}],
        scope=["p1"],
        hits=hits if hits is not None else [_hit("p1", "4.2 Implementation Setup")],
        answer=answer,
        citations=CitationReport(
            cited=cited if cited is not None else ["E1"],
            unknown=unknown or [],
            mode=mode,
            unknown_reason=reason,
            downgraded=(reason == "downgraded"),
        ),
        require_all_papers=require_all_papers,
        expect_reason=expect_reason,
        keypoints=keypoints or [],
        layer=layer,
    )


# ----------------------------------------------------------------------
# MRR@k：第一个命中排名的倒数（2026-09-25 新增）
# ----------------------------------------------------------------------
def test_first_gold_rank_counts_position() -> None:
    hits = [_hit("p1", "别的章节"), _hit("p1", "别的章节"), _hit("p1", "4.2 Implementation Setup")]
    assert first_gold_rank(_outcome(hits=hits), 5) == 3
    assert first_gold_rank(_outcome(hits=hits), 2) is None


def test_first_gold_rank_cross_paper_waits_for_both() -> None:
    """跨论文题要**两篇都到齐**的那个位置：第 1 条只中一篇，不算。"""
    gold = [{"paper_id": "p1", "section": "A"}, {"paper_id": "p2", "section": "B"}]
    hits = [_hit("p1", "A"), _hit("p1", "A"), _hit("p2", "B")]
    outcome = _outcome(hits=hits, gold=gold, require_all_papers=True)
    assert first_gold_rank(outcome, 5) == 3
    assert first_gold_rank(outcome, 2) is None


def test_first_gold_rank_is_none_without_gold() -> None:
    assert first_gold_rank(_outcome(gold=[]), 5) is None


def test_evaluate_reports_mrr() -> None:
    first = _outcome("q1", hits=[_hit("p1", "4.2 Implementation Setup")])
    third = _outcome("q2", hits=[_hit("p1", "别的"), _hit("p1", "别的"), _hit("p1", "4.2 Implementation Setup")])
    report = evaluate([first, third], k=5)
    assert report.mrr_at_k == pytest.approx((1.0 + 1 / 3) / 2)


# ----------------------------------------------------------------------
# 要点命中率（2026-09-25 新增）
# ----------------------------------------------------------------------
def test_keypoint_score_accepts_alternative_spellings() -> None:
    """备选写法任一命中即算；全角数字经 NFKC 归一化后也算中。"""
    outcome = _outcome(
        answer="batch size 8，adapter 学习率 2e-4。",
        keypoints=[["２e-4", "2e-4"], ["batch size 4", "batch size 8"]],
    )
    assert keypoint_score(outcome) == (2, 2)


def test_keypoint_score_reports_misses() -> None:
    outcome = _outcome(answer="只说了 8。", keypoints=["8", ["0.85", "0.9"]])
    assert keypoint_score(outcome) == (1, 2)


def test_keypoint_score_is_none_without_keypoints() -> None:
    assert keypoint_score(_outcome(keypoints=[])) is None


def test_evaluate_reports_keypoints_and_layers() -> None:
    good = _outcome(
        "q1", answer="batch size 8", keypoints=["8"], layer="detail",
    )
    missed = _outcome(
        "q2", answer="没有提到那个数字", keypoints=["40"], layer="global",
    )
    report = evaluate([good, missed], k=5)
    assert report.keypoint_rate == 0.5  # (1/1 + 0/1) / 2
    assert report.keypoint_misses == ["q2"]
    assert report.by_layer["detail"]["keypoints"] == 1.0
    assert report.by_layer["global"]["keypoints"] == 0.0
    assert report.by_layer["detail"]["mrr"] == 1.0


def test_keypoint_rate_is_none_in_retrieval_only() -> None:
    """--retrieval-only 不调 LLM，答案为空 —— 要点命中率显示"不适用"而不是 0。"""
    report = evaluate([_outcome("q1", answer="", keypoints=["8"])], k=5, retrieval_only=True)
    assert report.keypoint_rate is None


# ----------------------------------------------------------------------
# 召回
# ----------------------------------------------------------------------
def test_hits_gold_matches_paper_and_section() -> None:
    outcome = _outcome()
    assert hits_gold(outcome, 5) is True


def test_hits_gold_requires_the_right_paper() -> None:
    """章节标题对但论文不对 —— 这正是"串论文"，不能算命中。"""
    outcome = _outcome(hits=[_hit("p2", "4.2 Implementation Setup")])
    assert hits_gold(outcome, 5) is False


def test_hits_gold_respects_k() -> None:
    hits = [_hit("p2", "别的章节")] * 4 + [_hit("p1", "4.2 Implementation Setup")]
    assert hits_gold(_outcome(hits=hits), 5) is True
    assert hits_gold(_outcome(hits=hits), 4) is False


def test_hits_gold_is_none_without_gold() -> None:
    """错库题没标 gold —— 不该拉低 Recall。"""
    assert hits_gold(_outcome(gold=[]), 5) is None


def test_require_all_papers_for_cross_paper_questions() -> None:
    """跨论文对比题：每篇都要有代表；只命中一篇不算过。"""
    both = [
        _hit("p1", "3.1 Overview"),
        _hit("p2", "3.1 Task Formulation"),
    ]
    only_one = [_hit("p1", "3.1 Overview")]
    gold = [
        # 同一篇列多个候选章节 —— 它们之间是"或"
        {"paper_id": "p1", "section": "3.1 Overview"},
        {"paper_id": "p1", "section": "3.2 Detail"},
        {"paper_id": "p2", "section": "3.1 Task Formulation"},
    ]

    strict = _outcome(hits=only_one, gold=gold, require_all_papers=True)
    assert hits_gold(strict, 5) is False

    lenient = _outcome(hits=only_one, gold=gold, require_all_papers=False)
    assert hits_gold(lenient, 5) is True  # 默认"命中任一即可"

    assert hits_gold(_outcome(hits=both, gold=gold, require_all_papers=True), 5) is True


def test_cross_paper_candidate_sections_are_or_ed() -> None:
    """同一篇列了 3 个候选章节，命中其中任意一个就算这篇过了。"""
    gold = [
        {"paper_id": "p1", "section": "3.1 Overview"},
        {"paper_id": "p1", "section": "3.2 Detail"},
        {"paper_id": "p2", "section": "3.1 Task Formulation"},
    ]
    hits = [_hit("p1", "3.2 Detail"), _hit("p2", "3.1 Task Formulation")]
    assert hits_gold(_outcome(hits=hits, gold=gold, require_all_papers=True), 5) is True


# ----------------------------------------------------------------------
# 无依据
# ----------------------------------------------------------------------
def test_unknown_citation_is_ungrounded() -> None:
    assert is_ungrounded(_outcome(unknown=["E9"])) is True


def test_out_of_scope_question_answered_anyway_is_ungrounded() -> None:
    """错库题却给了实质答案 —— 最常见的"硬编"。"""
    # 错库题却挂了 grounded（默认档位）—— 最常见的"硬编"
    assert is_ungrounded(_outcome(expect_insufficient=True)) is True


def test_out_of_scope_question_saying_insufficient_is_fine() -> None:
    outcome = _outcome(
        expect_insufficient=True, mode="unknown", reason="out_of_scope", cited=[], answer="资料不足。"
    )
    assert is_ungrounded(outcome) is False


def test_long_answer_without_citation_is_ungrounded() -> None:
    long_answer = "这篇论文做了很多事，包括 A、B、C、D。" * 6
    assert is_ungrounded(_outcome(answer=long_answer, cited=[])) is True


def test_short_insufficient_answer_is_not_ungrounded() -> None:
    """「资料不足」本来就短、也不该有引用。"""
    outcome = _outcome(answer="资料不足。", cited=[], mode="unknown", reason="not_in_corpus")
    assert is_ungrounded(outcome) is False


# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------
def test_evaluate_aggregates() -> None:
    outcomes = [
        _outcome("ok"),
        _outcome("miss", hits=[_hit("p1", "别的章节")]),
        # 这题召回是命中的（用了默认 hits），坏在编造引用上 —— 两个指标互不干扰
        _outcome("oops", unknown=["E9"]),
    ]
    report = evaluate(outcomes, k=5)
    assert report.total == 3
    assert report.recall_at_k == 2 / 3
    assert report.unknown_citations == 1
    assert report.ungrounded_rate == 1 / 3
    assert "miss" in report.fails and "oops" in report.fails


def test_evaluate_excludes_out_of_scope_from_recall_and_citation() -> None:
    """错库题没 gold、也不该有引用 —— 不能拉低 Recall 和引用率。"""
    outcomes = [
        _outcome("normal"),
        _outcome(
            "oos", gold=[], expect_insufficient=True, mode="unknown",
            reason="not_in_corpus", cited=[], answer="资料不足。",
        ),
    ]
    report = evaluate(outcomes, k=5)
    assert report.recall_at_k == 1.0
    assert report.citation_rate == 1.0
    assert report.out_of_scope_accuracy == 1.0
    assert report.ungrounded_rate == 0.0


def test_evaluate_answered_rate_tracks_non_insufficient_answers() -> None:
    outcomes = [
        _outcome("a"),
        _outcome("b", answer="资料不足。", cited=[], mode="unknown", reason="not_in_corpus"),
    ]
    report = evaluate(outcomes, k=5)
    assert report.answered_rate == 0.5


def test_evaluate_empty() -> None:
    report = evaluate([], k=5)
    assert report.total == 0
    assert report.recall_at_k is None  # 没样本 = 不适用，不是 0%


def test_out_of_scope_type_shows_not_applicable_recall() -> None:
    """错库题没有 gold —— 那一类的 Recall 要显示"不适用"，不能显示 0%（会被当成全错）。"""
    report = evaluate([
        _outcome(
            "oos", gold=[], expect_insufficient=True, mode="unknown",
            reason="not_in_corpus", cited=[], answer="资料不足。",
        )
    ])
    assert report.by_type["detail"]["recall"] is None or report.by_type["detail"]["recall"] == 1.0
    assert report.recall_at_k is None


def test_retrieval_only_skips_citation_metrics() -> None:
    """--retrieval-only 下不调 LLM，引用类指标必须留 None，而且不能把题误标成"需要看"。"""
    outcomes = [
        _outcome("ok"),
        # 没调 LLM 时档位是 unknown（run_eval 也是这么填的）——
        # 否则"挂了 grounded 却没引用"会把每道题都误报成无依据
        _outcome("no_llm", answer="", cited=[], mode="unknown"),
    ]
    report = evaluate(outcomes, k=5, retrieval_only=True)
    assert report.citation_rate is None
    assert report.ungrounded_rate is None
    assert report.unknown_citations == 0
    assert report.fails == []  # 没调 LLM，不该因为"没引用"把题标成失败


# ----------------------------------------------------------------------
# 档位 / 拒答（Step 9 加的）
# ----------------------------------------------------------------------
def test_downgraded_is_not_ungrounded_but_is_counted() -> None:
    """降级**不算**"无依据"：正文已经被换成固定拒答文案，那是正确处置。
    但它必须被单独计数 —— 验收标准里这个数应当恒为 0。"""
    outcome = _outcome(
        "dg", answer=DOWNGRADE_ANSWER, cited=[], mode="unknown", reason="downgraded"
    )
    assert is_ungrounded(outcome) is False
    report = evaluate([outcome], k=5)
    assert report.downgraded_count == 1
    assert report.ungrounded_rate == 0.0
    # 处置正确，但**要出现在"需要看的题"里**：它意味着模型吐了一段拿不出出处的正文
    assert report.fails == ["dg"]


def test_refusal_cited_counts_models_that_ignore_the_protocol() -> None:
    """拒答里还带着 `[En]` —— 量的是模型听不听话，目标恒为 0。

    注意量的是**模型原始输出**（`citations` 是在剥之前解析的），不是最终展示的正文：
    展示那一步会把编号剥掉，所以拿展示后的文本量永远得 0，等于没量。
    """
    outcome = _outcome(
        "oos", expect_insufficient=True, expect_reason="out_of_scope",
        mode="unknown", reason="out_of_scope", cited=["E1", "E5"],
    )
    report = evaluate([outcome], k=5)
    assert report.refusal_cited == 1


def test_reason_accuracy_only_counts_questions_with_expect_reason() -> None:
    """两道拒绝题各标了期望 reason；分对没分对才是"超纲 vs 库里没有"的度量。
    没标 expect_reason 的题不参与（否则会被当成 0 分拉低）。"""
    counted = [
        _outcome(
            "oos1", expect_insufficient=True, expect_reason="out_of_scope",
            mode="unknown", reason="out_of_scope", cited=[],
        ),
        _outcome(
            "oos2", expect_insufficient=True, expect_reason="not_in_corpus",
            mode="unknown", reason="not_in_corpus", cited=[],
        ),
        _outcome("normal"),  # 没有 expect_reason
    ]
    assert evaluate(counted, k=5).reason_accuracy == 1.0

    # 把"超纲"判成"库里没有"：两档被合并 → 扣分
    wrong = [
        _outcome(
            "oos1", expect_insufficient=True, expect_reason="out_of_scope",
            mode="unknown", reason="not_in_corpus", cited=[],
        ),
        counted[1],
    ]
    assert evaluate(wrong, k=5).reason_accuracy == 0.5
    assert evaluate([], k=5).reason_accuracy is None  # 没样本 = 不适用
