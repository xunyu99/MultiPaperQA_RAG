"""Step 7：跑评测集，一条命令出指标表（这份数字就是后面所有优化的对比基线）。

用法：
    # 先看会跑哪些题、会花多少钱，不执行
    .venv\\Scripts\\python.exe -m scripts.run_eval --list

    # 完整跑（每道题 1 次 embedding + 1 次 rerank + 1 次 LLM）
    .venv\\Scripts\\python.exe -m scripts.run_eval

    # 只看检索层指标（不调 LLM，不花钱）
    .venv\\Scripts\\python.exe -m scripts.run_eval --retrieval-only

    # 详细模式：打印每题的检索结果和回答
    .venv\\Scripts\\python.exe -m scripts.run_eval --verbose

**这组数字是基线，不是成绩**。后面每加一个东西（BM25 / rerank / 资产解析器 / sticky scope）
都重跑这条命令，数字没动就回滚（PLAN §0.2）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db.connection import connect, init_db
from app.eval.metrics import (
    Outcome,
    evaluate,
    first_gold_rank,
    hits_gold,
    is_ungrounded,
    keypoint_score,
)
from app.generation.answerer import (
    MODE_UNKNOWN,
    REASON_DOWNGRADED,
    AnswerResult,
    CitationReport,
    answer_from_cards,
)
from app.generation.evidence import build_cards
from app.generation.evidence import render as render_cards
from app.eval.judge import Judge
from app.providers.llm import LLMClient
from app.retrieval.retriever import retrieve
from app.retrieval.vector_index import ChunkVectorIndex

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "app" / "eval" / "questions.jsonl"
DEFAULT_DUMP_PATH = Path(__file__).resolve().parent.parent / ".eval_out" / "last.json"

# 中译英（E9）：只用于**检索**，生成仍用原问句 —— 否则"检索语言"和"生成语言"
# 两个变量会混在一起。专有名词和编号写法必须原样保留，否则 scope 解析和
# 资产定点注入（"表 1"）都会崩。
_TRANSLATE_PROMPT = (
    "把下面这个问题翻译成英文，用于检索英文论文。\n\n"
    "硬性要求：\n"
    "1. 专有名词原样保留：论文名（PE-CLIP / ORSANet / FDSRM / FaceCaption-15M / EmotioNet）、"
    "模型名、数据集名、缩写，以及「表 N」「图 N」这类编号写法，一律不要改写或翻译。\n"
    "2. 只输出译文本身，一行，不要解释、不要引号、不要前后缀。\n\n"
    "问题：{question}"
)


def _translator(settings):
    """返回一个 问题 -> 英文译文 的函数（LLM 调用，结果在内存里缓存）。"""
    planner = LLMClient(settings).build_planner()
    cache: dict[str, str] = {}

    def translate(question: str) -> str:
        if question not in cache:
            reply = planner.invoke(_TRANSLATE_PROMPT.format(question=question))
            text = getattr(reply, "content", reply)
            cache[question] = str(text).strip().strip('"').splitlines()[0] if str(text).strip() else question
        return cache[question]

    return translate


def main() -> int:
    parser = argparse.ArgumentParser(description="评测基线（Step 7）")
    parser.add_argument("-k", type=int, default=None, help="Recall@k，默认取配置的 TOP_K")
    parser.add_argument("--list", action="store_true", help="只列出题目，不执行")
    parser.add_argument("--retrieval-only", action="store_true", help="只跑检索，不调 LLM（不花钱）")
    parser.add_argument("--verbose", action="store_true", help="打印每题的检索结果与回答")
    parser.add_argument("--only", default=None, help="只跑这些题，逗号分隔的 id（调试单题用）")
    parser.add_argument(
        "--dump", nargs="?", const=str(DEFAULT_DUMP_PATH), default=str(DEFAULT_DUMP_PATH),
        help="每题的检索明细写成 JSON（默认 .eval_out/last.json），传 0 表示不写",
    )
    parser.add_argument("--section-cap", type=int, default=None, help="同章节去重上限（0=关，默认取配置）")
    # ---- 消融开关（EVAL_PLAN §3.1）：一次运行只改一个变量，跑完对着数字看有没有用 ----
    parser.add_argument("--no-keyword", action="store_true", help="关掉关键词通道（纯向量 vs 双路 A/B）")
    parser.add_argument("--no-rerank", action="store_true", help="关掉重排（候选 50 -> 直接取前 k）")
    parser.add_argument("--no-asset", action="store_true", help="关掉资产定点注入（问「表 1」不再钉住本体）")
    parser.add_argument(
        "--layer", default=None, choices=["global", "detail", "asset", "refusal"],
        help="只跑某一层（分层是本次评测的核心，见 EVAL_PLAN §1.1）",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, metavar="N",
        help="整集重复跑 N 次，只打每轮汇总（看 LLM 的波动；成本 ×N）",
    )
    parser.add_argument(
        "--translate-query", action="store_true",
        help="中译英对照（E9）：检索用英文译文，生成仍用原问句；每题多 1 次 LLM 调用",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="接 LLM 裁判（M2）：groundedness + correctness，每题各 1 次调用（有缓存）",
    )
    parser.add_argument(
        "--ab-section-cap", type=int, default=None, metavar="N",
        help="单次跑里对比 cap=0（关）与 cap=N：每题只算 1 次 embedding，用来判断这个改动到底有没有用",
    )
    args = parser.parse_args()

    settings = get_settings()
    # 消融开关直接改 settings 副本：检索链路不认命令行，只认 settings（这也保证
    # "评测里关掉的"和"线上配置里的"是同一个开关，不会两套逻辑）
    overrides: dict[str, Any] = {}
    if args.no_keyword:
        overrides["keyword_enabled"] = False
    if args.no_rerank:
        overrides["rerank_enabled"] = False
    if overrides:
        settings = settings.model_copy(update=overrides)
        print(f"[消融] " + "、".join(f"{key}=False" for key in overrides))
    auto_asset = not args.no_asset
    translate = _translator(settings) if args.translate_query else None
    if translate is not None:
        print("[E9] 检索前把问题译成英文（专有名词/编号保持原样），生成仍用原问句")
    if args.judge and args.retrieval_only:
        print("--judge 需要真实回答，不能和 --retrieval-only 一起用")
        return 1
    judge = Judge(settings) if args.judge else None
    if judge is not None:
        print(f"[M2] 裁判：{judge.model}（groundedness + correctness，命中缓存不重复调用）")
    k = args.k or settings.top_k
    questions = _load_questions()
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        questions = [item for item in questions if item["id"] in wanted]
        if not questions:
            print(f"没有匹配的题：{args.only}")
            return 1
    if args.layer:
        questions = [item for item in questions if item.get("layer") == args.layer]
        if not questions:
            print(f"没有这一层的题：{args.layer}")
            return 1

    if args.list:
        return _list(questions)

    index = ChunkVectorIndex(settings)
    conn = connect()
    init_db(conn)
    try:
        if args.ab_section_cap is not None:
            return _run_ab(conn, index, questions, k, settings, args)

        print(f"=== 评测集：{len(questions)} 题，top-{k} ===")
        if not args.retrieval_only:
            print(f"预计调用：{len(questions)} 次 embedding + {len(questions)} 次 LLM")
        print()

        reports: list[Any] = []
        for run in range(1, max(1, args.repeat) + 1):
            if args.repeat > 1:
                print(f"--- 第 {run} / {args.repeat} 轮 ---")

            outcomes: list[Outcome] = []
            for item in questions:
                outcome = _run_one(
                    conn, index, item, k, settings,
                    retrieval_only=args.retrieval_only, section_cap=args.section_cap,
                    auto_asset=auto_asset, translate=translate, judge=judge,
                )
                outcomes.append(outcome)
                # 重复跑时逐题打印会刷屏，只留最终汇总
                if args.repeat == 1:
                    _print_one(outcome, k, args.verbose)
                # **增量落盘**：2026-09-26 两次长任务都因为"跑到最后才写 dump"而白跑
                # （进程非正常退出 → 200KB 结果全丢）。现在每判完一题就覆写一次，
                # 中断也能拿到已完成的部分。文件名逻辑与收尾那次保持一致。
                if args.dump and args.dump != "0":
                    partial = Path(args.dump)
                    if args.repeat > 1:
                        partial = partial.with_name(f"{partial.stem}-run{run}{partial.suffix}")
                    _dump_json(partial, outcomes, k, args.retrieval_only)

            report = evaluate(outcomes, k=k, retrieval_only=args.retrieval_only)
            reports.append(report)
            _print_report(report, k, args.retrieval_only)

            if args.dump and args.dump != "0":
                path = Path(args.dump)
                if args.repeat > 1:
                    path = path.with_name(f"{path.stem}-run{run}{path.suffix}")
                _dump_json(path, outcomes, k, args.retrieval_only)
                print(f"明细已写入：{path}")

        if len(reports) > 1:
            _print_spread(reports, k)
        return 0
    finally:
        conn.close()


def _load_questions() -> list[dict]:
    lines = [line for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def _diversity(outcome: Outcome, k: int) -> int:
    """top-k 覆盖了多少个**不同的章节**（论文+章节）。去重想改善的就是这个数。"""
    return len(
        {
            (hit.paper.get("paper_id"), hit.section_id or hit.section_title or "?")
            for hit in outcome.hits[:k]
        }
    )


def _evidence_tokens(outcome: Outcome, k: int) -> int:
    """top-k 的证据 token 合计 —— 去重的**代价**就体现在这里。

    拿"同章节的第二块"换"另一个章节的第一块"时，换进来的不一定相关，
    却一样占 token 预算。只涨多样性不看这个数，会把噪声当收益。
    """
    return sum(hit.token_count for hit in outcome.hits[:k])


def _run_ab(conn, index, questions: list[dict], k: int, settings, args) -> int:
    """同一批问题，用**同一个 query 向量**跑 cap=0 和 cap=N，直接比出改动有没有用。

    一次 embedding 喂两套参数（`retrieve(vector=...)` 跳过重复 embedding），
    所以整个 A/B 只花 N 次 embedding、0 次 LLM。
    """
    cap = args.ab_section_cap
    print(f"=== A/B：同章节去重 cap=0（关） vs cap={cap}，{len(questions)} 题 top-{k} ===")
    print(f"每题只算 1 次 embedding：{len(questions)} 次 embedding，0 次 LLM")
    print()

    pairs: list[tuple[Outcome, Outcome, float, float, int, int]] = []
    for item in questions:
        vector = index.embedder.embed_query(item["question"])
        base = _run_one(conn, index, item, k, settings, retrieval_only=True,
                        vector=vector, section_cap=0)
        cand = _run_one(conn, index, item, k, settings, retrieval_only=True,
                        vector=vector, section_cap=cap)
        pairs.append(
            (
                base, cand,
                _diversity(base, k), _diversity(cand, k),
                _evidence_tokens(base, k), _evidence_tokens(cand, k),
            )
        )

    print(f"{'题目':<18}{'章节数 关->开':>16}{'gold 关->开':>16}{'证据token 关->开':>18}   变化")
    changed = []
    for base, cand, div0, div1, tok0, tok1 in pairs:
        hit0, hit1 = _hit_of(base, k), _hit_of(cand, k)
        if hit0 is None:
            hit_text = "不适用"
        else:
            hit_text = f"{'OK' if hit0 else 'X'}->{'OK' if hit1 else 'X'}"
        div_text = f"{div0}->{div1}"
        tok_text = f"{tok0}->{tok1}"
        delta = div1 - div0
        flag = "章节更全" if delta > 0 else ("章节变少" if delta < 0 else "")
        if delta > 0 and tok1 > tok0:
            flag += "、token涨"
        if hit0 is not None and hit1 and not hit0:
            flag = "修好了" if not flag else flag + " + 修好了"
        elif hit0 is not None and hit0 and not hit1:
            flag = "搞坏了"
        if flag:
            changed.append(f"{base.question_id}({flag})")
        print(f"{base.question_id:<18}{div_text:>16}{hit_text:>16}{tok_text:>18}   {flag}")
    print()

    report0 = evaluate([pair[0] for pair in pairs], k=k, retrieval_only=True)
    report1 = evaluate([pair[1] for pair in pairs], k=k, retrieval_only=True)
    div0 = sum(pair[2] for pair in pairs) / len(pairs)
    div1 = sum(pair[3] for pair in pairs) / len(pairs)
    tok0 = sum(pair[4] for pair in pairs) / len(pairs)
    tok1 = sum(pair[5] for pair in pairs) / len(pairs)
    print("=== 汇总 ===")
    print(f"  Recall@{k}          关 = {report0.recall_at_k:.0%}   开 = {report1.recall_at_k:.0%}")
    print(f"  平均不同章节数      关 = {div0:.2f}     开 = {div1:.2f}")
    print(f"  平均证据 token      关 = {tok0:.0f}     开 = {tok1:.0f}")
    print(f"  有变化的题          {'、'.join(changed) if changed else '无'}")
    print()
    print("判据：Recall 涨了才留；只涨章节数、Recall 不动，说明去重是'卫生习惯'不是'药'；")
    print("      Recall 掉了就回滚（section_cap 改回 0，代码留着不影响）；")
    print("      章节数涨了但 token 也涨，要挨个看换进来的是不是噪声 —— 多样性不等于相关性。")

    if args.dump and args.dump != "0":
        path = Path(args.dump).with_name("ab_section_cap.json")
        _dump_ab(path, pairs, k, cap)
        print(f"明细已写入：{path}")
    return 0


def _hit_of(outcome: Outcome, k: int) -> bool | None:
    """没有 gold 的题（错库题）返回 None：Recall 对它们不适用。"""
    if not outcome.gold:
        return None
    return hits_gold(outcome, k)


def _dump_ab(path: Path, pairs, k: int, cap: int) -> None:
    payload = {
        "k": k,
        "ab_section_cap": cap,
        "questions": [
            {
                "id": base.question_id,
                "question": base.question,
                "scope": base.scope,
                "cap_0": [_brief(hit) for hit in base.hits],
                "cap_n": [_brief(hit) for hit in cand.hits],
            }
            for base, cand, *_ in pairs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _brief(hit) -> dict:
    return {
        "chunk_id": hit.chunk_id,
        "similarity": round(hit.similarity, 4),
        "paper_id": hit.paper.get("paper_id"),
        "section_title": hit.section_title,
    }


def _dump_json(path: Path, outcomes: list[Outcome], k: int, retrieval_only: bool) -> None:
    """每题的检索明细落盘（含每篇论文的 top-k 和命中了哪些 gold）。

    控制台里中文/长行容易被截断，落成 JSON 才能完整回看，也方便前后两次跑 diff。
    """
    payload = {
        "k": k,
        "retrieval_only": retrieval_only,
        "questions": [
            {
                "id": outcome.question_id,
                "type": outcome.qtype,
                "layer": outcome.layer,
                "difficulty": outcome.difficulty,
                "question": outcome.question,
                "scope": outcome.scope,
                "gold": outcome.gold,
                "require_all_papers": outcome.require_all_papers,
                "hit": hits_gold(outcome, k),
                "gold_rank": first_gold_rank(outcome, k),
                "keypoints": outcome.keypoints,
                "keypoint_score": keypoint_score(outcome),
                "mode": outcome.mode,
                "unknown_reason": outcome.unknown_reason,
                "expect_reason": outcome.expect_reason,
                # 裁判结果（M2）：写进 dump 才能在 dump 上手算/复判，不必再依赖缓存反推
                "judge_grounded": outcome.judge_grounded,
                "judge_correct": outcome.judge_correct,
                "judge_note": outcome.judge_note,
                "top1_rank_score": round(outcome.hits[0].rank_score, 4) if outcome.hits else None,
                "answer": outcome.answer,
                "raw_answer": outcome.raw_answer,
                "hits": [
                    {
                        "rank": rank,
                        "similarity": round(item.similarity, 4),
                        "rank_score": round(item.rank_score, 4),
                        "chunk_id": item.chunk_id,
                        "paper_id": item.paper.get("paper_id"),
                        "paper_title": item.paper.get("title"),
                        "section_title": item.section_title,
                        "section_type": item.section_type,
                        "page_start": item.page_start,
                        "token_count": item.token_count,
                        "assets": [asset.get("asset_id") for asset in item.assets],
                        "preview": " ".join(item.content.split())[:200],
                    }
                    for rank, item in enumerate(outcome.hits, 1)
                ],
            }
            for outcome in outcomes
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_one(
    conn, index, item: dict, k: int, settings, retrieval_only: bool,
    vector=None, section_cap: int | None = None, auto_asset: bool = True,
    translate=None, judge=None,
) -> Outcome:
    stats: dict = {}
    # E9：检索用译文，**scope 与生成仍按原问句**（题面显式给的 scope 不受影响；
    # 没给 scope 的题由 retrieve 从 query 里解析，所以要求译文保留论文名）
    query = translate(item["question"]) if translate is not None else item["question"]
    hits = retrieve(
        conn, query, k=k, index=index, settings=settings,
        paper_id=item.get("paper_id"),
        # 题面显式给的范围（跨论文题必须显式给：字面解析只会认出第一篇，
        # 另一篇被过滤掉，题目永远过不了 —— 2026-09-25 实测踩到）
        paper_ids=item.get("scope") or None,
        stats=stats,
        vector=vector, section_cap=section_cap, auto_asset=auto_asset,
    )

    common = {
        "question_id": item["id"],
        "qtype": item["type"],
        "question": item["question"],
        "expect_insufficient": item["expect_insufficient"],
        "gold": item.get("gold") or [],
        "scope": stats.get("scope") or [],
        "keypoints": item.get("keypoints") or [],
        "layer": item.get("layer") or "",
        "difficulty": item.get("difficulty") or "",
        "reference": item.get("reference"),
        "note": item.get("note", ""),
        "require_all_papers": bool(item.get("require_all_papers")),
        "expect_reason": item.get("expect_reason"),
    }

    if retrieval_only:
        # 不调 LLM：用占位结果，只让检索层指标有效。
        # 档位标成 unknown —— "这轮没生成"当然不是 grounded，
        # 否则"挂了 grounded 却没引用"那条判据会把每道题都报成无依据。
        return Outcome(
            **common,
            hits=hits, answer="", citations=CitationReport(mode=MODE_UNKNOWN),
        )

    cards = build_cards(hits)
    result: AnswerResult = answer_from_cards(item["question"], cards, settings=settings)
    grounded = correct = None
    judge_note = ""
    if judge is not None:
        # 证据用**生成时看到的那份文本**（默认不渲染 citation），保证"判的是同一件事"
        evidence = render_cards(cards)
        grounded_result = judge.groundedness(
            item["question"], evidence, result.answer, question_id=item["id"]
        )
        grounded = grounded_result.label
        reference = item.get("reference")
        if reference:
            correct_result = judge.correctness(
                item["question"], reference, result.answer, question_id=item["id"]
            )
            correct = correct_result.label
            judge_note = "cached" if (grounded_result.cached and correct_result.cached) else ""
        else:
            judge_note = "cached" if grounded_result.cached else ""
    return Outcome(
        **common,
        # 生成用的证据卡按文档顺序重排了，这里回传到检索顺序会误导人，
        # 所以 hits 保留检索顺序（Recall 只看"有没有命中"，与顺序无关）
        hits=hits,
        answer=result.answer, citations=result.citations,
        latency_ms=result.latency_ms,
        raw_answer=result.raw_answer,
        judge_grounded=grounded,
        judge_correct=correct,
        judge_note=judge_note,
    )


def _print_one(outcome: Outcome, k: int, verbose: bool) -> None:
    hit = hits_gold(outcome, k)
    marks = []
    if hit is not None:
        marks.append(("召回两篇都中OK" if outcome.require_all_papers else "召回OK") if hit else "召回X")
    if outcome.citations.unknown:
        marks.append(f"编引用X({','.join(outcome.citations.unknown)})")
    # 档位：两道拒答题看 reason 分对没有；正常题出了 unknown 就是降级或误拒
    if outcome.mode == MODE_UNKNOWN:
        reason = outcome.unknown_reason or "?"
        if outcome.expect_insufficient:
            if reason == "?":
                # --retrieval-only 没调 LLM，"拒答"这件事无从判断，别报成拒答失败
                marks.append("拒答?(未生成)")
            else:
                ok = outcome.expect_reason in (None, reason)
                marks.append(f"{'拒答OK' if ok else '拒答X'}({reason})")
        elif reason == REASON_DOWNGRADED:
            marks.append("降级X")
        else:
            marks.append(f"误拒X({reason})")
    if outcome.insufficient and outcome.citations.cited:
        marks.append(f"拒答带引用X({'、'.join(outcome.citations.cited)})")
    if is_ungrounded(outcome):
        marks.append("无依据X")

    scope = f"{len(outcome.scope)}篇" if outcome.scope else "全库"
    print(f"[{outcome.question_id:<16}] {scope:<5} {' '.join(marks) or '—'}")
    print(f"    Q: {outcome.question}")
    if verbose:
        for rank, item in enumerate(outcome.hits[:k], 1):
            top_section = item.section_title or "?"
            paper_id = item.paper.get("paper_id") or "?"
            print(f"      {rank}. {item.similarity:+.3f}  {paper_id:<42} {top_section[:40]}")
        if outcome.answer:
            print(f"    A: {' '.join(outcome.answer.split())[:300]}")
    elif outcome.answer:
        print(f"    A: {' '.join(outcome.answer.split())[:120]}…")
    print()


def _print_report(report, k: int, retrieval_only: bool) -> None:
    print("=" * 66)
    print(f"=== 基线指标（{report.total} 题，top-{k}）===")
    print(f"  Recall@{k}          : {_pct(report.recall_at_k)}     （top-{k} 命中预期章节的比例）")
    mrr_text = "—" if report.mrr_at_k is None else f"{report.mrr_at_k:.3f}"
    print(f"  MRR@{k}             : {mrr_text}     （首个 gold 命中排名的倒数均值：第一条就中 = 1.000）")
    if not retrieval_only:
        print(f"  要点命中率         : {_pct(report.keypoint_rate)}     （keypoints 命中条数 ÷ 总数，按题平均）")
        if report.groundedness_rate is not None or report.correctness_rate is not None:
            print(f"  groundedness      : {_pct(report.groundedness_rate)}     （裁判：答案是否只依据证据）")
            print(f"  correctness       : {_pct(report.correctness_rate)}     （裁判：与参考答案事实是否一致）")
            print(f"  裁判缓存命中       : {report.judge_cached} 题")
        print(f"  引用率             : {_pct(report.citation_rate)}     （非错库题里给出引用的比例）")
        print(f"  编造引用           : {report.unknown_citations} 处   （引用了不存在的 [En]，必须为 0）")
        print(f"  无依据率           : {_pct(report.ungrounded_rate)}     （编引用 / 该拒答却答了 / 挂了 grounded 却没引用）")
        print(f"  降级               : {report.downgraded_count} 次   （声明有据却拿不出引用，必须为 0）")
        print(f"  拒答带引用         : {report.refusal_cited} 条   （模型没守协议，必须为 0）")
        print(f"  拒答 reason 命中率 : {_pct(report.reason_accuracy)}     （两道拒绝题：超纲 / 库里没有 分对没有）")
        print(f"  错库题答对率       : {_pct(report.out_of_scope_accuracy)}     （该说「资料不足」的题里真的说了）")
        print(f"  回答率             : {_pct(report.answered_rate)}     （非错库题里给出实质答案的比例）")
    else:
        print("  （--retrieval-only：引用 / 无依据类指标没算，它们需要调 LLM）")
    print()
    print("=== 分类别 ===")
    print(f"  {'类型':<16}{'题数':>4}{'Recall':>9}{'引用率':>9}{'无依据':>9}{'拒答':>9}")
    for qtype, row in report.by_type.items():
        print(
            f"  {qtype:<16}{int(row['count']):>4}{_pct(row['recall'], 9)}"
            f"{_pct(row['citation'], 9)}{_pct(row['ungrounded'], 9)}"
            f"{_pct(row.get('unknown'), 9)}"
        )
    if report.by_layer:
        print()
        print("=== 分层（全局题看 rerank、图表题看资产注入、拒答单独看）===")
        print(f"  {'层':<10}{'题数':>4}{'Recall':>9}{'MRR':>8}{'引用率':>9}{'要点':>8}")
        for layer, row in report.by_layer.items():
            mrr = row.get("mrr")
            print(
                f"  {layer:<10}{int(row['count']):>4}{_pct(row['recall'], 9)}"
                f"{(f'{mrr:.3f}' if mrr is not None else '—'):>8}"
                f"{_pct(row['citation'], 9)}{_pct(row.get('keypoints'), 8)}"
            )
    if report.keypoint_misses:
        print()
        print(f"=== 要点没答全的题：{', '.join(report.keypoint_misses)} ===")
    if report.fails:
        print()
        print(f"=== 需要看的题：{', '.join(report.fails)} ===")
        print("（用 --verbose 重跑看检索结果和回答）")
    print("=" * 66)
    print()
    print("这组数字是**基线**。以后每加一个优化（BM25 / rerank / 资产解析器 / sticky scope）")
    print("都重跑本命令，数字没动就回滚（PLAN §0.2）。")


def _pct(value: float | None, width: int = 0) -> str:
    """None = 不适用（没算），不能显示成 0% —— 那会被当成"全错"。"""
    text = "—" if value is None else f"{value:.0%}"
    return text.rjust(width) if width else text


def _print_spread(reports: list[Any], k: int) -> None:
    """`--repeat N` 的收尾：把每轮的核心数字并排打出来，看波动有多大。

    LLM 的输出有随机性，单次 100% 很脆 —— 重复跑才能区分"稳定达标"和"这次运气好"。
    不合并成"多数票"：那是另一种口径（会掩盖单轮的坏结果），要看分布就给分布。
    """
    print()
    print("=" * 66)
    print(f"=== {len(reports)} 轮波动（同题集同参数）===")
    print(f"  {'轮次':<6}{'Recall':>8}{'MRR':>8}{'要点':>8}{'引用率':>9}{'无依据':>9}{'拒答带引用':>12}{'降级':>6}")
    for index, report in enumerate(reports, 1):
        mrr = "—" if report.mrr_at_k is None else f"{report.mrr_at_k:.3f}"
        print(
            f"  {index:<6}{_pct(report.recall_at_k, 8)}{mrr:>8}"
            f"{_pct(report.keypoint_rate, 8)}{_pct(report.citation_rate, 9)}"
            f"{_pct(report.ungrounded_rate, 9)}{report.refusal_cited:>12}{report.downgraded_count:>6}"
        )
    print("=" * 66)
    print()


def _list(questions: list[dict]) -> int:
    print(f"=== 评测集：{len(questions)} 题 ===")
    for item in questions:
        gold = item.get("gold") or []
        gold_label = "、".join(f"{g['paper_id']}：{g['section']}" for g in gold) or "（预期答资料不足）"
        if item.get("require_all_papers"):
            gold_label += "   [两篇都要有代表]"
        print(f"  [{item['id']:<16}] {item['type']:<14} {item['question']}")
        print(f"      预期命中：{gold_label}")
        if item.get("note"):
            print(f"      备注：{item['note']}")
        print()
    print(f"完整跑：python -m scripts.run_eval      （{len(questions)} 次 embedding + {len(questions)} 次 LLM）")
    print("只测检索：python -m scripts.run_eval --retrieval-only   （不花钱）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
