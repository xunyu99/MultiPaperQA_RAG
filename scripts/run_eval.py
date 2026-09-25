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

from app.config import get_settings
from app.db.connection import connect, init_db
from app.eval.metrics import Outcome, evaluate, hits_gold, is_ungrounded
from app.generation.answerer import (
    MODE_UNKNOWN,
    REASON_DOWNGRADED,
    AnswerResult,
    CitationReport,
    answer_from_cards,
)
from app.generation.evidence import build_cards
from app.retrieval.retriever import retrieve
from app.retrieval.vector_index import ChunkVectorIndex

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "app" / "eval" / "questions.jsonl"
DEFAULT_DUMP_PATH = Path(__file__).resolve().parent.parent / ".eval_out" / "last.json"


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
    parser.add_argument(
        "--ab-section-cap", type=int, default=None, metavar="N",
        help="单次跑里对比 cap=0（关）与 cap=N：每题只算 1 次 embedding，用来判断这个改动到底有没有用",
    )
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.top_k
    questions = _load_questions()
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        questions = [item for item in questions if item["id"] in wanted]
        if not questions:
            print(f"没有匹配的题：{args.only}")
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

        outcomes: list[Outcome] = []
        for item in questions:
            outcome = _run_one(
                conn, index, item, k, settings,
                retrieval_only=args.retrieval_only, section_cap=args.section_cap,
            )
            outcomes.append(outcome)
            _print_one(outcome, k, args.verbose)

        report = evaluate(outcomes, k=k, retrieval_only=args.retrieval_only)
        _print_report(report, k, args.retrieval_only)
        if args.dump and args.dump != "0":
            path = Path(args.dump)
            _dump_json(path, outcomes, k, args.retrieval_only)
            print(f"明细已写入：{path}")
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
                "question": outcome.question,
                "scope": outcome.scope,
                "gold": outcome.gold,
                "require_all_papers": outcome.require_all_papers,
                "hit": hits_gold(outcome, k),
                "mode": outcome.mode,
                "unknown_reason": outcome.unknown_reason,
                "expect_reason": outcome.expect_reason,
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
    vector=None, section_cap: int | None = None,
) -> Outcome:
    stats: dict = {}
    hits = retrieve(
        conn, item["question"], k=k, index=index, settings=settings,
        paper_id=item.get("paper_id"), stats=stats,
        vector=vector, section_cap=section_cap,
    )

    if retrieval_only:
        # 不调 LLM：用占位结果，只让检索层指标有效。
        # 档位标成 unknown —— "这轮没生成"当然不是 grounded，
        # 否则"挂了 grounded 却没引用"那条判据会把每道题都报成无依据。
        return Outcome(
            question_id=item["id"], qtype=item["type"], question=item["question"],
            expect_insufficient=item["expect_insufficient"], gold=item.get("gold") or [],
            scope=stats.get("scope") or [], hits=hits, answer="",
            citations=CitationReport(mode=MODE_UNKNOWN), note=item.get("note", ""),
            require_all_papers=bool(item.get("require_all_papers")),
            expect_reason=item.get("expect_reason"),
        )

    result: AnswerResult = answer_from_cards(item["question"], build_cards(hits), settings=settings)
    return Outcome(
        question_id=item["id"], qtype=item["type"], question=item["question"],
        expect_insufficient=item["expect_insufficient"], gold=item.get("gold") or [],
        scope=stats.get("scope") or [],
        # 生成用的证据卡按文档顺序重排了，这里回传到检索顺序会误导人，
        # 所以 hits 保留检索顺序（Recall 只看"有没有命中"，与顺序无关）
        hits=hits,
        answer=result.answer, citations=result.citations,
        latency_ms=result.latency_ms, note=item.get("note", ""),
        require_all_papers=bool(item.get("require_all_papers")),
        expect_reason=item.get("expect_reason"),
        raw_answer=result.raw_answer,
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
    if not retrieval_only:
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
