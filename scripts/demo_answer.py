"""Step 6 验收：带引用的回答（这一步会调 LLM，花钱）。

用法：
    # 先看 prompt 长什么样，不花钱
    .venv\\Scripts\\python.exe -m scripts.demo_answer "这篇论文的贡献是什么" --dry-run

    # 真问（调一次 LLM）
    .venv\\Scripts\\python.exe -m scripts.demo_answer "这篇论文的贡献是什么"
    .venv\\Scripts\\python.exe -m scripts.demo_answer "消融实验说明了什么" --paper-id pe-clip

    # 同时打印证据卡，核对引用对不对
    .venv\\Scripts\\python.exe -m scripts.demo_answer "表 1 的结果" --paper-id pe-clip --show-evidence

每次运行打印四段：证据卡概要 → 回答 → 引用校验 → 溯源坐标。
**引用校验**是重点：`unknown` 非空说明模型编了不存在的编号；回答很长却一条引用都没有，
多半是拿自己的知识在答。
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.db.connection import connect, init_db
from app.generation.answerer import (
    SYSTEM_PROMPT_PATH,
    answer_from_cards,
    answer_question,
    build_user_message,
)
from app.generation.evidence import build_cards, render, summarize
from app.retrieval.retriever import retrieve
from app.retrieval.vector_index import ChunkVectorIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="带引用的问答（Step 6）")
    parser.add_argument("question", help="问题")
    parser.add_argument("-k", type=int, default=None, help="召回几条证据")
    parser.add_argument("--paper-id", default=None, help="只在某篇论文里检索")
    parser.add_argument("--all", action="store_true", help="强制全库检索（不做论文范围解析）")
    parser.add_argument("--dry-run", action="store_true", help="只打印 prompt，不调 LLM（不花钱）")
    parser.add_argument("--show-evidence", action="store_true", help="打印完整证据卡")
    parser.add_argument("--width", type=int, default=200, help="证据预览宽度")
    args = parser.parse_args()

    settings = get_settings()
    index = ChunkVectorIndex(settings)
    conn = connect()
    init_db(conn)
    try:
        k = args.k or settings.top_k
        stats: dict = {}
        items = retrieve(
            conn, args.question, k=k, index=index, settings=settings,
            paper_id=args.paper_id, auto_scope=not args.all, stats=stats,
        )
        cards = build_cards(items)
        info = summarize(cards)

        print(f"问题：{args.question}")
        _print_scope(args, stats, conn)
        timing = stats.get("timing", {})
        print(
            f"检索耗时：query embedding {timing.get('embed_ms', 0):.0f} ms"
            f" + 向量检索 {timing.get('search_ms', 0):.1f} ms"
        )
        print(
            f"证据：{info['cards']} 张卡 / {info['papers']} 篇论文 / {info['sections']} 个章节"
            f" / 含资产 {info['with_assets']}"
        )
        if stats.get("dropped"):
            print(f"[警告] 丢弃了 {len(stats['dropped'])} 条孤儿向量：{stats['dropped']}")
        print()

        if args.dry_run:
            print(f"=== system prompt（{SYSTEM_PROMPT_PATH.name}）===")
            print(build_user_message(args.question, cards))
            print()
            print("（--dry-run：没有调 LLM，没花钱）")
            return 0

        if not cards:
            # 没证据就不调 LLM，省钱也避免编造
            print("=== 回答 ===")
            print("资料不足：没有检索到与这个问题相关的证据。")
            return 0

        result = answer_from_cards(args.question, cards, settings=settings)

        print("=== 回答 ===")
        print(result.answer)
        print()

        _print_check(result, info)

        if args.show_evidence:
            print()
            print("=== 证据卡 ===")
            print(render(cards))
        return 0
    finally:
        conn.close()


def _print_check(result, info: dict) -> None:
    citations = result.citations
    print("=== 引用校验 ===")
    print(f"  LLM 耗时      : {result.latency_ms:.0f} ms")
    print(f"  用到的证据    : {', '.join(citations.cited) if citations.cited else '（一条都没有）'}")
    print(f"  证据覆盖率    : {citations.coverage:.0%}")
    if citations.unknown:
        # 最硬的幻觉信号：编号在证据里根本不存在
        print(f"  [警告] 编造引用：{', '.join(citations.unknown)}（证据只到 {info['cards']} 张卡）")
    if citations.looks_uncited and len(result.answer) > 60:
        print("  [警告] 回答很长但一条引用都没有 —— 可能是在用自己的知识回答，不是在读证据")
    if citations.insufficient:
        print("  说明：这轮回答的是「资料不足」")


def _print_scope(args, stats: dict, conn) -> None:
    """说清楚这次到底搜了哪些论文 —— 范围解析是隐式的，必须打出来才看得见。"""
    from app.db.repositories import papers as papers_repo

    scope = stats.get("scope") or []
    if args.paper_id:
        source = "手动指定 --paper-id"
    elif args.all:
        source = "手动 --all"
    elif scope:
        source = "从问题里解析出来的"
    else:
        source = "没解析出论文 → 全库"

    if scope:
        names = []
        for paper_id in scope:
            paper = papers_repo.get(conn, paper_id)
            names.append(paper_id if paper is None else f"{paper_id}《{paper['title'][:28]}》")
        print(f"检索范围：{len(scope)} 篇（{source}）")
        for name in names:
            print(f"            {name}")
    else:
        print(f"检索范围：全库（{source}）")


if __name__ == "__main__":
    raise SystemExit(main())
