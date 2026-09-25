"""Step 5 验收：检索 + 组装证据卡（不调 LLM）。

用法：
    .venv\\Scripts\\python.exe -m scripts.demo_retrieve "微表情识别的难点是什么"
    .venv\\Scripts\\python.exe -m scripts.demo_retrieve "消融实验" --paper-id pe-clip
    .venv\\Scripts\\python.exe -m scripts.demo_retrieve "表 1 的结果" --card-only

输出两段：
  1. 检索原始结果（带 similarity，按相似度排）—— 看召回质量；
  2. 组装好的证据卡（按**文档顺序**排、`[En]` 编号）—— 就是 Step 6 要喂给模型的原文。
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.db.connection import connect, init_db
from app.generation.evidence import build_cards, render, summarize
from app.retrieval.retriever import retrieve
from app.retrieval.vector_index import ChunkVectorIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="检索 + 证据卡（Step 5）")
    parser.add_argument("query", help="问题")
    parser.add_argument("-k", type=int, default=None, help="召回几条，默认取配置的 TOP_K")
    parser.add_argument("--paper-id", default=None, help="只在某篇论文里检索")
    parser.add_argument("--card-only", action="store_true", help="只打印证据卡")
    parser.add_argument("--raw-only", action="store_true", help="只打印检索结果，不组装")
    parser.add_argument("--show-trace", action="store_true", help="打印每张卡的溯源坐标（前端高亮用）")
    parser.add_argument("--width", type=int, default=160, help="预览宽度")
    args = parser.parse_args()

    settings = get_settings()
    index = ChunkVectorIndex(settings)
    conn = connect()
    init_db(conn)
    try:
        k = args.k or settings.top_k
        stats: dict = {}
        items = retrieve(
            conn, args.query, k=k, index=index, settings=settings,
            paper_id=args.paper_id, stats=stats,
        )
        timing = stats.get("timing", {})
        dropped = stats.get("dropped", [])

        print(f"问题：{args.query}")
        scope = stats.get("scope") or []
        scope_label = f"{len(scope)} 篇：{'、'.join(scope)}" if scope else "全库"
        print(f"范围：{scope_label}    top-{k}")
        print(
            f"耗时：query embedding {timing.get('embed_ms', 0):.0f} ms"
            f" + 向量检索 {timing.get('search_ms', 0):.1f} ms"
        )
        if dropped:
            # 向量在、SQLite 里没有 —— 论文被删过或索引没同步
            print(f"[警告] 丢弃了 {len(dropped)} 条孤儿向量：{dropped}")
        print()

        if not items:
            print("没有命中。先确认索引非空：python -m scripts.demo_vector_search --stats")
            return 1

        if not args.card_only:
            print("=== 检索结果（按相似度）===")
            for item in items:
                preview = " ".join((item.raw_content or "").split())
                if len(preview) > args.width:
                    preview = preview[: args.width] + "…"
                print(
                    f"{item.rank + 1}. similarity={item.similarity:+.4f}  {item.chunk_id}"
                    f"  p.{item.page_start}  {item.token_count} token"
                )
                print(f"   {preview}")
            print()

        if args.raw_only:
            return 0

        cards = build_cards(items)
        print("=== 证据卡（按文档顺序，资产已展开）===")
        print(render(cards))
        print()
        _print_summary(cards)
        if args.show_trace:
            _print_trace(cards)
        return 0
    finally:
        conn.close()


def _print_summary(cards) -> None:
    """人工核对用的几个数：资产有没有展开、章节有没有归对、块数对不对。"""
    info = summarize(cards)
    print("=== 证据卡自检 ===")
    print(f"  卡片数        : {info['cards']}")
    print(f"  覆盖论文      : {info['papers']}")
    print(f"  覆盖章节      : {info['sections']}")
    print(f"  含资产的卡片  : {info['with_assets']}  类型：{info['asset_kinds'] or '无'}")
    print(f"  溯源坐标点    : {info['trace_points']}（前端高亮用）")
    for card in cards:
        if not card.assets:
            continue
        kinds = ", ".join(sorted({asset["asset_type"] for asset in card.assets}))
        print(f"    {card.label} {card.chunk_id} → {kinds}")
    if info["missing"]:
        print(f"  [警告] 这些卡片里有占位符没找到对应资产：{', '.join(info['missing'])}")


def _print_trace(cards) -> None:
    """溯源坐标：前端点引用高亮时，按这些 bbox 在 PDF 页面上画框（PLAN §2.4）。"""
    print()
    print("=== 溯源坐标（归一化 0-1000 的 [x0,y0,x1,y1]）===")
    for card in cards:
        print(f"  [{card.label}] {card.chunk_id}  （{len(card.trace)} 块）")
        for point in card.trace:
            print(f"      p{point['page_idx']}  {point['bbox']}  {point['block_id']}")


if __name__ == "__main__":
    raise SystemExit(main())
