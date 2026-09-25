"""Step 4 验收：向量索引自检 + 裸检索。

用法：
    # 看索引状态（条数对不对、每篇多少条）
    .venv\\Scripts\\python.exe -m scripts.demo_vector_search --stats

    # 检索（默认全库）
    .venv\\Scripts\\python.exe -m scripts.demo_vector_search "微表情识别的难点是什么"

    # 只在某篇论文里检索（验证 paper_id 过滤）
    .venv\\Scripts\\python.exe -m scripts.demo_vector_search "表 1 的结果" --paper-id pe-clip

    # 看命中的是 index_text 还是 content
    .venv\\Scripts\\python.exe -m scripts.demo_vector_search "消融实验" --view index

这一步**只做向量检索**，不组装证据卡、不调 LLM —— 那是 Step 5/6 的事。
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.db.connection import connect, init_db
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import papers as papers_repo
from app.retrieval.vector_index import ChunkVectorIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="向量检索自检（Step 4）")
    parser.add_argument("query", nargs="?", default=None, help="问题")
    parser.add_argument("-k", type=int, default=None, help="返回几条，默认取配置的 TOP_K")
    parser.add_argument("--paper-id", default=None, help="只在某篇论文里检索")
    parser.add_argument("--view", choices=("content", "index"), default="content")
    parser.add_argument("--width", type=int, default=220)
    parser.add_argument("--stats", action="store_true", help="只打印索引状态")
    parser.add_argument("--repeat", type=int, default=1, help="重复检索几次，看耗时稳定性")
    args = parser.parse_args()

    settings = get_settings()
    index = ChunkVectorIndex(settings)

    conn = connect()
    init_db(conn)
    try:
        if args.stats or not args.query:
            return _print_stats(conn, index)

        k = args.k or settings.top_k
        hits: list = []
        timing: dict[str, float] = {}
        for _ in range(max(1, args.repeat)):
            hits, timing = index.search_timed(args.query, k=k, paper_id=args.paper_id)

        scope = args.paper_id or "全库"
        print(f'问题：{args.query}')
        print(f"范围：{scope}    top-{k}")
        # 分开报：embedding 是网络往返，Chroma 查询才是"检索本身"（验收标准是后者 <50ms）
        print(
            f"耗时：query embedding {timing['embed_ms']:.0f} ms"
            f" + 向量检索 {timing['search_ms']:.1f} ms"
            f" = {timing['total_ms']:.0f} ms"
        )
        if timing["search_ms"] < 50:
            print("      （向量检索 <50ms [OK] 验收通过；embedding 那部分是网络往返，不算在内）")
        print()

        if not hits:
            print("没有命中。先确认索引非空：python -m scripts.demo_vector_search --stats")
            return 1

        for rank, hit in enumerate(hits, 1):
            row = chunks_repo.get(conn, hit.chunk_id)
            if row is None:
                # 向量在、SQLite 里没有 —— 论文被删过或索引没同步，直接丢弃
                print(f"{rank}. [丢弃] {hit.chunk_id} 在 SQLite 里不存在（向量是孤儿）")
                continue
            paper = papers_repo.get(conn, row["paper_id"])
            text = row[args.view] if args.view in row.keys() else row["content"]
            preview = " ".join((text or "").split())
            if len(preview) > args.width:
                preview = preview[: args.width] + "…"
            print(
                f"{rank}. similarity={hit.similarity:+.4f}  {hit.chunk_id}\n"
                f"   论文：《{(paper['title'] if paper else '?')[:56]}》\n"
                f"   页码：p.{row['page_start']}  类型：{row['chunk_type']}  "
                f"token：{row['token_count']}  metadata：{hit.metadata}\n"
                f"   {args.view}：{preview}\n"
            )
        return 0
    finally:
        conn.close()


def _print_stats(conn, index: ChunkVectorIndex) -> int:
    by_paper = index.counts_by_paper()
    total = sum(by_paper.values())
    print("=== 向量索引状态 ===")
    print(f"collection 条数 : {total}")
    print(f"持久化目录      : {index.settings.chroma_dir}")
    print()

    print("=== 与 SQLite 对账 ===")
    print(f"{'paper_id':<48}{'SQLite':>8}{'Chroma':>8}")
    print("-" * 66)
    sqlite_total = 0
    problems = 0
    for paper in papers_repo.list_all(conn):
        paper_id = paper["paper_id"]
        n_sqlite = chunks_repo.count(conn, paper_id)
        n_chroma = by_paper.get(paper_id, 0)
        sqlite_total += n_sqlite
        flag = "" if n_sqlite == n_chroma else "  ← 对不上"
        problems += 0 if n_sqlite == n_chroma else 1
        print(f"{paper_id:<48}{n_sqlite:>8}{n_chroma:>8}{flag}")
    print("-" * 66)
    print(f"{'合计':<48}{sqlite_total:>8}{total:>8}")
    print()

    if problems:
        print(f"有 {problems} 篇对不上。修复：python -m scripts.index_chunks --all")
        return 1
    print("结论：Step 4 索引对得上（collection 条数 == chunks 条数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
