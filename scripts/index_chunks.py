"""Step 3：把 blocks 切分成 chunks 并入库（含 assets 和绑定）。

用法：
    # 一篇
    .venv\\Scripts\\python.exe -m scripts.index_chunks pe-clip

    # 全部论文
    .venv\\Scripts\\python.exe -m scripts.index_chunks --all

这一步**不联网、不花钱**：全部从已入库的 blocks 重算。改切分参数后直接重跑。
"""

from __future__ import annotations

import argparse
import statistics

from app.config import get_settings
from app.db.connection import connect, init_db
from app.db.repositories import assets as assets_repo
from app.db.repositories import blocks as blocks_repo
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import papers as papers_repo
from app.ingest.indexer import index_paper
from app.ingest.versioning import index_version


def main() -> int:
    parser = argparse.ArgumentParser(description="切分入库（Step 3）")
    parser.add_argument("paper_id", nargs="?", default=None, help="不给就配合 --all 用")
    parser.add_argument("--all", action="store_true", help="处理库里所有论文")
    parser.add_argument(
        "--skip-vectors",
        action="store_true",
        help="只切分写 SQLite，不写向量（不花钱，改切分参数时用）",
    )
    args = parser.parse_args()

    if not args.all and not args.paper_id:
        print("要么给 paper_id，要么用 --all")
        return 1

    settings = get_settings()
    conn = connect()
    init_db(conn)
    try:
        papers = papers_repo.list_all(conn)
        targets = [p["paper_id"] for p in papers] if args.all else [args.paper_id]

        print("=== Step 3 · 切分入库 ===")
        print(f"index_version : {index_version(settings)}")
        print(
            f"切分参数      : target={settings.chunk_target_tokens} "
            f"max={settings.chunk_max_tokens} min={settings.chunk_min_tokens} "
            f"block_overlap={settings.block_overlap} "
            f"text_overlap={settings.text_overlap_chars}"
        )
        print()

        all_tokens: list[int] = []
        for paper_id in targets:
            if paper_id is None:
                continue
            try:
                stats = index_paper(
                    conn, paper_id, settings, write_vectors=not args.skip_vectors
                )
            except ValueError as exc:
                print(f"  [跳过] {exc}")
                continue

            chunk_rows = chunks_repo.list_by_paper(conn, paper_id)
            tokens = [row["token_count"] or 0 for row in chunk_rows]
            all_tokens.extend(tokens)
            print(f"### {paper_id}")
            print(
                f"    {stats['chunks']:>3} 个 chunk   "
                f"token 中位={statistics.median(tokens) if tokens else 0:.0f} "
                f"最小={min(tokens) if tokens else 0} 最大={max(tokens) if tokens else 0}"
            )
            print(
                f"    资产 {assets_repo.count(conn)} 总 / 本篇 "
                f"{len(assets_repo.list_by_paper(conn, paper_id))}   "
                f"绑定 {assets_repo.binding_count(conn)} 总"
            )
            print(f"    blocks 总数 {blocks_repo.count(conn, paper_id)}，丢弃 {stats['dropped_blocks']}")
            if not args.skip_vectors:
                print(
                    f"    向量：删旧 {stats.get('vectors_removed', 0)}，"
                    f"写新 {stats.get('vectors_added', 0)}"
                )
            print()

        if all_tokens:
            _print_distribution(all_tokens, settings.chunk_min_tokens, settings.chunk_max_tokens)

        print("结论：Step 3 切分完成")
        print("下一步核对：")
        target = targets[0] if targets and targets[0] else "<paper_id>"
        print(f"  .venv\\Scripts\\python.exe -m scripts.dump_chunks {target}")
        return 0
    finally:
        conn.close()


def _print_distribution(tokens: list[int], minimum: int, maximum: int) -> None:
    ordered = sorted(tokens)
    print("=== 全部 chunk 的 token 分布 ===")
    print(
        f"  共 {len(ordered)} 个   中位={statistics.median(ordered):.0f}   "
        f"最小={ordered[0]}  最大={ordered[-1]}   合计={sum(ordered)}"
    )
    buckets = [(0, minimum), (minimum, 250), (250, 500), (500, maximum), (maximum, 10**9)]
    for low, high in buckets:
        count = sum(1 for value in ordered if low <= value < high)
        label = f"{low}-{high}" if high < 10**9 else f"{low}+"
        print(f"  {label:>10} token : {count:>3}  {'#' * count}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
