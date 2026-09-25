"""看切分结果（Step 3 验收）。

用法：
    .venv\\Scripts\\python.exe -m scripts.dump_chunks pe-clip
    .venv\\Scripts\\python.exe -m scripts.dump_chunks pe-clip --limit 3 --full
    .venv\\Scripts\\python.exe -m scripts.dump_chunks pe-clip --view index

人工核对三件事：
  1. chunk 的边界是不是落在章节/块之间（没有把公式表格切两半）；
  2. token_count 是不是都在 min-max 之间（超大表格除外）；
  3. 资产引用块（TABLE_REF / FIGURE_REF / FORMULA_REF）有没有出现在该在的地方。
"""

from __future__ import annotations

import argparse
import statistics

from app.db.connection import connect, init_db
from app.db.repositories import assets as assets_repo
from app.db.repositories import blocks as blocks_repo
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import papers as papers_repo
from app.db.repositories._util import loads_json


def main() -> int:
    parser = argparse.ArgumentParser(description="查看切分结果")
    parser.add_argument("paper_id", nargs="?", default=None, help="不给就列出所有论文")
    parser.add_argument("--limit", type=int, default=5, help="打印几个 chunk（默认 5）")
    parser.add_argument("--offset", type=int, default=0, help="从第几个开始")
    parser.add_argument("--width", type=int, default=200, help="正文预览宽度")
    parser.add_argument("--full", action="store_true", help="不截断")
    parser.add_argument(
        "--view",
        choices=("both", "content", "index"),
        default="both",
        help="打印 content（给模型看）还是 index_text（进向量）",
    )
    args = parser.parse_args()

    conn = connect()
    init_db(conn)
    try:
        if not args.paper_id:
            return _list_papers(conn)

        paper = papers_repo.get(conn, args.paper_id)
        if paper is None:
            print(f"库里没有这篇论文：{args.paper_id}")
            return 1

        rows = chunks_repo.list_by_paper(conn, args.paper_id)
        if not rows:
            print(f"{args.paper_id} 还没有 chunks，先跑：python -m scripts.index_chunks {args.paper_id}")
            return 1

        tokens = [row["token_count"] or 0 for row in rows]
        print(f"=== {paper['title']} ===")
        print(
            f"chunk 数 {len(rows)}   token 中位={statistics.median(tokens):.0f} "
            f"最小={min(tokens)} 最大={max(tokens)}"
        )
        print(f"index_version : {paper['index_version']}")
        print()

        picked = rows[args.offset : args.offset + args.limit]
        for row in picked:
            print("-" * 78)
            print(_header(conn, row))
            if args.view in ("both", "content"):
                print("  content:")
                print(_indent(_clip(row["content"], args.width, args.full)))
            if args.view in ("both", "index"):
                print("  index_text:")
                print(_indent(_clip(row["index_text"], args.width, args.full)))
            print()

        if len(rows) > args.offset + len(picked):
            print(f"（还有 {len(rows) - args.offset - len(picked)} 个，用 --limit / --offset 翻页）")
        return 0
    finally:
        conn.close()


def _header(conn, row) -> str:
    block_ids = loads_json(row["block_ids"]) or []
    asset_ids = [asset["asset_id"] for asset in assets_repo.assets_of_chunk(conn, row["chunk_id"])]
    pages = "" if row["page_start"] is None else f" p.{row['page_start']}"
    if row["page_end"] not in (None, row["page_start"]):
        pages += f"-{row['page_end']}"
    return (
        f"{row['chunk_id']}  [{row['chunk_type']}]  {row['token_count']} token  "
        f"{len(block_ids)} 块{pages}\n"
        f"  assets     : {', '.join(asset_ids) if asset_ids else '无'}"
    )


def _list_papers(conn) -> int:
    rows = papers_repo.list_all(conn)
    if not rows:
        print("库里还没有论文。先跑：python -m scripts.run_mineru <pdf>")
        return 1
    print("=== 库里的论文 ===")
    for row in rows:
        n_chunks = chunks_repo.count(conn, row["paper_id"])
        n_blocks = blocks_repo.count(conn, row["paper_id"])
        print(
            f"  {row['paper_id']:<48} {row['status']:<8} "
            f"{n_blocks:>4} 块  {n_chunks:>3} chunk"
        )
    return 0


def _clip(text: str | None, width: int, full: bool) -> str:
    value = (text or "").strip()
    if full or len(value) <= width:
        return value
    return value[:width] + "…"


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
