"""打印一篇论文的块清单（Step 2 验收：随机抽 10 块人工核对）。

用法：
    # 不带参数 → 列出库里有哪些论文
    .venv\\Scripts\\python.exe -m scripts.dump_blocks

    # 看某一篇
    .venv\\Scripts\\python.exe -m scripts.dump_blocks vit-2021

    # 只看表格块
    .venv\\Scripts\\python.exe -m scripts.dump_blocks vit-2021 --type table

    # 随机抽 10 块（人工核对用：块是不是切对了、图注配对了没）
    .venv\\Scripts\\python.exe -m scripts.dump_blocks vit-2021 --sample 10

    # 不截断正文
    .venv\\Scripts\\python.exe -m scripts.dump_blocks vit-2021 --full

人工核对就看三件事：① 块类型对不对；② 图注有没有配到图片/表格上；
③ order_index 连起来读是不是原文顺序。
"""

from __future__ import annotations

import argparse
import random

from app.db.connection import connect, init_db
from app.db.repositories import blocks, papers
from app.db.repositories._util import loads_json


def main() -> int:
    parser = argparse.ArgumentParser(description="打印块清单")
    parser.add_argument("paper_id", nargs="?", default=None, help="不给就列出所有论文")
    parser.add_argument("--type", dest="block_type", default=None, help="只看某一种块类型")
    parser.add_argument("--limit", type=int, default=40, help="最多打印几块（--sample 时忽略）")
    parser.add_argument("--sample", type=int, default=0, help="随机抽 N 块")
    parser.add_argument("--width", type=int, default=90, help="正文预览宽度")
    parser.add_argument("--full", action="store_true", help="不截断正文")
    args = parser.parse_args()

    conn = connect()
    init_db(conn)
    try:
        if not args.paper_id:
            return _list_papers(conn)

        paper = papers.get(conn, args.paper_id)
        if paper is None:
            print(f"库里没有这篇论文：{args.paper_id}")
            return 1

        rows = blocks.list_by_paper(conn, args.paper_id)
        if args.block_type:
            rows = [row for row in rows if row["block_type"] == args.block_type]
        if not rows:
            print("没有符合条件的块。先跑 scripts.run_mineru 把论文入库。")
            return 1

        print(f"=== {paper['title']} ===")
        print(f"paper_id : {args.paper_id}")
        print(f"块数     : {len(rows)}" + (f"（已按 type={args.block_type} 过滤）" if args.block_type else ""))
        print()

        if args.sample:
            picked = sorted(random.sample(rows, min(args.sample, len(rows))), key=lambda r: r["order_index"])
            print(f"=== 随机抽 {len(picked)} 块 ===")
        else:
            picked = rows[: args.limit]
            if len(rows) > len(picked):
                print(f"=== 只显示前 {len(picked)} 块（共 {len(rows)}，用 --limit 调整）===")

        for row in picked:
            print(_format_row(row, args.width, args.full))
        return 0
    finally:
        conn.close()


def _list_papers(conn) -> int:
    rows = papers.list_all(conn)
    if not rows:
        print("库里还没有论文。先跑：python -m scripts.run_mineru <pdf>")
        return 1
    print("=== 库里的论文 ===")
    for row in rows:
        n_blocks = blocks.count(conn, row["paper_id"])
        print(f"  {row['paper_id']:<24} {row['status']:<8} {n_blocks:>5} 块  {row['title']}")
    print()
    print("看某一篇：python -m scripts.dump_blocks <paper_id>")
    return 0


def _format_row(row, width: int, full: bool) -> str:
    parts = [f"#{row['order_index']:>5} [{row['block_type']:<8}]"]
    if row["page_idx"] is not None:
        parts.append(f"p{row['page_idx']}")
    parts.append(f"section={row['section_id'] or '-'}")
    head = " ".join(parts)

    body = row["caption"] or row["text"] or row["latex"] or row["html"] or ""
    body = " ".join(str(body).split())
    if not full and len(body) > width:
        body = body[:width] + "…"

    lines = [head]
    if body:
        lines.append(f"        {body}")
    if row["image_path"]:
        lines.append(f"        image: {row['image_path']}")
    bbox = loads_json(row["bbox"])
    if bbox:
        lines.append(f"        bbox: {bbox}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
