"""打印章节树 + section_type 分布 + 关键词落点（Step 2 验收）。

用法：
    .venv\\Scripts\\python.exe -m scripts.dump_sections vit-2021
    .venv\\Scripts\\python.exe -m scripts.dump_sections vit-2021 --keywords 贡献,创新点,contribution,novelty

验收标准（PLAN Step 2）：章节能打印成缩进树；「贡献 / 创新点」落在
abstract / intro / conclusion 里 —— 最后一节就是查这个的。
"""

from __future__ import annotations

import argparse
from collections import Counter

from app.db.connection import connect, init_db
from app.db.repositories import blocks, papers, sections
from app.ingest.section_tree import format_tree

DEFAULT_KEYWORDS = ("贡献", "创新", "contribution", "novelty", "novel")


def main() -> int:
    parser = argparse.ArgumentParser(description="打印章节树")
    parser.add_argument("paper_id", nargs="?", default=None, help="不给就列出所有论文")
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="逗号分隔，查这些词落在哪些章节类型里",
    )
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

        section_rows = sections.list_by_paper(conn, args.paper_id)
        block_rows = blocks.list_by_paper(conn, args.paper_id)
        if not section_rows:
            print("这篇论文还没有章节数据，先跑 scripts.run_mineru")
            return 1

        counts = Counter(row["section_id"] for row in block_rows)

        print(f"=== {paper['title']} ===")
        print(f"paper_id : {args.paper_id}")
        print(f"章节数   : {len(section_rows)}    块数: {len(block_rows)}")
        print()
        print("=== 章节树 ===")
        print(format_tree([dict(row) for row in section_rows], dict(counts)))

        print()
        print("=== section_type 分布 ===")
        type_counts = Counter((row["section_type"] or "other") for row in section_rows)
        for section_type, n in sorted(type_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {section_type:<12} {n:>3} 个章节")

        print()
        print("=== 关键词落点 ===")
        return _keyword_report(args.keywords, section_rows, block_rows)
    finally:
        conn.close()


def _keyword_report(keywords_arg: str, section_rows, block_rows) -> int:
    keywords = [k.strip().lower() for k in keywords_arg.split(",") if k.strip()]
    section_type = {row["section_id"]: (row["section_type"] or "other") for row in section_rows}
    section_title = {row["section_id"]: row["title"] for row in section_rows}

    hits = 0
    for row in block_rows:
        text = " ".join(
            str(row[key] or "") for key in ("text", "caption", "html", "latex")
        ).lower()
        matched = [k for k in keywords if k in text]
        if not matched:
            continue
        hits += 1
        sid = row["section_id"]
        snippet = " ".join(str(row["text"] or row["caption"] or "").split())[:60]
        print(
            f"  [{section_type.get(sid, 'other'):<12}] {section_title.get(sid, '-'):<24} "
            f"#{row['order_index']:>5}  命中 {','.join(matched)}  {snippet}"
        )

    if hits == 0:
        print(f"  没命中。关键词：{keywords_arg}（多半是解析没认出标题，看上面的树对不对）")
    else:
        print(f"  共 {hits} 块命中 —— 期望落在 abstract / intro / conclusion 里，落在别处就要看看标题归一化对不对")
    return 0


def _list_papers(conn) -> int:
    rows = papers.list_all(conn)
    if not rows:
        print("库里还没有论文。先跑：python -m scripts.run_mineru <pdf>")
        return 1
    print("=== 库里的论文 ===")
    for row in rows:
        n_sections = sections.count(conn, row["paper_id"])
        print(f"  {row['paper_id']:<24} {row['status']:<8} {n_sections:>4} 章  {row['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
