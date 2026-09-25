"""诊断工具：一个 query，逐篇论文看完整排序（比 top-k 看得深）。

用途：判断某道题失败到底是"配额没生效""gold 章节压根排不进来"还是"就差一名"。
只花 **1 次 embedding**（query 向量算一次，后面按论文查都是本地向量库查询）。

用法：
    # 自动解析 scope（问题里提到论文名就缩范围，否则全库）
    .venv\\Scripts\\python.exe -m scripts.diag_scope "pe-clip 和 emotion-qwen-vl 用的方法有什么不同" -n 10

    # 手动指定只看哪几篇
    .venv\\Scripts\\python.exe -m scripts.diag_scope "表 1 的结果" --paper pe-clip -n 10

明细会写到 .eval_out/scope.json（可改 --dump，0 表示不写）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.db.connection import connect, init_db
from app.db.repositories import papers as papers_repo
from app.retrieval.retriever import _enrich
from app.retrieval.scope import resolve_scope
from app.retrieval.vector_index import ChunkVectorIndex
from app.db.repositories import chunks as chunks_repo

DEFAULT_DUMP_PATH = Path(__file__).resolve().parent.parent / ".eval_out" / "scope.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="逐篇看一个 query 的排序（诊断用，1 次 embedding）")
    parser.add_argument("query", nargs="?", default=None, help="问题原文；不给就只列论文和用法")
    parser.add_argument("-n", type=int, default=10, help="每篇论文看多深，默认 10")
    parser.add_argument("--paper", default=None, help="逗号分隔的 paper_id；不给就自动解析")
    parser.add_argument("--dump", default=str(DEFAULT_DUMP_PATH), help="明细写这里，0 表示不写")
    args = parser.parse_args()

    settings = get_settings()
    conn = connect()
    init_db(conn)
    try:
        papers = [dict(row) for row in papers_repo.list_all(conn)]
        if args.query is None:
            print("=== 库里的论文 ===")
            for paper in papers:
                count = chunks_repo.count(conn, paper["paper_id"])
                print(f"  {paper['paper_id']:<44} {count:>3} chunk  {paper.get('title') or ''}")
            print()
            print('用法：python -m scripts.diag_scope "你的问题" -n 10')
            return 0

        if args.paper:
            scope = [item.strip() for item in args.paper.split(",") if item.strip()]
        else:
            scope = resolve_scope(args.query, papers)

        index = ChunkVectorIndex(settings)
        vector = index.embedder.embed_query(args.query)
        print(f"问题：{args.query}")
        print(f"范围：{scope or '全库'}")
        print()

        report = {"query": args.query, "scope": scope, "papers": {}, "all": []}

        targets = scope or [paper["paper_id"] for paper in papers]
        for paper_id in targets:
            hits = index.search_by_vector(vector, k=args.n, paper_id=paper_id)
            rows = chunks_repo.get_many(conn, [hit.chunk_id for hit in hits])
            by_id = {row["chunk_id"]: dict(row) for row in rows}
            print(f"--- {paper_id}（{len(hits)} 条）---")
            entries = []
            for hit in hits:
                row = by_id.get(hit.chunk_id)
                if row is None:
                    continue
                item = _enrich(conn, row, hit.similarity, 0)
                line = (
                    f"  {hit.similarity:+.4f}  {item.section_title or '?'}"
                    f"  ({item.section_type or '?'}, {item.token_count}t)"
                )
                print(line)
                # 预览用 index_text：向量/BM25 实际比的就是它，看这个才知道"为什么排这么高"
                print(f"            {item.chunk_id}  {' '.join((row.get('index_text') or '').split())[:90]}")
                entries.append(
                    {
                        "similarity": round(hit.similarity, 4),
                        "chunk_id": item.chunk_id,
                        "section_title": item.section_title,
                        "section_type": item.section_type,
                        "token_count": item.token_count,
                        "preview": " ".join(item.content.split())[:200],
                    }
                )
            report["papers"][paper_id] = entries
            print()

        if not scope:
            all_hits = index.search_by_vector(vector, k=args.n)
            rows = chunks_repo.get_many(conn, [hit.chunk_id for hit in all_hits])
            by_id = {row["chunk_id"]: dict(row) for row in rows}
            print(f"--- 全库 top-{args.n} ---")
            for hit in all_hits:
                row = by_id.get(hit.chunk_id)
                if row is None:
                    continue
                item = _enrich(conn, row, hit.similarity, 0)
                print(f"  {hit.similarity:+.4f}  {item.paper.get('paper_id'):<40} {item.section_title or '?'}")
                report["all"].append(
                    {
                        "similarity": round(hit.similarity, 4),
                        "chunk_id": item.chunk_id,
                        "paper_id": item.paper.get("paper_id"),
                        "section_title": item.section_title,
                    }
                )
            print()

        if args.dump and args.dump != "0":
            path = Path(args.dump)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"明细已写入：{path}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
