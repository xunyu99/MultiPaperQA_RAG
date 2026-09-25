"""Step 2 验收脚本：PDF → MinerU 解析 → blocks + 章节树 → 落库。

用法（在 E:\\agent\\MultiPaperQA_RAG 下）：

    # 正常走一遍（会调 MinerU，花钱）
    .venv\\Scripts\\python.exe -m scripts.run_mineru storage/pdfs/xxx.pdf

    # 指定 paper_id（默认从文件名推）
    .venv\\Scripts\\python.exe -m scripts.run_mineru storage/pdfs/xxx.pdf --paper-id vit-2021

    # 已经有解析产物，跳过 MinerU 直接解析落库（离线调试 / 重跑用）
    .venv\\Scripts\\python.exe -m scripts.run_mineru storage/pdfs/xxx.pdf --from-dir storage/mineru/vit-2021

    # 连 PDF 都没有：用仓库自带的样例产物跑通整条链路，零成本
    .venv\\Scripts\\python.exe -m scripts.run_mineru --from-dir tests/fixtures/mineru_demo --paper-id demo

    # 无视 parse_cache，强制重新调 MinerU
    .venv\\Scripts\\python.exe -m scripts.run_mineru storage/pdfs/xxx.pdf --force

产出：papers / sections / blocks 三张表 + parse_cache 一行，并打印块统计与章节树。

注：`--from-dir` + 无 PDF 时不会写 parse_cache —— 缓存要靠 content_hash 做键，
没有 PDF 就没有这个键，硬造一个反而会污染缓存。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.ingest import converter, section_tree
from app.ingest.mineru_client import count_images, find_markdown
from app.ingest.pipeline import IngestError, ParseReport, parse_paper

# 章节层级是从哪来的 —— 打印出来才知道这棵树可不可信
_HIERARCHY_LABELS = {
    "mineru": "MinerU 自己给了层级差异，直接采用",
    "numbering": "MinerU 给的是平的 → 靠标题编号推的父子关系",
    "flat-fallback": "MinerU 平级且标题认不出编号 → 全按顶层处理，层级不可信",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="MinerU 解析入库（Step 2）")
    parser.add_argument("pdf", nargs="?", default=None, help="PDF 路径（用 --from-dir 时可省略）")
    parser.add_argument("--paper-id", default=None, help="默认从文件名推")
    parser.add_argument("--title", default=None, help="覆盖论文标题（默认取第一个标题块）")
    parser.add_argument("--citation", default=None, help="覆盖出处串（默认从解析文本里抽）")
    parser.add_argument("--from-dir", default=None, help="用本地已有产物，跳过 MinerU")
    parser.add_argument("--content-hash", default=None, help="没带 PDF 时手动给 sha256（可选）")
    parser.add_argument("--force", action="store_true", help="无视 parse_cache，强制重新解析")
    args = parser.parse_args()

    pdf: Path | None = Path(args.pdf).resolve() if args.pdf else None
    if pdf is None and not args.paper_id:
        print("没给 PDF 就必须用 --paper-id 指定论文 id，例如：")
        print("  python -m scripts.run_mineru --from-dir tests/fixtures/mineru_demo --paper-id demo")
        return 1

    try:
        report = parse_paper(
            pdf=pdf,
            paper_id=args.paper_id,
            title=args.title,
            citation=args.citation,
            from_dir=args.from_dir,
            content_hash=args.content_hash,
            force=args.force,
            on_progress=_tick_printer(),
        )
    except IngestError as exc:
        print(f"[失败] {exc}")
        return 1

    _print_report(report)
    return 0


def _tick_printer():
    """MinerU 轮询状态每秒一次，只在状态变化时打一行，别刷屏。"""
    last: dict[str, str | None] = {"value": None}

    def on_tick(state: str, item: dict) -> None:
        if state == last["value"]:
            return
        last["value"] = state
        print(f"  [{time.strftime('%H:%M:%S')}] state={state or '未知'}")

    return on_tick


def _print_report(report: ParseReport) -> None:
    paper_id = report.paper_id
    mineru_dir = report.mineru_dir
    build = report.build
    print()
    print("=== Step 2 · MinerU 解析入库 ===")
    print(f"paper_id      : {paper_id}")
    print(f"论文标题      : {report.title}")
    print(f"出处串        : {report.citation or '（没抽到）'}")
    print(f"产物来源      : {report.mineru_source}")
    print(f"产物目录      : {mineru_dir}")
    if report.content_hash:
        print(f"content_hash  : {report.content_hash[:16]}…（同一个 PDF 再传一次会命中缓存）")
    else:
        print("content_hash  : （没有 PDF，本次不写 parse_cache）")
    print(f"mineru_version: {report.mineru_fingerprint}   ← 缓存键，只有它变才会重调 MinerU")
    print(f"parse_version : {report.parse_version}")

    print()
    print("=== 解析产物 ===")
    markdown = find_markdown(mineru_dir)
    content_lists = sorted(mineru_dir.rglob("*content_list.json"))
    middle_files = sorted(mineru_dir.rglob("*_middle.json"))
    print(f"  markdown     : {markdown.name if markdown else '缺（没找到 full.md）'}")
    print(f"  content_list : {content_lists[0].name if content_lists else '缺'}")
    print(f"  middle.json  : {middle_files[0].name if middle_files else '缺'}")
    print(f"  images/      : {count_images(mineru_dir)} 张图")

    print()
    print("=== 块统计 ===")
    print(f"  总块数   : {len(build.blocks)}")
    counts = converter.type_counts(build.blocks)
    print("  类型分布 : " + " / ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for kind, (with_caption, total) in converter.caption_coverage(build.blocks).items():
        print(f"  {kind:<8} 图注配对 : {with_caption}/{total}")

    print()
    print("=== 章节树 ===")
    print(f"  层级来源 : {_HIERARCHY_LABELS.get(build.hierarchy_source, build.hierarchy_source)}")
    if build.untokenized_titles:
        print(f"             其中 {build.untokenized_titles} 个标题认不出编号，按顶层处理")
    print(section_tree.format_tree(build.sections, build.block_counts))

    print()
    print("结论：Step 2 入库完成（papers / sections / blocks 已写入）")
    print("下一步人工核对：")
    print(f"  .venv\\Scripts\\python.exe -m scripts.dump_blocks {paper_id} --sample 10")
    print(f"  .venv\\Scripts\\python.exe -m scripts.dump_sections {paper_id}")


if __name__ == "__main__":
    raise SystemExit(main())
