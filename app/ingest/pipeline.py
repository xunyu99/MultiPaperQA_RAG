"""入库链路的**可调用入口**：PDF -> MinerU 产物 -> blocks + 章节树 -> 落库。

为什么要有这一层：Step 8 的 `POST /papers`（上传论文）和 `scripts/run_mineru.py`
要走**同一条**链路。逻辑留在脚本的 `main()` 里的话，接口只能去 shell 调子进程，
失败信息、进度、事务都不好接。所以把逻辑搬到这儿，两边都调它。

**这一层只负责"解析入库"（papers / sections / blocks）**，不切分、不写向量 ——
那是 `app/ingest/indexer.index_paper()` 的活（Step 3）。上传接口会把两个都调一遍。

两个必须守住的点：

1. **重新解析同一篇论文时，旧 blocks/sections 必须先删**。否则新旧数据混在一起，
   章节树和块顺序都是错的，而且不会报错 —— 只是检索结果悄悄变差。
2. **`index_version` 显式置空**。blocks 变了，之前基于旧 blocks 切出来的 chunks
   和向量全部作废（`papers.index_version` 为空就是在说"这篇的索引是脏的"）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import Settings, get_settings
from app.db.connection import connect, init_db, transaction
from app.db.repositories import blocks, parse_cache, papers, sections
from app.ingest import converter, section_tree
from app.ingest.citation import extract_citation
from app.ingest.mineru_client import MinerUClient, find_content_list
from app.ingest.versioning import default_paper_id, mineru_version, parse_version, sha256_file


class IngestError(Exception):
    """用户能看懂的入库失败（路径不对、参数不够……），不是程序 bug。"""


@dataclass
class ParseReport:
    """一次解析入库的结果。够接口和 CLI 各自拼输出了。"""

    paper_id: str
    title: str
    citation: str | None
    mineru_dir: Path
    mineru_source: str
    content_hash: str | None
    parse_version: str
    mineru_fingerprint: str
    build: section_tree.SectionBuild


def parse_paper(
    *,
    pdf: Path | None = None,
    paper_id: str | None = None,
    title: str | None = None,
    citation: str | None = None,
    from_dir: Path | str | None = None,
    content_hash: str | None = None,
    force: bool = False,
    settings: Settings | None = None,
    conn: sqlite3.Connection | None = None,
    on_progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> ParseReport:
    """解析一篇论文并落库（papers / sections / blocks）。

    - `from_dir` 给了就用本地已有产物，**不调 MinerU**（离线调试、重跑用）；
    - 没给就找 `parse_cache`（同一份 PDF + 同一套解析配置命中缓存），
      命中同样跳过 MinerU，**重复上传不重复花钱**；
    - `conn` 不传就自己开一个（接口里传进来，方便复用连接）。
    """
    settings = settings or get_settings()
    settings.ensure_dirs()

    if pdf is not None:
        pdf = Path(pdf).resolve()
        if not pdf.is_file():
            raise IngestError(f"PDF 不存在：{pdf}")
    if pdf is None and not paper_id:
        raise IngestError("没给 PDF 就必须指定 paper_id")

    resolved_paper_id = paper_id or default_paper_id(pdf)
    resolved_hash = content_hash or (sha256_file(pdf) if pdf else None)
    parser_version = parse_version(settings)
    fingerprint = mineru_version(settings)

    own_conn = conn is None
    if own_conn:
        conn = connect()
        init_db(conn)
    try:
        mineru_dir, source = _resolve_mineru_dir(
            conn=conn,
            from_dir=from_dir,
            force=force,
            settings=settings,
            pdf=pdf,
            paper_id=resolved_paper_id,
            content_hash=resolved_hash,
            mineru_fingerprint=fingerprint,
            on_progress=on_progress,
        )

        content_list_path = find_content_list(mineru_dir)
        raw_blocks = converter.to_blocks(
            resolved_paper_id, converter.load_content_list(content_list_path)
        )
        if not raw_blocks:
            raise IngestError(f"解析出来的块数是 0，检查 {content_list_path.name} 是不是空的")

        build = section_tree.build_sections(resolved_paper_id, raw_blocks)
        resolved_title = title or converter.extract_title(
            raw_blocks, pdf.stem if pdf else resolved_paper_id
        )
        resolved_citation = citation or extract_citation(
            build.blocks, build.sections, fallback=resolved_title
        )

        with transaction(conn):
            # 见模块开头第 1 条：旧数据必须先删干净
            blocks.delete_by_paper(conn, resolved_paper_id)
            sections.delete_by_paper(conn, resolved_paper_id)
            papers.upsert(
                conn,
                {
                    "paper_id": resolved_paper_id,
                    "title": resolved_title,
                    "citation": resolved_citation,
                    "pdf_path": str(pdf) if pdf else None,
                    "mineru_dir": str(mineru_dir),
                    "content_hash": resolved_hash,
                    "parse_version": parser_version,
                    # 见模块开头第 2 条
                    "index_version": None,
                    "status": papers.STATUS_PARSED,
                },
            )
            sections.upsert(conn, build.sections)
            blocks.upsert(conn, build.blocks)
            if resolved_hash:
                parse_cache.upsert(
                    conn,
                    {
                        "content_hash": resolved_hash,
                        "mineru_version": fingerprint,
                        "paper_id": resolved_paper_id,
                        "mineru_dir": str(mineru_dir),
                    },
                )

        return ParseReport(
            paper_id=resolved_paper_id,
            title=resolved_title,
            citation=resolved_citation,
            mineru_dir=mineru_dir,
            mineru_source=source,
            content_hash=resolved_hash,
            parse_version=parser_version,
            mineru_fingerprint=fingerprint,
            build=build,
        )
    finally:
        if own_conn and conn is not None:
            conn.close()


def _resolve_mineru_dir(
    *,
    conn: sqlite3.Connection,
    from_dir: Path | str | None,
    force: bool,
    settings: Settings,
    pdf: Path | None,
    paper_id: str,
    content_hash: str | None,
    mineru_fingerprint: str,
    on_progress: Callable[[str, dict[str, Any]], None] | None,
) -> tuple[Path, str]:
    """决定这次用哪个产物目录，返回 (目录, 来源说明)。"""
    if from_dir is not None:
        return Path(from_dir).resolve(), "指定 from_dir"

    if pdf is None:
        raise IngestError("没有 PDF 又不能跳过 MinerU：请给 PDF 路径，或指定产物目录")

    if not force and content_hash:
        cached = parse_cache.get(conn, content_hash, mineru_fingerprint)
        if cached and Path(cached["mineru_dir"]).is_dir():
            return Path(cached["mineru_dir"]), "parse_cache 命中（同一份 PDF + 同一套解析配置）"

    dest = settings.storage_dir / "mineru" / paper_id
    with MinerUClient(settings) as client:
        client.parse_pdf(pdf, dest, on_tick=on_progress)
    return dest, "调用 MinerU"
