"""切分入库：一篇论文的 blocks → assets + chunks + chunk_assets（+ 向量）。

顺序（都包在一个事务里，要么全成要么全不成）：

    删旧的 chunks（级联清 chunk_assets）→ 删旧的 assets
    → 写 assets → 写 chunks → 重建关键词索引（chunks_fts，同一个库里能进事务）
    → 写 chunk_assets → 更新 papers.index_version
    → 事务提交后，再写向量（Chroma 不参与 SQLite 事务，见下）

**为什么先删后写**：重复入库同一篇论文必须幂等。`chunks` 有
`UNIQUE(paper_id, order_index)`，不先删就会撞唯一约束；而且重切分之后
旧的 chunk_id 可能不再存在，留着就是孤儿。

`papers.index_version` 写的是**本次切分用的参数指纹**（见 versioning.py），
所以"改切分参数 → 重跑切分"这件事是可判断的，不用重跑 MinerU。

**为什么向量写在事务之外**：Chroma 是另一个存储，跨不过去 SQLite 的事务。
顺序上让 SQLite 先提交、再写向量 —— SQLite 是权威源，向量是可以随时重建的派生索引；
反过来的话，向量写成功而 SQLite 回滚，就留下永远找不到主人的向量。

关键词索引（`chunks_fts`）**在事务之内**：它在同一个 SQLite 文件里，没有理由分开。
所以"chunks 有、关键词索引没有"这种不一致在这条路上不可能出现。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from app.config import Settings, get_settings
from app.db.connection import transaction
from app.db.repositories import assets as assets_repo
from app.db.repositories import blocks as blocks_repo
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import papers as papers_repo
from app.db.repositories import sections as sections_repo
from app.ingest import assets as assets_mod
from app.ingest import splitter as splitter_mod
from app.ingest.versioning import index_version
from app.retrieval import keyword_index
from app.retrieval.vector_index import ChunkVectorIndex

_ASSET_COLUMNS = (
    "asset_id",
    "block_id",
    "paper_id",
    "asset_type",
    "page_idx",
    "bbox",
    "raw_content",
    "image_path",
    "caption",
    "label_norm",
)

_CHUNK_COLUMNS = (
    "chunk_id",
    "paper_id",
    "order_index",
    "content",
    "index_text",
    "token_count",
    "block_ids",
    "page_start",
    "page_end",
    "chunk_type",
)


def index_paper(
    conn: sqlite3.Connection,
    paper_id: str,
    settings: Settings | None = None,
    write_vectors: bool = True,
    vector_index: ChunkVectorIndex | None = None,
) -> dict[str, Any]:
    """切分并入库一篇论文。返回统计信息。"""
    settings = settings or get_settings()

    paper = papers_repo.get(conn, paper_id)
    if paper is None:
        raise ValueError(f"库里没有这篇论文：{paper_id}（先跑 scripts.run_mineru）")

    blocks = [dict(row) for row in blocks_repo.list_by_paper(conn, paper_id)]
    if not blocks:
        raise ValueError(f"{paper_id} 没有 blocks，先跑 scripts.run_mineru")
    sections = [dict(row) for row in sections_repo.list_by_paper(conn, paper_id)]

    asset_rows = assets_mod.build_assets(paper_id, blocks)
    build = splitter_mod.split_paper(
        paper_id=paper_id,
        paper_title=paper["title"],
        blocks=blocks,
        sections=sections,
        assets=asset_rows,
        settings=settings,
    )

    # 向量库和 FTS 吃的是**同一份** items（chunk_id + index_text + metadata），
    # 在这儿算一次、两边共用。各自再拼一次文本迟早会不一致。
    items = _vector_items(build.chunks, sections, blocks)

    fts_rows = 0
    with transaction(conn):
        chunks_repo.delete_by_paper(conn, paper_id)
        assets_repo.delete_by_paper(conn, paper_id)
        assets_repo.upsert(conn, [_asset_row(asset) for asset in build.assets])
        chunks_repo.upsert(conn, [_chunk_row(chunk) for chunk in build.chunks])
        # 关键词索引跟 chunks 在**同一个事务**里重建 —— 这样"向量库和 SQLite 会漂"
        # 那类问题在这里不会发生：FTS 和 chunks 要么一起成，要么一起回滚。
        fts_rows = keyword_index.replace_paper(conn, paper_id, items)
        for chunk in build.chunks:
            if chunk["_asset_ids"]:
                assets_repo.link_chunk(conn, chunk["chunk_id"], chunk["_asset_ids"])
        papers_repo.set_indexed(conn, paper_id, index_version(settings))

    stats = dict(build.stats)
    stats["fts_rows"] = fts_rows
    if write_vectors:
        index = vector_index or ChunkVectorIndex(settings)
        # 先删后写：重复入库是幂等的，不会留下孤儿向量
        stats["vectors_removed"] = index.reset_paper(paper_id)
        stats["vectors_added"] = index.add_chunks(items)
    return stats


def delete_paper(
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    settings: Settings | None = None,
    index: ChunkVectorIndex | None = None,
) -> dict[str, int]:
    """删掉一篇论文的索引和数据。**顺序不能反**（PLAN §2.3）：

        1. 先删**向量**（Chroma 里按 paper_id 过滤）
        2. 再删**关键词镜像**（chunks_fts）
        3. 最后删 **SQLite**（`DELETE FROM papers`，外键级联清掉 sections /
           blocks / chunks / assets / chunk_assets）

    反过来做的话，SQLite 那行先没了、Chroma 里还留着 —— 那批向量永远找不到主人
    （检索时回表查不到，只能白占一个候选位被丢弃），而且再没人知道该删它们。
    先删外部索引还有个好处：中途失败可以直接重试，重试是幂等的。

    **刻意保留**：`parse_cache` 那一行和 `storage/mineru/{paper_id}/` 目录（含原图）。
    重传同一份 PDF 还能秒回。要彻底清干净是另一个显式动作，不在这儿顺手做。

    向量那一步在 SQLite 之外，先做完；SQLite 的两步在**一个事务**里。
    """
    settings = settings or get_settings()
    index = index or ChunkVectorIndex(settings)

    vectors = index.reset_paper(paper_id)
    with transaction(conn):
        keyword_rows = keyword_index.delete_paper(conn, paper_id)
        chunks = chunks_repo.count(conn, paper_id)
        removed = papers_repo.delete(conn, paper_id)
    return {
        "paper_id": paper_id,
        "removed": removed,
        "chunks": chunks,
        "vectors": vectors,
        "keyword_rows": keyword_rows,
    }


def _vector_items(
    chunks: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把 chunks 转成向量库要的形状。

    metadata 里的 `section_type` 从 chunk 的首块推：`block_ids[0]` 对应的
    `blocks.section_id` → `sections.section_type`。`chunks` 表没有 section_id 列
    （见 PLAN §2.1），所以这里查一次。
    """
    section_type = {section["section_id"]: section.get("section_type") for section in sections}
    block_section = {block["block_id"]: block.get("section_id") for block in blocks}
    items: list[dict[str, Any]] = []
    for chunk in chunks:
        block_ids = chunk.get("block_ids") or []
        first_block = block_ids[0] if block_ids else None
        first_section = block_section.get(first_block) if first_block else None
        items.append(
            {
                "chunk_id": chunk["chunk_id"],
                "index_text": chunk["index_text"],
                "metadata": {
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "section_type": section_type.get(first_section),
                    "page_start": chunk.get("page_start"),
                    "chunk_type": chunk.get("chunk_type"),
                },
            }
        )
    return items


def _asset_row(asset: dict[str, Any]) -> dict[str, Any]:
    """只取库里有列的那部分 —— build_assets 还带了 `_label` / `_preview` 两个切分用的临时字段。"""
    return {column: asset.get(column) for column in _ASSET_COLUMNS}


def _chunk_row(chunk: dict[str, Any]) -> dict[str, Any]:
    """同理：`_asset_ids` 只用于写 chunk_assets，不是 chunks 表的列。"""
    return {column: chunk.get(column) for column in _CHUNK_COLUMNS}
