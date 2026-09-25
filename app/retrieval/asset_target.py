"""查询里的资产编号 → 它**本体所在的那个 chunk**（"表 3 里写了什么"）。

这是"前置定点注入"的解析半：`(table, "3")` → `label_norm='table:3'` → 资产在哪个块 →
那个块在哪个 chunk。命中的 chunk 会被直接放进最终结果的前几位，**不参与重排** ——
用户指名了哪张表，就不该再让一个相似度模型投票把它翻下去（见 retriever 的 injected）。

**解析为什么先用正则**：跟正文里"见表 3"的提及识别共用
`assets.mention_pattern` 同一套规则（词边界、阿拉伯/罗马数字、制表符不会误吃），
两边不会各漂各的。等前置小 LLM 的结构化输出上来，把它产出的
`{asset_type, number}` 交给 `resolve_targets()` 就行，或者在正则之外叠加。

**三条硬约束**（都是实测出来的，写在 retriever 的注释里）：

1. `label_norm` **只在论文内唯一**：实测 `table:3` 在 4 篇论文里都存在，
   所以 scope 不唯一时一律不定向，退回普通检索；
2. 22/123 个资产没有编号（NULL），7 个资产的块被切分丢掉了 —— 命中不到要能优雅退化；
3. 定位"本体所在块"要用 `json_each(block_ids)`，**不能用 chunk_assets**
   （那里混着交叉引用绑定，而且 relation 全是 primary）。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from app.ingest import assets as assets_mod

ASSET_TYPES = (assets_mod.ASSET_TABLE, assets_mod.ASSET_FIGURE, assets_mod.ASSET_FORMULA)


def parse_targets(query: str) -> list[tuple[str, str]]:
    """从问句里抠出资产编号，按出现先后返回 `[(asset_type, 原始编号)]`，已去重。

    只认"点名"的写法（"表 3"、"Table 3"、"图 2"、"Eq. (5)"）。"第三张表"这种中文序数
    识别不了 —— 那属于前置小 LLM 该干的活（它输出 `{asset_type, number}`）。
    """
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for asset_type in ASSET_TYPES:
        pattern = assets_mod.mention_pattern(asset_type)
        if pattern is None:
            continue
        for match in pattern.finditer(query or ""):
            key = (asset_type, match.group(1))
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
    return found


def resolve_targets(
    conn: sqlite3.Connection,
    scope: Iterable[str],
    targets: Iterable[tuple[str, str]],
) -> list[str]:
    """`(asset_type, number)` → 本体所在 chunk_id（按入参顺序，去重）。

    任何一条不满足（scope 不唯一 / 编号为空 / 库里没有 / 块不在任何 chunk 里）就跳过它，
    全部跳过时返回空列表 = 调用方走普通检索。
    """
    papers = [paper for paper in scope if paper]
    if len(papers) != 1:
        return []  # label_norm 只论文内唯一，多篇/全库下"表 3"是歧义
    paper_id = papers[0]

    found: list[str] = []
    for asset_type, number in targets:
        label_norm = assets_mod.make_label_norm(asset_type, number)
        if not label_norm:
            continue
        row = conn.execute(
            """
            SELECT c.chunk_id
            FROM assets a
            JOIN chunks c ON c.paper_id = a.paper_id
            WHERE a.paper_id = ? AND a.label_norm = ?
              AND EXISTS (SELECT 1 FROM json_each(c.block_ids) je WHERE je.value = a.block_id)
            ORDER BY c.order_index
            LIMIT 1
            """,
            (paper_id, label_norm),
        ).fetchone()
        if row is not None and row["chunk_id"] not in found:
            found.append(row["chunk_id"])
    return found


def resolve(conn: sqlite3.Connection, query: str, scope: Iterable[str]) -> list[str]:
    """`parse_targets` + `resolve_targets` 的合体，给 retriever 一步调用。"""
    return resolve_targets(conn, scope, parse_targets(query))
