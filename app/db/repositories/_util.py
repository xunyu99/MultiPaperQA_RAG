"""repositories 的公共小工具。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def as_batch(rows: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """允许 upsert 既收一条 dict 也收一批 dict。"""
    if isinstance(rows, Mapping):
        return [rows]
    return list(rows)


def upsert_rows(
    conn: sqlite3.Connection,
    table: str,
    rows: Iterable[Mapping[str, Any]],
    key: str | Sequence[str],
    columns: Sequence[str],
) -> int:
    """批量写入，主键冲突则更新指定列，返回写入条数。

    两个刻意的设计：

    1. **不用 `INSERT OR REPLACE`**。REPLACE 是"先删后插"，会顺着
       `ON DELETE CASCADE` 把子表一起删掉 —— 重新索引一篇论文就等于清空它的
       sections / blocks / chunks / assets。这里用 `ON CONFLICT DO UPDATE`。

    2. **写入哪些列由 rows 里实际出现的字段决定，不是由表结构决定**。没给的列
       不参与 INSERT（默认值生效）、也不参与 UPDATE（不被覆盖成 NULL）。
       比如 `papers.created_at` 有 DEFAULT，重新 upsert 时不会把它冲掉。
       代价：同一批 rows 的字段必须一致。
    """
    batch = list(rows)
    if not batch:
        return 0

    keys = (key,) if isinstance(key, str) else tuple(key)
    present = tuple(batch[0].keys())
    for row in batch:
        if tuple(row.keys()) != present:
            raise ValueError(f"{table}: 同一批写入的字段必须完全一致")

    unknown = set(present) - set(columns)
    if unknown:
        raise ValueError(f"{table}: 不认识的字段 {sorted(unknown)}，合法字段 {sorted(columns)}")
    missing = set(keys) - set(present)
    if missing:
        raise ValueError(f"{table}: 缺少主键字段 {sorted(missing)}")

    # 冲突目标必须写全（ON CONFLICT (paper_id) 这种）。省略目标时，SQLite 会把
    # **任何**唯一约束冲突都当成本次冲突来 DO UPDATE —— 比如 blocks 的
    # UNIQUE(paper_id, order_index) 一旦撞上，它会去改那条已存在的块，而不是报错，
    # 静默改坏别人的数据。写全目标后，只有主键冲突才 upsert，其余约束照常报错。
    conflict_target = f"({', '.join(keys)})"
    placeholders = ", ".join("?" for _ in present)
    updates = ", ".join(f"{col} = excluded.{col}" for col in present if col not in keys)
    conflict = (
        f"ON CONFLICT {conflict_target} DO UPDATE SET {updates}"
        if updates
        else f"ON CONFLICT {conflict_target} DO NOTHING"
    )

    sql = f"INSERT INTO {table} ({', '.join(present)}) VALUES ({placeholders}) {conflict}"
    conn.executemany(sql, [tuple(row[col] for col in present) for row in batch])
    return len(batch)


def dumps_json(value: Any) -> str | None:
    """bbox / block_ids 这类结构化字段统一这样存。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def loads_json(value: str | None) -> Any:
    if value is None or value == "":
        return None
    return json.loads(value)
