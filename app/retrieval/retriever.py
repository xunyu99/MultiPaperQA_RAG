"""检索 + 回归事实源：向量库出 id，回 SQLite 把这条证据需要的东西全补齐。

`retrieve()` 返回的**不是裸的向量结果**，而是一个已经补齐的
`RetrievedChunk`：正文、论文、章节、资产本体、溯源坐标都在上面。这样调用方
（证据卡、评测、调试脚本）拿到的对象是自洽的，不用各自再查一遍库。

三个必须做对的地方：

1. **顺序必须跟向量库走。** 向量库返回 `[(chunk_id, distance), ...]`，而
   `WHERE chunk_id IN (...)` 的返回顺序按主键来 —— **跟相似度排名毫无关系**。
   所以先做成 `{chunk_id: row}` 字典，再按向量库给的顺序取。忘了这步，排名会
   静默错乱：不报错、看着也像那么回事，但 top-1 已经不是最像的那条了。
2. **孤儿向量直接丢弃。** 向量库里有、SQLite 里没有（论文删过、索引没同步、
   中途崩过），绝不能传到证据卡里。
3. **资产占位符在这里展开。** chunk 里只有 `[TABLE_REF asset_id=...]`，
   本体在 `assets` 表；这一步把它换成 caption + 表头 + HTML，并把用到的资产
   记在 `assets` 上，供证据卡和前端溯源使用。
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.db.repositories import assets as assets_repo
from app.db.repositories import blocks as blocks_repo
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import papers as papers_repo
from app.db.repositories import sections as sections_repo
from app.db.repositories._util import loads_json
from app.ingest.assets import display_label
from app.retrieval import asset_target, keyword_index
from app.providers.reranker import Reranker
from app.retrieval.vector_index import ChunkVectorIndex
from app.retrieval.scope import resolve_scope

# content 里的资产占位符：`[TABLE_REF asset_id=pe-clip:table0001]`
_PLACEHOLDER = re.compile(r"\[(?P<kind>TABLE|FIGURE|FORMULA)_REF\s+asset_id=(?P<asset_id>[^\]]+)\]")

# 开去重时每篇的候选深度下限。取宽才有的可去重；15 条对本地 Chroma 是毫秒级，
# 但每条都要回 SQLite 补齐事实源，所以也别无限放大。
_DEFAULT_POOL = 15

# 占位符里的 kind → 类型词。三级兜底的最后一级（见 _asset_phrase）。
_KIND_WORD = {"TABLE": "Table", "FIGURE": "Figure", "FORMULA": "Formula"}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    similarity: float
    rank: int  # 0 = 最像
    content: str  # 占位符已展开，可以直接给模型
    raw_content: str  # 原始内容（占位符还在）
    paper: dict[str, Any]
    chunk_order: int
    page_start: int | None
    page_end: int | None
    chunk_type: str
    token_count: int
    # 排序分：双路融合（RRF）之后的名次分，**排序只看它**。`similarity` 是向量通道
    # 的余弦相似度，只用于展示 —— 关键词独有命中的 chunk 没有相似度，是 0.0。
    # 两个分开的理由：RRF 分（约 0.01）和余弦（0~1）不同量纲，混在一个字段里
    # 迟早会有人拿它去比阈值。
    rank_score: float = 0.0
    section_id: str | None = None
    section_title: str | None = None
    section_type: str | None = None
    section_order: int = 0
    assets: list[dict[str, Any]] = field(default_factory=list)
    # 前端溯源高亮：每个块的 page_idx + bbox（见 PLAN §2.4）
    trace: list[dict[str, Any]] = field(default_factory=list)


def retrieve(
    conn: sqlite3.Connection,
    query: str,
    k: int | None = None,
    index: ChunkVectorIndex | None = None,
    settings: Settings | None = None,
    paper_id: str | None = None,
    paper_ids: list[str] | None = None,
    auto_scope: bool = True,
    stats: dict[str, Any] | None = None,
    vector: Sequence[float] | None = None,
    section_cap: int | None = None,
    reranker: Any | None = None,
    injected: Sequence[str] | None = None,
    auto_asset: bool = True,
) -> list[RetrievedChunk]:
    """检索 top-k 并补齐事实源。

    **论文范围**（多论文是目标形态，见 scope.py）：

    - `paper_id` / `paper_ids` 显式给了 → 就用它；
    - 没给且 `auto_scope=True` → 从问题里字面解析（用户提到论文名就自动缩范围）；
    - 解析不出来 → **空列表 = 全库**。宁可搜宽，不要猜错论文。

    范围确定后在**向量库层**过滤（不是检索完再筛）—— 每篇论文一次查询。
    多篇时按**配额**取（`k / N` 向上取整），否则一篇可能占满 top-k，
    跨论文对比的问题就只剩一篇的证据了。合并后再按距离统一排序截断。

    `stats` 是可选出参：传个空 dict 进来，会填上 `timing` 和 `dropped`
    （返回类型保持"干净的列表"，调用方按需取这两个诊断信息）。

    `vector` 给了就跳过 embedding（同一个 query 要跑多套参数做 A/B 时，
    只算一次向量，省一次网络往返）。

    `section_cap > 0` 时开启**同章节去重**：同一篇论文的同一个章节最多占
    `section_cap` 个名额。章节是展示层概念（Chroma 元数据里也没有 section_id），
    所以只在这一步生效，不动索引 —— 见铁律 2。

    **定点注入**（`injected` / `auto_asset`）：用户点名"表 3"这类编号时，命中的 chunk
    直接占住前几名，**不参与重排**，其余名额照常由双路检索填 —— 用户指名了哪张表，
    就不该再让相似度模型投票把它翻下去。`injected` 显式给了就用它（前置小 LLM 的
    结构化输出将来从这儿进来）；不给且 `auto_asset=True` 时，用正则从问句里抠编号
    （`asset_target.resolve`），抠不到就静默退回普通检索。
    """
    settings = settings or get_settings()
    index = index or ChunkVectorIndex(settings)
    k = k or settings.top_k
    cap = settings.section_cap if section_cap is None else section_cap
    scope = _resolve_scope(conn, query, paper_id, paper_ids, auto_scope)
    quota, total = _quota_and_total(settings, k, scope)
    pinned = _pinned_chunks(conn, query, scope, injected, auto_asset)

    # query embedding 只算一次（网络往返），后面按论文做的都是本地查询
    started = time.perf_counter()
    if vector is None:
        vector = index.embedder.embed_query(query)
    embedded = time.perf_counter()

    # 双路召回：向量（语义）+ 关键词（字面）。每路每篇取 `depth` 条候选。
    depth = _channel_depth(settings, scope)
    vector_hits = _search_in_scope(index, vector, depth, scope, cap, settings)
    keyword_hits = _search_keywords(conn, query, depth, scope) if settings.keyword_enabled else []
    merged, paper_of = _merge_channels(vector_hits, keyword_hits)
    candidate_count = len(merged)
    finished = time.perf_counter()

    # 注入的 chunk 也要取行（它们不在双路候选里）
    wanted = _unique([*pinned, *[item.chunk_id for item in merged]])
    rows = chunks_repo.get_many(conn, wanted)
    by_id = {row["chunk_id"]: dict(row) for row in rows}
    assets_by_id = _assets_in_rows(conn, by_id.values())

    # 重排：用 `chunks.content` 打分（还没 enrich，省掉占位符展开那一坨）。
    # **query 就是改写后的那个** —— retrieve() 收到的就是 prepare_turn 给的 standalone。
    # 注入的**不参与重排**：它们的名次是用户指定的，不由模型决定。
    merged = [item for item in merged if item.chunk_id not in set(pinned)]
    merged, rerank_info = _rerank(query, merged, by_id, settings, reranker, assets_by_id)
    if pinned:
        merged = [_Candidate(chunk_id, 1.0, 0.0) for chunk_id in pinned if chunk_id in by_id] + merged
        rerank_info["pinned"] = pinned
    if len(scope) > 1:
        # 按篇保额放在重排**之后**：否则"每篇留几条"由合并顺序决定，轮不到重排去挑
        # 每篇里真正相关的那些（对比题最怕这个）。
        merged = _trim_per_paper(merged, paper_of, scope, quota)
    reranked = time.perf_counter()

    timing = {
        "embed_ms": (embedded - started) * 1000,
        "search_ms": (finished - embedded) * 1000,
        "rerank_ms": (reranked - finished) * 1000,
        "total_ms": (reranked - started) * 1000,
    }

    wide: list[RetrievedChunk] = []
    dropped: list[str] = []
    for rank, item in enumerate(merged):
        row = by_id.get(item.chunk_id)
        if row is None:
            dropped.append(item.chunk_id)  # 孤儿向量/孤儿 FTS 行，见模块开头第 2 条
            continue
        wide.append(_enrich(conn, row, item.similarity, rank, item.score))

    items, duplicates = _select(wide, total, scope, cap, quota)

    if stats is not None:
        stats["timing"] = timing
        stats["dropped"] = dropped
        stats["query"] = query
        stats["scope"] = scope
        stats["section_cap"] = cap
        stats["duplicates"] = [item.chunk_id for item in duplicates]
        stats["quota"] = quota
        stats["total"] = total
        stats["channels"] = {
            "vector": len(vector_hits),
            "keyword": len(keyword_hits),
            "keyword_only": sum(1 for hit in keyword_hits if not _in(vector_hits, hit.chunk_id)),
            # 并集大小（去重后、按篇保额**之前**）。别拿 len(merged)：那已经是保额后的
            "candidates": candidate_count,
        }
        stats["rerank"] = rerank_info
    return items


@dataclass(frozen=True)
class _Candidate:
    """合并之后的一条候选。

    `score` 是**排序依据**：现在只是"合并名次分"（1/位次），rerank 上线后换成
    cross-encoder 的重排分。`similarity` 只用于展示（关键词独有命中的是 0.0）。
    """

    chunk_id: str
    score: float
    similarity: float


def _in(hits: Sequence[Any], chunk_id: str) -> bool:
    return any(hit.chunk_id == chunk_id for hit in hits)


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _pinned_chunks(
    conn: sqlite3.Connection,
    query: str,
    scope: list[str],
    injected: Sequence[str] | None,
    auto_asset: bool,
) -> list[str]:
    """要定点注入的 chunk。显式给了 `injected` 就用它，否则用正则从问句里抠（可关）。"""
    if injected is not None:
        return _unique(injected)
    if not auto_asset:
        return []
    return asset_target.resolve(conn, query, scope)


def _channel_depth(settings: Settings, scope: list[str]) -> int:
    """每路每篇取几条候选。多篇给得少一点（见 config 的注释）。"""
    if len(scope) > 1:
        return max(1, settings.channel_k_multi)
    return max(1, settings.channel_k)


def _search_keywords(
    conn: sqlite3.Connection,
    query: str,
    depth: int,
    scope: list[str],
) -> list[Any]:
    """关键词通道，形状跟向量那边对齐：每篇各取 `depth` 条。

    两路都按"每篇 depth"取，而不是一路深一路浅 —— 不然融合时分额会歪。
    """
    if not scope:
        return keyword_index.search(conn, query, k=depth)
    if len(scope) == 1:
        return keyword_index.search(conn, query, k=depth, paper_ids=scope)
    merged: list[Any] = []
    for scoped_paper_id in scope:
        merged.extend(keyword_index.search(conn, query, k=depth, paper_ids=[scoped_paper_id]))
    return merged


def _merge_channels(
    vector_hits: Sequence[Any],
    keyword_hits: Sequence[Any],
) -> tuple[list[_Candidate], dict[str, str | None]]:
    """两路候选**合并去重**（并集：向量在前、关键词独有在后）。

    **这里刻意不做 RRF**。向量分（余弦）和关键词分（BM25）确实不同量纲，但排序
    不归合并阶段管 —— 最终顺序交给 rerank，RRF 排出来的结果会被重排完全覆盖。
    合并只负责两件事：不漏候选、不重复。所以 `score` 现在只是**占位名次分**，
    rerank 成功时会被换成真正的重排分。

    返回 (候选, chunk_id → paper_id)，后者给多篇按篇保额用。
    """
    ordered: list[str] = []
    # `similarity` **只认向量通道**：关键词通道也有个叫 similarity 的字段（= -bm25），
    # 量纲完全不同（实测能到 11.4），混进展示层会变成"相似度 11.407"这种鬼话。
    # 关键词独有的 chunk 没有向量相似度，就是 0.0。
    similarity: dict[str, float] = {hit.chunk_id: float(hit.similarity) for hit in vector_hits}
    paper_of: dict[str, str | None] = {}
    seen: set[str] = set()
    for hits in (vector_hits, keyword_hits):
        for hit in hits:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            ordered.append(hit.chunk_id)
            paper_of[hit.chunk_id] = getattr(hit, "paper_id", None) or (
                getattr(hit, "metadata", None) or {}
            ).get("paper_id")

    ranked = [
        _Candidate(chunk_id, 1.0 / (position + 1), similarity.get(chunk_id, 0.0))
        for position, chunk_id in enumerate(ordered)
    ]
    return ranked, paper_of


def _rerank(
    query: str,
    candidates: list[_Candidate],
    by_id: dict[str, dict[str, Any]],
    settings: Settings,
    reranker: Any | None,
    assets_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[_Candidate], dict[str, Any]]:
    """用交叉编码器重排候选。**任何失败都降级成原顺序**，不能拖垮整个请求。

    降级后的顺序就是合并顺序（向量在前）—— 这也正是 RRF 该在的位置：以后要做
    降级排序再加，别在正常路径上先排一遍再被重排覆盖。

    打分用的是 `content`，**占位符要换成资产的 caption** 再送：重排是交叉编码器，
    它看不到向量索引里那份 `index_text`，如果占位符被直接剔成空格，它就完全不知道
    "这里有一张表 3"。caption 为空时退到 `label_norm` 的显示名（"Table 3"），
    再没有就退到类型词（"Table"）。
    """
    if not candidates:
        return candidates, {"enabled": False}
    if not settings.rerank_enabled:
        return candidates, {"enabled": False}

    documents = []
    kept: list[_Candidate] = []
    for item in candidates:
        row = by_id.get(item.chunk_id)
        if row is None:
            continue
        kept.append(item)
        documents.append(
            _PLACEHOLDER.sub(
                lambda match: _asset_phrase(
                    (assets_by_id or {}).get(match.group("asset_id").strip()), match.group("kind")
                ),
                row.get("content") or "",
            )
        )
    if not kept:
        return candidates, {"enabled": True, "ok": True, "resized": 0}

    try:
        model = reranker or Reranker(settings)
        scores = model.rerank(query, documents)
    except Exception as exc:  # noqa: BLE001 - 重排挂了也得把答案给出去
        return candidates, {
            "enabled": True,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    rescored = [
        _Candidate(item.chunk_id, float(scores[index]), item.similarity)
        for index, item in enumerate(kept)
    ]
    # 同分（比如服务端只回了部分 doc、其余补 0）时按原顺序兜底，保证可复现
    order = {item.chunk_id: index for index, item in enumerate(kept)}
    rescored.sort(key=lambda item: (-item.score, order[item.chunk_id]))
    return rescored, {"enabled": True, "ok": True, "docs": len(kept)}


def _assets_in_rows(
    conn: sqlite3.Connection, rows: Any
) -> dict[str, dict[str, Any]]:
    """候选 chunk 里出现过的资产，一次 IN 查询批量取回。

    给重排替换占位符用 —— 重排看的是 `content`（只有占位符），表的 caption / 编号
    在 `assets` 表里，不查一次它拿不到。
    """
    ids: set[str] = set()
    for row in rows:
        for match in _PLACEHOLDER.finditer(row.get("content") or ""):
            ids.add(match.group("asset_id").strip())
    return {row["asset_id"]: dict(row) for row in assets_repo.get_many(conn, sorted(ids))}


def _asset_phrase(asset: dict[str, Any] | None, kind: str) -> str:
    """占位符在重排文本里换成什么：caption → label_norm 显示名 → 类型词。

    三级兜底的理由：123 个资产里 62 个 caption 是空的，其中 22 个连编号都没有；
    但"这里有一张表/一张图"这个信息本身对重排也有用，别退化成空白。
    """
    if asset:
        caption = (asset.get("caption") or "").strip()
        if caption:
            return f" {caption} "
        label = display_label(asset.get("label_norm"))
        if label:
            return f" {label} "
    return f" {_KIND_WORD.get(kind, 'Asset')} "


def _trim_per_paper(
    ranked: list[_Candidate],
    paper_of: dict[str, str | None],
    scope: list[str],
    quota: int,
) -> list[_Candidate]:
    """多篇时**每篇先留 quota 条**再全局排。

    不这么做的话：候选池是 篇数 x 深度（比最终条数大），融合后按分数全局截断，
    一篇可能把名额吃光 —— 正是 `per_paper_k` 那条注释里踩过的坑。
    """
    grouped: dict[str, list[_Candidate]] = {}
    for item in ranked:
        grouped.setdefault(paper_of.get(item.chunk_id) or "", []).append(item)
    kept: list[_Candidate] = []
    for scoped_paper_id in scope:
        kept.extend(grouped.get(scoped_paper_id, [])[:quota])
    kept.sort(key=lambda item: -item.score)
    return kept


def _quota_and_total(settings: Settings, k: int, scope: list[str]) -> tuple[int, int]:
    """算出"每篇几条"（quota）和"一共几条"（total）。

    - **单篇 / 全库**：quota = k（`TOP_K`，**默认 5**）。全库没有"每篇"的概念，
      一条查询出 k 条就够。
    - **多篇**：quota = `PER_PAPER_K`（**默认 3**），total = N * quota ——
      所以勾 3 篇就返回 **9 条**证据。对比题要的是"每篇都有代表"，均分固定名额比按相似度
      全局截断可靠 —— 后者可以让一篇把名额吃光（Step 7 实测过）。
      注意 `PER_PAPER_K` 里的 3 是**每篇**，不是总数（`TOP_K` 试过 3，Recall@3 掉到 60%，
      所以退回 5，见 config 的注释）。
    """
    if len(scope) <= 1:
        return k, k
    return settings.per_paper_k, settings.per_paper_k * len(scope)


def _resolve_scope(
    conn: sqlite3.Connection,
    query: str,
    paper_id: str | None,
    paper_ids: list[str] | None,
    auto_scope: bool,
) -> list[str]:
    if paper_ids:
        return list(paper_ids)
    if paper_id:
        return [paper_id]
    if not auto_scope:
        return []
    return resolve_scope(query, papers_repo.list_all(conn))


def _search_in_scope(
    index: ChunkVectorIndex,
    vector: list[float],
    quota: int,
    scope: list[str],
    cap: int = 0,
    settings: Settings | None = None,
) -> list[Any]:
    """按范围取**宽候选**（不去重、不截断到 k，交给 `_select`）。

    `quota` 是**每篇几条**（单篇/全库时它就是 k，多篇时是 `PER_PAPER_K`）。
    `cap <= 0` 时每篇就取 quota 条。
    """
    if not scope:
        return index.search_by_vector(vector, k=_pool_depth(quota, cap, settings))
    if len(scope) == 1:
        return index.search_by_vector(
            vector, k=_pool_depth(quota, cap, settings), paper_id=scope[0]
        )

    depth = _pool_depth(quota, cap, settings)
    merged: list[Any] = []
    for scoped_paper_id in scope:
        merged.extend(index.search_by_vector(vector, k=depth, paper_id=scoped_paper_id))
    # 合并后统一按距离排序（不是按论文顺序拼）
    merged.sort(key=lambda hit: hit.distance)
    return merged


def _pool_depth(per_paper: int, cap: int, settings: Settings | None) -> int:
    """每篇取多深。开去重时必须取宽一点，否则去掉的空位补不回来。"""
    if cap <= 0:
        return per_paper
    candidate_k = settings.candidate_k if settings is not None else _DEFAULT_POOL
    return max(per_paper * 3, min(candidate_k, _DEFAULT_POOL))


def _select(
    wide: list[RetrievedChunk],
    total: int,
    scope: list[str],
    cap: int,
    quota: int,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """从宽候选里挑最终结果，返回 (选中的, 因同章节被挤掉的)。

    排序一律按 `rank_score`（双路融合分），**不是** `similarity` —— 后者只是向量通道
    的余弦值，关键词独有命中的 chunk 根本没有它。

    - `cap <= 0`：直接按融合顺序取前 `total` 条；
    - `cap > 0`：每篇内先压同章节重复，再按 `quota` 取，最后按融合分合并截断到 `total`。
      如果压完不够，**用被挤掉的按相似度补回来** —— 宁可重复也不能少给证据。
    """
    if cap <= 0:
        return wide[:total], []

    grouped: dict[str, list[RetrievedChunk]] = {}
    for item in wide:
        grouped.setdefault(item.paper.get("paper_id") or "", []).append(item)

    if not scope:
        kept = _cap_by_section(wide, cap)
    else:
        kept = []
        for scoped_paper_id in scope:
            kept.extend(_cap_by_section(grouped.get(scoped_paper_id, []), cap)[:quota])
    kept.sort(key=lambda item: item.rank_score, reverse=True)
    kept = kept[:total]

    kept_ids = {item.chunk_id for item in kept}
    dropped = [item for item in wide if item.chunk_id not in kept_ids]
    if len(kept) < min(total, len(wide)):  # 章节不够多样，补回重复
        for item in dropped:
            kept.append(item)
            if len(kept) >= total:
                break
        kept_ids = {item.chunk_id for item in kept}
        dropped = [item for item in wide if item.chunk_id not in kept_ids]
    return kept, dropped


def _cap_by_section(items: list[RetrievedChunk], cap: int) -> list[RetrievedChunk]:
    """按融合分顺序走一遍，同一章节最多留 cap 条（入参必须已按融合分降序）。"""
    counts: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for item in items:
        key = item.section_id or item.section_title or f"(no-section:{item.paper.get('paper_id')})"
        if counts.get(key, 0) >= cap:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(item)
    return kept


# ----------------------------------------------------------------------
# 单条补齐
# ----------------------------------------------------------------------
def _enrich(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    similarity: float,
    rank: int,
    rank_score: float = 0.0,
) -> RetrievedChunk:
    paper = papers_repo.get(conn, row["paper_id"])

    block_ids = loads_json(row["block_ids"]) or []
    blocks = [block for block in (blocks_repo.get(conn, bid) for bid in block_ids) if block]
    section = _section_of(conn, blocks)
    content, used_assets = expand_assets(conn, row["content"] or "")

    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        similarity=similarity,
        rank_score=rank_score,
        rank=rank,
        content=content,
        raw_content=row["content"] or "",
        paper=dict(paper) if paper else {},
        chunk_order=row["order_index"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        chunk_type=row["chunk_type"],
        token_count=row["token_count"] or 0,
        section_id=(section or {}).get("section_id"),
        section_title=(section or {}).get("title"),
        section_type=(section or {}).get("section_type"),
        section_order=(section or {}).get("order_index") or 0,
        assets=used_assets,
        trace=[
            {
                "block_id": block["block_id"],
                "page_idx": block["page_idx"],
                "bbox": loads_json(block["bbox"]),
            }
            for block in blocks
        ],
    )


def _section_of(conn: sqlite3.Connection, blocks: list[Any]) -> dict[str, Any] | None:
    """chunk 归到**第一个块**所属的章节。

    `chunks` 表没有 `section_id` 列（见 PLAN §2.1），所以从 `block_ids` 反查。
    跨章节的 chunk（补足合并产生的）只能归到第一个，代价是尾部文字的标签不精确。
    """
    for block in blocks:
        if block["section_id"]:
            section = sections_repo.get(conn, block["section_id"])
            if section is not None:
                return dict(section)
    return None


def expand_assets(
    conn: sqlite3.Connection,
    content: str,
) -> tuple[str, list[dict[str, Any]]]:
    """把 `[TABLE_REF asset_id=...]` 换成资产本体，返回 (展开后的文本, 用到的资产)。

    这就是"chunk 里不存本体、用的时候拼"的那一步（PLAN §2.1）。
    """
    used: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        kind = match.group("kind")
        asset_id = match.group("asset_id").strip()
        asset = assets_repo.get(conn, asset_id)
        if asset is None:
            return f"[{kind}_REF 缺失：{asset_id}]"
        row = dict(asset)
        used.append(row)
        return render_asset(row)

    return _PLACEHOLDER.sub(replace, content), used


def render_asset(asset: dict[str, Any]) -> str:
    """资产本体怎么给模型看。"""
    asset_type = asset["asset_type"]
    caption = (asset.get("caption") or "").strip()

    if asset_type == "table":
        body = (asset.get("raw_content") or "").strip()
        return "\n".join(part for part in (caption, body) if part)

    if asset_type == "formula":
        latex = (asset.get("raw_content") or "").strip()
        return "\n".join(part for part in (caption, latex) if part)

    # 图片：本体是图像文件，文本链路里给不了。v1 只有 caption，
    # 真正把图塞进多模态消息是 BACKLOG B11 的事。
    return caption or "（无图注的图）"
