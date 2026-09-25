"""blocks + sections + assets → chunks。

切分规则（每一条都有实测依据，见 PLAN Step 3）：

1. **单位是块**：公式、表格、图永不切（它们进 chunk 的是一个引用块，见 assets.py）。
2. **正文段落可以切**：超过 `CHUNK_MAX_TOKENS` 的纯文本块按句子边界切开。
   实测最长正文段 2926 字符（约 800 token），硬塞进一个 chunk 会让检索质量变差。
3. **边界限制在单篇论文内**，且优先落在章节边界。
4. **两遍聚合**：先按章节累加到 `CHUNK_TARGET_TOKENS`，再把低于 `CHUNK_MIN_TOKENS`
   的 chunk 并入相邻 chunk（同论文、合并后 ≤ max）。这是"不零碎"的关键 ——
   只做第一遍会有 15/178 个 chunk 小于 120 token，补足后只剩 1 个。
5. **chunk 之间不重叠**：边界落在完整块之间，本来就是自然断点，重叠只会让同一块
   出现在两个 chunk 的检索结果里、白占证据位。只有被切开的长正文块内部带 overlap。

   overlap **只在最终落到 chunk 边界时才拼出来**：切分时记在 `Unit.overlap_prefix` 里，
   渲染 chunk 时只有"它是这个 chunk 的第一个 unit"才拼到前面。否则两片被后面的补足合并
   又回到同一个 chunk 时，同一句话会在 chunk 内部出现两次（这是实测踩到的 bug）。

6. **跨章节合并要求两个章节"同父或互为祖先后代"**：`2.1 + 2.2`（同父）可以，
   `4 Experiments + 4.1`（父子相邻）可以，`2.3 + 3`（跨父）不行 —— 那样 `section_id`
   和章节路径就没法定位了。元信息章节（Abstract / Keywords / CCS Concepts /
   Acknowledgements）彼此都是顶层兄弟，同父，所以照样能合并。

`content` 和 `index_text` 是两套表示（铁律 2）：

    index_text 检索用（向量 + BM25 共用）：`[论文标题 > 章节路径]` + 正文 + 资产的检索文本
    content    生成用：正文 + 资产占位符，**不放标题**

标题**两边都放**，但理由不同：

- `index_text` 里是**检索需要** —— BM25 要能匹配"3.1 Pipeline"这种字面查询；
- `content` 里是**阅读需要** —— 模型要看懂这段属于哪一节。

多占的 token 很少（章节标题一行），换来的是证据卡组装时不用再把标题拼回去。

`chunks` 表**没有 section_id 列**：它能从 `chunk.block_ids → blocks.section_id` 推出来，
而跨章节的 chunk 本来也只能记一个（有歧义）。要按章节过滤时按 block_ids 反查即可
（90 个 chunk，Python 里过滤零成本）。**只留检索时要用来过滤的 `paper_id`** ——
它要进 Chroma metadata，每路召回都要用，不做 join。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.ingest import assets as assets_mod
from app.ingest.citation import author_block_ids
from app.ingest.converter import (
    BLOCK_CAPTION,
    BLOCK_EQUATION,
    BLOCK_IMAGE,
    BLOCK_NOISE,
    BLOCK_REFERENCE,
    BLOCK_TABLE,
    BLOCK_TITLE,
)
from app.ingest.tokenizer import chars_for_tokens, estimate_tokens
from app.ingest.versioning import index_version

# 句子边界：
#   - 中文标点（。！？；）后面**不需要空格**，"第一句。第二句。" 也要能切开；
#   - 英文标点必须跟空白，否则 "3.14"、"e.g." 会被当成句末切坏。
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])\s*|(?<=[.!?;])\s+")
# 正文里提到资产："as shown in Table 3" / "图 3" / "Equation (7)" / "TABLE VIII"。
# 正则来自 assets.mention_pattern —— 跟 caption 侧**共用同一个解析器**，
# 否则"入库怎么写名字"和"正文怎么认名字"会各漂各的（这就是之前的 bug 来源）。
_ASSET_MENTION = {
    asset_type: pattern
    for asset_type in (assets_mod.ASSET_TABLE, assets_mod.ASSET_FIGURE, assets_mod.ASSET_FORMULA)
    if (pattern := assets_mod.mention_pattern(asset_type)) is not None
}

# content 里的资产占位符。匹配前必须把它挖掉：占位符里的 "table0003" 自己也会被
# `_ASSET_MENTION` 当成"正文提到了表 3"。
_PLACEHOLDER = re.compile(r"\[[A-Z]+_REF[^\]]*\]")

# 这些块不进 chunk：噪声是版式垃圾，reference 是参考文献，title 的信息在章节路径里，
# caption 的内容已经进了对应资产的 caption（不重复放）。
DROP_BLOCK_TYPES = (BLOCK_NOISE, BLOCK_REFERENCE, BLOCK_TITLE, BLOCK_CAPTION)


@dataclass
class Unit:
    """切分的最小单位。一个块通常对应一个 unit，长正文块会被拆成多个。"""

    block_id: str
    order_index: int
    kind: str  # text / table / figure / formula
    content_text: str
    index_text: str
    section_id: str | None
    section_path: str
    page_start: int | None
    page_end: int | None
    asset_id: str | None = None
    # 上一片的尾句。只有它成为 chunk 的第一个 unit 时才拼到前面（见模块开头第 5 条）
    overlap_prefix: str = ""

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content_text)

    @property
    def head_text(self) -> str:
        """作为 chunk 第一块时的完整文本（含 overlap 前缀）。"""
        if self.overlap_prefix:
            return f"{self.overlap_prefix} {self.content_text}".strip()
        return self.content_text


@dataclass
class ChunkDraft:
    units: list[Unit] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(unit.tokens for unit in self.units)

    @property
    def section_id(self) -> str | None:
        return self.units[0].section_id if self.units else None

    @property
    def kinds(self) -> set[str]:
        return {unit.kind for unit in self.units}


@dataclass
class ChunkBuild:
    """切分结果：chunks 行 + 每篇论文的统计。"""

    chunks: list[dict[str, Any]]
    assets: list[dict[str, Any]]
    stats: dict[str, Any]


def split_paper(
    paper_id: str,
    paper_title: str,
    blocks: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    settings: Settings | None = None,
) -> ChunkBuild:
    """一篇论文的 blocks → chunks。`blocks` 需要已经填好 section_id。"""
    settings = settings or get_settings()

    section_by_id = {section["section_id"]: dict(section) for section in sections}
    # 作者行不进索引（PLAN BACKLOG B20）：占 15%-34% 的 token，对问答零价值。
    # 注意是**按块**排除，不是按章节 —— 摘要正文常常也落在 preamble 里。
    author_ids = author_block_ids(blocks)
    units, dropped = _build_units(
        paper_id, paper_title, blocks, section_by_id, assets, settings, author_ids
    )
    drafts = _group_by_section(units)
    drafts = [draft for group in drafts for draft in _split_section(group, settings)]
    drafts = _merge_small(drafts, settings, section_by_id)

    chunks = [_to_row(paper_id, draft, index) for index, draft in enumerate(drafts, 1)]
    _attach_cross_references(chunks, assets)

    tokens = [chunk["token_count"] for chunk in chunks]
    stats = {
        "paper_id": paper_id,
        "chunks": len(chunks),
        "tokens_total": sum(tokens),
        "tokens_min": min(tokens) if tokens else 0,
        "tokens_max": max(tokens) if tokens else 0,
        "dropped_blocks": dropped,
        "index_version": index_version(settings),
    }
    return ChunkBuild(chunks=chunks, assets=assets, stats=stats)


# ----------------------------------------------------------------------
# 1. blocks → units
# ----------------------------------------------------------------------
def _build_units(
    paper_id: str,
    paper_title: str,
    blocks: list[dict[str, Any]],
    section_by_id: dict[str, dict[str, Any]],
    assets: list[dict[str, Any]],
    settings: Settings,
    author_ids: set[str],
) -> tuple[list[Unit], int]:
    reference_sections = {
        sid for sid, section in section_by_id.items() if section.get("section_type") == "reference"
    }
    asset_by_block = {asset["block_id"]: asset for asset in assets}

    units: list[Unit] = []
    dropped = 0

    for block in sorted(blocks, key=lambda b: b.get("order_index") or 0):
        if block.get("block_type") in DROP_BLOCK_TYPES:
            dropped += 1
            continue
        if block["block_id"] in author_ids:
            dropped += 1
            continue
        if block.get("section_id") in reference_sections:
            dropped += 1
            continue

        section_path = _section_path(paper_title, block.get("section_id"), section_by_id)
        asset = asset_by_block.get(block["block_id"])

        if asset is not None:
            unit = Unit(
                block_id=block["block_id"],
                order_index=block.get("order_index") or 0,
                kind=asset["asset_type"],
                content_text=assets_mod.ref_text(asset),
                index_text=assets_mod.index_text_of(asset) or "",
                section_id=block.get("section_id"),
                section_path=section_path,
                page_start=block.get("page_idx"),
                page_end=block.get("page_idx"),
                asset_id=asset["asset_id"],
            )
        else:
            text = _block_text(block)
            if not text:
                dropped += 1
                continue
            unit = Unit(
                block_id=block["block_id"],
                order_index=block.get("order_index") or 0,
                kind="text",
                content_text=text,
                index_text=text,
                section_id=block.get("section_id"),
                section_path=section_path,
                page_start=block.get("page_idx"),
                page_end=block.get("page_idx"),
            )

        units.extend(_split_long_unit(unit, settings))

    return units, dropped


def _block_text(block: dict[str, Any]) -> str:
    return (block.get("text") or block.get("latex") or block.get("html") or "").strip()


def _section_path(
    paper_title: str,
    section_id: str | None,
    section_by_id: dict[str, dict[str, Any]],
) -> str:
    """从当前章节沿 parent_id 往上走，拼成 `论文标题 > 3 Method > 3.1 ...`。"""
    chain: list[str] = []
    current = section_id
    guard = 0
    while current and guard < 16:
        section = section_by_id.get(current)
        if section is None:
            break
        chain.append(str(section.get("title") or "").strip())
        current = section.get("parent_id")
        guard += 1
    chain.reverse()
    parts = [paper_title.strip(), *[part for part in chain if part]]
    return " > ".join(part for part in parts if part)


# ----------------------------------------------------------------------
# 2. 长正文块按句子切
# ----------------------------------------------------------------------
def _split_long_unit(unit: Unit, settings: Settings) -> list[Unit]:
    """超长正文按句子切。

    **预算要扣掉章节标题**：`token_count` 算的是最终 content，里面已经含
    `## 章节标题` 那一行。按纯正文算会差几个 token，最后正好超出 max —— 实测踩过。

    **预算还要扣掉 overlap**：`overlap_prefix` 会在这一片成为 chunk 首块时拼上来，
    不预留同样会超。
    """
    budget = settings.chunk_max_tokens - _heading_tokens(unit)
    if unit.kind != "text" or unit.tokens <= budget:
        return [unit]

    sentences = [s for s in _SENTENCE_SPLIT.split(unit.content_text) if s.strip()]
    if len(sentences) <= 1:
        return _split_unit_by_chars(unit, settings)

    pieces: list[Unit] = []
    buffer: list[str] = []
    carry = ""
    for sentence in sentences:
        candidate = " ".join([*buffer, sentence])
        if buffer and estimate_tokens(f"{carry} {candidate}".strip()) > budget:
            pieces.append(_clone(unit, " ".join(buffer), overlap_prefix=carry))
            carry = _overlap_text(buffer, settings.text_overlap_chars)
            buffer = []
        buffer.append(sentence)
    if buffer:
        pieces.append(_clone(unit, " ".join(buffer), overlap_prefix=carry))
    return pieces


def _split_unit_by_chars(unit: Unit, settings: Settings) -> list[Unit]:
    """连句子边界都找不到的极端长文本（整段没有标点）：按字符切。

    字符预算不能写死成"token × 3.6" —— 那是英文的密度，中文 1 个字就是 1 个 token，
    按英文换算会把一段 320 字的中文当成 115 token（实际 266），于是根本不切。
    这里用**这段文本自己的字符/token 比**换算，中英混排都自适应。
    """
    text = unit.content_text
    ratio = len(text) / max(1, estimate_tokens(text))
    budget = max(50, int((settings.chunk_max_tokens - _heading_tokens(unit)) * ratio))
    step = max(1, budget - settings.text_overlap_chars)
    pieces: list[Unit] = []
    start = 0
    carry = ""
    while start < len(text):
        piece = text[start : start + budget].strip()
        if piece:
            pieces.append(_clone(unit, piece, overlap_prefix=carry))
            carry = piece[-settings.text_overlap_chars :].strip() if settings.text_overlap_chars else ""
        if start + budget >= len(text):
            break
        start += step
    return pieces or [unit]


def _overlap_text(sentences: list[str], overlap_chars: int) -> str:
    """取末尾几句话，凑够 overlap_chars 字符，作为下一片的 overlap 前缀。"""
    if overlap_chars <= 0:
        return ""
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        tail.insert(0, sentence)
        total += len(sentence)
        if total >= overlap_chars:
            break
    return " ".join(tail)


def _clone(unit: Unit, text: str, overlap_prefix: str = "") -> Unit:
    return Unit(
        block_id=unit.block_id,
        order_index=unit.order_index,
        kind=unit.kind,
        content_text=text,
        index_text=text,
        section_id=unit.section_id,
        section_path=unit.section_path,
        page_start=unit.page_start,
        page_end=unit.page_end,
        asset_id=unit.asset_id,
        overlap_prefix=overlap_prefix,
    )


def _heading_tokens(unit: Unit) -> int:
    """这个 unit 在 chunk 里会占掉的章节标题开销（渲染成 `## 标题`）。"""
    heading = unit.section_path.split(" > ")[-1] if unit.section_path else ""
    return estimate_tokens(f"## {heading}") if heading else 0


# ----------------------------------------------------------------------
# 3. 章节内聚合
# ----------------------------------------------------------------------
def _group_by_section(units: list[Unit]) -> list[list[Unit]]:
    """按**连续的同一章节**分组。跨章节的合并留给第 4 步的补足。"""
    groups: list[list[Unit]] = []
    for unit in units:
        if groups and groups[-1][-1].section_id == unit.section_id:
            groups[-1].append(unit)
        else:
            groups.append([unit])
    return groups


def _split_section(group: list[Unit], settings: Settings) -> list[ChunkDraft]:
    """按 target 聚合。判断用的是**渲染后的 content**，不是 unit token 之和 ——
    因为 content 里还会插 `## 章节标题`，差的就是那几个 token。"""
    drafts: list[ChunkDraft] = []
    current: list[Unit] = []
    for unit in group:
        if current and _content_tokens([*current, unit]) > settings.chunk_target_tokens:
            drafts.append(ChunkDraft(units=current))
            current = []
        current.append(unit)
    if current:
        drafts.append(ChunkDraft(units=current))
    return drafts


def _content_tokens(units: list[Unit]) -> int:
    return estimate_tokens(_render_content(ChunkDraft(units=units)))


# ----------------------------------------------------------------------
# 4. 补足：低于 min 的 chunk 并进相邻 chunk
# ----------------------------------------------------------------------
# 论文边界不用判断：split_paper 每次只处理一篇论文的 blocks，物理上合并不了别的论文
def _merge_small(
    drafts: list[ChunkDraft],
    settings: Settings,
    section_by_id: dict[str, dict[str, Any]],
) -> list[ChunkDraft]:
    # 第一遍：当前的太小 → 并进**前一个**
    merged: list[ChunkDraft] = []
    for draft in drafts:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and draft.tokens < settings.chunk_min_tokens
            and previous.tokens + draft.tokens <= settings.chunk_max_tokens
            and _sections_compatible(previous, draft, section_by_id)
        ):
            previous.units.extend(draft.units)
        else:
            merged.append(ChunkDraft(units=list(draft.units)))

    # 第二遍：还是太小 → 并进**后一个**。
    # 这一遍不能省：父章节（"4 Experimental evaluation" 自己只有标题加一句话）往前并
    # 会被跨父规则挡住（前面是 3.1 的正文），只能往后并进它的第一个子章节 4.1。
    result: list[ChunkDraft] = []
    index = 0
    while index < len(merged):
        current = merged[index]
        following = merged[index + 1] if index + 1 < len(merged) else None
        if (
            following is not None
            and current.tokens < settings.chunk_min_tokens
            and current.tokens + following.tokens <= settings.chunk_max_tokens
            and _sections_compatible(current, following, section_by_id)
        ):
            following.units = current.units + following.units
            index += 1
            continue
        result.append(current)
        index += 1
    return result


def _sections_compatible(
    left: ChunkDraft,
    right: ChunkDraft,
    section_by_id: dict[str, dict[str, Any]],
) -> bool:
    """跨章节合并的准入条件：两个章节**同父**，或者**互为祖先后代**。

    允许：`2.1 + 2.2`（同父）、`4 Experiments + 4.1`（父子相邻）、
          Abstract / Keywords / CCS Concepts 之间（都是顶层兄弟，同父）
    禁止：`2.3 + 3` —— 一个挂在 2 下面、一个是顶层，合起来 section_id
          和章节路径都没法定位（实测出现过 `3.1 Pipeline || 4 Experimental
          evaluation || 4.1 VLMs` 这种）。
    """
    a = left.units[-1].section_id if left.units else None
    b = right.units[0].section_id if right.units else None
    if a == b or a is None or b is None:
        return True
    if _is_ancestor(a, b, section_by_id) or _is_ancestor(b, a, section_by_id):
        return True
    return _parent_of(a, section_by_id) == _parent_of(b, section_by_id)


def _parent_of(section_id: str, section_by_id: dict[str, dict[str, Any]]) -> str | None:
    section = section_by_id.get(section_id)
    return section.get("parent_id") if section else None


def _is_ancestor(
    maybe_ancestor: str,
    node: str,
    section_by_id: dict[str, dict[str, Any]],
) -> bool:
    current = _parent_of(node, section_by_id)
    guard = 0
    while current and guard < 16:
        if current == maybe_ancestor:
            return True
        current = _parent_of(current, section_by_id)
        guard += 1
    return False


# ----------------------------------------------------------------------
# 5. 生成 chunks 行
# ----------------------------------------------------------------------
def _to_row(paper_id: str, draft: ChunkDraft, index: int) -> dict[str, Any]:
    content = _render_content(draft)
    index_text = _render_index_text(draft)
    pages = [unit.page_start for unit in draft.units if unit.page_start is not None]
    page_ends = [unit.page_end for unit in draft.units if unit.page_end is not None]
    kinds = draft.kinds

    return {
        "chunk_id": f"{paper_id}:c{index:04d}",
        "paper_id": paper_id,
        "order_index": index,
        "content": content,
        "index_text": index_text or _fallback_index_text(draft),
        "token_count": estimate_tokens(content),
        "block_ids": _unique([unit.block_id for unit in draft.units]),
        "page_start": min(pages) if pages else None,
        "page_end": max(page_ends) if page_ends else None,
        "chunk_type": next(iter(kinds)) if len(kinds) == 1 else "mixed",
        # 供 indexer 写 chunk_assets，不入 chunks 表
        "_asset_ids": _unique([unit.asset_id for unit in draft.units if unit.asset_id]),
    }


def _render_content(draft: ChunkDraft) -> str:
    """content = `## 章节标题` + 正文 + 资产占位符（给生成模型看）。

    标题在这里是**阅读需要**：模型要知道这段属于哪一节。多占的 token 很少。
    chunk 跨章节时（补足合并会产生），每个章节各插一次标题，边界看得见。
    """
    parts: list[str] = []
    last_section: str | None = None
    for index, unit in enumerate(draft.units):
        if unit.section_id != last_section:
            heading = unit.section_path.split(" > ")[-1] if unit.section_path else ""
            if heading:
                parts.append(f"## {heading}")
            last_section = unit.section_id
        # overlap 前缀只在它成为 chunk 首块时拼出来（见模块开头第 5 条）
        text = unit.head_text if index == 0 else unit.content_text
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _render_index_text(draft: ChunkDraft) -> str:
    """index_text = 检索索引（向量 + BM25 共用），所以标题**要**在里面。"""
    parts: list[str] = []
    last_section: str | None = None
    for index, unit in enumerate(draft.units):
        if unit.section_id != last_section:
            if unit.section_path:
                parts.append(f"[{unit.section_path}]")
            last_section = unit.section_id
        text = unit.head_text if index == 0 else unit.index_text
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _fallback_index_text(draft: ChunkDraft) -> str:
    """整块都没可索引文本时（比如一个超大公式单独成 chunk），至少留下章节路径。"""
    path = draft.units[0].section_path if draft.units else ""
    return f"[{path}]" if path else ""


# ----------------------------------------------------------------------
# 6. 交叉引用：正文提到 Table 3 就把表 3 挂上
# ----------------------------------------------------------------------
def _attach_cross_references(chunks: list[dict[str, Any]], assets: list[dict[str, Any]]) -> None:
    """正文提到 "Table 3" 就把表 3 挂到这个 chunk 上。

    键只用 `assets.label_norm`（入库时从**原始** caption / `\\tag{}` 抽的，
    见 assets.parse_label_norm），精确相等才算命中。抽不出编号（label_norm 是
    NULL）的资产**不参与**：宁可漏挂，也不拿"第几个块"的计数器编号去猜。

    匹配前先把 `[TABLE_REF asset_id=...]` 占位符挖掉，否则占位符里的
    "table0003" 会被正则当成正文提及。
    """
    lookup = _label_lookup(assets)
    for chunk in chunks:
        found = set(chunk["_asset_ids"])
        text = _PLACEHOLDER.sub(" ", chunk["content"])
        for asset_type, pattern in _ASSET_MENTION.items():
            for number in pattern.findall(text):
                asset_id = lookup.get(assets_mod.make_label_norm(asset_type, number) or "")
                if asset_id:
                    found.add(asset_id)
        chunk["_asset_ids"] = sorted(found)


def _label_lookup(assets: list[dict[str, Any]]) -> dict[str, str]:
    """label_norm → asset_id；同一编号对多个资产的，整条不挂（见 _pick_unique）。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for asset in assets:
        norm = asset.get("label_norm")
        if norm:
            grouped.setdefault(norm, []).append(asset)
    return {norm: picked for norm, rows in grouped.items() if (picked := _pick_unique(rows))}


def _pick_unique(rows: list[dict[str, Any]]) -> str | None:
    """同一编号挂在多个资产上时不猜。

    "Figure 2" 在 facecaption-15m 里对应两个块（一张多子图被 MinerU 拆开、caption
    切碎），这时号码本身没有指向性，挂谁都是错的。只在"只有一个候选"，或者
    "只有一个候选的 caption 以该编号开头"（强匹配："Fig. 6: Overview..."）时才认。
    """
    if len(rows) == 1:
        return rows[0]["asset_id"]
    strong = [row for row in rows if row.get("_strong")]
    return strong[0]["asset_id"] if len(strong) == 1 else None


def _unique(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item is not None and item not in seen:
            seen.add(item)
            result.append(item)
    return result
