"""content_list.json → Block 列表（Step 2 的第一半）。

MinerU 交出来的 `*_content_list.json` 是一个**平铺数组**，标题、正文、图片、表格、
公式混在一起按阅读顺序排。这一步只做"翻译 + 配对"，不建树（建树是 section_tree 的事）：

1. 每个 entry → 一个 block 字典，字段与 `blocks` 表一一对应；
2. 图片/表格的**图注配对**：优先用 MinerU 自带的 caption 字段，
   没有就找同页相邻的、长得像图注的文本块；
3. 公式的 LaTeX 从 `text` 挪到 `latex`，去掉 `$$` 包裹。

刻意保持纯函数：输入 JSON、输出字典列表，不碰数据库、不碰网络 —— 这样能单测，
也能在 MinerU 挂了的时候拿本地已有产物重跑。

`order_index` 直接用 content_list 的下标（翻译过程中被丢掉的空 entry 会留下空档）。
留空档是故意的：它是"这块对应原文第几条"的可追溯线索，调试时要靠它回查原始 JSON。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# block_type 取值（与 schema 注释一致）
BLOCK_TEXT = "text"
BLOCK_TITLE = "title"
BLOCK_LIST = "list"
BLOCK_CAPTION = "caption"
BLOCK_IMAGE = "image"
BLOCK_TABLE = "table"
BLOCK_EQUATION = "equation"
BLOCK_REFERENCE = "reference"
BLOCK_FOOTNOTE = "footnote"
BLOCK_NOISE = "noise"
BLOCK_OTHER = "other"

# MinerU 的 type → 我们的 block_type。
#
# 这几个映射是真跑一篇 MM '25 论文之后补的：MinerU 实际会吐出 8 种类型，
# 不认识的会全部落进 other。其中 header / footer / page_number 是**版式噪声**
# （页眉的论文短标题、"Yujing Wang et al."、页码 13972），它们混进 chunk
# 会直接污染向量和 BM25 —— 一页一条页码，重复度还特别高。
_TYPE_MAP = {
    "text": BLOCK_TEXT,
    "image": BLOCK_IMAGE,
    "figure": BLOCK_IMAGE,
    "table": BLOCK_TABLE,
    # v1 的 "equation" 就是独立成块的 display 公式（= v2 的 equation_interline）。
    "equation": BLOCK_EQUATION,
    "interline_equation": BLOCK_EQUATION,
    # inline_equation 刻意**不**当资产：行内公式本来就该待在正文里（实测 v2 里
    # 232 个 inline 公式全部嵌在 paragraph 内，从不单独出块）。映射成公式块会让
    # 它凭空多出一个假公式资产，还会把假编号喂给交叉引用。
    "inline_equation": BLOCK_TEXT,
    "list": BLOCK_LIST,
    "ref_text": BLOCK_REFERENCE,
    "page_footnote": BLOCK_FOOTNOTE,
    "header": BLOCK_NOISE,
    "footer": BLOCK_NOISE,
    "page_number": BLOCK_NOISE,
}

# 版式噪声：入库保留（可追溯、可统计），但切分时**必须跳过**，不进任何 chunk。
# 详见 splitter（Step 3）。
NOISE_TYPES = (BLOCK_NOISE,)

# 图注/表注的标签：Fig. 3 / Figure3 / 图 3 / Table 2 / 表2 / 式(3) / Eq. (3)
_CAPTION_LABEL = re.compile(
    r"^(?:fig(?:ure)?|tab(?:le)?|图|表|式|eq(?:uation)?)\.?\s*[\(（]?\s*\d+\s*[\)）]?",
    re.IGNORECASE,
)
_CAPTION_SEPARATORS = (":", "：", ".", "、", "-", "—")
# 超过这个长度还以 "Table 1 ..." 开头的，当正文，不当图注
_CAPTION_MAX_LEN = 300
_CAPTION_SHORT_LEN = 60


def load_content_list(path: str | Path) -> list[dict[str, Any]]:
    """读 content_list.json。顶层必须是数组，否则直接报错别猜。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"content_list 顶层应该是数组，实际是 {type(data).__name__}：{path}")
    return [item for item in data if isinstance(item, dict)]


def is_caption_text(text: str | None) -> bool:
    """判断一段文本像不像图注/表注。

    靠两条：以标签+编号开头，**并且**后面紧跟分隔符（"Table 1: ..."），
    或者整段很短（"图 3 不同方法的对比"）。
    这样能把 "Table 1 shows the results of ..."（正文）排除掉。
    """
    value = (text or "").strip()
    if not value or len(value) > _CAPTION_MAX_LEN:
        return False
    match = _CAPTION_LABEL.match(value)
    if match is None:
        return False
    rest = value[match.end() :]
    if rest[:1] in _CAPTION_SEPARATORS:
        return True
    return len(value) <= _CAPTION_SHORT_LEN


def to_blocks(paper_id: str, content_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """content_list → blocks。返回的每个字典字段与 `blocks` 表完全对应。

    注意 `section_id` 这里统一是 None：章节归属要等 section_tree 扫完全文才能定，
    本函数不做半成品猜测。
    """
    blocks: list[dict[str, Any]] = []
    for order_index, entry in enumerate(content_list):
        block = _entry_to_block(paper_id, order_index, entry)
        if block is not None:
            blocks.append(block)
    pair_captions(blocks)
    return blocks


def pair_captions(blocks: list[dict[str, Any]]) -> int:
    """给缺图注的图片/表格补图注，返回补上的条数。

    MinerU 通常已经把 caption 放进同一个 entry 里，这条兜底处理的是它漏掉的：
    图注被单独识别成一个文本块。配对条件很保守 —— 同页、紧邻（前一块或后一块）、
    且文本确实长得像图注。配上的同时把那个文本块的类型改成 `caption`，
    但**内容保留**（原子块永不丢内容），免得后面谁把它当正文再切一遍。
    """
    fixed = 0
    for index, block in enumerate(blocks):
        if block["block_type"] not in (BLOCK_IMAGE, BLOCK_TABLE):
            continue
        if block["caption"]:
            continue
        for neighbour_index in (index + 1, index - 1):
            if not 0 <= neighbour_index < len(blocks):
                continue
            neighbour = blocks[neighbour_index]
            if neighbour["block_type"] != BLOCK_TEXT:
                continue
            if neighbour["page_idx"] != block["page_idx"]:
                continue
            if not is_caption_text(neighbour["text"]):
                continue
            block["caption"] = neighbour["text"]
            neighbour["block_type"] = BLOCK_CAPTION
            fixed += 1
            break
    return fixed


def caption_coverage(blocks: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """图注配对情况：{类型: (有图注数, 总数)}。Step 2 验收要打印它。"""
    stats: dict[str, tuple[int, int]] = {}
    for kind in (BLOCK_IMAGE, BLOCK_TABLE):
        items = [b for b in blocks if b["block_type"] == kind]
        with_caption = sum(1 for b in items if b["caption"])
        stats[kind] = (with_caption, len(items))
    return stats


def type_counts(blocks: list[dict[str, Any]]) -> dict[str, int]:
    """块类型分布。落库前后都能用（入参是 block 字典列表，不是数据库行）。"""
    counts: dict[str, int] = {}
    for block in blocks:
        kind = str(block.get("block_type") or BLOCK_OTHER)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def extract_title(blocks: list[dict[str, Any]], fallback: str) -> str:
    """论文标题取第一个 title 块。取不到就用文件名，不留空（papers.title 是 NOT NULL）。"""
    for block in blocks:
        if block["block_type"] == BLOCK_TITLE and (block["text"] or "").strip():
            return block["text"].strip()
    return fallback


# ----------------------------------------------------------------------
# 内部
# ----------------------------------------------------------------------
def _entry_to_block(paper_id: str, order_index: int, entry: dict[str, Any]) -> dict[str, Any] | None:
    raw_type = str(entry.get("type") or "").strip().lower()
    block_type = _TYPE_MAP.get(raw_type, BLOCK_OTHER)

    heading_level = _as_int(entry.get("text_level"))
    # MinerU 用 text_level 标记标题层级；只有 text 类型才可能变成标题
    if block_type == BLOCK_TEXT and heading_level:
        block_type = BLOCK_TITLE

    text = _as_text(entry.get("text"))
    latex: str | None = None
    html = _as_text(entry.get("table_body")) if raw_type == "table" else None
    caption = _join([
        _as_text(entry.get("image_caption")),
        _as_text(entry.get("table_caption")),
        _as_text(entry.get("image_footnote")),
        _as_text(entry.get("table_footnote")),
    ])
    image_path = _as_text(entry.get("img_path")) or _as_text(entry.get("image_path"))

    if block_type == BLOCK_LIST:
        text = _join_list_items(entry.get("list_items")) or text

    if block_type == BLOCK_EQUATION:
        # 公式：正文位置留空，LaTeX 单独存。行内公式被 MinerU 单独吐出来时也走这里，
        # 靠 order_index 在原文里的位置仍能判断它属于哪一章。
        latex = _strip_math_delimiters(text)
        text = None

    if block_type == BLOCK_TITLE:
        # 标题块不该带图片/公式
        image_path = image_path or None

    if not any((text, latex, html, caption, image_path)):
        return None  # 空 entry 直接丢掉，别在库里留空行

    return {
        "block_id": f"{paper_id}:b{order_index:05d}",
        "paper_id": paper_id,
        "section_id": None,
        "order_index": order_index,
        "block_type": block_type,
        "heading_level": heading_level if block_type == BLOCK_TITLE else None,
        "text": text,
        "latex": latex,
        "html": html,
        "caption": caption,
        "page_idx": _as_int(entry.get("page_idx")),
        "bbox": _as_bbox(entry.get("bbox")),
        "image_path": image_path,
    }


def _as_text(value: Any) -> str | None:
    """MinerU 的文本字段有时是 str，有时是 list[str]，统一成一段文本。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        return _join(_as_text(item) for item in value)
    return str(value).strip() or None


def _join(parts: Any) -> str | None:
    items = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    return "\n".join(items) if items else None


def _join_list_items(value: Any) -> str | None:
    if not isinstance(value, (list, tuple)):
        return None
    lines = [_as_text(item) for item in value]
    return _join(lines)


def _strip_math_delimiters(text: str | None) -> str | None:
    """去掉公式两端的 $$ / $ / \\[ \\]。"""
    value = (text or "").strip()
    if not value:
        return None
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("$", "$")):
        if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
            return value[len(left) : -len(right)].strip() or None
    return value


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bbox(value: Any) -> list[float] | None:
    """bbox 统一成 [x0, y0, x1, y1] 四个数；形状不对就当没有。"""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None
