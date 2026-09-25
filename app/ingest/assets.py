"""blocks → assets（表格 HTML / 公式 LaTeX / 图片），以及"资产在 chunk 里的引用块文本"。

设计来自同项目的上一版 `paper_research_agent`（用同一批论文验证过），见 PLAN §2.5。

核心约定：**本体不进 chunk，chunk 里只留一个占位符**。这两列各自只拿自己需要的那点东西：

    content    给生成模型看 → 只有占位符：  [TABLE_REF asset_id=pe-clip:table0001]
    index_text 进向量       → 只有轻量文本：图 → 图注；表 → 表注 + 表头；公式 → 编号

为什么不让 content 也带上 caption / preview：**那些在 `assets` 表里已经有了**，
chunk 里再存一遍就是同一份数据存两处。占位符里带 `asset_id`，代码在组装证据卡时
按 id 回 `assets` 表取 caption / 表头 / HTML 本体，一处数据一个家。

同理 `index_text` 里不放 `asset_id`、不放图片路径、不放表格数值 ——
进向量的东西只该是"这句话在讲什么"，不是元数据和文件路径。

**`caption` 只存 block 上真有的那份**，没有就是空串。编号另存 `label_norm`
（`table:3` / `figure:2` / `formula:7`，抽不出为 NULL），因为 caption 会真的进
证据卡喂给模型 —— 往里补一个论文里不存在的 "Table 7" 就是凭空造表。
`index_text` 同理**不入库**，要用时由 `index_text_of()` 现算（输入全在同一行里）。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

from app.ingest.converter import BLOCK_EQUATION, BLOCK_IMAGE, BLOCK_TABLE

ASSET_TABLE = "table"
ASSET_FIGURE = "figure"
ASSET_FORMULA = "formula"

# blocks.block_type → assets.asset_type
_ASSET_TYPES = {
    BLOCK_TABLE: ASSET_TABLE,
    BLOCK_IMAGE: ASSET_FIGURE,
    BLOCK_EQUATION: ASSET_FORMULA,
}

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
TABLE_PREVIEW_COLUMNS = 8
# 表格 HTML 解析失败时的纯文本兜底长度
TABLE_FALLBACK_CHARS = 300

# 公式编号：MinerU 把它放在 LaTeX 末尾，`... = 0 \tag{7}`。
_TAG = re.compile(r"\\tag\*?\s*\{([^}]*)\}")
# 罗马数字编号：论文里很常见（"TABLE VIII"），不认它就会白白丢掉一张表的身份。
_ROMAN = re.compile(r"^[IVXLCDM]+$")
_ROMAN_VALUE = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
# 每种资产只认**自己那类**标签：表注里写 "Figure 3" 是上游串了（实测有一张表
# 的 caption 就是图注），不能让这张表变成 figure:3。
_LABEL_WORDS = {
    ASSET_TABLE: r"(?:table|tab\.?|表)",
    ASSET_FIGURE: r"(?:figure|fig\.?|图)",
    ASSET_FORMULA: r"(?:equation|eq\.?|式)",
}
_DISPLAY_WORD = {ASSET_TABLE: "Table", ASSET_FIGURE: "Figure", ASSET_FORMULA: "Equation"}


def make_label_norm(asset_type: str, number: str | None) -> str | None:
    """`(table, "VIII")` → `"table:8"`；数字归一化不了就返回 None。"""
    token = _normalize_number(number)
    return f"{asset_type}:{token}" if token else None


def parse_label_norm(
    asset_type: str,
    caption: str | None,
    raw_content: str | None = None,
) -> str | None:
    """资产的**身份编号**：表格/图片从原始 caption 抠，公式取 LaTeX 末尾的 `\\tag{}`。

    抽不出就返回 None —— 宁可没有名字，也不退回"第几个块"的计数器编号：
    计数器跟论文里的真实编号不是一回事（实测 emotionet / pe-clip 已经各错位 1）。

    入参就是 `assets.caption`（= `block.caption` 原文，没有则为空）。
    """
    if asset_type == ASSET_FORMULA:
        tags = _TAG.findall(raw_content or "")
        return make_label_norm(asset_type, tags[-1]) if tags else None
    pattern = mention_pattern(asset_type)
    if pattern is None:
        return None
    match = pattern.search(caption or "")
    return make_label_norm(asset_type, match.group(1)) if match else None


def label_leads_caption(asset_type: str, caption: str | None) -> bool:
    """caption 是不是**以**这个编号开头（强匹配："Fig. 6: Overview..."）。

    用来在撞号时挑出唯一可信的那个候选，见 splitter._pick_unique。
    """
    pattern = mention_pattern(asset_type)
    if pattern is None:
        return False
    match = pattern.match(caption or "")
    return bool(match and make_label_norm(asset_type, match.group(1)))


def mention_pattern(asset_type: str) -> re.Pattern[str] | None:
    """"提到某个资产"的正则。**caption 侧和正文侧共用同一个**，两边不会各写一套。

    三道约束都是实测踩出来的误报，去掉任何一条都会挂错资产：

    1. 左边不能是字母/数字/反斜杠 —— 否则 "notable 8.68%" 里的 "table 8"、
       LaTeX "\\leq 3" 里的 "eq 3" 都会被当成在提表/提公式；
    2. 右边不能紧跟字母/数字 —— 挡住 "Table In..." 这种词内命中；
    3. 编号既认阿拉伯数字（含 3a 这种子编号），也认罗马数字（"TABLE VIII"）。
    """
    words = _LABEL_WORDS.get(asset_type)
    if words is None:
        return None
    return re.compile(
        rf"(?<![A-Za-z0-9]){words}\s*[（(]?\s*(\d+[a-zA-Z]?|[IVXLCDM]{{1,7}})(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def _normalize_number(value: str | None) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    if token.isdigit():
        return token
    upper = token.upper()
    if _ROMAN.match(upper):
        return str(_roman_to_int(upper))
    return token.lower()


def _roman_to_int(text: str) -> int:
    total = 0
    previous = 0
    for char in reversed(text):
        value = _ROMAN_VALUE.get(char, 0)
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return total


def build_assets(paper_id: str, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 table / figure / equation 块转成 `assets` 表的行。

    每种资产**各有一个计数器**，asset_id 形如 `{paper_id}:table0001`，
    既能一眼看出是第几张表，又不会和别的论文撞。
    """
    counters: dict[str, int] = {ASSET_TABLE: 0, ASSET_FIGURE: 0, ASSET_FORMULA: 0}
    rows: list[dict[str, Any]] = []

    for block in blocks:
        asset_type = _ASSET_TYPES.get(block.get("block_type") or "")
        if asset_type is None:
            continue

        counters[asset_type] += 1
        counter = counters[asset_type]
        asset_id = f"{paper_id}:{asset_type}{counter:04d}"
        # caption 只存 block 上**真有的**那份，没有就是空。
        # 以前这里会补一个假的 "Table 7" / "Figure 7"：那个编号论文里并不存在，
        # 而 caption 会真的进证据卡喂给模型 —— 等于凭空造了一张表出来。
        caption = _clean(block.get("caption") or "")
        raw_content = _raw_content_of(block, asset_type)
        label_norm = parse_label_norm(asset_type, caption, raw_content)
        strong = bool(label_norm) and (
            asset_type == ASSET_FORMULA or label_leads_caption(asset_type, caption)
        )

        rows.append(
            {
                "asset_id": asset_id,
                "block_id": block["block_id"],
                "paper_id": paper_id,
                "asset_type": asset_type,
                "page_idx": block.get("page_idx"),
                "bbox": block.get("bbox"),
                "raw_content": raw_content,
                "image_path": block.get("image_path"),
                "caption": caption,
                "label_norm": label_norm,
                # `_strong` 不进库，只在切分时给撞号的取舍用（见 splitter._pick_unique）
                "_strong": strong,
            }
        )

    return rows


def ref_text(asset: dict[str, Any]) -> str:
    """资产在 `content` 里的占位符 —— 只有 id，正文一律从 `assets` 表取。

    格式固定成 `[TABLE_REF asset_id=xxx]` 这种方括号标记，是为了让后续代码能用
    一个正则把它找出来、按 id 回表把 caption / 表头 / HTML 本体填进证据卡。
    """
    tag = f"{asset['asset_type'].upper()}_REF"
    return f'[{tag} asset_id={asset["asset_id"]}]'


def index_text_of(asset: dict[str, Any]) -> str:
    """资产进向量的检索文本：图 → 图注；表 → 表注 + 表头；公式 → 编号。

    **即时算，不落库**：输入全在 asset 这一行里（caption / raw_content /
    label_norm），存一份快照只会是"同一份数据两个家"，还会跟 caption 一起过时。
    """
    return (
        _index_text_of(
            asset["asset_type"],
            asset.get("caption") or "",
            display_label(asset.get("label_norm")) or "",
            asset.get("raw_content"),
        )
        or ""
    )


# ----------------------------------------------------------------------
# 内部
# ----------------------------------------------------------------------
def display_label(label_norm: str | None) -> str | None:
    """"table:3" → "Table 3"。caption 为空时拿它当资产的显示名（见 retriever._rerank）。"""
    if not label_norm:
        return None
    asset_type, _, number = label_norm.partition(":")
    word = _DISPLAY_WORD.get(asset_type)
    return f"{word} {number}" if word and number else None


def _index_text_of(asset_type: str, caption: str, display: str, raw_content: str | None) -> str:
    """资产贡献给 chunk 的 `index_text` 的那点文本。

    图：只有图注（图片本身没有可索引的文字）
    表：表注 + 表头行（"表 1: 数据集统计  Method | Accuracy | Memory"）
    公式：只有编号（LaTeX 不进向量，见 PLAN 铁律 2）；没编号就是空
    """
    if asset_type == ASSET_TABLE:
        return _join([caption, table_header(raw_content or "")]) or ""
    if asset_type == ASSET_FIGURE:
        return caption
    return display


def _raw_content_of(block: dict[str, Any], asset_type: str) -> str | None:
    if asset_type == ASSET_TABLE:
        return block.get("html")
    if asset_type == ASSET_FORMULA:
        return block.get("latex") or block.get("text")
    return None


def table_header(html: str, max_columns: int = TABLE_PREVIEW_COLUMNS) -> str:
    """表格 HTML → 第一行（表头）的纯文本，用来进 index_text。"""
    parser = _TableParser()
    try:
        parser.feed(html or "")
    except Exception:  # HTMLParser 很少抛，但外部数据不该让整篇论文入库失败
        return _clean(html)[:TABLE_FALLBACK_CHARS]

    rows = parser.rows
    if not rows:
        return _clean(html)[:TABLE_FALLBACK_CHARS]
    return " | ".join(cell for cell in rows[0][:max_columns] if cell)


class _TableParser(HTMLParser):
    """只关心 <tr> / <td> / <th>，其余标签忽略。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._finish_row()
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._finish_cell()
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def close(self) -> None:  # pragma: no cover - 收尾调用，逻辑同 handle_endtag
        super().close()
        self._finish_cell()
        self._finish_row()

    def _finish_cell(self) -> None:
        if self._cell is not None and self._row is not None:
            self._row.append(_clean(" ".join(self._cell)))
        self._cell = None

    def _finish_row(self) -> None:
        self._finish_cell()
        if self._row:
            self.rows.append(self._row)
        self._row = None


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", text or "")).strip()


def _join(parts: list[str | None]) -> str | None:
    items = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    return "\n".join(items) if items else None
