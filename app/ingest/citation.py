"""抽 `papers.citation`：出处串。

主来源是**论文标题到下一个标题之间的内容** —— 也就是首页那段作者信息
（姓名 + 单位 + 邮箱，一段一个作者）：

    Yujing Wang Lianxin Digital (Technology) Hangzhou, Zhejiang, China wangyj@lx-tech.com
    Zhiyuan Han Lianxin Digital (Technology) Hangzhou, Zhejiang, China hanzy@lx-tech.com
    ...

选它的理由：**每篇论文都有，位置由结构决定**（标题块之后、下一个标题块之前），
不依赖有没有 "ACM Reference Format" 那块。

但这段里**没有年份和会议**，而证据卡那行要写 `(Zhang et al., 2024)`。所以再补一段：
如果论文首页有 "ACM Reference Format"（出版社排好的标准引用），就从里面抠出
**年份 + 会议**这一小段接在后面。标题和 DOI 丢掉 —— 标题已经在 `papers.title` 里，
DOI 对问答没用。

结果形如：

    Yujing Wang ... wangyj@lx-tech.com; Zhiyuan Han ... hanzy@lx-tech.com · 2025. In
    Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)

**不解析姓名和单位** —— 它们挤在同一个文本块里（`Yujing Wang Lianxin Digital ...`），
没有可靠的分隔符，硬拆一定会拆错。存原文、展示时截断就够了。

这段是**派生数据**：信息全部来自 blocks，随时能重算，不需要重跑 MinerU。
"""

from __future__ import annotations

import re
from typing import Any

from app.ingest.converter import BLOCK_TEXT, BLOCK_TITLE

# "ACM Reference Format:" / "ACM Reference format:"
_REFERENCE_FORMAT = re.compile(r"reference\s*format", re.IGNORECASE)
# MinerU 用小标标记通讯作者/脚注：<sup>?</sup>、<sup>1</sup>。
# **连内容一起去掉** —— 只删标签会在串里留下一个孤零零的 "?"。
_SUP_ELEMENT = re.compile(r"<sup\b[^>]*>.*?</sup>", re.IGNORECASE | re.DOTALL)
# 其余标签只删标签本身
_HTML_TAG = re.compile(r"<[^>]+>")
# MinerU 有时会把 URL 拆出空格："https: //doi.org/..."
_BROKEN_URL = re.compile(r"(https?)\s*:\s*/\s*/", re.IGNORECASE)
# MinerU 会把 markdown 特殊字符转义："Benitez-Quiroz\*"
_MD_ESCAPE = re.compile(r"\\([\\*_`\[\]\(\)#+\-.!])")
# 出版年份
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*\w")
# 会议名："In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)"
_VENUE = re.compile(r"\bIn\s+[^,]*?\([^)]*\)")
# 看起来像正文段落的开头（摘要没被识别成标题时，用来收住作者段）
_BODY_START = re.compile(r"^\s*(abstract|摘要|摘\s*要)\b", re.IGNORECASE)
# 以句末标点结尾 = 是一个**完整句子** → 正文，不是作者行。
#
# 这条是为了堵住"危险方向"的错：作者行的判据里有"长度 ≤ 300 字符"，万一某篇论文的
# 摘要很短（实测三篇是 975–2179 字符，远超阈值，但短摘要存在），它就会被误判成作者行
# 而**被排除出索引 —— 那是丢内容**。作者行（姓名 + 单位 + 邮箱）几乎不以句号结尾，
# 所以这条判据的两个错误方向是不对称的：误判成正文只是多留一点噪声（安全），
# 误判成作者行会丢正文（危险）。
_SENTENCE_END = ("。", "！", "？", ".", "!", "?")
# 脚注标记：作者段的块如果**以 `<sup>` 开头**，它是脚注不是作者
# （"<sup>?</sup>These authors contributed equally"）。
# 用结构信号判断，不去维护"These authors / Corresponding author / ..."这种词表。
_FOOTNOTE_MARKER = re.compile(r"^\s*<sup\b", re.IGNORECASE)
# 独立的日期行，如 "May 20, 2025" —— 没有 ACM Reference Format 的论文靠它拿年份
_STANDALONE_DATE = re.compile(r"^[A-Z][a-z]+\s+\d{1,2},\s*((?:19|20)\d{2})$")

# 实测（5 篇论文）：真实作者行 58–177 字符；混进来的正文段 632–2179 字符。
# 门槛取 20 是为了滤掉 "May 20, 2025" 这种投稿日期行（12 字符），它不是作者。
MIN_CITATION_LEN = 20
# 上下限都留了足够余量：177 < 300 < 632
MAX_AUTHOR_LINE = 300
# 作者段最多取几块，防止某篇排版异常时把整页吞进来
MAX_AUTHOR_BLOCKS = 12


def extract_citation(
    blocks: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    fallback: str | None = None,
) -> str | None:
    """返回出处串；作者段抽不到就返回 fallback。"""
    ordered = sorted(blocks, key=lambda b: b.get("order_index") or 0)

    authors, date_year = _author_segment(ordered)
    if not authors:
        return fallback

    tail = _year_and_venue(ordered) or (f"{date_year}." if date_year else None)
    joined = "; ".join(authors)
    return _clean(f"{joined} · {tail}" if tail else joined)


# ----------------------------------------------------------------------
# 作者段：标题块之后、下一个标题块之前
# ----------------------------------------------------------------------
def author_block_ids(blocks: list[dict[str, Any]]) -> set[str]:
    """返回被判为"作者行"的块 id。

    **切分时用它把作者行排除出索引**（PLAN BACKLOG B20）：实测作者行占这些块
    15%–34% 的 token，却对问答零价值，只是把摘要挤在一起、稀释语义。

    注意：**不能按章节排除** —— 三篇论文的摘要正文都落在 `preamble` 里
    （MinerU 没把 "Abstract" 标成标题），其中 `pe-clip` 那块有 2179 字。
    整段排除 preamble 会把摘要一起丢掉。所以必须**按块**判断。
    """
    ordered = sorted(blocks, key=lambda b: b.get("order_index") or 0)
    rows, _ = _author_blocks(ordered)
    return {row["block_id"] for row in rows}


def strip_author_lines(text: str) -> str:
    """剔除含邮箱的行 —— 给 `index_text` 用，`content` 不动。

    为什么需要（2026-09-25 决定，PLAN B20 的"方案 b"）：MinerU 会把**摘要正文和作者
    联系信息并进同一个块**（实测 `pe-clip:c0002`，章节头是 `(preamble) abstract`，
    里面塞着 Authors' Contact Information 和一串邮箱）。这种块不是纯作者块，
    `author_block_ids` 的按块判据抓不到，于是作者行占掉这块约一半 token，
    把摘要向量稀释掉 —— 这正是 B20 想解决的问题本身。

    判据只留一条：**行内有邮箱**。姓名和单位挤在同一行、没有可靠分隔符，靠词表迟早会漂；
    邮箱是这里最稳定的结构信号。代价可控：正文里出现邮箱极少，真误删也只影响检索索引 ——
    `content` 原文保留，引用和回答照样看得到作者。

    整段都被剔光时**原样返回**：宁可留一点噪声，也不能让索引文本变成空。
    """
    kept = [line for line in text.splitlines() if not _EMAIL.search(line)]
    if not any(line.strip() for line in kept):
        return text
    return "\n".join(kept)


def is_author_contact_block(text: str) -> bool:
    """块里出现邮箱地址 → 判为作者联系/单位块，`index_text` 剔掉、`content` 保留。

    这是"方案 b"的主力判据，因为实测里作者联系信息**常常自己就是一整块**：
    `pe-clip:b00009`（632 字符的 "Authors' Contact Information: …"）、
    `4:b00011/b00012`（作者单位 + e-mail）、`facecaption-15m:b00020`（844 字符）。
    它们有两个共同特征：**单行长文本**（>300 字符）+ **含邮箱** —— 这正好让
    `_author_blocks` 的长度阈值把它们当正文收住，于是整块漏进索引。

    实测全库只有 19 个块命中邮箱，逐块看过：全部是作者/单位/ACM 版权样板，
    没有一处正文被误伤。整篇论文的正文从来不写邮箱。
    """
    return bool(_EMAIL.search(text or ""))


def _author_blocks(ordered: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """返回 (作者行块列表, 前言里独立日期行的年份)。

    日期行本身**不进作者段**（"May 20, 2025" 只有 12 字符，也不是作者），
    但它的年份有用 —— 没有 ACM Reference Format 的论文就靠它。
    """
    found: list[dict[str, Any]] = []
    date_year: str | None = None
    started = False

    for block in ordered:
        block_type = block.get("block_type")
        if block_type == BLOCK_TITLE:
            if not started:
                started = True  # 文档大标题，作者段从它后面开始
                continue
            break  # 又遇到标题 → 作者段结束

        if not started:
            continue
        # 脚注（"These authors contributed equally"）、页眉页脚页码都不算作者
        if block_type != BLOCK_TEXT:
            continue

        text = (block.get("text") or "").strip()
        # 脚注块（以 <sup> 标记开头）不是作者，但要先取它的年份线索
        if _FOOTNOTE_MARKER.match(text):
            continue
        date_match = _STANDALONE_DATE.match(text)
        if date_match is not None:
            date_year = date_year or date_match.group(1)
            continue
        if text.endswith(_SENTENCE_END):
            break  # 完整句子 → 正文开始了（见 _SENTENCE_END 的说明）
        if len(text) < MIN_CITATION_LEN:
            continue
        # 摘要/正文段落：PE-CLIP 那篇的摘要没被标成标题，靠这两条收住，
        # 否则整段摘要都会被当成作者信息
        if len(text) > MAX_AUTHOR_LINE or _BODY_START.match(text):
            break

        found.append(block)
        if len(found) >= MAX_AUTHOR_BLOCKS:
            break

    return found, date_year


def _author_segment(ordered: list[dict[str, Any]]) -> tuple[list[str], str | None]:
    """作者行的**文本**列表（extract_citation 用），判据与 author_block_ids 完全一致。"""
    rows, date_year = _author_blocks(ordered)
    return [(row.get("text") or "").strip() for row in rows], date_year


# ----------------------------------------------------------------------
# 年份 + 会议：从 "ACM Reference Format" 里抠一小段
# ----------------------------------------------------------------------
def _year_and_venue(ordered: list[dict[str, Any]]) -> str | None:
    reference = _reference_format_text(ordered)
    if reference is None:
        return None

    year_match = _YEAR.search(reference)
    if year_match is None:
        return None
    year = year_match.group(0)

    venue_match = _VENUE.search(reference)
    if venue_match is None:
        return f"{year}."
    return f"{year}. {venue_match.group(0)}"


def _reference_format_text(ordered: list[dict[str, Any]]) -> str | None:
    """找 "ACM Reference Format:" 标题块，返回它所在章节里后面那段长文本。"""
    for index, block in enumerate(ordered):
        if block.get("block_type") != BLOCK_TITLE:
            continue
        if not _REFERENCE_FORMAT.search(block.get("text") or ""):
            continue
        section_id = block.get("section_id")
        for candidate in ordered[index + 1 :]:
            if candidate.get("section_id") != section_id:
                break  # 走出这个章节了
            if candidate.get("block_type") == BLOCK_TEXT:
                text = (candidate.get("text") or "").strip()
                if len(text) >= MIN_CITATION_LEN:
                    return text
    return None


def _clean(text: str) -> str:
    without_sup = _SUP_ELEMENT.sub("", text)
    without_tags = _HTML_TAG.sub("", without_sup)
    fixed_url = _BROKEN_URL.sub(r"\1://", without_tags)
    unescaped = _MD_ESCAPE.sub(r"\1", fixed_url)
    return " ".join(unescaped.split())
