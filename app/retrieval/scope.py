"""从问题里认出用户问的是哪篇（哪几篇）论文。

**不用向量**：论文身份是结构信息，字面匹配最准也最可控（库里就几篇）。
向量在这里反而是错的工具 —— 5 篇都是 FER 论文，语义高度重叠，让向量去区分
"这是哪一篇"只会得到一堆 0.5 上下的分数（见 PLAN Step 4 的实测）。

返回**空列表**表示"没认出来，按全库处理" —— 宁可搜宽一点，也不要猜错论文。

别名来源（按可靠性排序）：

1. `paper_id` 本身（`pe-clip` / `emotion-qwen-vl`）—— 我们从文件名推的，用户可能直接用
2. 标题里**冒号之前**的部分（`Rethinking Occlusion in FER`）—— 论文的短名通常就在那儿
3. 标题**前 N 个词** —— 有些标题没有冒号（`AN EVALUATION OF A VISUAL QUESTION ANSWERING
   STRATEGY FOR ZERO-SHOT ...`），整条太长，用户只会说开头一段
4. 标题整体去标点 —— 只有用户把标题说全了才命中

保护：短于 `MIN_ALIAS_LEN` 的别名直接丢掉 —— 不然 `FER`（面部表情识别）会命中一堆论文。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

MIN_ALIAS_LEN = 6
# 标题前几个词也当别名。实测：标题没有冒号时（`AN EVALUATION OF A VISUAL QUESTION
# ANSWERING STRATEGY ...`），只靠"冒号前"和"整条标题"都认不出来，用户只会说开头一段。
TITLE_HEAD_WORDS = 6

# 标题里的副标题分隔符：冒号（中英）、破折号
_SUBTITLE_SPLIT = re.compile(r"[:：]| — | -- ")


def build_aliases(paper: dict[str, Any] | Any) -> set[str]:
    """一篇论文的所有可匹配别名（已转小写、去过短串）。"""
    aliases: set[str] = set()

    paper_id = str(_get(paper, "paper_id") or "").strip().lower()
    if paper_id:
        aliases.add(paper_id)
        # paper_id 的连字符 / 下划线，用户可能写成空格（"pe clip"、"orsanet rethinking ..."）
        for separator in ("-", "_"):
            aliases.add(paper_id.replace(separator, " "))

    title = str(_get(paper, "title") or "").strip()
    if title:
        head = _SUBTITLE_SPLIT.split(title, maxsplit=1)[0].strip()
        if head:
            aliases.add(head.lower())
        cleaned = re.sub(r"\s+", " ", re.sub(r"[^\w\s\-]", " ", title)).strip().lower()
        if cleaned:
            aliases.add(cleaned)
            words = cleaned.split()
            if len(words) > TITLE_HEAD_WORDS:
                aliases.add(" ".join(words[:TITLE_HEAD_WORDS]))

    return {alias for alias in aliases if len(alias) >= MIN_ALIAS_LEN}


def resolve_scope(
    question: str,
    papers: Iterable[dict[str, Any]],
    explicit: list[str] | None = None,
) -> list[str]:
    """返回这次检索该限定在哪些论文里；**空列表 = 全库**。

    `explicit` 非空时直接用它（调用方已经指定了范围，不用再猜）。
    """
    if explicit:
        return list(explicit)

    text = (question or "").lower()
    if not text:
        return []

    hits: list[str] = []
    for paper in papers:
        paper_id = str(_get(paper, "paper_id") or "")
        if not paper_id:
            continue
        if any(alias in text for alias in build_aliases(paper)):
            hits.append(paper_id)
    return hits


def _get(row: dict[str, Any] | Any, key: str) -> Any:
    """兼容 sqlite3.Row 和 dict。Row 没有 .get()。"""
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError):
        return None
