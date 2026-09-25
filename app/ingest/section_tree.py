"""blocks → 章节树（Step 2 的第二半）。

两件事：

1. **建树**：按标题块的 `heading_level` 维护一个栈，level 变小就弹栈，
   于是 "3 Method" 和 "3.1 Dataset" 的父子关系自然出来。
2. **归一化 section_type**：把 "3.1 Experiments and Results" 这种标题文本
   映射成 experiment。纯关键词规则，不用模型 —— 它要的是稳定可复现，
   不是聪明。

正文开头（标题、作者、没有标题的摘要）不属于任何标题，统一挂到一个
`(preamble)` 根节点下。如果这段里出现了 abstract/摘要，就把根节点的类型判成
abstract —— 不少论文的摘要是不带标题的，不这么兜一下，
「贡献/创新点落在 abstract」这条验收就永远过不了。

产物直接对应 `sections` 表：每个 block 拿到 section_id，sections 拿到
parent_id / level / page_start / page_end。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.ingest.converter import BLOCK_TITLE

PREAMBLE_TITLE = "(preamble)"

# 按优先级从上往下匹配，先命中的算数。顺序是刻意的：
# "Model Evaluation" 该算 experiment 而不是 method，所以 experiment 在前；
# "Related Work" 该算 related 而不是 method，所以 related 在前。
_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reference", ("references", "reference", "bibliography", "参考文献")),
    ("abstract", ("abstract", "摘要", "概要")),
    (
        "related",
        (
            "related work",
            "related",
            "previous work",
            "prior work",
            "literature",
            "相关工作",
            "文献综述",
            "研究现状",
        ),
    ),
    ("conclusion", ("conclusion", "discussion", "结论", "总结", "讨论")),
    ("experiment", ("experiment", "evaluation", "results", "ablation", "实验", "评估", "结果", "消融")),
    (
        "method",
        (
            "method",
            "approach",
            "methodology",
            "framework",
            "architecture",
            "model",
            "algorithm",
            "proposal",
            "proposed",
            "方法",
            "模型",
            "算法",
            "架构",
        ),
    ),
    ("intro", ("introduction", "intro", "background", "引言", "绪论", "背景")),
)

_ABSTRACT_HINTS = ("abstract", "摘要")

# 章节编号的 token 提取器。形状确实很多：1 / 1.2 / 1.2.3 / II / A / 第3章 / 一、，
# 但有一个共同点可以利用：**编号后面一定跟分隔符或空白**。
# 所以不用枚举所有形状，只用这一条就能安全地宽松匹配 ——
# "Introduction" 里的 I、"3D" 里的 3、"Table 2" 里的 2 都不会被当成编号。
# 注意 [IVXLivxl] 里刻意不放 C：这样 "V. Conclusion" 是罗马数字（一级），
# 而 "C. Implementation" 会掉到 letter 分支（子章节）。这两个在实际论文里都常见。
_TITLE_TOKEN = re.compile(
    r"""^\s*(?:
        (?P<arabic>\d+(?:\.\d+)*)
      | (?:[\(\（]\s*(?P<paren>\d+)\s*[\)\）])
      | (?:第\s*(?P<chapter>\d+|[一二三四五六七八九十]+)\s*[章节部分])
      | (?P<cjk>[一二三四五六七八九十]+)
      | (?P<roman>[IVXLivxl]{1,4})
      | (?P<letter>[A-Za-z])
    )\s*(?:[.、,，:：)）]\s*|\s+)(?=\S)""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class SectionBuild:
    """建树结果：sections 与"已填好 section_id 的 blocks"。"""

    sections: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    # 层级是从哪来的：mineru（MinerU 自己给了层级差异）/ numbering（MinerU 给的是平的，
    # 靠标题编号推）/ flat-fallback（连编号都认不出来，只能全按顶层处理）
    hierarchy_source: str = "mineru"
    # 认不出编号的标题数。>0 说明这棵树的层级不完全可信，值得人工看一眼。
    untokenized_titles: int = 0

    @property
    def block_counts(self) -> dict[str, int]:
        counter = Counter(block["section_id"] for block in self.blocks)
        return dict(counter)


def normalize_section_type(title: str | None) -> str:
    """标题文本 → section_type。认不出来就是 other（不留空）。"""
    core = _strip_title_prefix(title or "").lower()
    if not core:
        return "other"
    for section_type, keywords in _TYPE_RULES:
        if any(keyword in core for keyword in keywords):
            return section_type
    return "other"


def build_sections(paper_id: str, blocks: list[dict[str, Any]]) -> SectionBuild:
    """按阅读顺序扫一遍 blocks，建章节树并把每个块挂到所属章节。

    输入建议是 `converter.to_blocks()` 的产物（字段齐全、section_id 为 None）。
    本函数不修改入参，返回的是浅拷贝。
    """
    ordered = sorted(blocks, key=lambda b: b.get("order_index") or 0)
    doc_title_block_id = _document_title_block_id(ordered)
    heading_levels, heading_parents, hierarchy_source, untokenized = _effective_heading_levels(ordered)

    sections: list[dict[str, Any]] = []
    root = _make_section(paper_id, 0, title=PREAMBLE_TITLE, level=0, section_type="other", parent_id=None)
    sections.append(root)

    stack: list[dict[str, Any]] = []
    by_block_id: dict[str, dict[str, Any]] = {}
    current = root
    assigned: list[dict[str, Any]] = []

    for block in ordered:
        block_id = block["block_id"]
        level = heading_levels.get(block_id)

        if block_id == doc_title_block_id:
            # 文档大标题：不建 section，跟着前言走（见 _document_title_block_id）
            copy = dict(block)
            copy["section_id"] = current["section_id"]
            assigned.append(copy)
            continue

        if block.get("block_type") == BLOCK_TITLE and isinstance(level, int) and level >= 1:
            # 编号明确说了父是谁就照它来；否则退回按 level 维护的栈
            forced_parent = heading_parents.get(block_id)
            parent_section = by_block_id.get(forced_parent) if forced_parent else None
            if parent_section is None:
                while stack and stack[-1]["level"] >= level:
                    stack.pop()
                parent_section = stack[-1] if stack else None

            section = _make_section(
                paper_id,
                len(sections),
                title=(block.get("text") or "").strip() or PREAMBLE_TITLE,
                level=level,
                section_type=normalize_section_type(block.get("text")),
                parent_id=parent_section["section_id"] if parent_section else None,
            )
            # 子章节自己认不出来时继承父章节的类型。
            # "3.1 Encoder" 这类标题里没有任何类别关键词，但它显然属于 method；
            # 不继承的话，以后按 section_type='method' 定向过滤只能捞到父标题那一个块。
            if section["section_type"] == "other" and parent_section is not None:
                section["section_type"] = parent_section["section_type"]

            sections.append(section)
            by_block_id[block_id] = section
            # 维护栈：比当前不浅的都弹掉
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            stack.append(section)
            current = section

        copy = dict(block)
        copy["section_id"] = current["section_id"]
        assigned.append(copy)

    _fill_page_ranges(sections, assigned)
    sections, assigned = _drop_empty_root(paper_id, sections, assigned)
    _rebase_levels(sections)
    # 论文以标题开头时根节点会被丢掉，这时第一个 section 是真正的章节，不要动它
    if sections and sections[0]["level"] == 0:
        _refine_preamble(sections[0], assigned)
    return SectionBuild(
        sections=sections,
        blocks=assigned,
        hierarchy_source=hierarchy_source,
        untokenized_titles=untokenized,
    )


def format_tree(sections: list[dict[str, Any]], block_counts: dict[str, int] | None = None) -> str:
    """打印成缩进树。带 block 数，方便一眼看出哪里块数为 0（多半是解析没认出来）。"""
    counts = block_counts or {}
    children: dict[str | None, list[dict[str, Any]]] = {}
    for section in sections:
        children.setdefault(section["parent_id"], []).append(section)
    for group in children.values():
        group.sort(key=lambda s: s["order_index"])

    lines: list[str] = []

    def walk(parent_id: str | None, depth: int) -> None:
        for section in children.get(parent_id, []):
            pages = ""
            if section.get("page_start") is not None:
                pages = f" p.{section['page_start']}"
                if section.get("page_end") not in (None, section["page_start"]):
                    pages += f"-{section['page_end']}"
            lines.append(
                f"{'  ' * depth}- [{section.get('section_type') or 'other'}] "
                f"{section['title']} ({counts.get(section['section_id'], 0)} 块){pages}"
            )
            walk(section["section_id"], depth + 1)

    walk(None, 0)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 内部
# ----------------------------------------------------------------------
def _strip_title_prefix(title: str) -> str:
    """去掉标题开头的编号，留给关键词归一化用。复用同一套 token 规则。"""
    match = _TITLE_TOKEN.match(title or "")
    if match is not None:
        rest = title[match.end() :].strip()
        if rest:
            return rest
    return title.strip()


def _title_blocks(ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        block
        for block in ordered
        if block.get("block_type") == BLOCK_TITLE and isinstance(block.get("heading_level"), int)
    ]


def parse_title_token(title: str | None) -> tuple[str, str] | None:
    """抠出标题开头的章节编号，返回 (类别, 规范化 key)。

    "2.1 Micro-Expression" → ("arabic", "2.1")
    "II. RELATED WORK"     → ("roman", "II")
    "A. Dataset"           → ("letter", "A")
    "第三 章 方法"          → ("cjk", "三")
    "Introduction"         → None（没有编号）

    **不比"点的个数"**：那对罗马数字和字母无效。拿到 key 之后按**编号前缀**定父子
    关系 —— "2.1" 的父就是编号恰好为 "2" 的那个标题。这跟层级有多深、编号长什么样
    都无关，比数点数稳。
    """
    match = _TITLE_TOKEN.match(title or "")
    if match is None:
        return None
    groups = match.groupdict()
    if groups["arabic"]:
        return ("arabic", groups["arabic"])
    if groups["paren"]:
        return ("arabic", groups["paren"])
    if groups["chapter"]:
        return ("cjk", groups["chapter"])
    if groups["cjk"]:
        return ("cjk", groups["cjk"])
    if groups["roman"]:
        return ("roman", groups["roman"].upper())
    if groups["letter"]:
        return ("letter", groups["letter"].upper())
    return None


def _effective_heading_levels(
    ordered: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, str], str, int]:
    """算建树用的层级和父子关系。

    返回 (levels, parents, hierarchy_source, untokenized_titles)。

    真跑出来的第二个坑：MinerU（vlm 模型）会把**所有**章节标题都标成同一个 level ——
    论文标题 level 1，然后 Abstract、1 Introduction、2.1 Micro-Expression… 清一色 level 2。
    层级信息等于没有，子章节全变平级，section_type 也就继承不到父章节。

    判据是数据驱动的：MinerU 真给出了层级差异就听它的；**全是平的**才改用标题编号推。
    """
    all_titles = _title_blocks(ordered)
    raw = {t["block_id"]: int(t["heading_level"]) for t in all_titles}

    doc_title_block_id = _document_title_block_id(ordered)
    titles = [t for t in all_titles if t["block_id"] != doc_title_block_id]
    if len(titles) < 2:
        return raw, {}, "mineru", 0

    # MinerU 给了层级差异 → 听它的，不多事
    if len({int(t["heading_level"]) for t in titles}) > 1:
        return raw, {}, "mineru", 0

    # MinerU 是平的 → 用标题编号里的前缀关系定父子
    tokens = {t["block_id"]: parse_title_token(t.get("text")) for t in titles}
    untokenized = sum(1 for token in tokens.values() if token is None)
    if untokenized == len(titles):
        # 一个编号都认不出来（比如整篇用无编号标题），不硬猜，全按顶层处理
        return raw, {}, "flat-fallback", untokenized

    levels: dict[str, int] = dict(raw)
    parents: dict[str, str] = {}
    seen: dict[tuple[str, str], str] = {}
    last_top_block_id: str | None = None

    for block in titles:
        block_id = block["block_id"]
        token = tokens[block_id]
        parent_id: str | None = None

        if token is not None:
            kind, key = token
            if kind == "arabic" and "." in key:
                # "3.1.2" 的父是 "3.1"
                parent_id = seen.get((kind, key.rsplit(".", 1)[0]))
            elif kind == "letter":
                # IEEE 惯例：A. / B. 是上一个一级章节的子节
                parent_id = last_top_block_id
            seen.setdefault(token, block_id)

        levels[block_id] = levels[parent_id] + 1 if parent_id else 1
        if parent_id:
            parents[block_id] = parent_id
        if levels[block_id] == 1:
            last_top_block_id = block_id

    return levels, parents, "numbering", untokenized


def _document_title_block_id(ordered: list[dict[str, Any]]) -> str | None:
    """判断"第一个标题块其实是文档大标题"并返回它的 block_id。

    真跑出来的坑：MM '25 那篇论文里，论文标题是 level 1，**所有章节标题都是 level 2**。
    照原样建树的话，整篇论文都会挂在这个标题节点下面，而且标题里带 "Model" 一词，
    被归一化成 method，于是 CCS Concepts / Keywords / Acknowledgements 这些子节点
    统统继承了 method —— 全错。

    判据是数据驱动的，不是猜标题文本：**首标题比其它所有标题都浅**，说明它是
    文档标题而不是章节标题。反例是全部同级（我们自己的 demo 样例），这时不触发，
    原来怎么建还怎么建。
    """
    titles = _title_blocks(ordered)
    if len(titles) < 2:
        return None
    first = titles[0]
    if first.get("page_idx") not in (0, None):
        return None
    if all(other["heading_level"] > first["heading_level"] for other in titles[1:]):
        return str(first["block_id"])
    return None


def _rebase_levels(sections: list[dict[str, Any]]) -> None:
    """把章节层级压到 1 起（根节点仍是 0）。

    文档标题被摘掉之后，剩下的顶层章节还挂着 level 2，看着别扭。减一个常数
    不影响父子关系（建栈时已经定好了）。
    """
    levels = [section["level"] for section in sections if section["level"] >= 1]
    if not levels:
        return
    offset = min(levels) - 1
    if offset <= 0:
        return
    for section in sections:
        if section["level"] >= 1:
            section["level"] -= offset


def _make_section(
    paper_id: str,
    order_index: int,
    *,
    title: str,
    level: int,
    section_type: str,
    parent_id: str | None,
) -> dict[str, Any]:
    return {
        "section_id": f"{paper_id}:s{order_index:03d}",
        "paper_id": paper_id,
        "parent_id": parent_id,
        "level": level,
        "order_index": order_index,
        "title": title,
        "section_type": section_type,
        "page_start": None,
        "page_end": None,
    }


def _fill_page_ranges(sections: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> None:
    """章节页码 = 它名下所有块页码的最小值 / 最大值。"""
    spans: dict[str, list[int]] = {}
    for block in blocks:
        page = block.get("page_idx")
        if page is None:
            continue
        spans.setdefault(block["section_id"], []).append(int(page))
    for section in sections:
        pages = spans.get(section["section_id"])
        if pages:
            section["page_start"] = min(pages)
            section["page_end"] = max(pages)


def _drop_empty_root(
    paper_id: str,
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """根节点一个块都没有时把它删掉，并把 section_id / order_index 压紧。

    论文第一块就是标题（很常见）时会走这条路，否则库里每篇论文都会挂一条
    空的 `(preamble)` 章节，看着就邋遢。

    只删空的根节点。非空根节点、以及没有正文但有子章节的标题（"3 Method"
    紧跟 "3.1 Encoder"）都要保留 —— 那是真实的章节结构。
    """
    root = sections[0]
    if root["level"] != 0:
        return sections, blocks
    if any(block["section_id"] == root["section_id"] for block in blocks):
        return sections, blocks

    kept = sections[1:]
    mapping = {section["section_id"]: f"{paper_id}:s{i:03d}" for i, section in enumerate(kept)}

    new_sections: list[dict[str, Any]] = []
    for i, section in enumerate(kept):
        copy = dict(section)
        copy["section_id"] = mapping[section["section_id"]]
        copy["parent_id"] = mapping.get(section["parent_id"] or "", None)
        copy["order_index"] = i
        new_sections.append(copy)

    new_blocks: list[dict[str, Any]] = []
    for block in blocks:
        copy = dict(block)
        copy["section_id"] = mapping[block["section_id"]]
        new_blocks.append(copy)
    return new_sections, new_blocks


def _refine_preamble(root: dict[str, Any], blocks: list[dict[str, Any]]) -> None:
    """根节点若含"摘要"字样，就判成 abstract（论文摘要常常没有标题）。"""
    text = " ".join(
        (block.get("text") or "")
        for block in blocks
        if block["section_id"] == root["section_id"]
    ).lower()
    if any(hint in text for hint in _ABSTRACT_HINTS):
        root["section_type"] = "abstract"
        root["title"] = f"{PREAMBLE_TITLE} abstract"
