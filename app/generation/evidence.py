"""证据卡：把检索结果排成"论文 → 章节 → 块"，编上 `[En]`，渲染给模型。

`retriever.retrieve()` 已经把正文、论文、章节、资产本体、溯源坐标都补齐了
（见那个模块的说明），所以这里**只做两件事**：

1. **排序**：按**文档顺序**（论文 → 章节 `order_index` → chunk `order_index`），
   **不是按相似度**。证据卡呈现的是论文的阅读结构，模型读起来能理解上下文；
   相似度写在每块头部当参考，不参与排序。
2. **编号**：`[E1] [E2] ...` 按渲染顺序给，回答里引用它。

渲染出来的结构：

    【论文】EmotioNet: An accurate, real-time algorithm ... （citation）

    ### 2.2. Classification in face space                （p.3-5）

    [E1] We tested the algorithm derived in Sec. 3 on ...

    ### 4. Experimental Results                          （p.5-6）

    [E2] We provide extensive evaluations ...

**同一个章节的多个 chunk 共用一行标题** —— 实测 `pe-clip` 有个章节切成 5 个 chunk，
标题只出现一次。**论文标题 + citation 也只写一次**（多论文分组同理，按 paper_id 分）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.retrieval.retriever import RetrievedChunk


@dataclass
class EvidenceCard:
    label: str
    chunk_id: str
    paper_id: str
    paper_title: str
    citation: str | None
    section_id: str | None
    section_title: str | None
    section_order: int
    chunk_order: int
    page_start: int | None
    page_end: int | None
    similarity: float
    text: str
    assets: list[dict[str, Any]] = field(default_factory=list)
    # 前端溯源高亮用：每个块的 page_idx + bbox（见 PLAN §2.4）
    trace: list[dict[str, Any]] = field(default_factory=list)
    # 排序分：重排分；重排关掉或调用失败时是**合并名次分**（1/位置，不同量纲）。
    # **只用于展示和落盘，不进 prompt** —— 模型没有校准参考，把 0.87 这种浮点
    # 塞给它只会瞎锚（见 PLAN Step 9），而落盘是为了将来标定 τ。
    rank_score: float = 0.0

    @property
    def page_label(self) -> str:
        if self.page_start is None:
            return ""
        if self.page_end in (None, self.page_start):
            return f"p.{self.page_start}"
        return f"p.{self.page_start}-{self.page_end}"


def build_cards(items: list[RetrievedChunk]) -> list[EvidenceCard]:
    """把检索结果排成证据卡。入参是 `retrieve()` 的返回，不需要再查库。"""
    cards = [
        EvidenceCard(
            label="",  # 排完序再编
            chunk_id=item.chunk_id,
            paper_id=str(item.paper.get("paper_id") or ""),
            paper_title=str(item.paper.get("title") or ""),
            citation=item.paper.get("citation"),
            section_id=item.section_id,
            section_title=item.section_title,
            section_order=item.section_order,
            chunk_order=item.chunk_order,
            page_start=item.page_start,
            page_end=item.page_end,
            similarity=item.similarity,
            text=_strip_own_heading(item.content, item.section_title),
            assets=list(item.assets),
            trace=list(item.trace),
            rank_score=item.rank_score,
        )
        for item in items
    ]

    # 文档顺序：论文 → 章节 → chunk。**不按相似度**（见模块开头第 1 条）
    cards.sort(key=lambda c: (c.paper_id, c.section_order, c.chunk_order))
    for index, card in enumerate(cards, 1):
        card.label = f"E{index}"
    return cards


def render(cards: list[EvidenceCard], with_citation: bool = False) -> str:
    """渲染成给模型看的证据卡：按论文分组，同章节的块共用一行标题。

    **默认不显示 citation**（作者 / 年份 / 会议）。原因：`papers.citation` 存的是
    作者行的**原文**（姓名 + 单位 + 城市 + 邮箱挤在同一行，拆不干净 —— 见 §2.5），
    直接拼在标题后面会拉出一大串单位和邮箱，把证据卡开头撑爆。
    标识论文这件事，标题那行已经做完了；作者是谁原 PDF 首页一眼就有。

    数据没丢：`papers.citation` 仍然存全，前端展示、按作者筛选时随时能取。
    需要时传 `with_citation=True` 打开。
    """
    if not cards:
        return "（没有召回任何证据）"

    lines: list[str] = []
    current_paper: str | None = None
    current_section: str | None = None

    for card in cards:
        if card.paper_id != current_paper:
            citation = f"（{card.citation}）" if with_citation and card.citation else ""
            lines.append(f"【论文】{card.paper_title}{citation}")
            lines.append("")
            current_paper = card.paper_id
            current_section = None

        if card.section_id != current_section:
            title = card.section_title or "（无章节标题）"
            pages = f"   （{card.page_label}）" if card.page_label else ""
            lines.append(f"### {title}{pages}")
            lines.append("")
            current_section = card.section_id

        lines.append(f"[{card.label}] {card.text}")
        lines.append("")

    return "\n".join(lines).strip()


def _strip_own_heading(text: str, section_title: str | None) -> str:
    """去掉内容开头那行 `## 章节标题`。

    `content` 里带章节标题是为了**阅读时看得懂归属**（PLAN §2.1）。但证据卡里
    已经用 `### 标题` 把它打印成一行了，再显示一遍就是同一个标题出现两次 ——
    实测"消融实验"那张卡里 `### 4.4 Ablation Analysis` 和 `## 4.4 Ablation Analysis`
    各出现一次。这里只去掉**和分组标题完全一致**的那一行，别的标题不动。
    """
    if not section_title:
        return text
    lines = text.splitlines()
    if lines and lines[0].strip() == f"## {section_title}".strip():
        return "\n".join(lines[1:]).lstrip()
    return text


def summarize(cards: list[EvidenceCard]) -> dict[str, Any]:
    """人工核对用的小结：论文/章节覆盖、资产有没有展开、溯源坐标有几个。"""
    return {
        "cards": len(cards),
        "papers": len({card.paper_id for card in cards}),
        "sections": len({card.section_id for card in cards}),
        "with_assets": sum(1 for card in cards if card.assets),
        "asset_kinds": sorted({asset["asset_type"] for card in cards for asset in card.assets}),
        "trace_points": sum(len(card.trace) for card in cards),
        "missing": [card.label for card in cards if "缺失：" in card.text],
    }
