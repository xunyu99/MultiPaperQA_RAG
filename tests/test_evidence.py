"""Step 5 验收：证据卡的排序、编号、渲染。

不碰数据库也不碰向量库 —— `retrieve()` 已经把该补齐的都补齐了（那个链路的测试
在 `test_retriever.py`），这里只验证"排成什么样、编成什么号"。
"""

from __future__ import annotations

from typing import Any

from app.generation.evidence import build_cards, render, summarize
from app.retrieval.retriever import RetrievedChunk


def _item(
    chunk_id: str,
    *,
    paper_id: str = "p1",
    section_id: str = "p1:s1",
    section_title: str = "1 Introduction",
    section_order: int = 1,
    chunk_order: int = 1,
    similarity: float = 0.5,
    text: str = "正文",
    page_start: int | None = 1,
    page_end: int | None = 1,
    assets: list[dict[str, Any]] | None = None,
    trace: list[dict[str, Any]] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        similarity=similarity,
        rank=0,
        content=text,
        raw_content=text,
        paper={"paper_id": paper_id, "title": f"论文 {paper_id}", "citation": "Zhang et al. · 2025"},
        chunk_order=chunk_order,
        page_start=page_start,
        page_end=page_end,
        chunk_type="text",
        token_count=10,
        section_id=section_id,
        section_title=section_title,
        section_type="method",
        section_order=section_order,
        assets=assets or [],
        trace=trace or [],
    )


# ----------------------------------------------------------------------
# 排序与编号
# ----------------------------------------------------------------------
def test_cards_are_sorted_by_document_order_not_similarity() -> None:
    """相似度最高的在后，但证据卡必须按文档顺序 —— c0001 在前。"""
    items = [
        _item("p1:c0002", chunk_order=2, similarity=0.9),
        _item("p1:c0001", chunk_order=1, similarity=0.3),
    ]
    cards = build_cards(items)
    assert [card.chunk_id for card in cards] == ["p1:c0001", "p1:c0002"]


def test_sections_are_ordered_by_section_order() -> None:
    items = [
        _item("p1:c0009", section_id="p1:s2", section_title="2 Method", section_order=2, chunk_order=1),
        _item("p1:c0001", section_id="p1:s1", section_title="1 Introduction", section_order=1, chunk_order=1),
    ]
    cards = build_cards(items)
    assert [card.section_title for card in cards] == ["1 Introduction", "2 Method"]


def test_labels_follow_render_order() -> None:
    items = [
        _item("p1:c0003", chunk_order=3, similarity=0.9),
        _item("p1:c0001", chunk_order=1, similarity=0.1),
    ]
    cards = build_cards(items)
    assert [(card.chunk_id, card.label) for card in cards] == [("p1:c0001", "E1"), ("p1:c0003", "E2")]


# ----------------------------------------------------------------------
# 渲染
# ----------------------------------------------------------------------
def test_render_groups_by_paper_and_section() -> None:
    cards = build_cards(
        [
            _item("p1:c0001", paper_id="p1", text="引言正文。"),
            _item("p2:c0001", paper_id="p2", section_id="p2:s1",
                  section_title="1 Introduction", text="另一篇的引言。"),
        ]
    )
    text = render(cards)
    assert "【论文】论文 p1" in text
    assert "【论文】论文 p2" in text
    assert text.count("### 1 Introduction") == 2  # 两篇各自的引言各一行
    assert "[E1] 引言正文。" in text
    assert "[E2] 另一篇的引言。" in text


def test_render_hides_citation_by_default() -> None:
    """citation 存的是作者行原文（姓名+单位+邮箱），直接拼出来会把卡片开头撑爆。"""
    cards = build_cards([_card_with_long_citation()])
    assert "zhang@example.edu" not in render(cards)
    assert "zhang@example.edu" in render(cards, with_citation=True)


def _card_with_long_citation() -> RetrievedChunk:
    item = _item("p1:c0001")
    return RetrievedChunk(
        **{
            **item.__dict__,
            "paper": {
                "paper_id": "p1",
                "title": "论文 p1",
                "citation": "张三 某大学某学院 zhang@example.edu · 2025. In Proceedings of MM '25",
            },
        }
    )


def test_render_prints_section_heading_once_for_multiple_chunks() -> None:
    """同一个章节的多个 chunk 共用一行标题（实测 pe-clip 有章节切成 5 个 chunk）。"""
    items = [
        _item("p1:c0001", chunk_order=1, text="第一段。"),
        _item("p1:c0002", chunk_order=2, text="第二段。"),
        _item("p1:c0003", chunk_order=3, text="第三段。"),
    ]
    text = render(build_cards(items))
    assert text.count("### 1 Introduction") == 1
    assert "[E1] 第一段。" in text and "[E3] 第三段。" in text


def test_render_shows_page_range() -> None:
    cards = build_cards([_item("p1:c0001", page_start=3, page_end=5)])
    assert "（p.3-5）" in render(cards)


def test_content_own_heading_is_not_duplicated() -> None:
    """卡里已经用 `### 标题` 打印章节了，content 自带的同一行 `## 标题` 要去掉。"""
    cards = build_cards([_item("p1:c0001", section_title="4.4 Ablation Analysis",
                               text="## 4.4 Ablation Analysis\n\n正文内容。")])
    text = render(cards)
    assert text.count("4.4 Ablation Analysis") == 1
    assert "正文内容。" in text


def test_other_headings_in_content_are_kept() -> None:
    """只去掉和分组标题一致的那一行，正文里别的标题不动（跨章节合并会产生）。"""
    cards = build_cards([_item("p1:c0001", section_title="4 Experiments",
                               text="## 4 Experiments\n\n正文。\n\n## 4.1 Datasets\n\n子节正文。")])
    text = render(cards)
    assert "## 4.1 Datasets" in text


def test_render_without_cards() -> None:
    assert "没有召回" in render([])


# ----------------------------------------------------------------------
# 自检小结
# ----------------------------------------------------------------------
def test_summarize_counts_assets_and_trace() -> None:
    items = [
        _item(
            "p1:c0001",
            assets=[{"asset_id": "p1:table0001", "asset_type": "table"}],
            trace=[{"block_id": "p1:b1", "page_idx": 1, "bbox": [1, 2, 3, 4]}],
        ),
        _item("p1:c0002", chunk_order=2, section_id="p1:s2", section_title="2 Method", section_order=2),
    ]
    info = summarize(build_cards(items))
    assert info["cards"] == 2
    assert info["papers"] == 1
    assert info["sections"] == 2
    assert info["with_assets"] == 1
    assert info["asset_kinds"] == ["table"]
    assert info["trace_points"] == 1
    assert info["missing"] == []


def test_summarize_flags_missing_assets() -> None:
    info = summarize(build_cards([_item("p1:c0001", text="[TABLE_REF 缺失：p1:table9999]")]))
    assert info["missing"] == ["E1"]
