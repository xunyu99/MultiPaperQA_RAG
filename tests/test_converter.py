"""Step 2 验收（第一半）：content_list.json → blocks 的翻译与图注配对。

这些都是纯函数，不需要网络也不需要 MinerU 账号 —— 拿一份手工构造的
content_list 就能把规则钉死。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.ingest import converter
from app.ingest.converter import (
    BLOCK_CAPTION,
    BLOCK_EQUATION,
    BLOCK_FOOTNOTE,
    BLOCK_IMAGE,
    BLOCK_LIST,
    BLOCK_NOISE,
    BLOCK_REFERENCE,
    BLOCK_TABLE,
    BLOCK_TEXT,
    BLOCK_TITLE,
)

# 一份缩小的 content_list，覆盖：标题层级、摘要、图/表自带 caption、
# 公式、列表、图注被单独识别成文本块、以及"正文以 Table 2 开头"的干扰项。
CONTENT_LIST: list[dict[str, Any]] = [
    {"type": "text", "text": "Attention Is All You Need", "text_level": 1, "page_idx": 0},
    {"type": "text", "text": "Ashish Vaswani, Noam Shazeer", "page_idx": 0},
    {"type": "text", "text": "Abstract", "text_level": 1, "page_idx": 0},
    {"type": "text", "text": "The dominant sequence transduction models are based on RNNs.", "page_idx": 0},
    {"type": "text", "text": "1 Introduction", "text_level": 1, "page_idx": 1},
    {"type": "text", "text": "Recurrent neural networks have been firmly established.", "page_idx": 1},
    {
        "type": "image",
        "img_path": "images/f1.jpg",
        "image_caption": ["Figure 1: The Transformer architecture."],
        "page_idx": 1,
        "bbox": [10, 20, 300, 400],
    },
    {"type": "text", "text": "2 Background", "text_level": 1, "page_idx": 2},
    {
        "type": "table",
        "table_caption": ["Table 1: Maximum path lengths."],
        "table_body": "<table><tr><td>0.85</td></tr></table>",
        "img_path": "images/t1.jpg",
        "page_idx": 2,
    },
    {"type": "equation", "text": "$$\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}(QK^T)V$$", "page_idx": 3},
    {"type": "list", "list_items": ["第一条", "第二条"], "page_idx": 3},
    # 图注没被 MinerU 配进 image entry，而是单独成了一个文本块 → 触发兜底配对
    {"type": "image", "img_path": "images/f2.jpg", "page_idx": 4},
    {"type": "text", "text": "Figure 2: Training curves on WMT 2014.", "page_idx": 4},
    # 干扰项：正文以 "Table 2" 开头，但它不是图注，不能被配走
    {"type": "table", "table_body": "<table></table>", "page_idx": 5},
    {
        "type": "text",
        "text": (
            "Table 2 shows the results of our ablation study on the WMT 2014 "
            "English-to-German dataset, where we vary the number of attention heads."
        ),
        "page_idx": 5,
    },
]


@pytest.fixture()
def blocks() -> list[dict[str, Any]]:
    return converter.to_blocks("p1", CONTENT_LIST)


def _at(blocks: list[dict[str, Any]], order_index: int) -> dict[str, Any]:
    return next(b for b in blocks if b["order_index"] == order_index)


# ----------------------------------------------------------------------
# 翻译
# ----------------------------------------------------------------------
def test_every_entry_becomes_one_block(blocks: list[dict[str, Any]]) -> None:
    assert len(blocks) == len(CONTENT_LIST)
    assert [b["order_index"] for b in blocks] == list(range(len(CONTENT_LIST)))


def test_block_types(blocks: list[dict[str, Any]]) -> None:
    assert [b["block_type"] for b in blocks] == [
        BLOCK_TITLE,
        BLOCK_TEXT,
        BLOCK_TITLE,
        BLOCK_TEXT,
        BLOCK_TITLE,
        BLOCK_TEXT,
        BLOCK_IMAGE,
        BLOCK_TITLE,
        BLOCK_TABLE,
        BLOCK_EQUATION,
        BLOCK_LIST,
        BLOCK_IMAGE,
        BLOCK_CAPTION,
        BLOCK_TABLE,
        BLOCK_TEXT,
    ]


def test_heading_level_only_on_titles(blocks: list[dict[str, Any]]) -> None:
    assert _at(blocks, 0)["heading_level"] == 1
    assert _at(blocks, 4)["heading_level"] == 1
    assert _at(blocks, 1)["heading_level"] is None
    assert _at(blocks, 6)["heading_level"] is None


def test_block_ids_and_section_id(blocks: list[dict[str, Any]]) -> None:
    assert _at(blocks, 0)["block_id"] == "p1:b00000"
    assert _at(blocks, 14)["block_id"] == "p1:b00014"
    # 章节归属要等 section_tree 扫完全文，这里统一是 None
    assert {b["section_id"] for b in blocks} == {None}
    assert {b["paper_id"] for b in blocks} == {"p1"}


def test_equation_moves_latex_out_of_text(blocks: list[dict[str, Any]]) -> None:
    equation = _at(blocks, 9)
    assert equation["text"] is None
    assert equation["latex"] == "\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}(QK^T)V"


def test_list_items_are_joined(blocks: list[dict[str, Any]]) -> None:
    assert _at(blocks, 10)["text"] == "第一条\n第二条"


def test_bbox_kept_as_numbers(blocks: list[dict[str, Any]]) -> None:
    assert _at(blocks, 6)["bbox"] == [10.0, 20.0, 300.0, 400.0]
    assert _at(blocks, 5)["bbox"] is None


def test_table_html_and_image_path(blocks: list[dict[str, Any]]) -> None:
    table = _at(blocks, 8)
    assert table["html"] == "<table><tr><td>0.85</td></tr></table>"
    assert table["image_path"] == "images/t1.jpg"
    assert table["caption"] == "Table 1: Maximum path lengths."


# ----------------------------------------------------------------------
# 图注配对
# ----------------------------------------------------------------------
def test_inline_caption_is_kept(blocks: list[dict[str, Any]]) -> None:
    assert _at(blocks, 6)["caption"] == "Figure 1: The Transformer architecture."


def test_caption_is_paired_from_neighbour_block(blocks: list[dict[str, Any]]) -> None:
    image = _at(blocks, 11)
    assert image["caption"] == "Figure 2: Training curves on WMT 2014."
    # 被配走的文本块类型改成 caption，但内容保留（原子块永不丢内容）
    neighbour = _at(blocks, 12)
    assert neighbour["block_type"] == BLOCK_CAPTION
    assert neighbour["text"] == "Figure 2: Training curves on WMT 2014."


def test_body_text_starting_with_table_is_not_a_caption(blocks: list[dict[str, Any]]) -> None:
    # 第 13 块是空表格，第 14 块是长正文。正文不该被当成图注配过去。
    assert _at(blocks, 13)["caption"] is None
    assert _at(blocks, 14)["block_type"] == BLOCK_TEXT


def test_caption_coverage(blocks: list[dict[str, Any]]) -> None:
    # 图片两张都配上了；表格只有第一张有图注
    assert converter.caption_coverage(blocks) == {BLOCK_IMAGE: (2, 2), BLOCK_TABLE: (1, 2)}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Figure 1: The Transformer architecture.", True),
        ("Fig.3 Architecture", True),
        ("图 2 不同方法的对比", True),
        ("表3：消融实验结果", True),
        ("式(3) 是注意力公式", True),
        ("Table 1: Maximum path lengths.", True),
        ("", False),
        (None, False),
        ("Attention is all you need.", False),
        (
            "Table 2 shows the results of our ablation study on the WMT 2014 "
            "English-to-German dataset, where we vary the number of attention heads.",
            False,
        ),
    ],
)
def test_is_caption_text(text: str | None, expected: bool) -> None:
    assert converter.is_caption_text(text) is expected


# ----------------------------------------------------------------------
# 其它
# ----------------------------------------------------------------------
def test_type_counts(blocks: list[dict[str, Any]]) -> None:
    counts = converter.type_counts(blocks)
    assert counts[BLOCK_TEXT] == 4
    assert counts[BLOCK_TITLE] == 4
    assert counts[BLOCK_CAPTION] == 1


def test_extract_title(blocks: list[dict[str, Any]]) -> None:
    assert converter.extract_title(blocks, "fallback") == "Attention Is All You Need"
    assert converter.extract_title([], "fallback") == "fallback"


def test_empty_entries_are_dropped() -> None:
    blocks = converter.to_blocks("p1", [{"type": "text", "text": "   "}, {"type": "unknown"}])
    assert blocks == []


def test_load_content_list(tmp_path: Path) -> None:
    path = tmp_path / "x_content_list.json"
    path.write_text(json.dumps(CONTENT_LIST, ensure_ascii=False), encoding="utf-8")
    assert len(converter.load_content_list(path)) == len(CONTENT_LIST)

    bad = tmp_path / "bad_content_list.json"
    bad.write_text('{"type": "text"}', encoding="utf-8")
    with pytest.raises(ValueError, match="顶层应该是数组"):
        converter.load_content_list(bad)


# ----------------------------------------------------------------------
# MinerU 实际会吐的版式类型（真跑 MM '25 那篇论文之后补的映射）
# ----------------------------------------------------------------------
def test_mineru_layout_types_are_mapped() -> None:
    blocks = converter.to_blocks(
        "p1",
        [
            {"type": "ref_text", "text": "[1] Wu et al. 2010. Research on micro-expression", "page_idx": 6},
            {"type": "header", "text": "Yujing Wang et al.", "page_idx": 1},
            {"type": "footer", "text": "1 BY", "page_idx": 1},
            {"type": "page_number", "text": "13972", "page_idx": 1},
            {"type": "page_footnote", "text": "∗Yuhao Shan and Tong Chen are the corresponding authors", "page_idx": 0},
        ],
    )
    assert [b["block_type"] for b in blocks] == [
        BLOCK_REFERENCE,
        BLOCK_NOISE,
        BLOCK_NOISE,
        BLOCK_NOISE,
        BLOCK_FOOTNOTE,
    ]
    # 切分时必须跳过这些（Step 3 的 splitter 读这个常量）
    assert converter.NOISE_TYPES == (BLOCK_NOISE,)


def test_page_numbers_do_not_get_mistaken_for_content() -> None:
    """页码一页一条且高度重复，混进 chunk 会污染向量和 BM25，所以单独标成 noise。"""
    blocks = converter.to_blocks(
        "p1",
        [{"type": "page_number", "text": str(n), "page_idx": n} for n in range(13972, 13979)],
    )
    assert len(blocks) == 7
    assert {b["block_type"] for b in blocks} == {BLOCK_NOISE}
