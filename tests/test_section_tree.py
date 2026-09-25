"""Step 2 验收（第二半）：blocks → 章节树 + section_type 归一化。"""

from __future__ import annotations

import pytest

from app.ingest import converter, section_tree
from app.ingest.section_tree import PREAMBLE_TITLE, normalize_section_type


def _paper_starting_with_title() -> list[dict]:
    """最常见的排版：第一块就是论文标题。"""
    return converter.to_blocks(
        "p1",
        [
            {"type": "text", "text": "Attention Is All You Need", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "Ashish Vaswani", "page_idx": 0},
            {"type": "text", "text": "Abstract", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "We propose the Transformer.", "page_idx": 0},
            {"type": "text", "text": "1 Introduction", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "Recurrent models are slow.", "page_idx": 1},
            {"type": "text", "text": "2 Related Work", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "Much work on attention.", "page_idx": 2},
            {"type": "text", "text": "3 Model", "text_level": 1, "page_idx": 2},
            {"type": "text", "text": "3.1 Encoder", "text_level": 2, "page_idx": 2},
            {"type": "text", "text": "The encoder is a stack.", "page_idx": 3},
            {"type": "text", "text": "3.2 Decoder", "text_level": 2, "page_idx": 3},
            {"type": "text", "text": "The decoder is similar.", "page_idx": 3},
            {"type": "text", "text": "4 Experiments", "text_level": 1, "page_idx": 4},
            {"type": "text", "text": "We evaluate on WMT 2014.", "page_idx": 4},
            {"type": "text", "text": "5 Conclusion", "text_level": 1, "page_idx": 5},
            {"type": "text", "text": "We proposed the Transformer.", "page_idx": 5},
            {"type": "text", "text": "References", "text_level": 1, "page_idx": 6},
            {"type": "text", "text": "[1] Bahdanau et al. 2015", "page_idx": 6},
        ],
    )


def _paper_without_abstract_heading() -> list[dict]:
    """摘要没标题：MinerU 连论文标题都没标出来，正文直接顶到最前面。"""
    return converter.to_blocks(
        "p1",
        [
            {"type": "text", "text": "Some Title", "page_idx": 0},
            {"type": "text", "text": "Abstract—We present a new method.", "page_idx": 0},
            {"type": "text", "text": "1 Introduction", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "Body.", "page_idx": 1},
        ],
    )


def test_every_block_gets_a_section() -> None:
    build = section_tree.build_sections("p1", _paper_starting_with_title())
    assert len(build.blocks) == 19
    assert all(block["section_id"] for block in build.blocks)
    section_ids = {section["section_id"] for section in build.sections}
    assert {block["section_id"] for block in build.blocks} <= section_ids


def test_paper_starting_with_title_has_no_empty_preamble() -> None:
    build = section_tree.build_sections("p1", _paper_starting_with_title())
    titles = [section["title"] for section in build.sections]
    assert PREAMBLE_TITLE not in titles
    assert titles[0] == "Attention Is All You Need"
    # order_index 压紧，parent_id 指向仍然有效
    assert [s["order_index"] for s in build.sections] == list(range(len(build.sections)))


def test_nested_subsections() -> None:
    build = section_tree.build_sections("p1", _paper_starting_with_title())
    by_title = {section["title"]: section for section in build.sections}

    model = by_title["3 Model"]
    encoder = by_title["3.1 Encoder"]
    decoder = by_title["3.2 Decoder"]

    assert model["parent_id"] is None
    assert model["level"] == 1
    assert encoder["parent_id"] == model["section_id"]
    assert decoder["parent_id"] == model["section_id"]
    assert encoder["level"] == 2


def test_section_type_is_normalized() -> None:
    build = section_tree.build_sections("p1", _paper_starting_with_title())
    types = {section["title"]: section["section_type"] for section in build.sections}
    assert types["Abstract"] == "abstract"
    assert types["1 Introduction"] == "intro"
    assert types["2 Related Work"] == "related"
    assert types["3 Model"] == "method"
    assert types["3.1 Encoder"] == "method"
    assert types["4 Experiments"] == "experiment"
    assert types["5 Conclusion"] == "conclusion"
    assert types["References"] == "reference"


def test_block_counts_per_section() -> None:
    build = section_tree.build_sections("p1", _paper_starting_with_title())
    counts = build.block_counts
    by_title = {section["title"]: section for section in build.sections}
    # "3 Model" 自己只有标题那一块，正文都在 3.1 / 3.2 里
    assert counts[by_title["3 Model"]["section_id"]] == 1
    assert counts[by_title["3.1 Encoder"]["section_id"]] == 2


def test_page_range_per_section() -> None:
    build = section_tree.build_sections("p1", _paper_starting_with_title())
    by_title = {section["title"]: section for section in build.sections}
    encoder = by_title["3.1 Encoder"]
    assert encoder["page_start"] == 2
    assert encoder["page_end"] == 3


def test_preamble_abstract_heuristic() -> None:
    """摘要没标题时，根节点靠正文里的 abstract 字样判成 abstract。"""
    build = section_tree.build_sections("p1", _paper_without_abstract_heading())
    root = build.sections[0]
    assert root["level"] == 0
    assert root["section_type"] == "abstract"
    assert root["title"].startswith(PREAMBLE_TITLE)
    # 根节点收走了标题和摘要两段
    assert build.block_counts[root["section_id"]] == 2


def test_preamble_stays_other_without_abstract() -> None:
    blocks = converter.to_blocks(
        "p1",
        [
            {"type": "text", "text": "Just some front matter.", "page_idx": 0},
            {"type": "text", "text": "1 Introduction", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "Body.", "page_idx": 1},
        ],
    )
    build = section_tree.build_sections("p1", blocks)
    assert build.sections[0]["title"] == PREAMBLE_TITLE
    assert build.sections[0]["section_type"] == "other"


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Abstract", "abstract"),
        ("摘要", "abstract"),
        ("1 Introduction", "intro"),
        ("2 Background", "intro"),
        ("引言", "intro"),
        ("Related Work", "related"),
        ("2. 相关工作", "related"),
        ("2 Previous work", "related"),
        ("2. Prior Work", "related"),
        ("3 Method", "method"),
        ("3.1 Model Architecture", "method"),
        ("3 Proposal", "method"),
        ("3. Proposed Framework", "method"),
        ("第3章 算法设计", "method"),
        ("4 Experiments", "experiment"),
        ("4.1 Ablation Study", "experiment"),
        ("Model Evaluation", "experiment"),
        ("5 Conclusion and Future Work", "conclusion"),
        ("6 Discussion", "conclusion"),
        ("结论", "conclusion"),
        ("References", "reference"),
        ("参考文献", "reference"),
        ("5G Networks", "other"),
        ("", "other"),
        (None, "other"),
    ],
)
def test_normalize_section_type(title: str | None, expected: str) -> None:
    assert normalize_section_type(title) == expected


def test_format_tree_indents_by_level() -> None:
    build = section_tree.build_sections("p1", _paper_starting_with_title())
    text = section_tree.format_tree(build.sections, build.block_counts)
    lines = text.splitlines()
    top = next(line for line in lines if "3 Model" in line)
    child = next(line for line in lines if "3.1 Encoder" in line)
    assert not top.startswith(" ")
    assert child.startswith("  ")
    assert "[method]" in top


# ----------------------------------------------------------------------
# 文档大标题 vs 章节标题
# ----------------------------------------------------------------------
def _acm_style_paper() -> list[dict]:
    """MM '25 那篇的层级：论文标题 level 1，所有章节标题 level 2。"""
    return converter.to_blocks(
        "p1",
        [
            {"type": "text", "text": "Emotion-Qwen-VL: A Fine-Tuned Model", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "Yujing Wang, Zhiyuan Han", "page_idx": 0},
            {"type": "text", "text": "Abstract", "text_level": 2, "page_idx": 0},
            {"type": "text", "text": "This paper presents our solution.", "page_idx": 0},
            {"type": "text", "text": "CCS Concepts", "text_level": 2, "page_idx": 0},
            {"type": "text", "text": "1 Introduction", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": "Body of the intro.", "page_idx": 1},
            {"type": "text", "text": "2 Related Work", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": "2.1 Micro-Expression Recognition", "text_level": 3, "page_idx": 1},
            {"type": "text", "text": "Prior work.", "page_idx": 2},
        ],
    )


def test_document_title_does_not_become_a_section() -> None:
    build = section_tree.build_sections("p1", _acm_style_paper())
    titles = [section["title"] for section in build.sections]

    assert "Emotion-Qwen-VL: A Fine-Tuned Model" not in titles
    # 标题和作者落进前言根节点，没有被丢掉
    root = build.sections[0]
    assert root["level"] == 0
    assert build.block_counts[root["section_id"]] == 2


def test_chapters_become_top_level_and_levels_are_rebased() -> None:
    build = section_tree.build_sections("p1", _acm_style_paper())
    by_title = {section["title"]: section for section in build.sections}

    abstract = by_title["Abstract"]
    assert abstract["parent_id"] is None
    assert abstract["level"] == 1  # 原来是 2，压到 1 起
    assert abstract["section_type"] == "abstract"

    detail = by_title["2.1 Micro-Expression Recognition"]
    assert detail["parent_id"] == by_title["2 Related Work"]["section_id"]
    assert detail["level"] == 2


def test_metadata_headings_do_not_inherit_a_wrong_type() -> None:
    """CCS Concepts / Keywords 之前因为父节点被误判成 method，继承了 method。"""
    build = section_tree.build_sections("p1", _acm_style_paper())
    by_title = {section["title"]: section for section in build.sections}
    assert by_title["CCS Concepts"]["section_type"] == "other"


def _flat_levels_paper() -> list[dict]:
    """MinerU 把所有标题都标成 level 2 的真实情形（vlm 模型就是这样）。"""
    return converter.to_blocks(
        "p1",
        [
            {"type": "text", "text": "Some Paper Title", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "Author One", "page_idx": 0},
            {"type": "text", "text": "Abstract", "text_level": 2, "page_idx": 0},
            {"type": "text", "text": "摘要正文。", "page_idx": 0},
            {"type": "text", "text": "1 Introduction", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": "引言正文。", "page_idx": 1},
            {"type": "text", "text": "2 Related Work", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": "相关工作正文。", "page_idx": 1},
            {"type": "text", "text": "2.1 Micro-Expression Recognition", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": "子章节正文。", "page_idx": 2},
            {"type": "text", "text": "3 Proposed Method", "text_level": 2, "page_idx": 2},
            {"type": "text", "text": "方法正文。", "page_idx": 2},
            {"type": "text", "text": "3.1 Task Formulation", "text_level": 2, "page_idx": 2},
            {"type": "text", "text": "任务定义正文。", "page_idx": 3},
            {"type": "text", "text": "Acknowledgements", "text_level": 2, "page_idx": 3},
            {"type": "text", "text": "致谢正文。", "page_idx": 3},
            {"type": "text", "text": "References", "text_level": 2, "page_idx": 4},
            {"type": "text", "text": "[1] Someone et al. 2020.", "page_idx": 4},
        ],
    )


def test_flat_mineru_levels_fall_back_to_section_numbering() -> None:
    build = section_tree.build_sections("p1", _flat_levels_paper())
    by_title = {section["title"]: section for section in build.sections}

    # "2.1" 靠编号认出来是第 2 层，挂在 "2 Related Work" 下面
    related = by_title["2 Related Work"]
    sub = by_title["2.1 Micro-Expression Recognition"]
    assert sub["parent_id"] == related["section_id"]
    assert sub["level"] == related["level"] + 1

    # 继承了父章节的类型，而不是掉进 other
    assert sub["section_type"] == "related"
    assert by_title["3.1 Task Formulation"]["section_type"] == "method"


def test_flat_mineru_levels_keep_unnumbered_headings_top_level() -> None:
    """Acknowledgements 没有编号，不能因为 MinerU 给了 level 2 就挂到 5 Conclusion 下面。"""
    build = section_tree.build_sections("p1", _flat_levels_paper())
    by_title = {section["title"]: section for section in build.sections}
    assert by_title["Acknowledgements"]["parent_id"] is None
    assert by_title["References"]["parent_id"] is None
    assert by_title["Abstract"]["parent_id"] is None


# ----------------------------------------------------------------------
# 章节编号的解析（形状很多：阿拉伯、罗马、字母、中文）
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("1 Introduction", ("arabic", "1")),
        ("2 Related Work", ("arabic", "2")),
        ("2.1 Micro-Expression Recognition", ("arabic", "2.1")),
        ("3.1.2 Detail", ("arabic", "3.1.2")),
        ("(2) Related Work", ("arabic", "2")),
        ("II. RELATED WORK", ("roman", "II")),
        ("V. CONCLUSION", ("roman", "V")),
        ("IV Experiments", ("roman", "IV")),
        ("A. Dataset", ("letter", "A")),
        ("C. Implementation Details", ("letter", "C")),
        ("第3章 方法", ("cjk", "3")),
        ("第三章 方法", ("cjk", "三")),
        ("一、绪论", ("cjk", "一")),
        # 这些不是编号，不能被当成编号
        ("Introduction", None),
        ("Abstract", None),
        ("3D Reconstruction", None),
        ("Table 2 shows the results", None),
        ("Acknowledgements", None),
    ],
)
def test_parse_title_token(title: str, expected: tuple[str, str] | None) -> None:
    assert section_tree.parse_title_token(title) == expected


def test_deep_numbering_nests_by_prefix() -> None:
    """比"数点的个数"更本质：3.2.1 的父是 3.2，3.2 的父是 3。"""
    blocks = converter.to_blocks(
        "p1",
        [
            {"type": "text", "text": "Some Title", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "3 Method", "text_level": 2, "page_idx": 0},
            {"type": "text", "text": "3.2 Training", "text_level": 2, "page_idx": 0},
            {"type": "text", "text": "3.2.1 Optimizer", "text_level": 2, "page_idx": 0},
            {"type": "text", "text": "正文。", "page_idx": 0},
        ],
    )
    build = section_tree.build_sections("p1", blocks)
    by_title = {section["title"]: section for section in build.sections}

    method = by_title["3 Method"]
    training = by_title["3.2 Training"]
    optimizer = by_title["3.2.1 Optimizer"]

    assert method["parent_id"] is None
    assert training["parent_id"] == method["section_id"]
    assert optimizer["parent_id"] == training["section_id"]
    assert optimizer["level"] == training["level"] + 1


def test_ieee_roman_and_letter_style() -> None:
    """IEEE 惯例：罗马数字是一级章节，A./B. 是它的子节。"""
    blocks = converter.to_blocks(
        "p1",
        [
            {"type": "text", "text": "Some Title", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "I. INTRODUCTION", "text_level": 2, "page_idx": 0},
            {"type": "text", "text": "II. RELATED WORK", "text_level": 2, "page_idx": 0},
            {"type": "text", "text": "A. Attention Mechanisms", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": "正文。", "page_idx": 1},
            {"type": "text", "text": "B. Transformers", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": "正文。", "page_idx": 1},
            {"type": "text", "text": "III. METHOD", "text_level": 2, "page_idx": 2},
            {"type": "text", "text": "正文。", "page_idx": 2},
        ],
    )
    build = section_tree.build_sections("p1", blocks)
    by_title = {section["title"]: section for section in build.sections}

    assert by_title["I. INTRODUCTION"]["parent_id"] is None
    assert by_title["II. RELATED WORK"]["parent_id"] is None
    # A./B. 挂到最近的二级章节（II）
    assert by_title["A. Attention Mechanisms"]["parent_id"] == by_title["II. RELATED WORK"]["section_id"]
    assert by_title["B. Transformers"]["parent_id"] == by_title["II. RELATED WORK"]["section_id"]
    # 继承父章节类型
    assert by_title["A. Attention Mechanisms"]["section_type"] == "related"
    # III 之后字母计数重来，C./D. 会挂到 III 下面 —— 这里只验证层级不串
    assert by_title["III. METHOD"]["level"] == by_title["II. RELATED WORK"]["level"]


def test_hierarchy_source_is_reported() -> None:
    """层级不可靠时要能报出来，而不是让人以为树是对的。"""
    good = section_tree.build_sections("p1", _paper_starting_with_title())
    assert good.hierarchy_source == "mineru"

    by_number = section_tree.build_sections("p1", _flat_levels_paper())
    assert by_number.hierarchy_source == "numbering"

    # MinerU 平级 + 标题一个编号都没有 → 不硬猜，全按顶层
    no_numbers = converter.to_blocks(
        "p1",
        [
            {"type": "text", "text": "Some Title", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "Introduction", "text_level": 2, "page_idx": 0},
            {"type": "text", "text": "正文。", "page_idx": 0},
            {"type": "text", "text": "Method", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": "正文。", "page_idx": 1},
        ],
    )
    fallback = section_tree.build_sections("p1", no_numbers)
    assert fallback.hierarchy_source == "flat-fallback"
    assert fallback.untokenized_titles == 2
    assert all(section["parent_id"] is None for section in fallback.sections[1:])
