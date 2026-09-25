"""Step 3 验收：blocks → assets，以及资产在 chunk 里的引用块文本。"""

from __future__ import annotations

from app.ingest import converter
from app.ingest.assets import (
    ASSET_FORMULA,
    ASSET_TABLE,
    build_assets,
    index_text_of,
    mention_pattern,
    ref_text,
    table_header,
)

TABLE_HTML = (
    "<table>"
    "<tr><td>Method</td><td>Accuracy</td><td>Memory</td></tr>"
    "<tr><td>Full attention</td><td>0.912</td><td>24.5 GB</td></tr>"
    "<tr><td>Ours</td><td>0.908</td><td>6.1 GB</td></tr>"
    "</table>"
)


def _assets(*entries: dict) -> list[dict]:
    blocks = converter.to_blocks("p1", list(entries))
    return build_assets("p1", blocks)


def test_asset_ids_are_numbered_per_type() -> None:
    assets = _assets(
        {"type": "table", "table_caption": ["Table 1: A"], "table_body": TABLE_HTML, "page_idx": 1},
        {"type": "table", "table_caption": ["Table 2: B"], "table_body": TABLE_HTML, "page_idx": 2},
        {"type": "image", "img_path": "images/f1.jpg", "image_caption": ["Figure 1: C"], "page_idx": 3},
        {"type": "equation", "text": "$$E = mc^2$$", "page_idx": 4},
    )
    assert [a["asset_id"] for a in assets] == [
        "p1:table0001",
        "p1:table0002",
        "p1:figure0001",
        "p1:formula0001",
    ]
    assert [a["asset_type"] for a in assets] == ["table", "table", "figure", "formula"]


def test_asset_keeps_body_in_raw_content_only() -> None:
    (table,) = _assets(
        {"type": "table", "table_caption": ["Table 1: A"], "table_body": TABLE_HTML, "page_idx": 1}
    )
    assert table["raw_content"] == TABLE_HTML
    assert table["caption"] == "Table 1: A"
    # 本体不进占位符，也不进检索文本
    assert "<td>" not in ref_text(table)
    assert "<td>" not in (index_text_of(table) or "")


def test_ref_text_is_a_bare_placeholder() -> None:
    """content 里只有占位符 —— caption / 表头 / 本体都从 assets 表取，不重复存。"""
    (table,) = _assets(
        {"type": "table", "table_caption": ["Table 3: 主实验结果"], "table_body": TABLE_HTML, "page_idx": 7}
    )
    assert ref_text(table) == "[TABLE_REF asset_id=p1:table0001]"


def test_figure_index_text_is_only_the_caption() -> None:
    """图 → 只放图注。asset_id、图片路径都不能进向量。"""
    (figure,) = _assets(
        {
            "type": "image",
            "img_path": "images/f2.jpg",
            "image_caption": ["Figure 2: Training curves"],
            "page_idx": 5,
        }
    )
    text = index_text_of(figure) or ""
    assert "asset_id" not in text
    assert "images/f2.jpg" not in text
    assert text == "Figure 2: Training curves"


def test_table_index_text_is_caption_plus_header() -> None:
    """表 → 表注 + 表头行。表格数值和 HTML 都不进向量。"""
    (table,) = _assets(
        {"type": "table", "table_caption": ["Table 1: 主实验结果"], "table_body": TABLE_HTML, "page_idx": 1}
    )
    text = index_text_of(table) or ""
    assert text.startswith("Table 1: 主实验结果")
    assert "Method | Accuracy | Memory" in text
    # 数值和标签都不该进
    assert "0.912" not in text
    assert "<td>" not in text


def test_table_header_survives_empty_html() -> None:
    assert table_header("") == ""
    assert table_header("<table><tr><td>表头A</td><td>表头B</td></tr></table>") == "表头A | 表头B"


def test_missing_caption_stays_empty_instead_of_inventing_a_number() -> None:
    """没有表注就是空 caption + NULL 编号，绝不拿计数器编一个 "Table 2" 出来。

    caption 会真的进证据卡喂给模型 —— 补出来的编号等于凭空造了一张表。
    检索文本里也不该出现这个假编号（表那边只剩表头行）。
    """
    assets = _assets(
        {"type": "table", "table_caption": ["Table 4: 消融实验"], "table_body": TABLE_HTML, "page_idx": 1},
        {"type": "table", "table_body": TABLE_HTML, "page_idx": 2},
    )
    assert assets[0]["label_norm"] == "table:4"  # caption 里的真实编号
    assert assets[1]["label_norm"] is None  # 没有表注 → 不猜
    assert assets[1]["caption"] == ""
    assert "Table" not in index_text_of(assets[1])
    assert index_text_of(assets[1]) == "Method | Accuracy | Memory"  # 只剩表头


def test_roman_numeral_caption_is_normalized() -> None:
    """罗马数字要认（实测 facecaption-15m 的 10 张表全是 "TABLE I".."TABLE X"）。"""
    (table,) = _assets(
        {
            "type": "table",
            "table_caption": ["TABLE VIII: Text prompts"],
            "table_body": TABLE_HTML,
            "page_idx": 1,
        }
    )
    assert table["label_norm"] == "table:8"
    assert table["caption"] == "TABLE VIII: Text prompts"  # 原始 caption 一字不改


def test_caption_of_another_type_is_not_borrowed() -> None:
    """上游把图注串到表上时（实测真有一张表这样），不能让它变成 figure:3。"""
    (table,) = _assets(
        {
            "type": "table",
            "table_caption": ["Figure 3: FERPlus"],
            "table_body": TABLE_HTML,
            "page_idx": 1,
        }
    )
    assert table["label_norm"] is None


def test_formula_label_norm_comes_from_the_latex_tag() -> None:
    """公式编号取 LaTeX 末尾的 \\tag{} —— 那才是论文里的真实编号。

    退回"第几个公式块"会错位：实测 emotionet / pe-clip 各有一个公式没编号，
    后面的编号就整体差 1，用计数器会挂错资产。
    """
    tagged, untagged = _assets(
        {"type": "equation", "text": "$$E = mc^2 \\tag{7}$$", "page_idx": 1},
        {"type": "equation", "text": "$$\\mathcal{O}(nwd)$$", "page_idx": 2},
    )
    assert tagged["label_norm"] == "formula:7"
    assert index_text_of(tagged) == "Equation 7"
    assert tagged["raw_content"] == "E = mc^2 \\tag{7}"  # 本体只在这里
    assert untagged["label_norm"] is None
    assert index_text_of(untagged) == ""  # 没编号就不给检索文本，不编 "Equation 2"


def test_mention_pattern_ignores_words_that_merely_contain_the_label() -> None:
    """两条真实误报：正文 "notable 8.68%"、LaTeX "\\leq 3"。"""
    table = mention_pattern(ASSET_TABLE)
    formula = mention_pattern(ASSET_FORMULA)
    assert table is not None and formula is not None
    assert not table.search("marking a notable 8.68% improvement")
    assert not formula.search(r"$\theta \leq 3 6 0^{o}$")
    assert table.search("as shown in Table 8")
    assert formula.search("Eq. (3)")


def test_figure_keeps_image_path_in_asset_only() -> None:
    (figure,) = _assets(
        {"type": "image", "img_path": "images/f9.jpg", "image_caption": ["Figure 9: 示意图"], "page_idx": 2}
    )
    assert figure["image_path"] == "images/f9.jpg"
    assert figure["raw_content"] is None
    assert "[FIGURE_REF asset_id=p1:figure0001]" == ref_text(figure)
