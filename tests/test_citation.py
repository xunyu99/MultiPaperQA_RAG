"""出处串抽取：作者段取"标题到下一个标题之间"，年份会议从 Reference Format 补。"""

from __future__ import annotations

from app.ingest import converter, section_tree
from app.ingest.citation import extract_citation


def _build(*entries: dict) -> section_tree.SectionBuild:
    return section_tree.build_sections("p1", converter.to_blocks("p1", list(entries)))


def test_takes_all_author_lines_between_title_and_next_heading() -> None:
    """作者段是"标题到下一个标题之间"的**全部**作者行，不是只取第一行。"""
    build = _build(
        {"type": "text", "text": "Emotion-Qwen-VL: A Fully Fine-Tuned Model", "text_level": 1, "page_idx": 0},
        {
            "type": "text",
            "text": "Yujing Wang Lianxin Digital (Technology) Hangzhou, Zhejiang, China wangyj@lx-tech.com",
            "page_idx": 0,
        },
        {
            "type": "text",
            "text": "Zhiyuan Han Lianxin Digital (Technology) Hangzhou, Zhejiang, China hanzy@lx-tech.com",
            "page_idx": 0,
        },
        {
            "type": "text",
            "text": "Tong Chen* Southwest University, Chongqing, China c_tong@swu.edu.cn",
            "page_idx": 0,
        },
        {"type": "text", "text": "Abstract", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "正文。", "page_idx": 0},
    )
    citation = extract_citation(build.blocks, build.sections)
    assert citation is not None
    assert citation.count(";") == 2  # 三行作者，用 "; " 连起来
    assert "Yujing Wang" in citation
    assert "Zhiyuan Han" in citation
    assert "Tong Chen*" in citation


def test_year_and_venue_are_appended_from_reference_format() -> None:
    build = _build(
        {"type": "text", "text": "Some Title", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Jane Doe Some University jane@example.edu", "page_idx": 0},
        {"type": "text", "text": "ACM Reference Format:", "text_level": 2, "page_idx": 0},
        {
            "type": "text",
            "text": (
                "Jane Doe, John Smith, and Ann Lee. 2025. Some Title. "
                "In Proceedings of the 33rd ACM International Conference on "
                "Multimedia (MM '25), October 27-31, 2025, Dublin, Ireland. "
                "https://doi.org/10.1145/1234567.1234568"
            ),
            "page_idx": 0,
        },
        {"type": "text", "text": "1 Introduction", "text_level": 2, "page_idx": 1},
        {"type": "text", "text": "正文。", "page_idx": 1},
    )
    citation = extract_citation(build.blocks, build.sections)
    assert citation is not None
    # 作者段在前，年份+会议在后
    assert "Jane Doe Some University jane@example.edu" in citation
    assert " · 2025. In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)" in citation
    # 标题和 DOI 不该出现（标题已经在 papers.title 里）
    assert "10.1145" not in citation
    assert "Some Title. In Proceedings" not in citation


def test_stops_at_abstract_when_heading_is_missing() -> None:
    """PE-CLIP 那篇的摘要没被识别成标题，不能把整段摘要当成作者信息。"""
    # 真实摘要长度量级（实测 PE-CLIP 的正文段是 632–2179 字符）
    long_abstract = "Facial expression recognition has attracted increasing attention in recent years. " * 8
    build = _build(
        {"type": "text", "text": "PE-CLIP: A Parameter-Efficient Fine-Tuning", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Ibtissam Saadi<sup>?</sup>, Faculty of Graphical Systems, BTU Cottbus", "page_idx": 0},
        {"type": "text", "text": long_abstract, "page_idx": 0},
        {"type": "text", "text": "ACM Reference Format:", "text_level": 2, "page_idx": 1},
        {"type": "text", "text": "Ibtissam Saadi. 2025. PE-CLIP. 23 pages.", "page_idx": 1},
    )
    citation = extract_citation(build.blocks, build.sections)
    assert citation is not None
    assert "Ibtissam Saadi, Faculty of Graphical Systems" in citation
    assert "has attracted increasing attention" not in citation


def test_stops_at_next_heading() -> None:
    build = _build(
        {"type": "text", "text": "Title", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Jane Doe, Some University, jane@example.edu", "page_idx": 0},
        {"type": "text", "text": "1 Introduction", "text_level": 2, "page_idx": 1},
        {"type": "text", "text": "这段正文不属于作者信息，不该进来。", "page_idx": 1},
    )
    citation = extract_citation(build.blocks, build.sections)
    assert citation is not None
    assert "Jane Doe" in citation
    assert "这段正文" not in citation


def test_skips_footnote_blocks() -> None:
    build = _build(
        {"type": "text", "text": "EmotioNet: An accurate algorithm", "text_level": 1, "page_idx": 0},
        {
            "type": "text",
            "text": "C. Fabian Benitez-Quiroz\\*, Ramprakash Srinivasan\\*, Aleix M. Martinez Dept. ECE OSU",
            "page_idx": 0,
        },
        {"type": "page_footnote", "text": "*These authors contributed equally to this paper.", "page_idx": 0},
        {"type": "text", "text": "1 Introduction", "text_level": 2, "page_idx": 1},
    )
    citation = extract_citation(build.blocks, build.sections)
    assert citation is not None
    assert "contributed equally" not in citation
    # 转义残留要清掉
    assert "Benitez-Quiroz*," in citation
    assert "\\*" not in citation


def test_skips_footnote_marked_with_sup() -> None:
    """以 <sup> 标记开头的块是脚注（"These authors contributed equally"），不是作者。"""
    build = _build(
        {"type": "text", "text": "EmotioNet: An accurate algorithm", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "C. Fabian Benitez-Quiroz, Aleix M. Martinez Dept. ECE OSU", "page_idx": 0},
        {
            "type": "text",
            "text": "<sup>?</sup>These authors contributed equally to this paper.",
            "page_idx": 0,
        },
    )
    citation = extract_citation(build.blocks, build.sections)
    assert citation is not None
    assert "contributed equally" not in citation
    assert "Benitez-Quiroz" in citation


def test_standalone_date_line_supplies_the_year() -> None:
    """没有 ACM Reference Format 时，前言里那行 "May 20, 2025" 提供年份。"""
    build = _build(
        {"type": "text", "text": "AN EVALUATION OF A VQA STRATEGY", "text_level": 1, "page_idx": 0},
        {
            "type": "text",
            "text": "Modesto Castrillón-Santana SIANI - Universidad de Las Palmas de Gran Canaria Spain modesto@ulpgc.es",
            "page_idx": 0,
        },
        {"type": "text", "text": "May 20, 2025", "page_idx": 0},
    )
    citation = extract_citation(build.blocks, build.sections)
    assert citation is not None
    assert citation.endswith(" · 2025.")
    # 日期行本身不该混进作者段
    assert "May 20" not in citation


def test_cleans_html_tags_and_broken_url() -> None:
    build = _build(
        {"type": "text", "text": "Title", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Ibtissam Saadi<sup>?</sup>, BTU Cottbus, issaadi@b-tu.de", "page_idx": 0},
    )
    citation = extract_citation(build.blocks, build.sections)
    assert citation is not None
    assert "<sup>" not in citation
    assert "Ibtissam Saadi, BTU Cottbus" in citation


def test_returns_fallback_when_no_authors() -> None:
    build = _build(
        {"type": "text", "text": "Title", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "短", "page_idx": 0},
    )
    assert extract_citation(build.blocks, build.sections, fallback="备用串") == "备用串"


def test_year_only_when_no_venue_phrase() -> None:
    """没有 "In Proceedings" 的期刊式引用，至少把年份留下。"""
    build = _build(
        {"type": "text", "text": "Title", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Jane Doe, Some University, jane@example.edu", "page_idx": 0},
        {"type": "text", "text": "ACM Reference Format:", "text_level": 2, "page_idx": 1},
        {"type": "text", "text": "Jane Doe. 2025. Some Title. 1, 1 (March 2025), 23 pages.", "page_idx": 1},
    )
    citation = extract_citation(build.blocks, build.sections)
    assert citation is not None
    assert citation.endswith(" · 2025.")
