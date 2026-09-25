"""Step 3 验收：切分规则。"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.ingest import converter, section_tree
from app.ingest.assets import build_assets
from app.ingest.splitter import split_paper
from app.ingest.splitter import ChunkDraft, Unit, _render_content, _split_long_unit
from app.ingest.tokenizer import estimate_tokens


def test_author_contact_lines_stay_out_of_index_text() -> None:
    """作者联系块留 content、不进 index_text（2026-09-25 决定，方案 b）。

    复刻真实数据里的形状：作者联系信息**自己是一个长块**（pe-clip:b00009，632 字符），
    因为超过 300 字符的长度阈值被 `_author_blocks` 当成正文收住，于是漏进索引。
    这里的判据是"块里有邮箱"，content 必须原样保留 —— 作者问题仍然要答得出来。
    """
    entries = [
        {"type": "text", "text": "测试论文标题", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Abstract", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "这是摘要正文，讲的是方法。" * 6, "page_idx": 0},
        {
            "type": "text",
            "text": "Authors' Contact Information: 张三, zhang@example.edu, 某大学; 李四, li@example.com, 某大学",
            "page_idx": 0,
        },
    ]
    build = _split("p-author", entries)
    content = "\n".join(chunk["content"] for chunk in build.chunks)
    indexed = "\n".join(chunk["index_text"] for chunk in build.chunks)

    assert "zhang@example.edu" in content and "li@example.com" in content
    assert "zhang@example.edu" not in indexed and "li@example.com" not in indexed
    assert "Authors' Contact Information" not in indexed
    assert "这是摘要正文" in indexed  # 摘要本体没被误删


def _settings(**kw) -> Settings:
    base = {
        "chunk_target_tokens": 100,
        "chunk_max_tokens": 200,
        "chunk_min_tokens": 40,
        "block_overlap": 0,
        "text_overlap_chars": 20,
    }
    base.update(kw)
    return Settings(**base)


_PLAIN = "这是一段正文，用来撑长度。" * 4


def _paper_entries() -> list[dict]:
    return [
        {"type": "text", "text": "测试论文标题", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "张三 某大学 zhang@example.edu", "page_idx": 0},
        {"type": "text", "text": "Abstract", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "摘要正文。" * 8, "page_idx": 0},
        {"type": "text", "text": "1 Introduction", "text_level": 2, "page_idx": 1},
        {"type": "text", "text": _PLAIN, "page_idx": 1},
        {
            "type": "table",
            "table_caption": ["Table 1: 主实验结果"],
            "table_body": "<table><tr><td>Method</td><td>Accuracy</td></tr><tr><td>Ours</td><td>0.908</td></tr></table>",
            "page_idx": 1,
        },
        {"type": "text", "text": "As shown in Table 1, our method is competitive.", "page_idx": 1},
        {"type": "text", "text": "2 Related Work", "text_level": 2, "page_idx": 2},
        {"type": "text", "text": _PLAIN, "page_idx": 2},
        {"type": "text", "text": "References", "text_level": 2, "page_idx": 3},
        {"type": "ref_text", "text": "[1] Someone et al. 2020. A paper about things.", "page_idx": 3},
        {"type": "page_number", "text": "1234", "page_idx": 3},
    ]


def _split(paper_id: str, entries: list[dict], settings: Settings | None = None):
    raw = converter.to_blocks(paper_id, entries)
    build = section_tree.build_sections(paper_id, raw)
    asset_rows = build_assets(paper_id, build.blocks)
    return split_paper(
        paper_id=paper_id,
        paper_title="测试论文标题",
        blocks=build.blocks,
        sections=build.sections,
        assets=asset_rows,
        settings=settings or _settings(),
    )


# ----------------------------------------------------------------------
# 基本结构
# ----------------------------------------------------------------------
def test_chunk_ids_are_deterministic_and_prefixed() -> None:
    first = _split("p1", _paper_entries())
    second = _split("p1", _paper_entries())
    assert [c["chunk_id"] for c in first.chunks] == [c["chunk_id"] for c in second.chunks]
    assert all(c["chunk_id"].startswith("p1:c") for c in first.chunks)
    assert first.chunks[0]["chunk_id"] == "p1:c0001"


def test_order_index_is_contiguous() -> None:
    build = _split("p1", _paper_entries())
    assert [c["order_index"] for c in build.chunks] == list(range(1, len(build.chunks) + 1))


def test_heading_is_in_index_text_not_in_content() -> None:
    """标题两边都有：index_text 给 BM25/向量匹配，content 给模型看懂上下文。"""
    build = _split("p1", _paper_entries())
    content = "\n".join(c["content"] for c in build.chunks)
    index_text = "\n".join(c["index_text"] for c in build.chunks)

    assert "## 1 Introduction" in content, "模型要知道这段属于哪一节"
    assert "1 Introduction" in index_text, "BM25 要能匹配章节标题"


def test_index_text_carries_paper_title_and_section_path() -> None:
    build = _split("p1", _paper_entries())
    joined = "\n".join(c["index_text"] for c in build.chunks)
    assert "[测试论文标题 > 1 Introduction]" in joined


# ----------------------------------------------------------------------
# 哪些内容不进 chunk
# ----------------------------------------------------------------------
def test_noise_and_reference_blocks_do_not_enter_chunks() -> None:
    build = _split("p1", _paper_entries())
    all_block_ids = {bid for c in build.chunks for bid in c["block_ids"]}
    raw = converter.to_blocks("p1", _paper_entries())
    excluded = {
        b["block_id"] for b in raw if b["block_type"] in ("noise", "reference", "title")
    }
    assert not (all_block_ids & excluded)
    joined = "\n".join(c["content"] for c in build.chunks)
    assert "1234" not in joined
    assert "Someone et al." not in joined


def test_author_lines_do_not_enter_chunks() -> None:
    """作者行（姓名+单位+邮箱）不进索引，但**摘要要留下**。

    实测作者行占这些块 15%-34% 的 token，只是稀释语义（BACKLOG B20）。
    危险点是：摘要正文常常也落在 preamble 里，所以必须按块排除、不能按章节排除。
    """
    build = _split(
        "p1",
        [
            {"type": "text", "text": "测试论文标题", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "张三 某大学 zhang@example.edu", "page_idx": 0},
            {"type": "text", "text": "李四 另一所大学 lisi@example.edu", "page_idx": 0},
            # 摘要正文没有标题，落在同一个 preamble 里 —— 这块必须留下
            {"type": "text", "text": "微表情识别很难，因为动作细微且持续时间极短，需要专门的方法。" * 3, "page_idx": 0},
            {"type": "text", "text": "1 Introduction", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": _PLAIN, "page_idx": 1},
        ],
    )
    joined = "\n".join(c["content"] for c in build.chunks)
    assert "zhang@example.edu" not in joined, "作者行不该进 chunk"
    assert "lisi@example.edu" not in joined
    assert "微表情识别很难" in joined, "落在 preamble 里的摘要必须保留"


def test_no_chunk_contains_more_than_one_paper() -> None:
    """论文边界是结构保证的：split_paper 每次只吃一篇论文的 blocks。"""
    p1 = _split("p1", _paper_entries())
    p2 = _split("p2", _paper_entries())
    assert all(c["chunk_id"].startswith("p1:") for c in p1.chunks)
    assert all(c["chunk_id"].startswith("p2:") for c in p2.chunks)


# ----------------------------------------------------------------------
# 资产
# ----------------------------------------------------------------------
def test_asset_appears_as_reference_block_in_content() -> None:
    build = _split("p1", _paper_entries())
    chunk = next(c for c in build.chunks if any(a.endswith("table0001") for a in c["_asset_ids"]))
    # content 里只有占位符，caption / 表头都由代码按 asset_id 回 assets 表取
    assert "[TABLE_REF asset_id=p1:table0001]" in chunk["content"]
    assert "Caption:" not in chunk["content"]
    assert "<td>" not in chunk["content"]  # 本体不进 chunk


def test_index_text_has_no_asset_id_or_html() -> None:
    build = _split("p1", _paper_entries())
    index_text = "\n".join(c["index_text"] for c in build.chunks)
    assert "asset_id" not in index_text
    assert "<td>" not in index_text
    # 表注 + 表头留下，数值不留
    assert "Table 1: 主实验结果" in index_text
    assert "Method | Accuracy" in index_text
    assert "0.908" not in index_text


def test_assets_are_registered() -> None:
    build = _split("p1", _paper_entries())
    assert [a["asset_id"] for a in build.assets] == ["p1:table0001"]


# ----------------------------------------------------------------------
# 尺寸约束
# ----------------------------------------------------------------------
def test_token_counts_stay_within_bounds() -> None:
    settings = _settings()
    build = _split("p1", _paper_entries(), settings)
    for chunk in build.chunks:
        assert chunk["token_count"] <= settings.chunk_max_tokens, chunk["chunk_id"]
    # 允许最后一个 chunk 是真的没法再合并的碎片
    assert sum(1 for c in build.chunks if c["token_count"] < settings.chunk_min_tokens) <= 1


def test_long_text_is_split_at_sentence_boundaries() -> None:
    settings = _settings(chunk_target_tokens=60, chunk_max_tokens=120, chunk_min_tokens=10)
    # 30 遍只有约 100 token，够不到 max；这里要给到明显超过 max 的长度
    build = _split(
        "p1",
        [
            {"type": "text", "text": "标题", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "第一句。" * 80, "page_idx": 0},
        ],
        settings,
    )
    assert len(build.chunks) > 1, "超长正文段应该被切开"
    for chunk in build.chunks:
        assert chunk["token_count"] <= settings.chunk_max_tokens
    # overlap 只在切开处出现：末尾那句话会在下一片的开头重复
    assert build.chunks[0]["content"].rstrip().endswith("第一句。")


def test_oversized_asset_becomes_its_own_chunk() -> None:
    """表格本体再大也不切 —— 它只影响引用块（caption+preview），所以 chunk 不会爆。"""
    huge_table = "<table>" + "<tr><td>x</td><td>y</td></tr>" * 400 + "</table>"
    build = _split(
        "p1",
        [
            {"type": "text", "text": "标题", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "1 Method", "text_level": 2, "page_idx": 1},
            {"type": "table", "table_caption": ["Table 1: 大表"], "table_body": huge_table, "page_idx": 1},
        ],
        _settings(chunk_max_tokens=100),
    )
    chunk = next(c for c in build.chunks if "[TABLE_REF" in c["content"])
    assert chunk["token_count"] <= 100
    assert len(build.assets[0]["raw_content"]) > 10000


# ----------------------------------------------------------------------
# 交叉引用
# ----------------------------------------------------------------------
_TABLE_HTML = "<table><tr><td>Method</td><td>Accuracy</td></tr></table>"


def _split_mention(
    mention_text: str,
    asset_entries: list[dict],
    settings: Settings | None = None,
):
    """正文里写 mention_text，前面先放若干资产块 —— 看会不会被挂上。

    chunk_min_tokens 压到 1：不然资产的引用块（只有十几 token）会被"补足"并进
    后面那个 chunk，资产和提及就同 chunk 了，测不出交叉引用这条路。
    """
    entries = [
        {"type": "text", "text": "标题", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "1 Method", "text_level": 2, "page_idx": 1},
        *asset_entries,
        {"type": "text", "text": _PLAIN, "page_idx": 1},
        {"type": "text", "text": "2 Results", "text_level": 2, "page_idx": 2},
        {"type": "text", "text": mention_text, "page_idx": 2},
    ]
    if settings is None:
        settings = _settings(chunk_target_tokens=60, chunk_max_tokens=200, chunk_min_tokens=1)
    return _split("p1", entries, settings)


def test_body_mention_attaches_the_asset() -> None:
    """正文写 "As shown in Table 1" 时，即使表在别的 chunk 里也要挂上。"""
    build = _split_mention(
        "As shown in Table 1, the accuracy improves.",
        [
            {
                "type": "table",
                "table_caption": ["Table 1: 主实验结果"],
                "table_body": _TABLE_HTML,
                "page_idx": 1,
            }
        ],
    )
    mentioning = next(c for c in build.chunks if "As shown in Table 1" in c["content"])
    assert "p1:table0001" in mentioning["_asset_ids"]
    # 走的是交叉引用这条路：表本体不在这个 chunk 的块列表里
    assert build.assets[0]["block_id"] not in mentioning["block_ids"]


def test_roman_numeral_mention_attaches_the_asset() -> None:
    """"TABLE VIII" 和正文里的 "Table 8" 是同一个编号。"""
    build = _split_mention(
        "The prompts are listed in Table 8 for reference.",
        [
            {
                "type": "table",
                "table_caption": ["TABLE VIII: Text prompts"],
                "table_body": _TABLE_HTML,
                "page_idx": 1,
            }
        ],
    )
    mentioning = next(c for c in build.chunks if "Table 8" in c["content"])
    assert "p1:table0001" in mentioning["_asset_ids"]


def test_mention_inside_a_word_does_not_attach() -> None:
    """正文 "notable 8.68%" 里的 "table 8" 不是提及 —— 实测真的挂错过一张表。"""
    build = _split_mention(
        "Our model marks a notable 8.68% improvement over the previous best method.",
        [
            {
                "type": "table",
                "table_caption": ["Table 8: 消融实验"],
                "table_body": _TABLE_HTML,
                "page_idx": 1,
            }
        ],
    )
    mentioning = next(c for c in build.chunks if "notable" in c["content"])
    assert mentioning["_asset_ids"] == []


def test_latex_leq_does_not_attach_a_formula() -> None:
    r"""LaTeX "\leq 3 6 0" 里的 "eq 3" 不是公式引用 —— 实测也挂错过一个公式。"""
    build = _split_mention(
        r"We constrain $\sum_{k} \theta_{ak} \leq 3 6 0^{o}$ for every point.",
        [{"type": "equation", "text": "$$x = y \\tag{3}$$", "page_idx": 1}],
    )
    mentioning = next(c for c in build.chunks if "leq" in c["content"])
    assert mentioning["_asset_ids"] == []


def test_duplicate_label_is_not_guessed() -> None:
    """两个资产都叫 "Table 2" 时号码没有指向性 —— 宁可漏挂，也不挂错。"""
    build = _split_mention(
        "As shown in Table 2, the results improve.",
        [
            {
                "type": "table",
                "table_caption": ["Table 2: 第一张"],
                "table_body": _TABLE_HTML,
                "page_idx": 1,
            },
            {
                "type": "table",
                "table_caption": ["Table 2: 第二张"],
                "table_body": _TABLE_HTML,
                "page_idx": 1,
            },
        ],
    )
    mentioning = next(c for c in build.chunks if "As shown in Table 2" in c["content"])
    assert mentioning["_asset_ids"] == []


# ----------------------------------------------------------------------
# token 估算
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "low", "high"),
    [
        ("hello world", 2, 6),
        ("这是一段中文文本", 6, 12),
        ("", 0, 0),
    ],
)
def test_estimate_tokens_is_reasonable(text: str, low: int, high: int) -> None:
    assert low <= estimate_tokens(text) <= high


# ----------------------------------------------------------------------
# overlap 只在 chunk 边界拼出来（否则同一 chunk 内部会重复）
# ----------------------------------------------------------------------
def _long_unit(settings: Settings) -> Unit:
    # 每句话都不一样 —— 否则 overlap 前缀在正文里天然重复，count() 数不清
    text = "".join(f"这是第 {i} 句话，用来把长度撑起来。" for i in range(1, 61))
    return Unit(
        block_id="p1:b00001",
        order_index=1,
        kind="text",
        content_text=text,
        index_text=text,
        section_id="p1:s001",
        section_path="标题 > 1 Introduction",
        page_start=1,
        page_end=1,
    )


def test_overlap_prefix_is_not_inside_the_piece_itself() -> None:
    settings = _settings(chunk_max_tokens=120, text_overlap_chars=30)
    pieces = _split_long_unit(_long_unit(settings), settings)
    assert len(pieces) >= 2
    assert pieces[1].overlap_prefix, "第二片应该带上上一片的尾句"
    assert pieces[1].overlap_prefix not in pieces[1].content_text


def test_overlap_is_printed_once_when_pieces_share_a_chunk() -> None:
    """两片如果又落回同一个 chunk，重复段不能出现两次（实测踩过的 bug）。"""
    settings = _settings(chunk_max_tokens=120, text_overlap_chars=30)
    pieces = _split_long_unit(_long_unit(settings), settings)
    content = _render_content(ChunkDraft(units=pieces))
    assert content.count(pieces[1].overlap_prefix) == 1


def test_overlap_is_printed_when_piece_starts_a_chunk() -> None:
    settings = _settings(chunk_max_tokens=120, text_overlap_chars=30)
    pieces = _split_long_unit(_long_unit(settings), settings)
    content = _render_content(ChunkDraft(units=[pieces[1]]))
    assert pieces[1].overlap_prefix in content


# ----------------------------------------------------------------------
# 跨章节合并的准入条件
# ----------------------------------------------------------------------
_LONG = "这是一段足够长的正文，用来把章节撑满一个 chunk。" * 12


def test_merge_refuses_cross_parent_sections() -> None:
    """2.1（挂在 2 下面）和 3（顶层）跨父，不能合并 —— 否则 section 没法定位。"""
    settings = _settings(chunk_target_tokens=100, chunk_max_tokens=400, chunk_min_tokens=80)
    build = _split(
        "p1",
        [
            {"type": "text", "text": "标题", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "2 Related Work", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": "2.1 Sub", "text_level": 3, "page_idx": 1},
            {"type": "text", "text": _LONG, "page_idx": 1},
            {"type": "text", "text": "3 Method", "text_level": 2, "page_idx": 2},
            {"type": "text", "text": "很短。", "page_idx": 2},
        ],
        settings,
    )
    # 没有任何一个 chunk 同时包含 "2.1 Sub" 和 "3 Method" 两个标题
    merged = [
        c["chunk_id"] for c in build.chunks
        if "2.1 Sub" in c["content"] and "3 Method" in c["content"]
    ]
    assert not merged, f"出现了跨父合并：{merged}"


def test_merge_allows_sibling_sections() -> None:
    """2.1 和 2.2 同父，可以合并。"""
    settings = _settings(chunk_target_tokens=100, chunk_max_tokens=400, chunk_min_tokens=80)
    build = _split(
        "p1",
        [
            {"type": "text", "text": "标题", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "2 Related Work", "text_level": 2, "page_idx": 1},
            {"type": "text", "text": "2.1 A", "text_level": 3, "page_idx": 1},
            {"type": "text", "text": _LONG, "page_idx": 1},
            {"type": "text", "text": "2.2 B", "text_level": 3, "page_idx": 2},
            {"type": "text", "text": "很短。", "page_idx": 2},
        ],
        settings,
    )
    # 标题只在 index_text 里（content 不放标题），所以按 index_text 判断
    merged = [c for c in build.chunks if "2.1 A" in c["index_text"] and "2.2 B" in c["index_text"]]
    assert merged, "同父兄弟章节应该能合并"
