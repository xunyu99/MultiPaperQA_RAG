"""关键词通道（FTS5 + trigram）—— 不联网、不碰 storage。"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from app.config import Settings
from app.db.connection import connect, init_db, transaction
from app.db.repositories import blocks as blocks_repo
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import papers as papers_repo
from app.db.repositories import sections as sections_repo
from app.ingest import converter, section_tree
from app.ingest.indexer import index_paper
from app.retrieval import keyword_index


@pytest.fixture()
def conn() -> sqlite3.Connection:
    connection = connect(":memory:")
    init_db(connection)
    yield connection
    connection.close()


def _item(chunk_id: str, text: str, paper_id: str = "p1", **meta: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "index_text": text,
        "metadata": {"paper_id": paper_id, "section_type": None, "chunk_type": "text", **meta},
    }


# ---------------------------------------------------------------- 查询构造
def test_to_match_query_quotes_every_token() -> None:
    """每个词单独加引号：挡住 FTS5 保留字符，trigram 下还等价于短语匹配。"""
    # 整串短语排在最前，后面是各词 —— 见 to_match_query 的 docstring
    assert keyword_index.to_match_query("PE-CLIP adapter") == (
        '"PE-CLIP adapter" OR "PE-CLIP" OR "adapter"'
    )
    assert keyword_index.to_match_query("3.1") == '"3.1"'
    assert keyword_index.to_match_query('what is "PE-CLIP"?') == (
        '"what is  PE-CLIP ?" OR "what" OR "PE-CLIP"'
    )


def test_term_phrase_ranks_the_exact_chunk_first(conn: sqlite3.Connection) -> None:
    """用户复制的术语短语应该精确命中，而不是被按词 OR 冲淡。

    实测数据：字面 "adapter tuning" 全库只有 1 个 chunk，但按词 OR 会命中 49 个。
    """
    keyword_index.replace_paper(
        conn,
        "p1",
        [
            _item("p1:c1", "we use adapter modules and prompt tuning in this work"),
            _item("p1:c2", "the adapter tuning strategy is described as follows"),
        ],
    )
    assert keyword_index.search(conn, "adapter tuning")[0].chunk_id == "p1:c2"


def test_to_match_query_gives_up_on_short_or_empty_input() -> None:
    """trigram 至少要 3 个字符 —— 全是短词时直接返回 None，别发一个必然为空的查询。"""
    assert keyword_index.to_match_query("") is None
    assert keyword_index.to_match_query("???") is None
    assert keyword_index.to_match_query("a b") is None
    assert keyword_index.to_match_query("F1") is None


# ---------------------------------------------------------------- 召回
def test_english_keyword_finds_the_right_chunk(conn: sqlite3.Connection) -> None:
    keyword_index.replace_paper(
        conn,
        "p1",
        [
            _item("p1:c1", "PE-CLIP: adapter tuning for dynamic facial expression recognition"),
            _item("p1:c2", "the dataset contains fifteen million image-text pairs"),
        ],
    )
    assert [hit.chunk_id for hit in keyword_index.search(conn, "adapter tuning")] == ["p1:c1"]


def test_chinese_text_is_searchable(conn: sqlite3.Connection) -> None:
    """**这条是选 trigram 的理由本身**：默认的 unicode61 下这里必挂。

    unicode61 会把一整句中文当成一个 token，"微表情"永远查不到；trigram 按 3 字符
    滑窗建索引，所以中文论文（以及中文关键词）也能走这条路。
    """
    keyword_index.replace_paper(
        conn,
        "cn",
        [
            _item("cn:c1", "我们提出了一种微表情识别方法，并做了消融实验", paper_id="cn"),
            _item("cn:c2", "本文的数据集来自公开来源", paper_id="cn"),
        ],
    )
    assert [hit.chunk_id for hit in keyword_index.search(conn, "微表情识别")] == ["cn:c1"]


def test_short_acronym_is_a_known_blind_spot(conn: sqlite3.Connection) -> None:
    """"F1" 这种两个字符的词查不到 —— trigram 的硬边界，不是 bug。

    这类查询靠向量通道兜；真需要关键词命中，得另加一条 LIKE 兜底或换分词器。
    """
    keyword_index.replace_paper(conn, "p1", [_item("p1:c1", "F1 score on DFEW dataset")])
    assert keyword_index.search(conn, "F1") == []
    assert [hit.chunk_id for hit in keyword_index.search(conn, "DFEW")] == ["p1:c1"]


def test_punctuation_in_the_question_does_not_raise(conn: sqlite3.Connection) -> None:
    """用户问句里的引号/括号/星号会让 FTS5 直接 syntax error —— 必须挡住。"""
    keyword_index.replace_paper(conn, "p1", [_item("p1:c1", "adapter tuning for FER")])
    for query in ['what is "PE-CLIP"?', "PE-CLIP (2025): adapter*", "???", "", "a", "-"]:
        keyword_index.search(conn, query)  # 不抛异常就算过
    assert keyword_index.search(conn, "???") == []


def test_search_filters_by_paper_and_section(conn: sqlite3.Connection) -> None:
    keyword_index.replace_paper(
        conn, "p1", [_item("p1:c1", "adapter tuning details", paper_id="p1", section_type="method")]
    )
    keyword_index.replace_paper(
        conn,
        "p2",
        [_item("p2:c1", "adapter tuning results", paper_id="p2", section_type="experiment")],
    )
    assert [hit.chunk_id for hit in keyword_index.search(conn, "adapter", paper_ids=["p2"])] == [
        "p2:c1"
    ]
    assert [
        hit.chunk_id for hit in keyword_index.search(conn, "adapter", section_type="method")
    ] == ["p1:c1"]


def test_replace_paper_is_idempotent(conn: sqlite3.Connection) -> None:
    """重索引同一篇论文不能留重复行 —— 跟 chunks / 向量一样的先删后写。"""
    items = [_item("p1:c1", "adapter tuning"), _item("p1:c2", "datasets")]
    assert keyword_index.replace_paper(conn, "p1", items) == 2
    assert keyword_index.replace_paper(conn, "p1", items) == 2
    assert keyword_index.count(conn, "p1") == 2
    assert keyword_index.count(conn) == 2


def test_empty_index_text_is_skipped(conn: sqlite3.Connection) -> None:
    """什么都没有的 chunk 不写进 FTS（向量那边也是这么处理的）。"""
    items = [_item("p1:c1", "adapter"), _item("p1:c2", "   ")]
    assert keyword_index.replace_paper(conn, "p1", items) == 1
    assert keyword_index.count(conn, "p1") == 1


# ---------------------------------------------------------------- 与入库打通
def test_index_paper_fills_the_keyword_index(conn: sqlite3.Connection) -> None:
    """切分入库时 FTS 跟 chunks 一起写（同一个事务），行数必须对得上。"""
    entries = [
        {"type": "text", "text": "标题", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "1 Method", "text_level": 2, "page_idx": 1},
        {"type": "text", "text": "我们用一个 adapter 模块做参数高效微调。" * 6, "page_idx": 1},
    ]
    raw = converter.to_blocks("p1", entries)
    tree = section_tree.build_sections("p1", raw)
    with transaction(conn):
        papers_repo.upsert(conn, {"paper_id": "p1", "title": "标题"})
        sections_repo.upsert(conn, tree.sections)
        blocks_repo.upsert(conn, tree.blocks)

    settings = Settings(_env_file=None, chunk_target_tokens=100, chunk_max_tokens=200, chunk_min_tokens=10)
    stats = index_paper(conn, "p1", settings, write_vectors=False)

    assert stats["fts_rows"] == chunks_repo.count(conn, "p1")
    assert keyword_index.count(conn, "p1") == chunks_repo.count(conn, "p1")
