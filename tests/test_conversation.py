"""多轮编排：指代改写 + 范围解析链（不联网，不调真模型）。

这里测的是**判断逻辑**，不是模型质量：改写模型好不好用要拿真 key 跑（见 PLAN 的评测），
但"什么时候该改写""范围该听谁的""改写失败怎么办"这三件事必须能离线测。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.config import Settings
from app.db.connection import connect, init_db, transaction
from app.db.repositories import papers as papers_repo
from app.db.repositories import sessions as sessions_repo
from app.generation.conversation import (
    SCOPE_ALL,
    SCOPE_EXPLICIT,
    SCOPE_QUESTION,
    SCOPE_STICKY,
    prepare_turn,
    resolve_scope_chain,
    save_turn,
)
from app.generation.intent import SYSTEM_PROMPT, QueryIntent, parse_intent
from app.generation.rewrite import format_history


class FakeIntentLLM:
    """假的入口模型：返回预设的结构化理解，并记下被问了什么。"""

    def __init__(self, intent: QueryIntent | None = None, **fields) -> None:
        self.intent = intent or QueryIntent(
            rewritten_query="改写后的独立问句", **fields
        )
        self.calls: list[dict] = []

    def structured(self, schema, prompt, *, system=None, **kwargs):  # noqa: ANN001
        self.calls.append({"prompt": prompt, "system": system, **kwargs})
        return self.intent


class BrokenIntentLLM:
    def structured(self, schema, prompt, **kwargs):  # noqa: ANN001
        raise RuntimeError("网络断了")


class FakePlanner(FakeIntentLLM):
    """旧名字的兼容壳：后面的用例只关心"改写后是什么"，不关心走哪种调用。"""

    def __init__(self, reply: str = "改写后的独立问句") -> None:
        super().__init__(QueryIntent(rewritten_query=reply))


@pytest.fixture()
def conn() -> sqlite3.Connection:
    connection = connect(":memory:")
    init_db(connection)
    with transaction(connection):
        papers_repo.upsert(connection, {"paper_id": "pe-clip", "title": "PE-CLIP"})
        papers_repo.upsert(connection, {"paper_id": "emotionet", "title": "EmotioNet"})
    yield connection
    connection.close()


# ---------------------------------------------------------------- 入口理解（一次结构化调用）
def test_first_turn_also_calls_for_intent() -> None:
    """第一轮**也要调**：指代改写确实不需要，但"这句是不是闲聊"和"点名了表几"同样要判。"""
    llm = FakeIntentLLM(**{"is_chitchat": True})
    intent = parse_intent("你好", [], llm=llm)
    assert llm.calls, "第一轮不能省掉这次调用"
    assert intent.is_chitchat is True


def test_intent_prompt_carries_history_and_question() -> None:
    llm = FakeIntentLLM(QueryIntent(rewritten_query="PE-CLIP 的消融实验说明了什么？"))
    history = [
        {"role": "user", "content": "PE-CLIP 的方法是什么"},
        {"role": "assistant", "content": "PE-CLIP 用了两个适配器…"},
    ]
    intent = parse_intent("它的消融实验呢", history, llm=llm)
    assert intent.rewritten_query == "PE-CLIP 的消融实验说明了什么？"
    assert "PE-CLIP 的方法是什么" in llm.calls[0]["prompt"]
    assert "它的消融实验呢" in llm.calls[0]["prompt"]
    assert llm.calls[0]["system"] == SYSTEM_PROMPT


def test_intent_failure_falls_back_to_the_question() -> None:
    """结构化解析失败**不能让这一轮挂掉** —— 退回原问题、什么都不判。"""
    intent = parse_intent("它呢", [{"role": "user", "content": "上一句"}], llm=BrokenIntentLLM())
    assert intent.rewritten_query == "它呢"
    assert intent.is_chitchat is False
    assert intent.asset_type is None


def test_intent_rejects_an_answer_instead_of_a_question() -> None:
    """模型偶尔会"顺手回答"——改写字段长得离谱就当它没改写，用原问题。"""
    llm = FakeIntentLLM(QueryIntent(rewritten_query="PE-CLIP 的 batch size 是 8。" * 20))
    intent = parse_intent("它呢", [{"role": "user", "content": "上一句"}], llm=llm)
    assert intent.rewritten_query == "它呢"


def test_intent_cleans_prefixes_and_quotes() -> None:
    llm = FakeIntentLLM(QueryIntent(rewritten_query='改写结果："PE-CLIP 的 batch size 是多少"'))
    intent = parse_intent("它呢", [{"role": "user", "content": "上一句"}], llm=llm)
    assert intent.rewritten_query == "PE-CLIP 的 batch size 是多少"


def test_prepare_turn_resolves_asset_target_from_the_model(conn: sqlite3.Connection) -> None:
    """模型点名"表 1" → 直接算出要注入的 chunk；模型没点名 → 留空给正则兜底。"""
    from app.db.repositories import assets as assets_repo
    from app.db.repositories import blocks as blocks_repo
    from app.db.repositories import chunks as chunks_repo

    with transaction(conn):
        blocks_repo.upsert(
            conn,
            [{"block_id": "pe-clip:b1", "paper_id": "pe-clip", "section_id": None, "order_index": 1,
              "block_type": "table", "heading_level": None, "text": None, "latex": None,
              "html": "<table></table>", "caption": "Table 1: x", "page_idx": 1, "bbox": None,
              "image_path": None}],
        )
        chunks_repo.upsert(
            conn,
            [{"chunk_id": "pe-clip:c0001", "paper_id": "pe-clip", "order_index": 1,
              "content": "[TABLE_REF asset_id=pe-clip:table0001]", "index_text": "Table 1: x",
              "block_ids": ["pe-clip:b1"], "page_start": 1, "page_end": 1, "chunk_type": "table"}],
        )
        assets_repo.upsert(
            conn,
            [{"asset_id": "pe-clip:table0001", "block_id": "pe-clip:b1", "paper_id": "pe-clip",
              "asset_type": "table", "caption": "Table 1: x", "label_norm": "table:1"}],
        )

    llm = FakeIntentLLM(
        QueryIntent(rewritten_query="PE-CLIP 的表 1 说了什么", asset_type="table", asset_number="1")
    )
    turn = prepare_turn(conn, "表 1 说了什么", paper_id="pe-clip", llm=llm)
    assert turn.injected == ["pe-clip:c0001"]

    # 模型没点名 → 留空（retriever 的正则还会再兜一次底）
    turn2 = prepare_turn(conn, "表 1 说了什么", paper_id="pe-clip", llm=FakeIntentLLM())
    assert turn2.injected == []


def test_format_history_marks_roles_and_truncates() -> None:
    history = [
        {"role": "user", "content": "问题一"},
        {"role": "assistant", "content": "答" * 500},
    ]
    text = format_history(history)
    assert text.startswith("用户：问题一")
    assert "助手：" in text
    assert len(text) < 700  # 助手那条被截到 300


def test_format_history_keeps_only_the_last_two_rounds() -> None:
    """窗口是 2 轮 = 4 条消息，更早的话题不占位置。"""
    history = [
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "第二轮问题"},
        {"role": "assistant", "content": "第二轮回答"},
        {"role": "user", "content": "第三轮问题"},
        {"role": "assistant", "content": "第三轮回答"},
    ]
    text = format_history(history)
    assert "第一轮" not in text
    assert "第二轮问题" in text and "第三轮回答" in text


def test_history_skips_empty_assistant_messages(conn: sqlite3.Connection) -> None:
    """空的助手回复不占窗口 —— 实测库里 9 条助手消息有 6 条是空的。

    丢掉之后还要能凑满 2 轮，所以取数时留了缓冲（见 sessions._HISTORY_BUFFER）。
    窗口按"最近 rounds 个提问"切，第二轮的答案虽然丢了，提问本身仍是追问的锚点。
    """
    session = sessions_repo.create(conn)
    sid = session["session_id"]
    with transaction(conn):
        sessions_repo.append_exchange(conn, sid, question="第一轮问题", answer="第一轮回答")
        sessions_repo.append_exchange(conn, sid, question="第二轮问题", answer="")
        sessions_repo.append_exchange(conn, sid, question="第三轮问题", answer="第三轮回答")

    items = sessions_repo.history(conn, sid, rounds=2)
    contents = [item["content"] for item in items]
    assert "" not in contents
    assert contents == ["第二轮问题", "第三轮问题", "第三轮回答"]
    assert contents[0].endswith("问题"), "窗口不能从一条孤立的助手回复开始"


def test_history_skips_failed_answer_marked_in_citations(conn: sqlite3.Connection) -> None:
    """标了 `_failed` 的助手回复（生成失败兜底文案）同样不该进改写历史。"""
    session = sessions_repo.create(conn)
    sid = session["session_id"]
    with transaction(conn):
        sessions_repo.append_exchange(
            conn, sid, question="失败的那轮", answer="生成失败：模型没有返回内容，请重试。",
            citations={"_failed": True},
        )
        sessions_repo.append_exchange(conn, sid, question="正常的一轮", answer="正常回答")

    contents = [item["content"] for item in sessions_repo.history(conn, sid, rounds=2)]
    assert "失败的那轮" in contents
    assert all("生成失败" not in c for c in contents)


def test_history_skips_chitchat_rounds(conn: sqlite3.Connection) -> None:
    """闲聊轮进历史只会带偏指代消解："它呢？"不该去闲聊里找指代对象。

    用户那句话留着（不影响），闲聊的助手回复必须挡掉。
    """
    session = sessions_repo.create(conn)
    sid = session["session_id"]
    with transaction(conn):
        sessions_repo.append_exchange(
            conn, sid, question="你好", answer="你好！我是这个论文库的问答助手。",
            citations={"_chitchat": True},
        )
        sessions_repo.append_exchange(conn, sid, question="PE-CLIP 的方法", answer="它用了 adapter")

    contents = [item["content"] for item in sessions_repo.history(conn, sid, rounds=2)]
    assert "你好" in contents
    assert all("我是这个论文库" not in c for c in contents)
    assert "它用了 adapter" in contents


# ---------------------------------------------------------------- 范围链
def test_question_beats_explicit_selection(conn: sqlite3.Connection) -> None:
    """问句里点名了论文 → 它压过面板上的勾选（用户这一轮改主意了）。"""
    scope, source = resolve_scope_chain(
        conn,
        "emotionet 是怎么标注的",
        explicit_paper_id="pe-clip",
        sticky=["pe-clip"],
    )
    assert (scope, source) == (["emotionet"], SCOPE_QUESTION)


def test_explicit_selection_wins_when_question_names_nothing(conn: sqlite3.Connection) -> None:
    """问句里没提论文 → 听面板的勾选。"""
    scope, source = resolve_scope_chain(
        conn, "随便问", explicit_paper_id="emotionet", sticky=["pe-clip"]
    )
    assert (scope, source) == (["emotionet"], SCOPE_EXPLICIT)


def test_explicit_empty_selection_means_all_and_clears_sticky(conn: sqlite3.Connection) -> None:
    """取消勾选 / 点全库 = **显式全库**：不许 sticky 把上一轮的范围粘回来。

    前端现在总是发 `paper_ids`（空数组也发），所以这里能区分"没给"和"给了空"。
    """
    scope, source = resolve_scope_chain(conn, "随便问", explicit_paper_ids=[], sticky=["pe-clip"])
    assert (scope, source) == ([], SCOPE_ALL)
    # 对照：真"没给"（None）时，上一轮的范围照样继承
    assert resolve_scope_chain(conn, "随便问", explicit_paper_ids=None, sticky=["pe-clip"]) == (
        ["pe-clip"],
        SCOPE_STICKY,
    )


def test_question_beats_sticky(conn: sqlite3.Connection) -> None:
    """用户改口问另一篇时，问句本身比"上一轮的范围"可靠。"""
    scope, source = resolve_scope_chain(
        conn, "emotionet 是怎么标注的", sticky=["pe-clip"]
    )
    assert (scope, source) == (["emotionet"], SCOPE_QUESTION)


def test_sticky_used_when_question_has_no_paper_name(conn: sqlite3.Connection) -> None:
    """追问"它的消融实验呢"里没有论文名 —— 不能退回全库。"""
    scope, source = resolve_scope_chain(conn, "它的消融实验呢", sticky=["pe-clip"])
    assert (scope, source) == (["pe-clip"], SCOPE_STICKY)


def test_sticky_drops_papers_that_are_gone(conn: sqlite3.Connection) -> None:
    """粘住的范围里若有已下架的论文，过滤掉；全没了就退回全库。"""
    scope, source = resolve_scope_chain(conn, "它呢", sticky=["已删除的论文"])
    assert (scope, source) == ([], SCOPE_ALL)


def test_all_when_nothing_matches(conn: sqlite3.Connection) -> None:
    assert resolve_scope_chain(conn, "红烧肉怎么做") == ([], SCOPE_ALL)


def test_selecting_every_paper_means_the_whole_library(conn: sqlite3.Connection) -> None:
    """前端"全选"会勾满所有论文 —— 必须归一成全库。

    不归一的话，5 篇会走多篇配额（每篇 PER_PAPER_K 条）一次返回 15 条证据；
    而用户的意思是"不限定"。
    """
    every = ["pe-clip", "emotionet"]
    assert resolve_scope_chain(conn, "随便问", explicit_paper_ids=every) == ([], SCOPE_ALL)

    # 少选一篇仍然是"限定这几篇"，不能被误归一
    scope, source = resolve_scope_chain(conn, "随便问", explicit_paper_ids=["pe-clip"])
    assert (scope, source) == (["pe-clip"], SCOPE_EXPLICIT)


def test_selecting_every_paper_when_only_one_exists(conn: sqlite3.Connection) -> None:
    """库里只有一篇时不能归一 —— 那本来就是"就看这一篇"。"""
    with transaction(conn):
        conn.execute("DELETE FROM papers WHERE paper_id = 'emotionet'")
    scope, source = resolve_scope_chain(conn, "随便问", explicit_paper_ids=["pe-clip"])
    assert (scope, source) == (["pe-clip"], SCOPE_EXPLICIT)


# ---------------------------------------------------------------- 编排
def test_prepare_turn_rewrites_and_sticks(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        session = sessions_repo.create(conn, paper_id="pe-clip")
        sessions_repo.append_exchange(
            conn, session["session_id"], question="PE-CLIP 的方法是什么", answer="两个适配器"
        )

    turn = prepare_turn(
        conn,
        "它的消融实验呢",
        session_id=session["session_id"],
        settings=Settings(_env_file=None),
        # 故意让改写结果里**不带论文名**（模型完全可能这么输出）——
        # 这时候就该靠 sticky 兜住，而不是退回全库
        llm=FakePlanner("这篇论文的消融实验说明了什么？"),
    )
    assert turn.standalone == "这篇论文的消融实验说明了什么？"
    assert turn.rewritten is True
    assert turn.scope == ["pe-clip"]
    assert turn.scope_source == SCOPE_STICKY

    saved = sessions_repo.get(conn, session["session_id"])
    assert saved["scope_paper_id"] == "pe-clip"  # 范围记回了会话
    assert saved["message_count"] == 2


def test_rewrite_that_names_a_paper_wins_over_sticky(conn: sqlite3.Connection) -> None:
    """改写结果里带了另一篇论文名 —— 以问句为准，sticky 让位。"""
    with transaction(conn):
        session = sessions_repo.create(conn, paper_id="pe-clip")
        sessions_repo.append_exchange(conn, session["session_id"], question="问", answer="答")

    turn = prepare_turn(
        conn,
        "它呢",
        session_id=session["session_id"],
        settings=Settings(_env_file=None),
        llm=FakePlanner("emotionet 的训练集是什么"),
    )
    assert turn.scope == ["emotionet"]
    assert turn.scope_source == SCOPE_QUESTION


def test_prepare_turn_without_session_is_stateless(conn: sqlite3.Connection) -> None:
    turn = prepare_turn(
        conn, "pe-clip 的 batch size", settings=Settings(_env_file=None), llm=FakePlanner()
    )
    assert turn.session_id is None
    assert turn.scope == ["pe-clip"]
    assert turn.scope_source == SCOPE_QUESTION
    assert turn.history == []


def test_save_turn_keeps_debug_metadata(conn: sqlite3.Connection) -> None:
    """落库时把"实际用的问句 + 范围 + 来源"存进 citations —— 排查串论文靠它。"""
    with transaction(conn):
        session = sessions_repo.create(conn)
        # 先有一轮历史，改写才会真的发生（第一轮不需要指代消解）
        sessions_repo.append_exchange(conn, session["session_id"], question="上一句", answer="上一答")

    turn = prepare_turn(
        conn,
        "它呢",
        session_id=session["session_id"],
        settings=Settings(_env_file=None),
        llm=FakePlanner("PE-CLIP 是什么"),
    )
    save_turn(conn, turn, answer="答案", citations={"cited": ["E1"]})

    messages = sessions_repo.list_messages(conn, session["session_id"])
    assert [item["role"] for item in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[2]["content"] == "它呢"                      # 存的是用户原话
    assert messages[3]["citations"]["_standalone"] == "PE-CLIP 是什么"
    assert messages[3]["citations"]["_scope_source"] == SCOPE_QUESTION
    assert messages[3]["citations"]["cited"] == ["E1"]           # 原有字段没被覆盖


def test_save_turn_without_session_does_nothing(conn: sqlite3.Connection) -> None:
    turn = prepare_turn(
        conn, "随便问", settings=Settings(_env_file=None), llm=FakePlanner()
    )
    save_turn(conn, turn, answer="答案", citations={})
    assert sessions_repo.list_recent(conn) == []
