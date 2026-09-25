"""Step 8 验收：FastAPI 接口（真 SQLite + 真 Chroma，假向量、假 LLM、假 MinerU，不联网）。

测三件事，都按**真实请求路径**走：

1. `/health` 的对账（SQLite 与向量库条数不一致时要能看出来）；
2. 上传 -> 后台解析入库 -> 轮询任务 -> 论文出现在列表里（整条链路的接口层）；
3. `/chat` 的 SSE 协议：先 `evidence`、再若干 `token`、最后 `done`，且 `done` 里带引用校验。

假掉的三样都是**要花钱/要联网**的部分：embedding、LLM、MinerU。其余全是真的 ——
所以帧格式、事务、任务状态这些真会出问题的地方，测试都能抓到。
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app.api import main as api_main
from app.api import jobs as jobs_mod
from app.config import get_settings
from app.db.connection import connect, init_db, transaction
from app.db.repositories import assets as assets_repo
from app.db.repositories import blocks as blocks_repo
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import papers as papers_repo
from app.db.repositories import sections as sections_repo
from app.retrieval import keyword_index
from app.retrieval import vector_index as vector_index_mod

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "mineru_demo"


class FakeEmbedder(Embeddings):
    """固定向量：不联网。排名没有意义，但链路是真的。"""

    def __init__(self, settings=None) -> None:  # noqa: ANN001 - 对齐真 Embedder 的签名
        self.settings = settings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.01] * 8 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.01] * 8


class FakeLLM:
    """假的生成通道：吐带引用的文本，用来验证 SSE 帧和收尾。"""

    def __init__(self, settings=None, *, is_chitchat: bool = False,  # noqa: ANN001
                 pieces: list[str] | None = None) -> None:
        self.settings = settings
        self.is_chitchat = is_chitchat
        # 想演拒答就传 pieces（拒答标记要在第一个增量里）
        self.pieces = pieces or ["这是", "一段回答", "[E1]。"]
        # 记下 chat_stream 收到的 prompt / system，用来断言"闲聊走的是哪套提示词"
        self.stream_calls: list[dict] = []

    def chat(self, prompt: str, *, system=None, model=None, temperature=0.2,  # noqa: ANN001
             max_tokens=None, planner=False) -> str:
        """改写（规划通道）也走同一个 client 类，所以这里得一起假掉 ——
        否则多轮测试会真的去调 qwen-flash。"""
        assert planner is True, "指代改写必须走规划通道"
        return "改写后的独立问句"

    def structured(self, schema, prompt: str, *, system=None, **kwargs):  # noqa: ANN001
        """入口理解（改写 + 闲聊判定 + 资产编号）走结构化输出，同样得假掉。"""
        return schema(rewritten_query="改写后的独立问句", is_chitchat=self.is_chitchat)

    async def chat_stream(self, prompt: str, *, system=None, model=None, temperature=0.2):  # noqa: ANN001
        self.stream_calls.append({"prompt": prompt, "system": system})
        for piece in self.pieces:
            yield piece


class FakeMinerUClient:
    """假装调了 MinerU：直接把仓库里的离线产物拷到目标目录。"""

    def __init__(self, settings=None) -> None:  # noqa: ANN001
        self.settings = settings

    def __enter__(self) -> "FakeMinerUClient":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def parse_pdf(self, pdf: Path, dest: Path, on_tick=None) -> None:  # noqa: ANN001
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(FIXTURE_DIR, dest)
        if on_tick:
            on_tick("done", {})


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把库、向量库、上游三样全指到临时目录 / 假实现上。

    **`get_settings.cache_clear()` 这行不能省**：`get_settings()` 是
    `@lru_cache(maxsize=1)`（见 `app/config.py`），改了环境变量它照样返回上一次的
    Settings —— 于是第二个测试会连到第一个测试的库上，表现为"偏偏一起跑就失败"。
    跑完再清一次，别让临时路径留在缓存里。
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    # 重排走真实接口（百炼），接口层的用例不该为它付费也不该被它拖慢；见 test_rerank.py
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setattr(vector_index_mod, "Embedder", FakeEmbedder)
    monkeypatch.setattr(api_main, "LLMClient", FakeLLM)
    monkeypatch.setattr("app.ingest.pipeline.MinerUClient", FakeMinerUClient)
    get_settings.cache_clear()

    try:
        with TestClient(api_main.app) as test_client:
            yield test_client
    finally:
        get_settings.cache_clear()


def _seed_one_paper(paper_id: str = "pe-clip") -> None:
    """种一篇最小论文：一个章节、一个 chunk、一条向量。"""
    conn = connect()
    init_db(conn)
    try:
        with transaction(conn):
            papers_repo.upsert(
                conn,
                {"paper_id": paper_id, "title": "PE-CLIP", "citation": "Saadi et al., 2025"},
            )
            sections_repo.upsert(
                conn,
                [
                    {"section_id": f"{paper_id}:s1", "paper_id": paper_id, "parent_id": None,
                     "level": 1, "order_index": 1, "title": "3.1 Overview",
                     "section_type": "method"},
                ],
            )
            blocks_repo.upsert(
                conn,
                [
                    {"block_id": f"{paper_id}:b1", "paper_id": paper_id,
                     "section_id": f"{paper_id}:s1", "order_index": 1, "block_type": "text",
                     "heading_level": None, "text": "正文", "latex": None, "html": None,
                     "caption": None, "page_idx": 1, "bbox": None, "image_path": None},
                ],
            )
            chunks_repo.upsert(
                conn,
                [
                    {"chunk_id": f"{paper_id}:c0001", "paper_id": paper_id, "order_index": 1,
                     "content": "## 3.1 Overview\n\n正文内容", "index_text": "PE-CLIP 方法概述",
                     "block_ids": [f"{paper_id}:b1"], "page_start": 1, "page_end": 1,
                     "chunk_type": "text", "token_count": 20},
                ],
            )
    finally:
        conn.close()

    index = vector_index_mod.ChunkVectorIndex()
    index.reset_paper(paper_id)
    index.add_chunks(
        [
            {"chunk_id": f"{paper_id}:c0001", "index_text": "PE-CLIP 方法概述",
             "metadata": {"chunk_id": f"{paper_id}:c0001", "paper_id": paper_id,
                          "chunk_type": "text"}}
        ]
    )


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 文本切成 [(event, data), ...]，顺便当成帧格式的断言。"""
    events: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_line = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data_line = line[len("data: "):]
        assert event_name and data_line, f"帧格式不对：{block!r}"
        events.append((event_name, json.loads(data_line)))
    return events


def test_health_on_empty_db(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["papers"] == 0
    assert body["chunks"] == body["vectors"] == 0
    assert body["drift"] == 0


def test_health_reports_drift(client: TestClient) -> None:
    """向量库里多出一条孤儿（删论文没删向量）时必须报出来，而不是说一切正常。"""
    _seed_one_paper()
    index = vector_index_mod.ChunkVectorIndex()
    index.add_chunks(
        [
            {"chunk_id": "ghost:c0001", "index_text": "孤儿向量",
             "metadata": {"chunk_id": "ghost:c0001", "paper_id": "ghost"}}
        ]
    )
    body = client.get("/health").json()
    assert body["ok"] is False
    assert body["drift"] == 1


def test_papers_lists_one(client: TestClient) -> None:
    _seed_one_paper()
    body = client.get("/papers").json()
    assert body["total"] == 1
    assert body["items"][0]["paper_id"] == "pe-clip"
    assert body["items"][0]["chunks"] == 1


def test_paper_file_404_when_missing(client: TestClient) -> None:
    _seed_one_paper()
    response = client.get("/papers/pe-clip/file")
    assert response.status_code == 404
    assert "PDF" in response.json()["detail"]


def test_upload_runs_in_background_and_indexes(client: TestClient) -> None:
    """整条上传链路：POST /papers -> 轮询任务 -> 论文可查 -> 向量对上账。"""
    response = client.post(
        "/papers",
        files={"file": ("demo.pdf", b"%PDF-1.4 fake", "application/pdf")},
        params={"paper_id": "demo"},
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]

    deadline = time.time() + 60
    body: dict = {}
    while time.time() < deadline:
        body = client.get(f"/tasks/{task_id}").json()
        if body["status"] in {"done", "failed"}:
            break
        time.sleep(0.2)

    assert body.get("status") == "done", body
    assert {step["status"] for step in body["steps"]} == {"done"}
    assert body["result"]["chunks"] > 0
    assert body["result"]["vectors"] == body["result"]["chunks"]

    assert client.get("/papers").json()["total"] == 1
    assert client.get("/health").json()["ok"] is True


def test_task_404(client: TestClient) -> None:
    response = client.get("/tasks/nope")
    assert response.status_code == 404
    assert "内存" in response.json()["detail"]


def test_chat_streams_evidence_tokens_then_done(client: TestClient) -> None:
    _seed_one_paper()
    with client.stream("POST", "/chat", json={"question": "PE-CLIP 的方法是什么"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())

    events = _parse_sse(text)
    names = [name for name, _ in events]

    # **第一帧是 intent，不是 evidence**：入口理解（改写 / 范围 / 闲聊）做完就先发出去，
    # 前端的面板立刻能更新，不用等检索（那一步慢一个量级）
    assert names[0] == "intent"
    assert names[1] == "evidence"
    assert names[-1] == "done"
    assert names.count("token") == 3

    intent = dict(events)["intent"]
    assert intent["scope"] == ["pe-clip"]
    assert intent["scope_source"] == "question"
    assert intent["standalone"] == "改写后的独立问句"
    assert intent["chitchat"] is False

    evidence = dict(events)["evidence"]
    assert evidence["cards"][0]["chunk_id"] == "pe-clip:c0001"
    assert evidence["cards"][0]["label"] == "E1"

    tokens = "".join(payload["text"] for name, payload in events if name == "token")
    done = dict(events)["done"]
    assert done["answer"] == tokens == "这是一段回答[E1]。"
    assert done["citations"]["cited"] == ["E1"]
    assert done["citations"]["unknown"] == []
    # 档位（Step 9）：有有效引用 → grounded，前端据此不加任何徽标
    assert done["mode"] == "grounded"
    assert done["unknown_reason"] is None


def test_refusal_marker_becomes_unknown_and_strips_citations(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """拒答标记 → `mode=unknown` + reason；**展示的正文里不许留 `[En]`**。

    模型在拒答时引用证据是**实测过的真实行为**（prompt 要求"每个事实都标出处"，
    而"证据只涉及 X 和 Y"是在陈述证据内容），所以展示前剥干净这一道必须有。
    """
    _seed_one_paper()
    fake = FakeLLM(pieces=["[[REFUSE:out_of_scope]]\n证据只涉及 ", "A [E1]", "，没有相关内容。"])
    monkeypatch.setattr(api_main, "LLMClient", lambda settings=None: fake)

    with client.stream("POST", "/chat", json={"question": "红烧肉怎么做"}) as response:
        text = "".join(response.iter_text())
    done = dict(_parse_sse(text))["done"]

    assert done["mode"] == "unknown"
    assert done["unknown_reason"] == "out_of_scope"
    assert done["citations"]["insufficient"] is True
    assert "REFUSE" not in done["answer"] and "[E1]" not in done["answer"]
    # 引用校验看的是**模型原始输出**，所以那里照旧能看到 E1（保留诊断能力）
    assert done["citations"]["cited"] == ["E1"]


def test_chitchat_short_circuits_retrieval(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """闲聊：不检索、不给证据卡、不写引用，改用 chitchat.md 那套提示词接住。

    这条守着三件事：① 不花 embedding / rerank 的钱；② 前端能从 `chitchat` 标志区分；
    ③ 用的是**原话**（寒暄没什么可指代消解的）。
    """
    fake = FakeLLM(None, is_chitchat=True)
    monkeypatch.setattr(api_main, "LLMClient", lambda settings=None: fake)

    with client.stream("POST", "/chat", json={"question": "你好"}) as response:
        text = "".join(response.iter_text())
    events = _parse_sse(text)
    evidence = dict(events)["evidence"]
    done = dict(events)["done"]

    assert evidence["chitchat"] is True
    assert evidence["cards"] == []
    assert done["citations"]["_chitchat"] is True
    # 走的是闲聊提示词，不是 answer.md（那份只会回"资料不足"）
    assert "前台" in fake.stream_calls[0]["system"]
    assert fake.stream_calls[0]["prompt"] == "你好"


def test_chat_says_insufficient_when_nothing_retrieved(client: TestClient) -> None:
    """库里没东西时：不调 LLM，但协议照样走完（前端逻辑统一）。"""
    with client.stream("POST", "/chat", json={"question": "红烧肉怎么做"}) as response:
        text = "".join(response.iter_text())
    events = _parse_sse(text)
    assert [name for name, _ in events] == ["intent", "evidence", "done"]
    assert "资料不足" in dict(events)["done"]["answer"]


def test_chat_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


# ---------------------------------------------------------------- 会话
def test_session_crud(client: TestClient) -> None:
    created = client.post("/sessions", json={}).json()
    session_id = created["session_id"]
    assert created["title"] == ""
    assert created["scope"] == []
    assert created["message_count"] == 0

    renamed = client.patch(f"/sessions/{session_id}", json={"title": "改名了"}).json()
    assert renamed["title"] == "改名了"

    # 一条消息都没有的会话**默认不出现在历史里**：前端是懒创建，
    # 空壳出现在列表里只会让人以为"我建过一堆没用的东西"
    assert client.get("/sessions").json()["total"] == 0
    listed = client.get("/sessions", params={"include_empty": "true"}).json()
    assert [item["session_id"] for item in listed["items"]] == [session_id]

    detail = client.get(f"/sessions/{session_id}").json()
    assert detail["messages"] == []

    assert client.delete(f"/sessions/{session_id}").json()["deleted"] == 1
    assert client.get("/sessions").json()["total"] == 0


def test_session_shows_up_after_the_first_exchange(client: TestClient) -> None:
    """说过话的会话才进历史列表（这是懒创建的配套规则）。"""
    _seed_one_paper()
    session_id = client.post("/sessions", json={}).json()["session_id"]
    assert client.get("/sessions").json()["total"] == 0

    with client.stream("POST", "/chat", json={"question": "PE-CLIP 的方法", "session_id": session_id}) as response:
        "".join(response.iter_text())

    listed = client.get("/sessions").json()
    assert [item["session_id"] for item in listed["items"]] == [session_id]
    assert listed["items"][0]["message_count"] == 2
    assert listed["items"][0]["title"] == "PE-CLIP 的方法"


def test_session_404s(client: TestClient) -> None:
    assert client.get("/sessions/nope").status_code == 404
    assert client.patch("/sessions/nope", json={"title": "x"}).status_code == 404
    assert client.delete("/sessions/nope").status_code == 404


def test_session_title_comes_from_first_question(client: TestClient) -> None:
    _seed_one_paper()
    session_id = client.post("/sessions", json={}).json()["session_id"]

    with client.stream(
        "POST", "/chat", json={"question": "PE-CLIP 的方法是什么", "session_id": session_id}
    ) as response:
        "".join(response.iter_text())

    detail = client.get(f"/sessions/{session_id}").json()
    assert detail["title"] == "PE-CLIP 的方法是什么"
    assert detail["message_count"] == 2
    assert [item["role"] for item in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["content"] == "PE-CLIP 的方法是什么"
    assert detail["messages"][1]["citations"]["cited"] == ["E1"]
    # 证据卡快照要一起存：刷新页面后历史消息还得能显示来源、还得能点开原文
    snapshot = detail["messages"][1]["citations"]["_cards"]
    assert snapshot[0]["label"] == "E1"
    assert snapshot[0]["chunk_id"] == "pe-clip:c0001"
    assert "trace" in snapshot[0]


def test_chat_with_unknown_session_is_404(client: TestClient) -> None:
    """会话 id 不存在就明确报错，别悄悄当成无状态请求 —— 那会让前端"消息丢了"找不到原因。"""
    response = client.post("/chat", json={"question": "随便问", "session_id": "s-nope"})
    assert response.status_code == 404


def test_chat_scope_priority_end_to_end(client: TestClient) -> None:
    """范围链走一遍：勾选 → 问句压过勾选 → 继承 → 取消勾选回全库。

    这一串就是用户在界面上会做的事，四条规则一次跑全：
    ① 勾了 pe-clip → 就搜 pe-clip（explicit）
    ② 勾着 pe-clip 但问句点名 emotionet → **问句赢**（question），前端跟着换勾选
    ③ 不传 paper_ids → 继承上一轮记下的范围（sticky）
    ④ 传空数组 = 取消勾选/全库 → **显式全库**，不许 sticky 粘回来
    """
    _seed_one_paper("pe-clip")
    _seed_one_paper("emotionet")
    session_id = client.post("/sessions", json={}).json()["session_id"]

    def ask(question: str, paper_ids: list[str] | None = None) -> dict:
        body: dict = {"question": question, "session_id": session_id}
        if paper_ids is not None:
            body["paper_ids"] = paper_ids
        with client.stream("POST", "/chat", json=body) as response:
            text = "".join(response.iter_text())
        return dict(_parse_sse(text))["evidence"]

    ev = ask("这篇论文的方法是什么", ["pe-clip"])
    assert (ev["scope"], ev["scope_source"]) == (["pe-clip"], "explicit")

    ev = ask("emotionet 是怎么标注的", ["pe-clip"])
    assert (ev["scope"], ev["scope_source"]) == (["emotionet"], "question")

    ev = ask("它的数据集呢")
    assert (ev["scope"], ev["scope_source"]) == (["emotionet"], "sticky")

    ev = ask("随便问问", [])
    assert (ev["scope"], ev["scope_source"]) == ([], "all")


def test_chat_inherits_sticky_scope_and_records_it(client: TestClient) -> None:
    """多轮：第一轮定了范围，第二轮追问（问句里没有论文名）要继承下来。"""
    _seed_one_paper()
    session_id = client.post("/sessions", json={}).json()["session_id"]

    for question in ["PE-CLIP 的方法是什么", "它的 batch size 是多少"]:
        with client.stream("POST", "/chat", json={"question": question, "session_id": session_id}) as response:
            text = "".join(response.iter_text())

    events = _parse_sse(text)
    evidence = dict(events)["evidence"]
    done = dict(events)["done"]
    # 第二轮：问句里没有论文名，范围靠 sticky 继承第一轮的 pe-clip
    assert evidence["scope"] == ["pe-clip"]
    assert evidence["scope_source"] == "sticky"
    assert evidence["standalone"] == "改写后的独立问句"   # 走了指代改写
    assert evidence["rewritten"] is True
    assert done["session_id"] == session_id

    detail = client.get(f"/sessions/{session_id}").json()
    assert detail["message_count"] == 4
    last = detail["messages"][-1]
    assert last["citations"]["_scope_source"] == "sticky"
    assert last["citations"]["_standalone"] == "改写后的独立问句"


def test_stateless_chat_does_not_touch_sessions(client: TestClient) -> None:
    """不传 session_id = 老行为：一问一答，不落库。"""
    _seed_one_paper()
    with client.stream("POST", "/chat", json={"question": "PE-CLIP 的方法"}) as response:
        "".join(response.iter_text())
    assert client.get("/sessions").json()["total"] == 0


def test_root_redirects_to_ui(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/ui/"


def test_ui_serves_index_and_assets(client: TestClient) -> None:
    """前端页面和它的两个静态文件都要能取到（挂载路径写错就是整页 404）。"""
    page = client.get("/ui/")
    assert page.status_code == 200
    assert "论文阅读助手" in page.text
    assert "/ui/app.js" in page.text or 'src="app.js"' in page.text

    for asset, needle in [("/ui/app.js", "parseFrame"), ("/ui/app.css", ".card")]:
        response = client.get(asset)
        assert response.status_code == 200, asset
        assert needle in response.text, asset
        # 静态资源不许被强缓存：否则改了 app.js，浏览器还在跑旧 JS，
        # 表现为"后端改了、前端没反应"，非常难查（踩过一次）
        assert response.headers["cache-control"] == "no-cache", asset


def test_every_element_id_used_by_js_exists_in_html() -> None:
    """`el("health")` 写错一个字母的话，页面只是**静默不工作**，不报错。

    所以静态地查一遍：JS 里引用的每个 id，HTML 里都必须有。
    """
    import re

    html = (api_main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    js = (api_main.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r'\bel\("([^"]+)"\)', js))
    assert used, "没从 JS 里抓到任何 id，脚本或正则要改"
    assert used <= html_ids, f"JS 用到但 HTML 里没有：{sorted(used - html_ids)}"


def test_citation_lookup_is_scoped_to_its_own_message() -> None:
    """多轮里编号会重复（每条回答都有自己的 E1/E2…），所以引用定位**必须限定在它所在的那条消息**。

    曾经的写法是"全页按 `data-label` 找、取最后一张" —— 点旧回答的 `[E3]` 会跳到**最新那条**的
    卡片上（用户报过这个 bug）。JS 没有测试运行器（见本模块开头），所以只能用静态锚点钉住
    关键调用形状：改了这两处就该改这个测试，而不是让 bug 悄悄回来。
    """
    js = (api_main.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'cite.closest(".msg")' in js, "点引用时要把'这条消息'当查找范围传下去"
    assert "openCard(card, node)" in js, "点卡片要激活被点的那一个，不能按编号全页找"


def test_css_colors_live_in_variables() -> None:
    """**换配色只该改 `app.css` 顶部的调色板。**

    所以规则部分（`:root` 之后）不许出现硬编码颜色 —— 一旦有人"顺手写个 #ccc"，
    换主题时就会漏掉那一处，而且很难找。要新颜色就在调色板/语义层加变量。
    """
    import re

    css = (api_main.STATIC_DIR / "app.css").read_text(encoding="utf-8")
    start = css.index(":root {")
    end = css.index("\n}", start)          # 调色板 + 语义变量块的结尾
    rules = css[end:]
    found = sorted(set(re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", rules)))
    assert not found, f"规则里出现硬编码颜色：{found}（请改成语义变量）"

    # 调色板必须还在，且语义层有 accent —— 不写死具体配色名，
    # 否则每次换主题都要改测试（第一版就写死了 --color-iris，换橘粉时挂掉）
    palette = set(re.findall(r"(--[a-z-]+):", css[start:end]))
    neutral = {"--color-white", "--color-ink", "--color-mist", "--color-line",
               "--color-amber", "--color-amber-soft", "--color-moss"}
    theme_colors = {name for name in palette if name.startswith("--color-")} - neutral
    assert len(theme_colors) >= 3, f"调色板至少要 3 个主题色，现在是 {sorted(theme_colors)}"
    assert "--accent" in palette and "--tint-a" in palette


# ---------------------------------------------------------------- 删除论文
def test_delete_paper_clears_index_and_keeps_parse_cache(client: TestClient) -> None:
    """删一篇：向量 / 关键词 / SQLite 全清，**parse_cache 保留**（PLAN §2.3）。

    删除顺序在 `indexer.delete_paper` 里定死（先向量 → 再 FTS → 最后 SQLite），
    这里验的是结果：删干净了、别的论文没被带下水、该留的还在。
    """
    _seed_one_paper("pe-clip")
    _seed_one_paper("emotionet")

    # 塞两行"该留下来 / 该删掉"的样本：parse_cache 保留，FTS 镜像要删
    conn = connect()
    init_db(conn)
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO parse_cache (content_hash, mineru_version, paper_id, mineru_dir) "
                "VALUES (?, ?, ?, ?)",
                ("hash-pe-clip", "v1", "pe-clip", "storage/mineru/pe-clip"),
            )
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, index_text, paper_id, section_type, chunk_type) "
                "VALUES (?, ?, ?, ?, ?)",
                ("pe-clip:c0001", "PE-CLIP 方法概述", "pe-clip", "method", "text"),
            )
            # 另一篇也要有 FTS 行 —— /health 是三方对账（chunks == vectors == fts），
            # 夹具缺一行会让 ok=False，那是夹具不自洽，不是删除的锅
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, index_text, paper_id, section_type, chunk_type) "
                "VALUES (?, ?, ?, ?, ?)",
                ("emotionet:c0001", "EmotioNet 标注流程", "emotionet", "method", "text"),
            )
    finally:
        conn.close()

    body = client.delete("/papers/pe-clip").json()
    assert body["removed"] == 1
    assert body["chunks"] == 1
    assert body["vectors"] == 1
    assert body["keyword_rows"] == 1

    conn = connect()
    init_db(conn)
    try:
        assert papers_repo.get(conn, "pe-clip") is None
        assert chunks_repo.count(conn, "pe-clip") == 0
        assert keyword_index.count(conn, "pe-clip") == 0
        # 外键级联：子表跟着 papers 一起没
        for table in ("sections", "blocks", "assets"):
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE paper_id = ?", ("pe-clip",)
            ).fetchone()[0]
            assert count == 0, table
        # parse_cache 与 storage/mineru/{paper_id}/ 是**刻意保留**的
        kept = conn.execute(
            "SELECT COUNT(*) FROM parse_cache WHERE paper_id = ?", ("pe-clip",)
        ).fetchone()[0]
        assert kept == 1
        # 另一篇一点没动
        assert chunks_repo.count(conn, "emotionet") == 1
    finally:
        conn.close()

    assert vector_index_mod.ChunkVectorIndex().ids_of_paper("pe-clip") == []
    assert [p["paper_id"] for p in client.get("/papers").json()["items"]] == ["emotionet"]
    # 对账：SQLite 的 chunk 数必须还等于向量数（/health 靠的就是这个）
    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["chunks"] == health["vectors"]
    assert health["fts"] == health["chunks"]


def test_delete_unknown_paper_is_404(client: TestClient) -> None:
    assert client.delete("/papers/nope").status_code == 404


def test_asset_image_serves_the_file_and_rejects_unknown(client: TestClient) -> None:
    """证据卡缩略图用的取图接口：能取到真图，不知道的 id 一律 404。"""
    from PIL import Image

    _seed_one_paper("pe-clip")
    settings = get_settings()
    target = Path(settings.storage_dir) / "mineru" / "pe-clip" / "images"
    target.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(target / "fig.png")

    conn = connect()
    init_db(conn)
    try:
        with transaction(conn):
            assets_repo.upsert(
                conn,
                [{"asset_id": "pe-clip:figure0001", "block_id": "pe-clip:b1",
                  "paper_id": "pe-clip", "asset_type": "figure",
                  "image_path": "images/fig.png", "caption": "Figure 1"}],
            )
    finally:
        conn.close()

    response = client.get("/assets/pe-clip:figure0001/image")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")

    assert client.get("/assets/nope/image").status_code == 404


def test_delete_refuses_while_that_paper_is_ingesting(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """正在入库的论文不许删。

    否则会留下"papers 没了、blocks 还在"的半篇论文，而且后台那个任务还会继续往库里写 ——
    这种状态比"删不掉"难收拾得多。
    """
    _seed_one_paper("pe-clip")
    busy = SimpleNamespace(paper_id="pe-clip", status="running", task_id="t-123")
    monkeypatch.setattr(jobs_mod.REGISTRY, "list", lambda: [busy])

    response = client.delete("/papers/pe-clip")
    assert response.status_code == 409
    assert "正在入库" in response.json()["detail"]
    assert "t-123" in response.json()["detail"]
    # 拒绝就得真没动它
    assert client.get("/papers").json()["total"] == 1
