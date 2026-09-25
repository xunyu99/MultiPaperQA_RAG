"""Step 8 · FastAPI：把 CLI 那套链路包成 HTTP 服务（Step 8c 之后又加了会话和三栏前端）。

启动（**必须单 worker**，见 `app/api/jobs.py` 里任务表那段）：

    .venv\\Scripts\\python.exe -m uvicorn app.api.main:app --port 8000

打开 http://127.0.0.1:8000/docs 就能直接调。

接口一览：

| 方法 | 路径 | 做什么 |
| --- | --- | --- |
| GET | `/health` | 库里几篇论文 / 几个 chunk / 向量几条，**顺便对账** |
| GET | `/papers` | 论文列表 |
| POST | `/papers` | 上传 PDF -> 后台解析入库，立刻返回 `task_id` |
| GET | `/tasks/{task_id}` | 查入库进度（解析 / 切分两步） |
| POST | `/chat` | 提问 -> **SSE 流式**回答 + 证据 + 引用坐标 |
| GET | `/papers/{paper_id}/file` | 原始 PDF（前端用 PDF.js 渲染 + 按 bbox 高亮） |
| POST/GET/PATCH/DELETE | `/sessions...` | 会话 CRUD；会话里的消息带证据卡快照 |

前端（`/ui`）还调这些：`/sessions` 列表与详情、以及转义后的静态资源。

几个刻意的决定：

1. **上传是异步的**（`202` + 轮询）。MinerU 一篇要几分钟，同步等必然超时。
2. **`/chat` 先发证据再发字**。前端能在模型还在生成时就把来源渲染出来，
   而不是等整段回答完。协议见 `app/api/chat.py`。
3. **`/health` 带对账**。SQLite 和 Chroma 的条数不一致是最常见的事故
   （入库中途崩过、删论文没删向量），这里一眼能看出来，不用另写脚本。
4. **会话是可选的**。带 `session_id` 就落库 + 继承上一轮范围（sticky scope）；
   不带就是无状态的一问一答，范围全靠请求自己带。
5. **档位随 `done` 帧一起发**（`mode` / `unknown_reason`）。证据帧先发时还没有结论，
   所以那里给的是中性空报告 —— 别拿它当"这条是拒答"。
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api import jobs as jobs_mod
from app.api.chat import card_payload, citation_payload, sse_frame, stream_events
from app.config import Settings, get_settings
from app.db.connection import connect, init_db, transaction
from app.db.repositories import assets as assets_repo
from app.db.repositories import chunks as chunks_repo
from app.db.repositories import papers as papers_repo
from app.db.repositories import sessions as sessions_repo
from app.generation.answerer import (
    EMPTY_ANSWER_FALLBACK,
    MODE_UNKNOWN,
    REASON_CHITCHAT,
    REASON_FAILED,
    REASON_NOT_IN_CORPUS,
    CitationReport,
    assess,
    build_user_content,
    load_chitchat_prompt,
    load_system_prompt,
)
from app.generation.conversation import prepare_turn, save_turn
from app.generation.evidence import build_cards
from app.ingest import indexer
from app.ingest.versioning import default_paper_id
from app.providers.llm import LLMClient
from app.retrieval import keyword_index
from app.retrieval.retriever import retrieve
from app.retrieval.vector_index import ChunkVectorIndex


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    session_id: str | None = Field(
        default=None,
        description="多轮会话 id。给了就落库、并继承上一轮的论文范围；不给就是无状态的一问一答",
    )
    paper_id: str | None = Field(default=None, description="只看某一篇（不给就按问题里的论文名自动解析）")
    paper_ids: list[str] | None = Field(
        default=None,
        description=(
            "只看这几篇（跨论文对比用）。**空列表 = 显式全库**（用户取消勾选/点了全库），"
            "此时不再继承上一轮的范围；不给（null）才轮到继承"
        ),
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="覆盖默认配额，调试用")


class SessionCreate(BaseModel):
    title: str = Field(default="", description="不给就用第一个问题自动生成")
    paper_id: str | None = None
    paper_ids: list[str] | None = None


class SessionRename(BaseModel):
    title: str = Field(min_length=1)


def _settings() -> Settings:
    return get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时把库和向量索引准备好（顺便让明显的问题在启动阶段就暴露，而不是等第一次请求）。"""
    conn = connect()
    init_db(conn)
    conn.close()
    _settings().ensure_dirs()
    yield


app = FastAPI(
    title="MultiPaperQA_RAG · 多论文问答",
    description="MinerU 解析 -> 切分 -> 向量检索 -> 带引用的回答（v1 单用户 / 多篇论文）",
    lifespan=lifespan,
)

# 前端在别的端口跑（比如 vite 的 5173），浏览器会拦跨域请求。
# v1 是本机演示、无鉴权，所以放全开；**生产要收紧成白名单**。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端：一个静态页，没有构建步骤（npm / vite 那套对演示是负担）。
# 挂在 `/ui`，根路径 302 过去，这样 `--port 8000` 打开就是界面。
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


@app.middleware("http")
async def _no_cache_for_static(request: Any, call_next: Any) -> Any:
    """静态资源禁用**强缓存**。

    不然会出现这种很难查的现象：改了 `app.js`、后端也重启了，回答的行为变了，
    但浏览器还在跑缓存里那份旧 JS（前端面板不同步）—— 看着像"前端没实现"。

    用 `no-cache` 而不是 `no-store`：浏览器仍会带 `If-None-Match` 回来校验，
    文件没变就是 304，代价可以忽略。
    """
    response = await call_next(request)
    if request.url.path.startswith("/ui"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# 中间有反向代理时别缓存、别缓冲，否则"流式"会变成"一次性蹦出来"
STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


# ----------------------------------------------------------------------
# 健康检查 / 对账
# ----------------------------------------------------------------------
@app.get("/health", summary="健康检查 + SQLite 与向量库对账")
def health() -> dict[str, Any]:
    settings = _settings()
    conn = connect()
    try:
        init_db(conn)
        rows = [
            {"paper_id": row["paper_id"], "title": row["title"], "chunks": chunks_repo.count(conn, row["paper_id"])}
            for row in papers_repo.list_all(conn)
        ]
        sqlite_chunks = sum(item["chunks"] for item in rows)
        index = ChunkVectorIndex(settings)
        vectors = index.count()
        # 关键词索引在同一个库里，但也可能落后（老库升级后没重跑切分 → 表是空的）
        fts_rows = keyword_index.count(conn)
    finally:
        conn.close()

    return {
        "ok": sqlite_chunks == vectors == fts_rows,
        "papers": len(rows),
        "chunks": sqlite_chunks,
        # 对不上就是出过事（入库中途崩过 / 删论文没删向量），这条比 ok=false 更有用
        "vectors": vectors,
        "drift": vectors - sqlite_chunks,
        # 关键词索引：老库升级后没重跑切分的话这里是 0，而不是报错
        "fts": fts_rows,
        "top_k": settings.top_k,
        "per_paper_k": settings.per_paper_k,
        "section_cap": settings.section_cap,
        "embedding_model": settings.embedding_model,
        "chunk_params": {
            "target": settings.chunk_target_tokens,
            "max": settings.chunk_max_tokens,
            "min": settings.chunk_min_tokens,
        },
        "detail": rows,
    }


@app.get("/papers", summary="论文列表")
def list_papers() -> dict[str, Any]:
    conn = connect()
    try:
        init_db(conn)
        items = []
        for row in papers_repo.list_all(conn):
            paper = dict(row)
            paper["chunks"] = chunks_repo.count(conn, paper["paper_id"])
            items.append(paper)
    finally:
        conn.close()
    return {"total": len(items), "items": items}


@app.delete("/papers/{paper_id}", summary="删除一篇论文（向量 + 关键词 + SQLite）")
def delete_paper(paper_id: str) -> dict[str, Any]:
    """删一篇论文。**只删被点名的那一篇**，删什么、留什么见 `indexer.delete_paper`。

    两条拒绝的情况都在动手之前判掉：

    - 库里没有这篇 → 404；
    - 这篇**正在入库** → 409。边入库边删会留下"papers 没了、blocks 还在"的半篇论文，
      而且后台那个任务还会继续往库里写 —— 等它跑完再删。

    历史会话不受影响（`sessions` 刻意不做指向 `papers` 的外键）：会话里的范围会在
    下一轮被 `resolve_scope_chain` 自动过滤掉，历史回答也原样留着；只是点卡片看原文
    会 404 —— 前端会把它显示成"这篇论文已删除"。
    """
    settings = _settings()
    busy = [
        job
        for job in jobs_mod.REGISTRY.list()
        if job.paper_id == paper_id
        and job.status in (jobs_mod.STATUS_QUEUED, jobs_mod.STATUS_RUNNING)
    ]
    if busy:
        raise HTTPException(
            status_code=409,
            detail=f"这篇论文正在入库（任务 {busy[0].task_id}），等它跑完再删",
        )

    conn = connect()
    try:
        init_db(conn)
        if papers_repo.get(conn, paper_id) is None:
            raise HTTPException(status_code=404, detail=f"没有这篇论文：{paper_id}")
        return indexer.delete_paper(conn, paper_id, settings=settings)
    finally:
        conn.close()


@app.get("/assets/{asset_id}/image", summary="资产原图（证据卡缩略图）")
def asset_image(asset_id: str) -> FileResponse:
    """单张资产图。前端缩略图要用 —— `/papers/{id}/file` 给的是整个 PDF，取不了单张。

    图片路径是从 `assets.image_path` 现拼的（相对 `storage/mineru/{paper_id}/`），
    跟生成侧发多模态图用的是同一份数据。**不做路径穿越**：只接受库里的相对路径，
    拼完再确认它确实在 `storage/mineru/` 底下。
    """
    settings = _settings()
    conn = connect()
    try:
        init_db(conn)
        row = assets_repo.get(conn, asset_id)
    finally:
        conn.close()
    if row is None or not row["image_path"]:
        raise HTTPException(status_code=404, detail=f"没有这张图：{asset_id}")

    root = (Path(settings.storage_dir) / "mineru").resolve()
    path = (root / row["paper_id"] / str(row["image_path"])).resolve()
    if not str(path).startswith(str(root)) or not path.is_file():
        raise HTTPException(status_code=404, detail=f"图片不在磁盘上：{asset_id}")
    # media_type 交给 FileResponse 按扩展名猜（.jpg / .png 都有）
    return FileResponse(path)


@app.get("/papers/{paper_id}/file", summary="原始 PDF（前端 PDF.js 渲染 + 按 bbox 高亮）")
def paper_file(paper_id: str) -> FileResponse:
    conn = connect()
    try:
        init_db(conn)
        row = papers_repo.get(conn, paper_id)
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"没有这篇论文：{paper_id}")

    pdf_path = row["pdf_path"]
    if not pdf_path or not Path(pdf_path).is_file():
        # 服务端不渲染页面图（那要装 PyMuPDF），直接给原始 PDF 让前端渲染。
        # 库里存的是**上传时**的路径，论文挪过位置就会走到这里。
        raise HTTPException(status_code=404, detail=f"原始 PDF 不在磁盘上：{pdf_path}")
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"{paper_id}.pdf")


# ----------------------------------------------------------------------
# 上传（异步）
# ----------------------------------------------------------------------
@app.post("/papers", status_code=202, summary="上传 PDF，后台解析入库（返回 task_id）")
async def upload_paper(
    file: UploadFile = File(..., description="论文 PDF"),
    paper_id: str | None = Query(default=None, description="不给我就从文件名推"),
) -> dict[str, Any]:
    settings = _settings()
    settings.ensure_dirs()

    filename = Path(file.filename or "upload.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只收 PDF")

    resolved_id = paper_id or default_paper_id(Path(filename))
    dest = settings.storage_dir / "pdfs" / f"{resolved_id}.pdf"
    dest.write_bytes(await file.read())

    job = jobs_mod.REGISTRY.submit(
        paper_id=resolved_id,
        filename=filename,
        pdf_path=str(dest),
        steps=["parse", "index"],
        run=jobs_mod.run_ingest_job,
    )
    return {
        "task_id": job.task_id,
        "paper_id": job.paper_id,
        "status": job.status,
        "hint": f"轮询 GET /tasks/{job.task_id} 看进度；同一份 PDF 再传会命中 parse_cache，不会重复调 MinerU",
    }


@app.get("/tasks/{task_id}", summary="查入库进度")
def get_task(task_id: str) -> dict[str, Any]:
    job = jobs_mod.REGISTRY.get(task_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="没有这个任务（任务表只在内存里：进程重启就没了，多 worker 之间也不共享）",
        )
    return {
        "task_id": job.task_id,
        "paper_id": job.paper_id,
        "filename": job.filename,
        "status": job.status,
        "steps": [{"name": step.name, "status": step.status, "detail": step.detail} for step in job.steps],
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
    }


# ----------------------------------------------------------------------
# 会话（多轮）
# ----------------------------------------------------------------------
def _session_payload(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session["session_id"],
        "title": session["title"],
        "scope": session.get("scope_paper_ids") or ([session["scope_paper_id"]] if session.get("scope_paper_id") else []),
        "message_count": session.get("message_count") or 0,
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
    }


@app.post("/sessions", status_code=201, summary="新建会话")
def create_session(payload: SessionCreate) -> dict[str, Any]:
    conn = connect()
    try:
        init_db(conn)
        with transaction(conn):
            session = sessions_repo.create(
                conn,
                title=payload.title,
                paper_id=payload.paper_id,
                paper_ids=payload.paper_ids,
            )
    finally:
        conn.close()
    return _session_payload(session)


@app.get("/sessions", summary="历史会话（按最近使用排，默认不含空会话）")
def list_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    include_empty: bool = Query(default=False, description="把一条消息都没有的会话也列出来"),
) -> dict[str, Any]:
    conn = connect()
    try:
        init_db(conn)
        items = [
            _session_payload(row)
            for row in sessions_repo.list_recent(conn, limit, include_empty=include_empty)
        ]
    finally:
        conn.close()
    return {"total": len(items), "items": items}


@app.get("/sessions/{session_id}", summary="会话详情（含全部消息）")
def get_session(session_id: str) -> dict[str, Any]:
    conn = connect()
    try:
        init_db(conn)
        session = sessions_repo.get(conn, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"没有这个会话：{session_id}")
        messages = sessions_repo.list_messages(conn, session_id)
    finally:
        conn.close()
    return {**_session_payload(session), "messages": messages}


@app.patch("/sessions/{session_id}", summary="改会话标题")
def rename_session(session_id: str, payload: SessionRename) -> dict[str, Any]:
    conn = connect()
    try:
        init_db(conn)
        if sessions_repo.get(conn, session_id) is None:
            raise HTTPException(status_code=404, detail=f"没有这个会话：{session_id}")
        with transaction(conn):
            sessions_repo.rename(conn, session_id, payload.title)
        session = sessions_repo.get(conn, session_id)
    finally:
        conn.close()
    return _session_payload(session)


@app.delete("/sessions/{session_id}", summary="删会话（消息级联删除）")
def delete_session(session_id: str) -> dict[str, Any]:
    conn = connect()
    try:
        init_db(conn)
        with transaction(conn):
            removed = sessions_repo.delete(conn, session_id)
    finally:
        conn.close()
    if not removed:
        raise HTTPException(status_code=404, detail=f"没有这个会话：{session_id}")
    return {"deleted": removed, "session_id": session_id}


# ----------------------------------------------------------------------
# 提问（SSE 流式）
# ----------------------------------------------------------------------
@app.post("/chat", summary="提问，SSE 流式返回证据 + 回答 + 引用")
async def chat(request: ChatRequest) -> StreamingResponse:
    settings = _settings()

    # 会话不存在就直接 404 —— **必须在开流之前**。流一旦开出去，HTTP 状态码就改不了了，
    # 只能发一个 error 事件；而且后面的 append_exchange 会撞外键，报错会晚到没法收场。
    if request.session_id:
        conn = connect()
        try:
            init_db(conn)
            if sessions_repo.get(conn, request.session_id) is None:
                raise HTTPException(status_code=404, detail=f"没有这个会话：{request.session_id}")
        finally:
            conn.close()

    def build_turn() -> Any:
        """入口理解：一次 qwen-flash 调用（指代改写 / 闲聊判定 / 资产编号）+ 范围解析。

        **只理解、不检索**：这一步决定"搜哪儿、问什么、是不是闲聊"，做完就把 intent 帧
        发给前端（面板立刻更新），不用等后面那步慢一个量级的检索。
        """
        conn = connect()
        try:
            init_db(conn)
            return prepare_turn(
                conn,
                request.question,
                session_id=request.session_id,
                paper_id=request.paper_id,
                paper_ids=request.paper_ids,
                settings=settings,
                # 从这儿传进去（而不是让 prepare_turn 自己 new）：测试里替换 client 只要改一处，
                # 也避免"改写"和"生成"各建一个连接池
                llm=LLMClient(settings),
            )
        finally:
            conn.close()

    # 理解这一步在开流之前就跑掉：它的结果（范围/改写/闲聊）是 intent 帧的内容
    turn = await run_in_threadpool(build_turn)

    def build_retrieval(turn: Any) -> dict[str, Any]:
        """检索：同步阻塞（embedding 走网络 + 关键词 + rerank），丢线程池里跑。"""
        conn = connect()
        try:
            init_db(conn)
            stats: dict = {}
            if turn.is_chitchat:
                # 闲聊短路：**不检索**（不花 embedding、不花 rerank），证据卡给空列表。
                # 前端靠 evidence 帧里的 `chitchat` 标志区分这一轮。
                return {"turn": turn, "cards": [], "hits": [], "timing": {}, "chitchat": True}
            hits = retrieve(
                conn,
                turn.standalone,
                k=request.top_k,
                index=ChunkVectorIndex(settings),
                settings=settings,
                # 范围已经在这条链里定完了（显式 > 问句 > sticky > 全库），
                # 所以这里传显式范围、关掉 retriever 自己的自动解析，避免两套规则打架
                paper_ids=turn.scope or None,
                auto_scope=False,
                stats=stats,
                # 入口小模型点名的资产（"表 3"）直接定点注入；它没点名时传 None，
                # 让 retriever 用正则再兜一次底（见 prepare_turn 的 _resolve_injected）
                injected=turn.injected or None,
            )
            cards = build_cards(hits)
            return {
                "turn": turn,
                "cards": cards,
                "hits": hits,
                "timing": stats.get("timing") or {},
            }
        finally:
            conn.close()

    # 帧里带上前端要用的元信息：本轮实际用的问句、范围、范围是从哪来的。
    # 展示用 `question`，检索用 `standalone` —— 两者的差别得让前端看得见。
    meta = {
        "session_id": turn.session_id,
        "standalone": turn.standalone,
        "rewritten": turn.rewritten,
        "chitchat": turn.is_chitchat,
        "scope": turn.scope,
        "scope_source": turn.scope_source,
    }

    async def frames() -> AsyncIterator[str]:
        """SSE 帧序列：**先发意图，再发证据，最后流答案**。

        顺序是刻意的：入口理解（改写 / 范围 / 闲聊）只等一次小模型调用（几百毫秒），
        而检索慢一个量级（embedding + 关键词 + rerank，约 1 秒）。先把 `intent` 发出去，
        前端的面板（范围、改写提示）就能立刻更新，用户不用盯着空白等。

        注意：流一旦开出去，HTTP 状态码就改不了了 —— 所以会话 404 之类的检查必须留在
        `chat()` 开头（见上面）。
        """
        yield sse_frame("intent", meta)

        context = await run_in_threadpool(build_retrieval, turn)
        cards = context["cards"]
        # 存进会话的卡片快照：截断正文，只留回看和溯源够用的部分
        card_snapshot = card_payload(cards, text_limit=240)

        def save(answer: str, citations: dict[str, Any]) -> None:
            """落库：**流结束之后**才写，写的是最终答案（不是拼了一半的增量）。"""
            if not turn.session_id:
                return
            conn = connect()
            try:
                init_db(conn)
                save_turn(conn, turn, answer=answer, citations=citations, cards=card_snapshot)
            finally:
                conn.close()

        if not cards:
            if context.get("chitchat"):
                # 闲聊短路：不检索、没证据、也不写引用 —— 用小模型自己接住，顺势把用户引回正题。
                # 用的是**原话**（不是改写后的）：寒暄没什么可指代消解的。
                client = LLMClient(settings)
                chitchat_started = time.perf_counter()

                def chitchat_finalize(answer: str) -> dict[str, Any]:
                    raw = answer.strip()
                    if raw:
                        # 闲聊不是拒答：档位显式标成 chitchat，前端据此显示「闲聊」
                        # 而不是「未收录」—— 两件事的措辞不该一样。
                        text = raw
                        report = CitationReport(
                            mode=MODE_UNKNOWN, unknown_reason=REASON_CHITCHAT
                        )
                    else:
                        text = EMPTY_ANSWER_FALLBACK
                        report = CitationReport(
                            mode=MODE_UNKNOWN, unknown_reason=REASON_FAILED
                        )
                    payload = citation_payload(report)
                    if report.unknown_reason == REASON_FAILED:
                        payload["_failed"] = True
                    # 打标：闲聊轮不进改写历史（否则"你好"会污染下一轮的指代消解）
                    payload["_chitchat"] = True
                    save(text, payload)
                    return {
                        **meta,
                        "answer": text,
                        "citations": payload,
                        "mode": report.mode,
                        "unknown_reason": report.unknown_reason,
                        "latency_ms": round((time.perf_counter() - chitchat_started) * 1000, 1),
                    }

                async for frame in stream_events(
                    context={
                        **meta,
                        "cards": [],
                        # 证据帧先于生成发出，这时还没有引用报告；给一个**中性**的空报告，
                        # 别拿 parse_citations("") 去推 —— 那会推成"降级"，前端会闪一个徽标。
                        "citations": citation_payload(CitationReport()),
                        "timing": {},
                    },
                    answer_stream=client.chat_stream(
                        turn.question, system=load_chitchat_prompt()
                    ),
                    finalize=chitchat_finalize,
                ):
                    yield frame
                return

            # 一条证据都没有：不调 LLM（省钱，也避免硬编）。协议仍然走完，前端逻辑统一。
            async def empty_stream() -> AsyncIterator[str]:
                if False:  # pragma: no cover - 只是让它是个异步生成器
                    yield ""

            def empty_finalize(answer: str) -> dict[str, Any]:
                # 一张证据都没有：不调 LLM，直接拒答。档位和 reason **显式写死**，
                # 别拿 parse_citations("") 去推 —— 那推出来的是"降级"，语义不对。
                text = "资料不足：没有检索到与这个问题相关的证据。"
                report = CitationReport(
                    mode=MODE_UNKNOWN, unknown_reason=REASON_NOT_IN_CORPUS
                )
                payload = citation_payload(report)
                save(text, payload)
                return {
                    **meta,
                    "answer": text,
                    "citations": payload,
                    "mode": report.mode,
                    "unknown_reason": report.unknown_reason,
                    "latency_ms": 0.0,
                }

            async for frame in stream_events(
                context={
                    **meta,
                    "cards": [],
                    "citations": citation_payload(CitationReport()),
                    "timing": context["timing"],
                },
                answer_stream=empty_stream(),
                finalize=empty_finalize,
            ):
                yield frame
            return

        # 生成用**改写后**的问句：它自包含，历史里的指代已经在改写那一步消解掉了。
        # 用 request.question 的话，生成模型会收到一句带"它/那个"却没有指代对象的问题。
        # 展示仍用原始 `question`（见 meta），两者别混。
        # 证据里有图时这是图文混排的 content blocks（图带着自己的 [En] 一起发），
        # 没图时就是一个字符串 —— 生成侧不用分叉，见 answerer.build_user_content。
        user_message = build_user_content(turn.standalone, cards, settings)
        system_prompt = load_system_prompt()
        client = LLMClient(settings)
        started = time.perf_counter()

        def finalize(answer: str) -> dict[str, Any]:
            # 档位和最终正文都由 assess 定（跟同步路径共用同一份判据）：
            #   拒答 → 去掉标记 + 剥掉残留引用；降级 → 整段换文案；一个字没吐 → 失败文案。
            # **绝不能存空串** —— 会话里会留下"有问无答"的脏记录，空的助手消息还会
            # 白占改写历史的名额（sessions.history 靠 _failed 过滤）。
            text, report = assess(answer, cards)
            payload = citation_payload(report)
            if report.unknown_reason == REASON_FAILED:
                payload["_failed"] = True
            save(text, payload)
            return {
                **meta,
                "answer": text,
                "citations": payload,
                "mode": report.mode,
                "unknown_reason": report.unknown_reason,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }

        async for frame in stream_events(
            context={
                **meta,
                "cards": card_payload(cards),
                # 证据先发时还没生成，引用校验是空的；真正结果在 done 事件里
                "citations": citation_payload(CitationReport()),
                "timing": context["timing"],
            },
            answer_stream=client.chat_stream(user_message, system=system_prompt, temperature=0.2),
            finalize=finalize,
        ):
            yield frame

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )
