"""入库任务表（v1 版：进程内存 + 线程）。

**为什么要有它**：`POST /papers` 要调 MinerU，一篇论文几分钟。同步等它跑完，
HTTP 请求必然超时、浏览器也不会等。所以接口立刻返回 `task_id`，进度另开一个
`GET /tasks/{id}` 查。

**为什么不做成正经的任务队列**：v1 是单进程演示，Celery / Redis 那一套
装起来比接口本身还大。内存表够用，代价说清楚：

1. **进程重启任务就没了**（不是"失败"，是"查不到"）；
2. **多 worker 不共享**（`uvicorn --workers 4` 的话，任务在 A 进程、查询打到 B 就 404）。
   演示一律用单 worker 起，这条写进启动命令里。

真要多用户并发入库，那是 BACKLOG B16（PostgreSQL）那一步的事。
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Step:
    """任务里的一个步骤（解析 / 切分），前端能看见卡在哪一步。"""

    name: str
    status: str = STATUS_QUEUED
    detail: str = ""


@dataclass
class Job:
    task_id: str
    paper_id: str
    filename: str
    pdf_path: str = ""
    status: str = STATUS_QUEUED
    steps: list[Step] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None


class Progress:
    """任务里报告进度的小工具：`begin("parse")` / `done("parse", "26 个块")`。

    只按**步骤名**匹配，步骤顺序由 `steps` 列表定死 —— 跑的时候顺序有变也不用改别处。
    """

    def __init__(self, job: Job, lock: threading.Lock) -> None:
        self._job = job
        self._lock = lock

    def _set(self, name: str, status: str, detail: str) -> None:
        with self._lock:
            for step in self._job.steps:
                if step.name == name:
                    step.status = status
                    step.detail = detail
                    break

    def begin(self, name: str, detail: str = "") -> None:
        self._set(name, STATUS_RUNNING, detail)
        self._job.status = STATUS_RUNNING

    def done(self, name: str, detail: str = "") -> None:
        self._set(name, STATUS_DONE, detail)


class JobRegistry:
    """线程安全的任务表。**单进程内有效**，见模块开头。"""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        *,
        paper_id: str,
        filename: str,
        pdf_path: str,
        steps: list[str],
        run: Callable[[Job, Progress], dict[str, Any]],
    ) -> Job:
        """登记一个任务并**立刻返回**，真正的活在后台线程里跑。

        `run(job, progress)` 里用 `progress.begin/done` 报告进度。
        """
        job = Job(
            task_id=uuid.uuid4().hex[:12],
            paper_id=paper_id,
            filename=filename,
            pdf_path=pdf_path,
        )
        job.steps = [Step(name=name) for name in steps]
        with self._lock:
            self._jobs[job.task_id] = job

        progress = Progress(job, self._lock)

        def worker() -> None:
            job.status = STATUS_RUNNING
            try:
                job.result = run(job, progress) or {}
                job.status = STATUS_DONE
            except Exception as exc:  # noqa: BLE001 - 任务失败要如实回给调用方
                job.status = STATUS_FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                # 完整栈只打日志，不回给接口 —— 里面有本机路径和 key
                traceback.print_exc()
            finally:
                job.finished_at = _now()

        threading.Thread(target=worker, name=f"ingest-{job.task_id}", daemon=True).start()
        return job

    def get(self, task_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(task_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)


# 全局单例：接口和后台线程共用一个表
REGISTRY = JobRegistry()


def run_ingest_job(job: Job, progress: Progress) -> dict[str, Any]:
    """真正的入库活：解析（papers/sections/blocks）-> 切分 + 写向量。

    **每个线程自己开连接** —— SQLite 连接不能跨线程共用（`check_same_thread`），
    而且解析和切分是两个事务，中间崩了要能看出停在哪一步。
    """
    from app.config import get_settings
    from app.db.connection import connect, init_db
    from app.ingest.indexer import index_paper
    from app.ingest.pipeline import parse_paper

    settings = get_settings()
    conn = connect()
    init_db(conn)
    try:
        progress.begin("parse", "调用 MinerU 或命中缓存")
        report = parse_paper(pdf=job.pdf_path, paper_id=job.paper_id, settings=settings, conn=conn)
        progress.done("parse", f"{len(report.build.blocks)} 个块，来源：{report.mineru_source}")

        progress.begin("index", "切分 + 写向量")
        stats = index_paper(conn, job.paper_id, settings)
        progress.done(
            "index",
            f"{stats.get('chunks', 0)} 个 chunk，向量 {stats.get('vectors_added', 0)} 条",
        )

        return {
            "paper_id": job.paper_id,
            "title": report.title,
            "blocks": len(report.build.blocks),
            "chunks": stats.get("chunks", 0),
            "vectors": stats.get("vectors_added", 0),
            "mineru_source": report.mineru_source,
        }
    finally:
        conn.close()
