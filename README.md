# MultiPaperQA_RAG · 多论文问答系统

把一批 PDF 论文丢进去，用自然语言提问，回答带原文引用与图表。
链路：MinerU 解析 → 结构化切分 → 混合检索（向量 + 关键词）→ 重排 → 带引用的回答。

- 测什么、怎么对比、本地模型怎么落 见 [EVAL_PLAN.md](EVAL_PLAN.md)

## 能做什么

| 环节 | 说明 |
| --- | --- |
| 入库 | 上传 PDF → MinerU 解析 → 章节树 → 结构化切分 → 落 SQLite + 向量库；后台异步跑，可轮询进度 |
| 检索 | 向量召回 + 关键词召回 → RRF 融合 → 重排 → 意图与范围路由 → 充分性判断 |
| 回答 | 证据卡组装 → 带引用的生成；SSE 流式，**先发证据再发字** |
| 范围 | 单篇 / 全库提问；带 `session_id` 时会话内继承上一轮范围（sticky scope） |
| 多模态 | 表格、图片按需挂载，看图回答 |
| 前端 | 内置静态页（会话列表 + 对话 + PDF.js 按 bbox 高亮），无 npm 构建步骤 |
| 评测 | `app/eval/` 评测集 + 指标脚本，一条命令出基线数字 |

## 环境

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # 然后填入自己的 key（生成 / 规划 / Embedding / Rerank / MinerU）
```

`.env` 里的 `STORAGE_DIR`、`DB_PATH`、`MODEL_CACHE_DIR` 是绝对路径，**换机器或改目录名时要一起改**。

自检两条：

```powershell
.venv\Scripts\python.exe -m scripts.check_env        # 解释器、工程根目录、storage 目录
.venv\Scripts\python.exe -m scripts.check_providers  # key、网络、模型名真实可用
```

## 跑起来

```powershell
# 必须单 worker：入库任务表在进程内存里（见 app/api/jobs.py）
.venv\Scripts\python.exe -m uvicorn app.api.main:app --port 8000
```

打开 http://127.0.0.1:8000 就是前端（挂在 `/ui`，根路径 302 过去），
http://127.0.0.1:8000/docs 是接口文档。

## 跑测试

```powershell
.venv\Scripts\python.exe -m pytest
```

`conftest.py` 把 pytest 的临时目录固定成工程内的 `.pytest_tmp/<随机后缀>`（避开系统 Temp 里
别人建的只读 ACL）。`.pytest_tmp*/`、`.pytest_cache/` 都在 `.gitignore` 里，**积累的临时目录可以直接删**，
不影响任何东西。

## 常用脚本

| 命令 | 做什么 |
| --- | --- |
| `-m scripts.init_db` | 建库建表（幂等，重复跑不清数据） |
| `-m scripts.run_mineru storage/pdfs/xxx.pdf` | PDF → MinerU → blocks + 章节树 + 落库 |
| `-m scripts.index_chunks --all` | blocks → chunks + assets 入库（不联网、不花钱） |
| `-m scripts.demo_answer "问题" --show-evidence` | 单次带引用回答，顺便核对证据卡 |
| `-m scripts.run_eval --list` | 先看评测会跑哪些题、会花多少钱 |

命令前都要加 `.venv\Scripts\python.exe`；每个脚本的 docstring 里有完整参数。

## 目录

| 目录 | 作用 |
| --- | --- |
| `app/providers/` | 模型接入层（LLM / VLM / Embedding / Rerank），换模型只改这里 |
| `app/db/` | SQLite schema + repositories |
| `app/ingest/` | MinerU 调用、结构化转换、章节树、切分、入库 |
| `app/retrieval/` | 多路召回、RRF 融合、重排、Planner、充分性判断 |
| `app/generation/` | 证据卡组装、生成、引用、多模态挂图 |
| `app/api/` | FastAPI 接口 + 单页前端（`app/api/static/`） |
| `app/eval/` | 评测集与指标 |
| `scripts/` | 每一步的验收脚本 |
| `storage/` | PDF、MinerU 产物、向量库、数据库（不进 git） |

## 仓库约定

`.gitignore` 挡住了不该入库的东西：`.venv/`、`__pycache__/`、各类测试与类型检查缓存、
`.eval_out/`、`.env`、`storage/*`（只留 `storage/.gitkeep`）、`*.db`、`*.zip`、`*.log`。

- **`storage/` 不进 git**：`app.db`、Chroma 向量库、MinerU 产物体积大且能重建，换机器后重跑
  `scripts.run_mineru` + `scripts.index_chunks` 就行。
- **`.env` 不进 git**：里面有明文 key。换机器从 `.env.example` 复制，并把 `STORAGE_DIR` /
  `DB_PATH` / `MODEL_CACHE_DIR` 三个路径改成新位置。
- 提交前扫一眼 `git status`，确认没有 `*.db` / `storage/` / `.env` 混进来。它们在 `.gitignore` 里，
  但要是曾经被 `git add -f` 强行跟过，得先 `git rm --cached <文件>` 才会真的忽略。
