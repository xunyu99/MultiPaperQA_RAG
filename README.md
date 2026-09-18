# Agentic RAG 论文问答系统

基于 MinerU 解析 + 结构化切分 + 混合检索 + LangGraph 编排的论文问答系统。
完整路线图见 [PLAN.md](PLAN.md)。

## 环境

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # 然后填入自己的 key
```

## 跑测试

```powershell
.venv\Scripts\python.exe -m pytest
```

## 目录

| 目录 | 作用 |
| --- | --- |
| `app/providers/` | 模型接入层（LLM / VLM / Embedding / Rerank），换模型只改这里 |
| `app/db/` | SQLite schema + repositories |
| `app/ingest/` | MinerU 调用、结构化转换、章节树、切分、入库 |
| `app/retrieval/` | 多路召回、RRF 融合、重排、Planner、充分性判断 |
| `app/generation/` | 证据卡组装、生成、引用、多模态挂图 |
| `app/graph/` | LangGraph 状态机 |
| `app/api/` | FastAPI 接口 |
| `app/eval/` | 评测集与指标 |
| `scripts/` | 每一步的验收脚本 |
| `storage/` | PDF、MinerU 产物、数据库（不进 git） |
