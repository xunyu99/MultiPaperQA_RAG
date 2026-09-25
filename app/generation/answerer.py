"""证据卡 → prompt → 答案 → 引用校验 + 档位（Step 6，Step 9 扩成两档）。

分三层，每层都能单独测：

    build_user_message   拼 prompt（不碰网络）
    answer_from_cards    卡片 + LLM → 答案（LLM 可注入，测试用假的）
    answer_question      检索 → 卡片 → 答案（完整链路）

**两条互相独立的判据，合起来定出这一轮的档位**（见 PLAN Step 9）：

1. **引用校验**（这条是老的，仍然管用）：模型很爱编引用编号（`[E9]` 而证据只到 `[E5]`），
   所以拿到回答后一定要解析出来跟证据对一遍：

   - `cited`   回答里用到、**且证据里真实存在**的编号
   - `unknown` 回答里用了、但**证据里没有**的编号 —— 这是幻觉引用的硬信号
   - `coverage` 被引用的证据占全部证据的比例；太低说明召回的证据大半没用上

2. **拒答标记**（Step 9 加的）：模型自己知道"这条答不上来"，但**"答了"不能由它自己说了算** ——
   所以协议是不对称的：

   - 正常回答：**没有标记**，正文带 `[En]`（"我答了"由引用证据证明，不由模型声明）
   - 拒答：首行 `[[REFUSE:out_of_scope]]` 或 `[[REFUSE:not_in_corpus]]`，第二行起一句说明

   由此得到两档：`grounded`（有据可依）/ `unknown`（拒答或降级）。
   `unknown` 还带一个 reason（`out_of_scope` / `not_in_corpus` / `downgraded` / `failed`），
   前端的措辞和评测的 reason 命中率都读它。

**为什么不用 JSON 让模型"结构化输出"**：生成走 SSE 流式，JSON 没法边生成边渲染正文；
而且思考模式不支持强制 `tool_choice`、`response_format=json_schema` 直接 400，
只能靠"请你输出 JSON"，失效模式更多、失效还是全局的。最要紧的是 —— `answer` 字段照样是字符串，
模型照样能在里面写 `[E1]`：换的是信封，问题在信封里面。详见 PLAN Step 9。
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.generation.evidence import EvidenceCard, build_cards, render
from app.providers.llm import LLMClient, image_to_data_url
from app.retrieval.retriever import retrieve
from app.retrieval.vector_index import ChunkVectorIndex

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "answer.md"
# 闲聊短路的系统提示词。**不能复用 answer.md**：那份的第一条硬规则是"只能用证据、
# 不要用你自己的知识" —— 拿它回答"你是谁"，模型会一本正经回你"资料不足"。
CHITCHAT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "chitchat.md"

# 回答里出现的引用编号。
# **只认半角方括号一对一**（`[E2]` / `[E2][E3]`）；`[E2, E3]` / `【E2】` / `（E2）` 解析不到
# —— 实测 22 题的真实回答里一次都没出现（模型清一色写 `[E3]`），所以先不做宽容版，
# 见 PLAN Step 9「测量二」。真出现了再放宽，别提前写一套没人踩的兼容代码。
_CITATION = re.compile(r"\[E(\d+)\]")

# 拒答标记：**只用于拒答，只有两个值**。规则里刻意没有 `grounded` ——
# "我答了"必须由引用证据来证明，模型唯一能声明的是"我不答"（标记只能降级，不能升级）。
# 解析宽容：忽略大小写、容忍任意空白、容忍被 markdown 加粗包起来（`**[[REFUSE:...]]**`）。
_REFUSAL = re.compile(
    r"\[\[\s*REFUSE\s*:\s*(out_of_scope|not_in_corpus)\s*\]\]", re.IGNORECASE
)

MODE_GROUNDED = "grounded"
MODE_UNKNOWN = "unknown"

REASON_OUT_OF_SCOPE = "out_of_scope"
REASON_NOT_IN_CORPUS = "not_in_corpus"
REASON_DOWNGRADED = "downgraded"
REASON_FAILED = "failed"
REASON_CHITCHAT = "chitchat"

# 前端徽标用的短标签（也方便日志里一眼看懂）
REASON_LABELS = {
    REASON_OUT_OF_SCOPE: "不在知识库范围内",
    REASON_NOT_IN_CORPUS: "知识库未收录",
    REASON_DOWNGRADED: "依据不足",
    REASON_FAILED: "生成失败",
    REASON_CHITCHAT: "闲聊",
}

# 「资料不足」是拒答标记的**后备**判据：模型该吐标记却没吐时还能认出这种写法。
# 判断方式是**首行前缀**，不是"前 200 字里任意位置" —— 实测有个正常回答收尾写着
# 「资料不足的部分：证据中没有说明…」（keyword_en_3），那种**是有依据的回答**，
# 按老判据会被误判成拒答、整段丢掉。
_INSUFFICIENT_PREFIX = "资料不足"
# 超过这么多字的回答，一条引用都没有就可疑（评测那边用同一条判据）
UNCITED_LENGTH_HINT = 60
# 生成失败（模型连重试都没吐出内容）时的兜底文案。
# **不能返回空串**：会话里会留下"有问无答"的脏记录（实测库里 9 条助手消息有 6 条
# 是空的），而空的助手消息还会白占改写历史的名额。见 sessions.history 的过滤。
EMPTY_ANSWER_FALLBACK = "生成失败：模型没有返回内容，请重试。"
# 降级（声明有据却拿不出有效引用）时**换掉整段正文**的固定文案。
# 只改徽标、把那段没引用的正文留在屏幕上，等于换个方式把模型自己的知识展示出去 ——
# 和"不做 general"自相矛盾（见 PLAN Step 9）。
DOWNGRADE_ANSWER = "资料不足：证据里没有足够依据支撑这个回答。"
# 拒答说明被剥空时的兜底（剥掉标记和残留的 `[En]` 之后什么都不剩）。
EMPTY_REFUSAL_FALLBACK = "资料不足：证据里没有提到这条信息。"


@dataclass
class CitationReport:
    cited: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    coverage: float = 0.0
    # 两档：grounded（有据可依）/ unknown（拒答或降级）。判定见 parse_citations。
    mode: str = MODE_GROUNDED
    # 只有 unknown 才有值：out_of_scope / not_in_corpus / downgraded / failed / chitchat
    unknown_reason: str | None = None
    # 声明有据却拿不出有效引用 —— 质量指标，应当恒为 0（见 PLAN Step 9）
    downgraded: bool = False

    @property
    def insufficient(self) -> bool:
        """这一轮拒答了（老字段名，评测/前端还在用）。"""
        return self.mode == MODE_UNKNOWN

    @property
    def looks_uncited(self) -> bool:
        """挂了 grounded 却一条引用都没有。正常流程下不该出现（那种会被降级）。"""
        return self.mode == MODE_GROUNDED and not self.cited


@dataclass
class AnswerResult:
    question: str
    answer: str
    cards: list[EvidenceCard]
    citations: CitationReport
    user_message: str
    latency_ms: float
    # 模型原始输出（没被降级/剥标记处理过）。诊断用：对比它和 answer 就知道这一轮
    # 到底是被降级了、还是拒答说明被剥了引用。
    raw_answer: str = ""

    @property
    def mode(self) -> str:
        return self.citations.mode

    @property
    def unknown_reason(self) -> str | None:
        return self.citations.unknown_reason


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def load_chitchat_prompt() -> str:
    return CHITCHAT_PROMPT_PATH.read_text(encoding="utf-8")


def build_user_message(question: str, cards: list[EvidenceCard]) -> str:
    """【证据】+【问题】。**不用 str.format** —— 证据里全是 LaTeX 的 `{}`，format 会崩。"""
    return f"【证据】\n{render(cards)}\n\n【问题】\n{question}"


# 一次问答最多带几张图（护栏）。实测"本体块"最多带 4 张、且只有一个 chunk 触及，
# 所以这个上限今天永远不会触发 —— 它防的是以后换语料冒出"一个块挂 20 张图"。
MAX_EVIDENCE_IMAGES = 4


def collect_figures(
    cards: list[EvidenceCard], settings: Settings | None = None
) -> list[tuple[str, dict[str, Any], Path]]:
    """挑出**要真的送进多模态消息**的图，返回 [(证据编号, 资产行, 磁盘路径)]。

    图能进来只有一个原因：**它的本体就在这张证据卡里** —— 也就是 `card.assets` 里有它，
    而那只有正文里带 `[FIGURE_REF asset_id=...]` 占位符时才会成立
    （`expand_assets` 只按占位符展开）。

    **"正文只是提到、本体在别处"的图不会进来**：那种绑定只存在于 `chunk_assets`，
    进不了 `card.assets`。这不是漏了，是规则（见 PLAN §6.2「资产携带」）：
    提到资产的块往往是综述/结论句，把本体硬塞进来等于偷偷扩 top-k。

    图在盘上找不到就跳过 —— 一张图丢了不能让整个请求挂掉。
    """
    settings = settings or get_settings()
    root = Path(settings.storage_dir) / "mineru"

    found: list[tuple[str, dict[str, Any], Path]] = []
    for card in cards:
        for asset in card.assets:
            if asset.get("asset_type") != "figure" or not asset.get("image_path"):
                continue
            path = root / card.paper_id / str(asset["image_path"])
            if not path.is_file():
                continue
            found.append((card.label, asset, path))
            if len(found) >= MAX_EVIDENCE_IMAGES:
                return found
    return found


def build_user_content(
    question: str,
    cards: list[EvidenceCard],
    settings: Settings | None = None,
) -> str | list[dict[str, Any]]:
    """证据 + 问题 → 发给模型的内容。**没有图就返回纯字符串**，跟以前一字不差。

    有图时返回图文混排的 content blocks（顺序刻意是"证据 → 图（逐张）→ 问题"）：

        [text]  【证据】…            ← render(cards)
        [text]  【E3 的图】Figure 3: …
        [image] （E3 那张图）
        [text]  【问题】…

    **每张图前面必须先给一段带 `[En]` 的文字**：模型收到一堆图却不知道哪张对应哪条证据，
    就没法把"从图里读出来的东西"标成 `[En]`；标不出编号 = 零引用的长回答，
    会被兜底降级成"资料不足"（见 prompts/answer.md 的图规则）。
    """
    figures = collect_figures(cards, settings)
    if not figures:
        return build_user_message(question, cards)

    parts: list[dict[str, Any]] = [{"type": "text", "text": f"【证据】\n{render(cards)}"}]
    for label, asset, path in figures:
        caption = (asset.get("caption") or "").strip() or "（无图注）"
        parts.append({"type": "text", "text": f"\n【{label} 的图】{caption}"})
        parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    parts.append({"type": "text", "text": f"\n【问题】\n{question}"})
    return parts


def parse_citations(answer: str, cards: list[EvidenceCard]) -> CitationReport:
    """把 `[En]` 抠出来跟证据对一遍，并定出这一轮的档位。

    判定是一条**全函数**（顺序执行，先到先得）：

        1. 有拒答标记        → unknown + 标记里的 reason
        2. 有有效引用且无编造 → grounded
        3. 首行是「资料不足」 → unknown / not_in_corpus（标记没吐出来时的后备）
        4. 其余              → unknown / downgraded（**兜底降级**）

    第 1 条必须排在引用**前面**：实测两道拒答题在拒答的同时**引用了 E1/E4/E5**
    （prompt 规则要求"每个事实都标出处"，而"证据只涉及 X 和 Y"是在陈述证据内容），
    先看引用的话它们会被判成 grounded。
    第 3 条又必须排在引用**后面**：正常回答的收尾可能写着「资料不足的部分：…」
    （实测 keyword_en_3），那是有依据的回答，别整段丢掉。
    """
    text = answer or ""
    known = {card.label for card in cards}
    found = {f"E{match.group(1)}" for match in _CITATION.finditer(text)}
    cited = sorted(found & known, key=_label_order)
    unknown = sorted(found - known, key=_label_order)
    coverage = len(cited) / len(cards) if cards else 0.0

    refusal = find_refusal(text)
    if refusal:
        mode, reason = MODE_UNKNOWN, refusal
    elif cited and not unknown:
        mode, reason = MODE_GROUNDED, None
    elif looks_insufficient(text):
        mode, reason = MODE_UNKNOWN, REASON_NOT_IN_CORPUS
    else:
        mode, reason = MODE_UNKNOWN, REASON_DOWNGRADED

    return CitationReport(
        cited=cited,
        unknown=unknown,
        coverage=coverage,
        mode=mode,
        unknown_reason=reason,
        downgraded=(reason == REASON_DOWNGRADED),
    )


def find_refusal(answer: str) -> str | None:
    """抠出拒答标记里的 reason。没有标记就返回 None。"""
    match = _REFUSAL.search(answer or "")
    return match.group(1).lower() if match else None


def looks_insufficient(answer: str) -> bool:
    """**首行**是不是以「资料不足」开头（拒答标记的后备判据）。

    刻意收紧成首行前缀，而不是"前 200 字里任意位置"：正常回答的收尾可能写着
    「资料不足的部分：证据中没有说明…」，那是**有依据的回答**，按老判据会被误判成拒答。
    容忍 markdown 加粗 / 标题前缀（`**资料不足：**`、`## 资料不足`）。
    """
    text = (answer or "").lstrip()
    if not text:
        return False
    first = text.splitlines()[0].strip().lstrip("*# \u3000").strip()
    return first.startswith(_INSUFFICIENT_PREFIX)


def assess(raw: str, cards: list[EvidenceCard]) -> tuple[str, CitationReport]:
    """模型原始输出 → (给用户看的正文, 引用报告/档位)。

    **同步和流式两条路共用这一份**（`answer_from_cards` 和 `main.py` 的 `finalize`），
    否则两边会各自漂移出一套规则 —— 那正是"两套解析逻辑并存产生歧义"的老坑。
    """
    text = (raw or "").strip()
    if not text:
        return EMPTY_ANSWER_FALLBACK, CitationReport(
            mode=MODE_UNKNOWN, unknown_reason=REASON_FAILED
        )
    report = parse_citations(text, cards)
    return finalize_answer(text, report), report


def finalize_answer(raw: str, report: CitationReport) -> str:
    """按档位处理最终给用户看的正文。

    - `grounded`：原样（顺手去掉可能漏出来的标记，别让哨兵出现在用户眼前）
    - 降级：**整段换成固定文案** —— 那段没有依据的正文不能给用户看
    - 拒答：去掉标记，并剥掉任何残留的 `[En]` —— 这一道让"拒答里不许有引用"从
      **靠模型自觉**变成**结构上保证**：模型手滑写了编号，用户也看不到。
    """
    text = (raw or "").strip()
    if report.mode == MODE_GROUNDED:
        return _strip_refusal_marker(text)
    if report.downgraded:
        return DOWNGRADE_ANSWER
    body = _strip_refusal_marker(text)
    # 剥掉残留的编号；顺带清掉剥完留下的空格（"证据只涉及 A [E1] 和 B [E5]，" → "…和 B，"）
    body = _CITATION.sub("", body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\s+([，。；：、）】》！？])", r"\1", body)
    body = body.strip()
    return body or EMPTY_REFUSAL_FALLBACK


def _strip_refusal_marker(text: str) -> str:
    """去掉标记，并清掉**包着它的 markdown 加粗符号**。

    模型很爱写成 `**[[REFUSE:not_in_corpus]]**` —— 只 sub 掉标记的话，屏幕上会剩下一个
    孤零零的 `****`（实测踩到）。这里只吃行首的 `*` / `_` / 空白，不吃 `-`，
    免得把拒答说明里合法的列表符号也削掉。
    """
    body = _REFUSAL.sub("", text or "")
    lines = body.splitlines()
    if lines:
        lines[0] = re.sub(r"^[\s*_]+", "", lines[0])
        if not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def answer_from_cards(
    question: str,
    cards: list[EvidenceCard],
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> AnswerResult:
    """卡片 + LLM → 答案。**没有证据时直接返回"资料不足"，不调 LLM**（省钱，也避免编造）。"""
    settings = settings or get_settings()
    started = time.perf_counter()

    if not cards:
        return AnswerResult(
            question=question,
            answer="资料不足：没有检索到与这个问题相关的证据。",
            cards=[],
            citations=CitationReport(
                mode=MODE_UNKNOWN, unknown_reason=REASON_NOT_IN_CORPUS
            ),
            user_message="",
            latency_ms=0.0,
        )

    # 纯文本版留作记录/诊断（user_message 字段），真正发出去的是 build_user_content ——
    # 证据里有图时它是图文混排的 content blocks，没图时就是那个字符串。
    user_message = build_user_message(question, cards)
    content = build_user_content(question, cards, settings)
    client = llm or LLMClient(settings)
    raw = client.chat(content, system=load_system_prompt(), temperature=0.2)
    latency = (time.perf_counter() - started) * 1000
    # 档位和最终正文都由 assess 定：模型没吐内容 → failed 兜底文案；
    # 声明有据却拿不出引用 → 降级成固定拒答文案（不会把那段没依据的正文放出去）
    answer, citations = assess(raw, cards)

    return AnswerResult(
        question=question,
        answer=answer,
        cards=cards,
        citations=citations,
        user_message=user_message,
        latency_ms=latency,
        raw_answer=(raw or "").strip(),
    )


def answer_question(
    conn: sqlite3.Connection,
    question: str,
    paper_id: str | None = None,
    paper_ids: list[str] | None = None,
    k: int | None = None,
    index: ChunkVectorIndex | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> AnswerResult:
    """完整链路：检索（含论文范围解析）→ 证据卡 → 生成。"""
    settings = settings or get_settings()
    index = index or ChunkVectorIndex(settings)
    items = retrieve(
        conn, question, k=k, index=index, settings=settings,
        paper_id=paper_id, paper_ids=paper_ids,
    )
    return answer_from_cards(question, build_cards(items), llm=llm, settings=settings)


def _label_order(label: str) -> int:
    match = re.match(r"E(\d+)$", label)
    return int(match.group(1)) if match else 0
