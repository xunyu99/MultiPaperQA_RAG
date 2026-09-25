"""入口理解：一次小模型调用，拿到这一轮需要的全部结构化信息。

替代原来单一的"指代改写"。原来 `rewrite_question` 只把问句补完整，现在一次调用同时产出：

    rewritten_query  指代消解后的自包含问句 —— **检索和生成都用它**（见 PLAN 的链路）
    is_chitchat      明显闲聊（社交/元问题），给入口短路用
    asset_type       问句里点名的资产类型（table / figure / formula）
    asset_number     点名的编号（一律阿拉伯数字，代码还会再归一化一次兜底）

**为什么合成一次调用**：拆成三次就是每轮多两次模型往返（几百 ms 一次），而这几件事
读的是同一段上下文、同一个模型就能出，没有理由分开。走 `LLMClient.structured`
（function calling + pydantic 校验 + 校验失败自纠一次），见 providers/llm.py。

**第一轮也要调**。指代改写第一轮确实不需要（没有上下文可指代），但"这句是不是闲聊"
和"用户点名了表几"第一轮同样要判 —— 所以不能像旧版那样第一轮直接跳过。

**失败必须无害**：超时、模型不调工具、字段不合法时一律回退成
`QueryIntent(rewritten_query=原问题)`，这一轮照常跑下去。资产编号那一路还有正则兜底
（retriever 的 `auto_asset`），闲聊那一路最多是没短路成功。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.generation.rewrite import HISTORY_ROUNDS, format_history, sanitize_rewrite
from app.providers.llm import LLMClient


class QueryIntent(BaseModel):
    """一轮提问的结构化理解。字段名就是 function calling 的参数名。"""

    rewritten_query: str = Field(
        description="补全指代后的独立问句；不需要改写就原样返回用户的问题"
    )
    is_chitchat: bool = Field(
        default=False,
        description="跟论文知识库无关的社交/元问题（问候、自我介绍、你能做什么）",
    )
    asset_type: Literal["table", "figure", "formula"] | None = Field(
        default=None, description="用户点名的资产类型；没有点名就 null"
    )
    asset_number: str | None = Field(
        default=None, description="点名的编号，一律阿拉伯数字（表 VIII → 8）；没有点名就 null"
    )


SYSTEM_PROMPT = """你是论文问答系统的入口理解模块。一次调用同时判断四件事，只输出结构化字段。

1. rewritten_query：把用户最新问题里的指代（它 / 这个 / 上面那篇 / 那两篇）补全成**脱离上下文也能读懂**的独立问句。
   - 保持原语言：中文问就中文，英文问就英文；
   - 不改变意图，不添加历史里没有的信息；
   - 已经完整、或者历史帮不上忙，就**原样返回**用户的话，不要重写、不要美化。

2. is_chitchat：这句是不是**跟论文知识库无关**的社交或元问题（寒暄、自我介绍、你能做什么、天气）。
   - 注意区分：问"你知道 X 吗"是闲聊；问"论文里的 X 是什么"不是；
   - 拿不准就填 false（宁可漏判，也别把正经问题挡掉）。

3. asset_type / asset_number：用户**点名了某个具体编号**的表格、图片或公式时才填。
   - 只有指向具体编号才算（"表 3"、"Figure 2"、"式(5)"、"Table VII"）；
   - 泛指不算（"有哪些表"、"表格里写了什么"、"图注是什么"）；
   - asset_number 一律输出**阿拉伯数字**（"表 VIII" → "8"）；
   - 没有点名就都填 null。"""


def parse_intent(
    question: str,
    history: list[dict[str, str]],
    *,
    llm: Any = None,
    settings: Any = None,
    enabled: bool = True,
    rounds: int = HISTORY_ROUNDS,
) -> QueryIntent:
    """返回这一轮的结构化理解。任何失败都退回"原问题 + 什么都不判"。"""
    fallback = QueryIntent(rewritten_query=question)
    if not enabled:
        return fallback

    context = format_history(history, rounds) or "（无）"
    prompt = f"对话历史：\n{context}\n\n最新问题：{question}\n"
    client = llm or LLMClient(settings)
    try:
        intent = client.structured(QueryIntent, prompt, system=SYSTEM_PROMPT)
    except Exception:  # noqa: BLE001 - 见模块开头"失败必须无害"
        return fallback

    # 改写结果再做一次荒谬性检查：空、或者比原问题长太多（说明它在回答问题而不是改写）
    return intent.model_copy(
        update={"rewritten_query": sanitize_rewrite(intent.rewritten_query, question)}
    )
