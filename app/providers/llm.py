"""对话模型接入（LangChain ChatOpenAI）。

两条模型通道，各用各的 key / base_url / 模型：
  生成通道（DeepSeek）：chat() 正文回答 + ask_with_image() 看图
  规划通道（Qwen）    ：structured() 结构化路由、查询改写

为什么分开：
  1. Planner 本质是分类任务，用快而便宜的模型即可；思考模式模型慢，而且限制 tool_choice；
  2. 生成要的是中文表达和长上下文，DeepSeek 更合适；
  3. 两家的配额、限速、故障互相隔离。

设计约定：结构化输出只走 function calling 一条路（tool_choice="auto"），
不保留 response_format=json_object 的第二条通道，避免两套解析逻辑并存产生歧义。
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, AsyncIterator, TypeVar

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from app.config import ProviderConfig, Settings, get_settings

# 送进视觉模型的图片长边上限（像素）
MAX_IMAGE_SIDE = 1024

# structured() 的输出类型
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def image_to_data_url(path: Path, max_side: int = MAX_IMAGE_SIDE) -> str:
    """本地图片 → base64 data URL。

    默认等比缩放：A4 扫描图长边可能到 2000+ px，直接用会吃掉大量上下文并拖慢
    响应，而图表问答在 1024 长边下基本无损。
    """
    from PIL import Image  # 延迟导入，未装 pillow 不影响纯文本路径

    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    width, height = img.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))))

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def flatten_content(value: Any) -> str:
    """把 StrOutputParser 的输出统一成字符串。

    StrOutputParser 只取 message.content，不做展平：纯文本模型返回 str 没问题；
    多模态返回 content block 列表时它会原样交出 list，所以要在这里兜一层。

    **也要能吃 message 对象本身**（`AIMessageChunk` 之类）。踩过的坑：这里少写这一步，
    流式就整个哑掉 —— pydantic 的 BaseModel **是可迭代的**（迭代出 `(字段名, 值)` 元组），
    所以 `for block in value` 拿到的是元组，既不是 str 也不是 dict，每个 chunk 都展平成
    空字符串；上层 `if text:` 全部跳过，一个 token 都不发，最后返回一个空答案。
    它不报错、看着像"模型没说话"，非常难查。
    """
    content = getattr(value, "content", None)
    if content is not None and not isinstance(value, (str, list)):
        value = content
    if isinstance(value, str):
        return value

    parts: list[str] = []
    for block in value:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _describe_failure(raw: Any) -> str:
    """把模型上一次的失败输出压成一段可读文本，用于回灌自纠。"""
    if raw is None:
        return "(无输出)"

    calls = getattr(raw, "tool_calls", None) or []
    if calls:
        return str(calls)[:500]

    content = getattr(raw, "content", "")
    if isinstance(content, list):
        content = flatten_content(content)
    return str(content)[:500] if content else "(空)"


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._str_parser = StrOutputParser()

    # ------------------------------------------------------------------
    # 内部：构造 ChatOpenAI（key / base_url / 代理 / 超时只在这里处理一次）
    # ------------------------------------------------------------------
    def _build(self, provider: ProviderConfig, model: str, temperature: float) -> ChatOpenAI:
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": provider.api_key,
            "base_url": provider.base_url,
            "temperature": temperature,
            "timeout": provider.timeout,
            "max_retries": provider.max_retries,
        }
        if provider.extra_body:
            kwargs["extra_body"] = provider.extra_body
        if provider.proxy:
            kwargs["http_client"] = httpx.Client(
                proxy=provider.proxy,
                timeout=provider.timeout,
            )
        return ChatOpenAI(**kwargs)

    def build_llm(self, model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
        """生成通道（DeepSeek）的原生模型对象，供 bind_tools 等场景使用。"""
        return self._build(
            self.settings.generation_provider,
            model or self.settings.llm_model,
            temperature,
        )

    def build_planner(self, model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
        """规划通道（Qwen）的原生模型对象。"""
        return self._build(
            self.settings.planner_provider,
            model or self.settings.planner_model,
            temperature,
        )

    # ------------------------------------------------------------------
    # 普通生成
    # ------------------------------------------------------------------
    def chat(
        self,
        prompt: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        planner: bool = False,
        max_attempts: int = 2,
    ) -> str:
        """生成通道（默认）/ 规划通道（`planner=True`，走 Qwen）。
        `prompt` 也接受图文混排的 content blocks（证据里有图时用，见 answerer.build_user_content）。

        指代改写这类"小活"要走规划通道：便宜、快，而且不占用生成通道的配额。

        **返回空串时重试一次**：模型偶尔会"什么都不说"（思考模式把 token 全用在
        reasoning 上、上游返回空 choices）。这种失败重发一遍通常就好。
        只重试"拿到了回复但是空的"，不重试异常 —— 超时/限流重试要再等一遍 timeout，
        该由上层决定（改写退回原问题、生成给兜底文案）。
        """
        provider = self.settings.planner_provider if planner else self.settings.generation_provider
        default_model = self.settings.planner_model if planner else self.settings.llm_model
        llm = self._build(provider, model or default_model, temperature)
        if max_tokens:
            llm = llm.bind(max_tokens=max_tokens)

        messages: list[Any] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        return flatten_content(self._str_parser.invoke(llm.invoke(messages)))

    # ------------------------------------------------------------------
    # 结构化输出（function calling，参数全部显式）
    # ------------------------------------------------------------------
    def structured(
        self,
        schema: type[SchemaT],
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_attempts: int = 2,
        tool_choice: str = "auto",
    ) -> SchemaT:
        """按 schema 拿一个结构化对象，走 function calling。

        为什么不用 with_structured_output：

        1. langchain-openai 1.6.2 把它的默认 method 从 function_calling 改成
           json_schema，会发 response_format={"type": "json_schema", ...}，
           DeepSeek 直接 400：This response_format type is unavailable now。
        2. 就算显式指定 method="function_calling"，它内部会强制
           tool_choice=<工具名>，而 deepseek-v4-flash 的思考模式会回：
           Thinking mode does not support this tool_choice。

        所以这里手动 bind_tools，只发服务端确定支持的字段
        （tools + tool_choice="auto"）。

        工具选择不能强制之后，可靠性靠三件事补：

        1. 模型可能不调工具而是直接写文本 —— 重试时用文本明确要求它调；
        2. 拿到 arguments 后过 pydantic 校验（字段类型、必填项）；
        3. 校验失败时把错误和模型上次的输出一起回灌，让它自纠一次。
        """
        llm = self._build(
            self.settings.planner_provider,
            model or self.settings.planner_model,
            0.0,
        )
        tool = convert_to_openai_tool(schema)
        tool_name = tool["function"]["name"]
        runnable = llm.bind_tools([tool], tool_choice=tool_choice)

        messages: list[Any] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            resp = runnable.invoke(messages)
            calls = getattr(resp, "tool_calls", None) or []
            invalid = getattr(resp, "invalid_tool_calls", None) or []

            if calls:
                args = calls[0]["args"]
                # 个别端点不会替我们解析 arguments，这里兜一下
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError as exc:
                        last_error = exc
                        args = None
                if args is not None:
                    try:
                        return schema.model_validate(args)
                    except ValidationError as exc:
                        last_error = exc
            elif invalid:
                last_error = ValueError(f"工具调用参数无法解析：{invalid}")
            else:
                last_error = ValueError(f"模型没有调用工具 {tool_name}，而是直接返回了文本")

            if attempt < max_attempts:
                # 只有"模型压根没调工具"时才能安全地把这条回复追加进历史；
                # 带了 tool_calls 的 AIMessage 必须配 ToolMessage，否则服务端 400。
                if not calls and not invalid:
                    messages.append(resp)
                messages.append(
                    HumanMessage(
                        content=(
                            f"你上一次没有按要求调用工具 {tool_name}。\n"
                            f"问题：{last_error}\n"
                            f"你上次的输出：{_describe_failure(resp)}\n"
                            f"请只通过调用 {tool_name} 工具来输出结果，content 里不要写任何文字。"
                        )
                    )
                )

        raise ValueError(f"结构化输出失败（重试 {max_attempts} 次）：{last_error}")

    # ------------------------------------------------------------------
    # 看图（复用同一个模型）
    # ------------------------------------------------------------------
    def ask_with_image(
        self,
        question: str,
        image_path: Path,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 512,
    ) -> str:
        llm = self._build(
            self.settings.generation_provider,
            model or self.settings.llm_model,
            0.1,
        )
        if max_tokens:
            llm = llm.bind(max_tokens=max_tokens)

        content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_to_data_url(Path(image_path))}},
        ]
        messages: list[Any] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=content))

        for _ in range(max(1, max_attempts)):
            text = flatten_content(self._str_parser.invoke(llm.invoke(messages)))
            if text.strip():
                return text
        return ""

    async def chat_stream(
        self,
        prompt: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_attempts: int = 2,
    ) -> AsyncIterator[str]:
        """流式版 `chat()`：逐段吐文本，给 `/chat` 的 SSE 用。
        `prompt` 同样接受图文混排的 content blocks（理由见 `chat()`）。

        **只吐增量，不吐累积**：`astream` 给的是 chunk，我们 yield 其中的文本。
        调用方自己拼完整回答（拼完还要过一遍引用校验）。

        注：`temperature` / `model` 的默认值刻意和 `chat()` 保持一致，
        两条路径必须产出一致风格的答案，否则"流式看着好、存下来不一样"。

        **一个 token 都没吐出来时重试一次**（理由同 `chat()`）。只在第一次完全
        没输出时重试 —— 已经吐给前端的增量收不回来，重试会把内容重复一遍。
        """
        llm = self._build(
            self.settings.generation_provider,
            model or self.settings.llm_model,
            temperature,
        )
        messages: list[Any] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        for attempt in range(1, max(1, max_attempts) + 1):
            produced = False
            async for chunk in llm.astream(messages):
                # 传 chunk 本身就行：flatten_content 会先取 .content（见它的 docstring，
                # 这里踩过坑 —— 直接迭代 message 对象拿到的是字段元组，展平出来是空的）
                text = flatten_content(chunk)
                if text:
                    produced = True
                    yield text
            if produced or attempt >= max_attempts:
                return
