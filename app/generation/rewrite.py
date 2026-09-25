"""指代改写 / 意图抽取的**共用零件**：历史窗口 + 模型输出清洗。

真正的模型调用在 `intent.py`（一次结构化输出同时拿改写、闲聊判定、资产编号）。
这里只留两件被它复用的小事，以及"这个窗口多大"的唯一常量。

**历史窗口只给 `HISTORY_ROUNDS` 轮**。历史**只喂入口这一步**，不进检索、也不进生成：
检索和生成拿到的都是改写后的自包含问句 + 本轮的证据。

**为什么清洗不能省**：结构化输出也会跑偏 —— 模型可能把"改写"写成一段回答
（比原问题长得多），或者带上"改写结果："这种前缀。前者必须退回原问题，
否则检索会被一段答案带偏。
"""

from __future__ import annotations

# 历史窗口：只给改写模型最近 2 轮（一轮 = 用户一条 + 助手一条，共 4 条消息）。
#
# 为什么是 2 轮而不是 1 轮：`messages` 里存的是**用户原话**（见 save_turn），追问链里的
# 原话自己就带悬空指代（"它在 DFEW 上呢？"里的"它"）。只给最后一轮的话，用户接着说
# "那 F1 呢"就没有锚点了 —— 2 轮等于多留一个更早的话题锚点。
#
# 为什么不怕多带：这段历史只进改写模型（小模型 + 每条截 300 字），2 轮约 1200 字，
# 一次很短的调用；真正吃 token 的生成侧**不带历史**。这个数字是窗口的唯一来源，
# sessions_repo.history 由调用方把同一个常量传下去，别在两处各写一个数。
HISTORY_ROUNDS = 2

def format_history(history: list[dict[str, str]], rounds: int = HISTORY_ROUNDS) -> str:
    """历史拼成纯文本，**只取最近 rounds 轮**。

    只给问题和答案，不给证据卡 —— 证据每轮都重新检索，把旧的塞进去只会让改写模型
    分心（还白烧 token）。

    内容为空的消息直接跳过：库里实测有一批空的助手回复（流式中断时落的库），
    它们提供不了任何锚点，却会白占窗口里的一个位置。
    """
    if rounds <= 0:
        return ""
    lines = []
    for item in history[-rounds * 2 :]:
        content = " ".join((item.get("content") or "").split())
        if not content:
            continue
        role = "用户" if item.get("role") == "user" else "助手"
        lines.append(f"{role}：{content[:300]}")
    return "\n".join(lines)


def sanitize_rewrite(raw: str, fallback: str) -> str:
    """把模型输出收拾干净：去空行、去"改写结果："前缀、去引号。

    再做一道**荒谬性检查**：输出比原问题长太多（说明它在回答问题而不是改写）
    或者为空，就直接用原问题。
    """
    for line in (raw or "").splitlines():
        text = line.strip()
        if not text:
            continue
        for prefix in ("改写结果：", "改写结果:", "改写：", "答案：", "答："):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        text = text.strip("\"'“”‘’ 　")
        if not text:
            continue
        if len(text) > max(len(fallback) * 2, len(fallback) + 40):
            return fallback
        return text
    return fallback
