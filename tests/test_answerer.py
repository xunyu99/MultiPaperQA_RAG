"""Step 6 验收：prompt 拼装、引用校验、无证据短路（不联网）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.generation.answerer import (
    DOWNGRADE_ANSWER,
    EMPTY_ANSWER_FALLBACK,
    MAX_EVIDENCE_IMAGES,
    answer_from_cards,
    assess,
    build_user_content,
    build_user_message,
    finalize_answer,
    load_system_prompt,
    looks_insufficient,
    parse_citations,
)
from app.generation.evidence import EvidenceCard


def _card(
    label: str,
    text: str = "正文",
    section: str = "1 Introduction",
    assets: list[dict[str, Any]] | None = None,
) -> EvidenceCard:
    return EvidenceCard(
        label=label,
        chunk_id=f"p1:c{label[1:].zfill(4)}",
        paper_id="p1",
        paper_title="测试论文",
        citation="Zhang et al. · 2025",
        section_id="p1:s1",
        section_title=section,
        section_order=1,
        chunk_order=int(label[1:]),
        page_start=1,
        page_end=1,
        similarity=0.6,
        text=text,
        assets=assets or [],
    )


def _figure_asset(image_path: str = "images/fig.png") -> dict[str, Any]:
    """一张图资产。行形状照抄 `assets` 表（`expand_assets` 返回的就是整行）。"""
    return {
        "asset_id": "p1:figure0001",
        "asset_type": "figure",
        "caption": "Figure 1: 框架图",
        "image_path": image_path,
    }


def _settings_with_image(tmp_path: Path) -> Settings:
    """建一张**真图**，放在 `{storage_dir}/mineru/{paper_id}/images/` 下 —— 跟线上目录结构一致。"""
    from PIL import Image

    target = tmp_path / "mineru" / "p1" / "images"
    target.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(target / "fig.png")
    return Settings(storage_dir=tmp_path)


class FakeLLM:
    """记下调用次数和参数，返回固定答案 —— 测试里就能验证"有没有真的去调模型"。"""

    def __init__(self, answer: str = "论文提出 X 方法 [E1]。") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def chat(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.answer


# ----------------------------------------------------------------------
# prompt 拼装
# ----------------------------------------------------------------------
def test_user_message_contains_evidence_and_question() -> None:
    message = build_user_message("这篇论文的贡献是什么", [_card("E1", text="我们提出 X。")])
    assert "【证据】" in message and "【问题】" in message
    assert "我们提出 X。" in message
    assert message.rstrip().endswith("这篇论文的贡献是什么")
    assert "[E1]" in message


def test_user_message_survives_latex_braces() -> None:
    """证据里全是 LaTeX 的 `{}` —— 用 str.format 拼会直接崩，所以必须用 f-string 之外的方式。"""
    message = build_user_message("公式是什么", [_card("E1", text=r"$$\frac{a}{b} = \text{x}$$")])
    assert r"\frac{a}{b}" in message


def test_system_prompt_has_the_hard_rules() -> None:
    prompt = load_system_prompt()
    # Step 9 之后拒答走标记协议，prompt 里必须教会模型这两件事：
    # 标记怎么写、拒答时不许带 [En]
    for keyword in ("只用证据", "[[REFUSE:", "拒答", "[E1]", "照抄"):
        assert keyword in prompt, f"系统提示里少了约束：{keyword}"


# ----------------------------------------------------------------------
# 引用校验
# ----------------------------------------------------------------------
def test_parse_citations_matches_known_labels() -> None:
    cards = [_card("E1"), _card("E2"), _card("E3")]
    report = parse_citations("结论 A [E1]，结论 B [E1][E3]。", cards)
    assert report.cited == ["E1", "E3"]
    assert report.unknown == []
    assert report.coverage == 2 / 3


def test_parse_citations_flags_hallucinated_labels() -> None:
    """最硬的幻觉信号：模型引用了证据里根本不存在的编号。"""
    report = parse_citations("结论 [E9]。", [_card("E1"), _card("E2")])
    assert report.unknown == ["E9"]
    assert report.cited == []


def test_parse_citations_detects_insufficient_answer() -> None:
    report = parse_citations("资料不足：证据里没有提到训练用的硬件。", [_card("E1")])
    assert report.insufficient is True
    assert report.looks_uncited is False  # 「资料不足」本来就不该有引用


def test_long_answer_without_any_citation_is_downgraded() -> None:
    """长回答零引用 = 拿不出依据 → **降级成 unknown**（以前只是打个标记）。"""
    report = parse_citations("这篇论文做了很多事，包括 A、B、C。", [_card("E1")])
    assert report.mode == "unknown"
    assert report.unknown_reason == "downgraded"
    assert report.downgraded is True
    assert report.looks_uncited is False  # 它已经不是 grounded 了


def test_looks_insufficient_only_checks_the_head() -> None:
    assert looks_insufficient("资料不足：没有相关证据。")
    assert not looks_insufficient("论文提出了 X 方法。" + "（后面很长）" * 50)


def test_insufficient_prefix_only_counts_the_first_line() -> None:
    """"资料不足"出现在**收尾**不算拒答。

    实测 keyword_en_3 就是这种形状：正文正常带引用，最后一段写着「资料不足的部分：…」。
    老的"前 200 字里任意位置"判据会把这种**有依据的回答**整段丢掉。
    """
    assert looks_insufficient("资料不足：证据里没有提到硬件。")
    assert looks_insufficient("**资料不足：**证据里没有提到硬件。")  # 容忍 markdown 加粗
    assert not looks_insufficient("结论 A [E1]。\n\n**资料不足的部分**：证据中没有说明。")


# ----------------------------------------------------------------------
# 拒答标记（Step 9）
# ----------------------------------------------------------------------
def test_body_figure_is_sent_right_before_its_own_label(tmp_path: Path) -> None:
    """**本体在这个块里** → 图跟着它的 `[En]` 一起发。

    顺序是"证据 → 带编号的标注 → 图 → 问题"：图前面必须先有那段文字，否则模型收到一堆图
    不知道哪张对应哪条证据，就没法把从图里读出来的东西标成 `[En]`，整段回答会被降级掉。
    """
    settings = _settings_with_image(tmp_path)
    cards = [
        _card("E1", text="正文 [FIGURE_REF asset_id=p1:figure0001]", assets=[_figure_asset()])
    ]

    content = build_user_content("图里画了什么", cards, settings)

    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["text", "text", "image_url", "text"]
    assert "【E1 的图】Figure 1: 框架图" in content[1]["text"]
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[3]["text"].endswith("图里画了什么")


def test_mention_only_figure_is_not_sent(tmp_path: Path) -> None:
    """**正文只是提到、本体在别处** → 一张图都不带（PLAN §6.2「资产携带」规则 3）。

    依据是 `card.assets` 只装 `expand_assets` 按**占位符**展开的资产 —— "提到"的那种
    只存在于 `chunk_assets`，根本进不到卡片里。正文那句"如图 1 所示"照旧在，图不跟着走。
    """
    settings = _settings_with_image(tmp_path)
    cards = [_card("E1", text="As shown in Figure 1, UAR improves.", assets=[])]

    content = build_user_content("提升了多少", cards, settings)

    assert isinstance(content, str)  # 纯文本路径，没有 content blocks
    assert "Figure 1" in content


def test_missing_image_file_falls_back_to_text(tmp_path: Path) -> None:
    """图在盘上找不到就跳过 —— 一张图丢了不能让整个请求挂。"""
    settings = _settings_with_image(tmp_path)
    cards = [_card("E1", assets=[_figure_asset("images/not-there.png")])]

    assert isinstance(build_user_content("问题", cards, settings), str)


def test_figures_are_capped(tmp_path: Path) -> None:
    """上限是护栏：实测本体块最多带 4 张，今天不会触发，但别让它无限涨。"""
    settings = _settings_with_image(tmp_path)
    cards = [
        _card(f"E{i}", assets=[dict(_figure_asset(), asset_id=f"p1:figure000{i}")])
        for i in range(1, 7)  # 塞 6 张，只该带 4 张
    ]

    content = build_user_content("问题", cards, settings)

    assert isinstance(content, list)
    assert [p["type"] for p in content].count("image_url") == MAX_EVIDENCE_IMAGES
    # 末尾三项固定是：[第 4 张的标注] [第 4 张的图] [问题]
    assert "【E4 的图】" in content[-3]["text"]
    assert "【E5 的图】" not in str(content)


def test_answer_from_cards_sends_images_when_evidence_has_them(tmp_path: Path) -> None:
    """整条链路：证据带图 → 真发给模型的是 content blocks，不是纯文本。"""
    settings = _settings_with_image(tmp_path)
    llm = FakeLLM()

    answer_from_cards("图里画了什么", [_card("E1", assets=[_figure_asset()])], llm=llm, settings=settings)

    sent = llm.calls[0]["prompt"]
    assert isinstance(sent, list)
    assert any(part["type"] == "image_url" for part in sent)


def test_refusal_marker_sets_unknown_with_reason() -> None:
    report = parse_citations("[[REFUSE:out_of_scope]]\n这个问题不在知识库范围内。", [_card("E1")])
    assert report.mode == "unknown"
    assert report.unknown_reason == "out_of_scope"
    assert report.insufficient is True


def test_refusal_marker_wins_over_citations() -> None:
    """**实测过的真实形状**：拒答的同时还引用着证据。

    模型是在陈述"证据只涉及 X 和 Y"，而 prompt 规则要求"每个事实都标出处"，
    所以它照章办事引用了 E1/E5。按"先看引用"判，这道题会被判成 grounded ——
    所以标记必须排在引用**前面**。
    """
    cards = [_card("E1"), _card("E2")]
    report = parse_citations("[[REFUSE:out_of_scope]]\n证据只涉及 A [E1] 和 B [E2]。", cards)
    assert report.mode == "unknown"
    assert report.unknown_reason == "out_of_scope"
    assert report.cited == ["E1", "E2"]  # 引用照旧解析出来，只是不再决定档位


def test_refusal_marker_parsing_is_lenient() -> None:
    """模型会给它套 markdown 加粗、会大小写不一、会在冒号前后加空格。"""
    cards = [_card("E1")]
    for text in (
        "**[[REFUSE:not_in_corpus]]**\n知识库里没有提到。",
        "[[ refuse : NOT_IN_CORPUS ]]\n知识库里没有提到。",
        "[[REFUSE:not_in_corpus]]",  # 只有标记、没有说明
    ):
        report = parse_citations(text, cards)
        assert report.mode == "unknown"
        assert report.unknown_reason == "not_in_corpus"


def test_refusal_answer_is_stripped_of_marker_and_citations() -> None:
    """展示前剥干净：**模型手滑写了编号，用户也看不到**（结构上保证的那一层）。"""
    cards = [_card("E1"), _card("E2")]
    raw = "**[[REFUSE:out_of_scope]]**\n证据只涉及 A [E1] 和 B [E2]，没有相关内容。"
    shown = finalize_answer(raw, parse_citations(raw, cards))
    assert "REFUSE" not in shown  # 标记没了
    assert "[E1]" not in shown and "[E2]" not in shown  # 残留编号也没了
    assert shown == "证据只涉及 A 和 B，没有相关内容。"  # 顺带清掉剥完留下的空格


def test_refusal_without_body_falls_back_to_fixed_text() -> None:
    """只有标记、说明被剥空时给一句固定文案，不能返回空串。"""
    raw = "[[REFUSE:not_in_corpus]]"
    shown = finalize_answer(raw, parse_citations(raw, [_card("E1")]))
    assert shown.startswith("资料不足")


def test_downgrade_replaces_the_whole_answer() -> None:
    """挂了 grounded 却拿不出引用 → **整段换成固定文案**。

    只换徽标、把那段没依据的正文留着，等于从后门把 general 放回来。
    """
    raw = "这篇论文做了很多事，包括 A、B、C。" * 3
    report = parse_citations(raw, [_card("E1")])
    assert report.mode == "unknown" and report.unknown_reason == "downgraded"
    assert finalize_answer(raw, report) == DOWNGRADE_ANSWER
    assert "这篇论文做了很多事" not in finalize_answer(raw, report)


def test_fabricated_citation_downgrades() -> None:
    """编造编号是硬幻觉信号，同样降级。"""
    report = parse_citations("结论 [E9]。", [_card("E1")])
    assert report.unknown == ["E9"]
    assert report.mode == "unknown" and report.unknown_reason == "downgraded"


def test_grounded_answer_is_left_alone() -> None:
    raw = "论文提出 X [E1]。\n\n**资料不足的部分**：证据中没有说明硬件配置。"
    report = parse_citations(raw, [_card("E1")])
    assert report.mode == "grounded"
    assert finalize_answer(raw, report) == raw


def test_empty_answer_is_marked_failed_not_downgraded() -> None:
    """模型一个字没吐 → failed（"生成失败"），不能混成"依据不足"。"""
    text, report = assess("   ", [_card("E1")])
    assert report.unknown_reason == "failed"
    assert report.mode == "unknown"
    assert text == EMPTY_ANSWER_FALLBACK


# ----------------------------------------------------------------------
# 生成
# ----------------------------------------------------------------------
def test_answer_from_cards_calls_llm_with_system_prompt() -> None:
    llm = FakeLLM()
    result = answer_from_cards("问题", [_card("E1")], llm=llm, settings=Settings())
    assert len(llm.calls) == 1
    assert "只用证据" in llm.calls[0]["system"]
    assert result.answer.startswith("论文提出 X 方法")
    assert result.citations.cited == ["E1"]


def test_no_cards_short_circuits_without_calling_llm() -> None:
    """没证据就不调模型：省钱，也避免它拿自己的知识硬答。"""
    llm = FakeLLM()
    result = answer_from_cards("问题", [], llm=llm, settings=Settings())
    assert llm.calls == []
    assert "资料不足" in result.answer
    assert result.citations.insufficient is True
    assert result.cards == []
