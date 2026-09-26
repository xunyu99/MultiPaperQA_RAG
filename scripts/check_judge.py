"""裁判自检（M2 的闸门）：拿 6 条人工已知答案试跑，判据不对就不许跑全量。

用法：
    .venv\\Scripts\\python.exe -m scripts.check_judge

三类都要覆盖：**明显有据 / 明显编造 / 拒答**，外加 correctness 的对与错。
每条跑 2 次看一致性（双判一致率是裁判可信度的一部分，EVAL_PLAN §2.4）。
"""

from __future__ import annotations

from app.config import get_settings
from app.eval.judge import Judge

# (用例名, 裁判, 期望 label, question, second_field, answer)
# second_field：groundedness 用证据，correctness 用参考答案
CASES: list[tuple[str, str, int, str, str, str]] = [
    (
        "论文自身设置算有据（probe 里被判错的回归用例）",
        "groundedness", 1,
        "PE-CLIP 训练用了多大 batch size？",
        "The batch size is set to 8 for the DFEW dataset.",
        "PE-CLIP 的 batch size 是 8。",
    ),
    (
        "编造数字",
        "groundedness", 0,
        "PE-CLIP 训练用了多大 batch size？",
        "The batch size is set to 8 for the DFEW dataset.",
        "PE-CLIP 的 batch size 是 16，学习率 5e-3。",
    ),
    (
        "张冠李戴（把别的论文的结论安到这篇）",
        "groundedness", 0,
        "FDSRM 的核心模块是什么？",
        "ORSANet introduces SSGM and a multi-scale cross-interaction module.",
        "FDSRM 的核心模块是 SSGM（Spatial-Semantic Guidance Module）。",
    ),
    (
        "拒答不算幻觉",
        "groundedness", 1,
        "EmotioNet 的作者邮箱是什么？",
        "The paper describes AU recognition; no contact information is included.",
        "资料中没有相关信息，无法回答。",
    ),
    (
        "correctness 正确",
        "correctness", 1,
        "PE-CLIP 在 DFEW 上用的学习率是多少？",
        "初始学习率 2e-5；adapter 2e-4、prompt 3e-5，batch size 8。",
        "初始学习率是 2e-5，其中 adapter 用 2e-4、prompt 用 3e-5，batch size 8。",
    ),
    (
        "correctness 错误",
        "correctness", 0,
        "PE-CLIP 在 DFEW 上用的学习率是多少？",
        "初始学习率 2e-5；adapter 2e-4、prompt 3e-5，batch size 8。",
        "adapter 的学习率是 2e-3，batch size 16。",
    ),
]


def main() -> int:
    settings = get_settings()
    print(f"裁判模型：{settings.judge_model}")
    print()
    judge = Judge(settings)
    failures = 0
    inconsistent = 0
    for name, kind, expected, question, second, answer in CASES:
        results = [
            judge.groundedness(question, second, answer, question_id=f"selfcheck-{name}")
            if kind == "groundedness"
            else judge.correctness(question, second, answer, question_id=f"selfcheck-{name}")
            for _ in range(2)
        ]
        labels = [item.label for item in results]
        ok = all(label == expected for label in labels)
        stable = len(set(labels)) == 1
        failures += 0 if ok else 1
        inconsistent += 0 if stable else 1
        print(f"[{'通过' if ok else '不通过'}] {name}")
        print(f"         期望={expected} 实得={labels} 理由={results[0].reason[:90]}")
    print()
    print(f"不一致的用例：{inconsistent}   判错的用例：{failures}")
    if failures or inconsistent:
        print("→ **不要跑全量**：先把 prompt/判据改对（改完记得升 PROMPT_VERSION 让缓存失效）")
        return 1
    print("→ 可以跑全量")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
