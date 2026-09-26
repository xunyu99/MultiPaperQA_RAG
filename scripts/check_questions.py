"""题库自检闸门：**改完题、付钱跑之前必须先过这一关**。

为什么要有它（2026-09-26，血的教训）：重写 keyword 题时只换了问题、没换 `gold` 和
`paper_id`，三道题被锁到了别的论文上，模型"正确地"拒答，却被判成答错；`reference`
里还写进了 gold 章节里根本没有的事实（把正文 4.2 的初始学习率写进"表 2"的答案）。
结果白跑了两轮付费评测。**题库错误必须在不花钱的环节被发现。**

检查四类：

1. **gold 章节存在**：`sections` 表里真的有那么一节（拼错一个字就等于永远不可能命中）。
2. **问题指向的论文 = gold 的论文**：问题里的 ASCII 术语（缩写、论文名、模型名）
   至少要有一个出现在 gold 论文的正文里；一个都不出现 → ERROR（可能就是这次那种
   "问 EmotioNet 却标到 ORSANet"）。
3. **keypoints / reference 里的"硬 token"（带数字的、或大写缩写）必须在论文里出现**：
   全篇都找不到 → ERROR（多半是我编的或写错了论文）；只在别的章节出现、gold 章节没有
   → WARN（可能"答案的依据不在你标的这一节"，`asset_1` 那种就是这样）。
4. **结构和配额**：id 唯一、字段齐全、分层配额与题数。

用法：
    .venv\\Scripts\\python.exe -m scripts.check_questions
退出码非 0 = 有 ERROR，**不许跑付费评测**。
"""

from __future__ import annotations

import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "app" / "eval" / "questions.jsonl"

# 论文名 / 通用词不进"术语"判定，否则任何一篇都能"命中"
_STOPWORDS = {
    "THE", "AND", "FOR", "WITH", "THIS", "THAT", "WHAT", "HOW", "WHY", "WHICH", "FROM",
    "PAPER", "METHOD", "MODEL", "RESULTS", "SECTION", "TABLE", "FIGURE", "IEEE", "ACM",
}
_ASCII_TERM = re.compile(r"[A-Za-z][A-Za-z0-9\-_.]{2,}")
# "硬 token"分两种，**判据强度不同**（2026-09-26 收紧）：
#   - 缩写（KSDA / SPADE / FFHQ-Text）：全篇找不到 = ERROR，那基本就是写错论文或编的；
#   - 数字（2e-4 / 0.85 / 1500万）：论文里写法五花八门（$2 \times 10^{-4}$、"15 million"），
#     找不到多半是写法差异 → 只 WARN，别拿它拦人（实测第一版把 16 条好题全判成了坏题）。
_ACRONYM = re.compile(r"\b[A-Z]{2,}[A-Za-z0-9\-]*\b")
_NUMBER = re.compile(r"\b\d+(?:[eE][-+]?\d+|\.\d+)\b")


def _load_chunks(con: sqlite3.Connection) -> dict[str, list[tuple[str, str, str]]]:
    """paper_id → [(chunk_id, section 标题, 正文)]"""
    out: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
    # 表格 / 图的**本体在 assets 表**，chunk 正文里只有 [TABLE_REF ...] 占位符 ——
    # 不把 assets 算进来的话，所有"表里的数值"都会被误判成"全篇找不到"（实测踩过）。
    rows = con.execute(
        """
        select c.paper_id, c.chunk_id, coalesce(s.title, ''), coalesce(c.content, '') || ' ' ||
               coalesce(c.index_text, '') || ' ' ||
               coalesce((
                   select group_concat(coalesce(a.caption,'') || ' ' || coalesce(a.raw_content,''), ' ')
                   from chunk_assets ca join assets a on a.asset_id = ca.asset_id
                   where ca.chunk_id = c.chunk_id
               ), '')
        from chunks c
        left join sections s on s.section_id = (
            select b.section_id from blocks b where b.block_id = c.block_ids limit 1
        )
        """
    ).fetchall()
    for paper_id, chunk_id, section, text in rows:
        out[paper_id].append((chunk_id, section or "", text))
    return out


def main() -> int:
    items = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    settings_db = ROOT / "storage" / "app.db"
    con = sqlite3.connect(f"file:{settings_db}?mode=ro", uri=True)

    sections = collections.defaultdict(list)
    for paper_id, title in con.execute("select paper_id, title from sections"):
        sections[paper_id].append(title or "")
    chunks = _load_chunks(con)
    paper_text = {pid: " ".join(text for _, _, text in rows) for pid, rows in chunks.items()}
    papers = {pid for (pid,) in con.execute("select paper_id from papers")}

    errors: list[str] = []
    warnings: list[str] = []

    seen = collections.Counter()
    for item in items:
        qid = item["id"]
        seen[qid] += 1
        for field in ("id", "type", "layer", "difficulty", "question", "expect_insufficient", "gold", "keypoints", "note"):
            if field not in item:
                errors.append(f"{qid}: 缺字段 {field}")
        gold = item.get("gold") or []
        # 1. gold 章节存在
        for g in gold:
            if g["paper_id"] not in papers:
                errors.append(f"{qid}: gold 论文不存在 {g['paper_id']}")
            elif not any(g["section"] in title or title in g["section"] for title in sections[g["paper_id"]]):
                errors.append(f"{qid}: gold 章节不存在 {g['paper_id']} / {g['section']!r}")
        # 2. 问题指向的论文 == gold 的论文
        terms = [t for t in _ASCII_TERM.findall(item["question"]) if t.upper() not in _STOPWORDS]
        if gold and terms:
            gold_papers = {g["paper_id"] for g in gold}
            hit = [t for t in terms if any(t.lower() in paper_text.get(p, "").lower() for p in gold_papers)]
            if not hit:
                errors.append(
                    f"{qid}: 问题里的术语 {terms[:4]} 在 gold 论文 {sorted(gold_papers)} 的正文里一个都找不到"
                    "（题面和 gold 可能指向不同论文）"
                )
        # 3. keypoints / reference 的硬 token 是否在场
        gold_sections = {(g["paper_id"], g["section"]) for g in gold}
        in_gold_section = " ".join(
            text
            for pid, rows in chunks.items()
            for _, section, text in rows
            if any(pid == g and sec in section for g, sec in gold_sections)
        )
        hard: list[str] = []
        for value in [*(item.get("keypoints") or []), item.get("reference") or ""]:
            for part in (value if isinstance(value, list) else [value]):
                hard += _ACRONYM.findall(str(part)) + _NUMBER.findall(str(part))
        all_text = " ".join(text for rows in chunks.values() for _, _, text in rows)
        for token in sorted(set(hard)):
            in_paper = token.lower() in all_text.lower()
            is_number = bool(_NUMBER.fullmatch(token))
            if not in_paper and is_number:
                warnings.append(f"{qid}: 数字 {token!r} 字面找不到（多半是写法差异，如 $2 \\times 10^{{-4}}$）")
            elif not in_paper:
                errors.append(f"{qid}: reference/keypoints 里的 {token!r} 全篇找不到（疑似编造或写错论文）")
            elif token not in in_gold_section:
                warnings.append(f"{qid}: {token!r} 不在 gold 章节里（依据可能在别处）")

    for qid, count in seen.items():
        if count > 1:
            errors.append(f"重复 id：{qid} × {count}")

    layers = collections.Counter(item["layer"] for item in items)
    print(f"题目总数：{len(items)}   分层：{dict(layers)}")
    print(f"ERROR {len(errors)} 条，WARN {len(warnings)} 条")
    for line in errors:
        print("  [ERROR]", line)
    for line in warnings:
        print("  [WARN ]", line)
    if errors:
        print()
        print("→ 有 ERROR：**不许跑付费评测**，先修题")
        return 1
    print()
    print("→ 闸门通过（WARN 请人工扫一眼：确认 reference 的依据确实在对应位置）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
