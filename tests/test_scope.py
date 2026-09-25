"""论文范围解析：从问题里认出用户问的是哪篇（哪几篇）。"""

from __future__ import annotations

from app.config import Settings
from app.retrieval.retriever import _quota_and_total, _search_in_scope
from app.retrieval.scope import build_aliases, resolve_scope
from app.retrieval.vector_index import SearchHit

PAPERS = [
    {"paper_id": "emotion-qwen-vl",
     "title": "Emotion-Qwen-VL: A Fully Fine-Tuned Multimodal Large Language Model for Micro-Expression Visual Question Answering"},
    {"paper_id": "pe-clip",
     "title": "PE-CLIP: A Parameter-Eficient Fine-Tuning of Vision Language Models for Dynamic Facial Expression Recognition"},
    {"paper_id": "orsanet_rethinking_occlusion_in_fer",
     "title": "Rethinking Occlusion in FER: A Semantic-Aware Perspective and Go Beyond"},
]


# ----------------------------------------------------------------------
# 别名
# ----------------------------------------------------------------------
def test_aliases_include_id_and_subtitle() -> None:
    aliases = build_aliases(PAPERS[2])
    assert "orsanet_rethinking_occlusion_in_fer" in aliases
    assert "orsanet rethinking occlusion in fer" in aliases  # 分隔符换了也能认
    assert "rethinking occlusion in fer" in aliases  # 冒号前的短名


def test_alias_covers_title_head_without_subtitle_separator() -> None:
    """标题没有冒号时（实测 `AN EVALUATION OF A VISUAL QUESTION ANSWERING STRATEGY ...`），
    只靠"冒号前"和"整条标题"都认不出来 —— 得拿开头几个词当别名。"""
    paper = {
        "paper_id": "an_evaluation_of_a_vqa_strategy_for_zero-shot_fer",
        "title": "AN EVALUATION OF A VISUAL QUESTION ANSWERING STRATEGY FOR ZERO-SHOT "
                 "FACIAL EXPRESSION RECOGNITION IN STILL IMAGES",
    }
    aliases = build_aliases(paper)
    assert "an evaluation of a visual question" in aliases
    assert resolve_scope("AN EVALUATION OF A VISUAL QUESTION ANSWERING STRATEGY 说了什么",
                         [paper]) == [paper["paper_id"]]


def test_short_aliases_are_dropped() -> None:
    """`FER` 这种词会命中一堆论文，必须挡掉。"""
    assert build_aliases({"paper_id": "fer", "title": "FER"}) == set()


# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------
def test_resolves_paper_id_mentioned_in_a_chinese_question() -> None:
    """中文问题里夹着论文短名 —— 实测最常见的问法。"""
    assert resolve_scope("emotion-qwen-vl这篇论文的贡献是什么", PAPERS) == ["emotion-qwen-vl"]
    assert resolve_scope("PE-CLIP 训练用的 batch size 是多少", PAPERS) == ["pe-clip"]


def test_resolves_subtitle_mention() -> None:
    assert resolve_scope("Rethinking Occlusion in FER 讲了什么", PAPERS) == [
        "orsanet_rethinking_occlusion_in_fer"
    ]


def test_resolves_multiple_papers_for_comparison() -> None:
    """对比类问题会同时提到两篇 —— 必须两篇都搜。"""
    scope = resolve_scope("PE-CLIP 和 Emotion-Qwen-VL 有什么区别", PAPERS)
    assert sorted(scope) == ["emotion-qwen-vl", "pe-clip"]


def test_returns_empty_when_nothing_matches() -> None:
    """解析不出来 = 全库。宁可搜宽，不要猜错论文。"""
    assert resolve_scope("红烧肉怎么做", PAPERS) == []
    assert resolve_scope("这两篇论文的方法有什么不同", PAPERS) == []


def test_explicit_scope_wins() -> None:
    assert resolve_scope("随便问", PAPERS, explicit=["pe-clip"]) == ["pe-clip"]


# ----------------------------------------------------------------------
# 多论文配额
# ----------------------------------------------------------------------
class FakeIndex:
    """记录每次查询的范围，用来验证"每篇都查、每篇都有配额"。"""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str | None]] = []

    def search_by_vector(self, vector, k, paper_id=None):  # noqa: ANN001
        self.calls.append((k, paper_id))
        return [SearchHit(chunk_id=f"{paper_id or 'all'}:c{i}", similarity=1 - i * 0.1,
                          distance=i * 0.1, metadata={}) for i in range(k)]


def test_每个论文都要查到_and_配额均分() -> None:
    index = FakeIndex()
    _search_in_scope(index, [0.0], quota=2, scope=["p1", "p2", "p3"])
    # 每篇取 quota 条（quota 由 PER_PAPER_K 决定，见 _quota_and_total）
    assert index.calls == [(2, "p1"), (2, "p2"), (2, "p3")]


def test_single_paper_scope_uses_full_k() -> None:
    index = FakeIndex()
    _search_in_scope(index, [0.0], quota=3, scope=["p1"])
    assert index.calls == [(3, "p1")]


def test_empty_scope_searches_everything() -> None:
    index = FakeIndex()
    _search_in_scope(index, [0.0], quota=3, scope=[])
    assert index.calls == [(3, None)]


def test_merged_results_are_sorted_and_left_wide() -> None:
    """合并后按距离统一排序（不是按论文顺序拼）。

    `_search_in_scope` 现在返回**宽候选**（不去重、不截断到 k），截断交给
    `_select` —— 开同章节去重时要先在宽候选里挑，提前砍到 k 就没得挑了。
    关去重时每篇深度就是 quota，合并后 = N x quota 条，`_select` 再按 total 截断。
    """
    index = FakeIndex()
    hits = _search_in_scope(index, [0.0], quota=2, scope=["p1", "p2", "p3"])
    assert len(hits) == 6  # 3 篇各 2 条
    assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)


def test_多篇按每篇配额_而不是均分总数() -> None:
    """多篇时总数 = 篇数 x `PER_PAPER_K`：给每篇固定名额，别按相似度全局截断。

    老做法是 ceil(k/N)，k=3、N=2 时每篇只剩 1 条，对比题只有一条证据；
    新做法每篇 2 条、一共 4 条，篇数越多总条数越多（代价是上下文变长）。

    这里只断言**策略**（入参 k 怎么分配），不写死 `TOP_K` 的具体值 ——
    那个由 `.env` 决定，写死会让测试跟着配置文件一起碎。
    """
    settings = Settings()
    k = settings.top_k
    # 单篇 / 全库：就是 TOP_K
    assert _quota_and_total(settings, k, []) == (k, k)
    assert _quota_and_total(settings, k, ["p1"]) == (k, k)
    # 多篇：每篇 PER_PAPER_K
    assert _quota_and_total(settings, k, ["p1", "p2"]) == (settings.per_paper_k, settings.per_paper_k * 2)
    assert _quota_and_total(settings, k, ["p1", "p2", "p3"]) == (
        settings.per_paper_k, settings.per_paper_k * 3,
    )


def test_默认配额是单篇5_多篇每篇3() -> None:
    """代码里的默认值（不看 .env）：`TOP_K=5`、`PER_PAPER_K=3`。

    `TOP_K` 试过 3，Recall 从 80% 掉到 60%，退回 5；`PER_PAPER_K` 试过 2，
    每篇两个名额会被同一章节的重复块吃光，所以给 3。
    """
    settings = Settings(_env_file=None)
    assert settings.top_k == 5
    assert settings.per_paper_k == 3
