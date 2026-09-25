"""Step 2 验收（配套）：版本指纹与 paper_id 推导。"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.ingest import versioning
from app.ingest.versioning import (
    default_paper_id,
    index_version,
    mineru_version,
    parse_version,
    sha256_file,
)


def test_sha256_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    a = tmp_path / "a.pdf"
    a.write_bytes(b"%PDF-1.4 hello")
    b = tmp_path / "b.pdf"
    b.write_bytes(b"%PDF-1.4 hello!")

    assert sha256_file(a) == sha256_file(a)
    assert sha256_file(a) != sha256_file(b)
    assert len(sha256_file(a)) == 64


def test_default_paper_id_is_safe(tmp_path: Path) -> None:
    ascii_pdf = tmp_path / "Attention Is All You Need.pdf"
    ascii_pdf.write_bytes(b"x")
    assert default_paper_id(ascii_pdf) == "attention-is-all-you-need"

    cjk_pdf = tmp_path / "论文 A（修订版）.pdf"
    cjk_pdf.write_bytes(b"x")
    # 中文被清掉，剩下的 ASCII 还是可用的
    assert default_paper_id(cjk_pdf) == "a"

    pure_cjk = tmp_path / "论文修订版.pdf"
    pure_cjk.write_bytes(b"x")
    paper_id = default_paper_id(pure_cjk)
    # 一个 ASCII 字符都不剩 → 退回内容哈希前 12 位
    assert paper_id.startswith("paper-")
    assert len(paper_id) == len("paper-") + 12


def test_parse_version_tracks_mineru_settings() -> None:
    vlm = parse_version(Settings(mineru_model_version="vlm"))
    pipeline = parse_version(Settings(mineru_model_version="pipeline"))
    assert vlm != pipeline
    assert parse_version(Settings(mineru_model_version="vlm")) == vlm

    # 关掉表格开关也应该换指纹
    assert parse_version(Settings(mineru_enable_table=False)) != vlm


def test_mineru_version_ignores_our_own_code_changes(monkeypatch) -> None:
    """改 converter 不能顶掉 MinerU 缓存 —— 否则改一行本地解析代码就得重花一次钱。"""
    settings = Settings(mineru_model_version="vlm")
    before_cache_key = mineru_version(settings)
    before_audit = parse_version(settings)

    monkeypatch.setattr(versioning, "CONVERTER_VERSION", "999")

    assert mineru_version(settings) == before_cache_key  # 缓存键不动
    assert parse_version(settings) != before_audit  # 但产物来源记录要变


def test_parse_version_is_mineru_version_plus_converter() -> None:
    settings = Settings(mineru_model_version="vlm")
    assert parse_version(settings).startswith(mineru_version(settings))
    assert "converter:" in parse_version(settings)
    assert "converter:" not in mineru_version(settings)


def test_index_version_tracks_splitter_and_embedding() -> None:
    base = index_version(Settings(chunk_target_tokens=500))
    assert base != index_version(Settings(chunk_target_tokens=300))
    assert base != index_version(Settings(embedding_model="text-embedding-v3"))
    assert base != index_version(Settings(embedding_dim=768))
    assert base == index_version(Settings(chunk_target_tokens=500))


def test_two_fingerprints_are_independent() -> None:
    """改切分参数只该动 index_version，不该动 parse_version —— 这才省得掉 MinerU 那次调用。"""
    a = Settings(chunk_target_tokens=500)
    b = Settings(chunk_target_tokens=300)
    assert parse_version(a) == parse_version(b)
    assert index_version(a) != index_version(b)
