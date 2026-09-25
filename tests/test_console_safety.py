"""Windows 控制台的编码坑：两层防线，别只做一层。

踩过的坑：`print("✅ ...")` 直接 `UnicodeEncodeError: 'gbk' codec can't encode
character '\\u2705'`，脚本跑到一半崩掉。

**第一层（根治）**：`app/console.py` 在入口把 stdout 的错误处理改成 `replace`，
任何字符都不会再让脚本崩 —— 这才是根治，因为 MinerU 抽出来的正文里就有 `∗` 这种
GBK 装不下的字符，靠"源码别写怪字符"根本防不住。

**第二层（本测试）**：`app/` 和 `scripts/` 里我们自己写的字符串仍然是 GBK 安全的，
免得输出里出现一堆 `?`。`tests/` 不在检查范围内 —— 那里的样例数据是**真实论文文本**，
本来就该包含 `∗` 这类字符，改掉反而不真实。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("app", "scripts")


def _python_files() -> list[Path]:
    files: list[Path] = []
    for name in SOURCE_DIRS:
        files.extend(sorted((ROOT / name).rglob("*.py")))
    return files


def test_python_sources_are_gbk_encodable() -> None:
    offenders: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        try:
            text.encode("gbk")
        except UnicodeEncodeError as exc:
            bad = exc.object[exc.start : exc.end]
            line = text[: exc.start].count("\n") + 1
            offenders.append(f"  {path.relative_to(ROOT)}:{line}  字符 {bad!r}（U+{ord(bad[0]):04X}）")

    assert not offenders, (
        "下面这些文件里有 GBK 编不了的字符，脚本 print 出来会 UnicodeEncodeError：\n"
        + "\n".join(offenders)
    )


def test_stdout_is_reconfigured_to_never_crash() -> None:
    """入口就把错误处理设成 replace —— 这是真正的兜底。"""
    from app.console import make_streams_safe

    # 幂等：重复调用不该报错
    make_streams_safe()
    make_streams_safe()

    import sys

    if hasattr(sys.stdout, "errors"):
        assert sys.stdout.errors == "replace"
