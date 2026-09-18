"""Step 0 冒烟测试：确认包能导入、目录骨架完整。"""

from pathlib import Path

import app

ROOT = Path(__file__).resolve().parents[1]


def test_version() -> None:
    assert app.__version__ == "0.1.0"


def test_project_skeleton() -> None:
    for rel in ["app", "scripts", "tests", "storage/pdfs", "storage/mineru"]:
        assert (ROOT / rel).exists(), f"缺少 {rel}"


def test_config_files() -> None:
    for rel in ["pyproject.toml", ".gitignore", ".env.example"]:
        assert (ROOT / rel).is_file(), f"缺少 {rel}"

