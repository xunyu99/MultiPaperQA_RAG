"""Step 0 验收脚本：确认解释器、工程根目录、storage 目录都对。

运行：
    .venv\\Scripts\\python.exe -m scripts.check_env
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "app",
    "app/providers",
    "app/db/repositories",
    "app/ingest",
    "app/retrieval",
    "app/generation",
    "app/graph",
    "app/api",
    "app/eval",
    "scripts",
    "tests",
    "storage",
    "storage/pdfs",
    "storage/mineru",
]


def main() -> int:
    print("=== Step 0 环境自检 ===")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"解释器路径  : {sys.executable}")
    print(f"平台        : {platform.system()} {platform.release()}")
    print(f"工程根目录  : {ROOT}")

    in_venv = Path(sys.executable).parent.parent.name == ".venv"
    print(f"虚拟环境    : {'是' if in_venv else '否（请用 .venv 里的 python 运行本脚本）'}")

    ok = True
    print("\n=== 目录检查 ===")
    for rel in REQUIRED_DIRS:
        exists = (ROOT / rel).is_dir()
        ok &= exists
        print(f"  [{'OK' if exists else '缺失'}] {rel}")

    print("\n=== 其它文件 ===")
    for rel in ["pyproject.toml", ".gitignore", ".env.example"]:
        exists = (ROOT / rel).is_file()
        ok &= exists
        print(f"  [{'OK' if exists else '缺失'}] {rel}")

    has_env = (ROOT / ".env").is_file()
    print(f"  [{'OK' if has_env else '待办'}] .env（还没创建的话，Step 1 需要）")

    print("\n结论：", "Step 0 通过" if ok else "有缺失，先别往下走")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

