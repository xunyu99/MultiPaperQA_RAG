"""Step 1 验收脚本：建库建表，并打印结果。

运行（在 E:\\agent\\MultiPaperQA_RAG 下）：
    .venv\\Scripts\\python.exe -m scripts.init_db
    .venv\\Scripts\\python.exe -m scripts.init_db --db storage/test.db

幂等：表全部是 CREATE TABLE IF NOT EXISTS，重复跑不会清数据。
想从头来就自己删掉 .db 文件（连同 .db-wal / .db-shm）。
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.db.connection import (
    EXPECTED_TABLES,
    connect,
    count_rows,
    init_db,
    table_names,
    verify_schema,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化 SQLite 库（建表，幂等）")
    parser.add_argument("--db", default=None, help="数据库路径，默认取 .env 的 DB_PATH")
    args = parser.parse_args()

    settings = get_settings()
    conn = connect(args.db)
    try:
        init_db(conn)
        names = table_names(conn)

        print("=== Step 1 · 初始化数据库 ===")
        print(f"数据库    : {args.db or settings.db_path}")
        print(f"预期表数  : {len(EXPECTED_TABLES)}")
        print(f"实际表数  : {len(names)}")
        print()

        print("=== 表与行数 ===")
        for name in names:
            print(f"  {name:<14} {count_rows(conn, name):>6} 行")

        missing = [name for name in EXPECTED_TABLES if name not in names]
        extra = [name for name in names if name not in EXPECTED_TABLES]
        if missing:
            print(f"\n缺少表：{missing}")
        if extra:
            print(f"\n多出来的表（v1 不该有）：{extra}")

        problems = verify_schema(conn)
        if problems:
            print("\n=== schema 过期 ===")
            for problem in problems:
                print(f"  {problem}")
            print(f"\n这是老库没跟着 schema.sql 升级（CREATE TABLE IF NOT EXISTS 不动已存在的表）。")
            print(f"v1 阶段库里没重要数据，直接删掉重建：")
            print(f"  Remove-Item {args.db or settings.db_path} -ErrorAction SilentlyContinue")
            print(f"  然后再跑一次本脚本")

        ok = not missing and not extra and not problems
        print()
        print("结论：", "Step 1 建表通过" if ok else "表结构不对，先别往下走")
        if ok:
            print("下一步：Step 2 · MinerU 解析入库")
        return 0 if ok else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
