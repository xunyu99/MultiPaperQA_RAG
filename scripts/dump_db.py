"""看库里的数据（通用表格查看器）。

SQLite 是**单文件数据库**：7 张表全在 `storage/app.db` 这一个文件里，
项目目录下不会有 per-table 的文件，`.gitignore` 也把 `storage/*` 排除了 ——
所以想看数据只能用工具打开，本脚本就是干这个的。

用法：
    # 1. 有哪些表、各多少行
    .venv\\Scripts\\python.exe -m scripts.dump_db

    # 2. 打印建表语句（字段、类型、注释、约束都在里面）
    .venv\\Scripts\\python.exe -m scripts.dump_db --schema
    .venv\\Scripts\\python.exe -m scripts.dump_db --schema papers

    # 3. 看某张表（默认前 20 行）
    .venv\\Scripts\\python.exe -m scripts.dump_db papers
    .venv\\Scripts\\python.exe -m scripts.dump_db chunks --limit 5

    # 4. 只看某篇论文的某几列
    .venv\\Scripts\\python.exe -m scripts.dump_db blocks --where "paper_id='pe-clip'" --columns order_index,block_type,page_idx --limit 15

    # 5. 正文太长被截断了，想看全文
    .venv\\Scripts\\python.exe -m scripts.dump_db blocks --where "block_type='table'" --full

    # 6. 导出成 CSV，用 Excel 直接打开看（推荐，比命令行直观）
    .venv\\Scripts\\python.exe -m scripts.dump_db --csv
    .venv\\Scripts\\python.exe -m scripts.dump_db papers --csv

通用的看数据用它；看块清单/章节树那种带格式的视图，用 `dump_blocks` / `dump_sections`。
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from app.config import get_settings
from app.db.connection import connect, count_rows, table_names

# 一句话说明，看表的时候不用回头翻 PLAN
TABLE_NOTES = {
    "papers": "论文元信息 + 三个复用指纹",
    "sections": "章节树（splitter / 章节路径 / 证据卡定位串读它）",
    "blocks": "原子块，永不切分的最小单位",
    "chunks": "检索与生成单元（Step 3 才有数据）",
    "assets": "表格 HTML / 公式 LaTeX / 图片（Step 3 才有数据）",
    "chunk_assets": "chunk 与资产的绑定（Step 3 才有数据）",
    "parse_cache": "MinerU 产物复用表，删论文时保留",
}

_NULL = "NULL"


def main() -> int:
    parser = argparse.ArgumentParser(description="查看 SQLite 库里的数据")
    parser.add_argument("table", nargs="?", default=None, help="表名；不给就列出所有表")
    parser.add_argument("--where", default=None, help='SQL 过滤条件，如 "paper_id=\'pe-clip\'"')
    parser.add_argument("--columns", default=None, help="只看这几列，逗号分隔")
    parser.add_argument("--limit", type=int, default=20, help="最多打印几行（默认 20）")
    parser.add_argument("--width", type=int, default=40, help="单列最大宽度（默认 40）")
    parser.add_argument("--full", action="store_true", help="不截断")
    parser.add_argument("--schema", action="store_true", help="打印建表语句")
    parser.add_argument(
        "--csv",
        nargs="?",
        const="storage/exports",
        default=None,
        metavar="输出目录",
        help="导出成 CSV（Excel 能直接打开）；不给目录就用 storage/exports",
    )
    parser.add_argument("--db", default=None, help="数据库路径，默认取 .env 的 DB_PATH")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else get_settings().db_path
    conn = connect(args.db)
    try:
        names = table_names(conn)
        if args.schema:
            return _print_schema(conn, names, args.table)
        if args.csv:
            return _export_csv(conn, names, args.table, Path(args.csv))
        if not args.table:
            return _list_tables(conn, names, db_path)
        if args.table not in names:
            print(f"没有这张表：{args.table}")
            print("库里有：" + " / ".join(names))
            return 1
        return _print_rows(conn, args)
    finally:
        conn.close()


def _list_tables(conn: sqlite3.Connection, names: list[str], db_path: Path) -> int:
    print(f"数据库：{db_path}")
    exists = db_path.is_file()
    print(f"文件  ：{'存在 ' + str(round(db_path.stat().st_size / 1024, 1)) + ' KB' if exists else '不存在（先跑 scripts.init_db）'}")
    print()
    print(f"{'表':<14}{'行数':>7}   说明")
    print("-" * 76)
    for name in names:
        print(f"{name:<14}{count_rows(conn, name):>7}   {TABLE_NOTES.get(name, '')}")
    print()
    print("看某张表：python -m scripts.dump_db <表名>    建表语句：--schema   导出 Excel：--csv")
    return 0


def _export_csv(
    conn: sqlite3.Connection,
    names: list[str],
    table: str | None,
    dest_dir: Path,
) -> int:
    """每张表导一个 CSV。Excel 是双击就能看的，比让人装数据库 GUI 省事。

    编码用 utf-8-sig（带 BOM）—— 不带 BOM 的话，Windows 版 Excel 会把中文读成乱码。
    这是 Excel 的历史包袱，不是我们的选择。
    """
    targets = [table] if table else names
    if table and table not in names:
        print(f"没有这张表：{table}")
        print("库里有：" + " / ".join(names))
        return 1

    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int, Path]] = []
    for name in targets:
        rows = conn.execute(f"SELECT * FROM {name}").fetchall()  # 表名已过白名单
        path = dest_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            if rows:
                writer.writerow(list(rows[0].keys()))
                for row in rows:
                    writer.writerow(["" if value is None else value for value in row])
            else:
                # 空表也要写表头，否则 Excel 打开是一片空白，看着像导出失败
                columns = [r["name"] for r in conn.execute(f"PRAGMA table_info({name})")]
                writer.writerow(columns)
        written.append((name, len(rows), path.resolve()))

    print(f"已导出到：{dest_dir.resolve()}")
    print()
    for name, count, path in written:
        print(f"  {path.name:<18}{count:>6} 行   {path}")
    print()
    print("用 Excel 双击打开对应文件就行（中文不会乱码）。")
    return 0


def _print_schema(conn: sqlite3.Connection, names: list[str], table: str | None) -> int:
    targets = [table] if table else names
    for name in targets:
        if name not in names:
            print(f"没有这张表：{name}")
            return 1
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        print(f"--- {name} ---")
        print(row["sql"] if row and row["sql"] else "(没有建表语句)")
        print()
    return 0


def _print_rows(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    columns = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else []
    select = ", ".join(columns) if columns else "*"

    # 表名已经过白名单校验；--where / --columns 是开发者手写的调试输入，不做转义
    where = f" WHERE {args.where}" if args.where else ""
    total = conn.execute(f"SELECT count(*) AS n FROM {args.table}{where}").fetchone()["n"]
    sql = f"SELECT {select} FROM {args.table}{where} LIMIT {max(0, int(args.limit))}"
    rows = conn.execute(sql).fetchall()

    print(f"{args.table}：匹配 {total} 行，显示前 {len(rows)} 行")
    if not rows:
        print("（没有数据）")
        return 0

    headers = list(rows[0].keys())
    body = [[_cell(row[h], args.width, args.full) for h in headers] for row in rows]
    widths = [max(len(h), max(len(line[i]) for line in body)) for i, h in enumerate(headers)]

    print(_rule(widths))
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(_rule(widths))
    for line in body:
        print("| " + " | ".join(line[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(_rule(widths))
    if total > len(rows):
        print(f"（还有 {total - len(rows)} 行没显示，用 --limit 调整）")
    return 0


def _rule(widths: list[int]) -> str:
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"


def _cell(value: object, width: int, full: bool) -> str:
    if value is None:
        return _NULL
    # 正文里有换行，压成一行才看得清表格
    text = " ".join(str(value).split())
    if not full and len(text) > width:
        return text[:width] + "…"
    return text


if __name__ == "__main__":
    raise SystemExit(main())
