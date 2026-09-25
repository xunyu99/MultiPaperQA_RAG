"""MinerU 连通性自检（Step 2 开跑前用）。

和三件事有关，分开验证才不会互相掩盖：

    配置   .env 里的 key / base_url / 开关是不是填了
    接口   服务端认不认我们发的请求体（字段名对不对）
    链路   真跑一篇，看产物能不能落盘

用法：
    # 只查配置，不发任何请求
    .venv\\Scripts\\python.exe -m scripts.check_mineru

    # 申请一次上传链接来验证接口字段（**不消耗解析额度**，因为没有真的上传文件）
    .venv\\Scripts\\python.exe -m scripts.check_mineru storage/pdf/xxx.pdf

    # 完整跑一篇（这一步才真正消耗额度）
    .venv\\Scripts\\python.exe -m scripts.check_mineru storage/pdf/xxx.pdf --full

为什么要单独有"申请链接"这一档：`/file-urls/batch` 的请求体字段名是我照控制台文档
写的，没法离线核对。这个调用能验证字段对不对，又不触发解析 —— 字段写错的话，
在这里报 400 比跑完整流程才发现划算得多。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.ingest.mineru_client import MinerUClient, MinerUError
from app.ingest.versioning import mineru_version, parse_version, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="MinerU 连通性自检")
    parser.add_argument("pdf", nargs="?", default=None, help="要自检的 PDF（可选）")
    parser.add_argument("--full", action="store_true", help="真的跑一遍完整解析（消耗额度）")
    parser.add_argument("--dest", default=None, help="--full 时的产物目录，默认 storage/mineru/check")
    args = parser.parse_args()

    settings = get_settings()

    print("=== 1. 配置 ===")
    print(f"  base_url      : {settings.mineru_base_url}")
    print(f"  api_key       : {'已配置，长度 ' + str(len(settings.mineru_api_key)) if settings.mineru_api_key else '[缺] 没配（填 .env 的 MINERU_API_KEY）'}")
    print(f"  proxy         : {settings.mineru_proxy or '（不走代理）'}")
    print(f"  model_version : {settings.mineru_model_version}")
    print(f"  language      : {settings.mineru_language}")
    print(f"  is_ocr        : {settings.mineru_is_ocr}")
    print(f"  enable_formula: {settings.mineru_enable_formula}")
    print(f"  enable_table  : {settings.mineru_enable_table}")
    print(f"  parse_version : {parse_version(settings)}")
    print(f"  mineru_version: {mineru_version(settings)}   ← parse_cache 的键")
    print(f"  轮询          : 每 {settings.mineru_poll_interval_seconds}s 一次，上限 {settings.mineru_poll_timeout_seconds}s")

    if not settings.mineru_api_key:
        print("\n结论：先把 MINERU_API_KEY 填进 .env，再往下走")
        return 1

    if not args.pdf:
        print("\n结论：配置齐全。想验证接口就带上一个 PDF 路径（不加 --full 不会消耗额度）")
        return 0

    pdf = Path(args.pdf).resolve()
    if not pdf.is_file():
        print(f"\nPDF 不存在：{pdf}")
        return 1

    print("\n=== 2. PDF ===")
    print(f"  路径          : {pdf}")
    print(f"  大小          : {pdf.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  content_hash  : {sha256_file(pdf)[:16]}…")

    with MinerUClient(settings) as client:
        print("\n=== 3. 接口自检：申请上传链接 ===")
        try:
            batch_id, upload_url = client.request_upload_url(pdf)
        except MinerUError as exc:
            print(f"  [失败] {exc}")
            print("\n上面这段是服务端原样返回。如果是 400 且提到未知字段，改")
            print("app/ingest/mineru_client.py 里的 _upload_body()，然后重跑本脚本。")
            return 1
        print("  [通过] 成功了，说明 _upload_body() 的字段名服务端认")
        print(f"  batch_id      : {batch_id}")
        print(f"  预签名地址    : {'已拿到，长度 ' + str(len(upload_url)) + '（含签名，不打印）'}")
        print("  注意：**只申请了地址，没有上传文件**，所以没有消耗解析额度")

        if not args.full:
            print("\n结论：接口通。要真跑一遍就加 --full（这一步会消耗额度）")
            return 0

        dest = Path(args.dest) if args.dest else settings.storage_dir / "mineru" / "check"
        print(f"\n=== 4. 完整解析（会消耗额度）===")
        print(f"  产物目录      : {dest}")
        try:
            content_list = client.parse_pdf(pdf, dest, on_tick=_tick)
        except MinerUError as exc:
            print(f"  [失败] {exc}")
            return 1
        print(f"  [通过] 产物已落盘：{content_list}")
        print("\n结论：MinerU 链路完全打通。下一步：")
        print(f"  .venv\\Scripts\\python.exe -m scripts.run_mineru {pdf.name}")
        return 0


def _tick(state: str, item: dict) -> None:
    print(f"    轮询状态：state={state or '未知'}")


if __name__ == "__main__":
    raise SystemExit(main())
