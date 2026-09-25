"""MinerU 云端解析客户端（Step 2 的入口）。

走的是"批量文件上传"这条路，因为本地 PDF 没有公网 URL：

    1. POST /file-urls/batch      申请预签名上传地址 → 拿到 batch_id 和 file_urls
    2. PUT  file_urls[0]          直接把 PDF 字节传上去（**不能带 Authorization**，
                                  也不能带 Content-Type，否则签名校验会失败）
    3. GET  /extract-results/batch/{batch_id}   轮询到 state=done → full_zip_url
    4. GET  full_zip_url          下载 zip，解压到 storage/mineru/{paper_id}/

注意：接口字段以控制台文档为准。写这段时无法联网核对，所以每个请求都把服务端的
原始返回带在报错里 —— 真报 400 的时候照着 `msg` 改 `_upload_body()` 就行。

产物目录**永远不删**（删论文也不删）：它是 parse_cache 复用的依据，
也是 BACKLOG 里资产解析、VLM 挂图要用的原图来源。
"""

from __future__ import annotations

import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from app.config import Settings, get_settings

# 有 status_code 就一定是 HTTP 层的问题；没有就是业务层的
_OK_CODES = (0, 200, None)
_DONE_STATES = ("done", "success", "completed")
_FAILED_STATES = ("failed", "error")

TickCallback = Callable[[str, dict], None]


class MinerUError(RuntimeError):
    """MinerU 相关的所有失败。带上原始返回，方便照着改字段名。"""


class MinerUClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.mineru_base_url.rstrip("/")
        kwargs: dict = {"timeout": 120.0}
        if self.settings.mineru_proxy:
            kwargs["proxy"] = self.settings.mineru_proxy
        self._client = httpx.Client(**kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MinerUClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 一条龙
    # ------------------------------------------------------------------
    def parse_pdf(
        self,
        pdf_path: str | Path,
        dest_dir: str | Path,
        on_tick: TickCallback | None = None,
    ) -> Path:
        """上传 → 轮询 → 下载 → 解压，返回 content_list.json 的路径。"""
        pdf = Path(pdf_path)
        if not pdf.is_file():
            raise MinerUError(f"PDF 不存在：{pdf}")

        batch_id, upload_url = self.request_upload_url(pdf)
        self.upload_file(upload_url, pdf)
        item = self.poll_batch(batch_id, on_tick=on_tick)

        zip_url = item.get("full_zip_url")
        if not zip_url:
            raise MinerUError(f"任务已完成但没返回 full_zip_url，原始返回：{item}")
        self.download_zip(str(zip_url), Path(dest_dir))
        return find_content_list(Path(dest_dir))

    # ------------------------------------------------------------------
    # 分步接口（单步调试时用得上）
    # ------------------------------------------------------------------
    def request_upload_url(self, pdf_path: Path) -> tuple[str, str]:
        resp = self._client.post(
            f"{self.base_url}/file-urls/batch",
            headers=self._headers(json_body=True),
            json=self._upload_body(pdf_path),
        )
        data = self._unwrap(resp, "申请上传链接")
        urls = data.get("file_urls") or []
        batch_id = data.get("batch_id")
        if not batch_id or not urls:
            raise MinerUError(f"申请上传链接返回缺字段（要 batch_id + file_urls）：{data}")
        return str(batch_id), str(urls[0])

    def upload_file(self, upload_url: str, pdf_path: Path) -> None:
        """PUT 到预签名地址。这个请求**必须干净**：多一个 header 签名就失效。"""
        with pdf_path.open("rb") as fh:
            resp = self._client.put(upload_url, content=fh.read())
        if resp.status_code not in (200, 201, 204):
            raise MinerUError(f"上传 PDF 失败：HTTP {resp.status_code} {resp.text[:300]}")

    def poll_batch(self, batch_id: str, on_tick: TickCallback | None = None) -> dict:
        interval = max(1, int(self.settings.mineru_poll_interval_seconds))
        deadline = time.monotonic() + max(1, int(self.settings.mineru_poll_timeout_seconds))

        while True:
            resp = self._client.get(
                f"{self.base_url}/extract-results/batch/{batch_id}",
                headers=self._headers(json_body=False),
            )
            data = self._unwrap(resp, "查询解析任务")
            results = data.get("extract_result") or []
            item = results[0] if results else {}
            state = str(item.get("state") or "").strip().lower()
            if on_tick is not None:
                on_tick(state, item)

            if state in _DONE_STATES:
                return item
            if state in _FAILED_STATES:
                raise MinerUError(f"解析失败：state={state} err_msg={item.get('err_msg')} 原始返回={item}")
            if time.monotonic() > deadline:
                raise MinerUError(
                    f"轮询超时（{self.settings.mineru_poll_timeout_seconds}s），"
                    f"最后状态 {state or '未知'}，batch_id={batch_id}"
                )
            time.sleep(interval)

    def download_zip(self, zip_url: str, dest_dir: Path) -> Path:
        resp = self._client.get(zip_url)
        if resp.status_code != 200:
            raise MinerUError(f"下载解析产物失败：HTTP {resp.status_code} {resp.text[:200]}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        archive = dest_dir / "_mineru_download.zip"
        archive.write_bytes(resp.content)
        try:
            safe_extract(archive, dest_dir)
        finally:
            archive.unlink(missing_ok=True)
        return dest_dir

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _upload_body(self, pdf_path: Path) -> dict:
        """上传请求体。字段名若与服务端不符，报错里会带上 msg，照着改这一处即可。"""
        return {
            "enable_formula": self.settings.mineru_enable_formula,
            "enable_table": self.settings.mineru_enable_table,
            "language": self.settings.mineru_language,
            "model_version": self.settings.mineru_model_version,
            "files": [{"name": pdf_path.name, "is_ocr": self.settings.mineru_is_ocr}],
        }

    def _headers(self, *, json_body: bool) -> dict[str, str]:
        if not self.settings.mineru_api_key:
            raise MinerUError("MINERU_API_KEY 没配。填到 .env 里再跑（Step 0 的 .env.example 有模板）")
        headers = {"Authorization": f"Bearer {self.settings.mineru_api_key}"}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _unwrap(self, resp: httpx.Response, what: str) -> dict:
        if resp.status_code != 200:
            raise MinerUError(f"{what}失败：HTTP {resp.status_code} {resp.text[:300]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise MinerUError(f"{what}返回的不是 JSON：{resp.text[:300]}") from exc
        code = payload.get("code")
        if code not in _OK_CODES:
            raise MinerUError(f"{what}返回业务错误：code={code} msg={payload.get('msg')}")
        data = payload.get("data")
        return data if isinstance(data, dict) else {}


# ----------------------------------------------------------------------
# 产物查找 / 解压
# ----------------------------------------------------------------------
def safe_extract(archive: Path, dest_dir: Path) -> None:
    """解压，并且挡住 zip slip（成员路径里写 ../.. 逃出目标目录）。

    解析产物是外部服务给的文件，不能无条件信任路径。
    """
    dest = dest_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest)):
                raise MinerUError(f"zip 里有越界路径，拒绝解压：{member.filename}")
        zf.extractall(dest)


def find_content_list(root: str | Path) -> Path:
    """在产物目录里找 *_content_list.json（zip 里可能套了一层子目录）。"""
    base = Path(root)
    matches = sorted(base.rglob("*content_list.json"))
    if not matches:
        raise MinerUError(f"产物目录里没找到 *_content_list.json：{base}")
    return matches[0]


def find_markdown(root: str | Path) -> Path | None:
    base = Path(root)
    for name in ("full.md",):
        candidate = base / name
        if candidate.is_file():
            return candidate
    matches = sorted(base.rglob("*.md"))
    return matches[0] if matches else None


def count_images(root: str | Path) -> int:
    base = Path(root)
    return sum(1 for path in base.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
