"""入库版本指纹：回答"哪一层可以跳过、哪一层必须重做"。

四个指纹各管一段，成本差着数量级，所以必须分开：

    content_hash   PDF 字节       变了 = 用户传了另一份文件
    mineru_version MinerU 调用参数 变了 = 要重新调 MinerU（花钱、几分钟）
    parse_version  上面两个 + converter 版本 = 这批 blocks/assets 的完整来源（只用于审计）
    index_version  切分+embedding  变了 = chunks/向量要重做（本地切分免费，embedding 花钱但快）

**为什么 parse_cache 的键是 mineru_version 而不是 parse_version**：
converter 是我们本地的代码，改它只需要重跑本地解析（读磁盘上的 content_list.json，几秒、零成本），
不该触发一次收费的 MinerU 调用。把 converter 版本塞进缓存键，就会出现"改了一行解析代码，
缓存全部失效，几十篇论文的钱白花"。产物来源还是要记（papers.parse_version），但那是审计用的，
不用来判缓存。

关键设计：**指纹从 settings 自动拼出来，不是手写的 v1/v2**。
手写的版本号一定会忘改，然后就出现"改了 chunk 大小却复用了旧 chunk"这种脏数据。

注意 parse_version 里带了 CONVERTER_VERSION：改 converter/section_tree 的产物结构
（比如给 block 加一列）时手动 +1，否则旧产物会被错误复用。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.config import Settings, get_settings

# converter / section_tree 的产物结构版本，改了结构就 +1
CONVERTER_VERSION = "1"

# 资产身份逻辑版本：改 assets.py 的编号解析 / splitter.py 的交叉引用规则就 +1。
# 1 = 用"第几个块"的计数器编号 + 宽松正则（会挂错）；
# 2 = 从原始 caption / \tag{} 抽 label_norm，抽不出留 NULL，撞号不猜。
# 3 = 不再给没有编号的资产补假 caption；索引文本改成现算，不再落库。
# 它进 index_version：这批 chunk_assets 是哪版逻辑产出的，靠它分辨。
ASSET_VERSION = "3"

_SLUG_KEEP = re.compile(r"[^A-Za-z0-9_-]+")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """PDF 字节的 sha256。流式读，几百 MB 的文件也不会吃内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def mineru_version(settings: Settings | None = None) -> str:
    """MinerU 调用参数指纹。

    这是 **parse_cache 的键**：它只包含真要发给 MinerU 的参数。
    改这里的任何一项，磁盘上的 content_list.json 就不保证还符合预期，必须重调 MinerU。
    """
    s = settings or get_settings()
    return (
        f"mineru:{s.mineru_model_version}"
        f"|lang:{s.mineru_language}"
        f"|ocr:{int(bool(s.mineru_is_ocr))}"
        f"|formula:{int(bool(s.mineru_enable_formula))}"
        f"|table:{int(bool(s.mineru_enable_table))}"
    )


def parse_version(settings: Settings | None = None) -> str:
    """这批 blocks/assets 的完整来源 = MinerU 调用参数 + converter 版本。

    存进 `papers.parse_version`，用于审计和排查「为什么这两篇的 blocks 长得不一样」。
    **不要拿它当缓存键**，理由见模块开头。
    """
    return f"{mineru_version(settings)}|converter:{CONVERTER_VERSION}"


def index_version(settings: Settings | None = None) -> str:
    """索引层指纹：切分参数 + 资产逻辑版本 + embedding 模型与维度。"""
    s = settings or get_settings()
    return (
        f"split:{s.chunk_target_tokens}-{s.chunk_max_tokens}-{s.chunk_min_tokens}"
        f"-{s.block_overlap}-{s.text_overlap_chars}"
        f"|assets:{ASSET_VERSION}"
        f"|embed:{s.embedding_model}-{s.embedding_dim}"
    )


def default_paper_id(pdf_path: str | Path) -> str:
    """从文件名推一个稳定的 paper_id。

    它会被拼进 chunk_id 和目录名，所以必须是安全字符集：非 [A-Za-z0-9_-] 一律换成 `-`。
    中文文件名会因此被清空，这时退回内容哈希前 12 位 —— 仍然稳定，只是不好看。
    """
    path = Path(pdf_path)
    slug = _SLUG_KEEP.sub("-", path.stem).strip("-").lower()
    if slug:
        return slug[:64]
    return f"paper-{sha256_file(path)[:12]}"
