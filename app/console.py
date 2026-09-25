"""让 print 永远不会因为编码崩掉。

这不是"优雅处理"，是**真实会崩**：Windows 控制台默认 GBK，脚本一 `print()` 到
GBK 里没有的字符就 `UnicodeEncodeError` 直接退出，而且往往是在跑到一半的时候。

踩过的两次：

1. 自己写的 `print(...)` 里放了个 U+2705（白色对勾）—— 不在 GBK 里，脚本崩；
2. **更麻烦的**：MinerU 从论文里抽出来的正文带了 U+2217（通讯作者标记用的那个星号），
   `dump_blocks` 把这个块打印出来时就崩了。这种情况"源码里别写怪字符"根本防不住 ——
   论文里的字符什么都有，防的是**数据**，不是代码。

所以正确做法是在入口处把**编码错误处理器**改成 `replace`：编码（GBK）不动，
中文照样正常显示，只有极个别字符会显示成 `?`，但脚本绝不会崩。
改编码本身是错的方向 —— 把 stdout 设成 UTF-8 会在 GBK 控制台里显示成乱码。
"""

from __future__ import annotations

import sys
from typing import Any


def make_streams_safe() -> list[str]:
    """把 stdout / stderr 的错误处理改成 replace，返回被改过的流名。

    幂等；对没有 `reconfigure` 的流（比如 pytest 抓取用的替身）自动跳过。
    """
    patched: list[str] = []
    for name in ("stdout", "stderr"):
        stream: Any = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            # 流已经关闭或不允许重配置 —— 跳过就好，不值得让整个脚本挂掉
            continue
        patched.append(name)
    return patched
