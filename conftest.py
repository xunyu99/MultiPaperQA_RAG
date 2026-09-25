"""pytest 全局配置：把临时目录固定到工程内，且**每次运行换一个名字**。

这条是真踩出来的，前后踩了两次，所以写清楚：

**第一次**：pytest 默认把 `tmp_path` 建在系统 Temp 下
（`C:\\Users\\<你>\\AppData\\Local\\Temp\\pytest-of-<你>\\...`）。那个目录可能被
别的进程先建过、带上了只读 ACL —— 于是所有用到 `tmp_path` 的测试集体
`PermissionError: [WinError 5]`。

**第二次**：改成 `--basetemp=.pytest_tmp`（放进工程目录）也没解决。因为 pytest
拿到显式 basetemp 时会**先 `rm -rf` 它**，只要这个目录是别的用户/进程建的、
删不掉，一样报 `WinError 5`。

**最终做法**：basemp 的**父目录**固定成工程内的 `.pytest_tmp`（谁跑都在工程里，
不碰系统 Temp），但每次运行的**实际目录带上随机后缀** —— pytest 找不到就不用删，
不同进程之间永远不会互相踩。

`.pytest_tmp/` 已经进了 `.gitignore`，积累的目录可以直接删，不影响任何东西。
"""

from __future__ import annotations

import os
import uuid

import pytest


def pytest_configure(config: pytest.Config) -> None:
    root = getattr(config.option, "basetemp", None) or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".pytest_tmp"
    )
    # pytest 只 mkdir 叶子目录，不建中间层 —— 父目录得我们自己保证存在。
    os.makedirs(root, exist_ok=True)
    # 关键：加随机后缀。pytest 会对 basetemp 做 rm -rf，目录不存在就不会踩权限问题。
    config.option.basetemp = os.path.join(root, uuid.uuid4().hex[:8])
