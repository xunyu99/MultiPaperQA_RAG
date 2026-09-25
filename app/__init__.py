"""MultiPaperQA_RAG 多论文问答系统。"""

__version__ = "0.1.0"

# 导入 app 的任何东西都会顺带把控制台设成"不会因编码崩"。
# 放这里而不是每个脚本里，是因为漏一个脚本就等于留一个坑 —— 详见 app/console.py。
# 只改编码的**错误处理**（replace），不改编码本身，所以中文显示不受影响。
from app.console import make_streams_safe as _make_streams_safe

_make_streams_safe()
