# -*- coding: utf-8 -*-
"""后台日志：tee stdout/stderr 到进程内环形缓冲，供前端「后台日志」面板查看

_TeeWriter 按行入缓冲并拼接半行；install() 幂等，且会改写已存在的
logging StreamHandler 的流引用（uvicorn CLI 先配日志后导入应用的场景）
"""
import sys
import threading
from collections import deque
from datetime import datetime

_MAX_LINES = 5000
_lock = threading.Lock()
_records = deque(maxlen=_MAX_LINES)
_next_id = 1


def _emit(stream_tag: str, line: str):
    global _next_id
    with _lock:
        _records.append({"id": _next_id,
                         "ts": datetime.now().strftime("%H:%M:%S"),
                         "stream": stream_tag,
                         "text": line})
        _next_id += 1


class _TeeWriter:
    """分流写入器：原始终端 + 日志缓冲（行缓冲，含半行拼接）"""

    def __init__(self, original, tag: str):
        self._original = original
        self._tag = tag
        self._pending = ""

    def write(self, s):
        try:
            self._original.write(s)
        except Exception:
            pass
        if not s:
            return 0
        self._pending += s
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            _emit(self._tag, line)
        return len(s)

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        # isatty / encoding / fileno 等属性透传原始终端
        return getattr(self._original, name)


class _AccessLogFilter:
    """uvicorn access log 过滤器：丢弃前端高频轮询请求的访问日志。

    record.args = (client, method, path, http_version, status_code)；
    不过滤的话，系统状态页每 2 秒的轮询会把日志面板刷成纯访问流水。
    """

    PREFIXES = ("/api/system/", "/api/crowd/camera/stream")

    def filter(self, record):
        try:
            args = record.args
            if args and len(args) >= 3:
                path = str(args[2])
                if path.startswith(self.PREFIXES):
                    return False
        except Exception:
            pass
        return True


def install():
    """安装 stdout / stderr 分流捕获（进程级幂等）"""
    if isinstance(sys.stdout, _TeeWriter):
        return
    sys.stdout = _TeeWriter(sys.stdout, "out")
    sys.stderr = _TeeWriter(sys.stderr, "err")
    # 接管已存在的 logging StreamHandler（uvicorn CLI 启动顺序场景）
    try:
        import logging
        names = [""] + [n for n in logging.Logger.manager.loggerDict]
        for n in names:
            lg = logging.getLogger(n)
            for h in lg.handlers:
                if isinstance(h, logging.StreamHandler):
                    if h.stream is sys.__stdout__:
                        h.stream = sys.stdout
                    elif h.stream is sys.__stderr__:
                        h.stream = sys.stderr
        # 轮询类请求的 access log 不进后台日志面板
        logging.getLogger("uvicorn.access").addFilter(_AccessLogFilter())
    except Exception:
        pass


def get_logs(tail: int = 200, cursor: int = 0):
    """读取缓冲：cursor>0 时返回其后新记录；否则返回末尾 tail 条"""
    with _lock:
        items = list(_records)
    if cursor > 0:
        items = [r for r in items if r["id"] > cursor]
    if len(items) > tail:
        items = items[-tail:]
    return items


def clear_logs():
    with _lock:
        _records.clear()
