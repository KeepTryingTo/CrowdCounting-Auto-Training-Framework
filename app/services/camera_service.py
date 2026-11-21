# -*- coding: utf-8 -*-
"""摄像头实时检测服务（OpenCV 采集 + 复用推理管线 + MJPEG 推流）

- 单例会话：同一时刻仅一个摄像头会话（物理摄像头独占），开启新会话自动停止旧会话
- 采集线程：cv2.VideoCapture 读帧 → 复用人群计数推理函数 →
  画面左上角绘制「平均 FPS + 人群数量」信息面板 → JPEG 编码
- 统计口径：
    密度图估计 → 人群数量（count，密度图求和）
    P2PNet 点预测 → 人群数量（count，置信点计数）
"""
import base64
import io
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 支持的图像类子任务（人群计数；三条路线 + 旧版合并 key）
CROWD_ROUTE_TASKS = ("crowd_counting", "crowd_density", "crowd_point",
                     "crowd_box")
SUPPORTED_TASKS = {
    "pytorch": set(CROWD_ROUTE_TASKS),
}

# 中断运行前的最大连续推理错误帧数
_MAX_INFER_ERRORS = 20
# 连续读帧失败上限（超过则判定摄像头不可用）
_MAX_READ_FAILURES = 40


# ------------------------- 字体（支持中文） -------------------------

_FONT_CACHE = {}


def _get_font(size=20):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    candidates = []
    if __import__("sys").platform == "win32":
        candidates = [Path("C:/Windows/Fonts") / n for n in
                      ("msyh.ttc", "simhei.ttf", "simfang.ttf",
                       "Microsoft YaHei UI.ttf", "arial.ttf")]
    candidates += [n for n in ("DejaVuSans.ttf", "arial.ttf")]
    for c in candidates:
        try:
            font = ImageFont.truetype(str(c), size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


# ------------------------- 推理分发（复用人群计数服务） -------------------------

def _infer_pytorch(img: Image.Image, p: dict) -> dict:
    task = p["task"]
    wp = p.get("weight_path") or None
    device = p.get("device") or "cpu"
    # CLTR 官方验证/推理阈值为 0.35；未显式传参时不要回退到 0.5
    conf = float(p.get("conf_threshold") or
                 (0.35 if p.get("model") == "cltr" else 0.5))
    if task == "crowd_box":
        from app.services import yolo_service
        return yolo_service.headbox_infer(
            img, p["model"], weight_path=wp,
            conf_threshold=conf, device=device,
            pretrained=bool(p.get("pretrained")))
    if task in CROWD_ROUTE_TASKS:
        from app.services import crowd_counting
        return crowd_counting.crowd_infer(
            img, model_name=p["model"], weight_path=wp, device=device,
            pretrained=bool(p.get("pretrained")), conf_threshold=conf)
    raise ValueError(f"摄像头实时计数不支持任务: {task}")


_INFER_DISPATCH = {
    "pytorch": _infer_pytorch,
}

# 任务 → 统计字段/文案（zh / en）
_TASK_STATS = {
    "crowd_counting": ("人群数量", "Crowd"),
    "crowd_density": ("人群数量", "Crowd"),
    "crowd_point": ("人群数量", "Crowd"),
    "crowd_box": ("人头数量", "Heads"),
}


def _overlay_lines(task: str, result: dict, avg_fps: float, lang: str):
    """左上角信息面板内容：平均 FPS + 人群数量"""
    zh = lang != "en"
    lines = [f"{'平均 FPS' if zh else 'Avg FPS'}: {avg_fps:.1f}"]
    if task in _TASK_STATS:
        label = _TASK_STATS[task][0 if zh else 1]
        if task in ("crowd_counting", "crowd_density"):
            # 密度图路线：count 为密度求和（浮点）
            val = result.get("count", 0)
            val = f"{float(val):.1f}"
        elif task == "crowd_point":
            # 点预测路线：count 为整数（置信点计数）
            val = int(result.get("count", 0) or 0)
        else:
            # 头部框路线：检测框计数
            val = int(result.get("num_detected", result.get("count", 0)) or 0)
        lines.append(f"{label}: {val}")
    return lines


def _draw_overlay(img: Image.Image, lines, error=None):
    """在画面左上角绘制半透明信息面板"""
    draw = ImageDraw.Draw(img, "RGBA")
    font = _get_font(20)
    if error:
        lines = lines + [f"[{error[:60]}]"]
    pad, line_h = 10, 28
    widths = [draw.textlength(ln, font=font) for ln in lines]
    panel_w = int(max(widths) if widths else 100) + pad * 2
    panel_h = len(lines) * line_h + pad * 2 - 6
    draw.rectangle([8, 8, 8 + panel_w, 8 + panel_h], fill=(0, 0, 0, 160))
    for i, ln in enumerate(lines):
        color = (0, 255, 120, 255) if i == 0 else (235, 240, 245, 255)
        if error and i == len(lines) - 1:
            color = (80, 200, 255, 255)
        draw.text((8 + pad, 8 + pad + i * line_h), ln, font=font, fill=color)
    return img


def _decode_vis(vis_b64: str):
    """base64 PNG → PIL 图（失败返回 None）"""
    try:
        return Image.open(io.BytesIO(base64.b64decode(vis_b64))).convert("RGB")
    except Exception:
        return None


# ------------------------- 摄像头会话 -------------------------

class CameraSession:
    def __init__(self, framework: str, params: dict):
        self.framework = framework
        self.params = dict(params)
        self.task = params["task"]
        self.model = params.get("model") or ""
        self.lang = params.get("lang") or "zh"
        self.camera_index = int(params.get("camera_index") or 0)
        self.width = int(params.get("width") or 0)
        self.height = int(params.get("height") or 0)

        self.cap = None
        self.thread = None
        self.running = False
        self.latest_jpeg = None       # 最近的 JPEG 帧（供 MJPEG 推流）
        self.frame_seq = 0            # 帧序号（有新帧时递增）
        self.frames = 0               # 已处理帧数
        self.avg_fps = 0.0
        self.error = None
        self._infer_errors = 0
        self._start_ts = None
        self._lock = threading.Lock()

    # ---- 生命周期 ----

    def start(self):
        import cv2
        import sys as _sys
        if self.running:
            return
        cap = None
        if _sys.platform == "win32":
            # DSHOW 打开更快（避免 MSMF 后端数秒延迟）；失败回退默认后端
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            raise ValueError(
                f"无法打开摄像头 {self.camera_index}"
                f"（请检查设备连接 / 是否被其他程序占用 / 编号是否正确）")
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap = cap
        self.running = True
        self.error = None
        self._start_ts = time.perf_counter()
        self.thread = threading.Thread(target=self._loop, daemon=True,
                                       name=f"camera-{self.framework}")
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive() and \
                threading.current_thread() is not self.thread:
            self.thread.join(timeout=5)
        self.thread = None
        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    # ---- 采集 + 推理循环 ----

    def _push_jpeg(self, cv2, base):
        """FPS 统计 + JPEG 编码 + 更新最新帧（供 MJPEG 推流）"""
        # 平均 FPS：自会话启动以来的累计平均
        self.frames += 1
        elapsed = time.perf_counter() - self._start_ts
        if elapsed > 0:
            self.avg_fps = self.frames / elapsed
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(np.asarray(base), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with self._lock:
                self.latest_jpeg = buf.tobytes()
                self.frame_seq += 1

    def _loop(self):
        import cv2
        read_failures = 0
        while self.running:
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                read_failures += 1
                if read_failures >= _MAX_READ_FAILURES:
                    self.error = "摄像头读取失败（设备断开或被占用）"
                    self.running = False
                    break
                time.sleep(0.05)
                continue
            read_failures = 0

            pil = None
            try:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = _INFER_DISPATCH[self.framework](pil, self.params)
                self._infer_errors = 0
                # 展示底图：优先用推理可视化（密度热力图 / 预测点叠加图）
                base = pil
                if result is not None and result.get("visualization"):
                    base = _decode_vis(result["visualization"]) or pil
                lines = _overlay_lines(self.task, result or {}, self.avg_fps,
                                       self.lang)
                self._push_jpeg(cv2, _draw_overlay(base, lines))
            except Exception as e:
                # 单帧任何环节异常都不终止线程（否则 running 仍为 True，
                # 前端流连接一直等不到帧 → 表现为「没有实时画面」）；
                # 累积超限才判定会话失败
                self._infer_errors += 1
                err = f"{type(e).__name__}: {e}"
                if self._infer_errors >= _MAX_INFER_ERRORS:
                    self.error = f"实时检测连续失败: {err}"
                    self.running = False
                    break
                if pil is not None:
                    try:
                        self._push_jpeg(cv2, _draw_overlay(
                            pil, _overlay_lines(self.task, {}, self.avg_fps,
                                                self.lang), error=err))
                    except Exception:
                        pass
            time.sleep(0.01)   # 让出 GIL，避免占满 CPU
        # 循环结束（错误 / stop）：保留最后一帧，状态由 status() 上报

    # ---- 状态 ----

    def status(self) -> dict:
        w = h = 0
        if self.cap is not None:
            try:
                import cv2
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            except Exception:
                pass
        return {
            "running": self.running,
            "framework": self.framework,
            "task": self.task,
            "model": self.model,
            "camera_index": self.camera_index,
            "avg_fps": round(self.avg_fps, 1),
            "frames": self.frames,
            "error": self.error,
            "width": w, "height": h,
        }


class CameraManager:
    """全局单例：同一时刻仅一个摄像头会话（物理摄像头独占）"""

    def __init__(self):
        self.session = None   # CameraSession | None
        self._lock = threading.Lock()

    def start(self, framework: str, params: dict) -> CameraSession:
        supported = SUPPORTED_TASKS.get(framework)
        if not supported:
            raise ValueError(f"未知框架: {framework}")
        task = params.get("task")
        if task not in supported:
            raise ValueError(
                f"摄像头实时检测不支持任务「{task}」，"
                f"当前框架支持: {sorted(supported)}")
        if not params.get("model"):
            raise ValueError("请先选择网络模型")
        with self._lock:
            # 开启新会话前自动停止旧会话（摄像头独占）
            if self.session is not None:
                self.session.stop()
                self.session = None
            session = CameraSession(framework, params)
            session.start()   # 打开失败在此抛 ValueError
            self.session = session
            return session

    def stop(self):
        with self._lock:
            if self.session is not None:
                self.session.stop()
                self.session = None

    def get(self):
        return self.session

    def status(self) -> dict:
        if self.session is None:
            return {"running": False, "session": False}
        return {"session": True, **self.session.status()}


manager = CameraManager()


def stream_response():
    """当前会话的 MJPEG StreamingResponse（人群计数路由用）"""
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse

    session = manager.get()
    if session is None or not session.running:
        raise HTTPException(404, "摄像头会话未运行（请先开启检测）")

    def gen():
        import time as _time
        last_seq = -1
        try:
            while True:
                s = manager.get()
                if s is not session or not s.running:
                    break
                with session._lock:
                    frame, seq = session.latest_jpeg, session.frame_seq
                if frame and seq != last_seq:
                    last_seq = seq
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + frame + b"\r\n")
                    continue
                _time.sleep(0.02)
        finally:
            # 注意：这里【不能】因客户端断开就停止会话。浏览器在切换页面 /
            # 模态框重绘 / 网络抖动时会短暂断流并自动重连，若断流即 stop，
            # 重连时会话已停，表现为「摄像头打开了但一直没有实时画面」。
            # 会话生命周期由「停止」按钮、关闭弹窗（/camera/stop）或服务
            # 退出显式管理；manager.start 开新会话时也会先停旧会话。
            pass

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame")
