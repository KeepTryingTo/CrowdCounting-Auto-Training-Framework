# -*- coding: utf-8 -*-
"""ONNX Runtime 推理服务：部署产物验证 + 上传 onnx 模型推理

与 onnx_export 同款的【子进程隔离】：onnxruntime 与 torch 同进程存在
OpenMP 原生库冲突风险（偶发 0xC0000005），因此 API 进程绝不直接创建
InferenceSession —— 探测与推理一律经「子进程 + __RESULT__ JSON 协议」
执行，同时与用户的独立部署环境（snippet.py）保持一致。

类型识别（与导出契约一致，两条路径）：
1. 同目录 meta.json（本框架导出产物 100% 命中：route / model / imgsz 精确）
2. ONNX 图结构启发式（上传的外部 onnx）：
   - 双输出且均为 [B,M,2]            → crowd_point / p2pnet（logits+points）
   - 单输出 3 维且末维=6             → crowd_box / yolo 端到端（NMS 内嵌）
   - 单输出 3 维且第 2 维 ≥5         → crowd_box / yolo 标准（[B,4+nc,N]）
   - 单输出 4 维且通道=1，下采样 /4   → crowd_point / p2r（logit 图）
   - 单输出 4 维且通道=1，其余        → crowd_density（密度图）

推理结果字段与 torch 路线完全对齐（count / visualization / point_based /
box_based / density_map …），前端渲染逻辑零改动即可复用。
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 子进程运行标记前缀（父进程解析结果 JSON）
_RESULT_MARK = "__RESULT__"

# 各路线的默认输入边长与对齐约束（图维度动态时的回退值，与导出侧一致）
_ROUTE_DEFAULTS = {
    "crowd_density": {"imgsz": 384, "multiple": 8},
    "crowd_point": {"imgsz": 384, "multiple": 16},       # p2pnet / cp
    "crowd_point_p2r": {"imgsz": 384, "multiple": 32},
    "crowd_box": {"imgsz": 640, "multiple": 32},
    "crowd_sdg": {"imgsz": 320, "multiple": 16},          # 单域泛化
}


# ------------------------- 子进程侧：图像 IO（与 snippet 同款） -------------------------

def _imread(path):
    """读取图像（np.fromfile + imdecode：兼容 Windows 非 ASCII 路径）"""
    import cv2
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _b64_png(img_bgr):
    """BGR ndarray → PNG base64（结果可视化字段编码）"""
    import cv2
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode() if ok else ""


# ------------------------- 子进程侧：类型识别 -------------------------

def _load_meta(onnx_path):
    """同目录 meta.json（本框架 ONNX 导出产物必带）。

    utf-8-sig 容错：既兼容导出侧的纯 utf-8，也兼容 Windows 编辑器
    手工改写后可能带上的 BOM。
    """
    mf = Path(onnx_path).parent / "meta.json"
    if not mf.is_file():
        return None
    try:
        return json.loads(mf.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _shape_ints(tensor_info):
    """维度列表 → int（动态维 / 字符串维 → 0）"""
    return [int(d) if isinstance(d, int) and d > 0 else 0
            for d in (tensor_info.shape or [])]


def _detect_format(onnx_path, session):
    """识别 onnx 的路线 / 模型 / 输入尺寸（meta.json 优先，图结构启发式兜底）"""
    meta = _load_meta(onnx_path)
    inp = session.get_inputs()[0]
    iname = inp.name
    in_sh = _shape_ints(inp)
    ih = in_sh[2] if len(in_sh) >= 4 else 0
    iw = in_sh[3] if len(in_sh) >= 4 else 0
    outs = session.get_outputs()

    if meta and meta.get("route"):
        # 本框架导出产物：meta.json 为权威来源
        route = meta["route"]
        model = str(meta.get("model") or "")
        defaults = (_ROUTE_DEFAULTS["crowd_point_p2r"]
                    if (route == "crowd_point" and model == "p2r")
                    else _ROUTE_DEFAULTS.get(route, {"imgsz": 384,
                                                     "multiple": 8}))
        imgsz = int(meta.get("imgsz") or 0) or ih or iw or defaults["imgsz"]
        return {"source": "meta.json", "route": route, "model": model,
                "model_label": str(meta.get("model_label") or model or route),
                "input_name": iname,
                "input_h": ih or imgsz, "input_w": iw or imgsz,
                "imgsz": imgsz, "multiple": int(meta.get("multiple")
                                                or defaults["multiple"]),
                "end2end": bool(meta.get("end2end")),
                "classes": meta.get("classes"),
                "dynamic_batch": bool(meta.get("dynamic_batch"))}

    # ---- 图结构启发式（上传的外部 onnx） ----
    if len(outs) >= 2 and len(_shape_ints(outs[0])) == 3 \
            and len(_shape_ints(outs[1])) == 3:
        d = _ROUTE_DEFAULTS["crowd_point"]
        imgsz = ih or iw or d["imgsz"]
        return {"source": "graph", "route": "crowd_point", "model": "p2pnet",
                "model_label": "P2PNet 系（双输出 logits+points）",
                "input_name": iname, "input_h": ih or imgsz,
                "input_w": iw or imgsz, "imgsz": imgsz,
                "multiple": d["multiple"], "end2end": False, "classes": None,
                "dynamic_batch": not bool(ih and iw)}

    o_sh = _shape_ints(outs[0]) if outs else []
    if len(o_sh) == 3 and o_sh[2] == 6:
        d = _ROUTE_DEFAULTS["crowd_box"]
        imgsz = ih or iw or d["imgsz"]
        return {"source": "graph", "route": "crowd_box", "model": "yolo",
                "model_label": "YOLO（端到端 NMS 内嵌）",
                "input_name": iname, "input_h": ih or imgsz,
                "input_w": iw or imgsz, "imgsz": imgsz,
                "multiple": d["multiple"], "end2end": True, "classes": None,
                "dynamic_batch": not bool(ih and iw)}
    if len(o_sh) == 3 and o_sh[1] >= 5:
        d = _ROUTE_DEFAULTS["crowd_box"]
        imgsz = ih or iw or d["imgsz"]
        return {"source": "graph", "route": "crowd_box", "model": "yolo",
                "model_label": "YOLO（标准输出，需 NMS 后处理）",
                "input_name": iname, "input_h": ih or imgsz,
                "input_w": iw or imgsz, "imgsz": imgsz,
                "multiple": d["multiple"], "end2end": False, "classes": None,
                "dynamic_batch": not bool(ih and iw)}
    if len(o_sh) == 4 and o_sh[1] == 1:
        # 单通道图输出：下采样 /4 → P2R；其余（/8 等）→ 密度网
        oh = o_sh[2]
        down = (ih // oh) if (ih and oh) else 0
        if down == 4:
            d = _ROUTE_DEFAULTS["crowd_point_p2r"]
            imgsz = ih or iw or d["imgsz"]
            return {"source": "graph", "route": "crowd_point", "model": "p2r",
                    "model_label": "P2R（1/4 分辨率 logit 图）",
                    "input_name": iname, "input_h": ih or imgsz,
                    "input_w": iw or imgsz, "imgsz": imgsz,
                    "multiple": d["multiple"], "end2end": False,
                    "classes": None, "dynamic_batch": not bool(ih and iw)}
        d = _ROUTE_DEFAULTS["crowd_density"]
        imgsz = ih or iw or d["imgsz"]
        return {"source": "graph", "route": "crowd_density", "model": "density",
                "model_label": "密度网（单通道密度图输出）",
                "input_name": iname, "input_h": ih or imgsz,
                "input_w": iw or imgsz, "imgsz": imgsz,
                "multiple": d["multiple"], "end2end": False, "classes": None,
                "dynamic_batch": not bool(ih and iw)}
    raise ValueError(
        "无法识别该 ONNX 模型的输出契约（须为本框架导出的部署产物，或"
        " P2PNet / P2R / 密度网 / YOLO 系输出结构）")


def _probe_core(onnx_path):
    """探测入口（子进程）：创建会话 → 类型识别 → 汇总信息"""
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    fmt = _detect_format(onnx_path, session)
    return {"route": fmt["route"], "model": fmt["model"],
            "model_label": fmt["model_label"], "imgsz": fmt["imgsz"],
            "multiple": fmt["multiple"], "end2end": fmt["end2end"],
            "source": fmt["source"], "dynamic_batch": fmt["dynamic_batch"],
            "input_name": fmt["input_name"],
            "output_names": [o.name for o in session.get_outputs()],
            "classes": fmt.get("classes"),
            "size_mb": round(Path(onnx_path).stat().st_size / 1048576, 2)}


# ------------------------- 子进程侧：预处理 / 后处理（与 snippet.py 契约一致） -------------------------

def _preprocess_std(img_bgr, th, tw):
    """密度网 / P2PNet / P2R 预处理：RGB /255 + 缩放 + ImageNet 归一化"""
    import cv2
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = cv2.resize(x, (tw, th)).transpose(2, 0, 1)[None]
    return (x - IMAGENET_MEAN.reshape(1, 3, 1, 1)) / IMAGENET_STD.reshape(
        1, 3, 1, 1)


def _letterbox(img_bgr, th, tw):
    """YOLO 预处理：等比缩放 + 灰边填充（与 ultralytics 训练/验证一致）
    返回 (输入张量 [1,3,th,tw], 缩放比 r, 填充 dw, dh)"""
    import cv2
    h, w = img_bgr.shape[:2]
    r = min(th / h, tw / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img_bgr, (nw, nh))
    dh, dw = (th - nh) // 2, (tw - nw) // 2
    canvas = cv2.copyMakeBorder(resized, dh, th - nh - dh, dw, tw - nw - dw,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None], r, dw, dh


def _draw_points(img_bgr, pts):
    """预测点可视化：红色实心圆（与 p2pnet_infer / p2r_infer 一致）"""
    import cv2
    h, w = img_bgr.shape[:2]
    size = max(2, int(max(h, w) / 300))
    for p in pts:
        cv2.circle(img_bgr, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    return img_bgr


def _infer_density(img_bgr, session, iname, fmt):
    """密度网：count = 密度图求和；可视化 = JET 热力图叠加"""
    import cv2
    th, tw = fmt["input_h"], fmt["input_w"]
    h0, w0 = img_bgr.shape[:2]
    density = session.run(None, {iname: _preprocess_std(
        img_bgr, th, tw)})[0][0, 0]
    count = float(density.sum())

    d = density - density.min()
    if d.max() > 1e-8:
        d = d / d.max()
    heat = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # 密度热力图放大到【原图尺寸】（模型原生为 1/N 分辨率，直接返回会很小）
    heat_full = cv2.resize(heat, (w0, h0))
    overlay = cv2.addWeighted(img_bgr, 0.5, heat_full, 0.5, 0)
    return {"count": round(count, 2),
            "visualization": _b64_png(overlay),
            "density_map": _b64_png(heat_full),
            "density_size": [h0, w0],
            "model": fmt["model"], "onnx": True, "engine": "onnxruntime"}


def _infer_p2pnet(img_bgr, session, iname, fmt, conf_threshold):
    """P2PNet：softmax 人概率 → 阈值筛点；坐标按输入尺寸映射回原图"""
    th, tw = fmt["input_h"], fmt["input_w"]
    h0, w0 = img_bgr.shape[:2]
    pred_logits, pred_points = session.run(
        None, {iname: _preprocess_std(img_bgr, th, tw)})
    prob = 1.0 / (1.0 + np.exp(-(pred_logits[0, :, 1] - pred_logits[0, :, 0])))
    pts = pred_points[0][prob > conf_threshold]
    scale = np.array([w0 / tw, h0 / th], dtype=np.float32)
    pts = pts * scale
    vis = _draw_points(img_bgr.copy(), pts)
    return {"count": int(len(pts)), "visualization": _b64_png(vis),
            "num_points": int(len(pts)),
            "points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in pts[:2000]],
            "conf_threshold": conf_threshold, "model": "p2pnet",
            "point_based": True, "onnx": True, "engine": "onnxruntime"}


def _infer_p2r(img_bgr, session, iname, fmt):
    """P2R：logit>0 即人头点（与训练期评估口径一致，阈值无效）"""
    th, tw = fmt["input_h"], fmt["input_w"]
    h0, w0 = img_bgr.shape[:2]
    logits = session.run(None, {iname: _preprocess_std(
        img_bgr, th, tw)})[0][0, 0]          # [th/4, tw/4]
    down = th // logits.shape[0] if logits.shape[0] else 4
    count = int((logits > 0).sum())
    rows, cols = np.nonzero(logits > 0)
    pts = np.stack([cols, rows], 1) * down + (down - 1) / 2.0   # (x,y)
    pts = pts[(pts[:, 1] < th) & (pts[:, 0] < tw)]
    scale = np.array([w0 / tw, h0 / th], dtype=np.float32)
    pts = pts * scale
    vis = _draw_points(img_bgr.copy(), pts)
    return {"count": count, "visualization": _b64_png(vis),
            "num_points": count,
            "points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in pts[:2000]],
            "model": "p2r", "point_based": True,
            "onnx": True, "engine": "onnxruntime"}


def _nms_numpy(boxes, scores, iou_thr=0.45):
    """纯 numpy NMS（与部署片段代码同款，无需 torch）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def _infer_yolo(img_bgr, session, iname, fmt, conf_threshold):
    """YOLO：letterbox 输入；端到端免后处理 / 标准输出需 numpy NMS"""
    import cv2
    th, tw = fmt["input_h"], fmt["input_w"]
    x, r, dw, dh = _letterbox(img_bgr, th, tw)
    out = session.run(None, {iname: x})[0][0]

    if fmt.get("end2end"):
        dets = out[out[:, 4] > conf_threshold]      # [N,6] xyxy conf cls
    else:
        boxes_xywh, scores = out[:4], out[4:]        # [4,N] / [nc,N]
        cls_ids = scores.argmax(0)
        conf = scores.max(0)
        keep = conf > conf_threshold
        boxes, conf, cls_ids = boxes_xywh.T[keep], conf[keep], cls_ids[keep]
        xy, wh = boxes[:, :2], boxes[:, 2:]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)  # cxcywh→xyxy
        keep2 = _nms_numpy(boxes, conf)
        boxes, conf, cls_ids = boxes[keep2], conf[keep2], cls_ids[keep2]
        dets = np.concatenate([boxes, conf[:, None], cls_ids[:, None]], 1)

    # letterbox 还原到原图坐标
    if len(dets):
        dets[:, [0, 2]] = (dets[:, [0, 2]] - dw) / r
        dets[:, [1, 3]] = (dets[:, [1, 3]] - dh) / r

    classes = fmt.get("classes") or []
    vis = img_bgr.copy()
    detections, summary = [], {}
    for x1, y1, x2, y2, cf, ci in dets:
        cname = (classes[int(ci)] if int(ci) < len(classes)
                 else str(int(ci)))
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 220, 120), 2)
        cv2.putText(vis, f"{cname} {cf:.2f}", (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 1)
        detections.append({"class": cname, "label": int(ci),
                           "score": round(float(cf), 4),
                           "box": [round(float(x1), 1), round(float(y1), 1),
                                   round(float(x2), 1), round(float(y2), 1)]})
        summary[cname] = summary.get(cname, 0) + 1
    return {"count": int(len(detections)), "visualization": _b64_png(vis),
            "detections": detections, "num_detected": len(detections),
            "summary": summary, "conf_threshold": conf_threshold,
            "model": fmt["model"], "box_based": True,
            "onnx": True, "engine": "onnxruntime"}


def _infer_one(img_bgr, session, fmt, conf_threshold):
    iname = fmt["input_name"]
    if fmt["route"] in ("crowd_density", "crowd_sdg"):
        return _infer_density(img_bgr, session, iname, fmt)
    if fmt["route"] == "crowd_box":
        return _infer_yolo(img_bgr, session, iname, fmt, conf_threshold)
    if fmt["model"] == "p2r":
        return _infer_p2r(img_bgr, session, iname, fmt)
    return _infer_p2pnet(img_bgr, session, iname, fmt, conf_threshold)


def _run_core(params):
    """推理入口（子进程）：批量图像 → onnxruntime → 结果列表"""
    import onnxruntime as ort
    onnx_path = params["onnx_path"]
    conf = float(params.get("conf_threshold", 0.5))
    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    fmt = _detect_format(onnx_path, session)
    results = []
    for ip in params["image_paths"]:
        img = _imread(ip)
        if img is None:
            raise ValueError(f"无法读取图像: {ip}")
        results.append(_infer_one(img, session, fmt, conf))
    return results


# ------------------------- API 侧：子进程调度 -------------------------

def _run_subprocess(args, timeout=600):
    """隔离子进程执行（与 onnx_export 同款协议），返回 __RESULT__ JSON"""
    root = Path(__file__).resolve().parents[2]   # Codes 根
    env = {**os.environ, "PYTHONPATH": str(root),
           "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run(
            [sys.executable, "-u", "-m", "app.services.onnx_infer", *args],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace", env=env, cwd=str(root))
    except subprocess.TimeoutExpired:
        raise ValueError(f"ONNX 推理超时（>{timeout}s）：请减小输入边长或"
                         "图像数量后重试")
    result = None
    for line in (r.stdout or "").splitlines():
        if line.startswith(_RESULT_MARK):
            try:
                result = json.loads(line[len(_RESULT_MARK):])
            except ValueError:
                pass
    if r.returncode == 0 and result is not None:
        return result
    msg = (r.stderr or r.stdout or "").strip()
    if "No module named" in msg and "onnxruntime" in msg:
        msg += "\n提示：缺少 onnxruntime，请安装后重试: pip install onnxruntime"
    if "bad allocation" in msg or "out of memory" in msg.lower():
        msg += ("\n提示：onnxruntime 内存不足 —— 请关闭其他占用内存的程序"
                "后重试，或减小输入边长")
    raise ValueError(f"ONNX 推理失败:\n{msg[-1600:]}")


def probe_format(onnx_path):
    """ONNX 模型类型探测（API 侧）：子进程创建会话识别路线/模型/输入尺寸"""
    p = Path(onnx_path)
    if not p.is_file():
        raise ValueError(f"ONNX 模型文件不存在: {onnx_path}")
    return _run_subprocess(["probe", str(p)], timeout=180)


def run_infer_batch(images, onnx_path, conf_threshold=0.5):
    """批量图像 ONNX 推理（API 侧）。

    images: PIL.Image 列表（来自上传）→ 临时 PNG → 一次子进程批量推理
    （避免逐张重复启动会话）→ 结果列表（字段与 torch 推理对齐）。
    """
    from config import RESULTS_DIR
    p = Path(onnx_path)
    if not p.is_file():
        raise ValueError(f"ONNX 模型文件不存在: {onnx_path}")
    if not images:
        return []
    tmp_dir = RESULTS_DIR / f"_tmp_onnx_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        image_paths = []
        for i, img in enumerate(images):
            fp = tmp_dir / f"img_{i:03d}.png"
            img.convert("RGB").save(str(fp), format="PNG")
            image_paths.append(str(fp))
        result_json = tmp_dir / "_results.json"
        params = {"onnx_path": str(p), "image_paths": image_paths,
                  "conf_threshold": float(conf_threshold),
                  "result_json": str(result_json)}
        params_file = tmp_dir / "_params.json"
        params_file.write_text(json.dumps(params, ensure_ascii=False),
                               encoding="utf-8")
        _run_subprocess(["run", str(params_file)])
        if not result_json.is_file():
            raise ValueError("ONNX 推理子进程未产出结果文件")
        data = json.loads(result_json.read_text(encoding="utf-8"))
        return data.get("results") or []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ------------------------- 子进程入口 -------------------------

if __name__ == "__main__":
    # python -m app.services.onnx_infer probe <onnx_path>
    # python -m app.services.onnx_infer run <params.json>
    _mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if _mode == "probe":
            _info = _probe_core(sys.argv[2])
            print(_RESULT_MARK + json.dumps(_info, ensure_ascii=False),
                  flush=True)
        elif _mode == "run":
            _params = json.loads(
                Path(sys.argv[2]).read_text(encoding="utf-8"))
            _results = _run_core(_params)
            Path(_params["result_json"]).write_text(
                json.dumps({"results": _results}, ensure_ascii=False),
                encoding="utf-8")
            print(_RESULT_MARK + json.dumps({"ok": True}), flush=True)
        else:
            raise ValueError(f"未知子进程模式: {_mode}")
    except Exception as e:
        print(f"ONNX 推理失败: {e}", flush=True)
        raise SystemExit(1)
