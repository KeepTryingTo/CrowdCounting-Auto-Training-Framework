# -*- coding: utf-8 -*-
"""ONNX 部署导出服务：权重 → ONNX + 部署片段代码

参考 PyTorch-DeepLearning-Visual-LLM/ObjectDetection/PyTorch_Load_pth2onnx
的导出（torch.onnx.export + onnx.checker）与 onnxruntime 推理范式：
导出后自动生成与权重模型类型匹配的「部署片段代码」（snippet.py），
用户复制片段 + onnx 文件即可在部署设备上直接推理，无需本框架。

三条路线的输出契约（片段代码据此编写）：
- crowd_density  密度网      : [B,1,H',W'] 密度图，count=sum；预处理
                               /255 + ImageNet 归一化
- crowd_point    P2PNet / CP : (pred_logits [B,M,2], pred_points [B,M,2])，
                               softmax→人概率，阈值筛选点坐标；16 倍数输入
- crowd_point    P2R         : [B,1,H/4,W/4] logit 图，logit>0 即人头点；
                               32 倍数输入
- crowd_box      YOLO        : ultralytics 原生导出（含元数据）；
                               标准 [B,4+nc,N] 需 NMS 后处理，
                               nms=True 内嵌版输出 [B,max_det,6] 免后处理

导出参数（可调）：opset / imgsz / dynamic_batch / simplify（onnxslim）/
half（仅 YOLO，CUDA）/ nms（仅 YOLO，端到端免后处理）。
"""
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

# 旧版 torch（<2.0）不支持 torch.load(weights_only=...)：独立运行本模块时
# 也要生效（通过 main 启动时已由 main 打过补丁，幂等）
from app.core.torch_compat import TORCH_LOAD_WEIGHTS_ONLY_PATCHED  # noqa: F401

from app.services import weight_format

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 密度网 / 点预测路线导出的默认 opset（P2PNet/密度网均为常规卷积算子）
_TORCH_EXPORT_OpSET_DEFAULT = 13


def _align_to(x, multiple):
    return max(multiple, int(round(x)) // multiple * multiple)


class _P2POnnxWrapper(nn.Module):
    """P2PNet / Consistent-Point / CU 导出包装：dict 输出 -> 有序元组
    （pred_logits, pred_points）；三者导出契约一致（CU 的不确定性分支
    不参与推理端计数，导出时略去）。"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out["pred_logits"], out["pred_points"]


class _SdgOnnxWrapper(nn.Module):
    """单域泛化（DCCUS/LLB/MPCount）导出包装：统一输出密度图

    MPCount 前向返回 (density, cls_map)，其余返回 density；导出只保留密度图。
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def _torch_onnx_export(model, dummy, out_path, *, input_name, output_names,
                        opset, dynamic_batch):
    """经典 Tracer 导出（参考 pth2onnx.py：torch.onnx.export + checker）"""
    dynamic = None
    if dynamic_batch:
        # dynamic_axes 以输入/输出【名字】为键（torch>=1.13 校验键名）
        dynamic = {input_name: {0: "batch"}}
        dynamic.update({n: {0: "batch"} for n in output_names})
    common = dict(opset_version=opset, input_names=[input_name],
                  output_names=output_names, dynamic_axes=dynamic)
    try:
        # 新版本 torch 支持 dynamo 开关，显式 False 强制走经典 Tracer；
        # 旧版本（如部分 2.x）没有该关键字，传了会 TypeError
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **common)
    except TypeError as e:
        if "dynamo" not in str(e):
            raise
        torch.onnx.export(model, dummy, str(out_path), **common)
    import onnx
    onnx.checker.check_model(onnx.load(str(out_path)))


def _slim_if_requested(onnx_path, simplify, log):
    """onnxslim 简化（与 ultralytics 引擎同款）；未安装或失败时跳过"""
    if not simplify:
        return False
    try:
        import onnx
        import onnxslim
        m = onnx.load(str(onnx_path))
        m = onnxslim.slim(m)
        onnx.save(m, str(onnx_path))
        log(f"onnxslim {onnxslim.__version__} 简化完成")
        return True
    except Exception as e:
        log(f"onnxslim 简化跳过（{e}）")
        return False


def _ort_probe(onnx_path, imgsz, dynamic_batch, log):
    """onnxruntime 冒烟验证（固定 batch=1，动态时另测 batch=2）。

    在【子进程】中执行：onnxruntime 与 torch 同进程存在 OpenMP 原生库
    冲突风险（偶发 0xC0000005），隔离后同时更贴近用户的独立部署环境。
    返回 (input_name, output_names, output_shapes)。
    """
    import subprocess
    import sys
    code = (
        "import json, numpy as np, onnxruntime as ort\n"
        f"s = ort.InferenceSession(r'{onnx_path}', "
        "providers=['CPUExecutionProvider'])\n"
        "iname = s.get_inputs()[0].name\n"
        "onames = [o.name for o in s.get_outputs()]\n"
        f"batches = {sorted(set([1, 2] if dynamic_batch else [1]))}\n"
        "out = {}\n"
        "for b in batches:\n"
        "    outs = s.run(None, {iname: np.zeros((b, 3, int(%d), int(%d)), "
        "dtype=np.float32)})\n" % (imgsz, imgsz) +
        "    out[str(b)] = [list(np.shape(o)) for o in outs]\n"
        "print(json.dumps({'input': iname, 'outputs': onames, "
        "'shapes': out}))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       timeout=300, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()
        if "bad allocation" in msg or "out of memory" in msg.lower():
            msg += ("\n提示：onnxruntime 初始化内存不足 —— 请关闭其他占用内存"
                    "的程序后重试，或减小输入边长 imgsz")
        raise ValueError(f"onnxruntime 验证失败:\n{msg}")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    shapes = info["shapes"]
    ok = all(len(shapes[str(b)]) == len(info["outputs"])
             for b in shapes)
    if not ok:
        raise ValueError("onnxruntime 输出数量不一致，请检查导出图")
    log(f"onnxruntime 验证通过 | 输入 {info['input']} "
        f"[B,3,{imgsz},{imgsz}] | 输出 {info['outputs']} -> "
        f"{shapes.get('1')}"
        + (f"（batch=2 {shapes.get('2')}）" if "2" in shapes else ""))
    return info["input"], info["outputs"], shapes.get("1")


# ------------------------- 片段代码生成（部署模板） -------------------------

def _snippet_header(meta, extra_note=""):
    route_names = {"crowd_density": "密度网（人群计数 · 密度图）",
                   "crowd_point": "点预测（人群计数）",
                   "crowd_box": "YOLO 头部框检测（人群计数）"}
    return f'''# -*- coding: utf-8 -*-
"""{route_names.get(meta['route'], meta['route'])} ONNX 部署片段代码
由人群计数框架「部署模式」自动生成 —— 直接复制本文件与 model.onnx
（保持同目录）到部署设备即可运行。

依赖：onnxruntime / numpy / opencv-python（无需本框架与 PyTorch）
模型: {meta['model_label']}    权重来源: {Path(meta['weight_path']).name}
输入: {meta['imgsz']}x{meta['imgsz']}（RGB，0-1，ImageNet 归一化）
opset: {meta['opset']}    动态 batch: {meta['dynamic_batch']}
{extra_note}用法:
  python snippet.py <图像路径> [阈值]        # 单张图像推理并保存可视化
  python snippet.py camera [摄像头编号]      # 打开摄像头实时推理（q/ESC 退出）
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

ONNX_PATH = "model.onnx"   # 片段代码默认读取同目录 model.onnx
IMG_SIZE = {meta['imgsz']}
CONF_THR = {meta.get('conf_threshold', 0.5)}
MEAN = np.array({IMAGENET_MEAN}, dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array({IMAGENET_STD}, dtype=np.float32).reshape(1, 3, 1, 1)


def imread(path):
    """读取图像（np.fromfile + imdecode：兼容 Windows 非 ASCII 路径）"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    """保存图像（imencode + tofile：兼容 Windows 非 ASCII 路径）"""
    cv2.imencode(Path(path).suffix or ".png", img)[1].tofile(path)
'''


_SNIPPET_RUNNER = '''


def run_image(img_path, thr=CONF_THR):
    """单张图像推理，并把可视化结果保存到同目录。"""
    img = imread(img_path)
    sess = ort.InferenceSession(ONNX_PATH,
                                providers=["CPUExecutionProvider"])
    vis, count = predict(img, sess, thr)
    print(f"估计人数: {count}")
    out_path = img_path.rsplit(".", 1)[0] + OUT_SUFFIX + ".png"
    imwrite(out_path, vis)
    print(f"可视化已保存: {out_path}")


def _gui_available():
    """检测 OpenCV 是否带 GUI（headless 构建无 imshow/destroyAllWindows）"""
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return True
    for line in info.splitlines():
        if "GUI:" in line:
            return "NONE" not in line.upper()
    return True


def run_camera(cam_index=0, thr=CONF_THR):
    """打开摄像头实时推理（按 q / ESC 退出）。"""
    if not _gui_available():
        print("当前 OpenCV 为 headless 构建（GUI: NONE），无法显示摄像头窗口。")
        print("请安装带 GUI 的 OpenCV 后重试，例如：")
        print("  pip uninstall -y opencv-python opencv-python-headless "
              "opencv-contrib-python opencv-contrib-python-headless")
        print("  pip install opencv-python")
        return
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"无法打开摄像头 {cam_index}（请检查设备编号/是否被占用）")
        return
    sess = ort.InferenceSession(ONNX_PATH,
                                providers=["CPUExecutionProvider"])
    print("摄像头已开启：按 q / ESC 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取摄像头失败（设备断开或被占用）")
            break
        vis, count = predict(frame, sess, thr)
        cv2.putText(vis, f"Count: {count}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
        try:
            cv2.imshow("Crowd Counting (ONNX)", vis)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as e:
            print("无法创建显示窗口（当前环境无图形界面？）:", e)
            break
        if key in (27, ord("q")):
            break
    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].lower() in ("camera", "cam", "--camera", "-c"):
        run_camera(int(args[1]) if len(args) > 1 else 0)
    else:
        run_image(args[0] if args else "test.jpg",
                  float(args[1]) if len(args) > 1 else CONF_THR)
'''


def _snippet_density(meta):
    return _snippet_header(meta) + '''

OUT_SUFFIX = "_density"


def preprocess(img_bgr):
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE)).transpose(2, 0, 1)[None]
    return (x - MEAN) / STD


def predict(img_bgr, sess, thr=CONF_THR):
    """返回 (已叠加密度热力图的图像, 估计人数)。密度图路线忽略 thr。"""
    density = sess.run(None, {"input": preprocess(img_bgr)})[0][0, 0]
    count = round(float(density.sum()), 2)

    d = density - density.min()
    if d.max() > 1e-8:
        d /= d.max()
    heat = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # 密度热力图放大到输入图像尺寸（与原图一致）
    heat = cv2.resize(heat, (img_bgr.shape[1], img_bgr.shape[0]))
    vis = cv2.addWeighted(img_bgr, 0.5, heat, 0.5, 0)
    return vis, count
''' + _SNIPPET_RUNNER


def _snippet_p2pnet(meta):
    return _snippet_header(meta, "注意：输入宽高须为 16 的倍数（FPN 三级严格 "
                                 "上采样）；点坐标按输入尺寸推理，再映射回原图。\n") + '''

OUT_SUFFIX = "_points"


def preprocess(img_bgr):
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE)).transpose(2, 0, 1)[None]
    return (x - MEAN) / STD


def predict(img_bgr, sess, thr=CONF_THR):
    """返回 (已标注预测点的图像, 估计人数)。"""
    h, w = img_bgr.shape[:2]
    pred_logits, pred_points = sess.run(
        None, {"input": preprocess(img_bgr)})     # [1,M,2] logits, [1,M,2] (x,y)
    prob = 1.0 / (1.0 + np.exp(-(pred_logits[0, :, 1] - pred_logits[0, :, 0])))
    pts = pred_points[0][prob > thr]           # 输入尺寸坐标系下的点
    scale = np.array([w / IMG_SIZE, h / IMG_SIZE], dtype=np.float32)
    pts = pts * scale                          # 映射回原图坐标
    vis = img_bgr.copy()
    for p in pts.astype(int):
        cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
    return vis, int(len(pts))
''' + _SNIPPET_RUNNER


def _snippet_p2r(meta):
    return _snippet_header(meta, "注意：输入宽高须为 32 的倍数；输出为 1/4 分\n"
                                 "辨率 logit 图，logit>0 的位置即人头点（与\n"
                                 "训练判定一致，置信度阈值无效）。\n") + '''

OUT_SUFFIX = "_points"


def preprocess(img_bgr):
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE)).transpose(2, 0, 1)[None]
    return (x - MEAN) / STD


def predict(img_bgr, sess, thr=CONF_THR):
    """返回 (已标注预测点的图像, 估计人数)。P2R 路线忽略 thr。"""
    h, w = img_bgr.shape[:2]
    logits = sess.run(None, {"input": preprocess(img_bgr)})[0][0, 0]  # [H/4,W/4]
    down = IMG_SIZE // logits.shape[0]                            # 下采样倍率
    rows, cols = np.nonzero(logits > 0)
    pts = np.stack([cols, rows], 1) * down + (down - 1) / 2.0      # (x,y)
    scale = np.array([w / IMG_SIZE, h / IMG_SIZE], dtype=np.float32)
    pts = (pts * scale).astype(int)
    vis = img_bgr.copy()
    for p in pts:
        cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
    return vis, int(len(pts))
''' + _SNIPPET_RUNNER


def _snippet_yolo(meta):
    cls_list = meta.get("classes") or []
    names_block = "\n".join(
        f"    {i}: {c!r}," for i, c in enumerate(cls_list)) or "    0: 'head',"
    # 端到端（nms=True 内嵌）：输出 [B, max_det, 6]，免手工 NMS
    if meta.get("end2end"):
        predict = '''
OUT_SUFFIX = "_boxes"


def predict(img_bgr, sess, thr=CONF_THR):
    """返回 (已绘制检测框的图像, 检测框数=人头数)。"""
    ib, r, dw, dh = letterbox(img_bgr)
    dets = sess.run(None, {"%s": ib})[0][0]      # [max_det, 6] xyxy conf cls
    dets = dets[dets[:, 4] > thr]
    if len(dets):
        dets[:, [0, 2]] = (dets[:, [0, 2]] - dw) / r   # letterbox 还原到原图
        dets[:, [1, 3]] = (dets[:, [1, 3]] - dh) / r
    vis = img_bgr.copy()
    for x1, y1, x2, y2, conf, cls_id in dets:
        c = CLASS_NAMES.get(int(cls_id), str(cls_id))
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 120), 2)
        cv2.putText(vis, f"{c} {conf:.2f}", (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 1)
    return vis, int(len(dets))''' % meta["input_name"]
    else:
        predict = '''
OUT_SUFFIX = "_boxes"


def nms_numpy(boxes, scores, iou_thr=0.45):
    """纯 numpy NMS（部署无需 torch / torchvision）"""
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


def predict(img_bgr, sess, thr=CONF_THR):
    """返回 (已绘制检测框的图像, 检测框数=人头数)。"""
    ib, r, dw, dh = letterbox(img_bgr)
    out = sess.run(None, {"%s": ib})[0][0]        # [4+nc, N]（已解码 xywh+分数）
    boxes_xywh, scores = out[:4], out[4:]          # xywh 像素坐标 / 各类分数
    cls_ids = scores.argmax(0)
    conf = scores.max(0)
    keep = conf > thr
    boxes, conf, cls_ids = boxes_xywh.T[keep], conf[keep], cls_ids[keep]
    xy = boxes[:, :2]; wh = boxes[:, 2:]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)   # cxcywh -> xyxy
    keep2 = nms_numpy(boxes, conf)
    boxes, conf, cls_ids = boxes[keep2], conf[keep2], cls_ids[keep2]
    if len(boxes):
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r   # letterbox 还原到原图
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
    vis = img_bgr.copy()
    for (x1, y1, x2, y2), cf, ci in zip(boxes, conf, cls_ids):
        c = CLASS_NAMES.get(int(ci), str(ci))
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 120), 2)
        cv2.putText(vis, f"{c} {cf:.2f}", (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 1)
    return vis, int(len(boxes))''' % meta["input_name"]
    return f'''# -*- coding: utf-8 -*-
"""YOLO 头部框检测 ONNX 部署片段代码（人群计数）
由人群计数框架「部署模式」自动生成 —— 复制本文件与 model.onnx 到部署
设备即可运行（依赖：onnxruntime / numpy / opencv-python）。

模型: {meta['model_label']}    权重来源: {Path(meta['weight_path']).name}
输入: letterbox 到 {meta['imgsz']}x{meta['imgsz']} (RGB, 0-1，不归一化)
输出: {'端到端 NMS 内嵌 [max_det, 6]，免手工后处理' if meta.get('end2end') else '[4+nc, N]，需 numpy NMS 后处理'}
opset: {meta['opset']}    动态 batch: {meta['dynamic_batch']}
用法:
  python snippet.py <图像路径> [置信度阈值]     # 单张图像推理并保存可视化
  python snippet.py camera [摄像头编号]         # 打开摄像头实时推理（q/ESC 退出）
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

ONNX_PATH = "model.onnx"   # 片段代码默认读取同目录 model.onnx
IMG_SIZE = {meta['imgsz']}
CONF_THR = {meta.get('conf_threshold', 0.5)}
CLASS_NAMES = {{
{names_block}
}}


def imread(path):
    """读取图像（np.fromfile + imdecode：兼容 Windows 非 ASCII 路径）"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    """保存图像（imencode + tofile：兼容 Windows 非 ASCII 路径）"""
    cv2.imencode(Path(path).suffix or ".png", img)[1].tofile(path)


def letterbox(img):
    """等比缩放 + 灰边填充到 IMG_SIZE（与 ultralytics 训练/验证一致）
    返回 (输入张量 [1,3,S,S], 缩放比 r, 填充 dw, dh)"""
    h, w = img.shape[:2]
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh))
    dh, dw = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
    canvas = cv2.copyMakeBorder(resized, dh, IMG_SIZE - nh - dh,
                                dw, IMG_SIZE - nw - dw,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None], r, dw, dh

''' + predict + _SNIPPET_RUNNER


def build_snippet(meta):
    """按模型类型生成部署片段代码"""
    if meta["route"] == "crowd_density":
        return _snippet_density(meta)
    if meta["route"] == "crowd_sdg":
        return _snippet_density(meta)
    if meta["route"] == "crowd_box":
        return _snippet_yolo(meta)
    if meta["model"] == "p2r":
        return _snippet_p2r(meta)
    return _snippet_p2pnet(meta)


# ------------------------- 导出任务入口 -------------------------

# 子进程运行标记前缀（父进程解析：进度 / 结果 JSON）
_RESULT_MARK = "__RESULT__"
_PROGRESS_MARK = "__PROGRESS__"


def _export_core(params: dict, log, progress):
    """导出核心（无 Task 耦合，可在子进程独立运行）：
    权重 → 格式推断 → 构建/导出 → 简化/校验 → 片段代码与产物，
    返回结果 dict（调用方据此构造任务 result）。"""
    weight_path = params["weight_path"]
    imgsz = int(params.get("imgsz", 0))
    opset = int(params.get("opset", _TORCH_EXPORT_OpSET_DEFAULT))
    dynamic_batch = bool(params.get("dynamic_batch", False))
    simplify = bool(params.get("simplify", True))
    half = bool(params.get("half", False))
    nms = bool(params.get("nms", False))

    save_dir = Path(params["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    progress(5)
    fmt = weight_format.infer_weight_format(weight_path)
    route = fmt["route"]
    model_name = fmt["model"]
    log(f"权重格式推断: {fmt['detail']}")
    if model_name == "auto":
        log("密度网具体骨干探测中（逐候选架构严格加载校验）...")
        model_name = weight_format.probe_density_model(weight_path, log=log)
        log(f"骨干探测命中: {model_name}")
    if not imgsz:
        imgsz = fmt["imgsz"]
    if route == "crowd_point":
        imgsz = _align_to(imgsz, fmt["multiple"])
    if route == "crowd_sdg":
        imgsz = _align_to(imgsz, fmt["multiple"])
    if route == "crowd_box":
        imgsz = max(32, imgsz - imgsz % 32)
    log(f"导出参数 | 路线 {route} | 模型 {model_name} | "
        f"输入 {imgsz} | opset {opset} | 动态batch {dynamic_batch} | "
        f"简化 {simplify}" + (f" | half {half} | nms {nms}"
                              if route == "crowd_box" else ""))

    onnx_path = save_dir / "model.onnx"
    progress(20)

    # ------------------------- 路线一：YOLO（ultralytics 原生导出） -------------------------
    if route == "crowd_box":
        try:
            from app.services import yolo_service as ys
            YOLO = ys._require_ultralytics()
        except Exception as e:
            raise ValueError(f"头部框路线导出需要 ultralytics 引擎: {e}")
        y = YOLO(str(weight_path))
        classes = y.names if isinstance(y.names, dict) else dict(
            enumerate(y.names or []))
        classes = [str(classes[i]) for i in range(len(classes))]
        kwargs = dict(format="onnx", imgsz=imgsz, opset=opset,
                      dynamic=dynamic_batch, simplify=simplify,
                      device="cpu")
        if nms:
            kwargs["nms"] = True
        if half:
            kwargs["half"] = True       # CPU 导出时引擎自动回退 fp32 并提示
        log("ultralytics 原生 ONNX 导出中（含元数据，输入 letterbox）...")
        try:
            out = y.export(**kwargs)
        except Exception as e:
            raise ValueError(f"ultralytics ONNX 导出失败: {e}")
        shutil.copyfile(out, onnx_path)
        progress(55)
        input_name, output_names, out_shapes = _ort_probe(
            onnx_path, imgsz, dynamic_batch, log)
        # 端到端判定：输出 3 维且末维=6（xyxy conf cls）
        end2end = (len(out_shapes) == 1 and len(out_shapes[0]) == 3
                   and out_shapes[0][-1] == 6)
        if end2end:
            log("检测到端到端 NMS 内嵌输出 [max_det, 6]，"
                "片段代码将免手工后处理")

    # ------------------------- 路线二：torch.onnx.export（密度网 / 点预测） -------------------------
    else:
        model = _build_torch_model(route, model_name, weight_path, log)
        model.eval()
        dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
        input_name = "input"
        if route == "crowd_point" and model_name != "p2r":
            out_names = ["pred_logits", "pred_points"]
            export_model = _P2POnnxWrapper(model)
        elif route == "crowd_sdg":
            out_names = ["density"]
            export_model = _SdgOnnxWrapper(model)
        else:
            out_names = (["density"] if route == "crowd_density"
                         else ["density_logits"])
            export_model = model
        with torch.no_grad():
            # 记录导出尺寸下的实际输出（尺寸校验 + 元信息）
            probe = export_model(dummy)
            probe_shapes = [list(p.shape) for p in
                            (probe if isinstance(probe, tuple) else (probe,))]
        log(f"torch.onnx.export 导出中（{input_name} -> {out_names}，"
            f"输出形状 {probe_shapes}）...")
        with torch.no_grad():
            _torch_onnx_export(export_model, dummy, onnx_path,
                               input_name=input_name, output_names=out_names,
                               opset=opset, dynamic_batch=dynamic_batch)
        progress(55)
        _slim_if_requested(onnx_path, simplify, log)
        input_name, output_names, out_shapes = _ort_probe(
            onnx_path, imgsz, dynamic_batch, log)
        end2end = False
        classes = fmt.get("classes")

    # ------------------------- 片段代码与产物 -------------------------
    progress(80)
    meta = {"route": route, "model": model_name,
            "model_label": fmt.get("model_label", model_name),
            "weight_path": str(weight_path), "imgsz": imgsz, "opset": opset,
            "dynamic_batch": dynamic_batch, "end2end": end2end,
            "onnx_path": str(onnx_path), "input_name": input_name,
            "conf_threshold": float(params.get("conf_threshold", 0.5))}
    if classes:
        meta["classes"] = classes
    snippet = build_snippet(meta)
    (save_dir / "snippet.py").write_text(snippet, encoding="utf-8")
    (save_dir / "meta.json").write_text(
        json.dumps({**meta, "input_name": input_name,
                    "output_names": output_names,
                    "output_shapes": out_shapes,
                    "multiple": fmt["multiple"], "source": fmt["source"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # 说明文件（离线可读：参数 / 输出契约 / 用法 / 部署建议）
    (save_dir / "README.txt").write_text(
        f"ONNX 部署包（人群计数框架 · 部署模式）\n"
        f"{'=' * 46}\n"
        f"模型类型 : {fmt['route_label']}\n"
        f"模型     : {meta['model_label']}\n"
        f"权重来源 : {weight_path}\n"
        f"输入尺寸 : {imgsz}x{imgsz}（{'动态 batch' if dynamic_batch else '固定 batch=1'}）"
        f"  opset {opset}\n"
        f"输出契约 : {output_names} -> {out_shapes}\n"
        f"对齐约束 : 输入宽高须为 {fmt['multiple']} 的倍数"
        + ("\n提示     : half 导出仅 CUDA 有效，CPU 已自动回退 fp32"
           if route == "crowd_box" and half else "") + "\n\n"
        f"文件:\n"
        f"  model.onnx  ONNX 中间表示模型（可部署到 ONNX Runtime /\n"
        f"              TensorRT / OpenVINO / 移动端 NNAPI 等）\n"
        f"  snippet.py  部署推理片段代码（onnxruntime + numpy + opencv，\n"
        f"              无需本框架）\n"
        f"  meta.json   导出元信息\n\n"
        f"用法: python snippet.py <图像路径>"
        + (" [置信度阈值]" if route != "crowd_point" or model_name != "p2r"
           else "") + "\n", encoding="utf-8")

    progress(100)
    size_mb = round(onnx_path.stat().st_size / 1048576, 2)
    log(f"ONNX 导出完成: {onnx_path}（{size_mb} MB）| "
        f"片段代码 {save_dir / 'snippet.py'}")
    return {
        "onnx_path": str(onnx_path),
        "onnx_url": f"/runs/onnx/{save_dir.name}/model.onnx",
        "snippet_path": str(save_dir / "snippet.py"),
        "snippet_url": f"/runs/onnx/{save_dir.name}/snippet.py",
        "snippet": snippet,
        "meta": {k: v for k, v in meta.items() if k != "onnx_path"},
        "output_names": output_names, "output_shapes": out_shapes,
        "size_mb": size_mb, "save_dir": str(save_dir),
        "route": route, "model": model_name,
    }


def run_onnx_export(task, params: dict):
    """ONNX 导出后台任务：在【隔离子进程】中执行导出核心。

    子进程化原因（生产健壮性）：
    1. 模型构建（528MB 级 P2PNet 骨干 / ultralytics 引擎）与 onnx 导出
       的原生库在长驻服务进程中偶发访问冲突（0xC0000005），隔离后崩溃
       只影响导出子进程，错误清晰上报，服务本身不受影响；
    2. 导出峰值内存随子进程退出完全释放；
    3. onnxruntime 校验天然独立（与用户的独立部署环境一致）。
    日志逐行回传任务；__PROGRESS__ / __RESULT__ 标记解析进度与结果。
    """
    import os
    import subprocess
    import sys
    import threading
    from collections import deque

    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    task.progress = 3
    params = {**params, "save_dir": str(save_dir)}
    params_file = save_dir / "_export_params.json"
    params_file.write_text(json.dumps(params, ensure_ascii=False),
                           encoding="utf-8")

    root = Path(__file__).resolve().parents[2]   # Codes 根
    env = {**os.environ, "PYTHONPATH": str(root),
           "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-u", "-m", "app.services.onnx_export",
           str(params_file)]
    task.log("ONNX 导出子进程启动（与主服务进程隔离）...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace",
                            env=env, cwd=str(root))

    def _watch_stop():
        """用户停止任务 → 终止导出子进程"""
        while proc.poll() is None:
            if task.stop_event.is_set():
                proc.kill()
                return
            task.stop_event.wait(1.0)

    threading.Thread(target=_watch_stop, daemon=True).start()

    tail = deque(maxlen=30)
    result = None
    for line in proc.stdout:
        line = line.rstrip("\r\n")
        if line.startswith(_PROGRESS_MARK):
            try:
                task.progress = max(task.progress,
                                    float(line[len(_PROGRESS_MARK):]))
            except ValueError:
                pass
            continue
        if line.startswith(_RESULT_MARK):
            result = json.loads(line[len(_RESULT_MARK):])
            continue
        if line:
            task.log(line)
            tail.append(line)
    code = proc.wait()
    params_file.unlink(missing_ok=True)
    if task.stop_event.is_set():
        task.log("收到停止信号，ONNX 导出已终止")
        raise StopIteration()
    if code != 0 or result is None:
        msg = "\n".join(list(tail)[-12:])
        if "not enough memory" in msg or "bad allocation" in msg:
            msg += ("\n提示：导出内存不足 —— 请关闭其他占用内存的程序后重试，"
                    "或减小输入边长 imgsz")
        raise ValueError(f"ONNX 导出失败（子进程退出码 {code}）:\n{msg}")
    task.progress = 100
    task.result = result


def _build_torch_model(route, model_name, weight_path, log):
    """按路线构建 torch 模型并加载权重（密度网 / 点预测）"""
    # 单域泛化（DCCUS / LLB_Count / MPCount）：密度图输出，结构由
    # checkpoint 内 sdg_meta 决定（模型名 / 骨干 / 聚类数）
    if route == "crowd_sdg":
        from app.services import sdg_service
        ck = torch.load(str(weight_path), map_location="cpu",
                        weights_only=False)
        meta = ck.get("sdg_meta", {}) if isinstance(ck, dict) else {}
        model = sdg_service._build_model(
            meta.get("model_name") or model_name,
            meta.get("backbone"),
            int(meta.get("num_clusters") or 4))
        sdg_service._load_weights(model, weight_path)
        return model
    # PET 使用 NestedTensor + 四叉树动态查询，经典 ONNX Tracer 无法导出；
    # 在构建模型前按 checkpoint 键名提前给出明确提示
    try:
        from app.services.weight_format import safe_load_ckpt
        _ck = safe_load_ckpt(weight_path)
        if isinstance(_ck, dict):
            _sd = (_ck.get("model") or _ck.get("model_state_dict")
                   or _ck.get("state_dict") or _ck)
        else:
            _sd = _ck
        if isinstance(_sd, dict) and any(
                str(k).startswith(("quadtree_", "context_encoder."))
                for k in _sd):
            raise ValueError(
                "PET 模型暂不支持 ONNX 导出（NestedTensor + 四叉树动态查询"
                "结构）；请使用 PyTorch 权重进行推理")
    except ValueError:
        raise
    except Exception:
        pass

    if route == "crowd_point" and model_name not in ("p2pnet", "p2r", "cp", "cu"):
        raise ValueError(
            f"模型 {model_name} 暂不支持 ONNX 导出（当前仅支持 P2PNet / "
            f"P2R / Consistent-Point / CU；PET / CLTR / 自定义点预测模型"
            f"请使用 PyTorch 权重推理）")

    if route == "crowd_density":
        from app.services.crowd_counting import (_build_crowd_model,
                                                 _load_weights_compat)
        model = _build_crowd_model(model_name, pretrained=False)
        _load_weights_compat(model, weight_path, context="ONNX 导出权重")
        return model
    # 点预测
    if model_name == "p2r":
        from app.services.crowd_counting import _build_p2r
        model = _build_p2r(pretrained=False)
    elif model_name == "cp":
        from app.services.crowd_counting import _build_cpoint
        model = _build_cpoint(pretrained=False, training=False)
    elif model_name == "cu":
        from app.services.crowd_counting import _build_cu
        model = _build_cu(pretrained=False, training=False)
    else:
        # P2PNet：构建器固定加载 528MB VGG16_bn 预训练权重，但导出时会被
        # 用户权重 strict 全量覆盖 —— 本进程内临时打补丁跳过该加载，
        # 将导出峰值内存降低约 1GB（补丁仅在隔离子进程内生效）
        from app.services.crowd_counting import _build_p2pnet
        from nets.p2pnet.models import vgg_ as _vgg
        _orig_vgg = _vgg.vgg16_bn
        _vgg.vgg16_bn = lambda pretrained=True: _orig_vgg(pretrained=False)
        try:
            model = _build_p2pnet(training=False, device="cpu")
        finally:
            _vgg.vgg16_bn = _orig_vgg
    from app.services.weight_format import safe_load_ckpt
    ck = safe_load_ckpt(weight_path)
    sd = None
    if isinstance(ck, dict):
        sd = ck.get("model_state_dict") or ck.get("state_dict")
    if not isinstance(sd, dict):
        sd = ck
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model


if __name__ == "__main__":
    # 子进程入口（须置于文件末尾，确保依赖函数均已定义）：
    # python -m app.services.onnx_export <params.json>
    import sys
    _params = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    def _log(msg):
        print(str(msg), flush=True)

    def _prog(v):
        print(f"{_PROGRESS_MARK}{v}", flush=True)

    try:
        _result = _export_core(_params, log=_log, progress=_prog)
    except Exception as e:
        print(f"导出失败: {e}", flush=True)
        raise SystemExit(1)
    print(_RESULT_MARK + json.dumps(_result, ensure_ascii=False), flush=True)
