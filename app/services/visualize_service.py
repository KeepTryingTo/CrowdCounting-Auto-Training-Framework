# -*- coding: utf-8 -*-
"""可视化服务：标签可视化 + 模型推理可视化

两种模式：
1. 标签可视化（无需模型，仅用标注文件生成叠加图）：
   - 密度图路线：.npy 密度图 → 热力图叠加（jet 着色 + count 标注）
   - 点预测路线：.mat 点标注 → 红色圆点叠加（count 标注）
   - 头部框路线：.txt（YOLO 格式）或 .xml（VOC 格式）→ 彩色框叠加

2. 模型推理可视化（与推理面板一致，但支持批量目录）：
   - 复用 crowd_infer / p2pnet_infer / headbox_infer 的可视化输出

支持：
- 单张图像（同步请求，返回 base64 可视化图）
- 目录批量（后台任务，逐图处理，结果保存到指定目录；
  完成后任务结果仅携带前 MAX_PREVIEW_IMAGES 张缩略预览供前端展示，
  全部结果以文件形式落盘）
"""
import base64
import io
import os
import random
import shutil
import uuid
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

from config import RESULTS_DIR

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 稳定类别颜色（与 yolo_service._PALETTE 一致）
_PALETTE = [(random.Random(i % 24).randint(50, 255),
             random.Random(i % 24 + 7).randint(50, 255),
             random.Random(i % 24 + 13).randint(50, 255))
            for i in range(24)]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _try_font(size=16):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_count(draw, count, W, H):
    """左上角标注 count"""
    font = _try_font(max(16, H // 25))
    label = f"count = {count:.1f}" if isinstance(count, float) else f"count = {count}"
    tb = draw.textbbox((8, 8), label, font=font)
    draw.rectangle([tb[0] - 4, tb[1] - 4, tb[2] + 4, tb[3] + 4], fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 0), font=font)


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# 标签可视化：各路线
# ---------------------------------------------------------------------------

def vis_label_density(img: Image.Image, density_npy) -> dict:
    """密度图标签可视化：jet 热力图 + 原图叠加 + count 标注"""
    density = np.load(density_npy) if isinstance(density_npy, (str, Path)) \
        else density_npy
    if density.ndim == 3:
        density = density[:, :, 0]
    img = img.convert("RGB")
    W, H = img.size
    if density.shape != (H, W):
        density = np.array(Image.fromarray(density).resize((W, H)))

    import matplotlib.cm as cm
    count = float(density.sum())
    d = density - density.min()
    if d.max() > 1e-8:
        d = d / d.max()
    heat = (cm.jet(d)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat)
    overlay = Image.blend(img, heat_img, alpha=0.5)

    draw = ImageDraw.Draw(overlay)
    _draw_count(draw, count, W, H)
    return {"count": round(count, 2), "visualization": _to_b64(overlay),
            "density_map": _to_b64(heat_img),
            "vis_type": "density"}


def vis_label_points(img: Image.Image, points_mat_or_arr) -> dict:
    """点标注可视化：红色圆点 + count 标注（参考 P2PNet engine.vis）"""
    if isinstance(points_mat_or_arr, (str, Path)):
        from app.services.crowd_counting import load_mat_points
        points, _ = load_mat_points(Path(points_mat_or_arr))
    else:
        points = np.asarray(points_mat_or_arr)

    import cv2
    img = img.convert("RGB")
    W, H = img.size
    canvas = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    size = max(2, int(max(H, W) / 300))
    for p in points:
        canvas = cv2.circle(canvas, (int(p[0]), int(p[1])), size,
                            (0, 0, 255), -1)
    vis = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    vis_img = Image.fromarray(vis)
    draw = ImageDraw.Draw(vis_img)
    _draw_count(draw, len(points), W, H)
    return {"count": int(len(points)), "visualization": _to_b64(vis_img),
            "num_points": len(points), "vis_type": "points"}


def vis_label_boxes_yolo(img: Image.Image, label_txt) -> dict:
    """YOLO 格式标签可视化：.txt 每行 'cls cx cy w h'（归一化）"""
    img = img.convert("RGB")
    W, H = img.size
    boxes = []
    if isinstance(label_txt, (str, Path)):
        content = Path(label_txt).read_text(encoding="utf-8-sig")
    else:
        content = label_txt
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
                boxes.append((cid, x1, y1, x2, y2))
            except (ValueError, IndexError):
                continue

    draw = ImageDraw.Draw(img)
    font = _try_font(14)
    for cid, x1, y1, x2, y2 in boxes:
        color = _PALETTE[cid % len(_PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"cls{cid}"
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2],
                       fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    _draw_count(draw, len(boxes), W, H)
    return {"count": len(boxes), "visualization": _to_b64(img),
            "num_boxes": len(boxes), "vis_type": "boxes"}


def vis_label_boxes_voc(img: Image.Image, label_xml) -> dict:
    """VOC XML 标签可视化：解析 xml 的 object/bndbox"""
    import xml.etree.ElementTree as ET
    img = img.convert("RGB")
    W, H = img.size
    if isinstance(label_xml, (str, Path)):
        tree = ET.parse(str(label_xml))
    else:
        tree = ET.fromstring(label_xml)

    boxes = []
    for obj in tree.iter("object"):
        name_el = obj.find("name")
        bb = obj.find("bndbox")
        if bb is None:
            continue
        cname = (name_el.text or "?").strip() if name_el is not None else "?"
        try:
            x1 = float(bb.findtext("xmin"))
            y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax"))
            y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        boxes.append((cname, x1, y1, x2, y2))

    draw = ImageDraw.Draw(img)
    font = _try_font(14)
    for cname, x1, y1, x2, y2 in boxes:
        # crc32 稳定散列（内置 hash() 进程随机化，同一类别重启后颜色会变）
        color = _PALETTE[zlib.crc32(str(cname).encode("utf-8"))
                         % len(_PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tb = draw.textbbox((x1, y1), cname, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2],
                       fill=color)
        draw.text((x1, y1), cname, fill=(255, 255, 255), font=font)
    _draw_count(draw, len(boxes), W, H)
    return {"count": len(boxes), "visualization": _to_b64(img),
            "num_boxes": len(boxes), "vis_type": "boxes_voc"}


# ---------------------------------------------------------------------------
# 模型推理可视化（复用推理函数）
# ---------------------------------------------------------------------------

def vis_model(img: Image.Image, route, model_name, weight_path=None,
              device="cpu", pretrained=True, conf_threshold=0.5) -> dict:
    """用模型对单张图像推理并返回可视化结果"""
    if route == "crowd_density":
        from app.services.crowd_counting import crowd_infer
        return crowd_infer(img, model_name=model_name,
                           weight_path=weight_path, device=device,
                           pretrained=pretrained)
    if route == "crowd_point":
        # Use the unified dispatcher so RAZ / D2C / PET / CLTR and custom
        # point models work in the visualization module too.
        from app.services.crowd_counting import crowd_infer
        return crowd_infer(img, model_name=model_name,
                           weight_path=weight_path, device=device,
                           pretrained=pretrained,
                           conf_threshold=conf_threshold)
    if route == "crowd_box":
        from app.services.yolo_service import headbox_infer
        return headbox_infer(img, model_name, weight_path=weight_path,
                             conf_threshold=conf_threshold, device=device,
                             pretrained=pretrained)
    if route == "crowd_sdg":
        from app.services.sdg_service import sdg_infer
        return sdg_infer(img, weight_path=weight_path, model_name=model_name,
                         device=device, pretrained=pretrained,
                         conf_threshold=conf_threshold)
    raise ValueError(f"未知路线: {route}")


# ---------------------------------------------------------------------------
# 标签文件匹配 + 批量可视化
# ---------------------------------------------------------------------------

def _find_label(img_path: Path, label_dir: Path, label_format: str):
    """按图像主名在标签目录中查找匹配的标签文件"""
    stem = img_path.stem
    ext_map = {
        "npy": ".npy", "mat": ".mat",
        "yolo_txt": ".txt", "voc_xml": ".xml",
    }
    ext = ext_map.get(label_format, ".txt")
    cand = label_dir / (stem + ext)
    if cand.exists():
        return cand
    # 尝试常见前缀（GT_IMG_1 ↔ IMG_1）
    for prefix in ("GT_", "gt_", "ann_", "label_"):
        if stem.startswith(prefix):
            cand = label_dir / (stem[len(prefix):] + ext)
            if cand.exists():
                return cand
        stripped = stem.replace(prefix, "")
        cand = label_dir / (prefix + stripped + ext)
        if cand.exists():
            return cand
    return None


# 批量可视化完成后随任务结果返回的预览图数量上限（前端「可视化结果」
# 仅展示前若干张，避免大量图像一次性回传；全部结果仍以文件形式落盘）
MAX_PREVIEW_IMAGES = 5


def _make_preview(vis_b64: str, name: str, count, max_side=640):
    """成功结果 → 展示用缩略图（base64 JPEG，长边 ≤ max_side）

    原图为全尺寸 PNG（叠加 count 标注），直接回传多张会撑大任务详情
    响应；这里降采样为缩略图，仅供前端预览展示
    """
    img = Image.open(io.BytesIO(base64.b64decode(vis_b64))).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {"image": name, "count": count,
            "vis_b64": base64.b64encode(buf.getvalue()).decode()}


def run_visualize_batch(task, params: dict):
    """后台任务：批量可视化（标签模式或模型模式）

    params:
      mode:         "label" | "model"
      route:        "crowd_density" | "crowd_point" | "crowd_box"
      image_dir:    图像目录
      label_dir:    标签目录（label 模式）
      label_format: "npy" | "mat" | "yolo_txt" | "voc_xml"（label 模式）
      model:        模型名（model 模式）
      weight_path:  权重路径（model 模式）
      device:       设备
      conf_threshold: 置信度阈值
      save_dir:     输出目录
    """
    mode = params.get("mode", "label")
    route = params.get("route", "crowd_density")
    image_dir = Path(params["image_dir"])
    save_result = bool(params.get("save_result", True))
    save_dir = Path(params.get("output_dir") or params["save_dir"])
    if not save_result:
        save_dir = RESULTS_DIR / f"_tmp_vis_{uuid.uuid4().hex[:8]}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 收集图像
    images = sorted(p for p in image_dir.rglob("*")
                    if p.suffix.lower() in IMG_EXTS and p.is_file())
    if not images:
        raise ValueError(f"目录中无图像: {image_dir}")

    # 标签模式下预检查
    label_dir = Path(params["label_dir"]) if params.get("label_dir") else None
    label_format = params.get("label_format", "yolo_txt")
    if mode == "label":
        if not label_dir or not label_dir.is_dir():
            raise ValueError(f"标签目录不存在: {label_dir}")
        # 预匹配
        matched = []
        for img_p in images:
            lbl = _find_label(img_p, label_dir, label_format)
            if lbl:
                matched.append((img_p, lbl))
        if not matched:
            raise ValueError(
                f"未找到匹配的标签文件。图像目录: {image_dir}，"
                f"标签目录: {label_dir}，格式: {label_format}")
        task.log(f"匹配到 {len(matched)}/{len(images)} 对图像-标签")
        images = matched  # (img_path, label_path) 元组

    task.total_epochs = len(images)
    task.log(f"可视化模式: {mode} | 路线: {route} | 图像数: {len(images)}")
    task.log(f"输出目录: {save_dir}")

    ok, failed = 0, 0
    results = []
    previews = []   # 前 MAX_PREVIEW_IMAGES 张成功结果的缩略预览
    for i, item in enumerate(images, 1):
        if task.stop_event.is_set():
            raise StopIteration()
        task.epoch = i
        task.progress = i / len(images) * 100.0
        try:
            if mode == "label":
                img_p, lbl_p = item
                img = Image.open(img_p).convert("RGB")
                if label_format == "npy":
                    r = vis_label_density(img, lbl_p)
                elif label_format == "mat":
                    r = vis_label_points(img, lbl_p)
                elif label_format == "voc_xml":
                    r = vis_label_boxes_voc(img, lbl_p)
                else:
                    r = vis_label_boxes_yolo(img, lbl_p)
            else:
                img_p = item
                img = Image.open(img_p).convert("RGB")
                r = vis_model(img, route,
                              params.get("model", ""),
                              weight_path=params.get("weight_path"),
                              device=params.get("device", "cpu"),
                              conf_threshold=float(
                                  params.get("conf_threshold", 0.5)))

            # 保存可视化图
            vis_b64 = r.get("visualization")
            if vis_b64:
                out_path = save_dir / (img_p.stem + "_vis.png")
                out_path.write_bytes(base64.b64decode(vis_b64))
                if len(previews) < MAX_PREVIEW_IMAGES:
                    previews.append(_make_preview(vis_b64, img_p.name,
                                                   r.get("count", 0)))
            ok += 1
            results.append({
                "image": img_p.name,
                "count": r.get("count", 0),
                "vis_file": str(out_path) if vis_b64 else "",
            })
            if i % 10 == 0 or i == len(images):
                task.log(f"[{i}/{len(images)}] {img_p.name} → "
                         f"count={r.get('count', 0)}")
        except Exception as e:
            failed += 1
            task.log(f"[{i}/{len(images)}] 失败 {item[0].name}: {e}")

    # 保存汇总
    import json
    (save_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_name = (f"vis_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:6]}.zip")
    zip_path = RESULTS_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in save_dir.glob("*_vis.png"):
            zf.write(f, f.name)
        if (save_dir / "results.json").is_file():
            zf.write(save_dir / "results.json", "results.json")
    task.result = {
        "save_dir": str(save_dir) if save_result else "",
        "output_dir": str(save_dir),
        "total": len(images), "processed": ok, "failed": failed,
        "results_preview": results[:20],
        "preview_images": previews,
        "zip_url": f"/results/{zip_name}",
    }
    task.log(f"可视化完成: 成功 {ok} 失败 {failed} | 输出: {save_dir}")
    if not save_result:
        shutil.rmtree(save_dir, ignore_errors=True)
