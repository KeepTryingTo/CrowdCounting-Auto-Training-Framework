# -*- coding: utf-8 -*-
"""缩放区域提议（Zoom Region Proposal）与 RAZ 推理工具。

本文件实现论文中“注意力引导的局部放大”全流程：

1. Main-Net 输出一张注意力图 attn，值越大表示该区域越拥挤、越需要放大；
2. 在注意力图上随机采样 m 个兴趣点；
3. 每个兴趣点套用 3 种比例 × 2 种尺度的锚框，得到候选子图；
4. 用积分图计算每个候选子图的平均注意力，丢弃低于 theta_z 的候选；
5. 按平均注意力排序，做 IoU NMS（默认 IoU>0.5 删除低分框），最多保留 kz 个；
6. 把选中的子图裁剪并放大 scale 倍（论文为 2 倍），送入 RAZ-Net 再定位；
7. 把子图内的预测点映射回原图坐标，并用距离去重后合并到 Main-Net 结果。

坐标系统约定（很容易混淆，务必区分）：
- “原图坐标”：用户输入图像的像素坐标，宽 W、高 H；
- “处理后图像坐标”：按 max_side 等比缩放后的坐标（推理时默认长边 1536），
  本文件返回的 points 默认都在这个坐标系；
- “crop 坐标”：从处理后图像裁出的子图坐标，经过 scale 放大后送网络；
- 锚框/返回值 select_zoom_boxes 的框坐标属于“处理后图像坐标”。

关键参数：
- m        : 随机兴趣点数量（论文 200）
- theta_z  : 候选框平均注意力阈值（论文 0.01）
- kz       : 每张图最多保留的缩放框数量（论文 10）
- iou_thr  : NMS 的 IoU 阈值（论文 0.5）
- scale    : 子图放大倍数（论文 2.0）
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .datasets.points import load_points
from .postprocess import extract_peaks, merge_points

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_numpy_attn(attn) -> np.ndarray:
    """把注意力图统一成 CPU 上的二维 float ndarray。

    兼容模型输出常见的 [B,1,H,W]、[1,H,W]、[H,W] 以及 numpy 输入；
    多余的 batch/channel 维只取第 0 个。
    """
    if isinstance(attn, torch.Tensor):
        a = attn.detach().float().cpu()
        if a.ndim == 4:
            a = a[0]
        if a.ndim == 3:
            a = a[0]
        return a.numpy()
    a = np.asarray(attn, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    return a


def _iou(a, b) -> float:
    """计算两个框的 IoU。

    参数:
        a, b: 形如 (x1, y1, x2, y2) 的矩形，x2>x1、y2>y1。

    返回:
        float，范围 [0,1]；交集为 0 时返回 0。
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, aa + bb - inter)


def _nms(boxes: List[Dict], iou_thr: float = 0.5) -> List[Dict]:
    """按 score 从高到低做贪心 NMS。

    每个候选框与已保留框比较：IoU > iou_thr 的框被抑制，否则保留。
    boxes 里的每个 dict 至少包含 x1/y1/x2/y2/score。
    """
    boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)
    keep = []
    for b in boxes:
        if all(_iou((b["x1"], b["y1"], b["x2"], b["y2"]),
                    (k["x1"], k["y1"], k["x2"], k["y2"])) <= iou_thr
               for k in keep):
            keep.append(b)
    return keep


def select_zoom_boxes(attn, image_size: Tuple[int, int], *,
                      m: int = 200, theta_z: float = 0.01,
                      kz: int = 10, iou_thr: float = 0.5,
                      seed: Optional[int] = None) -> List[Dict]:
    """在注意力图上选择最多 kz 个需要放大的子图。

    参数:
        attn:
            注意力图，可为 [B,1,H,W] / [1,H,W] / [H,W] 的 torch Tensor
            或 numpy 数组。这里取第 0 个 batch/channel。
        image_size:
            输出框所在的坐标尺寸，格式 (W, H)，通常是“处理后图像”尺寸。
            attn 的 H/W 可以和它不同，函数内部会自动按比例映射。
        m:
            随机兴趣点数量；论文默认 200。若大于注意力图像素数会被截断。
        theta_z:
            候选框平均注意力下限；论文默认 0.01。低于它的框直接丢弃。
        kz:
            最终最多保留多少个框；论文默认 10。
        iou_thr:
            NMS 的 IoU 阈值；论文默认 0.5。
        seed:
            随机种子；同一种子可复现兴趣点采样。

    返回:
        List[dict]，每个元素格式为：
        {
            "x1": float, "y1": float, "x2": float, "y2": float,  # image_size 坐标
            "score": float,   # 框内平均注意力
        }
        按 score 从高到低排序，且数量 <= kz。
    """
    a = _to_numpy_attn(attn)
    if a.size == 0:
        return []
    W, H = int(image_size[0]), int(image_size[1])
    ah, aw = a.shape

    # 积分图：integral[y, x] 表示 a[:y, :x] 的累加和。
    # 这样任意矩形区域的注意力和可在 O(1) 内算出，避免对 200×6 个候选
    # 框逐个遍历像素。
    integral = np.pad(a.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def mean_box(x1, y1, x2, y2) -> float:
        """计算 image_size 坐标系下矩形框内的平均注意力。"""
        if x2 <= x1 or y2 <= y1:
            return 0.0
        # 把 image_size 坐标映射到注意力图坐标：x 按 W/aw、y 按 H/ah 缩放。
        ax1 = int(np.clip(math.floor(x1 / max(1, W) * aw), 0, aw))
        ax2 = int(np.clip(math.ceil(x2 / max(1, W) * aw), 0, aw))
        ay1 = int(np.clip(math.floor(y1 / max(1, H) * ah), 0, ah))
        ay2 = int(np.clip(math.ceil(y2 / max(1, H) * ah), 0, ah))
        if ax2 <= ax1 or ay2 <= ay1:
            return 0.0
        # 积分图四点公式：区域和 = 右下 - 右上 - 左下 + 左上。
        s = (integral[ay2, ax2] - integral[ay1, ax2] -
             integral[ay2, ax1] + integral[ay1, ax1])
        return float(s / max(1, (ax2 - ax1) * (ay2 - ay1)))

    # 论文锚框：1:1 / 1:2 / 2:1 三种比例 × 两种尺度（尺度2面积是尺度1的4倍）。
    # 小尺度 1:1 的边长 = 图像短边 / 3；同尺度不同比例保持面积相同：
    #   w = sqrt(area * ratio)，h = sqrt(area / ratio)。
    short = max(1.0, min(W, H))
    base = short / 3.0
    ratios = (1.0, 0.5, 2.0)
    anchors = []
    for scale in (1.0, 2.0):
        area = (base * scale) ** 2
        for r in ratios:
            w = math.sqrt(area * r)
            h = math.sqrt(area / r)
            anchors.append((w, h))

    rng = random.Random(seed)
    # 兴趣点数量不能超过注意力图像素数；避免空采样。
    m = min(int(m), max(1, ah * aw))
    if m <= 0:
        return []

    # 在注意力图上随机采样兴趣点（论文 m=200），再套用所有锚框。
    # 注意：采样是在注意力图坐标 (xs,ys)，下面要换算成 image_size 坐标 (cx,cy)。
    ys = np.array([rng.randrange(ah) for _ in range(m)])
    xs = np.array([rng.randrange(aw) for _ in range(m)])
    cx = (xs + 0.5) / max(1, aw) * W
    cy = (ys + 0.5) / max(1, ah) * H

    candidates = []
    for x, y in zip(cx, cy):
        for w, h in anchors:
            # 以兴趣点为中心生成框，并裁剪到图像边界内。
            x1 = max(0.0, x - w / 2)
            y1 = max(0.0, y - h / 2)
            x2 = min(float(W), x + w / 2)
            y2 = min(float(H), y + h / 2)
            # 太小的框（贴边裁剪后）没有放大价值，跳过。
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            sc = mean_box(x1, y1, x2, y2)
            # 当前框平均注意力低于 theta_z：说明该区域不拥挤，不值得放大。
            if sc < float(theta_z):
                continue
            candidates.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                               "score": float(sc)})
    # 按平均注意力排序后做 IoU NMS（论文 IoU>0.5 删除低分框），最多保留 kz 个。
    candidates = _nms(candidates, iou_thr=iou_thr)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:max(0, int(kz))]


def _crop_and_resize(img: Image.Image, box: Dict, scale: float = 2.0):
    """按框裁剪图像，并放大 scale 倍。

    参数:
        img:   父图像（PIL RGB）
        box:   {"x1","y1","x2","y2"}，坐标属于 img
        scale: 放大倍数，论文取 2.0

    返回:
        (crop, (x1, y1))
        crop 为放大后的 PIL 子图；
        (x1, y1) 是实际裁剪框左上角（已做边界裁剪），
        用于把子图预测点映射回父图像坐标：
            parent_x = child_x / scale + x1
            parent_y = child_y / scale + y1
    """
    x1, y1, x2, y2 = (int(round(box["x1"])), int(round(box["y1"])),
                     int(round(box["x2"])), int(round(box["y2"])))
    # 防止框越界导致 PIL crop 报错；同时保证宽高至少 1 像素。
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, max(x1 + 1, x2)), min(img.height, max(y1 + 1, y2))
    crop = img.crop((x1, y1, x2, y2))
    # 论文：选中的子图统一放大 2 倍再送入 RAZ-Net。
    nw, nh = max(1, int(round(crop.width * scale))), \
        max(1, int(round(crop.height * scale)))
    crop = crop.resize((nw, nh), Image.BILINEAR)
    # 返回左上角偏移，供预测点从裁剪坐标映射回原图。
    return crop, (x1, y1)


def _preprocess(img: Image.Image, max_side: int = 1536):
    """把 PIL 图像缩放到长边 <= max_side，并转成 ImageNet 归一化 tensor。

    参数:
        img:      RGB PIL 图像
        max_side: 长边上限；0 表示不缩放。QNRF 原图很大，推理时通常限制
                  到 1536/2048 以控制显存/内存。

    返回:
        (resized_img, tensor, scale)
        resized_img: 缩放后的 PIL 图（坐标系统与后续框一致）
        tensor:      [3,H,W] float tensor，已归一化
        scale:       resized_img 相对原图的缩放比例
    """
    W, H = img.size
    scale = 1.0
    if max_side and max(W, H) > max_side:
        scale = max_side / max(W, H)
        img = img.resize((max(1, int(round(W * scale))),
                          max(1, int(round(H * scale)))), Image.BILINEAR)
    # 0~1 归一化后再减 ImageNet mean、除 std；与训练集预处理保持一致。
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / \
        np.array(IMAGENET_STD, dtype=np.float32)
    t = torch.from_numpy(x.transpose(2, 0, 1)).float()
    return img, t, scale


def generate_zoom_crops(model, items: Sequence[Tuple[Path, Path]],
                        out_dir: str | Path, device: torch.device, *,
                        max_side: int = 1536, m: int = 200,
                        theta_z: float = 0.01, kz: int = 10,
                        scale: float = 2.0, seed: int = 42,
                        allow_density_gt: bool = False,
                        density_threshold: float = 0.05,
                        log=print) -> Dict:
    """用训练好的 Main-Net 生成 RAZ-Net 的缩放子图训练集。

    流程:
        1. 对每张训练图做与推理相同的预处理（长边 <= max_side）；
        2. Main-Net 前向，取注意力图 sigmoid 后的结果；
        3. select_zoom_boxes 选出高注意力子图；
        4. 裁剪并放大 scale 倍，保存子图与“子图坐标系”下的点标注；
        5. 输出到 out_dir/Train/{Images,GT}，可直接作为 CrowdDataset 的输入。

    参数:
        model:      训练好的 Main-Net（eval 模式）
        items:      [(image_path, gt_path), ...] 训练集图像与点标注
        out_dir:    子图缓存根目录；会创建 Train/Images 与 Train/GT
        device:     推理设备
        max_side:   全图推理时长边上限（控制显存）
        m:随机兴趣点数量；论文默认 200。若大于注意力图像素数会被截断。
        theta_z:候选框平均注意力下限；论文默认 0.01。低于它的框直接丢弃。
        kz:最终最多保留多少个框；论文默认 10。
        iou_thr:NMS 的 IoU 阈值；论文默认 0.5。
        seed:随机种子；同一种子可复现兴趣点采样。
        allow_density_gt: GT 为密度图时是否允许转伪点
        density_threshold: 密度图转伪点的阈值
        log:        日志函数

    返回:
        meta dict：{"images": 有效图像数, "crops": 子图数, ...}
    """
    out = Path(out_dir)
    img_dir = out / "Train" / "Images"
    gt_dir = out / "Train" / "GT"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    n_images = 0
    n_crops = 0
    for idx, (img_path, gt_path) in enumerate(items):
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        # 预处理：proc 是缩放后的图，tensor 是 [3,H,W] 归一化输入。
        proc, tensor, _ = _preprocess(img, max_side=max_side)
        # 读取原始图像坐标系下的点标注。
        pts = load_points(gt_path, image_size=img.size,
                          allow_density=allow_density_gt,
                          density_threshold=density_threshold)
        if len(pts):
            # 点坐标同步缩放到 proc 坐标系，后续裁剪/保存都在 proc 上做。
            sx, sy = proc.width / max(1, img.width), proc.height / max(1, img.height)
            pts_proc = pts * np.array([sx, sy], dtype=np.float32)
        else:
            pts_proc = pts

        # Main-Net 在全图上的注意力图决定“哪些区域值得放大”。
        # attn 是 logits，sigmoid 后才是 0~1 的注意力分数。
        with torch.no_grad():
            out_main = model(tensor.unsqueeze(0).to(device))
        attn = torch.sigmoid(out_main["attn"])[0]
        # 返回的 boxes 坐标属于 proc（处理后图像）坐标系。
        boxes = select_zoom_boxes(
            attn,
            proc.size,
            m=m,
            theta_z=theta_z,
            kz=kz,
            seed=seed + idx
        )
        if not boxes:
            log(f"[zoom] {img_path.name}: 无候选区域，跳过")
            continue
        n_images += 1
        for j, box in enumerate(boxes):
            crop, (x1, y1) = _crop_and_resize(proc, box, scale=scale)
            if len(pts_proc):
                # 只保留落在当前框内的点；先减去框左上角，再乘放大倍数，
                # 得到“放大后子图”的局部坐标。
                sel = pts_proc.copy()
                inside = ((sel[:, 0] >= x1) & (sel[:, 0] < box["x2"]) &
                          (sel[:, 1] >= y1) & (sel[:, 1] < box["y2"]))
                sel = sel[inside]
                sel[:, 0] = (sel[:, 0] - x1) * scale
                sel[:, 1] = (sel[:, 1] - y1) * scale
            else:
                sel = pts_proc.reshape(0, 2)
            # 命名规则：原图名_z0/z1...，图像与同名 .txt 点标注一一对应。
            stem = f"{img_path.stem}_z{j}"
            crop.save(img_dir / f"{stem}.jpg", quality=95)
            (gt_dir / f"{stem}.txt").write_text(
                "\n".join(f"{float(x):.4f} {float(y):.4f}" for x, y in sel) +
                ("\n" if len(sel) else ""), encoding="utf-8")
            n_crops += 1
        if (idx + 1) % 50 == 0:
            log(f"[zoom] {idx + 1}/{len(items)} 图像，已生成 {n_crops} 子图")
    meta = {"images": n_images, "crops": n_crops, "max_side": max_side,
            "m": m, "theta_z": theta_z, "kz": kz, "scale": scale}
    (out / "meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8")
    log(f"[zoom] 完成: {n_images} 图像 -> {n_crops} 子图: {out}")
    return meta


def _predict_crop(model, crop: Image.Image, device: torch.device,
                  peak_threshold: float, max_side: int = 4096):
    """对单个已放大的子图做 RAZ-Net 推理。

    参数:
        model:          RAZ-Net（或其定位/注意力分支结构）
        crop:           已放大 scale 倍的子图
        device:         推理设备
        peak_threshold: 定位峰值阈值
        max_side:       子图长边上限（通常子图已不大，默认 4096 足够）

    返回:
        (pts, scores, attn)
        pts/scores 在 **当前 crop 坐标**（经过 max_side 缩放后）；
        attn 是原始注意力 logits，用于下一轮再选子图。
    """
    _, tensor, _ = _preprocess(crop, max_side=max_side)
    with torch.no_grad():
        out = model(tensor.unsqueeze(0).to(device))
    pts, scores = extract_peaks(out["loc"], threshold=peak_threshold)
    # 注意：这里的 pts 还没有除以放大倍数，也还没有加裁剪偏移；
    # 由 predict_with_zoom 负责映射回父图坐标。
    return pts, scores, out["attn"]


def predict_with_zoom(main_model, raz_model, image: Image.Image,
                      device: torch.device, *, max_side: int = 1536,
                      peak_threshold: float = 0.3, rounds: int = 1,
                      m: int = 200, theta_z: float = 0.01,
                      kz: int = 10, zoom_scale: float = 2.0,
                      seed: int = 42):
    """Main-Net 全图推理 + 可选 RAZ-Net 局部精修。

    参数:
        main_model:      Main-Net（必须输出 loc / density / attn）
        raz_model:       RAZ-Net；为 None 或 rounds<=0 时只跑 Main-Net
        image:           输入 RGB PIL 图像（原图）
        device:          推理设备
        max_side:        全图长边上限。所有返回坐标都在缩放后的
                         “处理后图像”坐标系中
        peak_threshold:  定位图峰值阈值
        rounds:          RAZ 循环次数：
                         0/1=不精修或只精修一轮；
                         >1=在子图内继续用 RAZ 的注意力选子图（最多 2 轮实用）
        m/theta_z/kz:    缩放框搜索参数，见 select_zoom_boxes
        zoom_scale:      子图放大倍数（论文 2.0）
        seed:            随机种子

    返回 dict:
        main_points/main_scores: Main-Net 原始定位点/分数
        points/scores:           合并去重后的最终点/分数
        image:                   缩放后的 PIL 图（与 points 坐标一致）
        density:                 Main-Net 密度图
        attn:                    Main-Net 注意力 logits
        boxes:                   （仅 RAZ 路径）选中的缩放框
    """
    # 先跑 Main-Net 得到基础定位点/密度/注意力；RAZ 只做局部精修。
    proc, tensor, _ = _preprocess(image, max_side=max_side)
    with torch.no_grad():
        out_main = main_model(tensor.unsqueeze(0).to(device))
    # Main-Net 的定位分支已经给出全图点；RAZ 的结果只作为补充。
    main_pts, main_scores = extract_peaks(out_main["loc"],
                                          threshold=peak_threshold)
    merged_pts, merged_scores = main_pts, main_scores
    if raz_model is None or rounds <= 0:
        return {"main_points": main_pts, "main_scores": main_scores,
                "points": merged_pts, "scores": merged_scores,
                "image": proc, "density": out_main["density"],
                "attn": out_main["attn"]}

    # 用 Main-Net 的注意力选择第一批高密度子图；框坐标属于 proc。
    boxes = select_zoom_boxes(out_main["attn"], proc.size, m=m,
                              theta_z=theta_z, kz=kz, seed=seed)
    all_pts = []
    all_scores = []
    for bi, box in enumerate(boxes):
        crop, (x1, y1) = _crop_and_resize(proc, box, scale=zoom_scale)
        pts, scores, attn = _predict_crop(raz_model, crop, device,
                                          peak_threshold)
        if len(pts):
            # 子图内坐标：先除以放大倍数，再加裁剪框左上角偏移。
            pts = pts / float(zoom_scale)
            pts[:, 0] += x1
            pts[:, 1] += y1
            all_pts.append(pts)
            all_scores.append(scores)
        # rounds>1：用 RAZ-Net 在当前 crop 上的注意力图继续找更小的子图。
        # m/kz 减半，避免第二轮产生过多裁剪。
        if rounds > 1:
            sub_boxes = select_zoom_boxes(attn, crop.size, m=max(50, m // 2),
                                          theta_z=theta_z, kz=max(1, kz // 2),
                                          seed=seed + bi + 1)
            for sb in sub_boxes:
                sub, (sx, sy) = _crop_and_resize(crop, sb, scale=zoom_scale)
                sp, ss, _ = _predict_crop(raz_model, sub, device, peak_threshold)
                if len(sp):
                    # 两重坐标映射：子子图 -> 当前 crop -> 处理后的原图。
                    sp = sp / float(zoom_scale)
                    sp[:, 0] += sx
                    sp[:, 1] += sy
                    # crop coords -> processed image coords
                    sp = sp / float(zoom_scale)
                    sp[:, 0] += x1
                    sp[:, 1] += y1
                    all_pts.append(sp)
                    all_scores.append(ss)
    if all_pts:
        # 合并多个子图的 RAZ 点数，再与 Main-Net 点数做距离去重，
        # 避免同一人头在 3 像素内被重复计数。
        zoom_pts = np.concatenate(all_pts, axis=0)
        zoom_scores = np.concatenate(all_scores, axis=0)
        merged_pts, merged_scores = merge_points(
            main_pts, main_scores, zoom_pts, zoom_scores, min_dist=3.0)
    return {"main_points": main_pts, "main_scores": main_scores,
            "points": merged_pts, "scores": merged_scores,
            "image": proc, "density": out_main["density"],
            "attn": out_main["attn"], "boxes": boxes}
