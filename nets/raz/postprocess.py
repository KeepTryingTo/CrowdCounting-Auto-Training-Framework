# -*- coding: utf-8 -*-
"""Localization peak extraction and count fusion helpers."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_peaks(loc_logits: torch.Tensor, threshold: float = 0.3,
                  avg_pool: int = 3, max_pool: int = 3,
                  max_points: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Paper post-processing: 3x3 average pooling + 3x3 local-max NMS.

    Returns ``(points_xy, scores)`` for the first batch element.
    """
    # 将定位输出缩放到0-1之间
    # 先转概率图，再按论文做“平均池化增强峰值 + 3x3 局部极大值 NMS”。
    prob = torch.sigmoid(loc_logits)
    # 平均池化
    if avg_pool > 1:
        prob = F.avg_pool2d(prob, kernel_size=avg_pool, stride=1,
                            padding=avg_pool // 2)
    pooled = F.max_pool2d(prob, kernel_size=max_pool, stride=1,
                          padding=max_pool // 2)
    mask = (prob >= pooled - 1e-6) & (prob > float(threshold))
    # 响应值最大的位置作为预测点
    ys, xs = torch.nonzero(mask[0, 0], as_tuple=True)
    if ys.numel() == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
    scores = prob[0, 0, ys, xs]
    order = torch.argsort(scores, descending=True)[:max_points]
    ys, xs, scores = ys[order], xs[order], scores[order]
    pts = torch.stack([xs, ys], dim=1).float().cpu().numpy()
    return pts.astype(np.float32), scores.cpu().numpy().astype(np.float32)


def resize_density_to(density: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if density.shape[-2:] == size:
        return density
    return F.interpolate(density, size=size, mode="bilinear",
                         align_corners=False)


def grid_counts(density: torch.Tensor, points: np.ndarray,
                height: int, width: int, grid: int = 4):
    """Return per-cell density/localization counts as CPU float arrays."""
    g = max(1, int(grid))
    if density.dim() == 2:
        density = density[None, None]
    elif density.dim() == 3:
        density = density[None]
    # 将预测密度图大小缩放到指定尺寸
    dens = resize_density_to(density, (height, width))[0, 0]
    # cden/cloc 分别为每个 4x4 格子的密度积分与定位点数。
    cden = torch.zeros((g, g), dtype=torch.float32, device=dens.device)
    cloc = torch.zeros((g, g), dtype=torch.float32, device=dens.device)
    # 按照网格大小划分得到对应的patch大小
    hstep = height / g
    wstep = width / g

    ys = torch.arange(height, device=dens.device, dtype=torch.float32)
    xs = torch.arange(width, device=dens.device, dtype=torch.float32)
    gy = torch.clamp((ys / hstep).long(), 0, g - 1)
    gx = torch.clamp((xs / wstep).long(), 0, g - 1)

    for yy in range(g):
        # 当前网格内的密度图
        row = dens[gy == yy]
        if row.numel():
            # 当前网格大小对应的人群计数值
            cden[yy] = torch.stack(
                [row[:, gx == xx].sum() for xx in range(g)]
            )
    if len(points):
        p = np.asarray(points, dtype=np.float32)
        # 将坐标缩放到网格大小内
        ix = np.clip((p[:, 0] / max(1.0, wstep)).astype(int), 0, g - 1)
        iy = np.clip((p[:, 1] / max(1.0, hstep)).astype(int), 0, g - 1)
        # 计算对应网格区域的人群计数
        for xx, yy in zip(ix, iy):
            cloc[yy, xx] += 1.0
    return cden.cpu().numpy(), cloc.cpu().numpy()


def fuse_density_and_points(density: torch.Tensor, points: np.ndarray,
                            scores: np.ndarray, height: int, width: int,
                            grid: int = 4, theta_f: float = 5.0):
    """论文式 4x4 场景自适应融合。

    高密度格子（平均计数 >= θ_f）信任密度分支；低密度格子取
    密度与定位两路平均。原因是定位分支在高密度区域更容易漏检。
    """
    cden, cloc = grid_counts(density, points, height, width, grid)
    use_den = (cden + cloc) / 2.0 >= float(theta_f)
    fused = np.where(use_den, cden, (cden + cloc) / 2.0)
    return (float(fused.sum()), float(cden.sum()), float(cloc.sum()))


def merge_points(base_points: np.ndarray, base_scores: np.ndarray,
                 new_points: np.ndarray, new_scores: np.ndarray,
                 min_dist: float = 3.0):
    """Union zoom detections with Main-Net detections (local de-duplication)."""
    if len(base_points) == 0:
        return new_points, new_scores
    if len(new_points) == 0:
        return base_points, base_scores
    pts = [p for p in base_points]
    scs = [float(s) for s in base_scores]
    for p, s in zip(new_points, new_scores):
        if all(float(np.linalg.norm(p - q)) >= float(min_dist) for q in pts):
            pts.append(p)
            scs.append(float(s))
    return (np.asarray(pts, dtype=np.float32),
            np.asarray(scs, dtype=np.float32))
