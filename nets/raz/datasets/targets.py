# -*- coding: utf-8 -*-
"""Ground-truth construction for density / localization / zoom attention."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def _as_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 2)
    return arr.reshape(-1, 2)


def gaussian_density(points, height: int, width: int, beta: float = 0.1,
                     k: int = 4) -> np.ndarray:
    """Geometric-adaptive Gaussian density (Eq.1 in the RAZ paper).

    Each kernel is normalised to mass 1; boundary kernels are re-normalised
    after clipping so that ``sum(density) == len(points)`` (for in-bounds
    points).
    """
    pts = _as_points(points)
    density = np.zeros((height, width), dtype=np.float32)
    if len(pts) == 0:
        return density

    # 论文式几何自适应 sigma：取每个点最近 k 个邻居的平均距离。
    # 稀疏区域 sigma 大、密集区域 sigma 小。
    if cKDTree is not None and len(pts) > 1:
        tree = cKDTree(pts)
        kk = min(max(2, int(k) + 1), len(pts))
        dists, _ = tree.query(pts, k=kk)
        if dists.ndim == 1:
            dists = dists[:, None]
        nn = dists[:, 1:] if dists.shape[1] > 1 else dists[:, :1]
        sigma = beta * nn.mean(axis=1)
    else:
        sigma = np.full(len(pts), max(2.0, beta * min(height, width) / 10.0),
                        dtype=np.float32)
    sigma = np.maximum(1.0, sigma.astype(np.float32))

    for (x, y), s in zip(pts, sigma):
        x, y = int(round(float(x))), int(round(float(y)))
        if not (0 <= x < width and 0 <= y < height):
            continue
        r = max(1, int(math.ceil(3.0 * float(s))))
        x0, x1 = max(0, x - r), min(width, x + r + 1)
        y0, y1 = max(0, y - r), min(height, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1, dtype=np.float32) - x
        ys = np.arange(y0, y1, dtype=np.float32) - y
        g = np.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) /
                   (2.0 * float(s) ** 2))
        # 边界处核被裁剪后要重新归一化，保证每个点总质量为 1，
        # 这样整张密度图求和约等于人数。
        total = float(g.sum())
        if total > 0:
            density[y0:y1, x0:x1] += g / total
    return density


def density_at_stride(points, height: int, width: int, stride: int = 8,
                      beta: float = 0.1, k: int = 4) -> np.ndarray:
    """Density target at 1/stride resolution, preserving total mass."""
    # 先在全分辨率生成密度图，再平均池化到 1/stride；
    # 乘 stride^2 抵消均值池化造成的缩放，保持总人数不变。
    full = gaussian_density(points, height, width, beta=beta, k=k)
    t = torch.from_numpy(full).float()[None, None]
    t = F.avg_pool2d(t, kernel_size=stride, stride=stride) * (stride * stride)
    return t[0, 0].numpy()


def localization_target(points, height: int, width: int) -> np.ndarray:
    """Cross-kernel binary target (Eq.2)."""
    # 每个头心 + 上下左右邻域置 1（即 3x3 十字核），
    # 用 BCE 学习极度稀疏的定位概率图。
    pts = _as_points(points)
    target = np.zeros((height, width), dtype=np.float32)
    for x, y in pts:
        x, y = int(round(float(x))), int(round(float(y)))
        if not (0 <= x < width and 0 <= y < height):
            continue
        target[y, x] = 1.0
        if x > 0:
            target[y, x - 1] = 1.0
        if x + 1 < width:
            target[y, x + 1] = 1.0
        if y > 0:
            target[y - 1, x] = 1.0
        if y + 1 < height:
            target[y + 1, x] = 1.0
    return target


def attention_target(points, height: int, width: int, stride: int = 4,
                     sigma_z: float = 3.0) -> np.ndarray:
    """Crowd-density attention target.

    ``v_i = exp(-mean_distance_to_3_nearest_heads^2 / (2*sigma_z^2))``.
    The map is computed on a coarser grid (default stride 4) for efficiency
    and then upsampled to the requested size; this is a practical
    approximation of the per-pixel k-d tree target in the paper.
    """
    # 逐像素 k-d 树在 QNRF 这类超大图上非常慢，因此默认在 stride 网格上
    # 计算 v_i，再双线性上采样；stride=1 即论文原始逐像素版本。
    stride = max(1, int(stride))
    ht = max(1, int(math.ceil(height / stride)))
    wt = max(1, int(math.ceil(width / stride)))
    pts = _as_points(points)
    if len(pts) == 0:
        return np.zeros((height, width), dtype=np.float32)

    if cKDTree is None:
        # Fallback: nearest three distances via broadcasting (small grids).
        ys, xs = np.mgrid[0:ht, 0:wt]
        coords = np.stack([xs.ravel() * stride, ys.ravel() * stride], 1).astype(np.float32)
        d2 = ((coords[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        kk = min(3, len(pts))
        part = np.partition(d2, kk - 1, axis=1)[:, :kk]
        mean_d = np.sqrt(part).mean(1)
    else:
        ys, xs = np.mgrid[0:ht, 0:wt]
        coords = np.stack([xs.ravel() * stride, ys.ravel() * stride], 1).astype(np.float32)
        tree = cKDTree(pts)
        kk = min(3, len(pts))
        dists, _ = tree.query(coords, k=kk)
        if dists.ndim == 1:
            dists = dists[:, None]
        mean_d = dists.mean(axis=1)
    v = np.exp(-(mean_d ** 2) / (2.0 * float(sigma_z) ** 2))
    v = v.reshape(ht, wt).astype(np.float32)
    if (ht, wt) != (height, width):
        t = torch.from_numpy(v)[None, None]
        t = F.interpolate(t, size=(height, width), mode="bilinear",
                          align_corners=False)
        v = t[0, 0].numpy()
    return np.clip(v, 0.0, 1.0)


def make_targets(points, height: int, width: int, *, density_stride: int = 8,
                 attn_stride: int = 4, beta: float = 0.1, k: int = 4,
                 sigma_z: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一次性生成三个分支的监督：
    密度图（1/8）+ 定位图（全分辨率）+ 注意力图（全分辨率）。"""
    dens = density_at_stride(points, height, width, density_stride, beta, k)
    loc = localization_target(points, height, width)
    attn = attention_target(points, height, width, attn_stride, sigma_z)
    return dens, loc, attn
