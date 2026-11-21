# -*- coding: utf-8 -*-
"""D2C 训练所需的四类真值图（概率图 / 密度图 / 计数图 / 点图）。

论文中的两种中间表示（给定点标注 {(x_i, y_i)}_{i=1..N}）：

    概率图 P(x,y) = max_i [ δ(x-x_i, y-y_i) * G_{σ_i}(x,y) ]        # 峰值保留
    密度图 D(x,y) = Σ_i [ δ(x-x_i, y-y_i) * G_{σ_i}(x,y)
                          / Σ_{(x,y)} G_{σ_i}(x,y) ]               # 质量和为 1

- 概率图 P：每个高斯核**按峰值归一化为 1**，再逐像素取 max。多个个体即使
  重叠也各自保留一个峰（不会被相加合并），因此既能做定位又能做计数支撑，
  作为第一阶段 PMR 的回归目标；
- 密度图 D：每个高斯核**按核内总和归一化为 1**，再逐像素相加，全图积分
  ≈ 人数，用于积分计数；
- 计数图 C：把密度图按 32×32（rf）做 **sum-pooling**，得到 CMR 的回归目标
  （每个 cell 表示该区域局部人数）；
- 点图 dot：把标注点四舍五入到整像素并置 1，用于定位监督/可视化/评估。

本模块只负责“由点标注生成上述四种真值图”，不涉及网络与训练流程。
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.spatial import cKDTree   # 自适应 sigma 需要 k 近邻查询
except Exception:  # pragma: no cover - scipy 缺失时退化为固定 sigma
    cKDTree = None


def _points_array(points) -> np.ndarray:
    """把任意点标注容器整理成 (N,2) 的 float32 数组。

    兼容 list / tuple / ndarray / tensor（tensor 会经 np.asarray 自动转），
    空输入返回 (0,2)，避免后续 zip/循环时的形状歧义。
    注意：输入必须是“成对的坐标”，形如 (N,2) 或长度为 2N 的扁平序列。
    """
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 2)
    return arr.reshape(-1, 2)


def _sigmas(points: np.ndarray, height: int, width: int,
            mode: str = "adaptive", sigma: float = 15.0) -> np.ndarray:
    """为每个点计算高斯核的标准差 σ_i。

    mode:
      - "fixed"    : 所有点使用同一个固定 σ（论文对稀疏数据集 SHA-B /
                     WorldExpo 使用 σ=15）；
      - "adaptive" : 几何自适应（拥挤数据集 SHA-A / QNRF / UCF-CC-50 /
                     JHU / NWPU）：σ_i 与该点邻近人群的间距相关——人越密，
                     σ 越小，避免相邻个体的核糊成一片。
    自适应实现（与 D2C 官方一致）：取每个点最近的 3 个邻居平均距离 d̄，
    σ_i = 0.3 * d̄（代码里用 (d1+d2+d3)*0.1 等价于 0.3*均值）；
    点数不足 4 时用可用邻居的均值*0.3；单点特殊处理；σ 下限为 1 像素。
    """
    # 固定 σ；或点数 <=1 时无需近邻
    if mode == "fixed" or len(points) <= 1:
        if len(points) == 1 and mode == "adaptive":
            # 官方对单点使用图像平均尺寸的一半再除 2
            return np.array([max(2.0, (height + width) / 8.0)], np.float32)
        return np.full(len(points), float(sigma), np.float32)
    # scipy 不可用：退化为固定 σ，保证流程可跑
    if cKDTree is None:
        return np.full(len(points), float(sigma), np.float32)
    tree = cKDTree(points)
    # 查询最近的 k 个点（含自身，d0=0）；k 最多 4，避免小点集越界
    k = min(4, len(points))
    dists, _ = tree.query(points, k=k)
    if dists.ndim == 1:
        # k=1 时 query 返回一维数组，补一列便于统一处理
        dists = dists[:, None]
    if dists.shape[1] >= 4:
        # 标准情形：d1/d2/d3 为最近三个邻居的距离（d0 是自身 0）
        s = (dists[:, 1] + dists[:, 2] + dists[:, 3]) * 0.1
    else:
        # 只有 1~2 个邻居：可用邻居平均距离 * 0.3
        s = dists[:, 1:].mean(axis=1) * 0.3
    # σ 下限 1 像素，防止 0σ 导致除零/退化成 δ 函数
    return np.maximum(1.0, s).astype(np.float32)


def gaussian_maps(points, height: int, width: int,
                  sigma_mode: str = "adaptive",
                  sigma: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """由点标注生成**原图分辨率**的概率图与密度图。

    返回:
      prob    : (height, width) float32，取值 [0,1]，PMR 的回归目标；
      density : (height, width) float32，全图求和 ≈ 点数（边界截断会略少）。

    实现要点：
      - 逐点生成局部窗口内的可分离高斯核，窗口半径 r = ceil(3σ)
        （覆盖约 99.7% 的核质量，兼顾精度与速度）；
      - 概率图：核按**峰值**归一化（g / g.max()），再与已有值逐像素取
        maximum —— 这样重叠的两个个体不会相加成一个峰，而是保留双峰；
      - 密度图：核按**总和**归一化（g / g.sum()），再累加 —— 每个核贡献
        1 个人头，全图积分即人数；
      - 落在图像外的点被跳过；靠近边界的核会被裁剪，密度图总和会略小于
        真实人数（概率图不受影响，因为峰值在核中心）。
    """
    pts = _points_array(points)
    prob = np.zeros((height, width), np.float32)
    density = np.zeros((height, width), np.float32)
    # 无点：返回全零图（与有标注样本保持相同形状，便于批处理）
    if len(pts) == 0:
        return prob, density
    sigmas = _sigmas(pts, height, width, sigma_mode, sigma)
    for (x, y), s in zip(pts, sigmas):
        # 点坐标取整到像素（标注一般是浮点；取整保证核中心落在网格上）
        x, y = int(round(float(x))), int(round(float(y)))
        # 越界点直接忽略（裁剪/缩放后的标注可能落到图像外）
        if not (0 <= x < width and 0 <= y < height):
            continue
        # 局部窗口：半径 3σ，并裁剪到图像边界，避免大核在图上逐点循环
        r = max(1, int(math.ceil(3.0 * float(s))))
        x0, x1 = max(0, x - r), min(width, x + r + 1)
        y0, y1 = max(0, y - r), min(height, y + r + 1)
        # 相对核中心的坐标偏移（可分离构造：x 方向一维 + y 方向一维）
        xs = np.arange(x0, x1, dtype=np.float32) - x
        ys = np.arange(y0, y1, dtype=np.float32) - y
        g = np.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) /
                   (2.0 * float(s) ** 2))
        if g.size == 0:
            continue
        # 概率图：每个核峰值归一化为 1，再做逐点 max（保留双峰）
        peak = float(g.max())
        if peak > 1e-9:
            prob[y0:y1, x0:x1] = np.maximum(
                prob[y0:y1, x0:x1], (g / peak).astype(np.float32))
        # 密度图：每个核质量和归一化为 1，保证积分≈人数
        mass = float(g.sum())
        if mass > 1e-9:
            density[y0:y1, x0:x1] += (g / mass).astype(np.float32)
    return prob, density


def dot_map(points, height: int, width: int) -> np.ndarray:
    """把点标注画成 (height, width) 的二值点图（背景 0，点像素 1）。

    坐标四舍五入到整像素；越界点忽略；同一像素的重复点仍为 1。
    主要用于定位监督、可视化以及 AP/AR/F1、GAME 等定位指标的对齐。
    """
    dots = np.zeros((height, width), np.float32)
    for x, y in _points_array(points):
        x, y = int(round(float(x))), int(round(float(y)))
        if 0 <= x < width and 0 <= y < height:
            dots[y, x] = 1.0
    return dots


def count_map_from_density(density: np.ndarray, rf: int = 32) -> np.ndarray:
    """密度图 → 计数图：按 rf×rf 做 sum-pooling（每个 cell = 局部人数）。

    实现说明：
      - 先把 (h,w) 右侧/下侧零填充到 rf 的整数倍，保证池化不丢尾部像素；
        零填充不会改变任何 cell 的人数总和；
      - 用 avg_pool2d(kernel=rf, stride=rf) * rf² 等价于 sum-pooling
        （PyTorch 无原生 sum-pool，用均值乘回面积实现）；
      - 输出形状为 (ceil(h/rf), ceil(w/rf))，即 CMR 的目标分辨率。
    """
    h, w = density.shape
    h1 = int(math.ceil(h / rf) * rf)
    w1 = int(math.ceil(w / rf) * rf)
    if (h1, w1) != (h, w):
        density = np.pad(density, ((0, h1 - h), (0, w1 - w)))
    t = torch.from_numpy(density.astype(np.float32))[None, None]
    t = F.avg_pool2d(t, kernel_size=rf, stride=rf) * (rf * rf)
    return t[0, 0].numpy()


def make_targets(points, height: int, width: int,
                 sigma_mode: str = "adaptive", sigma: float = 15.0,
                 rf: int = 32):
    """一站式生成 (概率图, 密度图, 计数图, 点图) 四种真值。

    参数:
      points     : 点标注 (N,2)，(x, y) 像素坐标；
      height/width : 目标图尺寸（通常为网络输入/裁剪尺寸）；
      sigma_mode : "adaptive"（拥挤数据集，默认）或 "fixed"（稀疏数据集）；
      sigma      : fixed 模式下的固定 σ（默认 15）；
      rf         : 计数图的下采样倍率（CMR 输出步长，默认 32）。

    返回:
      prob    : (H,W)        概率图，PMR 目标；
      density : (H,W)        密度图，用于积分计数；
      count   : (ceil(H/rf), ceil(W/rf)) 计数图，CMR 目标；
      dot     : (H,W)        点图，定位监督/评估。
    """
    prob, density = gaussian_maps(points, height, width, sigma_mode, sigma)
    dot = dot_map(points, height, width)
    count = count_map_from_density(density, rf)
    return prob, density, count, dot
