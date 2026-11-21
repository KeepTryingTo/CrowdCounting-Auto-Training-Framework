# -*- coding: utf-8 -*-
"""Local-Count-Guided Peak Point Detection (LCG-PPD, 论文 Algorithm 1)。

D2C 框架的第二阶段输出是「概率图 P」与「计数图 C」：
  - 概率图 P:    与原图同分辨率，每个像素表示“该位置属于目标的概率”，
                 每个目标对应一个概率峰；
  - 计数图 C:    CMR 输出，分辨率为原图的 1/rf（rf=32）；
                 每个 cell 的数值表示该 32x32 区域内的人数（局部计数）。

LCG-PPD 的作用：在概率图上找候选峰，但**不直接按概率阈值取点**（论文指出
那样误报很多），而是用计数图给出每个局部区域的“应取人数 k”，在区域内按
概率取 top-k，再通过**稠密采样 + 投票**筛掉不稳定候选：

  1. 概率图做 3x3 均值池化（平滑噪声）后，再接 3x3 最大池化，
     取“等于邻域最大值”的像素作为候选峰（允许平台，容差 1e-6）；
  2. 在计数图坐标系上，为每个 cell 生成一个局部区域（默认 64x64 像素，
     即 2x2 个 count cell）；若区域内计数和 <=1，则不断向外扩展成更大的
     方形区域，直到计数和 >1（论文规则：合并直到局部人数不小于 1，
     避免稀疏区域反复产生无效候选）；
  3. 对每个区域，按 round(局部计数) 作为 k，在区域内的候选峰里按概率值
     取 top-k 作为“真正例”，给这些像素各投一票；
  4. 由于区域以步长 1 稠密滑动（同一像素会被多个区域覆盖），用
     「得票数 / 被覆盖次数 >= vote_threshold（论文 0.5）」作为最终判定：
     一个点在大多数包含它的区域里都排进 top-k，才认为是真实目标点。

返回值包含二值点图（便于可视化/与 GT 对齐）与 (x, y) 坐标数组。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def lcg_ppd(probability, count_map, rf: int = 32,
            vote_threshold: float = 0.5,
            max_points: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """从概率图 + 计数图检测目标点（LCG-PPD）。

    参数
    ----
    probability : (H,W) / (1,H,W) / (1,1,H,W) 的 tensor 或 ndarray，
                  取值 [0,1]，与**原图同分辨率**（PMR 输出，通常已 clamp）。
                  多余的 batch/channel 维会被自动去掉（只取第 0 个）。
    count_map   : (h,w) / (1,h,w) / (1,1,h,w) 的 tensor 或 ndarray，
                  通常 h=H/rf、w=W/rf，每个 cell 是该 rf×rf 区域的局部人数。
                  多余的 batch/channel 维同样会被去掉。
    rf          : CMR 的下采样倍率（论文/本实现默认 32）。
                  用于把 count 坐标 (x,y) 映射回概率图/原图像素坐标
                  （x*rf ~ (x+i)*rf）。
    vote_threshold : 最终投票比例阈值，论文取 0.5：
                  votes / cover >= vote_threshold 时保留该点。
    max_points  : 输出点数上限（防止极端情况下点数爆炸；超出时按
                  “先出现先保留”截断，仅用于安全兜底）。

    返回
    ----
    dot    : (H,W) float32 二值图（0/1），1 表示该像素被判定为目标点。
    points : (N,2) float32 数组，按 (x, y) 排列（图像像素坐标，非归一化）。

    说明
    ----
    - 全函数在 @torch.no_grad() 下运行，纯推理用途，不参与反向传播；
    - 输入会自动搬到 CPU 上计算（该后处理是 numpy/torch CPU 实现）；
    - 概率图与计数图的空间尺寸不需要严格整除：最后的上采样若尺寸对不上
      会改用 cv2 最近邻缩放（见步骤 4）。
    """
    # ---------- 0) 统一成 CPU 上的 2D 张量/数组 ----------
    # 概率图：转 float32 + CPU，并去掉 batch/channel 维（(1,H,W) → (H,W)）
    if isinstance(probability, torch.Tensor):
        p = probability.detach().float().cpu()
    else:
        p = torch.from_numpy(np.asarray(probability, np.float32))
    while p.dim() > 2:
        p = p[0]
    H, W = p.shape

    # 计数图：同样转成 CPU 上的 numpy 2D 数组（h, w 为 count cell 网格大小）
    if isinstance(count_map, torch.Tensor):
        c = count_map.detach().float().cpu().numpy()
    else:
        c = np.asarray(count_map, np.float32)
    while c.ndim > 2:
        c = c[0]
    h, w = c.shape

    # ---------- 1) 候选峰：3x3 均值池化平滑 → 3x3 最大池化取局部极大 ----------
    # p4: (1,1,H,W)，加 batch/channel 维以适配 pytorch 池化算子；
    # padding=1 + stride=1 保证输出与原图同尺寸（边界零填充，边缘略有衰减）。
    p4 = p[None, None]
    avg = F.avg_pool2d(p4, 3, stride=1, padding=1)   # 3x3 邻域平均，抑制噪声
    mx = F.max_pool2d(avg, 3, stride=1, padding=1)   # 3x3 邻域最大值

    # cand: 与邻域最大值相等（容差 1e-6，兼容平坦/平台区域）且大于 0 的像素。
    # 这些像素就是“局部概率峰”，作为后续 top-k 选择的候选集合。
    cand = ((avg >= mx - 1e-6) & (avg > 1e-6))[0, 0]

    # ---------- 2) 局部区域生成（在 count cell 坐标系上滑动） ----------
    # 每个 cell 生成一个方形区域 (x, y, i, s)：
    #   (x, y)  区域左上角所在的 count cell 坐标；
    #   i       区域边长（单位：count cell），最小 2（对应 64x64 像素）；
    #   s       区域内的计数和（用浮点累加）。
    # 规则：从 2x2 个 cell 开始尝试，一旦计数和 >1 就停止扩展；
    #       若扩到最大仍 <=1，则取能取到的最大方形（s 可能为 0/很小）。
    regions = []
    # cover: 每个 count cell 被多少个区域覆盖（用于步骤 4 的投票归一化）
    cover = np.zeros((h, w), np.float32)
    # 注意 h-1 / w-1：保证区域至少能取 2x2 个 cell（避免越界）
    for y in range(max(0, h - 1)):
        for x in range(max(0, w - 1)):
            chosen = None
            # i 从 2 开始逐步扩大方形区域，寻找第一个计数和 >1 的尺寸
            for i in range(2, min(h - y, w - x) + 1):
                s = float(c[y:y + i, x:x + i].sum())
                if s > 1.0:
                    chosen = (i, s)
                    break
            if chosen is None:
                # 扩展到边界仍不足 1 人：使用最大可用方形（稀疏/空旷区域）
                i = min(h - y, w - x)
                s = float(c[y:y + i, x:x + i].sum())
                chosen = (i, s)
            i, s = chosen
            regions.append((x, y, i, s))
            # 该方形覆盖到的所有 cell 覆盖计数 +1（稠密滑动，互相重叠）
            cover[y:y + i, x:x + i] += 1.0

    # ---------- 3) 每个区域内按 k=round(局部计数) 选 top-k 候选峰并投票 ----------
    # votes: 与原图同分辨率的得票图；每被某个区域选中一次 +1。
    votes = np.zeros((H, W), np.float32)
    cand_np = cand.numpy()          # (H,W) bool：候选峰掩码
    avg_np = avg[0, 0].numpy()      # (H,W) 平滑后的概率值，用于排序
    for x, y, i, s in regions:
        # k: 该区域应取出的点数（四舍五入局部人数）；<=0 的区域直接跳过
        k = int(round(s))
        if k <= 0:
            continue
        # 把 count cell 区域 (x, y, i) 映射到原图像素区域 [x1,x2) x [y1,y2)，
        # 并裁剪到图像边界（最后一个 cell 可能超出 1/rf 个像素对齐）。
        x1, y1 = x * rf, y * rf
        x2, y2 = min(W, (x + i) * rf), min(H, (y + i) * rf)
        if x2 <= x1 or y2 <= y1:
            continue
        # 取出该像素区域内所有候选峰坐标（ys, xs 为相对窗口的局部坐标）
        ys, xs = np.nonzero(cand_np[y1:y2, x1:x2])
        if len(xs) == 0:
            continue
        # 候选峰按概率值（avg）从大到小排序，取前 k 个作为真正例
        vals = avg_np[y1:y2, x1:x2][ys, xs]
        order = np.argsort(-vals)[:max(0, k)]
        for j in order:
            # 局部坐标加窗口偏移 → 原图坐标；该像素得票 +1
            votes[y1 + ys[j], x1 + xs[j]] += 1.0

    # ---------- 4) 投票归一化 + 阈值判定 ----------
    # cover 是 count 分辨率 (h,w)，先上采样到原图分辨率 (H,W)：
    #   np.repeat 按 rf 放大（要求 H=h*rf、W=w*rf 严格成立）；
    #   若不整除，退化为 cv2 最近邻缩放（保持覆盖计数不被插值污染）。
    cover_full = np.repeat(np.repeat(cover, rf, axis=0), rf, axis=1)
    if cover_full.shape != (H, W):
        import cv2
        cover_full = cv2.resize(cover, (W, H),
                                interpolation=cv2.INTER_NEAREST)
    # 避免除 0（未被任何区域覆盖的像素覆盖次数记为 1）
    cover_full = np.maximum(cover_full, 1.0)
    # ratio = 该点在包含它的区域中被选为 top-k 的比例；
    # 比例 >= 阈值（论文 0.5）→ 认为是稳定的真实目标点
    ratio = votes / cover_full
    dot = (ratio >= float(vote_threshold)).astype(np.float32)

    # 输出坐标：按行优先顺序取所有置 1 的像素（x=列, y=行）
    ys, xs = np.nonzero(dot)
    if len(xs) > max_points:
        # 安全兜底：点数过多时仅截断输出坐标，dot 图本身保持不变
        xs, ys = xs[:max_points], ys[:max_points]
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    return dot, points
