# -*- coding: utf-8 -*-
"""LLB_Count 的局部感知 / 边界感知损失（移植自 LLB_Count/losses/）

- LocalPerceptionLoss：按 grid×grid 切块，块内 L1(sum) × 块均密度 加权
- BoundAwareLoss：GT 密度图边界 mask 上的加权 MSE；边界提取支持
  sobel（纯 torch，默认）与 canny（cv2，需 opencv，逐样本 CPU 计算）
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalPerceptionLoss(nn.Module):
    """局部感知损失（LLB 核心）：稠密区域块权重更大"""

    def __init__(self, grid_size=8, lambda_local=1.0, device=None):
        super().__init__()
        self.grid_size = max(1, int(grid_size))
        self.lambda_local = float(lambda_local)
        self.local_loss = nn.L1Loss(reduction="sum")

    def forward(self, pred_map, gt_map):
        if gt_map.dim() == 3:
            gt_map = gt_map.unsqueeze(1)
        if gt_map.dim() == 2:
            gt_map = gt_map.unsqueeze(0).unsqueeze(0)
        if pred_map.dim() == 3:
            pred_map = pred_map.unsqueeze(1)
        if pred_map.dim() == 2:
            pred_map = pred_map.unsqueeze(0).unsqueeze(0)
        h, w = gt_map.shape[-2:]
        gh, gw = h / self.grid_size, w / self.grid_size
        local_loss = pred_map.new_zeros(())
        for i in range(self.grid_size):
            gis = int(max(min(h - gh, i * gh), 0))
            gie = int(min(h, (i + 1) * gh))
            if gie <= gis:
                continue
            for j in range(self.grid_size):
                gjs = int(max(min(w - gw, j * gw), 0))
                gje = int(min(w, (j + 1) * gw))
                if gje <= gjs:
                    continue
                pred_patch = pred_map[..., gis:gie, gjs:gje]
                gt_patch = gt_map[..., gis:gie, gjs:gje]
                local_loss = local_loss + gt_patch.mean() * \
                    self.local_loss(pred_patch, gt_patch)
        return local_loss / (self.grid_size * self.grid_size) * \
            self.lambda_local


class BoundAwareLoss(nn.Module):
    """边界感知损失：仅在 GT 密度图边界处计算加权 MSE

    extractor: "sobel"（默认，纯 torch，可反传前向；GT 边界由梯度阈值给出）
               "canny"（cv2.Canny(0, 20)，与官方实现一致；逐样本 CPU）
    """

    def __init__(self, lambda_boundary=2.0, gradient_threshold=0.05,
                 extractor="sobel", device=None):
        super().__init__()
        self.lambda_boundary = float(lambda_boundary)
        self.gradient_threshold = float(gradient_threshold)
        self.extractor = extractor
        self.bound_aware_mseloss = nn.MSELoss(reduction="sum")
        self.sobel_x = nn.Parameter(
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                         dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False)
        self.sobel_y = nn.Parameter(
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                         dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False)

    def _sobel_mask(self, gt_map):
        w_x = self.sobel_x.to(gt_map.device)
        w_y = self.sobel_y.to(gt_map.device)
        grad_x = F.conv2d(gt_map, w_x, padding=1)
        grad_y = F.conv2d(gt_map, w_y, padding=1)
        mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return (mag > self.gradient_threshold).float()

    def _canny_mask(self, gt_map):
        import cv2
        masks = []
        for k in range(gt_map.shape[0]):
            g = gt_map[k, 0].detach().cpu().numpy()
            g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            m = cv2.Canny(g8, 0, 20).astype(np.float32)
            m = m / (m.max() - m.min() + 1e-6)
            masks.append(torch.from_numpy(m))
        return torch.stack(masks).unsqueeze(1).to(gt_map.device)

    def forward(self, pred_map, gt_map):
        if gt_map.dim() == 2:
            gt_map = gt_map.unsqueeze(0).unsqueeze(0)
        elif gt_map.dim() == 3:
            gt_map = gt_map.unsqueeze(1)
        if pred_map.dim() == 2:
            pred_map = pred_map.unsqueeze(0).unsqueeze(0)
        elif pred_map.dim() == 3:
            pred_map = pred_map.unsqueeze(1)
        mask = self._canny_mask(gt_map) if self.extractor == "canny" \
            else self._sobel_mask(gt_map)
        loss = self.bound_aware_mseloss(pred_map * mask, gt_map * mask)
        return loss / (mask.sum() + 1e-6) * self.lambda_boundary
