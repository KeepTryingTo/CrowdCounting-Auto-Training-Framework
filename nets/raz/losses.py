# -*- coding: utf-8 -*-
"""Paper losses (Eq.1-3) and stage helpers."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def density_loss(pred: torch.Tensor, target: torch.Tensor,
                 factor: float = 8.0) -> torch.Tensor:
    """论文 Eq.1：1/8 密度图上的 RMSE，再乘 8 补偿分辨率。"""
    mse = F.mse_loss(pred, target, reduction="mean")
    return float(factor) * torch.sqrt(mse + 1e-12)


def localization_loss(logits: torch.Tensor, target: torch.Tensor,
                      gamma: float = 100.0) -> torch.Tensor:
    """论文 Eq.3：正样本（头心十字邻域）损失乘 gamma=100。

    定位图极度稀疏，普通 BCE 会被海量负样本淹没；这里通过加权让
    少量正像素对梯度有足够贡献。
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = 1.0 + (float(gamma) - 1.0) * target
    return (bce * weight).mean()


def attention_loss(logits: torch.Tensor, target: torch.Tensor,
                   pos_weight: float = 1.0) -> torch.Tensor:
    """BCE for the continuous zoom-attention target."""
    weight = 1.0 + (float(pos_weight) - 1.0) * target
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weight).mean()


def stage_loss(stage: str, outputs: dict, batch: dict,
               gamma: float = 100.0, density_factor: float = 8.0,
               attn_pos_weight: float = 1.0):
    """按论文的“顺序训练”只返回当前阶段该优化的损失。

    density/localization/attention 分别只训练对应分支；
    raz 同时训练 RAZ-Net 的定位 + 注意力两个分支。
    """
    stage = stage.lower()
    if stage == "density":
        return density_loss(outputs["density"], batch["density"],
                            factor=density_factor)
    if stage in ("localization", "loc"):
        return localization_loss(outputs["loc"], batch["loc"], gamma=gamma)
    if stage in ("attention", "attn", "proposal"):
        return attention_loss(outputs["attn"], batch["attn"],
                              pos_weight=attn_pos_weight)
    if stage == "raz":
        return (localization_loss(outputs["loc"], batch["loc"], gamma=gamma) +
                attention_loss(outputs["attn"], batch["attn"],
                               pos_weight=attn_pos_weight))
    raise ValueError(f"未知训练阶段: {stage}")
