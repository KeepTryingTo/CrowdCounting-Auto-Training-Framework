# -*- coding: utf-8 -*-
"""PET (Point-Query Quadtree, ICCV 2023) vendored for the crowd-counting framework.

模型与损失入口：build_pet(args) -> (model, criterion)
- 模型输入：NestedTensor（见 nets.pet.util.misc）
- 训练：model(samples, epoch=..., train=True, criterion=..., targets=...)
- 推理：model(samples, test=True) -> pred_logits / pred_points / split_map_raw
"""
from .models import build_pet

__all__ = ["build_pet"]
