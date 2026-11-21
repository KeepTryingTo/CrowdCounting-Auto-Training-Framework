# -*- coding: utf-8 -*-
"""CLTR (Crowd Localization TRansformer, ECCV 2022) vendored for the framework.

入口：build_model(args) -> (model, criterion, postprocessors)
- 模型输入：tensor [B,3,H,W] 或 NestedTensor
- 输出：pred_logits [B,Q,2]、pred_points [B,Q,channel_point]（归一化）
- 类别：point / no-object（num_classes=2）
"""
from .Networks.CDETR import build_model

__all__ = ["build_model"]
