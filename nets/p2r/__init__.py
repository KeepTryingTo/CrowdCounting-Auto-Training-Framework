# -*- coding: utf-8 -*-
"""P2RLoss 点预测模型（CVPR 2025 Point-to-Region Loss）

vendor 来源：P2RLoss-main（"Point-to-Region Loss for Semi-Supervised
Point-Based Crowd Counting"）。模型 / 损失 / 增强与源实现保持一致，
仅做如下适配：
- VGG16_BN 构造参数改为 pretrained（源实现传 yacs config 但未使用）
- torchvision 新旧权重 API 兼容（weights 枚举 / pretrained 布尔）
- NormalSample 增加 augment 与 resize_factor 参数（对接框架
  「数据增强」开关与「增强强度」）
"""
from .vgg16bn import VGG16_BN
from .p2rloss import P2RLoss
from .utils import NormalSample


def build_p2r_model(pretrained=True):
    """P2R 点预测网（VGG16-BN 编码 + FPN 融合 + 解码，输出 /4 分数图）

    forward 输出 [N,1,H/4,W/4] 的 logits：logit>0 的位置即预测人头点，
    人数 = (output > 0) 的元素个数（与源实现 validate 口径一致）
    """
    return VGG16_BN(pretrained=pretrained)
