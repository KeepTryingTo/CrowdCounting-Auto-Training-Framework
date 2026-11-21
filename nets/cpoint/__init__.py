# -*- coding: utf-8 -*-
"""Consistent-Point 点预测模型（arXiv:2503.12441 半监督人群计数）

vendor 自本仓库 ../Point-based Crowd Counting Algorithm/Consistent-Point
（独立复现实现，官方代码未开源）。结构：
- p2pnet.py / matcher.py / backbone.py / vgg_.py：P2PNet 基座
  （输出额外携带 anchor_points，供 PA 邻居选择使用）
- pa.py / iuc.py / ema.py：Consistent-Point 三件套——
  PA 位置聚合（K=4 最近辅助提议均值）、IUC 不确定性加权
  （w=(c-0.5)/0.5）、Mean-Teacher EMA（decay=0.999）
- SetCriterion_Crowd.forward_unlabeled：无标注分支（伪点 + IUC 加权 CE，
  按 M^U 归一化 + λ₂ 定位损失）
"""
from types import SimpleNamespace


def cp_default_args():
    """build() 所需的默认超参（论文设置）"""
    return SimpleNamespace(
        backbone='vgg16_bn', row=2, line=2,
        set_cost_class=1, set_cost_point=0.05,
        point_loss_coef=0.0002, eos_coef=0.5,
        no_pretrained=False, vgg_weights='')


def build_cpoint(pretrained=True):
    """Consistent-Point 模型（P2PNet 基座；forward 返回
    pred_logits / pred_points / anchor_points）"""
    from .p2pnet import build
    args = cp_default_args()
    args.no_pretrained = not pretrained
    return build(args, training=False)
