# -*- coding: utf-8 -*-
"""CU 点预测模型（"Calibrating Uncertainty for Semi-Supervised Crowd
Counting", ICCV 2023，arXiv:2308.09887）——vendor 自本仓库
../Point-based Crowd Counting Algorithm/Uncertainty Semi-Supervised Crowd
Counting（独立复现实现，官方参考 superlc1995/Calibrating_count）。结构：
- p2pnet.py / matcher.py / backbone.py / vgg_.py：P2PNet 基座
  （forward 额外输出 pred_uncer：patch 级不确定性分支，64x64 patch 一值）
- ema.py：Mean-Teacher EMA（teacher <- EMA(student)，decay=0.999）
- asm.py：空间匹配距离 dist(P,Q)（匈牙利配对 + 未配对满额罚 C=patch 对角线）
- bank.py：有标注图像 patch 库——时间累积 ASM 代理真值（教师快照），
  训练时把随机裁剪子块映射回原图坐标系查询
- ranking.py：不确定性分支的成对排序损失（Eq.2，batch min-max 归一化 ASM）
"""
from types import SimpleNamespace


def cu_default_args():
    """build() 所需的默认超参（论文设置：patch_size=64 最优，Table 8）"""
    return SimpleNamespace(
        backbone='vgg16_bn', row=2, line=2,
        set_cost_class=1, set_cost_point=0.05,
        point_loss_coef=0.0002, eos_coef=0.5,
        patch_size=64, no_pretrained=False, vgg_weights='')


def build_cu(pretrained=True):
    """CU 模型（P2PNet 基座 + 不确定性分支；forward 返回
    pred_logits / pred_points / anchor_points / pred_uncer）"""
    from .p2pnet import build
    args = cu_default_args()
    args.no_pretrained = not pretrained
    return build(args, training=False)


from .ema import ModelEMA
from .asm import spatial_matching_distance
from .bank import PatchBank
from .ranking import ranking_loss
