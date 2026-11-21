# -*- coding: utf-8 -*-
"""Density Map-based Crowd Counting Algorithm 参考模型 vendor

vendor 自参考项目「Density Map-based Crowd Counting Algorithm」（GitHub:
KeepTryingTo/CrowdCounting-framework-PyTorch）的 models/ 目录：
- backbones/vgg.py     VGG16 前端特征 + FPN 风格多尺度解码（896 通道）
- backbones/resnet.py  ResNet-IBN（IBN-Net）骨干
- heads/pred_density.py PreDensityHead 密度预测头
- utils/modules.py     IBN / 上采样等公共模块
- crowd_vgg.py         CrowdVGG（参考 load_model.py 的 crowdvgg）
- crowd_resnet.py      CrowdResNet（参考 load_model.py 的 crowdresnet）

vendored 改动：包内绝对导入改为相对导入（IDE 可解析，无 sys.path 注入）；
torchvision 预训练加载兼容新旧 API（weights="DEFAULT" → pretrained=True）。
"""

from nets.crowd.crowd_vgg import CrowdVGG
from nets.crowd.crowd_resnet import CrowdResNet

__all__ = ["CrowdVGG", "CrowdResNet"]


def build_reference_crowd_model(name, pretrained=True):
    """按参考项目 load_model.py 的注册名构建模型

    - crowdvgg    : CrowdVGG（VGG16 骨干 + 多尺度解码 + 密度头）
    - crowdresnet : CrowdResNet（ResNet50-IBN-a 骨干 + 密度头）
    """
    if name == "crowdvgg":
        return CrowdVGG(pretrained=pretrained)
    if name == "crowdresnet":
        return CrowdResNet(pretrained=pretrained)
    raise ValueError(f"未知参考计数模型: {name}（支持 crowdvgg / crowdresnet）")
