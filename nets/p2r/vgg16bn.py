# -*- coding: utf-8 -*-
"""P2R 点预测网：VGG16-BN 编码 + FPN 融合 + 解码

vendor 自 P2RLoss-main/models/vgg16bn.py（CVPR 2025），结构不变：
- encoders：VGG16-BN features[0:33]（/8）与 [33:43]（/16）两级
- fuse_layer：两级特征 1×1 对齐到 /8 尺寸后相加 + 3×3 融合（256ch）
- decoders：两层 3×3 卷积 + PixelShuffle(×2) 上采样，输出 2 通道，
  den = den2 - den1（差分 logits，收敛更稳）
输出空间分辨率 = 输入 / 4（下采样率 down = 4，输入需 padding 到 /32）
"""
import torch
import torch.nn as nn
from torchvision import models

from .utils import UpSample_P2P, SimpleDecoder


def _vgg16_bn(pretrained=True):
    """torchvision 权重 API 新旧兼容（0.13+ 枚举 / 旧版布尔）"""
    try:
        from torchvision.models import VGG16_BN_Weights
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        return models.vgg16_bn(weights=weights)
    except (ImportError, AttributeError):
        return models.vgg16_bn(pretrained=pretrained)


class VGG16_BN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = _vgg16_bn(pretrained)
        features = list(vgg.features.children())
        lids = [0, 33, 43]
        self.encoders = nn.ModuleList(
            nn.Sequential(*features[a:b]) for a, b in zip(lids[:-1], lids[1:])
        )
        self.num_channels = [512, 512]
        self.num_stage = len(self.num_channels)
        self.fuse_layer = UpSample_P2P(self.num_channels, ouc=256, bn=False,
                                       relu=False)

        self.decoders = SimpleDecoder(
            in_channel=self.fuse_layer.fuse_channel,
            fea_channel=self.fuse_layer.fuse_channel,
            up_scale=2,
            out_channel=2
        )

    def forward(self, image, need_fp=False):
        fea2 = self.encoding(image)
        if need_fp:
            import torch.nn.functional as F
            fea2 = F.dropout2d(fea2, p=0.5)
        denmap = self.decoding(fea2)
        return denmap

    def encoding(self, x):
        feas = []
        for module in self.encoders:
            feas.append(x := module(x))
        # 返回倒数 num_stage 层的特征图
        feas = feas[-self.num_stage:]
        fea = self.fuse_layer(feas)
        return fea

    def decoding(self, fea2):
        denmap = self.decoders(fea2)
        # 默认输出通道为 2：差分 logits
        if denmap.size(1) > 1:
            den1, den2 = denmap[:, :1], denmap[:, 1:2]
            den = den2 - den1
        else:
            den = denmap
        return den
