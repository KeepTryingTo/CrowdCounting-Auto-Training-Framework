# -*- coding: utf-8 -*-
"""CrowdVGG 密度网，vendor 自参考项目 models/crowd_vgg.py

VGG16 骨干（前端特征 + FPN 风格多尺度解码）+ PreDensityHead，
输出上采样回原图尺寸的 1 通道密度图（sum ≈ 人数）。
"""
import torch
from torch import nn

from nets.crowd.backbones.vgg import Backbone
from nets.crowd.utils.modules import upsample_bilinear
from nets.crowd.heads.pred_density import PreDensityHead


class CrowdVGG(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = Backbone(pretrained=pretrained)
        self.head = PreDensityHead(in_channels=896)

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        out = upsample_bilinear(out, x.size())
        return out


if __name__ == '__main__':
    model = CrowdVGG(True)
    x = torch.randn(size=(1, 3, 512, 512))
    out = model(x)

    print('out.size: {}'.format(out.size()))
