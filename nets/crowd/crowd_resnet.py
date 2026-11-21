# -*- coding: utf-8 -*-
"""CrowdResNet 密度网，vendor 自参考项目 models/crowd_resnet.py

ResNet50-IBN-a 骨干（IBN-Net，域泛化友好）+ PreDensityHead，
输出上采样回原图尺寸的 1 通道密度图（sum ≈ 人数）。
"""
import torch
from torch import nn

from nets.crowd.backbones import resnet
from nets.crowd.utils.modules import upsample_bilinear
from nets.crowd.heads.pred_density import PreDensityHead


class CrowdResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = resnet.resnet50_ibn_a(pretrained=pretrained)
        self.head = PreDensityHead(in_channels=1024)

    def forward(self, x):
        out, final_out = self.backbone(x)
        out = self.head(out)
        out = upsample_bilinear(out, x.size())
        return out


if __name__ == '__main__':
    model = CrowdResNet(True)
    x = torch.randn(size=(1, 3, 512, 512))
    out = model(x)

    print('out.size: {}'.format(out.size()))
