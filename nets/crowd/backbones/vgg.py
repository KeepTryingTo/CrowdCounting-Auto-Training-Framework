# -*- coding: utf-8 -*-
"""VGG16 前端特征 + FPN 风格多尺度解码骨干，vendor 自参考项目 models/backbones/vgg.py

vendored 改动：torchvision 预训练加载兼容新旧 API。
"""
import torch
from torch import nn
from torchvision import models

from nets.crowd.utils.modules import upsample_bilinear


def _vgg16_features(pretrained):
    """torchvision vgg16 features（兼容 weights= / pretrained= 两种 API）"""
    try:
        if pretrained:
            return models.vgg16(weights="DEFAULT").features
        return models.vgg16(weights=None).features
    except (TypeError, ValueError):
        # 旧版 torchvision（< 0.13）无 weights 关键字
        return models.vgg16(pretrained=pretrained).features


class Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(Backbone, self).__init__()

        # TODO frontend feature exactor
        model = list(_vgg16_features(pretrained).children())
        self.feblock1 = nn.Sequential(*model[:16])
        self.feblock2 = nn.Sequential(*model[16:23])
        self.feblock3 = nn.Sequential(*model[23:30])

        # backend
        self.beblock3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.beblock2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.beblock1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pred_den = nn.Sequential(
            nn.Conv2d(in_channels=896, out_channels=1, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.feblock1(x)
        x1 = x
        x = self.feblock2(x)
        x2 = x
        x = self.feblock3(x)

        # decoding stage
        x = self.beblock3(x)
        x3_ = x
        x = upsample_bilinear(x, x2.shape)
        x = torch.cat([x, x2], 1)

        x = self.beblock2(x)
        x2_ = x
        x = upsample_bilinear(x, x1.shape)
        x = torch.cat([x, x1], 1)

        x1_ = self.beblock1(x)

        x2_ = upsample_bilinear(x2_, x1.shape)
        x3_ = upsample_bilinear(x3_, x1.shape)

        x = torch.cat([x1_, x2_, x3_], 1)

        return x


def demo():
    x = torch.zeros(size=(1, 3, 320, 320))
    model = Backbone()
    out = model(x)
    print('out.shape: {}'.format(out.size()))


if __name__ == '__main__':
    demo()
