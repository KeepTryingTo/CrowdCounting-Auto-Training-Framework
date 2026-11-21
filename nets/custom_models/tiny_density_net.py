# -*- coding: utf-8 -*-
"""测试用轻量密度网（会出现在下拉框中）"""
import torch.nn as nn

from app.core.registry import register_custom_model


class TinyDensityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        return self.head(self.features(x))


@register_custom_model("tiny_density_net", desc="测试用轻量密度网",
                       tasks=["crowd_density"])
def build(num_classes=1, pretrained=False, **kwargs):
    return TinyDensityNet()
