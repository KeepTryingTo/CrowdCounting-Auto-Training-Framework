# -*- coding: utf-8 -*-
"""自定义人群计数密度网示例：3 层 CNN 密度头。

复制为 my_density_net.py 改成自己的网络即可出现在下拉框
（名为 example_model.py 的文件会被扫描器跳过）。

注册约定: factory(num_classes, pretrained) 返回的模型
forward(x) -> [N,1,H',W'] 密度图（sum ≈ 人数）；
与 GT 尺寸不一致时训练管线自动双线性对齐。
"""
import torch.nn as nn

from app.core.registry import register_custom_model


class SimpleDensityNet(nn.Module):
    """轻量密度网示例：下采样 1/8 + 1x1 出 1 通道密度图"""

    def __init__(self):
        super(SimpleDensityNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        return self.head(self.features(x))


@register_custom_model("simple_density_net", desc="3 层 CNN 密度网示例",
                       tasks=["crowd_density"])
def build(num_classes=1, pretrained=False, **kwargs):
    return SimpleDensityNet()
