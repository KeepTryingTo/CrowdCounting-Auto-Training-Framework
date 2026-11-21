# -*- coding: utf-8 -*-
"""公共模块（IBN 归一化 / 双线性上采样），vendor 自参考项目 models/utils/modules.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(num_features=self.half, affine=True)
        self.BN = nn.BatchNorm2d(num_features=planes - self.half)

    def forward(self, x):
        split = torch.split(tensor=x, split_size_or_sections=self.half, dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), dim=1)
        return out


def upsample_bilinear(x, size):
    return F.interpolate(x, size=size[2:],
                         mode='bilinear', align_corners=False)
