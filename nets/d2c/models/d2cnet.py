# -*- coding: utf-8 -*-
"""PMR (VGG16-UNet) and CMR (VGG16 local counter)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import VGG16Encoder


def _conv(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
        nn.ReLU(inplace=True),
    )


class PMR(nn.Module):
    """Probability Map Regression: VGG16 encoder + UNet-like decoder."""

    def __init__(self, pretrained: bool = True, weights_path: str = ""):
        super().__init__()
        self.encoder = VGG16Encoder(3, pretrained=pretrained,
                                    weights_path=weights_path or None)
        self.decode4 = nn.Sequential(_conv(1024, 512), _conv(512, 256))
        self.decode3 = nn.Sequential(_conv(512, 256), _conv(256, 128))
        self.decode2 = nn.Sequential(_conv(256, 128), _conv(128, 64))
        self.decode1 = nn.Sequential(_conv(128, 64))
        self.prob = nn.Conv2d(64, 1, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        d4 = self.decode4(torch.cat([f["c4"], self._up(f["c5"], f["c4"])], 1))
        d3 = self.decode3(torch.cat([f["c3"], self._up(d4, f["c3"])], 1))
        d2 = self.decode2(torch.cat([f["c2"], self._up(d3, f["c2"])], 1))
        d1 = self.decode1(torch.cat([f["c1"], self._up(d2, f["c1"])], 1))
        p = self.prob(d1)
        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear",
                          align_corners=False)
        # 论文：用 clamp 到 [0,1] 代替 sigmoid。
        return torch.clamp(p, 0.0, 1.0)


class CMR(nn.Module):
    """Count Map Regression: VGG16 + 1x1 head, output stride 32."""

    rf = 32

    def __init__(self, pretrained: bool = True, weights_path: str = ""):
        super().__init__()
        self.encoder = VGG16Encoder(1, pretrained=pretrained,
                                    weights_path=weights_path or None)
        self.head = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, 1),
            nn.Softplus(),
        )

    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        f = self.encoder(prob)
        return self.head(f["pool5"])


class D2CNet(nn.Module):
    def __init__(self, pretrained: bool = True, weights_path: str = ""):
        super().__init__()
        self.pmr = PMR(pretrained, weights_path)
        self.cmr = CMR(pretrained, weights_path)

    def forward(self, x: torch.Tensor):
        prob = self.pmr(x)
        count = self.cmr(prob)
        return prob, count
