# -*- coding: utf-8 -*-
"""VGG16_bn encoder shared by PMR and CMR."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn


def _weight_candidates() -> list:
    here = Path(__file__).resolve()
    cands = [here.parents[2] / "weights" / "vgg16_bn.pth"]
    for base in here.parents:
        cands += [
            base / "Codes" / "weights" / "pretrained" / "vgg16_bn.pth",
            base / "Point-based Crowd Counting Algorithm" / "PET" /
            "pretrained" / "vgg16_bn-6c64b313.pth",
        ]
    return cands


def _load_sd(model, path: Path) -> None:
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(str(path), map_location="cpu")
    if isinstance(ck, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(ck.get(key), dict):
                ck = ck[key]
                break
    if not isinstance(ck, dict):
        return
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ck.items()}
    model.load_state_dict(sd, strict=False)


class VGG16Encoder(nn.Module):
    """VGG16_bn feature pyramid: c1(1/1), c2(1/2), c3(1/4), c4(1/8),
    c5(1/16), pool5(1/32).

    PMR uses c1..c5; CMR uses c5+pool5 (rf=32).
    """

    channels = {"c1": 64, "c2": 128, "c3": 256, "c4": 512, "c5": 512}

    def __init__(self, in_channels: int = 3, pretrained: bool = True,
                 weights_path: Optional[str] = None,
                 freeze_bn: bool = False):
        super().__init__()
        import torchvision
        try:
            base = torchvision.models.vgg16_bn(weights=None)
        except TypeError:
            base = torchvision.models.vgg16_bn(pretrained=False)

        if weights_path:
            _load_sd(base, Path(weights_path))
        elif pretrained:
            local = next((p for p in _weight_candidates() if p.is_file()), None)
            if local is not None:
                _load_sd(base, local)
            else:
                try:
                    w = torchvision.models.VGG16_BN_Weights.DEFAULT
                    base = torchvision.models.vgg16_bn(weights=w)
                except Exception as exc:
                    print(f"[warn] VGG16_bn 预训练权重未加载: {exc}")

        self.features = base.features
        if in_channels == 1:
            # CMR 输入为概率图单通道；对 RGB 第一层卷积权重取平均（官方做法）。
            old = self.features[0]
            new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                            old.stride, old.padding, bias=old.bias is not None)
            with torch.no_grad():
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            self.features[0] = new

        self.in_channels = in_channels
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.features
        c1 = f[:6](x)       # 1/1
        c2 = f[6:13](c1)    # 1/2
        c3 = f[13:23](c2)   # 1/4
        c4 = f[23:33](c3)   # 1/8
        c5 = f[33:43](c4)   # 1/16
        pool5 = f[43:44](c5)  # 1/32
        return {"c1": c1, "c2": c2, "c3": c3, "c4": c4,
                "c5": c5, "pool5": pool5}
