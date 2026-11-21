# -*- coding: utf-8 -*-
"""Truncated VGG-16 backbone, output stride 8."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn


def _candidate_weights() -> list[Path]:
    here = Path(__file__).resolve()
    roots = list(here.parents)
    candidates = [
        here.parents[1] / "weights" / "vgg16_bn.pth",
        here.parents[2] / "weights" / "vgg16_bn.pth",
    ]
    # Workspace-relative fallbacks (this repository ships a local copy).
    for base in roots:
        candidates += [
            base / "Codes" / "weights" / "pretrained" / "vgg16_bn.pth",
            base / "Point-based Crowd Counting Algorithm" / "PET" /
            "pretrained" / "vgg16_bn-6c64b313.pth",
        ]
    return candidates


def _load_state_dict(model: nn.Module, path: Path) -> None:
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


class VGG16Frontend(nn.Module):
    """截断 VGG-16 骨干（论文：只保留前 13 个卷积层，输出 1/8）。

    具体取 conv1_1~conv4_3，再拼 conv5_1~conv5_3，但跳过 pool4/pool5；
    因此空间分辨率保持 1/8，通道数为 512，后面接三个任务分支。
    """

    out_channels = 512
    stride = 8

    def __init__(self, pretrained: bool = True,
                 weights_path: Optional[str] = None):
        super().__init__()
        try:
            import torchvision
            try:
                base = torchvision.models.vgg16_bn(weights=None)
            except TypeError:  # torchvision < 0.13
                base = torchvision.models.vgg16_bn(pretrained=False)
            feats = list(base.features.children())
            # VGG16_bn features 索引：[:33] 到 conv4_3+ReLU（1/8），
            # [34:43] 为 conv5_1~conv5_3；中间跳过 pool4(33)/pool5(43)。
            body = nn.Sequential(*feats[:33], *feats[34:43])
            self.body = body
            # 权重优先级：显式路径 > 工程/工作区本地权重 > torchvision 下载。
            if weights_path:
                _load_state_dict(base, Path(weights_path))
            elif pretrained:
                found = next((p for p in _candidate_weights() if p.is_file()), None)
                if found is not None:
                    _load_state_dict(base, found)
                else:
                    try:
                        w = torchvision.models.VGG16_BN_Weights.DEFAULT
                        base = torchvision.models.vgg16_bn(weights=w)
                        feats = list(base.features.children())
                        self.body = nn.Sequential(*feats[:33], *feats[34:43])
                    except AttributeError:
                        try:
                            base = torchvision.models.vgg16_bn(pretrained=True)
                            feats = list(base.features.children())
                            self.body = nn.Sequential(*feats[:33],
                                                      *feats[34:43])
                        except Exception as exc:
                            print(f"[warn] 未加载 VGG16_bn 预训练权重: {exc}")
                    except Exception as exc:
                        print(f"[warn] 未加载 VGG16_bn 预训练权重: {exc}")
            # Important: build from the *loaded* base's children.
            self.body = nn.Sequential(*list(base.features.children())[:33],
                                      *list(base.features.children())[34:43])
        except Exception as exc:
            raise RuntimeError(f"构建 VGG16 骨干失败: {exc}") from exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
