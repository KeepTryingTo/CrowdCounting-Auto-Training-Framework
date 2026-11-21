# -*- coding: utf-8 -*-
"""Main-Net and RAZ-Net.

Main-Net:
  truncated VGG16 -> counting branch, localization branch, zoom proposal
  branch (count+loc features -> attention).

RAZ-Net:
  same backbone/localization/attention configuration as Main-Net, with the
  counting branch removed; attention input comes from localization features
  only.  It can load all overlapping parameters from Main-Net.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import VGG16Frontend


def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, p: int = 1,
                  d: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=d if k == 3 else p, dilation=d,
                  bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class LocalizationHead(nn.Module):
    """定位分支：从 1/8 特征经 3 次反卷积恢复到输入分辨率。

    返回保留通道的 loc_feat（供注意力分支复用）和单通道 logits
    （BCE 用 logits，推理时再 sigmoid）。
    """

    def __init__(self, in_ch: int = 512, width: int = 256):
        super().__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, width, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.conv1 = _conv_bn_relu(width, width)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(width, width // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width // 2), nn.ReLU(inplace=True),
        )
        self.conv2 = _conv_bn_relu(width // 2, width // 2)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(width // 2, width // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width // 4), nn.ReLU(inplace=True),
        )
        self.conv3 = _conv_bn_relu(width // 4, width // 4)
        self.out = nn.Conv2d(width // 4, 1, 1)
        self.out_ch = width // 4

    def forward(self, x: torch.Tensor):
        x = self.deconv1(x)
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv2(x)
        x = self.deconv3(x)
        feat = self.conv3(x)
        logits = self.out(feat)
        return feat, logits


class AttentionHead(nn.Module):
    """缩放区域提议分支。

    Main-Net 用 count+loc 两路特征融合后预测注意力；
    RAZ-Net 只保留 loc 特征（论文要求），但共享 loc_proj/head 参数。
    """

    def __init__(self, loc_ch: int = 64, proj_ch: int = 16):
        super().__init__()
        self.loc_proj = nn.Sequential(
            nn.Conv2d(loc_ch, proj_ch, 1, bias=False),
            nn.BatchNorm2d(proj_ch), nn.ReLU(inplace=True),
        )
        self.count_proj = nn.Sequential(
            nn.Conv2d(256, proj_ch, 1, bias=False),
            nn.BatchNorm2d(proj_ch), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(proj_ch * 2, proj_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_ch), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(proj_ch, proj_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_ch), nn.ReLU(inplace=True),
            nn.Conv2d(proj_ch, 1, 1),
        )

    def forward_loc_only(self, loc_feat: torch.Tensor) -> torch.Tensor:
        return self.head(self.loc_proj(loc_feat))

    def forward_fused(self, count_feat: torch.Tensor,
                      loc_feat: torch.Tensor) -> torch.Tensor:
        # 计数特征是 1/8 分辨率，先上采样到定位特征的全分辨率再拼接。
        count_up = F.interpolate(self.count_proj(count_feat),
                                 size=loc_feat.shape[-2:], mode="bilinear",
                                 align_corners=False)
        fused = torch.cat([count_up, self.loc_proj(loc_feat)], dim=1)
        return self.head(self.fuse(fused))


class MainNet(nn.Module):
    def __init__(self, pretrained: bool = True, weights_path: str | None = None):
        super().__init__()
        self.backbone = VGG16Frontend(pretrained=pretrained,
                                      weights_path=weights_path)
        self.count_trunk = nn.Sequential(
            _conv_bn_relu(512, 512, d=2),
            _conv_bn_relu(512, 512, d=2),
            _conv_bn_relu(512, 256),
        )
        self.count_out = nn.Conv2d(256, 1, 1)
        self.loc_head = LocalizationHead(512, 256)
        self.attn_head = AttentionHead(self.loc_head.out_ch, 16)

    def forward(self, x: torch.Tensor, need_density: bool = True) -> Dict:
        feat = self.backbone(x)
        out = {}
        # 计数分支：两层 dilation=2 扩大感受野 + 1 层融合；softplus
        # 保证密度非负。
        count_feat = self.count_trunk(feat)
        loc_feat, loc_logits = self.loc_head(feat)
        loc_logits = F.interpolate(loc_logits, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
        if need_density:
            out["density"] = F.softplus(self.count_out(count_feat))
        out["loc"] = loc_logits
        # 注意力输出可能比输入略大（非 8 倍数尺寸），统一插值回输入大小。
        attn = self.attn_head.forward_fused(count_feat, loc_feat)
        out["attn"] = F.interpolate(attn, size=x.shape[-2:], mode="bilinear",
                                    align_corners=False)
        out["count_feat"] = count_feat
        out["loc_feat"] = loc_feat
        return out


class RAZNet(nn.Module):
    def __init__(self, pretrained: bool = True, weights_path: str | None = None):
        super().__init__()
        self.backbone = VGG16Frontend(pretrained=pretrained,
                                      weights_path=weights_path)
        self.loc_head = LocalizationHead(512, 256)
        self.attn_head = AttentionHead(self.loc_head.out_ch, 16)

    def forward(self, x: torch.Tensor) -> Dict:
        feat = self.backbone(x)
        loc_feat, loc_logits = self.loc_head(feat)
        loc_logits = F.interpolate(loc_logits, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
        attn = self.attn_head.forward_loc_only(loc_feat)
        attn = F.interpolate(attn, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)
        return {"loc": loc_logits, "attn": attn,
                "loc_feat": loc_feat}

    def load_from_main(self, main: MainNet, strict: bool = False):
        """用 Main-Net 初始化 RAZ-Net 的骨干/定位/注意力参数。

        RAZ-Net 没有计数分支，因此只加载同名且 shape 一致的参数；
        Main-Net 的 count_*、attn_head.count_proj/fuse 会被自然跳过。
        """
        src = main.state_dict()
        dst = self.state_dict()
        loaded = {k: v for k, v in src.items()
                  if k in dst and dst[k].shape == v.shape}
        missing, unexpected = self.load_state_dict(loaded, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(f"RAZ 初始化参数不完整: missing={missing}, "
                               f"unexpected={unexpected}")
        return missing, unexpected


def set_trainable(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(flag)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
