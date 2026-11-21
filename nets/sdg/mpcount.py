# -*- coding: utf-8 -*-
"""MPCount（CVPR 2024）模型移植

来源：DCCUS/MPCount-main/models/models.py（DGModel_base → DGModel_final）
原样保留模块命名（enc1..3 / dec1..3 / den_dec / den_head / cls_head / mem），
以便加载官方训练权重。

- 输入：x [B,3,H,W]（ImageNet 归一化）
- 推理：model(x) -> (density [B,1,H,W], cls_map [B,1,H/16,W/16])
- 训练：model.forward_train(img1, img2, c_gt) -> 7 元组（双分支 + 一致性）
"""
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


def _vgg16_bn(pretrained: bool):
    """构建 torchvision vgg16_bn（兼容新旧 API；预训练失败回退随机初始化）"""
    from torchvision import models
    if pretrained:
        try:
            return models.vgg16_bn(
                weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        except Exception:
            try:
                return models.vgg16_bn(pretrained=True)
            except Exception:
                pass
    try:
        return models.vgg16_bn(weights=None)
    except TypeError:
        return models.vgg16_bn(pretrained=False)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False, bn=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu is not None:
            y = self.relu(y)
        return y


def upsample(x, scale_factor=2, mode="bilinear"):
    if mode == "nearest":
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    return F.interpolate(x, scale_factor=scale_factor, mode=mode,
                         align_corners=False)


class DGModelBase(nn.Module):
    def __init__(self, pretrained=True, den_dropout=0.5):
        super().__init__()
        self.den_dropout = den_dropout
        vgg = _vgg16_bn(pretrained)
        self.enc1 = nn.Sequential(*list(vgg.features.children())[:23])
        self.enc2 = nn.Sequential(*list(vgg.features.children())[23:33])
        self.enc3 = nn.Sequential(*list(vgg.features.children())[33:43])

        self.dec3 = nn.Sequential(ConvBlock(512, 1024, bn=True),
                                  ConvBlock(1024, 512, bn=True))
        self.dec2 = nn.Sequential(ConvBlock(1024, 512, bn=True),
                                  ConvBlock(512, 256, bn=True))
        self.dec1 = nn.Sequential(ConvBlock(512, 256, bn=True),
                                  ConvBlock(256, 128, bn=True))
        self.den_dec = nn.Sequential(
            ConvBlock(512 + 256 + 128, 256, kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout))
        self.den_head = nn.Sequential(
            ConvBlock(256, 1, kernel_size=1, padding=0))

    def forward_fe(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x = self.dec3(x3)
        y3 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        y2 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        y1 = x

        y2 = upsample(y2, scale_factor=2)
        y3 = upsample(y3, scale_factor=4)
        return torch.cat([y1, y2, y3], dim=1), x3

    def forward(self, x):
        y_cat, _ = self.forward_fe(x)
        d = self.den_head(self.den_dec(y_cat))
        return upsample(d, scale_factor=4)


class DGModelMem(DGModelBase):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256,
                 den_dropout=0.5):
        super().__init__(pretrained, den_dropout)
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.mem = nn.Parameter(
            torch.FloatTensor(1, self.mem_dim, self.mem_size).normal_(
                mean=0.0, std=1.0))
        self.den_dec = nn.Sequential(
            ConvBlock(512 + 256 + 128, out_channels=self.mem_dim,
                      kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout))
        self.den_head = nn.Sequential(
            ConvBlock(self.mem_dim, out_channels=1, kernel_size=1,
                      padding=0))

    def forward_mem(self, y):
        b, k, h, w = y.shape
        m = self.mem.repeat(b, 1, 1)
        m_key = m.transpose(1, 2)
        y_ = y.view(b, k, -1)
        logits = torch.bmm(m_key, y_) / sqrt(k)
        y_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        return y_new.view(b, k, h, w), logits

    def forward(self, x):
        y_cat, _ = self.forward_fe(x)
        y_den, _ = self.forward_mem(self.den_dec(y_cat))
        d = self.den_head(y_den)
        return upsample(d, scale_factor=4)


class DGModelMemCls(DGModelMem):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256,
                 den_dropout=0.5, cls_dropout=0.3, cls_thrs=0.5):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout)
        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs
        self.cls_head = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            nn.Dropout2d(p=self.cls_dropout),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid())

    def transform_cls_map_gt(self, c_gt):
        return upsample(c_gt, scale_factor=4, mode="nearest")

    def transform_cls_map_pred(self, c):
        c_new = c.clone().detach()
        c_new[c < self.cls_thrs] = 0
        c_new[c >= self.cls_thrs] = 1
        return upsample(c_new, scale_factor=4, mode="nearest")

    def transform_cls_map(self, c, c_gt=None):
        return self.transform_cls_map_gt(c_gt) if c_gt is not None \
            else self.transform_cls_map_pred(c)

    def forward(self, x, c_gt=None):
        y_cat, x3 = self.forward_fe(x)
        y_den, _ = self.forward_mem(self.den_dec(y_cat))
        c = self.cls_head(x3)
        c_resized = self.transform_cls_map(c, c_gt)
        dc = upsample(self.den_head(y_den) * c_resized, scale_factor=4)
        return dc, c


class DGModelFinal(DGModelMemCls):
    """MPCount 最终模型（AMB + patch classification + 双分支一致性）"""

    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256,
                 cls_thrs=0.5, err_thrs=0.5, den_dropout=0.5,
                 cls_dropout=0.3, has_err_loss=False):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout,
                         cls_dropout, cls_thrs)
        self.err_thrs = err_thrs
        self.has_err_loss = has_err_loss
        self.den_dec = nn.Sequential(
            ConvBlock(512 + 256 + 128, self.mem_dim, kernel_size=1,
                      padding=0, bn=True))

    @staticmethod
    def jsd(logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        return F.mse_loss(p1, p2)

    def forward_train(self, img1, img2, c_gt=None):
        y_cat1, x3_1 = self.forward_fe(img1)
        y_cat2, x3_2 = self.forward_fe(img2)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)
        y_in1 = F.instance_norm(y_den1, eps=1e-5)
        y_in2 = F.instance_norm(y_den2, eps=1e-5)
        e_y = torch.abs(y_in1 - y_in2)
        e_mask = (e_y < self.err_thrs).clone().detach()
        loss_err = F.l1_loss(y_in1, y_in2) if self.has_err_loss else 0
        y_den_masked1 = F.dropout2d(y_den1 * e_mask, self.den_dropout)
        y_den_masked2 = F.dropout2d(y_den2 * e_mask, self.den_dropout)
        y_den_new1, logits1 = self.forward_mem(y_den_masked1)
        y_den_new2, logits2 = self.forward_mem(y_den_masked2)
        loss_con = self.jsd(logits1, logits2)
        c1 = self.cls_head(x3_1)
        c2 = self.cls_head(x3_2)
        c_resized_gt = self.transform_cls_map_gt(c_gt)
        c_resized1 = self.transform_cls_map_pred(c1)
        c_resized2 = self.transform_cls_map_pred(c2)
        c_err = torch.abs(c_resized1 - c_resized2)
        c_resized = torch.clamp(c_resized_gt + c_err, min=0, max=1)
        dc1 = upsample(self.den_head(y_den_new1) * c_resized, scale_factor=4)
        dc2 = upsample(self.den_head(y_den_new2) * c_resized, scale_factor=4)
        c_err = upsample(c_err, scale_factor=4)
        return dc1, dc2, c1, c2, c_err, loss_con, loss_err


def build_mpcount(pretrained=True, **kw):
    """构建 MPCount（DGModelFinal）；kw 透传 mem_size/mem_dim 等"""
    return DGModelFinal(pretrained=pretrained, **kw)
