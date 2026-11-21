# -*- coding: utf-8 -*-
"""DCCUS / LLB_Count 的 MetaMemNet 移植（AAAI 2023 / 局部-边界感知扩展）

来源：
- DCCUS/networks/MetaMemNet.py + backbones/{vgg,convnext}.py
- LLB_Count/networks/MetaMemNet.py + backbones/convnext.py（CCA 启用、通道更宽）

保留原模块命名（sem_mem/sty_mem/sem_down/sty_down/output_layer/backbone.*），
便于加载官方权重。MetaModule 用普通 nn.Module 替代（仅需 state_dict/deepcopy）。

- 推理：model(x) -> density [B,1,H,W]
- 训练：model.train_forward(x, label) -> (den, binmap|None, sim, sim2, orth)
  label 为在线聚类得到的伪域 id（sty_mem 索引）
- 聚类特征：model.conv_features(x) -> [B,256,H/4,W/4]
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_bilinear(x, size):
    """size 可为 torch.Size（取后两维）或 (h, w)；align_corners=False（同源）"""
    if not torch.is_tensor(size) and not isinstance(size, (tuple, list)):
        size = (size, size)
    hw = tuple(size[-2:]) if len(size) >= 2 else (size[0], size[0])
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)


def _torchvision(name, pretrained):
    from torchvision import models
    if not pretrained:
        try:
            return getattr(models, name)(weights=None)
        except TypeError:
            return getattr(models, name)(pretrained=False)
    try:
        weights = {"vgg16": models.VGG16_Weights.IMAGENET1K_V1,
                   "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1}[name]
        return getattr(models, name)(weights=weights)
    except Exception:
        try:
            return getattr(models, name)(pretrained=True)
        except Exception:
            return getattr(models, name)(weights=None)


# ------------------------- 骨干 -------------------------

class VGGBackbone(nn.Module):
    """DCCUS vgg 分支（torchvision vgg16，无 BN），输出 896 通道 /4 特征"""

    def __init__(self, pretrained=False):
        super().__init__()
        model = list(_torchvision("vgg16", pretrained).features.children())
        self.feblock1 = nn.Sequential(*model[:16])
        self.feblock2 = nn.Sequential(*model[16:23])
        self.feblock3 = nn.Sequential(*model[23:30])
        self.beblock3 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(inplace=True))
        self.beblock2 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True))
        self.beblock1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(512 + 256 + 128, 512 + 256 + 128, 1)

    def forward(self, x):
        x = self.feblock1(x)
        x1 = x
        x = self.feblock2(x)
        x2 = x
        x = self.feblock3(x)

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
        return self.final(x)


class _Conv1x1(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 1, bias=False))


class CrissCrossAttention(nn.Module):
    """CCNet 十字交叉注意力（LLB convNeXt 分支启用）"""

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()
        q = self.query_conv(x)
        q_h = q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        q_w = q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
        k = self.key_conv(x)
        k_h = k.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        k_w = k.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        v = self.value_conv(x)
        v_h = v.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        v_w = v.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        energy_h = torch.bmm(q_h, k_h)
        idx = torch.arange(h, device=x.device)
        energy_h = energy_h.view(b, w, h, h)
        energy_h[:, :, idx, idx] = float("-inf")
        energy_h = energy_h.permute(0, 2, 1, 3)
        energy_w = torch.bmm(q_w, k_w).view(b, h, w, w)
        att = self.softmax(torch.cat([energy_h, energy_w], dim=3))
        att_h = att[:, :, :, :h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        att_w = att[:, :, :, h:h + w].contiguous().view(b * h, w, w)
        out_h = torch.bmm(v_h, att_h.permute(0, 2, 1)).view(
            b, w, -1, h).permute(0, 2, 3, 1)
        out_w = torch.bmm(v_w, att_w.permute(0, 2, 1)).view(
            b, h, -1, w).permute(0, 2, 1, 3)
        return self.gamma * (out_h + out_w) + x


class ConvNeXtBackbone(nn.Module):
    """convnext_tiny + 多尺度解码（DCCUS / LLB 两种通道配置，LLB 启用 CCA）"""

    def __init__(self, pretrained=False, variant="dccus", threshold=0.1):
        super().__init__()
        self.variant = variant
        self.threshold = threshold
        model = _torchvision("convnext_tiny", pretrained)
        self.layer1 = nn.Sequential(model.features[:2])
        self.layer2 = nn.Sequential(model.features[2:4])
        self.layer3 = nn.Sequential(model.features[4:6])

        if variant == "llb":
            self.layer3_backend = nn.Sequential(_Conv1x1(384, 256), _Conv1x1(256, 256))
            self.layer2_backend = nn.Sequential(_Conv1x1(192 + 256, 128), _Conv1x1(128, 128))
            self.layer1_backend = nn.Sequential(_Conv1x1(96 + 128, 128), _Conv1x1(128, 128))
            out_ch = 256 + 128 + 128
            self.ccA3 = CrissCrossAttention(out_ch)
            self.binary_den_map = None
        else:
            self.layer3_backend = nn.Sequential(_Conv1x1(384, 256), _Conv1x1(256, 128))
            self.layer2_backend = nn.Sequential(_Conv1x1(192 + 128, 128), _Conv1x1(128, 64))
            self.layer1_backend = nn.Sequential(_Conv1x1(96 + 64, 64), _Conv1x1(64, 64))
            out_ch = 128 + 64 + 64
            self.ccA3 = None
            self.binary_den_map = nn.Sequential(nn.Conv2d(out_ch, 1, 1), nn.Sigmoid())
        self.final = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x):
        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        x2 = x
        x = self.layer3(x)

        x = self.layer3_backend(x)
        x3_ = x
        x = upsample_bilinear(x, x2.shape)
        x = torch.cat([x, x2], 1)
        x = self.layer2_backend(x)
        x2_ = x
        x = upsample_bilinear(x, x1.shape)
        x = torch.cat([x, x1], 1)
        x1_ = self.layer1_backend(x)
        x2_ = upsample_bilinear(x2_, x1.shape)
        x3_ = upsample_bilinear(x3_, x1.shape)
        x = self.final(torch.cat([x1_, x2_, x3_], 1))
        if self.ccA3 is not None:      # LLB：启用 CCA
            x = self.ccA3(x)
            return x
        bmap = self.binary_den_map(x)  # DCCUS：前景 mask 监督分支
        bmap = torch.where(bmap > self.threshold, bmap,
                           torch.zeros((), device=bmap.device))
        return x, bmap


# ------------------------- MetaMemNet -------------------------

class MetaMemNet(nn.Module):
    """DCCUS / LLB 的记忆增强网络（DICM 语义 memory + DSCM 风格 memory）

    backbone: "vgg" | "convNeXt_tiny"
    variant:  "dccus" | "llb"（仅 convNeXt 生效：通道配置与 CCA）
    """

    def __init__(self, pretrained=True, backbone_name="convNeXt_tiny",
                 variant="dccus", num_clusters=4, threshold=0.1, part_num=1024):
        super().__init__()
        self.backbone_name = backbone_name
        self.variant = variant
        self.threshold = threshold
        if backbone_name == "vgg":
            self.input_channels = 512 + 256 + 128
            self.backbone = VGGBackbone(pretrained=pretrained)
        elif backbone_name == "convNeXt_tiny":
            self.backbone = ConvNeXtBackbone(pretrained=pretrained,
                                             variant=variant,
                                             threshold=threshold)
            self.input_channels = (256 + 128 + 128) if variant == "llb" \
                else (128 + 64 + 64)
        else:
            raise ValueError(f"不支持的单域泛化骨干: {backbone_name}"
                             "（可选 vgg / convNeXt_tiny）")

        variance = math.sqrt(1.0)
        self.part_num = part_num
        self.sem_mem = nn.Parameter(
            torch.FloatTensor(1, 256, part_num).normal_(0.0, variance))
        self.sty_mem = nn.Parameter(
            torch.FloatTensor(num_clusters, 1, 256, part_num // num_clusters)
            .normal_(0.0, variance))
        self.sem_down = nn.Sequential(
            nn.Conv2d(self.input_channels, 256, 1, bias=False),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.5))
        self.sty_down = nn.Sequential(
            nn.Conv2d(self.input_channels, 256, 1, bias=False),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.5))
        self.output_layer = nn.Sequential(
            nn.Conv2d(256, 1, 1, bias=False), nn.ReLU(inplace=True))

    # ---- 特征（在线聚类用） ----
    def conv_features(self, x):
        if self.backbone_name == "vgg":
            x = self.backbone(x)
        else:
            x = self.backbone(x)
            if isinstance(x, tuple):
                x = x[0]
        return self.sty_down(x)

    def _backbone_forward(self, x):
        out = self.backbone(x)
        if isinstance(out, tuple):
            return out[0], out[1]
        return out, None

    def _density(self, x):
        """记忆重编码 → 密度图（/4 分辨率）"""
        memory = self.sem_mem.repeat(x.shape[0], 1, 1)
        memory_key = memory.transpose(1, 2)
        sem_pre = self.sem_down(x)
        sem_pre_ = sem_pre.view(sem_pre.shape[0], sem_pre.shape[1], -1)
        di_logits = torch.bmm(memory_key, sem_pre_)
        inv = torch.bmm(memory_key.transpose(1, 2), F.softmax(di_logits, dim=1))
        inv = inv.view(inv.shape[0], inv.shape[1], sem_pre.shape[2],
                       sem_pre.shape[3])
        return self.output_layer(inv), sem_pre_, inv

    def forward(self, x):
        feat, _ = self._backbone_forward(x)
        den, _, _ = self._density(feat)
        return upsample_bilinear(den, x.shape)

    def train_forward(self, x, label):
        """训练前向：返回 (den, binmap|None, sim, sim2, orth)

        binmap：DCCUS convNeXt 的前景 mask（vgg / LLB 为 None）
        sim / sim2 / orth：语义/风格重建 CE 与正交损失（未加权）
        """
        feat, binmap = self._backbone_forward(x)
        den, sem_pre_, _ = self._density(feat)
        den = upsample_bilinear(den, x.shape)

        b = sem_pre_.shape[0]
        hw = sem_pre_.shape[2]
        sim_gt = torch.arange(hw, device=x.device).unsqueeze(0).repeat(b, 1)
        recon_sim = torch.bmm(sem_pre_.transpose(1, 2), sem_pre_)
        sim = F.cross_entropy(recon_sim, sim_gt.long(), reduction="none") * 0.1

        sty_pre = self.sty_down(feat)
        sty_pre_ = sty_pre.view(sty_pre.shape[0], sty_pre.shape[1], -1)
        if torch.is_tensor(label):
            idx = label.to(self.sem_mem.device).long().reshape(-1)
        else:
            idx = torch.full((b,), int(label), dtype=torch.long,
                             device=self.sem_mem.device)
        memory2 = self.sty_mem[idx]          # int → (1,256,P/K)；tensor → (B,1,256,P/K)
        if memory2.dim() == 4:
            memory2 = memory2.squeeze(1)
        if memory2.shape[0] != b:
            memory2 = memory2.expand(b, -1, -1)
        mem2_key = memory2.transpose(1, 2)
        ds_logits = torch.bmm(mem2_key, sty_pre_)
        spe = torch.bmm(mem2_key.transpose(1, 2), F.softmax(ds_logits, dim=1))
        recon_sim2 = torch.bmm(spe.transpose(1, 2), sty_pre_)
        sim2 = F.cross_entropy(recon_sim2, sim_gt.long(), reduction="sum") * 0.1

        orth_pre = torch.bmm(sty_pre_.transpose(1, 2), sem_pre_)
        orth = 0.01 * torch.sum(torch.diagonal(
            orth_pre, dim1=-2, dim2=-1).pow(2))
        return den, binmap, sim, sim2, orth


def build_metamemnet(pretrained=True, backbone_name="convNeXt_tiny",
                     variant="dccus", num_clusters=4, threshold=0.1, **kw):
    """构建 DCCUS / LLB 的 MetaMemNet（variant 区分两者骨干配置与 CCA）"""
    return MetaMemNet(pretrained=pretrained, backbone_name=backbone_name,
                      variant=variant, num_clusters=num_clusters,
                      threshold=threshold, **kw)
