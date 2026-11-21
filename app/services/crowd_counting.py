# -*- coding: utf-8 -*-
"""人群计数服务（PyTorch）：密度图回归 + P2PNet 点预测

密度网：GhostNet 系列（本地权重）/ torchvision 分类骨干 /
        参考项目 crowdvgg / crowdresnet（vendor 于 nets/crowd/）/
        自定义密度网（nets/custom_models/）
点预测：P2PNet（vendor 于 nets/p2pnet/，VGG16_bn + FPN + 匈牙利匹配，
        含 SHTechA 预训练检查点）
        P2RLoss（vendor 于 nets/p2r/，CVPR 2025 半监督：教师-学生 EMA +
        Point-to-Region 匹配损失，.txt/.mat/.npy(N×2) 点标注）
        Consistent-Point（vendor 于 nets/cpoint/，arXiv 2503.12441 半监督：
        P2PNet 基座 + PA 位置聚合 + IUC 不确定性校准 + Mean-Teacher）
        CU（vendor 于 nets/cu/，ICCV 2023 半监督：P2PNet 基座 + patch 级
        不确定性分支 + PatchBank ASM 代理真值 + 成对排序损失 + 课程
        伪标签过滤 + Mean-Teacher）
损失：  MSE / MAE / SSIM / MSE+SSIM（crowd_losses.py，另含 OT / MMD 库函数）

数据集格式（训练）：
    dataset_root/
        Train/Images/xxx.jpg         # 图像
        Train/GT/xxx.npy|.txt|.mat  # 密度图 / 点标注（每行 "x y"；.mat 点预测路线；
                                    #   .npy 点预测路线亦支持 (N,2) 坐标，P2RLoss 格式）
        Test/...
    或扁平结构：root/imgs + root/gt(.txt) / root/npys(.npy)
"""
import csv
import re
import threading
import time
from pathlib import Path

from app.services.loss_weight_utils import (all_weights, apply_criterion_weights,
                                       apply_matcher_costs, aux_weight,
                                       describe)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import WEIGHTS_DIR
from nets.ghostnetv2 import ghostnetv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _safe_model_tag(model_name):
    """模型名 → 安全文件名标签（保留字母/数字/下划线/连字符）"""
    tag = re.sub(r"[^\w\-]", "_", str(model_name or "")).strip("_")
    return tag or "model"


def _ckpt_paths(save_dir, model_name):
    """训练 checkpoint 路径：以所选模型名为前缀，便于区分。

    例：model=mobilenet_v2 → mobilenet_v2_best.pth / mobilenet_v2_last.pth
    """
    tag = _safe_model_tag(model_name)
    base = Path(save_dir)
    return base / f"{tag}_best.pth", base / f"{tag}_last.pth"


# ------------------------- 学习率调度器 -------------------------

CROWD_SCHEDULERS = [
    "step", "cosine", "multistep", "poly", "exponential",
    "warmup_cosine", "warm_restarts", "plateau", "none",
]


def build_crowd_scheduler(optimizer, name, total_epochs, params=None):
    """统一调度器构建（密度图 / P2PNet / 点预测 / 单域泛化共用）。

    支持：step / cosine / multistep / poly / exponential /
          warmup_cosine / warm_restarts / plateau / none
    默认（未指定 / 未知名称）使用 cosine 余弦退火。
    返回 (scheduler, is_plateau)：is_plateau=True 时训练循环须
    用 scheduler.step(val_metric) 而非 scheduler.step()。
    """
    params = params or {}
    name = str(name or "cosine").lower()
    T = max(1, total_epochs)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T), False
    if name == "multistep":
        m1 = max(1, int(T * 0.33))
        m2 = max(m1 + 1, int(T * 0.66))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[m1, m2], gamma=0.5), False
    if name == "poly":
        power = float(params.get("poly_power", 0.9))
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: max(0.01, (1 - ep / T) ** power)), False
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95), False
    if name == "warmup_cosine":
        warm = min(max(1, int(params.get("warmup_epochs", 5))), T - 1)
        lin = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warm)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, T - warm))
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [lin, cos], milestones=[warm]), False
    if name == "warm_restarts":
        t0 = max(1, int(params.get("t0_epochs", 10)))
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=2), False
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-8), True
    if name == "none":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0), False
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, T // 4), gamma=0.5), False
    # 默认 cosine（未指定或未知名称也回退到余弦退火）
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T), False


# ------------------------- 模型 -------------------------

def _conv_bn_relu(in_ch, out_ch, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class DensityHead(nn.Module):
    """CSRNet 风格后端：上采样到 1/8 后空洞卷积保持分辨率，输出 1 通道密度图"""

    def __init__(self, in_ch=160):
        super().__init__()
        self.up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.convs = nn.Sequential(
            _conv_bn_relu(in_ch, 256, dilation=2),
            _conv_bn_relu(256, 128, dilation=2),
            _conv_bn_relu(128, 64, dilation=2),
            nn.Conv2d(64, 1, 1))

    def forward(self, x):
        return self.convs(self.up(x))


class CrowdNet(nn.Module):
    """骨干 + 密度头，输出 [N,1,H/8,W/8] 密度图（sum ≈ 人数）

    骨干：GhostNet 系列（本地权重）或 torchvision 分类模型
    （自动提取特征部分，输出通道探测）。
    """

    def __init__(self, backbone="ghostnetv2_10", pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        if backbone.startswith("ghostnetv2"):
            width = float(backbone.split("_")[-1]) / 10
            net = ghostnetv2(num_classes=1000, width=width, dropout=0.0, args=None)
            if pretrained:
                wf = WEIGHTS_DIR / f"ck_ghostnetv2_{int(width*10)}.pth.tar"
                if wf.exists():
                    sd = torch.load(wf, map_location="cpu")
                    sd = {k: v for k, v in sd.items()
                          if "total_ops" not in k and "total_params" not in k}
                    net.load_state_dict(sd, strict=True)
            try:
                in_ch = net.blocks[-1][0].conv.out_channels
            except Exception:
                in_ch = 160
            self.backbone = nn.Sequential(
                net.conv_stem, net.bn1, net.act1, net.blocks)
        elif backbone == "ghostnet":
            from nets.ghostnet import ghostnet
            net = ghostnet(num_classes=1000, width=1.0, dropout=0.0)
            try:
                in_ch = net.blocks[-1][0].conv.out_channels
            except Exception:
                in_ch = 160
            self.backbone = nn.Sequential(
                net.conv_stem, net.bn1, net.act1, net.blocks)
        else:
            # torchvision 通用骨干：resnet 系手工拆层，
            # 其余（vgg/densenet/mobilenet/efficientnet/convnext/shufflenet）用 .features
            self.backbone, in_ch = _build_tv_backbone(backbone, pretrained)
        self.head = DensityHead(in_ch)

    def forward(self, x):
        return self.head(self.backbone(x))


def _build_tv_backbone(name, pretrained):
    """torchvision 分类模型 -> (特征骨干 nn.Module, 输出通道数)"""
    import torchvision.models as tvm
    if not hasattr(tvm, name):
        raise ValueError(
            f"未知计数骨干: {name}（支持 ghostnet 系列 / torchvision 分类模型）")
    try:
        m = getattr(tvm, name)(weights="DEFAULT" if pretrained else None)
    except (TypeError, AttributeError):
        m = getattr(tvm, name)(pretrained=pretrained)
    if name.startswith(("resnet", "wide_resnet", "resnext")):
        feat = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                             m.layer1, m.layer2, m.layer3, m.layer4)
    elif hasattr(m, "features"):
        feat = m.features
    else:
        raise ValueError(f"模型 {name} 无可提取的特征部分")
    with torch.no_grad():
        out = feat(torch.zeros(1, 3, 64, 64))
    return feat, out.shape[1]


# torchvision 可用计数骨干清单（与 GhostNet 系列一起构成全部选项）
TV_CROWD_BACKBONES = [
    "resnet18", "resnet34", "resnet50", "resnet101", "wide_resnet50_2",
    "vgg16", "densenet121", "mobilenet_v2", "mobilenet_v3_large",
    "efficientnet_b0", "efficientnet_b3", "efficientnet_v2_s",
    "convnext_tiny", "convnext_small", "shufflenet_v2_x1_0",
]


# 参考项目模型注册名（vendor 于 nets/crowd/，与参考 load_model.py 同名）
REFERENCE_CROWD_MODELS = ["crowdvgg", "crowdresnet"]


def _build_crowd_model(name, pretrained=True):
    """构建密度网：内置 CrowdNet / 参考项目模型 / 自定义密度网。

    forward(x) 输出 [N,1,H',W']，与 GT 尺寸不一致时由训练管线双线性对齐。
    """
    from app.core.registry import is_custom_model_for, get_custom_model
    if name in REFERENCE_CROWD_MODELS:
        from nets.crowd import build_reference_crowd_model
        return build_reference_crowd_model(name, pretrained=pretrained)
    if is_custom_model_for(name, "crowd_density"):
        return get_custom_model(name, 1, pretrained=pretrained,
                                task="crowd_density")
    return CrowdNet(name, pretrained=pretrained)


def list_models():
    """pretrained 指骨干 ImageNet 权重（GhostNetV2 为本地离线，其余在线）"""
    out = []
    out.append({"name": "ghostnetv2_10", "desc": "GhostNetV2(1.0) 密度网（本地预训练骨干）",
                "type": "builtin", "pretrained": True, "pretrained_local": True,
                "need_weight": True})
    out.append({"name": "ghostnetv2_13", "desc": "GhostNetV2(1.3) 密度网（本地预训练骨干）",
                "type": "builtin", "pretrained": True, "pretrained_local": True,
                "need_weight": True})
    out.append({"name": "ghostnetv2_16", "desc": "GhostNetV2(1.6) 密度网（本地预训练骨干）",
                "type": "builtin", "pretrained": True, "pretrained_local": True,
                "need_weight": True})
    out.append({"name": "ghostnet", "desc": "GhostNet v1 密度网（无预训练）",
                "type": "builtin", "pretrained": False, "pretrained_local": False,
                "need_weight": True})
    # 参考项目模型（vendor 于 nets/crowd/，骨干 ImageNet 预训练在线下载）
    for n in REFERENCE_CROWD_MODELS:
        out.append({"name": n,
                    "desc": (f"参考项目密度网 {n}（VGG16+FPN 多尺度解码）"
                             if n == "crowdvgg" else
                             "参考项目密度网 crowdresnet（ResNet50-IBN-a + 密度头）"),
                    "type": "reference", "pretrained": True,
                    "pretrained_local": False, "need_weight": True})
    for n in TV_CROWD_BACKBONES:
        out.append({"name": n, "desc": f"torchvision {n} 密度网（ImageNet 预训练骨干）",
                    "type": "torchvision", "pretrained": True,
                    "pretrained_local": False, "need_weight": True})
    # P2PNet：基于点预测的人群计数（vendor 于 nets/p2pnet/）
    from config import WEIGHTS_DIR as _WD
    p2p_ckpt = _WD / "p2pnet" / "SHTechA.pth"
    out.append({"name": "p2pnet",
                "desc": "P2PNet 点预测网（VGG16_bn+FPN，点回归/分类；"
                        f"{'含 SHTechA 预训练检查点，离线可推理' if p2p_ckpt.is_file() else '训练后使用'}）",
                "type": "p2pnet", "pretrained": True,
                "pretrained_local": p2p_ckpt.is_file(),
                "need_weight": not p2p_ckpt.is_file()})
    # P2RLoss：CVPR 2025 半监督点预测（vendor 于 nets/p2r/，教师-学生 +
    # Point-to-Region 损失）；type 沿用 p2pnet（同属点预测路线）
    out.append({"name": "p2r",
                "desc": "P2R 点预测网（CVPR 2025 半监督，VGG16-BN 教师-学生 + "
                        "Point-to-Region 损失；label_ratio<1 时仅少量标注即可"
                        "训练，训练后使用）",
                "type": "p2pnet", "pretrained": True,
                "pretrained_local": False, "need_weight": True})
    # Consistent-Point：半监督点预测（vendor 于 nets/cpoint/，P2PNet 基座 +
    # PA 位置聚合 + IUC 不确定性校准 + Mean-Teacher）
    out.append({"name": "cp",
                "desc": "Consistent-Point 点预测网（arXiv 2503.12441 半监督，"
                        "P2PNet 基座 + PA 位置聚合 + IUC 不确定性校准 + "
                        "Mean-Teacher；label_ratio<1 时仅少量标注即可训练，"
                        "训练后使用）",
                "type": "p2pnet", "pretrained": True,
                "pretrained_local": False, "need_weight": True})
    # CU：半监督点预测（vendor 于 nets/cu/，ICCV 2023：P2PNet 基座 + patch 级
    # 不确定性分支 + PatchBank ASM 代理真值 + 排序损失 + 课程伪标签过滤）
    out.append({"name": "cu",
                "desc": "CU 点预测网（ICCV 2023 半监督，P2PNet 基座 + 不确定性"
                        "分支 + PatchBank ASM + 排序损失 + 课程伪标签过滤 + "
                        "Mean-Teacher；label_ratio<1 时仅少量标注即可训练，"
                        "训练后使用）",
                "type": "p2pnet", "pretrained": True,
                "pretrained_local": False, "need_weight": True})
    # PET：ICCV 2023 Point-Query Quadtree（vendor 于 nets/pet/）
    pet_ckpt = _WD / "pet" / "SHA_model.pth"
    out.append({"name": "pet",
                "desc": "PET 点预测网（ICCV 2023 点查询四叉树；"
                        f"{'含 SHA 预训练权重，离线可推理' if pet_ckpt.is_file() else '训练后使用'}）",
                "type": "p2pnet", "pretrained": True,
                "pretrained_local": pet_ckpt.is_file(),
                "need_weight": not pet_ckpt.is_file()})
    # CLTR：ECCV 2022 Crowd Localization Transformer（vendor 于 nets/cltr/）
    out.append({"name": "cltr",
                "desc": "CLTR 人群定位 Transformer（ECCV 2022 Conditional "
                        "DETR + ResNet50，训练/上传权重后使用）",
                "type": "p2pnet", "pretrained": False,
                "pretrained_local": False, "need_weight": True})
    # RAZ（Recurrent Attentive Zooming）：Main-Net 顺序训练密度/定位/
    # 注意力分支，再用 zoom crops 训练 RAZ-Net；单权重同时保存 Main/RAZ。
    out.append({"name": "raz",
                "desc": "RAZ 人群计数与精确定位（Main-Net 密度/定位/注意力 "
                        "+ RAZ-Net 循环局部放大；训练后使用 raz_best.pth）",
                "type": "p2pnet", "pretrained": True,
                "pretrained_local": False, "need_weight": True})
    out.append({"name": "d2c",
                "desc": "D2C 解耦两阶段人群计数（PMR 概率图 + CMR 计数图 + "
                        "LCG-PPD 定位；训练后使用 d2c_best.pth）",
                "type": "p2pnet", "pretrained": True,
                "pretrained_local": False, "need_weight": True})
    # 自定义模型清单 /tasks 按 tasks 过滤后追加（密度图/点预测/头部框）
    return out


# ------------------------- 推理模型缓存 + 权重加载校验 -------------------------

# 推理模型 LRU 缓存：避免每次推理 / 每帧摄像头都重建模型并重读权重
# （P2PNet 骨干为 528MB 的 vgg16_bn，逐帧重建会使摄像头帧率跌至秒级）
_MODEL_CACHE = {}                    # key -> nn.Module（eval 态）
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE_MAX = 4                 # 最多同时驻留的 (模型, 权重, 设备) 组合数


def _model_cache_key(model_name, weight_path, device):
    """缓存键：权重文件带 mtime——同路径权重被重新训练覆盖后自动失效"""
    key = [model_name, str(weight_path or ""), str(device)]
    if weight_path:
        try:
            key.append(Path(weight_path).stat().st_mtime)
        except OSError:
            key.append("missing")
    return tuple(key)


def _model_cache_get(key):
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.pop(key, None)
        if model is not None:
            _MODEL_CACHE[key] = model   # 重新插入 = 移到最近使用端
        return model


def _model_cache_put(key, model):
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[key] = model
        while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))   # 淘汰最久未用


def clear_inference_cache():
    """清空推理模型缓存（各训练管线启动时调用），返回清掉的模型数。

    训练是本框架的内存峰值场景（模型 + 优化器状态约 3 倍参数量 + 数据
    增强缓冲），缓存驻留的推理模型（LRU=4，P2PNet 级可达 GB）应让位；
    摄像头/推理按需重建（正在使用的会话持有自身引用，不受清空影响）。
    """
    with _MODEL_CACHE_LOCK:
        n = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
    return n


def _load_weights_compat(model, weight_path, *, context="权重"):
    """加载权重并校验匹配度（H4：strict=False 会静默吞掉架构错配）。

    从 checkpoint 提取 state_dict（兼容 model_state_dict / state_dict /
    "model" 三种键），剥掉 DP/DDP 的 module. 前缀后加载；若实际匹配的参数
    不足模型参数的一半则判定为架构不匹配并抛错。
    """
    ck = torch.load(weight_path, map_location="cpu")
    sd = None
    if isinstance(ck, dict):
        sd = ck.get("model_state_dict") or ck.get("state_dict")
        if sd is None and isinstance(ck.get("model"), dict):
            sd = ck["model"]
    if not isinstance(sd, dict) or not sd:
        sd = ck if isinstance(ck, dict) else None
        if not sd:
            raise ValueError(f"{context}文件中未找到模型参数: {weight_path}")
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    total_model = len(model.state_dict())
    result = model.load_state_dict(sd, strict=False)
    matched = len(sd) - len(result.unexpected_keys)
    # 阈值 95%：同架构 checkpoint 应 100% 匹配；近亲架构（如 ghostnet v1
    # 权重配 v2 模型）键名大量重合，过低的阈值拦不住，输出看似正常实则无意义
    if matched == 0 or matched < total_model * 0.95:
        raise ValueError(
            f"{context}与模型架构不匹配（{matched}/{total_model} 个参数键命中，"
            f"missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}）: {weight_path}")


# ------------------------- 密度图生成 -------------------------

def gaussian_kernel_density(shape, points, sigma=8.0, adaptive=False):
    """点标注 -> 密度图 numpy [H,W]。sum(密度图)=点数"""
    h, w = shape
    density = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return density
    pts = np.asarray(points, dtype=np.float32)
    for p in pts:
        x, y = int(round(p[0])), int(round(p[1]))
        if not (0 <= x < w and 0 <= y < h):
            continue
        if adaptive and len(pts) > 1:
            d = np.sqrt(((pts - p) ** 2).sum(1))
            d.sort()
            sigma_i = max(2.0, d[1] * 0.3) if len(d) > 1 else sigma
        else:
            sigma_i = sigma
        r = int(3 * sigma_i)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1) - x
        ys = np.arange(y0, y1) - y
        g = np.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) / (2 * sigma_i ** 2))
        g /= g.sum()
        density[y0:y1, x0:x1] += g
    return density


def _eval_from_test(root, factory):
    """验证集优先使用数据集自带的 Test/Val 划分（factory 构建评估模式数据集）。

    返回 (dataset, 来源说明)；未找到/不可用时返回 (None, 原因)，由调用方
    回退到「训练集按 val_ratio 切分」。
    """
    test_img, test_gt = _crowd_dirs(Path(root), False)
    if test_img is None or test_gt is None:
        return None, "未找到 Test/Val 划分，按训练集比例切分"
    try:
        ds = factory()
    except Exception as e:
        return None, f"Test/Val 划分不可用（{e}），按训练集比例切分"
    if len(ds) == 0:
        return None, "Test/Val 划分为空，按训练集比例切分"
    return ds, f"数据集 Test/Val 划分（{len(ds)} 张，完整划分）"


def _crowd_dirs(root: Path, train: bool):
    """定位人群计数数据目录。

    两种布局：
    1. Train/Test 子目录：root/Train/Images + root/Train/GT（GT 可为 .npy/.txt）
       （目录名兼容 Train/train、Test/test、Val/val、Images/images/imgs、
        GT/gt/npys/masks/labels）
    2. 扁平结构：root/imgs + root/gt(.txt) 或 root/npys(.npy)
    返回 (img_dir, gt_dir 或 None)；gt_dir=None 表示扁平模式（gt/npys 分别查找）
    """
    root = Path(root)
    marker = root / ".unify_incomplete"
    if marker.exists():
        raise ValueError(
            f"数据集上次统一转换未完成（发现 {marker}）。为避免使用不完整"
            f"的 Train/Test，请重新执行统一转换（重新上传或下载该数据集）"
            f"后再训练；原始数据未被清理。路径: {root}")

    def pick_dir(base, names):
        for n in names:
            d = base / n
            if d.exists():
                return d
        return None

    split_names = ["Train", "train"] if train else ["Test", "test", "Val", "val"]
    for split in split_names:
        sub = root / split
        if not sub.is_dir():
            continue
        img_dir = pick_dir(sub, ["Images", "images", "imgs", "img"])
        gt_dir = pick_dir(sub, ["GT", "gt", "npys", "masks", "labels"])
        if img_dir is not None and gt_dir is not None:
            return img_dir, gt_dir
    # 扁平结构
    img_dir = pick_dir(root, ["imgs", "images", "Images", "img"])
    if img_dir is None and any(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                               for p in root.iterdir() if p.is_file()):
        img_dir = root
    if img_dir is None:
        return None, None
    gt_dir = pick_dir(root, ["GT", "gt", "npys", "masks", "labels"])
    return img_dir, gt_dir


def _resize_density(density, nw, nh):
    """密度图重采样：双线性 + 总量校准。

    PIL resize 默认 BICUBIC，不保总且过冲可产生负值——密度图的总量即人数，
    重采样后必须按源总量缩放校准。
    """
    if density.ndim == 3:
        density = density[:, :, 0]   # 兼容 (H,W,1) 存储
    out = np.array(Image.fromarray(density.astype(np.float32)).resize(
        (nw, nh), Image.BILINEAR), dtype=np.float32)
    s_src, s_out = float(density.sum()), float(out.sum())
    if s_src > 1e-8 and s_out > 1e-8:
        out *= s_src / s_out
    return out


def _fit_min_side(img, density, min_side):
    """保证图像与密度图短边 >= min_side（不足则等比放大）。

    密度图训练采用「多尺度缩放 + 随机裁剪」：若某张图（含缩放后）在任一维
    小于裁剪尺寸，旧逻辑会跳过裁剪、保持原尺寸，导致同一 batch 内样本尺寸
    不一致（默认 collate 的 torch.stack 报
    "stack expects each tensor to be equal size"）。这里在裁剪前统一把短边
    补足到裁剪尺寸，配合随后的定尺裁剪，保证每个样本恒为
    min_side×min_side。
    """
    W, H = img.size
    if H >= min_side and W >= min_side:
        return img, density, W, H
    scale = max(min_side / max(1, H), min_side / max(1, W))
    nw = max(min_side, int(round(W * scale)))
    nh = max(min_side, int(round(H * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    density = _resize_density(density, nw, nh)
    return img, density, nw, nh


class CrowdDataset(Dataset):
    """密度图训练数据集，支持扩展数据增强策略。

    增强管线（train=True 且 augment=True 时按序执行，全部受 augment_strength
    [0, 2] 缩放——空间变换同步图像与密度图，光度变换仅图像）：
      1. 多尺度缩放（Multi-Scale）: 随机 resize 到 [1-0.25s, 1+0.25s]
      2. 色彩抖动（Color Jitter）: brightness/contrast/saturation/hue
      3. 随机裁剪: 固定 crop_size
      4. 随机水平翻转: p = 0.5·min(s,1.5)
      5. 随机垂直翻转: p = 0.25·min(s,1.5)
      6. 高斯噪声: σ = 0.01·s
      7. 随机擦除（Cutout）: p = 0.1·s, 擦除区域 5%~15%
    """
    def __init__(self, root, crop_size=384, sigma=8.0, adaptive=False,
                 train=True, augment=True, augment_strength=1.0):
        self.root = Path(root)
        self.crop = max(16, int(crop_size or 384))
        self.sigma = sigma
        self.adaptive = adaptive
        self.train = train
        self.augment = augment and augment_strength > 0
        self.aug_strength = max(0.0, min(2.0, augment_strength))
        img_dir, gt_dir = _crowd_dirs(self.root, train)
        if img_dir is None:
            raise ValueError(
                f"未找到图像目录。支持 root/Train/Images+GT 结构或 root/imgs 扁平结构。"
                f"路径: {root}")
        self.gt_dir = gt_dir
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.items = []
        for p in sorted(img_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            if gt_dir is not None:
                # Train/Test 结构：GT 目录内 .npy（密度图）或 .txt（点标注）
                npy = gt_dir / (p.stem + ".npy")
                txt = gt_dir / (p.stem + ".txt")
                if npy.exists():
                    self.items.append((p, "npy"))
                elif txt.exists():
                    self.items.append((p, "txt"))
            else:
                # 扁平结构：gt/ 或 npys/ 目录
                if (self.root / "gt" / (p.stem + ".txt")).exists():
                    self.items.append((p, "txt"))
                elif (self.root / "npys" / (p.stem + ".npy")).exists():
                    self.items.append((p, "npy"))
        if not self.items:
            raise ValueError(
                f"未找到有效样本。需要 Train/Images+GT(或 imgs+gt/npys)。路径: {root}")

    def __len__(self):
        return len(self.items)

    def _gt_path(self, img_path, suffix):
        """GT 文件定位：优先 gt_dir（Train/Test 结构），否则扁平的 gt//npys/"""
        if self.gt_dir is not None:
            p = self.gt_dir / (img_path.stem + suffix)
            if p.exists():
                return p
        for d in ("gt", "npys"):
            p = self.root / d / (img_path.stem + suffix)
            if p.exists():
                return p
        return None

    def __getitem__(self, idx):
        import random as _rnd
        img_path, gt_type = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        if gt_type == "txt":
            txt = self._gt_path(img_path, ".txt")
            pts = []
            for line in txt.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line:
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 2:
                        try:
                            pts.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            continue
            density = gaussian_kernel_density((H, W), pts, self.sigma, self.adaptive)
        else:
            density = np.load(self._gt_path(img_path, ".npy"))
            if density.ndim == 3:
                density = density[:, :, 0]   # (H,W,1) → (H,W)
            if density.shape != (H, W):
                density = _resize_density(density, W, H)

        # ---- 训练时数据增强（augment_strength 全局缩放） ----
        if self.train and self.augment:
            s = self.aug_strength
            # 1) 多尺度缩放（图像 + 密度图同步，密度总量保持）
            if s > 0:
                scale = _rnd.uniform(max(0.5, 1 - 0.25 * s),
                                     min(1.5, 1 + 0.25 * s))
                if abs(scale - 1.0) > 0.02:
                    nw, nh = max(64, int(W * scale)), max(64, int(H * scale))
                    img = img.resize((nw, nh), Image.BILINEAR)
                    density = _resize_density(density, nw, nh)
                    W, H = nw, nh

        # 统一短边 >= crop（原图偏小或多尺度缩小后）：保证训练裁剪恒为
        # crop×crop，否则同 batch 尺寸不一致会在 collate 的 torch.stack 报
        # "stack expects each tensor to be equal size"
        if self.train:
            img, density, W, H = _fit_min_side(img, density, self.crop)

        img_t = transforms.functional.to_tensor(img)
        img_t = transforms.functional.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)

        if self.train and self.augment:
            s = self.aug_strength
            # 2) 色彩抖动（仅图像；概率 = min(0.8, 0.5·s)）
            if _rnd.random() < min(0.8, 0.5 * s):
                b = 1 + _rnd.uniform(-0.3, 0.3) * s
                c = 1 + _rnd.uniform(-0.3, 0.3) * s
                sat = 1 + _rnd.uniform(-0.2, 0.2) * s
                hue = _rnd.uniform(-0.05, 0.05) * s
                img_t = transforms.functional.adjust_brightness(img_t, max(0.1, b))
                img_t = transforms.functional.adjust_contrast(img_t, max(0.1, c))
                img_t = transforms.functional.adjust_saturation(img_t, max(0.1, sat))
                img_t = transforms.functional.adjust_hue(img_t, max(-0.5, min(0.5, hue)))

            # 3) 随机裁剪（已保证 H,W >= crop，恒裁剪为 crop×crop）
            ch, cw = self.crop, self.crop
            y0 = np.random.randint(0, H - ch + 1)
            x0 = np.random.randint(0, W - cw + 1)
            img_t = img_t[:, y0:y0 + ch, x0:x0 + cw]
            density = density[y0:y0 + ch, x0:x0 + cw]
            # 4) 随机水平翻转
            if np.random.rand() < 0.5 * min(s, 1.5):
                img_t = torch.flip(img_t, [2])
                density = np.ascontiguousarray(density[:, ::-1])
            # 5) 随机垂直翻转（较低概率）
            if np.random.rand() < 0.25 * min(s, 1.5):
                img_t = torch.flip(img_t, [1])
                density = np.ascontiguousarray(density[::-1, :])
            # 6) 高斯噪声（仅图像）
            if _rnd.random() < 0.3 * s:
                img_t = img_t + torch.randn_like(img_t) * (0.01 * s)
            # 7) 随机擦除 / Cutout（图像与密度图同步清零）
            if _rnd.random() < 0.1 * s and img_t.shape[1] > 32 and img_t.shape[2] > 32:
                hh, ww = img_t.shape[1], img_t.shape[2]
                eh = max(8, int(hh * _rnd.uniform(0.05, 0.15)))
                ew = max(8, int(ww * _rnd.uniform(0.05, 0.15)))
                ey = _rnd.randint(0, hh - eh)
                ex = _rnd.randint(0, ww - ew)
                img_t[:, ey:ey + eh, ex:ex + ew] = 0
                density[ey:ey + eh, ex:ex + ew] = 0
        elif self.train:
            # 增强关闭时仅保留基础随机裁剪（同样恒为 crop×crop）
            ch, cw = self.crop, self.crop
            y0 = np.random.randint(0, H - ch + 1)
            x0 = np.random.randint(0, W - cw + 1)
            img_t = img_t[:, y0:y0 + ch, x0:x0 + cw]
            density = density[y0:y0 + ch, x0:x0 + cw]
        else:
            # 验证：限制最大边长，避免显存溢出
            max_side = 1536
            if max(H, W) > max_side:
                scale = max_side / max(H, W)
                nh, nw = int(H * scale), int(W * scale)
                img_t = torch.nn.functional.interpolate(
                    img_t.unsqueeze(0), size=(nh, nw), mode="bilinear",
                    align_corners=False)[0]
                density = _resize_density(density, nw, nh)

        # 密度图下采样到 1/8（与模型输出对齐）
        d = torch.from_numpy(density).float().unsqueeze(0).unsqueeze(0)
        d = F.avg_pool2d(d, 8) * 64.0  # 面积不变：均值池化 * 64 保持总和
        d = d[0, 0]
        return img_t, d


# ------------------------- .mat -> GT 密度图工具 -------------------------

def _dig_ann_points(node, depth=0, acc=None):
    """递归收集 image_info 嵌套结构中的 (N,2) 数值数组候选。

    下钻三类容器：object 数组（MATLAB cell）、结构化数组/void 标量
    （MATLAB struct，按字段展开）、普通嵌套 ndarray。
    img_size 是 (1,2) 小数组，取行数最多者即点标注。
    """
    if acc is None:
        acc = []
    if depth > 8:
        return acc
    try:
        arr = np.asarray(node)
    except Exception:
        return acc
    if arr.dtype == object:
        for item in arr.ravel():
            _dig_ann_points(item, depth + 1, acc)
    elif getattr(arr.dtype, "names", None):
        # MATLAB struct（结构化数组 / void 标量）：按字段名展开递归
        for name in arr.dtype.names:
            try:
                subs = np.atleast_1d(arr[name]).ravel()
            except Exception:
                continue
            for sub in subs:
                _dig_ann_points(sub, depth + 1, acc)
    elif arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1 and \
            np.issubdtype(arr.dtype, np.number):
        acc.append(arr.astype(np.float32))
    return acc


def _pick_ann_points(node):
    cands = _dig_ann_points(node)
    if not cands:
        return None
    # 行数最多的候选作为 annPoints（单点场景按首个兜底）
    return max(cands, key=lambda a: a.shape[0])


def _dig_img_size(node, depth=0):
    """在嵌套结构中查找 img_size：形状 (1,2)/(2,) 的小数值数组（H,W）"""
    if depth > 8:
        return None
    try:
        arr = np.asarray(node)
    except Exception:
        return None
    if arr.dtype == object:
        for item in arr.ravel():
            r = _dig_img_size(item, depth + 1)
            if r is not None:
                return r
        return None
    if getattr(arr.dtype, "names", None):
        # MATLAB struct：优先 height/width 字段精确读取；
        # 盲挖时跳过 (N,2) 点标注数组（防止单点 location 被误判为尺寸）
        names = arr.dtype.names
        if "height" in names and "width" in names:
            try:
                h = float(np.asarray(arr["height"]).ravel()[0])
                w = float(np.asarray(arr["width"]).ravel()[0])
                if 0 < h < 50000 and 0 < w < 50000:
                    return int(h), int(w)
            except Exception:
                pass
        for name in names:
            try:
                subs = np.atleast_1d(arr[name]).ravel()
            except Exception:
                continue
            for sub in subs:
                try:
                    sub_arr = np.asarray(sub)
                except Exception:
                    continue
                if sub_arr.ndim == 2 and sub_arr.shape[1] == 2 and \
                        sub_arr.shape[0] >= 1:
                    continue   # 点标注数组（location 等）
                r = _dig_img_size(sub, depth + 1)
                if r is not None:
                    return r
        return None
    flat = np.asarray(arr).ravel()
    if flat.size == 2 and np.issubdtype(flat.dtype, np.number) and \
            flat.min() > 0 and flat.max() < 50000:
        return tuple(int(v) for v in flat)
    return None


def load_mat_points(mat_path):
    """读取 .mat 点标注，自动探测格式。

    - annPoints 根键（QNRF / NWPU / UCF_CC_50）
    - image_info 结构（ShanghaiTech A/B）：递归解析嵌套层级，
      兼容不同来源 .mat 的层级差异
    返回 (points[N,2], (H, W) 或 None)
    """
    from scipy.io import loadmat
    mat = loadmat(str(mat_path))
    if "annPoints" in mat:
        raw = np.asarray(mat["annPoints"])
        if raw.dtype == object:
            raw = np.asarray(raw.tolist())
        return raw.astype(np.float32).reshape(-1, 2), None
    if "image_info" in mat:
        pts = _pick_ann_points(mat["image_info"])
        if pts is None:
            raise ValueError(
                f"image_info 结构中未找到 (N,2) 点标注数组: {mat_path}")
        size = _dig_img_size(mat["image_info"])
        return pts, size
    raise ValueError(
        f"不支持的 .mat 格式（键: {[k for k in mat if not k.startswith('__')]}），"
        f"支持 annPoints 或 image_info（ShanghaiTech）结构")


def _gaussian_kernel2d(sigma: float):
    """归一化 2D 高斯核（截断半径 3σ，奇数边长；中心和为 1）"""
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    s = k.sum()
    if s <= 0:
        k = np.zeros_like(k)
        k[radius, radius] = 1.0
    else:
        k /= s
    return k


def make_density_map(h, w, points, sigma=2.0, adaptive=False):
    """生成密度图：每个高斯核质量≈1，sum(density)≈点数

    adaptive=True 时用 k 近邻自适应 sigma。按点局部贴片叠加归一化小核
    （与逐点整幅 gaussian_filter 数值等价），复杂度 O(点数×核面积)。
    """
    density = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return density
    pts = np.asarray(points, dtype=np.float64)
    kernels = {}   # sigma → 核缓存（自适应模式下复用）
    for pt in pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < w and 0 <= y < h):
            continue
        if adaptive and len(pts) > 3:
            d = np.sqrt(((pts - pt) ** 2).sum(1))
            d.sort()
            sigma_i = max(1.0, (d[1] + d[2] + d[3]) * 0.1)
        else:
            sigma_i = sigma
        key = round(float(sigma_i), 3)
        if key not in kernels:
            kernels[key] = _gaussian_kernel2d(sigma_i)
        k = kernels[key]
        r = k.shape[0] // 2
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        kx0 = x0 - (x - r)
        ky0 = y0 - (y - r)
        patch = k[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]
        # 边界点的高斯核被图像范围裁剪后会丢质量，逐点重归一化保证
        # 每个点贡献的质量恒为 1（sum(密度图) ≈ 点数）
        ps = float(patch.sum())
        if ps <= 0:
            continue
        density[y0:y1, x0:x1] += patch / ps
    return density


# .mat 标签文件常见的标记性前缀/后缀（大小写不敏感剥离；
# 仅匹配带分隔符的形式，避免误伤 gtxxx 一类普通文件名）
_MAT_MARK_PREFIXES = ("gt_", "gt-", "groundtruth_", "ground_truth_",
                      "ground-truth_", "label_", "labels_", "label-",
                      "ann_", "annotation_", "annotations_")
_MAT_MARK_SUFFIXES = ("_ann", "_gt", "_label", "_labels", "_mat",
                      "_annotation", "_annotations", "_groundtruth")


def _norm_mat_stem(stem: str) -> str:
    """剥离 .mat 标签文件常见的标记性前缀/后缀（可叠加多层）。

    例：GT_IMG_1 → IMG_1；img_x_ann → img_x；GT_img_1_ann → img_1
    """
    s = stem
    changed = True
    while changed and s:
        changed = False
        low = s.lower()
        for p in _MAT_MARK_PREFIXES:
            if low.startswith(p) and len(s) > len(p):
                s = s[len(p):]
                changed = True
                break
        for x in _MAT_MARK_SUFFIXES:
            if low.endswith(x) and len(s) > len(x):
                s = s[: -len(x)]
                changed = True
                break
    return s


def _stem_tokens(stem: str):
    """文件主名按分隔符切分为 token（小写；IMG_1 → ['img','1']）"""
    return [t.lower() for t in re.split(r"[_\-.\s]+", stem) if t]


def _tokens_align(mat_tokens, img_tokens):
    """图像 token 序列是标签 token 序列的头部/尾部连续子列则视为同一文件。

    前缀型：GT_IMG_1.mat ↔ IMG_1.jpg（['gt','img','1'] 尾部含 ['img','1']）
    后缀型：img_1_ann.mat ↔ img_1.jpg（['img','1','ann'] 头部含 ['img','1']）
    token 级比较避免 IMG_1 误配 IMG_11。返回多出的标记 token 数（用于排序），
    不匹配返回 None。
    """
    n, m = len(mat_tokens), len(img_tokens)
    if m == 0 or m > n:
        return None
    if mat_tokens[:m] == img_tokens or mat_tokens[n - m:] == img_tokens:
        return n - m
    return None


def _find_image_for_mat(mat_path: Path, img_dir: Path):
    """按命名规则为 .mat 定位图像（标签文件可能带任意标记前缀/后缀）。

    匹配策略（依次，搜索目录：img_dir（可选）→ .mat 同级 → 上级，各自递归）：
    1. 归一化同名：剥离常见标记后同名（GT_IMG_x.mat ↔ IMG_x.jpg、
       img_x_ann.mat ↔ img_x.jpg、0001.mat ↔ 0001.jpg）
    2. 原始 stem 同名（标签本身无标记）
    3. token 级对齐：图像主名 token 是标签 token 的头部/尾部连续子列
      （任意前缀如 ann_IMG_1.mat / label_IMG_1.mat ↔ IMG_1.jpg；
       多候选时取标记 token 最少、路径字典序最小者）
    """
    import glob as _glob
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG")
    dirs = [d for d in ([img_dir] if img_dir else []) +
            [mat_path.parent, mat_path.parent.parent]
            if d is not None and d.exists()]
    stem_raw = mat_path.stem
    stem_norm = _norm_mat_stem(stem_raw)

    def _rglob_images(d: Path, pattern: str):
        try:
            return [p for p in sorted(d.rglob(pattern))
                    if p.suffix in exts and p.is_file()]
        except (OSError, ValueError):
            return []

    # 1) 归一化同名精确匹配
    if stem_norm:
        for d in dirs:
            hits = _rglob_images(d, _glob.escape(stem_norm) + ".*")
            if hits:
                return hits[0]
    # 2) 原始 stem 匹配（标签无标记 / 归一化未命中时兜底）
    if stem_raw != stem_norm:
        for d in dirs:
            hits = _rglob_images(d, _glob.escape(stem_raw) + ".*")
            if hits:
                return hits[0]
    # 3) token 级前缀/后缀对齐（任意标记，大小写不敏感）
    mat_toks = _stem_tokens(stem_raw)
    if mat_toks:
        cands = []
        for d in dirs:
            for p in sorted(d.rglob("*")):
                if p.suffix not in exts or not p.is_file():
                    continue
                extra = _tokens_align(mat_toks, _stem_tokens(p.stem))
                if extra is not None:
                    cands.append((extra, p))
        if cands:
            cands.sort(key=lambda x: (x[0], str(x[1])))
            return cands[0][1]
    return None


def run_mat2density(task, params: dict):
    """后台任务：.mat 点标注 -> GT 密度图 .npy（sum≈人数，尺寸=原图）"""
    from PIL import Image
    mat_dir = Path(params["mat_dir"])
    img_dir = Path(params["img_dir"]) if params.get("img_dir") else None
    save_dir = Path(params["save_dir"])
    sigma = float(params.get("sigma", 2.0))
    adaptive = bool(params.get("adaptive", False))
    save_dir.mkdir(parents=True, exist_ok=True)

    mats = sorted(p for p in mat_dir.rglob("*.mat"))
    if not mats:
        raise ValueError(f"目录中无 .mat 文件: {mat_dir}")
    task.total_epochs = len(mats)
    task.log(f"待转换 .mat: {len(mats)} 个 | sigma={sigma} | "
             f"自适应={adaptive} | 输出: {save_dir}")

    ok, skipped, failed = 0, 0, 0
    first_error = ""
    for i, mp in enumerate(mats, 1):
        if task.stop_event.is_set():
            raise StopIteration()
        task.epoch = i
        task.progress = i / len(mats) * 100.0
        try:
            # 先定位图像：输出 .npy 以图像主名命名（与训练配对规则一致；
            # 标签带任意标记前缀/后缀也能正确对齐，如 GT_IMG_1.mat ↔ IMG_1.jpg）
            img_path = _find_image_for_mat(mp, img_dir)
            base = img_path.stem if img_path is not None \
                else _norm_mat_stem(mp.stem)
            out = save_dir / (base + ".npy")
            if out.exists():
                # 增量跳过：源文件（.mat/图像）更新过则重新生成
                src_mt = mp.stat().st_mtime
                if img_path is not None and img_path.exists():
                    src_mt = max(src_mt, img_path.stat().st_mtime)
                if out.stat().st_mtime >= src_mt:
                    skipped += 1
                    continue
            points, mat_size = load_mat_points(mp)
            if img_path is not None:
                with Image.open(img_path) as im:
                    w, h = im.size
            elif mat_size is not None:
                h, w = mat_size  # image_info 的 img_size 为 [H, W]
            else:
                raise ValueError("无法定位同名图像，且 .mat 内无尺寸信息，"
                                 "请提供图像目录")
            density = make_density_map(h, w, points, sigma, adaptive)
            np.save(out, density)
            ok += 1
            if i % 10 == 0 or i == len(mats):
                task.log(f"[{i}/{len(mats)}] {mp.name} -> {out.name} "
                         f"(点数≈{len(points)})")
        except Exception as e:
            failed += 1
            if not first_error:
                first_error = f"{mp.name}: {e}"
            task.log(f"[{i}/{len(mats)}] 失败 {mp.name}: {e}")

    task.result = {"converted": ok, "skipped": skipped, "failed": failed,
                   "save_dir": str(save_dir)}
    task.log(f"转换完成：成功 {ok}，跳过 {skipped}，失败 {failed}")
    if ok == 0 and skipped == 0 and failed > 0:
        raise ValueError(
            f"全部转换失败（{failed} 个），首个错误：{first_error}"
            f"——请检查 .mat 格式与图像路径（详见任务日志）")


def _reference_overlay(img_bgr, density, alpha=0.67):
    """密度热力图与原图融合（vis_img_den.py save_density_map 移植）：
    密度归一化 → 反转 → JET 着色 → (1-alpha)*img + alpha*density → 重归一化"""
    import cv2
    pred_density = density / max(float(density.max()), 1e-8)
    pred_density_write = 1. - pred_density
    pred_density_write = cv2.applyColorMap(
        np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)
    pred_density_write = pred_density_write / 255

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255
    # 权重舍入到 10 位小数：alpha=0.67 时与参考字面量 0.33/0.67 的浮点值
    # 完全一致（否则 1-0.67=0.32999... 在整数边界截断会产生 ±1 像素差）
    w_img = round(1.0 - alpha, 10)
    w_den = round(alpha, 10)
    heatmap_pred = w_img * img + w_den * pred_density_write
    heatmap_pred = heatmap_pred / max(float(heatmap_pred.max()), 1e-8)
    query_img = cv2.cvtColor((heatmap_pred * 255).astype(np.uint8),
                             cv2.COLOR_BGR2RGB)
    return query_img


def run_npy2png(task, params: dict):
    """后台任务：.npy 密度图 → PNG 可视化（批量，保存到指定目录）

    mode:
    - heat（默认）：伪彩色热力图 jet，与推理热力图一致
    - gray：灰度
    - overlay：**密度图与原图按比例融合**（参考 CrowdCounting-framework-PyTorch
      vis_img_den.py 的 save_density_map：jet 热力图 + 原图加权叠加），
      需提供原图目录 img_dir（按 .npy 主名自动定位同名/前缀匹配图像），
      alpha 为密度权重（0=纯原图，1=纯热力图，默认 0.67）
    annotate: 左上角标注 count≈密度总和；输出 <stem>.png 与 .npy 同名
    """
    import matplotlib.cm as cm
    from PIL import ImageDraw, ImageFont

    npy_dir = Path(params["npy_dir"])
    save_dir = Path(params["save_dir"])
    mode = (params.get("mode") or "heat").lower()
    annotate = bool(params.get("annotate", True))
    alpha = min(1.0, max(0.0, float(params.get("alpha", 0.67))))
    img_dir = Path(params["img_dir"]) if params.get("img_dir") else None
    if mode == "overlay" and (img_dir is None or not img_dir.is_dir()):
        raise ValueError("叠加融合模式（overlay）需要提供原图目录 img_dir"
                         "（.npy 同名/前缀匹配的图像所在目录）")
    save_dir.mkdir(parents=True, exist_ok=True)

    npys = sorted(p for p in npy_dir.rglob("*.npy"))
    if not npys:
        raise ValueError(f"目录中无 .npy 文件: {npy_dir}")
    task.total_epochs = len(npys)
    task.log(f"待可视化 .npy: {len(npys)} 个 | 模式: {mode} | 标注: {annotate} | "
             + (f"融合比例(密度权重): {alpha} | 原图目录: {img_dir} | "
                if mode == "overlay" else "")
             + f"输出: {save_dir}")

    def _try_font(size):
        for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    ok, skipped, failed = 0, 0, 0
    first_error = ""
    for i, np_path in enumerate(npys, 1):
        if task.stop_event.is_set():
            raise StopIteration()
        task.epoch = i
        task.progress = i / len(npys) * 100.0
        try:
            out = save_dir / (np_path.stem + ".png")
            if out.exists():
                skipped += 1
                continue
            arr = np.load(np_path)
            if arr.ndim == 3:
                arr = arr[:, :, 0]   # 带通道维的密度图
            if arr.ndim != 2:
                raise ValueError(f"非二维密度图 (shape={arr.shape})")
            d = arr.astype(np.float64) - float(arr.min())
            mx = d.max()
            if mx > 1e-8:
                d = d / mx
            if mode == "gray":
                img = Image.fromarray((d * 255).astype(np.uint8), "L").convert("RGB")
            elif mode == "overlay":
                # 融合逻辑见 _reference_overlay
                src = _find_image_for_mat(np_path, img_dir)
                if src is None:
                    raise ValueError("未定位到同名原图（可用原图目录参数指定）")
                import cv2
                # cv2.imread 在 Windows 不支持非 ASCII 路径，imdecode 等价（同一解码器）
                img_bgr = cv2.imdecode(
                    np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise ValueError(f"原图读取失败: {src}")
                if img_bgr.shape[:2] != arr.shape[:2]:   # 尺寸对齐安全网
                    img_bgr = cv2.resize(img_bgr,
                                         (int(arr.shape[1]), int(arr.shape[0])))
                fused = _reference_overlay(img_bgr, arr.astype(np.float64),
                                           alpha=alpha)
                img = Image.fromarray(fused)
            else:
                heat = (cm.jet(d)[..., :3] * 255).astype(np.uint8)
                img = Image.fromarray(heat)
            if annotate:
                draw = ImageDraw.Draw(img)
                font = _try_font(max(14, img.height // 24))
                label = f"count~{float(arr.sum()):.1f}"
                tb = draw.textbbox((6, 6), label, font=font)
                draw.rectangle([tb[0] - 4, tb[1] - 4, tb[2] + 4, tb[3] + 4],
                               fill=(0, 0, 0))
                draw.text((6, 6), label, fill=(255, 255, 0), font=font)
            img.save(out)
            ok += 1
            if i % 20 == 0 or i == len(npys):
                task.log(f"[{i}/{len(npys)}] {np_path.name} -> {out.name}")
        except Exception as e:
            failed += 1
            if not first_error:
                first_error = f"{np_path.name}: {e}"
            task.log(f"[{i}/{len(npys)}] 失败 {np_path.name}: {e}")

    task.result = {"visualized": ok, "skipped": skipped, "failed": failed,
                   "save_dir": str(save_dir)}
    task.log(f"可视化完成：成功 {ok}，跳过 {skipped}，失败 {failed}")
    if ok == 0 and skipped == 0 and failed > 0:
        raise ValueError(
            f"全部可视化失败（{failed} 个），首个错误：{first_error}"
            f"（详见任务日志）")


# ------------------------- P2PNet 点预测：训练与推理 -------------------------

def _build_p2pnet(training: bool, device):
    """构建 vendored P2PNet 模型 + criterion"""
    from types import SimpleNamespace
    from nets.p2pnet.models.p2pnet import build
    args = SimpleNamespace(
        backbone="vgg16_bn", row=2, line=2,
        set_cost_class=1.0, set_cost_point=0.05,
        point_loss_coef=0.0002, eos_coef=0.5)
    return build(args, training=training, device=torch.device(device))


def _build_p2pnet_criterion():
    """只构建 P2PNet 匹配损失（自定义点预测模型复用，无需再建骨干）"""
    from types import SimpleNamespace
    from nets.p2pnet.models.matcher import build_matcher_crowd
    from nets.p2pnet.models.p2pnet import SetCriterion_Crowd
    args = SimpleNamespace(set_cost_class=1.0, set_cost_point=0.05,
                           point_loss_coef=0.0002, eos_coef=0.5)
    return SetCriterion_Crowd(
        1, matcher=build_matcher_crowd(args),
        weight_dict={"loss_ce": 1, "loss_points": args.point_loss_coef},
        eos_coef=args.eos_coef, losses=["labels", "points"])


def _load_points_gt(gt_path: Path):
    """点标注加载：.txt（每行 x y）或 .mat（image_info/annPoints，复用现有解析）"""
    if gt_path.suffix.lower() == ".txt":
        pts = []
        for line in gt_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line:
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    try:
                        pts.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
        return np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    pts, _ = load_mat_points(gt_path)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def _load_points_gt_any(gt_path: Path):
    """点标注加载（P2R 路线）：.txt / .mat 之外另支持 .npy 的 (N,2) 坐标数组
    （P2RLoss 源实现的数据格式）。.npy 若为 (H,W) 密度图则显式报错
    （点预测路线需要坐标而非密度图）
    """
    if gt_path.suffix.lower() == ".npy":
        arr = np.load(gt_path)
        if arr.ndim == 2 and 2 <= arr.shape[1] <= 4:
            return arr[:, :2].astype(np.float32)
        raise ValueError(
            f"该 .npy 不是点坐标数组（shape={arr.shape}）：点预测路线的 "
            f".npy 标注应为 (N,2) 坐标（P2RLoss 格式）；(H,W) 密度图请"
            f"改用「人群计数 · 密度图」路线或提供 .txt/.mat 点标注: {gt_path}")
    return _load_points_gt(gt_path)


class P2PPointsDataset(Dataset):
    """P2PNet 点预测数据集：图像 + 点标注（.txt x y / ShanghaiTech .mat）。
    ImageNet 归一化；训练时随机缩放 [0.7, 1.3]（图像与坐标同步）。
    """
    def __init__(self, root, train=True, augment=True, augment_strength=1.0,
                 resize_to=0):
        self.root = Path(root)
        self.train = train
        self.augment = augment
        self.aug_strength = max(0.0, min(2.0, augment_strength))
        # 长边目标分辨率（0=保持原始分辨率）
        self.resize_to = max(0, int(resize_to or 0))
        img_dir, gt_dir = _crowd_dirs(self.root, train)
        if img_dir is None:
            raise ValueError(f"未找到图像目录: {root}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.items = []
        if gt_dir is not None:
            for p in sorted(img_dir.iterdir()):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                for suf, kind in ((".txt", "txt"), (".mat", "mat")):
                    g = gt_dir / (p.stem + suf)
                    if g.exists():
                        self.items.append((p, g, kind))
                        break
        else:
            for p in sorted(img_dir.iterdir()):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                # 扁平回退：gt/ 下 .txt/.mat 均可；ground_truth、ground-truth 仅约定 .mat
                for d, suf in (("gt", ".txt"), ("gt", ".mat"),
                               ("ground_truth", ".mat"), ("ground-truth", ".mat")):
                    g = self.root / d / (p.stem + suf)
                    if g.exists():
                        self.items.append((p, g, suf[1:]))
                        break
        if not self.items:
            raise ValueError(
                f"未找到点标注样本（P2PNet 需要 .txt 点标注或 ShanghaiTech "
                f".mat）：{root}。可用「.mat → GT 密度图工具」所在目录的原始 "
                f".mat，或每行 'x y' 的 .txt")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import random as _rnd
        img_path, gt_path, _kind = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        points = _load_points_gt(gt_path)
        # 基础分辨率：长边缩放到 resize_to（0=保持原始分辨率）。
        # 放在随机尺度增强之前，保证训练时 [0.7,1.3] 随机缩放仍然生效。
        if self.resize_to > 0:
            W, H = img.size
            long_side = max(W, H)
            if long_side > 0 and long_side != self.resize_to:
                s = float(self.resize_to) / long_side
                nw = max(1, int(round(W * s)))
                nh = max(1, int(round(H * s)))
                img = img.resize((nw, nh), Image.BILINEAR)
                if len(points):
                    points = points * np.array([nw / W, nh / H],
                                               dtype=np.float32)
        if self.train and self.augment:
            W, H = img.size
            # 1) 随机缩放（图像与坐标同步）
            scale = _rnd.uniform(0.7, 1.3)
            if scale * min(W, H) > 128:
                img = img.resize((max(1, int(W * scale)),
                                  max(1, int(H * scale))), Image.BILINEAR)
                points = points * scale
            # 2) 随机水平翻转（图像与 x 坐标同步：x' = W - x）
            if _rnd.random() < 0.5:
                W2, H2 = img.size
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if len(points):
                    points = points.copy()
                    points[:, 0] = W2 - points[:, 0]
        # FPN 对齐：尺寸对齐到 16 的倍数（P2PNet 三级特征须逐级翻倍，
        # 实测非 16 倍数前向崩溃；参考实现训练按 128 倍数 crop，同满足）
        W, H = img.size
        new_w = max(16, W // 16 * 16)
        new_h = max(16, H // 16 * 16)
        if (new_w, new_h) != (W, H):
            if len(points):
                points = points * np.array(
                    [new_w / max(1, W), new_h / max(1, H)], dtype=np.float32)
            img = img.resize((new_w, new_h), Image.BILINEAR)
        img_t = self.transform(img)
        target = {"point": torch.from_numpy(points).float(),
                  "labels": torch.ones(len(points), dtype=torch.long)}
        return img_t, [target]


def run_p2pnet_train(task, params: dict):
    """P2PNet 训练"""
    from app.services.train_common import (set_seed, EarlyStopper,
                                            save_checkpoint, save_loss_curves,
                                            release_training_memory,
                                            report_batch_progress,
                                            is_main_process)
    from nets.p2pnet.util.misc import collate_fn_crowd as _collate
    dataset_path = params["dataset_path"]
    epochs = int(params.get("epochs", 30))
    eval_every = max(1, int(params.get("eval_every") or 1))
    batch_size = int(params.get("batch_size", 2))
    # lr=0 / 缺省 → 论文默认（P2PNet Adam 主 lr 1e-4、骨干 1e-5）
    lr = float(params.get("lr") or 1e-4)
    lr_backbone = float(params.get("lr_backbone") or 1e-5)
    weight_decay = float(params.get("weight_decay", 1e-4))
    lr_drop = int(params.get("lr_drop", 0)) or max(1, epochs)
    clip_max_norm = float(params.get("clip_max_norm", 0.1))
    pretrained = bool(params.get("pretrained", True))
    augment = bool(params.get("augment", True))
    device_str = params.get("device", "cpu")
    num_workers = int(params.get("num_workers", 0))
    early_stop = int(params.get("early_stopping", 0))
    val_ratio = float(params.get("val_ratio", 0.2))
    # 输入分辨率：长边缩放目标（0=保持原始分辨率）
    img_size = int(params.get("img_size", 0) or 1024)   # 默认 1024（原 0=原始分辨率）
    model_name = params.get("model", "p2pnet")
    loss_name = str(params.get("loss", "p2pnet"))
    from app.core.registry import (is_custom_model_for, get_custom_model,
                                   is_custom_loss_for, get_custom_loss)
    custom_point_model = is_custom_model_for(model_name, "crowd_point")

    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    task.total_epochs = epochs
    task.log("P2PNet 点预测训练 | 参考 CrowdCounting-P2PNet（点回归/分类 + "
             "匈牙利匹配）")
    task.log(f"数据集: {dataset_path} | 设备: {device_str} | lr={lr} "
             f"(backbone {lr_backbone}) | 分辨率: "
             f"{img_size if img_size > 0 else '原始'}")
    set_seed(int(params.get("seed", 42)))
    # 训练是内存峰值场景（P2PNet 骨干 528MB 级）：驻留的推理模型缓存让位
    _n_cached = clear_inference_cache()
    if _n_cached:
        task.log(f"已清空推理模型缓存（{_n_cached} 个），为训练让出内存")

    augment = bool(params.get("augment", True))
    augment_strength = float(params.get("augment_strength", 1.0))
    full = P2PPointsDataset(dataset_path, train=True, augment=augment,
                            augment_strength=augment_strength,
                            resize_to=img_size)
    # 验证集：优先数据集自带 Test 划分；否则按 val_ratio 从训练集切分
    val_eval_ds, val_src = _eval_from_test(
        dataset_path,
        lambda: P2PPointsDataset(dataset_path, train=False, augment=False,
                                 augment_strength=0.0, resize_to=img_size))
    if val_eval_ds is not None:
        train_set = full
        val_set = val_eval_ds
        n_val, n_train = len(val_set), len(train_set)
    else:
        n_val = max(1, int(len(full) * val_ratio)) if len(full) > 1 else 0
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(full), generator=g).tolist()
        # 验证集复用同一批样本，但走评估分支（关闭训练增强：随机缩放/翻转），
        # 否则验证 MAE 每轮都在随机增强视图上计算，不可复现且偏噪声
        import copy as _copy
        val_full = _copy.copy(full)
        val_full.train = False
        val_full.augment = False
        val_set = torch.utils.data.Subset(val_full, idx[:n_val])
        train_set = torch.utils.data.Subset(full, idx[n_val:])
        n_train = len(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True,
                               collate_fn=_collate)
    if len(train_loader) == 0:
        # drop_last 生效：样本数 < batch_size 时批数为 0，静默空训必须显式报错
        raise ValueError(
            f"训练批数为 0（训练样本 {len(train_set)} < Batch Size "
            f"{batch_size}，P2PNet 启用 drop_last），请减小 Batch Size 或补充数据")
    val_loader = (DataLoader(val_set, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=_collate)
                  if n_val else None)
    task.log(f"训练样本: {len(train_set)} | 验证样本: {len(val_set)}（{val_src}）")

    device = torch.device("cuda" if (
        device_str.startswith("cuda") and torch.cuda.is_available())
        else "cpu")
    if params.get("multi_gpu_mode") and torch.cuda.device_count() > 1:
        task.log("P2PNet 暂不支持多 GPU 并行，本次按单卡训练")
    if custom_point_model:
        # 自定义点预测模型：须实现 P2PNet 输出契约
        #   forward(x) -> {"pred_logits": [B,M,2], "pred_points": [B,M,2]}
        # 损失仍复用内置匈牙利匹配 criterion（可另选自定义附加损失）
        model = get_custom_model(model_name, 1, pretrained=pretrained,
                                 task="crowd_point")
        criterion = _build_p2pnet_criterion()
        task.log(f"自定义点预测模型: {model_name}（需输出 "
                 f"pred_logits / pred_points）")
    else:
        model, criterion = _build_p2pnet(training=True, device=device)
    if not pretrained:
        # 从头训练：重新初始化（build 默认加载 VGG16_bn ImageNet 骨干）
        for m in model.modules():
            if hasattr(m, "reset_parameters"):
                try:
                    m.reset_parameters()
                except Exception:
                    pass
    model.to(device)
    criterion.to(device)
    # 损失分项权重（缺省 = P2PNet 论文设置：ce 1 / points 2e-4 / 代价 1、0.05）
    _lw = all_weights(params, {"ce_w": 1.0, "points_w": 0.0002,
                               "cost_class": 1.0, "cost_point": 0.05})
    apply_criterion_weights(criterion, _lw)
    apply_matcher_costs(criterion, _lw)
    task.log(f"损失权重: {describe(_lw)}")
    # 自定义点损失：作为附加项与内置匹配损失相加（默认不启用）
    _aux_w = aux_weight(params)
    aux_loss = None
    if loss_name.startswith("custom:"):
        if not is_custom_loss_for(loss_name, "crowd_point"):
            raise ValueError(
                f"自定义损失 {loss_name} 未注册到 crowd_point 路线"
                f"（请在注册时写 tasks=[\"crowd_point\"]）")
        aux_loss = get_custom_loss(loss_name, task="crowd_point")
        if hasattr(aux_loss, "to"):
            aux_loss = aux_loss.to(device)
        task.log(f"自定义点预测附加损失: {loss_name}")
    # 骨干与其余部分分组学习率；优化器可选（缺省 Adam，与 P2PNet 论文一致）
    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": lr_backbone},
    ]
    from app.core.optim_builders import build_torch_optimizer
    optimizer = build_torch_optimizer(params, param_dicts, lr,
                                      weight_decay=weight_decay)
    # 统一调度器（与密度图路线共用 9 种选项；默认 step）
    scheduler, _is_plateau = build_crowd_scheduler(
        optimizer, params.get("scheduler") or "cosine", epochs, params)
    task.log(f"优化器: {params.get('optimizer') or 'adam'}"
             f"（骨干 lr {lr_backbone}） | "
             f"调度器: {params.get('scheduler') or 'cosine'}")

    csv_path = save_dir / "metrics.csv"
    _csv_header = ["epoch", "train_loss", "val_loss", "val_mae"]
    best_mae, start_epoch = float("inf"), 1
    best_path, last_path = _ckpt_paths(save_dir, params.get("model"))
    if params.get("resume_weight"):
        from app.services.train_common import (resume_training_state,
                                               reload_task_history)
        start_epoch, best_mae, _bm = resume_training_state(
            params["resume_weight"], model, optimizer, scheduler,
            best_key="val_mae", best_mode="min")
        task.best = _bm
        task.log(f"断点续训：从 epoch {start_epoch} 继续（best_mae={best_mae}）")
        reload_task_history(task, csv_path, start_epoch, _csv_header)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_csv_header)

    stopper = EarlyStopper(early_stop, mode="min")

    def _evaluate():
        """验证：计数 MAE + 与训练同口径的损失（匹配损失 + 自定义附加损失）

        同时按批打印验证进度（loss / MAE 滚动均值），便于观察测试进度。
        返回 (mae, loss)；loss 为 None 时表示无有效批次。
        """
        model.eval()
        criterion.eval()
        if aux_loss is not None and hasattr(aux_loss, "eval"):
            aux_loss.eval()
        maes, losses = [], []
        total_batches = len(val_loader)
        log_every = max(1, total_batches // 10)
        task.log(f"  开始验证（测试集）：{total_batches} 批（含匹配损失）")
        t0 = time.time()
        with torch.no_grad():
            for bi, (samples, targets) in enumerate(val_loader, 1):
                if task.stop_event.is_set():
                    raise StopIteration()
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                losses_v = sum(loss_dict[k] * criterion.weight_dict[k]
                               for k in loss_dict.keys()
                               if k in criterion.weight_dict)
                if aux_loss is not None:
                    extra = aux_loss(outputs, targets)
                    if isinstance(extra, dict):
                        extra = sum(v for v in extra.values())
                    losses_v = losses_v + _aux_w * extra
                if torch.isfinite(losses_v):
                    losses.append(float(losses_v))
                scores = torch.nn.functional.softmax(
                    outputs["pred_logits"], -1)[:, :, 1][0]
                pred_cnt = int((scores > 0.5).sum())
                gt_cnt = targets[0]["point"].shape[0]
                maes.append(abs(pred_cnt - gt_cnt))
                if bi % log_every == 0 or bi == total_batches:
                    task.log(
                        f"  验证 {bi}/{total_batches} | loss "
                        f"{np.mean(losses) if losses else 0.0:.4f} | "
                        f"MAE {np.mean(maes):.2f} | "
                        f"用时 {time.time() - t0:.1f}s")
        return (float(np.mean(maes)) if maes else float("inf"),
                float(np.mean(losses)) if losses else None)

    for epoch in range(start_epoch, epochs + 1):
        if task.stop_event.is_set():
            raise StopIteration()
        do_eval = ((eval_every <= 1) or (epoch % eval_every == 0)
                   or (epoch == epochs))
        model.train()
        criterion.train()
        if aux_loss is not None and hasattr(aux_loss, "train"):
            aux_loss.train()
        total_loss, n_batches = 0.0, 0
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)
        _main = is_main_process(params)
        for batch_idx, (samples, targets) in enumerate(train_loader):
            if task.stop_event.is_set():
                raise StopIteration()
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * criterion.weight_dict[k]
                         for k in loss_dict.keys()
                         if k in criterion.weight_dict)
            if aux_loss is not None:
                extra = aux_loss(outputs, targets)
                if isinstance(extra, dict):
                    extra = sum(v for v in extra.values())
                losses = losses + _aux_w * extra
            if not torch.isfinite(losses):
                task.log(f"损失非有限值，跳过该 batch: {loss_dict}")
                continue
            optimizer.zero_grad()
            losses.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               clip_max_norm)
            optimizer.step()
            total_loss += float(losses)
            n_batches += 1
            report_batch_progress(
                task, epoch=epoch, epochs=epochs, batch_idx=batch_idx,
                total_batches=total_batches, running_loss=total_loss,
                n_batches=n_batches, log_every=log_every, is_main=_main)
        avg_loss = total_loss / max(1, n_batches)
        val_mae, val_loss = _evaluate() if (val_loader and do_eval) \
            else (None, None)
        # plateau 按 val_mae 步进（与密度图路线一致），其余按 epoch
        if _is_plateau:
            if val_mae is not None:
                scheduler.step(val_mae)
        else:
            scheduler.step()
        task.epoch = epoch
        task.progress = epoch / epochs * 100.0
        task.metrics = {"train_loss": round(avg_loss, 4)}
        row = {"epoch": epoch, "train_loss": round(avg_loss, 4)}
        if val_loss is not None:
            task.metrics["val_loss"] = round(val_loss, 4)
            row["val_loss"] = round(val_loss, 4)
        if val_mae is not None:
            task.metrics["val_mae"] = round(val_mae, 4)
            row["val_mae"] = round(val_mae, 4)
        task.history.append(row)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get("epoch"), row.get("train_loss"),
                                    row.get("val_loss", ""),
                                    row.get("val_mae", "")])
        task.log(f"Epoch {epoch}/{epochs} | loss {avg_loss:.4f}"
                 + (f" | val_loss {val_loss:.4f}"
                    if val_loss is not None else "")
                 + (f" | val_mae {val_mae:.2f}" if val_mae is not None else ""))
        save_checkpoint(last_path, net=model, optimizer=optimizer,
                        scheduler=scheduler, epoch=epoch,
                        best_metric={"val_mae": val_mae}, model="p2pnet")
        if val_mae is not None and val_mae < best_mae:
            best_mae = val_mae
            task.best = {"epoch": epoch, "val_mae": round(val_mae, 4)}
            save_checkpoint(best_path, net=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch,
                            best_metric={"val_mae": val_mae},
                            model="p2pnet")
        stop_metric = val_mae if val_mae is not None else (
            avg_loss if val_loader is None else None)
        if early_stop and stop_metric is not None and stopper(stop_metric):
            task.log(f"早停触发（patience={early_stop}）")
            break
        # 每轮收尾及时释放：gc 循环垃圾 + CUDA 缓存显存归还
        release_training_memory()

    save_loss_curves(save_dir, task.history)
    task.result = {"best_weight": str(best_path),
                   "last_weight": str(last_path),
                   "best_val_mae": best_mae if best_mae < float("inf")
                   else None,
                   "save_dir": str(save_dir)}
    task.log(f"训练结束，最优 val_mae={best_mae:.4f}"
             if best_mae < float("inf") else "训练结束（无验证集）")


def p2pnet_infer(image: Image.Image, weight_path=None, device="cpu",
                 pretrained=True, conf_threshold=0.5):
    """P2PNet 推理：返回人数 + 点可视化图（红色圆点）+ 预测点坐标列表"""
    import base64
    import io
    import cv2

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    # 权重：显式路径 → 内置 SHTechA 检查点
    wp = weight_path
    if not wp:
        from config import WEIGHTS_DIR
        cand = WEIGHTS_DIR / "p2pnet" / "SHTechA.pth"
        if cand.is_file():
            wp = str(cand)
        elif pretrained:
            raise ValueError("P2PNet 推理需要权重：请选择训练产物/上传 .pth，"
                             "或将 SHTechA.pth 放到 weights/pretrained/p2pnet/")
        else:
            raise ValueError("P2PNet 推理需要权重文件")
    # 模型缓存（H3）：同 (权重, 设备) 只构建一次；权重文件被覆盖训练后
    # mtime 变化自动失效重建。构建成本高（骨干 528MB），摄像头逐帧场景必须缓存
    cache_key = _model_cache_key("p2pnet", wp, device)
    model = _model_cache_get(cache_key)
    if model is None:
        model = _build_p2pnet(training=False, device=dev)
        ck = torch.load(wp, map_location="cpu")
        # 兼容两种 checkpoint：框架格式（model_state_dict / state_dict）
        # 与官方 SHTechA.pth（"model" 键直接为 state_dict）
        sd = None
        if isinstance(ck, dict):
            sd = ck.get("model_state_dict") or ck.get("state_dict")
            if sd is None and isinstance(ck.get("model"), dict):
                sd = ck["model"]
        if not isinstance(sd, dict):
            sd = ck
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
              for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.to(dev).eval()
        _model_cache_put(cache_key, model)

    img = image.convert("RGB")
    W0, H0 = img.size
    # 尺寸对齐到 16 的倍数：P2PNet 的 FPN 用固定 ×2 上采样，特征三级
    # （1/4、1/8、1/16）须逐级严格翻倍——实测 H/W 非 16 倍数（如 200、56）
    # 必崩，16 倍数（192、288、96 等）全部通过；同时限制最大边长，
    # 整体等比缩放不破坏长宽比（L3）
    max_side = 1536
    scale = min(1.0, max_side / max(H0, W0))
    new_w = max(16, int(round(W0 * scale)) // 16 * 16)
    new_h = max(16, int(round(H0 * scale)) // 16 * 16)
    img_in = img.resize((new_w, new_h), Image.BILINEAR) \
        if (new_w, new_h) != (W0, H0) else img
    x = transforms.functional.to_tensor(img_in)
    x = transforms.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    x = x.unsqueeze(0).to(dev)

    with torch.no_grad():
        outputs = model(x)
    scores = torch.nn.functional.softmax(
        outputs["pred_logits"], -1)[:, :, 1][0]
    points_all = outputs["pred_points"][0]
    keep = scores > float(conf_threshold)
    points = points_all[keep].detach().cpu().numpy()
    # 坐标从推理尺寸映射回原图分辨率（可视化与返回值均基于原图）
    if (new_w, new_h) != (W0, H0):
        points = points * np.array([W0 / new_w, H0 / new_h],
                                   dtype=np.float32)

    # 可视化：预测点画红色实心圆
    canvas = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    size = max(2, int(max(H0, W0) / 300))
    for p in points:
        canvas = cv2.circle(canvas, (int(p[0]), int(p[1])), size,
                            (0, 0, 255), -1)
    vis = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(vis).save(buf, format="PNG")
    vis_b64 = base64.b64encode(buf.getvalue()).decode()

    return {"count": int(len(points)), "visualization": vis_b64,
            "num_points": int(len(points)),
            "points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in points[:2000]],
            "conf_threshold": conf_threshold, "model": "p2pnet",
            "point_based": True}


def _custom_point_infer(image: Image.Image, model_name, weight_path=None,
                        device="cpu", conf_threshold=0.5):
    """自定义点预测模型推理（输出契约同 P2PNet: pred_logits/pred_points）"""
    import base64
    import io
    import cv2
    if not weight_path:
        raise ValueError(
            f"自定义点预测模型 {model_name} 推理需要权重："
            f"请选择训练产物 / 上传 .pth")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    cache_key = _model_cache_key(model_name, weight_path, device)
    model = _model_cache_get(cache_key)
    if model is None:
        from app.core.registry import get_custom_model
        model = get_custom_model(model_name, 1, pretrained=False,
                                 task="crowd_point")
        _load_weights_compat(model, weight_path, context="自定义点预测权重")
        model.to(dev).eval()
        _model_cache_put(cache_key, model)

    img = image.convert("RGB")
    W0, H0 = img.size
    max_side = 1536
    scale = min(1.0, max_side / max(H0, W0))
    new_w = max(16, int(round(W0 * scale)) // 16 * 16)
    new_h = max(16, int(round(H0 * scale)) // 16 * 16)
    img_in = (img.resize((new_w, new_h), Image.BILINEAR)
              if (new_w, new_h) != (W0, H0) else img)
    x = transforms.functional.to_tensor(img_in)
    x = transforms.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    x = x.unsqueeze(0).to(dev)
    with torch.no_grad():
        outputs = model(x)
    scores = torch.nn.functional.softmax(
        outputs["pred_logits"], -1)[:, :, 1][0]
    points_all = outputs["pred_points"][0]
    keep = scores > float(conf_threshold)
    points = points_all[keep].detach().cpu().numpy()
    if (new_w, new_h) != (W0, H0):
        points = points * np.array([W0 / new_w, H0 / new_h],
                                   dtype=np.float32)

    canvas = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    size = max(2, int(max(H0, W0) / 300))
    for p in points:
        canvas = cv2.circle(canvas, (int(p[0]), int(p[1])), size,
                            (0, 0, 255), -1)
    vis = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(vis).save(buf, format="PNG")
    vis_b64 = base64.b64encode(buf.getvalue()).decode()

    return {"count": int(len(points)), "visualization": vis_b64,
            "num_points": int(len(points)),
            "points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in points[:2000]],
            "conf_threshold": float(conf_threshold),
            "model": model_name, "point_based": True}


# ------------------------- P2RLoss 点预测（CVPR 2025，半监督） -------------------------

class P2RDataset(Dataset):
    """P2R 半监督点预测数据集（vendor 逻辑来自 P2RLoss-main/datasets/shha.py）

    - 标注格式：.txt（每行 x y）/ .mat（annPoints|image_info）/ .npy（(N,2)
      坐标，P2RLoss 源格式）
    - 训练 __getitem__ 返回（有标注弱视图, 点序列, id, 无标注弱+强视图, 裁剪
      掩码, id）六元组：有标注样本按 id 随机采样，无标注样本按索引遍历
      （源实现语义：无标注池 = 全部训练图像）
    - label_ratio < 1 时按种子确定性划分「有标注子集」，模拟论文的
      protocol 文件（如 sha-5.txt = 5% 有标注）
    - 评估返回全图（不裁剪，32 倍数 padding），批大小须为 1
    """

    def __init__(self, root, train=True, label_ratio=1.0, crop_size=256,
                 augment=True, augment_strength=1.0, seed=42, val_ratio=0.2):
        self.root = Path(root)
        self.train = train
        crop_size = max(32, int(crop_size) or 256)
        from nets.p2r import NormalSample
        self.norm = NormalSample(
            mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_size=(crop_size, crop_size),
            resize_factor=0.3 * max(0.0, min(2.0, augment_strength)),
            train=train, augment=augment and train)
        # 评估专用预处理器（不裁剪不翻转，仅 32 倍数 padding）
        self.norm_eval = NormalSample(
            mean=IMAGENET_MEAN, std=IMAGENET_STD,
            crop_size=(crop_size, crop_size), train=False)

        img_dir, gt_dir = _crowd_dirs(self.root, train)
        if img_dir is None:
            raise ValueError(f"未找到图像目录: {root}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        items = []
        if gt_dir is not None:
            for p in sorted(img_dir.iterdir()):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                for suf in (".txt", ".mat", ".npy"):
                    g = gt_dir / (p.stem + suf)
                    if g.exists():
                        items.append((p, g))
                        break
        else:
            # 扁平回退：gt/npys/labels 等目录下 .txt/.mat/.npy
            for p in sorted(img_dir.iterdir()):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                for d in ("gt", "npys", "labels", "ground_truth"):
                    hit = next((suf for suf in (".txt", ".mat", ".npy")
                                if (self.root / d / (p.stem + suf)).exists()),
                               None)
                    if hit:
                        items.append((p, self.root / d / (p.stem + hit)))
                        break
        if not items:
            raise ValueError(
                f"未找到点标注样本（P2R 需要 .txt / .mat 点标注或 (N,2) 坐标 "
                f".npy）：{root}")
        self.items = items

        if not train:
            return
        # 有标注 / 无标注划分（种子确定性，模拟论文 protocol 文件）
        stems = [p.stem for p, _ in items]
        label_ratio = max(0.0, min(1.0, float(label_ratio)))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(stems), generator=g).tolist()
        n_lab = max(1, min(len(stems), round(len(stems) * label_ratio)))
        self.label_ids = [stems[i] for i in perm[:n_lab]]
        # 源实现语义：无标注池 = 全部训练图像（有标注图也在其中）
        self.unlabel_ids = stems
        self._by_id = {p.stem: (p, g) for p, g in items}
        self._test_items = None
        self._eval_from_test = False
        self._val_ratio = val_ratio
        self._seed = seed

    def eval_set(self):
        """评估集：优先 Test 目录；无 Test 时按 val_ratio 从训练集确定性切分"""
        if self._test_items is not None:
            return self._test_items
        test_img, test_gt = _crowd_dirs(self.root, False)
        if test_img is not None and test_gt is not None:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            its = []
            for p in sorted(test_img.iterdir()):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                for suf in (".txt", ".mat", ".npy"):
                    g = test_gt / (p.stem + suf)
                    if g.exists():
                        its.append((p, g))
                        break
            if its:
                self._test_items = its
                self._eval_from_test = True
                return its
        g = torch.Generator().manual_seed(self._seed)
        perm = torch.randperm(len(self.items), generator=g).tolist()
        n_val = max(1, int(len(self.items) * self._val_ratio))
        self._test_items = [self.items[i] for i in perm[:n_val]]
        return self._test_items

    def _eval_item(self, img_p, gt_p):
        img = Image.open(img_p).convert("RGB")
        pts = _load_points_gt_any(gt_p)
        img_t = self.norm_eval.im2tensor(img)
        dotseq = torch.from_numpy(pts).float()
        # eval 模式：不裁剪不翻转，仅 32 倍数 padding；坐标转 (y,x)
        images, dotseqs = self.norm_eval.process_lable(img_t, dotseq)
        return images[0], dotseqs[0], img_p.stem

    def __len__(self):
        return len(self.unlabel_ids) if self.train else len(self.items)

    def __getitem__(self, index):
        import random as _rnd
        if not self.train:
            img_p, gt_p = self.items[index]
            return self._eval_item(img_p, gt_p)
        lid = _rnd.choice(self.label_ids)
        img_p, gt_p = self._by_id[lid]
        img = Image.open(img_p).convert("RGB")
        pts = _load_points_gt_any(gt_p)
        limg, lseq = self.norm.process_lable(
            self.norm.im2tensor(img), torch.from_numpy(pts).float())

        uid = self.unlabel_ids[index]
        uimg_p, _ = self._by_id[uid]
        uimg_raw = Image.open(uimg_p).convert("RGB")
        # 弱视图（教师输入）+ 强视图（学生输入）拼 6 通道 + 随机裁剪掩码
        wa = self.norm.im2tensor(uimg_raw)
        sa = self.norm.strong_aug(uimg_raw)
        uimg = self.norm.process_unlabel(torch.cat((wa, sa), dim=0))
        umask = self._random_mask(uimg)
        return limg[0], lseq, lid, uimg[0], umask, uid

    @staticmethod
    def _random_mask(uimgs):
        """无标注强视图的随机擦除掩码（0=擦除区），源实现语义"""
        import random as _rm
        bsize, _, img_h, img_w = uimgs.shape
        cut_img_mask = torch.ones((bsize, 1, img_h, img_w))
        min_cut, max_cut = 1 / 8, 1 / 4
        for i in range(bsize):
            cut_w = int(img_w * (min_cut + _rm.random() * (max_cut - min_cut)))
            cut_h = int(img_h * (min_cut + _rm.random() * (max_cut - min_cut)))
            cut_top = _rm.randint(0, img_h - cut_h)
            cut_left = _rm.randint(0, img_w - cut_w)
            cut_img_mask[i, :, cut_top:cut_top + cut_h,
                         cut_left:cut_left + cut_w] = 0
        return cut_img_mask

    @staticmethod
    def collate_fn(samples):
        if len(samples[0]) == 6:
            limgs, lseqs, lids, uimgs, umasks, uids = zip(*samples)
            return (torch.stack(limgs, 0), sum(list(lseqs), []), lids,
                    torch.stack(uimgs, 0), torch.cat(umasks, 0), uids)
        imgs, seqs, ids = zip(*samples)
        return torch.stack(imgs, 0), sum(list(seqs), []), ids


def _build_p2r(pretrained=True):
    from nets.p2r import build_p2r_model
    return build_p2r_model(pretrained=pretrained)


# P2R 半监督伪标签阈值（logit 域；sigmoid(0.8473)≈0.7，源实现常量）
_P2R_PSEUDO_LOGIT_THR = 0.8472978603872036


def run_p2r_train(task, params: dict):
    """P2RLoss 训练（CVPR 2025 半监督点预测，教师-学生 + Point-to-Region 损失）

    - 教师模型为学生参数的 EMA（momentum 0.998，每步更新；初始为学生的
      深拷贝——比源实现的两独立初始化更符合 mean-teacher 惯例）
    - 阶段一（纯监督）：P2R 匹配损失训练有标注子集
    - 阶段二（半监督）：教师在无标注弱视图上生成伪标签（logit>0.847），
      置信度传播加权 + 随机擦除掩码，权重随轮次线性爬升至 2
    - 优化器 / 学习率 / 调度器 / 损失均可在界面调整；默认对齐论文
      （Adam 5e-5 / 骨干 1e-5 + 恒定学习率 + P2R 匹配损失，1e-3 级 lr
      会发散）。损失下拉默认 P2R，另可追加注册到 crowd_point 的自定义
      附加损失（契约：aux(pred_density, target_seq)）
    """
    import copy as _copy
    from app.services.train_common import (set_seed, EarlyStopper,
                                            save_checkpoint, save_loss_curves,
                                            resume_training_state,
                                            reload_task_history,
                                            release_training_memory,
                                            report_batch_progress,
                                            is_main_process)
    from nets.p2r import P2RLoss as _P2RLoss

    dataset_path = params["dataset_path"]
    epochs = int(params.get("epochs", 30))
    eval_every = max(1, int(params.get("eval_every") or 1))
    batch_size = int(params.get("batch_size", 2))
    weight_decay = float(params.get("weight_decay", 1e-4))
    # lr=0 / 缺省 → 论文默认（P2R Adam 主 lr 5e-5、骨干 1e-5）
    lr = float(params.get("lr") or 5e-5)
    lr_backbone = float(params.get("lr_backbone") or 1e-5)
    loss_name = str(params.get("loss") or "p2r")
    crop_size = int(params.get("img_size", 0)) or 256
    pretrained = bool(params.get("pretrained", True))
    weight_path = params.get("weight_path") or None
    device_str = params.get("device", "cpu")
    num_workers = int(params.get("num_workers", 0))
    early_stop = int(params.get("early_stopping", 0))
    seed = int(params.get("seed", 42))
    label_ratio = float(params.get("label_ratio", 1.0))
    augment = bool(params.get("augment", True))
    augment_strength = float(params.get("augment_strength", 1.0))
    val_ratio = float(params.get("val_ratio", 0.2))

    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    task.total_epochs = epochs
    task.log("P2RLoss 点预测训练（半监督）| CVPR 2025 Point-to-Region Loss | "
             "教师-学生 EMA + P2R 匹配损失")
    task.log(f"数据集: {dataset_path} | 设备: {device_str} | 裁剪: {crop_size}")
    task.log(f"超参: lr={lr}（骨干 {lr_backbone}）| 损失: P2R"
             + (f" + {loss_name}" if loss_name.startswith("custom:") else "")
             + "（缺省对齐论文；界面优化器/调度器/损失可按需调整）")
    set_seed(seed)
    # 教师-学生双模型常驻（约 2 倍单模型内存）：推理模型缓存让位
    _n_cached = clear_inference_cache()
    if _n_cached:
        task.log(f"已清空推理模型缓存（{_n_cached} 个），为训练让出内存")
    stopper = EarlyStopper(early_stop, mode="min")

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    if params.get("multi_gpu_mode") and torch.cuda.device_count() > 1:
        task.log("P2R 暂不支持多 GPU 并行，本次按单卡训练")
    device = torch.device(device_str)

    full = P2RDataset(dataset_path, train=True, label_ratio=label_ratio,
                      crop_size=crop_size, augment=augment,
                      augment_strength=augment_strength, seed=seed,
                      val_ratio=val_ratio)
    task.log(f"训练图像 {len(full.unlabel_ids)} 张，其中有标注 "
             f"{len(full.label_ids)} 张（label_ratio={label_ratio}，种子划分）")
    train_loader = DataLoader(full, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              collate_fn=P2RDataset.collate_fn)
    if len(train_loader) == 0:
        raise ValueError(
            f"训练批数为 0（训练图像 {len(full.unlabel_ids)} < Batch Size "
            f"{batch_size}，P2R 启用 drop_last），请减小 Batch Size 或补充数据")
    eval_items = full.eval_set()
    task.log(f"评估样本: {len(eval_items)} 张（" +
             ("数据集 Test/Val 划分（完整划分）"
              if getattr(full, "_eval_from_test", False)
              else "训练集 val_ratio 切分（未找到 Test/Val 划分）") + "）")

    student = _build_p2r(pretrained=pretrained)
    if weight_path:
        _load_weights_compat(student, weight_path, context="初始权重")
    teacher = _copy.deepcopy(student)   # mean-teacher：初始为学生的深拷贝
    student.to(device)
    teacher.to(device)
    _lw = all_weights(params, {"factor": 1.0})
    criterion = _P2RLoss(factor=_lw.get("factor", 1.0)).to(device)
    task.log(f"损失权重: {describe(_lw)}")

    # 骨干（encoders）与其余部分分组学习率；优化器/调度器可选
    # （默认 adam + 论文恒定学习率，可切换 8 种内置优化器与 9 种调度器）
    param_dicts = [
        {"params": [p for n, p in student.named_parameters()
                    if "encoders" not in n and p.requires_grad]},
        {"params": [p for n, p in student.named_parameters()
                    if "encoders" in n and p.requires_grad],
         "lr": lr_backbone},
    ]
    from app.core.optim_builders import build_torch_optimizer
    optimizer = build_torch_optimizer(params, param_dicts, lr,
                                      weight_decay=weight_decay)
    # 论文 StepLR decay 于 3500 轮（1500 轮内等效不衰减）→ 默认常量学习率
    scheduler, _is_plateau = build_crowd_scheduler(
        optimizer, params.get("scheduler") or "cosine", epochs, params)
    task.log(f"优化器: {params.get('optimizer') or 'adam'}（骨干 lr "
             f"{lr_backbone}） | 调度器: {params.get('scheduler') or 'cosine'}")
    # 自定义点预测附加损失（默认不启用；契约 aux(pred_density, target_seq)）
    _aux_w = aux_weight(params)
    aux_loss = None
    if loss_name.startswith("custom:"):
        from app.core.registry import is_custom_loss_for, get_custom_loss
        if not is_custom_loss_for(loss_name, "crowd_point"):
            raise ValueError(
                f"自定义损失 {loss_name} 未注册到 crowd_point 路线"
                f"（请在注册时写 tasks=[\"crowd_point\"]）")
        aux_loss = get_custom_loss(loss_name, task="crowd_point")
        if hasattr(aux_loss, "to"):
            aux_loss = aux_loss.to(device)
        task.log(f"自定义点预测附加损失: {loss_name}"
                 "（P2R 契约：aux(pred_density, target_seq)）")

    # 半监督启动轮：源实现 1500 轮中第 50 轮启动（约 1/30 训练量 + 50 上限），
    # 此处按 epochs 等比缩放（至少 3 轮纯监督热身）
    stage2 = min(50, max(3, round(epochs / 6)))
    task.log(f"半监督阶段：第 {stage2 + 1} 轮起启用无标注伪标签"
             f"（前 {stage2} 轮纯监督热身）")

    csv_path = save_dir / "metrics.csv"
    _csv_header = ["epoch", "train_loss", "val_loss", "val_mae"]
    best_mae, start_epoch = float("inf"), 1
    best_path, last_path = _ckpt_paths(save_dir, params.get("model"))
    if params.get("resume_weight"):
        start_epoch, best_mae, _bm = resume_training_state(
            params["resume_weight"], student, optimizer, scheduler,
            best_key="val_mae", best_mode="min")
        task.best = _bm
        task.log(f"断点续训：从 epoch {start_epoch} 继续（best_mae={best_mae}）")
        # 教师状态恢复：续训权重中的 teacher_state_dict（框架保存），
        # 缺省（旧版/外部权重）回退为学生深拷贝
        try:
            _ck = torch.load(params["resume_weight"], map_location="cpu")
            if isinstance(_ck, dict) and isinstance(
                    _ck.get("teacher_state_dict"), dict):
                teacher.load_state_dict(_ck["teacher_state_dict"])
                task.log("教师模型状态已从续训权重恢复")
        except Exception as e:
            task.log(f"教师状态恢复失败（回退为学生拷贝）: {e}")
        reload_task_history(task, csv_path, start_epoch, _csv_header)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_csv_header)

    def _validate():
        """全图逐张评估：MAE = |(logit>0)计数 - GT点数| 均值 + P2R 匹配损失"""
        student.eval()
        mae_sum, n = 0.0, 0
        loss_sum, nl = 0.0, 0
        total = max(1, len(eval_items))
        log_every = max(1, total // 10)
        task.log(f"  开始验证（测试集）：{total} 张图像（整图逐张，含损失）")
        t0 = time.time()
        with torch.no_grad():
            for img_p, gt_p in eval_items:
                if task.stop_event.is_set():
                    raise StopIteration()
                img_t, seq, _ = full._eval_item(img_p, gt_p)
                inp = img_t.unsqueeze(0).to(device)
                out = student(inp)
                down_rate = inp.size(-1) // out.size(-1)
                try:
                    seq_t = seq.to(device)
                    lv = criterion(out, [seq_t], down_rate)
                    if aux_loss is not None:
                        extra = aux_loss(out, [seq_t])
                        if isinstance(extra, dict):
                            extra = sum(v for v in extra.values())
                        lv = lv + _aux_w * extra
                    if torch.isfinite(lv):
                        loss_sum += float(lv)
                        nl += 1
                except Exception as e:
                    task.log(f"  验证损失计算跳过: {e}")
                pred_cnt = int((out > 0).sum().item())
                mae_sum += abs(pred_cnt - int(seq.size(0)))
                n += 1
                if n % log_every == 0 or n >= total:
                    task.log(
                        f"  验证 {n}/{total} | loss "
                        f"{loss_sum / max(1, nl):.4f} | "
                        f"MAE {mae_sum / max(1, n):.2f} | "
                        f"用时 {time.time() - t0:.1f}s")
        student.train()
        return (mae_sum / max(1, n), loss_sum / nl if nl else None)

    for epoch in range(start_epoch, epochs + 1):
        if task.stop_event.is_set():
            raise StopIteration()
        do_eval = ((eval_every <= 1) or (epoch % eval_every == 0)
                   or (epoch == epochs))
        student.train()
        teacher.eval()
        total_loss, n_batches = 0.0, 0
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)
        _main = is_main_process(params)
        for batch_idx, (limg, lseq, _lid, uimg, umasks, _uid) in \
                enumerate(train_loader):
            if task.stop_event.is_set():
                raise StopIteration()
            limg = limg.to(device, non_blocking=True)
            lseq = [d.to(device, non_blocking=True) for d in lseq]

            lden = student(limg)
            down_rate = limg.size(-1) // lden.size(-1)
            loss = criterion(lden, lseq, down_rate)
            if aux_loss is not None:
                extra = aux_loss(lden, lseq)
                if isinstance(extra, dict):
                    extra = sum(v for v in extra.values())
                loss = loss + _aux_w * extra

            # ---- 半监督阶段：教师伪标签 + 置信度传播 + 随机擦除掩码 ----
            if epoch - 1 >= stage2:
                wa_img = uimg[:, :3].to(device, non_blocking=True)
                sa_img = uimg[:, 3:].to(device, non_blocking=True)
                cut_img_mask = umasks.to(device, non_blocking=True)
                cut_den_mask = (F.avg_pool2d(
                    cut_img_mask, kernel_size=down_rate, stride=down_rate)
                    >= 0.5)
                sa_img_mask = sa_img * cut_img_mask

                with torch.inference_mode():
                    tu_den = teacher(wa_img)
                    tu_mask, tu_seq2 = [], []
                    for tden in tu_den:
                        tseq = torch.nonzero((tden >= 0).squeeze())
                        peaks = tden.squeeze()[tseq[:, 0], tseq[:, 1]]
                        tu_mask.append(peaks > _P2R_PSEUDO_LOGIT_THR)
                        tseq2 = tseq * down_rate + (down_rate - 1) / 2
                        tu_seq2.append(tseq2)

                weight = min(max((epoch - 1 - stage2) * 0.01, 0), 2)
                su_den = student(sa_img_mask)
                semi_loss = criterion(su_den, tu_seq2, down_rate, tu_mask,
                                      crop_den_masks=cut_den_mask)
                loss = (loss + weight * semi_loss) / (1 + weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 教师 EMA 更新（源实现 momentum=0.998，每步一次）
            momentum = 0.998
            for para_t, para_s in zip(teacher.parameters(),
                                      student.parameters()):
                para_t.data.copy_(para_t.data * momentum
                                  + para_s.data * (1.0 - momentum))
            total_loss += loss.item()
            n_batches += 1
            report_batch_progress(
                task, epoch=epoch, epochs=epochs, batch_idx=batch_idx,
                total_batches=total_batches, running_loss=total_loss,
                n_batches=n_batches, log_every=log_every, is_main=_main)

        train_loss = total_loss / max(1, n_batches)
        val_mae, val_loss = _validate() if do_eval else (None, None)
        if _is_plateau:
            if val_mae is not None:
                scheduler.step(val_mae)
        else:
            scheduler.step()

        task.epoch = epoch
        task.progress = epoch / epochs * 100.0
        task.metrics = {"train_loss": round(train_loss, 5)}
        if val_mae is not None:
            task.metrics["val_mae"] = round(val_mae, 3)
        if val_loss is not None:
            task.metrics["val_loss"] = round(val_loss, 5)
        task.history.append({"epoch": epoch, **task.metrics})
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.5f}",
                 "" if val_loss is None else f"{val_loss:.5f}",
                 "" if val_mae is None else f"{val_mae:.3f}"])
        task.log(f"Epoch {epoch}/{epochs} | loss {train_loss:.5f}"
                 + (f" | val_loss {val_loss:.5f}"
                    if val_loss is not None else "")
                 + (f" | val_MAE {val_mae:.3f}"
                    if val_mae is not None else ""))
        save_loss_curves(save_dir, task.history, ("val_mae", "val_loss"))

        is_best = val_mae is not None and val_mae < best_mae
        if is_best:
            best_mae = val_mae
            task.best = {"epoch": epoch, "val_mae": round(val_mae, 3)}
        # last.pth 附带教师状态（续训恢复用）；best.pth 仅学生（推理/部署）
        save_checkpoint(last_path, net=student, optimizer=optimizer,
                        scheduler=scheduler, epoch=epoch,
                        best_metric=task.best, model="p2r",
                        val_mae=val_mae, teacher_state_dict=teacher.state_dict())
        if is_best:
            save_checkpoint(best_path, net=student, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch,
                            best_metric=task.best, model="p2r",
                            val_mae=val_mae)
        if val_mae is not None and stopper.step(val_mae):
            task.log(f"早停触发：MAE 已 {stopper.patience} 轮未提升")
            break
        # 每轮收尾及时释放：gc 循环垃圾 + CUDA 缓存显存归还（教师-学生
        # 双模型 + 伪标签缓冲是本路线的内存峰值，逐轮归还尤为重要）
        release_training_memory()

    task.result = {"best_weight": str(best_path), "last_weight": str(last_path),
                   "best_val_mae": round(best_mae, 3), "save_dir": str(save_dir),
                   "model": "p2r"}
    task.log(f"训练结束，最优 val_MAE={best_mae:.3f}")


def p2r_infer(image: Image.Image, weight_path=None, device="cpu",
              pretrained=True, conf_threshold=0.5):
    """P2R 推理：返回人数 + 预测点可视化（红色圆点）+ 预测点坐标列表

    人数 = logits>0 的位置数（与训练期评估口径一致）；conf_threshold 对
    P2R 不生效（点由 logit>0 判定，与源实现 validate 完全一致）
    """
    import base64
    import io
    import cv2

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    if not weight_path:
        raise ValueError("P2R 推理需要权重：请选择训练产物"
                         "（*_best.pth / *_last.pth）或上传 .pth"
                         "（P2R 无内置检查点）")
    cache_key = _model_cache_key("p2r", weight_path, device)
    model = _model_cache_get(cache_key)
    if model is None:
        model = _build_p2r(pretrained=False)
        ck = torch.load(weight_path, map_location="cpu")
        sd = None
        if isinstance(ck, dict):
            sd = ck.get("model_state_dict") or ck.get("state_dict")
        if not isinstance(sd, dict):
            sd = ck
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
              for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.to(dev).eval()
        _model_cache_put(cache_key, model)

    img = image.convert("RGB")
    W0, H0 = img.size
    # 限制最大边长（显存/内存），再对齐到 32 倍数（编码器 /8 + 解码 ×2 → /4）
    max_side = 1536
    scale = min(1.0, max_side / max(H0, W0))
    new_w = max(32, int(round(W0 * scale)) // 32 * 32)
    new_h = max(32, int(round(H0 * scale)) // 32 * 32)
    img_in = img.resize((new_w, new_h), Image.BILINEAR) \
        if (new_w, new_h) != (W0, H0) else img
    x = transforms.functional.to_tensor(img_in)
    x = transforms.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    x = x.unsqueeze(0).to(dev)

    with torch.no_grad():
        den = model(x)                       # [1,1,H/4,W/4] logits
    down_rate = x.size(-1) // den.size(-1)
    count = int((den > 0).sum().item())

    # 预测点：(row,col) → 原图坐标（像素中心对齐），剔除 32 倍数 padding 区
    idx = torch.nonzero((den > 0).squeeze()).cpu().numpy()   # [K,2] (y,x)
    pts = idx * down_rate + (down_rate - 1) / 2
    pts = pts[(pts[:, 0] < new_h) & (pts[:, 1] < new_w)][:, ::-1]  # → (x,y)
    if (new_w, new_h) != (W0, H0):
        pts = pts * np.array([W0 / new_w, H0 / new_h], dtype=np.float32)

    canvas = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    size = max(2, int(max(H0, W0) / 300))
    for p in pts:
        canvas = cv2.circle(canvas, (int(p[0]), int(p[1])), size,
                            (0, 0, 255), -1)
    vis = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(vis).save(buf, format="PNG")
    vis_b64 = base64.b64encode(buf.getvalue()).decode()

    return {"count": count, "visualization": vis_b64,
            "num_points": count,
            "points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in pts[:2000]],
            "model": "p2r", "point_based": True}


# ------------------------- Consistent-Point 点预测（arXiv 2503.12441，半监督） -------------------------

class CPointSemiDataset(Dataset):
    """Consistent-Point 半监督数据集（vendor 逻辑来自本仓库 Consistent-Point 复现）

    - 标注格式与 P2R 路线一致：.txt / .mat / .npy((N,2) 坐标)
    - label_ratio < 1：种子确定性划分「有标注子集」，其余进无标注池
      （Consistent-Point 语义：无标注池 = 非有标注图像；=1 时纯监督）
    - 有标注样本：弱几何增强（缩放 [0.7,1.3] → 裁 crop → 翻转，图像与
      坐标同步）+ GT；无标注样本：同一几何变换得到弱视图（教师输入），
      其上仅叠加光度强增强（色彩抖动/灰度/模糊）得到强视图（学生输入）
      ——几何共享，伪点坐标对两个视图均有效
    - 评估：全图归一化 + 16 倍数右/下 padding（点坐标不变），batch=1
    """

    def __init__(self, root, train=True, label_ratio=1.0, crop_size=128,
                 augment=True, augment_strength=1.0, seed=42, val_ratio=0.2):
        self.root = Path(root)
        self.train = train
        self.crop = max(16, int(crop_size) or 128)
        self.augment = augment and train
        self.aug_strength = max(0.0, min(2.0, augment_strength))
        self._seed = seed
        self._val_ratio = val_ratio
        self._test_items = None
        self._eval_from_test = False

        img_dir, gt_dir = _crowd_dirs(self.root, train)
        if img_dir is None:
            raise ValueError(f"未找到图像目录: {root}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        # GT 候选目录：标准 GT 目录优先，扁平结构（gt/npys/labels）回退
        cand_dirs = [gt_dir] if gt_dir is not None else \
            [self.root / d for d in ("gt", "npys", "labels")]
        items = []
        for p in sorted(img_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            for d in cand_dirs:
                if d is None or not d.is_dir():
                    continue
                hit = next((d / (p.stem + suf)
                            for suf in (".txt", ".mat", ".npy")
                            if (d / (p.stem + suf)).exists()), None)
                if hit is not None:
                    items.append((p, hit))
                    break
        if not items:
            raise ValueError(
                f"未找到点标注样本（Consistent-Point 需要 .txt / .mat 点标注"
                f"或 (N,2) 坐标 .npy）：{root}")
        self.items = items
        if not train:
            return

        stems = [p.stem for p, _ in items]
        label_ratio = max(0.0, min(1.0, float(label_ratio)))
        if label_ratio < 1.0:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(stems), generator=g).tolist()
            n_lab = max(1, min(len(stems) - 1,
                               round(len(stems) * label_ratio)))
            labeled = {stems[i] for i in perm[:n_lab]}
        else:
            labeled = set(stems)
        self.label = [(p, g) for p, g in items if p.stem in labeled]
        self.unlabel = [(p, g) for p, g in items if p.stem not in labeled]

    def eval_set(self):
        """评估集：优先 Test 目录；无 Test 时按 val_ratio 从训练集确定性切分"""
        if self._test_items is not None:
            return self._test_items
        test_img, test_gt = _crowd_dirs(self.root, False)
        if test_img is not None and test_gt is not None:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            its = []
            for p in sorted(test_img.iterdir()):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                for suf in (".txt", ".mat", ".npy"):
                    g = test_gt / (p.stem + suf)
                    if g.exists():
                        its.append((p, g))
                        break
            if its:
                self._test_items = its
                self._eval_from_test = True
                return its
        g = torch.Generator().manual_seed(self._seed)
        perm = torch.randperm(len(self.items), generator=g).tolist()
        n_val = max(1, int(len(self.items) * self._val_ratio))
        self._test_items = [self.items[i] for i in perm[:n_val]]
        return self._test_items

    def _load(self, item):
        img = Image.open(item[0]).convert("RGB")
        pts = _load_points_gt_any(item[1])
        return img, pts

    def _weak_geometric(self, img, points):
        """缩放 + 随机裁剪 + 翻转（图像与坐标同步；Consistent-Point 弱增强）"""
        import random as _rnd
        c = self.crop
        w, h = img.size
        s = _rnd.uniform(0.7, 0.7 + 0.6 * self.aug_strength) \
            if self.augment else 1.0
        if s * min(w, h) > c:
            nw, nh = max(int(w * s), c), max(int(h * s), c)
            img = img.resize((nw, nh), Image.BILINEAR)
            if points.shape[0] > 0:
                points = points * np.array([nw / w, nh / h],
                                           dtype=np.float32)
            w, h = nw, nh
        if w < c or h < c:
            canvas = Image.new("RGB", (max(w, c), max(h, c)))
            canvas.paste(img, (0, 0))
            img = canvas
            w, h = img.size
        sx = _rnd.randint(0, w - c)
        sy = _rnd.randint(0, h - c)
        img = img.crop((sx, sy, sx + c, sy + c))
        if points.shape[0] > 0:
            inside = ((points[:, 0] >= sx) & (points[:, 0] < sx + c) &
                      (points[:, 1] >= sy) & (points[:, 1] < sy + c))
            points = points[inside] - np.array([sx, sy], dtype=np.float32)
        if self.augment and _rnd.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if points.shape[0] > 0:
                points = points.copy()
                points[:, 0] = (c - 1) - points[:, 0]
        return img, points

    def _strong_photometric(self, img):
        """仅光度强增强（几何与弱视图共享，伪点坐标保持有效）"""
        if not self.augment:
            return img
        import random as _rnd
        if _rnd.random() < 0.8:
            img = img.convert("HSV")
            if _rnd.random() < 0.8:
                img = Image.eval(img, lambda v: int(v * _rnd.uniform(0.6, 1.4)))
            img = img.convert("RGB")
        if _rnd.random() < 0.25:
            img = img.convert("L").convert("RGB")
        if _rnd.random() < 0.8:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(
                radius=_rnd.uniform(0.1, 2.0)))
        return img

    @staticmethod
    def _pack(points, sid, weight=None):
        pts = torch.as_tensor(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
        t = {"point": pts,
             "labels": torch.ones(pts.shape[0], dtype=torch.long),
             "image_id": sid}
        if weight is not None:
            t["weight"] = weight
        return t

    def _eval_item(self, img_p, gt_p):
        img, pts = self._load((img_p, gt_p))
        x = transforms.functional.to_tensor(img)
        x = transforms.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        h, w = x.shape[-2:]
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, self._pack(pts, img_p.stem), img_p.stem

    def __len__(self):
        if not self.train:
            return len(self.items)
        # 训练迭代按无标注池索引（无无标注图时按有标注池长度）
        return len(self.unlabel) or len(self.label)

    def __getitem__(self, index):
        import random as _rnd
        if not self.train:
            img_p, gt_p = self.items[index]
            return self._eval_item(img_p, gt_p)
        # 有标注：随机抽一张弱增强 + GT
        litem = _rnd.choice(self.label)
        lid = litem[0].stem
        limg, lpts = self._load(litem)
        limg, lpts = self._weak_geometric(limg, lpts)
        limg = transforms.functional.normalize(
            transforms.functional.to_tensor(limg), IMAGENET_MEAN,
            IMAGENET_STD)
        ltarget = self._pack(lpts, lid)
        # 无标注：同一几何 → 弱视图（教师）+ 光度强增强 → 强视图（学生）
        if self.unlabel:
            uitem = self.unlabel[index % len(self.unlabel)]
            uid = uitem[0].stem
            uimg, _ = self._load(uitem)
            uimg, _ = self._weak_geometric(
                uimg, np.zeros((0, 2), dtype=np.float32))
            wa = transforms.functional.normalize(
                transforms.functional.to_tensor(uimg), IMAGENET_MEAN,
                IMAGENET_STD)
            sa = transforms.functional.normalize(
                transforms.functional.to_tensor(self._strong_photometric(uimg)),
                IMAGENET_MEAN, IMAGENET_STD)
            return limg, ltarget, lid, wa, sa, uid
        # 纯监督（label_ratio=1）：无标注分支缺席，占位对齐批结构
        dummy = torch.zeros_like(limg)
        return limg, ltarget, lid, dummy, dummy, lid

    @staticmethod
    def collate_train(samples):
        limgs, ltargets, lids, was, sas, uids = zip(*samples)
        return (torch.stack(limgs, 0), list(ltargets), list(lids),
                torch.stack(was, 0), torch.stack(sas, 0), list(uids))


def _build_cpoint(pretrained=True, training=False):
    """构建 Consistent-Point 模型（training=True 时同时返回损失器）"""
    from nets.cpoint import cp_default_args
    from nets.cpoint.p2pnet import build as _build_cp
    args = cp_default_args()
    args.no_pretrained = not pretrained
    return _build_cp(args, training=training)


def run_cpoint_train(task, params: dict):
    """Consistent-Point 训练（arXiv 2503.12441 半监督点预测）

    Mean-Teacher：教师在弱视图上产出伪点（score>0.5），PA 取 anchor 最近
    K=4 邻居均值聚合位置，IUC 以 (c-0.5)/0.5 加权正样本分类项；学生在光度
    强视图上计算 L^U；总损失 L = L^L + λ_u·L^U（λ_u=0.1，无热身阶段，
    有无标注池即生效）。教师为学生的 EMA（0.999，每步更新）。
    优化器 / 学习率 / 调度器 / 损失均可在界面调整；默认对齐论文
    （Adam 1e-4 / 骨干 1e-5 + 恒定学习率 + P2PNet 匹配损失）。损失下拉
    默认 Consistent-Point，另可追加注册到 crowd_point 的自定义附加损失。
    """
    import copy as _copy
    from app.services.train_common import (set_seed, EarlyStopper,
                                            save_checkpoint, save_loss_curves,
                                            resume_training_state,
                                            reload_task_history,
                                            release_training_memory,
                                            report_batch_progress,
                                            is_main_process)
    from nets.cpoint.pa import position_aggregation
    from nets.cpoint.iuc import iuc_weights
    from nets.cpoint.ema import ModelEMA

    dataset_path = params["dataset_path"]
    epochs = int(params.get("epochs", 30))
    eval_every = max(1, int(params.get("eval_every") or 1))
    batch_size = int(params.get("batch_size", 2))
    # lr=0 / 缺省 → 论文默认（CP Adam 主 lr 1e-4、骨干 1e-5）
    lr = float(params.get("lr") or 1e-4)
    lr_backbone = float(params.get("lr_backbone") or 1e-5)
    loss_name = str(params.get("loss") or "cp")
    crop_size = int(params.get("img_size", 0)) or 256
    pretrained = bool(params.get("pretrained", True))
    weight_path = params.get("weight_path") or None
    device_str = params.get("device", "cpu")
    num_workers = int(params.get("num_workers", 0))
    early_stop = int(params.get("early_stopping", 0))
    seed = int(params.get("seed", 42))
    label_ratio = float(params.get("label_ratio", 1.0))
    augment = bool(params.get("augment", True))
    augment_strength = float(params.get("augment_strength", 1.0))
    val_ratio = float(params.get("val_ratio", 0.2))
    lambda_u = float(params.get("lambda_u", 0.1))   # 论文 Fig.4 最优
    pa_k = int(params.get("pa_k", 4))               # 论文 Tab.6 最优
    pseudo_thr = float(params.get("pseudo_threshold", 0.5))
    ema_decay = float(params.get("ema_decay", 0.999))

    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    task.total_epochs = epochs
    task.log("Consistent-Point 点预测训练（半监督）| P2PNet 基座 + "
             "PA 位置聚合 + IUC 不确定性校准 + Mean-Teacher")
    task.log(f"数据集: {dataset_path} | 设备: {device_str} | 裁剪: {crop_size}")
    task.log(f"超参: lr={lr}（骨干 {lr_backbone}）| 损失: Consistent-Point"
             + (f" + {loss_name}" if loss_name.startswith("custom:") else "")
             + "（缺省对齐论文；界面优化器/调度器/损失可按需调整）")
    set_seed(seed)
    # 教师-学生双模型常驻（约 2 倍单模型内存）：推理模型缓存让位
    _n_cached = clear_inference_cache()
    if _n_cached:
        task.log(f"已清空推理模型缓存（{_n_cached} 个），为训练让出内存")
    stopper = EarlyStopper(early_stop, mode="min")

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    if params.get("multi_gpu_mode") and torch.cuda.device_count() > 1:
        task.log("Consistent-Point 暂不支持多 GPU 并行，本次按单卡训练")
    device = torch.device(device_str)

    full = CPointSemiDataset(dataset_path, train=True,
                             label_ratio=label_ratio, crop_size=crop_size,
                             augment=augment, augment_strength=augment_strength,
                             seed=seed, val_ratio=val_ratio)
    if label_ratio < 1.0:
        task.log(f"训练图像 {len(full.items)} 张：有标注 {len(full.label)} / "
                 f"无标注 {len(full.unlabel)}（label_ratio={label_ratio}，"
                 f"种子确定性划分）")
    else:
        task.log(f"训练图像 {len(full.items)} 张（label_ratio=1，纯监督，"
                 f"无 L^U 分支）")
    train_loader = DataLoader(full, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              collate_fn=CPointSemiDataset.collate_train)
    if len(train_loader) == 0:
        raise ValueError(
            f"训练批数为 0（训练图像 {len(full.items)} < Batch Size "
            f"{batch_size}，启用 drop_last），请减小 Batch Size 或补充数据")
    eval_items = full.eval_set()
    task.log(f"评估样本: {len(eval_items)} 张（" +
             ("数据集 Test/Val 划分（完整划分）"
              if getattr(full, "_eval_from_test", False)
              else "训练集 val_ratio 切分（未找到 Test/Val 划分）") + "）")

    student, criterion = _build_cpoint(pretrained=pretrained, training=True)
    _lw = all_weights(params, {"ce_w": 1.0, "points_w": 0.0002,
                               "cost_class": 1.0, "cost_point": 0.05})
    apply_criterion_weights(criterion, _lw)
    apply_matcher_costs(criterion, _lw)
    task.log(f"损失权重: {describe(_lw)}")
    if weight_path:
        _load_weights_compat(student, weight_path, context="初始权重")
    teacher = ModelEMA(student, decay=ema_decay)
    student.to(device)
    teacher.module.to(device)
    criterion.to(device)
    weight_dict = criterion.weight_dict

    param_dicts = [
        {"params": [p for n, p in student.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in student.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": lr_backbone},
    ]
    from app.core.optim_builders import build_torch_optimizer
    optimizer = build_torch_optimizer(
        params, param_dicts, lr,
        weight_decay=float(params.get("weight_decay", 1e-4)))
    # 论文 StepLR 在训练周期内不衰减 → 默认常量学习率（可切换 9 种调度器）
    scheduler, _is_plateau = build_crowd_scheduler(
        optimizer, params.get("scheduler") or "cosine", epochs, params)
    task.log(f"优化器: {params.get('optimizer') or 'adam'}（骨干 lr "
             f"{lr_backbone}） | 调度器: {params.get('scheduler') or 'cosine'}")
    # 自定义点预测附加损失（默认不启用；契约 aux(outputs, targets)）
    _aux_w = aux_weight(params)
    aux_loss = None
    if loss_name.startswith("custom:"):
        from app.core.registry import is_custom_loss_for, get_custom_loss
        if not is_custom_loss_for(loss_name, "crowd_point"):
            raise ValueError(
                f"自定义损失 {loss_name} 未注册到 crowd_point 路线"
                f"（请在注册时写 tasks=[\"crowd_point\"]）")
        aux_loss = get_custom_loss(loss_name, task="crowd_point")
        if hasattr(aux_loss, "to"):
            aux_loss = aux_loss.to(device)
        task.log(f"自定义点预测附加损失: {loss_name}（契约：aux(outputs, targets)）")
    clip_max_norm = float(params.get("clip_max_norm", 0.1))

    csv_path = save_dir / "metrics.csv"
    _csv_header = ["epoch", "train_loss", "val_loss", "val_mae"]
    best_mae, start_epoch = float("inf"), 1
    best_path, last_path = _ckpt_paths(save_dir, params.get("model"))
    if params.get("resume_weight"):
        start_epoch, best_mae, _bm = resume_training_state(
            params["resume_weight"], student, optimizer, scheduler,
            best_key="val_mae", best_mode="min")
        task.best = _bm
        task.log(f"断点续训：从 epoch {start_epoch} 继续（best_mae={best_mae}）")
        try:
            _ck = torch.load(params["resume_weight"], map_location="cpu")
            if isinstance(_ck, dict) and isinstance(
                    _ck.get("teacher_state_dict"), dict):
                teacher.load_state_dict(_ck["teacher_state_dict"])
                task.log("教师模型状态已从续训权重恢复")
        except Exception as e:
            task.log(f"教师状态恢复失败（回退为学生拷贝）: {e}")
        reload_task_history(task, csv_path, start_epoch, _csv_header)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_csv_header)

    def _validate():
        """全图逐张评估：MAE + 与训练同口径的检测损失（含自定义附加损失）"""
        student.eval()
        mae_sum, n = 0.0, 0
        loss_sum, nl = 0.0, 0
        total = max(1, len(eval_items))
        log_every = max(1, total // 10)
        task.log(f"  开始验证（测试集）：{total} 张图像（整图逐张，含损失）")
        t0 = time.time()
        with torch.no_grad():
            for img_p, gt_p in eval_items:
                if task.stop_event.is_set():
                    raise StopIteration()
                x, target, _ = full._eval_item(img_p, gt_p)
                out = student(x.unsqueeze(0).to(device))
                tgt = [{k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in target.items()}]
                try:
                    ld = criterion(out, tgt)
                    lv = sum(ld[k] * weight_dict[k]
                             for k in ld if k in weight_dict)
                    if aux_loss is not None:
                        extra = aux_loss(out, tgt)
                        if isinstance(extra, dict):
                            extra = sum(v for v in extra.values())
                        lv = lv + _aux_w * extra
                    if torch.isfinite(lv):
                        loss_sum += float(lv)
                        nl += 1
                except Exception as e:
                    task.log(f"  验证损失计算跳过: {e}")
                scores = out["pred_logits"].softmax(-1)[0, :, 1]
                pred_cnt = int((scores > pseudo_thr).sum().item())
                mae_sum += abs(pred_cnt - int(target["point"].shape[0]))
                n += 1
                if n % log_every == 0 or n >= total:
                    task.log(
                        f"  验证 {n}/{total} | loss "
                        f"{loss_sum / max(1, nl):.4f} | "
                        f"MAE {mae_sum / max(1, n):.2f} | "
                        f"用时 {time.time() - t0:.1f}s")
        student.train()
        return (mae_sum / max(1, n), loss_sum / nl if nl else None)

    has_unlabeled = len(full.unlabel) > 0
    for epoch in range(start_epoch, epochs + 1):
        if task.stop_event.is_set():
            raise StopIteration()
        do_eval = ((eval_every <= 1) or (epoch % eval_every == 0)
                   or (epoch == epochs))
        student.train()
        teacher.module.eval()
        total_loss, n_batches = 0.0, 0
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)
        _main = is_main_process(params)
        for batch_idx, (limg, ltarget, _lid, uimg_w, uimg_s, uids) in \
                enumerate(train_loader):
            if task.stop_event.is_set():
                raise StopIteration()
            limg = limg.to(device)
            ltarget = [{k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in t.items()} for t in ltarget]

            out_l = student(limg)
            loss_dict_l = criterion(out_l, ltarget)
            loss_l = sum(loss_dict_l[k] * weight_dict[k]
                         for k in loss_dict_l if k in weight_dict)

            if has_unlabeled:
                uimg_w = uimg_w.to(device)
                uimg_s = uimg_s.to(device)
                with torch.no_grad():
                    out_t = teacher.module(uimg_w)
                    probs = out_t["pred_logits"].softmax(-1)[..., 1]
                    pa = position_aggregation(
                        out_t["pred_points"], probs,
                        out_t["anchor_points"], threshold=pseudo_thr,
                        K=pa_k, img_size=uimg_w.shape[-2:])
                    utargets = []
                    for r, uid in zip(pa, uids):
                        utargets.append({
                            "point": r["points"],
                            "labels": torch.ones(
                                r["points"].shape[0], dtype=torch.long,
                                device=device),
                            "weight": iuc_weights(r["scores"]),
                            "image_id": uid})
                out_u = student(uimg_s)
                loss_dict_u = criterion.forward_unlabeled(out_u, utargets)
                loss_u = sum(loss_dict_u[k] * weight_dict[k]
                             for k in loss_dict_u if k in weight_dict)
                loss = loss_l + lambda_u * loss_u
            else:
                loss_dict_u = {}
                loss_u = torch.zeros((), device=device)
                loss = loss_l
            if aux_loss is not None:
                extra = aux_loss(out_l, ltarget)
                if isinstance(extra, dict):
                    extra = sum(v for v in extra.values())
                loss = loss + _aux_w * extra

            optimizer.zero_grad()
            loss.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(),
                                               clip_max_norm)
            optimizer.step()
            teacher.update(student)
            total_loss += loss.item()
            n_batches += 1
            report_batch_progress(
                task, epoch=epoch, epochs=epochs, batch_idx=batch_idx,
                total_batches=total_batches, running_loss=total_loss,
                n_batches=n_batches, log_every=log_every, is_main=_main)

        train_loss = total_loss / max(1, n_batches)
        val_mae, val_loss = _validate() if do_eval else (None, None)
        if _is_plateau:
            if val_mae is not None:
                scheduler.step(val_mae)
        else:
            scheduler.step()

        task.epoch = epoch
        task.progress = epoch / epochs * 100.0
        task.metrics = {"train_loss": round(train_loss, 5)}
        if val_mae is not None:
            task.metrics["val_mae"] = round(val_mae, 3)
        if val_loss is not None:
            task.metrics["val_loss"] = round(val_loss, 5)
        task.history.append({"epoch": epoch, **task.metrics})
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.5f}",
                 "" if val_loss is None else f"{val_loss:.5f}",
                 "" if val_mae is None else f"{val_mae:.3f}"])
        task.log(f"Epoch {epoch}/{epochs} | loss {train_loss:.5f}"
                 + (f" | val_loss {val_loss:.5f}"
                    if val_loss is not None else "")
                 + (f" | val_MAE {val_mae:.3f}"
                    if val_mae is not None else ""))
        save_loss_curves(save_dir, task.history, ("val_mae", "val_loss"))

        is_best = val_mae is not None and val_mae < best_mae
        if is_best:
            best_mae = val_mae
            task.best = {"epoch": epoch, "val_mae": round(val_mae, 3)}
        save_checkpoint(last_path, net=student, optimizer=optimizer,
                        scheduler=scheduler, epoch=epoch,
                        best_metric=task.best, model="cp",
                        val_mae=val_mae, teacher_state_dict=teacher.state_dict())
        if is_best:
            save_checkpoint(best_path, net=student, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch,
                            best_metric=task.best, model="cp",
                            val_mae=val_mae)
        if val_mae is not None and stopper.step(val_mae):
            task.log(f"早停触发：MAE 已 {stopper.patience} 轮未提升")
            break
        # 每轮收尾及时释放：gc 循环垃圾 + CUDA 缓存显存归还
        release_training_memory()

    task.result = {"best_weight": str(best_path), "last_weight": str(last_path),
                   "best_val_mae": round(best_mae, 3), "save_dir": str(save_dir),
                   "model": "cp"}
    task.log(f"训练结束，最优 val_MAE={best_mae:.3f}")


def cpoint_infer(image: Image.Image, weight_path=None, device="cpu",
                 pretrained=True, conf_threshold=0.5):
    """Consistent-Point 推理：score>阈值的提议点即人头（与训练期评估口径
    一致），返回人数 + 红点可视化 + 坐标列表"""
    import base64
    import io
    import cv2

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    if not weight_path:
        raise ValueError("Consistent-Point 推理需要权重：请选择训练产物"
                         "（*_best.pth / *_last.pth）或上传 .pth"
                         "（无内置检查点）")
    cache_key = _model_cache_key("cp", weight_path, device)
    model = _model_cache_get(cache_key)
    if model is None:
        model = _build_cpoint(pretrained=False)
        ck = torch.load(weight_path, map_location="cpu")
        sd = ck.get("model_state_dict") or ck.get("state_dict") \
            if isinstance(ck, dict) else ck
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
              for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.to(dev).eval()
        _model_cache_put(cache_key, model)

    img = image.convert("RGB")
    W0, H0 = img.size
    # 限制长边（显存/内存），16 倍数右/下 padding（FPN 对齐；点坐标不变）
    max_side = 1536
    scale = min(1.0, max_side / max(H0, W0))
    new_w = max(16, int(round(W0 * scale)) // 16 * 16)
    new_h = max(16, int(round(H0 * scale)) // 16 * 16)
    img_in = img.resize((new_w, new_h), Image.BILINEAR) \
        if (new_w, new_h) != (W0, H0) else img
    x = transforms.functional.to_tensor(img_in)
    x = transforms.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    x = x.unsqueeze(0).to(dev)

    with torch.no_grad():
        out = model(x)
    scores = out["pred_logits"].softmax(-1)[0, :, 1]
    keep = scores > float(conf_threshold)
    points = out["pred_points"][0][keep].detach().cpu().numpy()
    if (new_w, new_h) != (W0, H0):
        points = points * np.array([W0 / new_w, H0 / new_h],
                                   dtype=np.float32)

    canvas = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    size = max(2, int(max(H0, W0) / 300))
    for p in points:
        canvas = cv2.circle(canvas, (int(p[0]), int(p[1])), size,
                            (0, 0, 255), -1)
    vis = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(vis).save(buf, format="PNG")
    vis_b64 = base64.b64encode(buf.getvalue()).decode()

    return {"count": int(len(points)), "visualization": vis_b64,
            "num_points": int(len(points)),
            "points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in points[:2000]],
            "conf_threshold": float(conf_threshold), "model": "cp",
            "point_based": True}


# ------------------- CU（Calibrating Uncertainty）半监督点预测 -------------------

class CUSemiDataset(Dataset):
    """CU 半监督数据集（vendor 逻辑来自本仓库 Uncertainty 复现的
    SemiCrowdDataset，数据布局遵循框架约定）

    - 标注格式与 P2R 路线一致：.txt / .mat / .npy((N,2) 坐标)
    - label_ratio < 1：种子确定性划分「有标注子集」，其余进无标注池
      （CU 论文协议：10% 标注，其余作无标注数据）
    - 有标注样本：弱几何增强（缩放 [0.7,1.3] → 裁 crop → 翻转）+ GT +
      裁剪几何信息（有标注索引 / 有效缩放 / 裁剪原点 / 翻转标志）——
      训练时把不确定性子块映射回原图坐标系查询 PatchBank 的 ASM
    - 无标注样本：同一几何变换得到弱视图（教师输入），其上叠加 CUTOUT
      强增强得到强视图（学生输入），并返回擦除框坐标（框内伪点丢弃）
    - 评估：全图归一化 + 16 倍数右/下 padding（点坐标不变），batch=1
    """

    def __init__(self, root, train=True, label_ratio=1.0, crop_size=128,
                 augment=True, augment_strength=1.0, seed=42, val_ratio=0.2,
                 cutout_size=32, cutout_p=0.5):
        self.root = Path(root)
        self.train = train
        self.crop_size = max(16, int(crop_size) or 128)
        self.augment = augment and train
        self.aug_strength = max(0.0, min(2.0, augment_strength))
        self._seed = seed
        self._val_ratio = val_ratio
        self.cutout_size = max(4, int(cutout_size))
        self.cutout_p = float(cutout_p)
        self._test_items = None
        self._eval_from_test = False

        img_dir, gt_dir = _crowd_dirs(self.root, train)
        if img_dir is None:
            raise ValueError(f"未找到图像目录: {root}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        cand_dirs = [gt_dir] if gt_dir is not None else \
            [self.root / d for d in ("gt", "npys", "labels")]
        items = []
        for p in sorted(img_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            for d in cand_dirs:
                if d is None or not d.is_dir():
                    continue
                hit = next((d / (p.stem + suf)
                            for suf in (".txt", ".mat", ".npy")
                            if (d / (p.stem + suf)).exists()), None)
                if hit is not None:
                    items.append((p, hit))
                    break
        if not items:
            raise ValueError(
                f"未找到点标注样本（CU 需要 .txt / .mat 点标注或 (N,2) 坐标 "
                f".npy）：{root}")
        self.items = items
        if not train:
            return

        stems = [p.stem for p, _ in items]
        label_ratio = max(0.0, min(1.0, float(label_ratio)))
        if label_ratio < 1.0:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(stems), generator=g).tolist()
            n_lab = max(1, min(len(stems) - 1,
                               round(len(stems) * label_ratio)))
            labeled = {stems[i] for i in perm[:n_lab]}
        else:
            labeled = set(stems)
        self.label = [(p, g) for p, g in items if p.stem in labeled]
        self.unlabel = [(p, g) for p, g in items if p.stem not in labeled]

    def eval_set(self):
        """评估集：优先 Test 目录；无 Test 时按 val_ratio 从训练集确定性切分"""
        if self._test_items is not None:
            return self._test_items
        test_img, test_gt = _crowd_dirs(self.root, False)
        if test_img is not None and test_gt is not None:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            its = []
            for p in sorted(test_img.iterdir()):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                for suf in (".txt", ".mat", ".npy"):
                    g = test_gt / (p.stem + suf)
                    if g.exists():
                        its.append((p, g))
                        break
            if its:
                self._test_items = its
                self._eval_from_test = True
                return its
        g = torch.Generator().manual_seed(self._seed)
        perm = torch.randperm(len(self.items), generator=g).tolist()
        n_val = max(1, int(len(self.items) * self._val_ratio))
        self._test_items = [self.items[i] for i in perm[:n_val]]
        return self._test_items

    # bank 依赖：整图 + GT 点（与训练裁剪同一坐标系）；norm：归一化张量
    def _load(self, item):
        img = Image.open(item[0]).convert("RGB")
        pts = _load_points_gt_any(item[1])
        return img, pts

    def norm(self, img):
        return transforms.functional.normalize(
            transforms.functional.to_tensor(img), IMAGENET_MEAN,
            IMAGENET_STD)

    def _weak_geometric(self, img, points):
        """缩放 + 随机裁剪 + 翻转（图像与坐标同步；CU 弱增强）。
        返回 (img, points, geo)：geo 携带裁剪几何（供 bank 查询回原图坐标系）"""
        import random as _rnd
        c = self.crop_size
        w, h = img.size
        s = _rnd.uniform(0.7, 0.7 + 0.6 * self.aug_strength) \
            if self.augment else 1.0
        if s * min(w, h) > c:
            nw, nh = max(int(w * s), c), max(int(h * s), c)
            img = img.resize((nw, nh), Image.BILINEAR)
            if points.shape[0] > 0:
                points = points * np.array([nw / w, nh / h],
                                           dtype=np.float32)
            w, h = nw, nh
        if w < c or h < c:
            canvas = Image.new("RGB", (max(w, c), max(h, c)))
            canvas.paste(img, (0, 0))
            img = canvas
            w, h = img.size
        sx = _rnd.randint(0, w - c)
        sy = _rnd.randint(0, h - c)
        img = img.crop((sx, sy, sx + c, sy + c))
        if points.shape[0] > 0:
            inside = ((points[:, 0] >= sx) & (points[:, 0] < sx + c) &
                      (points[:, 1] >= sy) & (points[:, 1] < sy + c))
            points = points[inside] - np.array([sx, sy], dtype=np.float32)
        flip = self.augment and _rnd.random() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if points.shape[0] > 0:
                points = points.copy()
                points[:, 0] = (c - 1) - points[:, 0]
        geo = {"scale": s, "crop": (sx, sy), "flip": flip}
        return img, points, geo

    @staticmethod
    def _pack(points, sid):
        pts = torch.as_tensor(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
        return {"point": pts,
                "labels": torch.ones(pts.shape[0], dtype=torch.long),
                "image_id": sid}

    def _eval_item(self, img_p, gt_p):
        img, pts = self._load((img_p, gt_p))
        x = self.norm(img)
        h, w = x.shape[-2:]
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, self._pack(pts, img_p.stem), img_p.stem

    def __len__(self):
        if not self.train:
            return len(self.items)
        # 训练迭代按无标注池索引（无无标注图时按有标注池长度）
        return len(self.unlabel) or len(self.label)

    def __getitem__(self, index):
        import random as _rnd
        if not self.train:
            img_p, gt_p = self.items[index]
            return self._eval_item(img_p, gt_p)
        # 有标注：随机抽一张弱增强 + GT + 裁剪几何（bank 查询用）
        lidx = _rnd.randrange(len(self.label))
        litem = self.label[lidx]
        lid = litem[0].stem
        limg, lpts = self._load(litem)
        limg, lpts, geo = self._weak_geometric(limg, lpts)
        limg = self.norm(limg)
        ltarget = self._pack(lpts, lid)
        geo["idx"] = lidx
        # 无标注：同一几何 → 弱视图（教师）+ CUTOUT 强增强 → 强视图（学生）
        if self.unlabel:
            uitem = self.unlabel[index % len(self.unlabel)]
            uid = uitem[0].stem
            uimg, _ = self._load(uitem)
            uimg, _, _ = self._weak_geometric(
                uimg, np.zeros((0, 2), dtype=np.float32))
            wa = self.norm(uimg)
            sa = wa.clone()
            cutbox = None
            if self.augment and _rnd.random() < self.cutout_p:
                c = self.crop_size
                bs = min(self.cutout_size, c // 2)
                bx = _rnd.randint(0, c - bs)
                by = _rnd.randint(0, c - bs)
                sa[:, by:by + bs, bx:bx + bs] = 0
                cutbox = (bx, by, bs)
            return limg, ltarget, geo, wa, sa, cutbox, uid
        # 纯监督（label_ratio=1）：无标注分支缺席，占位对齐批结构
        dummy = torch.zeros_like(limg)
        return limg, ltarget, geo, dummy, dummy, None, lid

    @staticmethod
    def collate_train(samples):
        limgs, ltargets, lgeos, was, sas, cutboxes, uids = zip(*samples)
        return (torch.stack(limgs, 0), list(ltargets), list(lgeos),
                torch.stack(was, 0), torch.stack(sas, 0),
                list(cutboxes), list(uids))


def _build_cu(pretrained=True, training=False):
    """构建 CU 模型（training=True 时同时返回损失器）"""
    from nets.cu import cu_default_args
    from nets.cu.p2pnet import build as _build_cu_model
    args = cu_default_args()
    args.no_pretrained = not pretrained
    return _build_cu_model(args, training=training)


def run_cu_train(task, params: dict):
    """CU 训练（ICCV 2023 "Calibrating Uncertainty for Semi-Supervised
    Crowd Counting"，arXiv:2308.09887 半监督点预测）

    学生损失（论文 Eq. 3）：L = L_pred(D_L) + L_uncer(D_L) + λ₁·L_pred(D_伪)
    - L_pred(D_L)：有标注 P2PNet 损失（分类 + 定位）
    - L_uncer：不确定性分支 vs PatchBank ASM 代理真值的成对排序损失
      （batch min-max 归一化，论文 Eq. 2；有标注数据）
    - L_pred(D_伪)：课程不确定性阈值 u_t 过滤后的伪点 P2PNet 损失
      （λ₁=0.3；u_t 从 0 线性升至 u_max，仅保留可靠 patch 的伪点）
    教师为学生的 EMA（0.999，每步更新），并周期性对 PatchBank 做快照
    （时间累积 ASM）。优化器 / 学习率 / 调度器 / 损失均可在界面调整；
    默认对齐论文（Adam 1e-4 / 骨干 1e-5 + 恒定学习率 + P2PNet 匹配损失），
    损失下拉默认 CU，另可追加注册到 crowd_point 的自定义附加损失。
    """
    from app.services.train_common import (set_seed, EarlyStopper,
                                            save_checkpoint, save_loss_curves,
                                            resume_training_state,
                                            reload_task_history,
                                            release_training_memory,
                                            report_batch_progress,
                                            is_main_process)
    from nets.cu import ModelEMA, PatchBank, ranking_loss

    dataset_path = params["dataset_path"]
    epochs = int(params.get("epochs", 30))
    eval_every = max(1, int(params.get("eval_every") or 1))
    batch_size = int(params.get("batch_size", 2))
    # lr=0 / 缺省 → 论文默认（CU Adam 主 lr 1e-4、骨干 1e-5）
    lr = float(params.get("lr") or 1e-4)
    lr_backbone = float(params.get("lr_backbone") or 1e-5)
    loss_name = str(params.get("loss") or "cu")
    patch_size = int(params.get("patch_size", 64))
    crop_size = int(params.get("img_size", 0)) or 256
    # 裁剪尺寸对齐 patch 网格（128 / 64 = 2x2 不确定性单元）
    crop_size = max(patch_size, (crop_size // patch_size) * patch_size)
    pretrained = bool(params.get("pretrained", True))
    weight_path = params.get("weight_path") or None
    device_str = params.get("device", "cpu")
    num_workers = int(params.get("num_workers", 0))
    early_stop = int(params.get("early_stopping", 0))
    seed = int(params.get("seed", 42))
    label_ratio = float(params.get("label_ratio", 1.0))
    augment = bool(params.get("augment", True))
    augment_strength = float(params.get("augment_strength", 1.0))
    val_ratio = float(params.get("val_ratio", 0.2))
    # CU 超参（论文 Sec. 3.4 / 实现细节；warmup 未指定时按 10% 轮数）
    lambda_u = float(params.get("lambda_u", 0.3))       # λ₁ 论文 Table 8
    uncer_coef = float(params.get("uncer_coef", 1.0))   # 排序损失权重
    warmup = int(params.get("warmup_epochs", 0)) or \
        max(1, round(epochs * 0.1))
    u_max = float(params.get("u_max", 0.5))             # 课程阈值上限
    pseudo_thr = float(params.get("pseudo_threshold", 0.5))
    ema_decay = float(params.get("ema_decay", 0.999))
    asm_every = int(params.get("asm_every", 1))
    bank_context = int(params.get("bank_context", 128))
    bank_batch = int(params.get("bank_batch", 16))
    cutout_size = int(params.get("cutout_size", 32))
    cutout_p = float(params.get("cutout_p", 0.5))
    if bank_context < patch_size:
        bank_context = patch_size

    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    task.total_epochs = epochs
    task.log("CU 点预测训练（半监督）| P2PNet 基座 + 不确定性分支 + "
             "PatchBank ASM + 排序损失 + 课程伪标签过滤 + Mean-Teacher")
    task.log(f"数据集: {dataset_path} | 设备: {device_str} | 裁剪: {crop_size} "
             f"| patch: {patch_size}")
    task.log(f"超参: lr={lr}（骨干 {lr_backbone}）| 损失: CU"
             + (f" + {loss_name}" if loss_name.startswith("custom:") else "")
             + "（缺省对齐论文；界面优化器/调度器/损失可按需调整）")
    task.log(f"λ₁={lambda_u} | warmup={warmup} 轮 | u_max={u_max} | "
             f"EMA={ema_decay} | 快照间隔={asm_every} 轮")
    set_seed(seed)
    # 教师-学生双模型常驻（约 2 倍单模型内存）：推理模型缓存让位
    _n_cached = clear_inference_cache()
    if _n_cached:
        task.log(f"已清空推理模型缓存（{_n_cached} 个），为训练让出内存")
    stopper = EarlyStopper(early_stop, mode="min")

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    if params.get("multi_gpu_mode") and torch.cuda.device_count() > 1:
        task.log("CU 暂不支持多 GPU 并行，本次按单卡训练")
    device = torch.device(device_str)

    full = CUSemiDataset(dataset_path, train=True, label_ratio=label_ratio,
                         crop_size=crop_size, augment=augment,
                         augment_strength=augment_strength, seed=seed,
                         val_ratio=val_ratio, cutout_size=cutout_size,
                         cutout_p=cutout_p)
    if label_ratio < 1.0:
        task.log(f"训练图像 {len(full.items)} 张：有标注 {len(full.label)} / "
                 f"无标注 {len(full.unlabel)}（label_ratio={label_ratio}，"
                 f"种子确定性划分）")
    else:
        task.log(f"训练图像 {len(full.items)} 张（label_ratio=1，纯监督，"
                 f"无伪标签分支）")
    train_loader = DataLoader(full, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              collate_fn=CUSemiDataset.collate_train)
    if len(train_loader) == 0:
        raise ValueError(
            f"训练批数为 0（训练图像 {len(full.items)} < Batch Size "
            f"{batch_size}，启用 drop_last），请减小 Batch Size 或补充数据")
    eval_items = full.eval_set()
    task.log(f"评估样本: {len(eval_items)} 张（" +
             ("数据集 Test/Val 划分（完整划分）"
              if getattr(full, "_eval_from_test", False)
              else "训练集 val_ratio 切分（未找到 Test/Val 划分）") + "）")

    student, criterion = _build_cu(pretrained=pretrained, training=True)
    _lw = all_weights(params, {"ce_w": 1.0, "points_w": 0.0002,
                               "cost_class": 1.0, "cost_point": 0.05})
    apply_criterion_weights(criterion, _lw)
    apply_matcher_costs(criterion, _lw)
    task.log(f"损失权重: {describe(_lw)}")
    if weight_path:
        _load_weights_compat(student, weight_path, context="初始权重")
    teacher = ModelEMA(student, decay=ema_decay)
    student.to(device)
    teacher.module.to(device)
    criterion.to(device)
    weight_dict = criterion.weight_dict

    param_dicts = [
        {"params": [p for n, p in student.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in student.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": lr_backbone},
    ]
    from app.core.optim_builders import build_torch_optimizer
    optimizer = build_torch_optimizer(
        params, param_dicts, lr,
        weight_decay=float(params.get("weight_decay", 1e-4)))
    # 论文 StepLR 在训练周期内不衰减 → 默认常量学习率（可切换 9 种调度器）
    scheduler, _is_plateau = build_crowd_scheduler(
        optimizer, params.get("scheduler") or "cosine", epochs, params)
    task.log(f"优化器: {params.get('optimizer') or 'adam'}（骨干 lr "
             f"{lr_backbone}） | 调度器: {params.get('scheduler') or 'cosine'}")
    # 自定义点预测附加损失（默认不启用；契约 aux(outputs, targets)）
    _aux_w = aux_weight(params)
    aux_loss = None
    if loss_name.startswith("custom:"):
        from app.core.registry import is_custom_loss_for, get_custom_loss
        if not is_custom_loss_for(loss_name, "crowd_point"):
            raise ValueError(
                f"自定义损失 {loss_name} 未注册到 crowd_point 路线"
                f"（请在注册时写 tasks=[\"crowd_point\"]）")
        aux_loss = get_custom_loss(loss_name, task="crowd_point")
        if hasattr(aux_loss, "to"):
            aux_loss = aux_loss.to(device)
        task.log(f"自定义点预测附加损失: {loss_name}（契约：aux(outputs, targets)）")
    clip_max_norm = float(params.get("clip_max_norm", 0.1))

    # PatchBank：有标注图像的 P×P 网格（原图坐标系），时间累积 ASM 代理真值
    bank = PatchBank.build(full, patch_size=patch_size,
                           bank_context=bank_context, log=task.log)
    n_grid = max(1, crop_size // patch_size)
    has_unlabeled = len(full.unlabel) > 0

    csv_path = save_dir / "metrics.csv"
    _csv_header = ["epoch", "train_loss", "val_loss", "val_mae"]
    best_mae, start_epoch = float("inf"), 1
    best_path, last_path = _ckpt_paths(save_dir, params.get("model"))
    if params.get("resume_weight"):
        start_epoch, best_mae, _bm = resume_training_state(
            params["resume_weight"], student, optimizer, scheduler,
            best_key="val_mae", best_mode="min")
        task.best = _bm
        task.log(f"断点续训：从 epoch {start_epoch} 继续（best_mae={best_mae}）")
        try:
            _ck = torch.load(params["resume_weight"], map_location="cpu")
            if isinstance(_ck, dict):
                if isinstance(_ck.get("teacher_state_dict"), dict):
                    teacher.load_state_dict(_ck["teacher_state_dict"])
                    task.log("教师模型状态已从续训权重恢复")
                if isinstance(_ck.get("bank_state_dict"), dict):
                    try:
                        bank.load_state_dict(_ck["bank_state_dict"])
                        task.log("PatchBank ASM 累积状态已恢复")
                    except ValueError as e:
                        task.log(f"bank 状态忽略（数据/patch 变更）: {e}")
        except Exception as e:
            task.log(f"教师/bank 状态恢复失败（回退默认）: {e}")
        reload_task_history(task, csv_path, start_epoch, _csv_header)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_csv_header)

    def _curriculum(epoch):
        """课程伪标签阈值 u_t：warmup 期 0（关闭无标注分支），之后线性升至 u_max"""
        if epoch < warmup:
            return 0.0
        span = max(epochs - warmup, 1)
        return u_max * min(1.0, (epoch - warmup) / span)

    def _validate():
        """全图逐张评估：MAE + 与训练同口径的检测损失（含自定义附加损失）"""
        student.eval()
        mae_sum, n = 0.0, 0
        loss_sum, nl = 0.0, 0
        total = max(1, len(eval_items))
        log_every = max(1, total // 10)
        task.log(f"  开始验证（测试集）：{total} 张图像（整图逐张，含损失）")
        t0 = time.time()
        with torch.no_grad():
            for img_p, gt_p in eval_items:
                if task.stop_event.is_set():
                    raise StopIteration()
                x, target, _ = full._eval_item(img_p, gt_p)
                out = student(x.unsqueeze(0).to(device))
                tgt = [{k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in target.items()}]
                try:
                    ld = criterion(out, tgt)
                    lv = sum(ld[k] * weight_dict[k]
                             for k in ld if k in weight_dict)
                    if aux_loss is not None:
                        extra = aux_loss(out, tgt)
                        if isinstance(extra, dict):
                            extra = sum(v for v in extra.values())
                        lv = lv + _aux_w * extra
                    if torch.isfinite(lv):
                        loss_sum += float(lv)
                        nl += 1
                except Exception as e:
                    task.log(f"  验证损失计算跳过: {e}")
                scores = out["pred_logits"].softmax(-1)[0, :, 1]
                pred_cnt = int((scores > pseudo_thr).sum().item())
                mae_sum += abs(pred_cnt - int(target["point"].shape[0]))
                n += 1
                if n % log_every == 0 or n >= total:
                    task.log(
                        f"  验证 {n}/{total} | loss "
                        f"{loss_sum / max(1, nl):.4f} | "
                        f"MAE {mae_sum / max(1, n):.2f} | "
                        f"用时 {time.time() - t0:.1f}s")
        student.train()
        return (mae_sum / max(1, n), loss_sum / nl if nl else None)

    for epoch in range(start_epoch, epochs + 1):
        if task.stop_event.is_set():
            raise StopIteration()
        do_eval = ((eval_every <= 1) or (epoch % eval_every == 0)
                   or (epoch == epochs))
        student.train()
        teacher.module.eval()
        u_t = _curriculum(epoch)
        use_unlabeled = has_unlabeled and epoch > warmup and lambda_u > 0
        total_loss, n_batches = 0.0, 0
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)
        _main = is_main_process(params)
        for batch_idx, (limgs, ltargets, lgeos, uimgs_w, uimgs_s, cutboxes,
                        uids) in enumerate(train_loader):
            if task.stop_event.is_set():
                raise StopIteration()
            limgs = limgs.to(device)
            ltargets = [{k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in t.items()} for t in ltargets]

            # ---- 有标注：L_pred(D_L) + L_uncer(D_L)（排序损失） ----
            out_l = student(limgs)
            loss_dict_l = criterion(out_l, ltargets)
            loss_l = sum(loss_dict_l[k] * weight_dict[k]
                         for k in loss_dict_l if k in weight_dict)
            uncer_map = out_l["pred_uncer"]          # [B,1,G,G]
            a_vals, k_vals = [], []
            for b, geo in enumerate(lgeos):
                for qv in range(n_grid):
                    for qu in range(n_grid):
                        a = bank.query(geo["idx"], geo["scale"],
                                       geo["crop"], geo["flip"], qu, qv)
                        if a is not None:
                            a_vals.append(a)
                            k_vals.append(uncer_map[b, 0, qv, qu])
            if len(a_vals) >= 2:
                asm = torch.tensor(a_vals, dtype=torch.float32,
                                   device=device)
                kappa = torch.stack(k_vals)
                loss_uncer = ranking_loss(asm, kappa)
            else:
                loss_uncer = uncer_map.new_zeros(())
            loss = loss_l + uncer_coef * loss_uncer

            # ---- 无标注：课程过滤伪点（score 阈值 + 不确定性阈值 + 擦除框） ----
            loss_u = limgs.new_zeros(())
            if use_unlabeled:
                uimgs_w = uimgs_w.to(device)
                uimgs_s = uimgs_s.to(device)
                with torch.no_grad():
                    out_t = teacher.module(uimgs_w)
                    probs = out_t["pred_logits"].softmax(-1)[..., 1]
                    unc_t = out_t["pred_uncer"]
                    crop = uimgs_w.shape[-1]
                    utargets = []
                    for b in range(uimgs_w.shape[0]):
                        keep = probs[b] > pseudo_thr
                        pts = out_t["pred_points"][b][keep].cpu()
                        pts = pts.clamp(min=0, max=crop - 1)
                        cutbox = cutboxes[b]
                        sel = torch.ones(pts.shape[0], dtype=torch.bool)
                        for t in range(pts.shape[0]):
                            qu_ = min(int(pts[t, 0]) // patch_size,
                                      n_grid - 1)
                            qv_ = min(int(pts[t, 1]) // patch_size,
                                      n_grid - 1)
                            # 不可靠子块（不确定性 >= u_t）的伪点丢弃
                            if not (unc_t[b, 0, qv_, qu_] < u_t):
                                sel[t] = False
                                continue
                            # CUTOUT 擦除框内的伪点丢弃（无监督信号的区域）
                            if cutbox is not None:
                                bx, by, bs = cutbox
                                if (bx <= pts[t, 0] < bx + bs and
                                        by <= pts[t, 1] < by + bs):
                                    sel[t] = False
                        pts = pts[sel]
                        if pts.shape[0] == 0:
                            continue    # 无可靠 patch -> 无监督
                        utargets.append({
                            "point": pts.to(device),
                            "labels": torch.ones(
                                pts.shape[0], dtype=torch.long,
                                device=device),
                            "image_id": uids[b]})
                if utargets:
                    out_s = student(uimgs_s)
                    loss_dict_u = criterion(out_s, utargets)
                    loss_u = sum(loss_dict_u[k] * weight_dict[k]
                                 for k in loss_dict_u if k in weight_dict)
                    loss = loss + lambda_u * loss_u

            if aux_loss is not None:
                extra = aux_loss(out_l, ltargets)
                if isinstance(extra, dict):
                    extra = sum(v for v in extra.values())
                loss = loss + _aux_w * extra

            optimizer.zero_grad()
            loss.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(),
                                               clip_max_norm)
            optimizer.step()
            teacher.update(student)
            total_loss += loss.item()
            n_batches += 1
            report_batch_progress(
                task, epoch=epoch, epochs=epochs, batch_idx=batch_idx,
                total_batches=total_batches, running_loss=total_loss,
                n_batches=n_batches, log_every=log_every, is_main=_main)

        # 训练完成后先做测试集评估（日志/进度立即可见），再刷新 PatchBank
        train_loss = total_loss / max(1, n_batches)
        val_mae, val_loss = _validate() if do_eval else (None, None)
        if _is_plateau:
            if val_mae is not None:
                scheduler.step(val_mae)
        else:
            scheduler.step()

        task.epoch = epoch
        task.progress = epoch / epochs * 100.0
        task.metrics = {"train_loss": round(train_loss, 5)}
        if val_mae is not None:
            task.metrics["val_mae"] = round(val_mae, 3)
        if val_loss is not None:
            task.metrics["val_loss"] = round(val_loss, 5)
        task.history.append({"epoch": epoch, **task.metrics})
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.5f}",
                 "" if val_loss is None else f"{val_loss:.5f}",
                 "" if val_mae is None else f"{val_mae:.3f}"])
        task.log(f"Epoch {epoch}/{epochs} | loss {train_loss:.5f}"
                 + (f" | val_loss {val_loss:.5f}"
                    if val_loss is not None else "")
                 + (f" | val_MAE {val_mae:.3f}"
                    if val_mae is not None else "")
                 + f" | u_t {u_t:.3f}"
                 f"{'（无标注分支启用）' if use_unlabeled else ''}")
        save_loss_curves(save_dir, task.history, ("val_mae", "val_loss"))

        # PatchBank 快照（时间累积 ASM，EMA 教师；间隔 asm_every 轮）
        # 放于评估之后：测试集损失/进度先输出，ASM 快照过程内部也会打印进度
        if asm_every > 0 and epoch % asm_every == 0:
            bank.update_asm(teacher.module, full, device,
                            score_thresh=pseudo_thr,
                            batch_size=bank_batch, seed=seed + epoch,
                            log=task.log)

        is_best = val_mae is not None and val_mae < best_mae
        if is_best:
            best_mae = val_mae
            task.best = {"epoch": epoch, "val_mae": round(val_mae, 3)}
        save_checkpoint(last_path, net=student, optimizer=optimizer,
                        scheduler=scheduler, epoch=epoch,
                        best_metric=task.best, model="cu",
                        val_mae=val_mae,
                        teacher_state_dict=teacher.state_dict(),
                        bank_state_dict=bank.state_dict())
        if is_best:
            save_checkpoint(best_path, net=student, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch,
                            best_metric=task.best, model="cu",
                            val_mae=val_mae)
        if val_mae is not None and stopper.step(val_mae):
            task.log(f"早停触发：MAE 已 {stopper.patience} 轮未提升")
            break
        # 每轮收尾及时释放：gc 循环垃圾 + CUDA 缓存显存归还
        release_training_memory()

    task.result = {"best_weight": str(best_path), "last_weight": str(last_path),
                   "best_val_mae": round(best_mae, 3), "save_dir": str(save_dir),
                   "model": "cu"}
    task.log(f"训练结束，最优 val_MAE={best_mae:.3f}")


def cu_infer(image: Image.Image, weight_path=None, device="cpu",
             pretrained=True, conf_threshold=0.5):
    """CU 推理：score>阈值的提议点即人头（与训练期评估口径一致），
    返回人数 + 红点可视化 + 坐标列表（兼容框架训练产物与独立复现项目的
    {'model': ...} checkpoint 两种格式）"""
    import base64
    import io
    import cv2

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    if not weight_path:
        raise ValueError("CU 推理需要权重：请选择训练产物"
                         "（*_best.pth / *_last.pth）或上传 .pth"
                         "（无内置检查点）")
    cache_key = _model_cache_key("cu", weight_path, device)
    model = _model_cache_get(cache_key)
    if model is None:
        model = _build_cu(pretrained=False)
        ck = torch.load(weight_path, map_location="cpu")
        sd = None
        if isinstance(ck, dict):
            sd = ck.get("model_state_dict") or ck.get("state_dict")
            if sd is None and isinstance(ck.get("model"), dict):
                sd = ck["model"]          # 独立复现项目 checkpoint 格式
        if not isinstance(sd, dict) or not sd:
            raise ValueError(f"权重文件中未找到模型参数: {weight_path}")
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
              for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.to(dev).eval()
        _model_cache_put(cache_key, model)

    img = image.convert("RGB")
    W0, H0 = img.size
    # 限制长边（显存/内存），16 倍数右/下 padding（FPN 对齐；点坐标不变）
    max_side = 1536
    scale = min(1.0, max_side / max(H0, W0))
    new_w = max(16, int(round(W0 * scale)) // 16 * 16)
    new_h = max(16, int(round(H0 * scale)) // 16 * 16)
    img_in = img.resize((new_w, new_h), Image.BILINEAR) \
        if (new_w, new_h) != (W0, H0) else img
    x = transforms.functional.to_tensor(img_in)
    x = transforms.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    x = x.unsqueeze(0).to(dev)

    with torch.no_grad():
        out = model(x)
    scores = out["pred_logits"].softmax(-1)[0, :, 1]
    keep = scores > float(conf_threshold)
    points = out["pred_points"][0][keep].detach().cpu().numpy()
    if (new_w, new_h) != (W0, H0):
        points = points * np.array([W0 / new_w, H0 / new_h],
                                   dtype=np.float32)

    canvas = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    size = max(2, int(max(H0, W0) / 300))
    for p in points:
        canvas = cv2.circle(canvas, (int(p[0]), int(p[1])), size,
                            (0, 0, 255), -1)
    vis = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(vis).save(buf, format="PNG")
    vis_b64 = base64.b64encode(buf.getvalue()).decode()

    return {"count": int(len(points)), "visualization": vis_b64,
            "num_points": int(len(points)),
            "points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in points[:2000]],
            "conf_threshold": float(conf_threshold), "model": "cu",
            "point_based": True}


def run_crowd_train(task, params: dict):
    model_name = params.get("model", "ghostnetv2_10")
    if model_name == "raz":
        from app.services import raz_service
        return raz_service.run_raz_train(task, params)
    if model_name == "d2c":
        from app.services import d2c_service
        return d2c_service.run_d2c_train(task, params)
    if model_name == "p2pnet":
        return run_p2pnet_train(task, params)
    if model_name == "p2r":
        return run_p2r_train(task, params)
    if model_name == "cp":
        return run_cpoint_train(task, params)
    if model_name == "cu":
        return run_cu_train(task, params)
    if model_name == "pet":
        from app.services import pet_service
        return pet_service.run_pet_train(task, params)
    if model_name == "cltr":
        from app.services import cltr_service
        return cltr_service.run_cltr_train(task, params)
    # 自定义点预测模型：走 P2PNet 管线（模型须实现 P2PNet 输出契约）
    from app.core.registry import is_custom_model_for
    if params.get("task") == "crowd_point" and \
            is_custom_model_for(model_name, "crowd_point"):
        return run_p2pnet_train(task, params)
    dataset_path = params["dataset_path"]
    epochs = int(params.get("epochs", 30))
    eval_every = max(1, int(params.get("eval_every") or 1))
    batch_size = int(params.get("batch_size", 2))
    # lr=0 / 缺省 → 论文默认（密度网 Adam 1e-5）
    lr = float(params.get("lr") or 1e-5)
    weight_decay = float(params.get("weight_decay", 1e-4))
    crop_size = int(params.get("img_size", 384))
    loss_name = params.get("loss", "mse")
    pretrained = bool(params.get("pretrained", True))
    weight_path = params.get("weight_path") or None
    device_str = params.get("device", "cpu")
    num_workers = int(params.get("num_workers", 0))
    sigma = float(params.get("sigma", 8.0))
    adaptive = bool(params.get("adaptive_sigma", False))
    val_ratio = float(params.get("val_ratio", 0.2))
    early_stop = int(params.get("early_stopping", 0))
    seed = int(params.get("seed", 42))
    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    task.total_epochs = epochs
    task.log(f"人群计数训练 | 模型: {model_name} | 损失: {loss_name}")
    from app.services.train_common import (set_seed, EarlyStopper,
                                           save_checkpoint, save_loss_curves,
                                           release_training_memory,
                                           report_batch_progress,
                                           is_main_process)
    set_seed(seed)
    # 训练是内存峰值场景：驻留的推理模型缓存及时让位（推理/摄像头按需重建）
    _n_cached = clear_inference_cache()
    if _n_cached:
        task.log(f"已清空推理模型缓存（{_n_cached} 个），为训练让出内存")
    stopper = EarlyStopper(early_stop, mode="min")

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    augment = bool(params.get("augment", True))
    augment_strength = float(params.get("augment_strength", 1.0))
    full = CrowdDataset(dataset_path, crop_size=crop_size, sigma=sigma,
                        adaptive=adaptive, train=True, augment=augment,
                        augment_strength=augment_strength)
    # 验证集：优先数据集自带 Test 划分；否则按 val_ratio 从训练集切分
    val_eval_ds, val_src = _eval_from_test(
        dataset_path,
        lambda: CrowdDataset(dataset_path, crop_size=crop_size, sigma=sigma,
                             adaptive=adaptive, train=False, augment=False))
    if val_eval_ds is not None:
        train_ds = full
        val_ds = val_eval_ds
        n_val, n_train = len(val_ds), len(full)
    else:
        n_val = max(1, int(len(full) * val_ratio))
        n_train = len(full) - n_val
        if n_train < 1:
            raise ValueError("训练样本不足")
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full), generator=g).tolist()
        train_ds = torch.utils.data.Subset(full, indices[n_val:])
        # 验证集复用同一批样本，但走评估分支（不增强、不随机裁剪到 crop）：
        # 否则验证 MAE 每轮都在随机增强视图上计算，不可复现且偏噪声
        import copy as _copy
        val_full = _copy.copy(full)
        val_full.train = False
        val_full.augment = False
        val_ds = torch.utils.data.Subset(val_full, indices[:n_val])
    # GT 来源构成（M11）：.npy 的 σ 以生成时为准，.txt 现场生成用本次 sigma，
    # 两者混用时监督目标尺度不同，显式告知
    n_npy = sum(1 for _, t in full.items if t == "npy")
    task.log(f"GT 构成: .npy {n_npy} 个（密度图以文件为准） | "
             f".txt {len(full.items) - n_npy} 个（现场生成，sigma={sigma}）")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers)
    task.log(f"训练样本: {n_train} | 验证样本: {n_val}（{val_src}）")

    model = _build_crowd_model(model_name, pretrained=pretrained)
    if weight_path:
        # 初始权重（H4）：校验匹配度，架构错配直接报错而非静默半加载
        _load_weights_compat(model, weight_path, context="初始权重")
    from app.services.train_common import (wrap_model_distributed,
                                            maybe_distributed_sampler,
                                            ddp_set_epoch)
    model = wrap_model_distributed(model, device_str, params, task=task)
    model.to(device)
    _ddp_sampler = maybe_distributed_sampler(train_ds, params)
    if _ddp_sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=False, sampler=_ddp_sampler,
                                  num_workers=num_workers, pin_memory=True)

    # 损失：mse/mae/ssim/mse_ssim/dmcount（crowd_losses）+ custom: 自定义
    if str(loss_name).startswith("custom:"):
        from app.core.registry import get_custom_loss
        criterion = get_custom_loss(loss_name, task="crowd_counting")
    elif loss_name in ("mse", "mae", "ssim", "mse_ssim", "dmcount"):
        from app.services.crowd_losses import build_crowd_loss
        _kw = {}
        if loss_name == "mse_ssim":
            # MSE / SSIM 组合权重（缺省 0.5 / 0.5，与框架原设置一致）
            _kw = all_weights(params, {"mse_w": 0.5, "ssim_w": 0.5})
            task.log(f"损失权重: {describe(_kw)}")
        elif loss_name == "dmcount":
            # DM-Count（NeurIPS 2020）：wot 0.1 / wtv 0.01 / count 1.0
            _lw = all_weights(params, {"ot_w": 0.1, "tv_w": 0.01,
                                       "count_w": 1.0})
            _kw = {"wot": _lw["ot_w"], "wtv": _lw["tv_w"],
                   "count_weight": _lw["count_w"]}
            task.log(f"损失权重: {describe(_lw)}")
        criterion = build_crowd_loss(loss_name, **_kw)
    else:
        raise ValueError(
            f"未知计数损失: {loss_name}"
            f"（支持 mse/mae/ssim/mse_ssim/dmcount/custom:*）")
    # SSIM 系损失持有 Parameter（高斯核），必须与输入同设备，否则 GPU 训练崩溃
    criterion.to(device)
    if loss_name == "dmcount":
        task.log(
            f"DM-Count 损失（NeurIPS 2020）：计数 + {criterion.wot}·OT"
            f"（Sinkhorn 熵正则 reg={criterion.reg}，{criterion.iters} 次迭代）"
            f" + {criterion.wtv}·TV；OT 目标测度 = 归一化 GT 密度"
            f"（stride={criterion.stride}，网格上限 "
            f"{criterion.max_ot_cells} 单元）；验证阶段跳过 OT 项")
    # 优化器可选，默认 adam
    from app.core.optim_builders import build_torch_optimizer
    optimizer = build_torch_optimizer(params, model.parameters(), lr,
                                      weight_decay=weight_decay)
    # 统一调度器（9 种，详见 build_crowd_scheduler）
    scheduler, is_plateau = build_crowd_scheduler(
        optimizer, params.get("scheduler"), epochs, params)
    if is_plateau:
        task.log("调度器: ReduceLROnPlateau（按 val_mae 自适应）")
    else:
        task.log(f"调度器: {params.get('scheduler') or 'cosine'}")

    csv_path = save_dir / "metrics.csv"
    _csv_header = ["epoch", "train_loss", "val_loss", "val_mae"]
    best_mae = float("inf")
    best_path, last_path = _ckpt_paths(save_dir, model_name)
    # 断点续训：恢复 模型/optimizer/scheduler/epoch/最优指标
    start_epoch = 1
    if params.get("resume_weight"):
        from app.services.train_common import (resume_training_state,
                                               reload_task_history)
        start_epoch, best_mae, _bm = resume_training_state(
            params["resume_weight"], model, optimizer, scheduler,
            best_key="val_mae", best_mode="min")
        task.best = _bm
        task.log(f"断点续训：从 epoch {start_epoch} 继续"
                 f"（best_mae={best_mae}）")
        reload_task_history(task, csv_path, start_epoch, _csv_header)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_csv_header)

    for epoch in range(start_epoch, epochs + 1):
        if task.stop_event.is_set():
            raise StopIteration()
        do_eval = ((eval_every <= 1) or (epoch % eval_every == 0)
                   or (epoch == epochs))
        ddp_set_epoch(train_loader, epoch, params)
        model.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)
        _main = is_main_process(params)
        for batch_idx, (imgs, dens) in enumerate(train_loader):
            if task.stop_event.is_set():
                raise StopIteration()
            imgs = imgs.to(device, non_blocking=True)
            dens = dens.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(imgs)
            if pred.shape[-2:] != dens.shape[-2:]:
                pred = F.interpolate(pred, size=dens.shape[-2:], mode="bilinear",
                                     align_corners=False)
            loss = criterion(pred, dens.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            report_batch_progress(
                task, epoch=epoch, epochs=epochs, batch_idx=batch_idx,
                total_batches=total_batches, running_loss=total_loss,
                n_batches=batch_idx + 1, log_every=log_every,
                is_main=_main)
        train_loss = total_loss / max(1, len(train_loader))

        # 验证（MAE = |预测计数 - 真实计数| 均值；同时输出验证损失与进度）
        # eval_every 控制评估间隔：非评估轮跳过（最后一轮总会评估）
        model.eval()
        val_loss = val_mae = None
        total_val = max(1, len(val_loader))
        log_every = max(1, total_val // 10)
        if do_eval:
            task.log(f"  开始验证（测试集）：{total_val} 批（含验证损失）")
            t0 = time.time()
            val_loss, val_mae = 0.0, 0.0
            with torch.no_grad():
                for vi, (imgs, dens) in enumerate(val_loader, 1):
                    if task.stop_event.is_set():
                        raise StopIteration()
                    imgs = imgs.to(device)
                    dens = dens.to(device)
                    pred = model(imgs)
                    if pred.shape[-2:] != dens.shape[-2:]:
                        pred = F.interpolate(pred, size=dens.shape[-2:],
                                             mode="bilinear",
                                             align_corners=False)
                    val_loss += criterion(pred, dens.unsqueeze(1)).item()
                    val_mae += abs(pred.sum().item() - dens.sum().item())
                    if vi % log_every == 0 or vi >= total_val:
                        task.log(
                            f"  验证 {vi}/{total_val} | loss "
                            f"{val_loss / max(1, vi):.5f} | "
                            f"MAE {val_mae / max(1, vi):.3f} | "
                            f"用时 {time.time() - t0:.1f}s")
            val_loss /= total_val
            val_mae /= total_val

        # 调度器步进：plateau 按 val_mae（仅评估轮），其余按 epoch
        if is_plateau:
            if do_eval:
                scheduler.step(val_mae)
        else:
            scheduler.step()

        task.epoch = epoch
        task.progress = epoch / epochs * 100.0
        task.metrics = {"train_loss": round(train_loss, 5)}
        if do_eval:
            task.metrics["val_loss"] = round(val_loss, 5)
            task.metrics["val_mae"] = round(val_mae, 3)
        task.history.append({"epoch": epoch, **task.metrics})
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.5f}",
                 "" if not do_eval else f"{val_loss:.5f}",
                 "" if not do_eval else f"{val_mae:.3f}"])
        task.log(f"Epoch {epoch}/{epochs} | loss {train_loss:.5f}"
                 + (f" | val_loss {val_loss:.5f} | val_MAE {val_mae:.3f}"
                    if do_eval else "（本轮跳过测试集评估）"))
        save_loss_curves(save_dir, task.history, ("val_mae", "val_loss"))

        is_best = do_eval and val_mae < best_mae
        if is_best:
            best_mae = val_mae
            task.best = {"epoch": epoch, "val_mae": round(val_mae, 3)}
        save_checkpoint(best_path if is_best else last_path,
                        net=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch, best_metric=task.best, model=model_name,
                        val_mae=val_mae)
        if is_best:
            save_checkpoint(last_path, net=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch,
                            best_metric=task.best, model=model_name,
                            val_mae=val_mae)
        if val_mae is not None and stopper.step(val_mae):
            task.log(f"早停触发：MAE 已 {stopper.patience} 轮未提升")
            break
        # 每轮收尾及时释放：gc 回收图窗等循环垃圾 + CUDA 缓存显存归还驱动
        # （同卡并发训练 / 训练后推理不再被空闲缓存块拖到 OOM）
        release_training_memory()

    task.result = {"best_weight": str(best_path), "last_weight": str(last_path),
                   "best_val_mae": round(best_mae, 3), "save_dir": str(save_dir)}
    task.log(f"训练结束，最优 val_MAE={best_mae:.3f}")


# ------------------------- 独立测试模式 -------------------------

def _test_summary(errs):
    """Count errors -> MAE / RMSE (and the legacy mse alias)."""
    arr = np.asarray(errs, dtype=np.float64)
    mae = float(np.mean(np.abs(arr))) if arr.size else 0.0
    rmse = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
    return round(mae, 3), round(rmse, 3)


def _test_write_csv(path, rows):
    if not path:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "pred", "gt", "error"])
        w.writeheader()
        w.writerows(rows)


def _density_tiled_count(model, img_t, patch):
    """Density model count with optional non-overlapping patch inference.

    img_t is a normalized [1,3,H,W] tensor. Tiles are padded to a multiple of
    32 so common stride-8 density backbones accept the last row/column.
    """
    _, _, h, w = img_t.shape
    if not patch or (h <= patch and w <= patch):
        return float(model(img_t)[0, 0].sum().item()), 0
    total = 0.0
    tiles = 0
    for y in range(0, h, patch):
        for x in range(0, w, patch):
            tile = img_t[:, :, y:y + patch, x:x + patch]
            th, tw = tile.shape[-2:]
            ph = (32 - th % 32) % 32
            pw = (32 - tw % 32) % 32
            if ph or pw:
                tile = F.pad(tile, (0, pw, 0, ph))
            total += float(model(tile)[0, 0].sum().item())
            tiles += 1
    return total, tiles


def run_density_test(task, params: dict):
    """Density route: evaluate a trained weight on a dataset test split.

    Only inference is performed (no optimizer / no gradients). Metrics are
    count-level MAE and RMSE; per-image rows are written to test_results.csv.
    """
    model_name = params.get("model") or "ghostnetv2_10"
    weight_path = params.get("weight_path")
    if not weight_path or not Path(weight_path).is_file():
        raise ValueError("测试需要训练产物或上传的权重文件（.pth/.pt）")
    root = params["dataset_path"]
    split = (params.get("split") or "test").lower()
    device_str = params.get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    patch = max(0, int(params.get("patch_size") or 0))
    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "test_results.csv"

    def _make(train):
        return CrowdDataset(root, crop_size=patch or 384, train=train,
                            augment=False)

    use_train = split == "train"
    try:
        ds = _make(use_train)
    except Exception as exc:
        if use_train:
            raise
        task.log(f"未找到 Test/Val 划分（{exc}），回退到 Train 划分")
        use_train = True
        ds = _make(True)
    task.log(f"密度图测试 | 模型: {model_name} | 权重: {weight_path}")
    task.log(f"数据集: {root} | 划分: {'Train' if use_train else 'Test/Val'} "
             f"| 样本: {len(ds)} | patch: {patch or '整图'}")
    model = _build_crowd_model(model_name, pretrained=False)
    _load_weights_compat(model, weight_path, context="测试权重")
    model.to(device).eval()
    rows, errs = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            if task.stop_event.is_set():
                raise StopIteration()
            img_t, target = ds[i]
            gt_count = float(target.sum().item())
            if patch:
                pred_count, n_tiles = _density_tiled_count(
                    model, img_t.unsqueeze(0).to(device), patch)
            else:
                pred_count = float(model(img_t.unsqueeze(0).to(device))
                                   [0, 0].sum().item())
                n_tiles = 0
            pred_count = max(0.0, pred_count)
            err = pred_count - gt_count
            errs.append(err)
            rows.append({"name": ds.items[i][0].stem,
                         "pred": round(pred_count, 4),
                         "gt": round(gt_count, 4),
                         "error": round(err, 4)})
            task.progress = (i + 1) / len(ds) * 100.0
            if (i + 1) % 20 == 0 or i + 1 == len(ds):
                task.log(f"  [{i + 1}/{len(ds)}] pred={pred_count:.2f} "
                         f"gt={gt_count:.2f}"
                         + (f" tiles={n_tiles}" if n_tiles else "")
                         + f" | 耗时 {time.time() - t0:.1f}s")
    if not errs:
        raise ValueError("测试集没有有效样本")
    mae, rmse = _test_summary(errs)
    _test_write_csv(csv_path, rows)
    task.progress = 100.0
    task.result = {"mae": mae, "rmse": rmse, "mse": rmse,
                   "num_images": len(errs), "model": model_name,
                   "dataset": str(root), "patch_size": patch,
                   "per_image_csv": str(csv_path), "save_dir": str(save_dir)}
    task.log(f"测试完成：MAE={mae:.3f} | RMSE={rmse:.3f} | "
             f"样本={len(errs)} | 逐图结果={csv_path}")


def _point_gt_count(gt_path: Path, kind: str):
    if kind == "npy":
        arr = np.load(gt_path)
        return float(np.asarray(arr, dtype=np.float64).sum())
    return float(len(_load_points_gt(gt_path)))


def run_point_test(task, params: dict):
    """Point route: run the standard inference once per image and compare counts."""
    model_name = params.get("model") or "p2pnet"
    weight_path = params.get("weight_path")
    if not weight_path or not Path(weight_path).is_file():
        raise ValueError("测试需要训练产物或上传的权重文件（.pth/.pt）")
    root = params["dataset_path"]
    split = (params.get("split") or "test").lower()
    device_str = params.get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    patch = max(0, int(params.get("patch_size") or 0))
    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "test_results.csv"

    use_train = split == "train"
    try:
        ds = P2PPointsDataset(root, train=use_train, augment=False)
    except Exception as exc:
        if use_train:
            raise
        task.log(f"未找到 Test/Val 划分（{exc}），回退到 Train 划分")
        use_train = True
        ds = P2PPointsDataset(root, train=True, augment=False)
    task.log(f"点预测测试 | 模型: {model_name} | 权重: {weight_path}")
    task.log(f"数据集: {root} | 划分: {'Train' if use_train else 'Test/Val'} "
             f"| 样本: {len(ds)}"
             + (f" | 点模型忽略 patch={patch}" if patch else ""))
    rows, errs = [], []
    t0 = time.time()
    for i, (img_p, gt_p, kind) in enumerate(ds.items):
        if task.stop_event.is_set():
            raise StopIteration()
        gt_count = _point_gt_count(gt_p, kind)
        img = Image.open(img_p).convert("RGB")
        out = crowd_infer(img, model_name=model_name,
                          weight_path=str(weight_path), device=device_str,
                          pretrained=False)
        pred_count = float(out.get("count") or 0.0)
        err = pred_count - gt_count
        errs.append(err)
        rows.append({"name": Path(img_p).stem, "pred": round(pred_count, 4),
                     "gt": round(gt_count, 4), "error": round(err, 4)})
        task.progress = (i + 1) / len(ds) * 100.0
        if (i + 1) % 20 == 0 or i + 1 == len(ds):
            task.log(f"  [{i + 1}/{len(ds)}] pred={pred_count:.2f} "
                     f"gt={gt_count:.2f} | 耗时 {time.time() - t0:.1f}s")
    if not errs:
        raise ValueError("测试集没有有效样本")
    mae, rmse = _test_summary(errs)
    _test_write_csv(csv_path, rows)
    task.progress = 100.0
    task.result = {"mae": mae, "rmse": rmse, "mse": rmse,
                   "num_images": len(errs), "model": model_name,
                   "dataset": str(root), "patch_size": patch,
                   "per_image_csv": str(csv_path), "save_dir": str(save_dir)}
    task.log(f"测试完成：MAE={mae:.3f} | RMSE={rmse:.3f} | "
             f"样本={len(errs)} | 逐图结果={csv_path}")


# ------------------------- 推理 -------------------------

def crowd_infer(image: Image.Image, model_name="ghostnetv2_10",
                weight_path=None, device="cpu", pretrained=True,
                conf_threshold=0.5):
    """人群计数推理：返回估计人数 + 密度热力图（base64）"""
    import base64
    import io
    import matplotlib.cm as cm

    if model_name == "p2pnet":
        return p2pnet_infer(image, weight_path=weight_path, device=device,
                            pretrained=pretrained,
                            conf_threshold=conf_threshold)
    if model_name == "p2r":
        return p2r_infer(image, weight_path=weight_path, device=device,
                         pretrained=pretrained,
                         conf_threshold=conf_threshold)
    if model_name == "cp":
        return cpoint_infer(image, weight_path=weight_path, device=device,
                            pretrained=pretrained,
                            conf_threshold=conf_threshold)
    if model_name == "cu":
        return cu_infer(image, weight_path=weight_path, device=device,
                        pretrained=pretrained,
                        conf_threshold=conf_threshold)
    if model_name == "pet":
        from app.services import pet_service
        return pet_service.pet_infer(
            image, weight_path=weight_path, device=device,
            pretrained=pretrained, conf_threshold=conf_threshold)
    if model_name == "cltr":
        from app.services import cltr_service
        return cltr_service.cltr_infer(
            image, weight_path=weight_path, device=device,
            pretrained=pretrained, conf_threshold=conf_threshold)
    if model_name == "raz":
        from app.services import raz_service
        return raz_service.raz_infer(
            image, weight_path=weight_path, device=device,
            pretrained=pretrained, conf_threshold=conf_threshold)
    if model_name == "d2c":
        from app.services import d2c_service
        return d2c_service.d2c_infer(
            image, weight_path=weight_path, device=device,
            pretrained=pretrained, conf_threshold=conf_threshold)
    # 自定义点预测模型（按 tasks 分区识别）
    from app.core.registry import is_custom_model_for
    if is_custom_model_for(model_name, "crowd_point"):
        return _custom_point_infer(
            image, model_name, weight_path=weight_path, device=device,
            conf_threshold=conf_threshold)

    image = image.convert("RGB")
    W0, H0 = image.size
    orig_image = image          # 原图（可视化输出与密度图统一到该尺寸）
    max_side = 1536
    scale = min(1.0, max_side / max(H0, W0))
    if scale < 1.0:
        image = image.resize((int(W0 * scale), int(H0 * scale)))

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    # 模型缓存（H3）：同 (模型, 权重, 设备) 只构建一次，权重更新自动失效
    cache_key = _model_cache_key(model_name, weight_path, device)
    model = _model_cache_get(cache_key)
    if model is None:
        model = _build_crowd_model(model_name, pretrained=pretrained)
        if weight_path:
            # H4：校验匹配度，架构错配直接报错而非静默半加载
            _load_weights_compat(model, weight_path, context="推理权重")
        model.to(dev).eval()
        _model_cache_put(cache_key, model)

    x = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / \
        torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = x.to(dev)

    with torch.no_grad():
        density = model(x)[0, 0].cpu().numpy()
    count = float(density.sum())

    # 热力图可视化
    d = density - density.min()
    if d.max() > 1e-8:
        d = d / d.max()
    heat = (cm.jet(d)[..., :3] * 255).astype(np.uint8)
    # 密度热力图放大到【原图尺寸】（模型原生为 1/8 分辨率，直接返回会很小）
    heat_img = Image.fromarray(heat).resize((W0, H0), Image.BILINEAR)
    overlay = Image.blend(orig_image, heat_img, alpha=0.5)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    vis_b64 = base64.b64encode(buf.getvalue()).decode()

    buf2 = io.BytesIO()
    heat_img.save(buf2, format="PNG")
    heat_b64 = base64.b64encode(buf2.getvalue()).decode()

    return {"count": round(count, 2), "visualization": vis_b64,
            "density_map": heat_b64,
            "density_size": [H0, W0],
            "model": model_name}

