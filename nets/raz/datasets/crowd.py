# -*- coding: utf-8 -*-
"""Torch datasets for RAZ Main-Net / RAZ-Net."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .adapters import DatasetSpec, build_dataset_spec
from .points import find_gt_for_image, iter_images, load_points
from .targets import make_targets

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if img.size == size:
        return img
    return img.resize(size, Image.BILINEAR)


class CrowdDataset(Dataset):
    """Point-annotated crowd dataset.

    Training samples are random square crops with density/localization/
    attention targets.  Evaluation samples are the full image (optionally
    limited to ``max_side``) plus the transformed points.
    """

    def __init__(self, items: Sequence[Tuple[Path, Path]], train: bool = True,
                 patch_size: int = 256, augment: bool = True,
                 max_side: int = 1536, attn_stride: int = 4,
                 density_stride: int = 8, beta: float = 0.1,
                 sigma_z: float = 3.0, allow_density_gt: bool = False,
                 density_threshold: float = 0.05):
        self.items = list(items)
        self.train = bool(train)
        self.patch_size = max(32, int(patch_size))
        self.augment = bool(augment and train)
        self.max_side = max(0, int(max_side))
        self.attn_stride = max(1, int(attn_stride))
        self.density_stride = max(1, int(density_stride))
        self.beta = float(beta)
        self.sigma_z = float(sigma_z)
        self.allow_density_gt = bool(allow_density_gt)
        self.density_threshold = float(density_threshold)

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, index: int):
        img_path, gt_path = self.items[index]
        with Image.open(img_path) as im:
            orig_size = im.size
            img = im.convert("RGB")
        pts = load_points(gt_path, image_size=orig_size,
                          allow_density=self.allow_density_gt,
                          density_threshold=self.density_threshold)
        return img, pts, Path(img_path), Path(gt_path), orig_size

    def __getitem__(self, index: int) -> Dict:
        img, points, img_path, gt_path, orig_size = self._load(index)
        W, H = img.size

        # 验证/推理：保持长宽比，把长边限制在 max_side；点坐标同步缩放。
        if not self.train:
            scale = 1.0
            if self.max_side > 0 and max(W, H) > self.max_side:
                scale = float(self.max_side) / max(W, H)
                nw, nh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
                img = _resize_image(img, (nw, nh))
                if len(points):
                    points = points * np.array([nw / W, nh / H], dtype=np.float32)
                W, H = nw, nh
            x = np.asarray(img, dtype=np.float32) / 255.0
            x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / \
                np.array(IMAGENET_STD, dtype=np.float32)
            tensor = torch.from_numpy(x.transpose(2, 0, 1)).float()
            return {
                "image": tensor,
                "pil": img,
                "points": torch.from_numpy(points.reshape(-1, 2)),
                "size": (W, H),
                "orig_size": (int(orig_size[0]), int(orig_size[1])),
                "scale": float(scale),
                "image_path": str(img_path),
                "gt_path": str(gt_path),
            }

        # ---- 训练增强：先保证短边 >= patch，再随机缩放/裁剪/翻转 ----
        # 所有空间变换都同步修改点坐标，保证图像与密度/定位/注意力一致。
        patch = self.patch_size
        if min(W, H) < patch:
            s = patch / max(1, min(W, H))
            nw, nh = max(patch, int(round(W * s))), max(patch, int(round(H * s)))
            img = _resize_image(img, (nw, nh))
            if len(points):
                points = points * np.array([nw / W, nh / H], dtype=np.float32)
            W, H = nw, nh

        if self.augment and random.random() < 0.5:
            s = random.uniform(0.8, 1.2)
            if min(int(W * s), int(H * s)) >= patch:
                nw, nh = max(1, int(W * s)), max(1, int(H * s))
                img = _resize_image(img, (nw, nh))
                if len(points):
                    points = points * np.array([nw / W, nh / H], dtype=np.float32)
                W, H = nw, nh

        # 随机裁剪到固定 patch×patch，保证同一 batch 能直接 stack。
        x0 = random.randint(0, max(0, W - patch))
        y0 = random.randint(0, max(0, H - patch))
        img = img.crop((x0, y0, x0 + patch, y0 + patch))
        if len(points):
            points = points.copy()
            points[:, 0] -= x0
            points[:, 1] -= y0
            inside = ((points[:, 0] > -1) & (points[:, 0] < patch + 1) &
                      (points[:, 1] > -1) & (points[:, 1] < patch + 1))
            points = points[inside]
            points[:, 0] = np.clip(points[:, 0], 0, patch - 1)
            points[:, 1] = np.clip(points[:, 1], 0, patch - 1)

        # 水平翻转时点坐标同步镜像；垂直翻转在拥挤人群里会破坏头部
        # 朝向先验，因此这里只做水平翻转。
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(points):
                points = points.copy()
                points[:, 0] = (patch - 1) - points[:, 0]

        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / \
            np.array(IMAGENET_STD, dtype=np.float32)
        image_t = torch.from_numpy(x.transpose(2, 0, 1)).float()

        # 三个分支共用同一组点，分别生成密度/定位/注意力监督。
        dens, loc, attn = make_targets(
            points, patch, patch,
            density_stride=self.density_stride,
            attn_stride=self.attn_stride,
            beta=self.beta, sigma_z=self.sigma_z
        )
        return {
            "image": image_t,
            "density": torch.from_numpy(dens).float().unsqueeze(0),
            "loc": torch.from_numpy(loc).float().unsqueeze(0),
            "attn": torch.from_numpy(attn).float().unsqueeze(0),
            "points": torch.from_numpy(points.reshape(-1, 2)),
            "image_path": str(img_path),
            "gt_path": str(gt_path),
        }

def collate_train(batch: List[Dict]) -> Dict:
    out = {}
    for key in ("image", "density", "loc", "attn"):
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["points"] = [b["points"] for b in batch]
    out["image_path"] = [b["image_path"] for b in batch]
    out["gt_path"] = [b["gt_path"] for b in batch]
    return out


def collate_eval(batch: List[Dict]) -> List[Dict]:
    return batch


def build_datasets(args) -> Tuple[DatasetSpec, CrowdDataset, CrowdDataset, CrowdDataset]:
    spec = build_dataset_spec(
        args.data_root, dataset=args.dataset, part=getattr(args, "part", None),
        val_ratio=args.val_ratio, seed=args.seed,
        manifest=getattr(args, "manifest", ""),
        train_sub=getattr(args, "train_sub", ""),
        test_sub=getattr(args, "test_sub", ""))
    if getattr(args, "max_train", 0):
        spec.train = spec.train[:int(args.max_train)]
    if getattr(args, "max_val", 0):
        spec.val = spec.val[:int(args.max_val)]
        spec.test = spec.test[:int(args.max_val)]
    if not spec.train:
        raise ValueError(f"训练集为空: {args.data_root}")
    if not spec.val:
        # Unified/custom datasets often have no Val split; derive one from
        # train instead of failing (the paper only needs a validation set).
        items = list(spec.train)
        random.Random(args.seed).shuffle(items)
        n_val = min(len(items) - 1, max(1, int(round(len(items) * args.val_ratio))))
        if n_val <= 0:
            raise ValueError("训练样本不足，无法切分验证集；请提供更多数据或官方 val")
        spec.train, spec.val = items[n_val:], items[:n_val]
    if not spec.test:
        spec.test = spec.val
    common = dict(patch_size=args.patch_size, max_side=args.max_side,
                  attn_stride=args.attn_stride, beta=args.beta,
                  sigma_z=args.sigma_z, allow_density_gt=args.allow_density_gt,
                  density_threshold=args.density_threshold)
    train_ds = CrowdDataset(spec.train, train=True,
                            augment=not args.no_augment, **common)
    val_ds = CrowdDataset(spec.val, train=False, augment=False, **common)
    test_items = spec.test or spec.val
    test_ds = CrowdDataset(test_items, train=False, augment=False, **common)
    return spec, train_ds, val_ds, test_ds
