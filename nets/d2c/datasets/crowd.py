# -*- coding: utf-8 -*-
"""Datasets for D2CNet."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .adapters import DatasetSpec, build_dataset_spec
from .points import load_points
from .targets import dot_map, make_targets

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _to_rgb_tensor(img: Image.Image) -> torch.Tensor:
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


class CrowdDataset(Dataset):
    """Point-annotated dataset with probability/density/count/dot targets.

    Training: random scale + crop + flip -> 384x384 targets.
    Evaluation: resize long side to max_side, pad to multiple of 32.
    """

    def __init__(self, items: List[Tuple[Path, Path]], train: bool = True,
                 patch_size: int = 384, max_side: int = 1920,
                 sigma_mode: str = "adaptive", sigma: float = 15.0,
                 augment: bool = True, rf: int = 32,
                 allow_density_gt: bool = False,
                 density_threshold: float = 0.05):
        self.items = list(items)
        self.train = bool(train)
        self.patch = max(32, int(np.ceil(int(patch_size) / 32) * 32))
        self.max_side = max(0, int(max_side))
        self.sigma_mode = sigma_mode
        self.sigma = float(sigma)
        self.augment = bool(augment and train)
        self.rf = int(rf)
        self.allow_density_gt = bool(allow_density_gt)
        self.density_threshold = float(density_threshold)

    def __len__(self):
        return len(self.items)

    def _read(self, idx):
        img_path, gt_path = self.items[idx]
        with Image.open(img_path) as im:
            orig_size = im.size
            img = im.convert("RGB")
        pts = load_points(gt_path, image_size=orig_size,
                          allow_density=self.allow_density_gt,
                          density_threshold=self.density_threshold)
        return img, pts, Path(img_path), orig_size

    def __getitem__(self, idx) -> Dict:
        img, points, img_path, orig_size = self._read(idx)
        W, H = img.size

        if not self.train:
            # 长边限制，保证与官方 “QNRF 长边<=1920” 的设置一致。
            scale = 1.0
            if self.max_side > 0 and max(W, H) > self.max_side:
                scale = self.max_side / max(W, H)
                nw, nh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
                img = img.resize((nw, nh), Image.BILINEAR)
                points = points * np.array([nw / W, nh / H], np.float32)
                W, H = nw, nh
            rf = self.rf
            # 填充到 32 的倍数，使 CMR 的 1/32 计数图正好覆盖整图。
            nw = int(np.ceil(W / rf) * rf)
            nh = int(np.ceil(H / rf) * rf)
            if (nw, nh) != (W, H):
                canvas = Image.new("RGB", (nw, nh), (0, 0, 0))
                canvas.paste(img, (0, 0))
                img = canvas
                W, H = nw, nh
            dot = dot_map(points, H, W)
            return {
                "image": _to_rgb_tensor(img),
                "dot": torch.from_numpy(dot).float().unsqueeze(0),
                "points": torch.from_numpy(points.reshape(-1, 2)),
                "size": (W, H),
                "orig_size": (int(orig_size[0]), int(orig_size[1])),
                "scale": float(scale),
                "image_path": str(img_path),
            }

        # ---- training augmentation ----
        if self.augment:
            s = random.uniform(0.75, 1.0)
            if min(W, H) * s >= self.patch:
                nw, nh = max(1, int(W * s)), max(1, int(H * s))
                img = img.resize((nw, nh), Image.BILINEAR)
                if len(points):
                    points = points * np.array([nw / W, nh / H], np.float32)
                W, H = nw, nh

        # 小图先补黑边，保证可以裁出 patch。
        if W < self.patch or H < self.patch:
            nw = max(W, self.patch)
            nh = max(H, self.patch)
            canvas = Image.new("RGB", (nw, nh), (0, 0, 0))
            canvas.paste(img, (0, 0))
            img = canvas
            W, H = nw, nh

        x0 = random.randint(0, max(0, W - self.patch))
        y0 = random.randint(0, max(0, H - self.patch))
        img = img.crop((x0, y0, x0 + self.patch, y0 + self.patch))
        if len(points):
            points = points.copy()
            points[:, 0] -= x0
            points[:, 1] -= y0
            keep = ((points[:, 0] >= 0) & (points[:, 0] < self.patch) &
                    (points[:, 1] >= 0) & (points[:, 1] < self.patch))
            points = points[keep]

        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(points):
                points = points.copy()
                points[:, 0] = (self.patch - 1) - points[:, 0]

        prob, density, count, dot = make_targets(
            points, self.patch, self.patch,
            self.sigma_mode, self.sigma, self.rf)
        return {
            "image": _to_rgb_tensor(img),
            "prob": torch.from_numpy(prob).float().unsqueeze(0),
            "density": torch.from_numpy(density).float().unsqueeze(0),
            "count": torch.from_numpy(count).float().unsqueeze(0),
            "dot": torch.from_numpy(dot).float().unsqueeze(0),
            "points": torch.from_numpy(points.reshape(-1, 2)),
            "image_path": str(img_path),
        }


def collate_train(batch):
    keys = ("image", "prob", "density", "count", "dot")
    out = {k: torch.stack([b[k] for b in batch], 0) for k in keys}
    out["points"] = [b["points"] for b in batch]
    out["image_path"] = [b["image_path"] for b in batch]
    return out


def collate_eval(batch):
    return batch


def build_datasets(args):
    spec = build_dataset_spec(
        args.data_root, dataset=args.dataset, part=args.part or None,
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
        raise ValueError("训练集为空")
    if not spec.val:
        items = list(spec.train)
        random.Random(args.seed).shuffle(items)
        n_val = min(len(items) - 1,
                    max(1, int(round(len(items) * args.val_ratio))))
        if n_val <= 0:
            raise ValueError("样本不足，无法切分验证集")
        spec.train, spec.val = items[n_val:], items[:n_val]
    if not spec.test:
        spec.test = spec.val

    common = dict(patch_size=args.patch_size, max_side=args.max_side,
                  sigma_mode=args.sigma_mode, sigma=args.sigma,
                  allow_density_gt=args.allow_density_gt,
                  density_threshold=args.density_threshold)
    train_ds = CrowdDataset(spec.train, train=True,
                            augment=not args.no_augment, **common)
    val_ds = CrowdDataset(spec.val, train=False, augment=False, **common)
    test_ds = CrowdDataset(spec.test, train=False, augment=False, **common)
    return spec, train_ds, val_ds, test_ds
