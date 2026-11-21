# -*- coding: utf-8 -*-
"""Balanced synthetic probability/count patches for CMR augmentation.

The official code produces these maps with MATLAB.  This Python version
samples 32x32 cell counts with inverse-frequency probabilities from the real
training set, then scatters points and builds the same probability/count
targets on the fly.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .points import load_points
from .targets import count_map_from_density, gaussian_maps


class BalancedSyntheticSampler:
    def __init__(self, items: Sequence[Tuple[Path, Path]],
                 patch_size: int = 384, rf: int = 32,
                 sigma_mode: str = "adaptive", sigma: float = 15.0,
                 stat_images: int = 100, max_count: int = 40,
                 allow_density_gt: bool = False,
                 density_threshold: float = 0.05,
                 seed: int = 42):
        self.patch = max(32, int(patch_size // 32 * 32))
        self.rf = int(rf)
        self.sigma_mode = sigma_mode
        self.sigma = float(sigma)
        self.max_count = max(1, int(max_count))
        self.rng = random.Random(seed)
        self.counts = self._collect_counts(
            items, stat_images, allow_density_gt, density_threshold)
        if not self.counts:
            self.counts = list(range(1, self.max_count + 1))
        hist = np.bincount(np.asarray(self.counts, dtype=np.int64),
                           minlength=self.max_count + 1).astype(np.float64)
        # 低频计数给更高概率：对应论文的样本平衡。
        weights = np.zeros(self.max_count + 1, dtype=np.float64)
        for k in range(1, self.max_count + 1):
            if hist[k] > 0:
                weights[k] = 1.0 / hist[k]
        if weights.sum() <= 0:
            weights[1:] = 1.0
        self.weights = weights / weights.sum()

    @staticmethod
    def _collect_counts(items, n_images, allow_density, threshold):
        counts = []
        for img_path, gt_path in list(items)[:max(1, int(n_images))]:
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
                pts = load_points(gt_path, image_size=(W, H),
                                  allow_density=allow_density,
                                  density_threshold=threshold)
            except Exception:
                continue
            if len(pts) == 0:
                continue
            cx = np.clip((pts[:, 0] // 32).astype(int), 0, max(0, W // 32))
            cy = np.clip((pts[:, 1] // 32).astype(int), 0, max(0, H // 32))
            # 统计非空 32x32 单元的局部计数
            m = {}
            for x, y in zip(cx, cy):
                key = (int(x), int(y))
                m[key] = m.get(key, 0) + 1
            counts.extend(m.values())
        return counts

    def sample_batch(self, batch_size: int = 1):
        probs, counts = [], []
        for _ in range(max(1, int(batch_size))):
            patch = self.patch
            cells = patch // self.rf
            pts = []
            cell_counts = np.zeros((cells, cells), np.float32)
            ks = np.random.choice(np.arange(self.max_count + 1),
                                  size=(cells, cells), p=self.weights)
            for gy in range(cells):
                for gx in range(cells):
                    k = int(ks[gy, gx])
                    cell_counts[gy, gx] = k
                    for _k in range(k):
                        x = (gx + self.rng.random()) * self.rf
                        y = (gy + self.rng.random()) * self.rf
                        pts.append([x, y])
            prob, density = gaussian_maps(pts, patch, patch,
                                          self.sigma_mode, self.sigma)
            count = count_map_from_density(density, self.rf)
            probs.append(torch.from_numpy(prob).float()[None])
            counts.append(torch.from_numpy(count).float()[None])
        return torch.stack(probs, 0), torch.stack(counts, 0)
