# Patch Bank (CU paper Sec. 3.2, Fig. 4 "dual-stream" design).
#
# Random training crops change every epoch, so the ASM surrogate cannot be
# accumulated on them. Instead every LABELED training image is divided ONCE
# into a fixed grid of non-overlapping patch_size x patch_size patches (64x64
# in the original-image frame, i.e. after the deterministic long-side resize).
# The bank keeps, per patch, a running accumulation of the spatial matching
# distance between model predictions and GT:
#
#   ASM(patch) = (1/T) sum_t dist(Pred_t(patch), GT(patch))
#
# T snapshots are produced by evaluating the (EMA) teacher every
# `asm_every` epochs -- the running mean implements the paper's
# "temporal accumulation" and suppresses single-epoch randomness.
#
# During training, the uncertainty branch is supervised on the RANDOM 128x128
# labeled crops (data diversity stream): each crop contains
# (crop_size/patch_size)^2 = 2x2 uncertainty units, whose region is mapped
# back to the original-image frame (undo scale -> crop -> flip) and queried
# against the bank with an overlap-area-weighted mean of accumulated ASM.
import math
import time

import numpy as np
import torch

from .asm import spatial_matching_distance


class PatchBank:
    def __init__(self, patch_size=64, crop_size=128, bank_context=128):
        assert patch_size >= 8 and (patch_size & (patch_size - 1)) == 0
        assert bank_context >= patch_size
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.context = bank_context
        self.C = patch_size * math.sqrt(2)   # unmatched penalty (patch diagonal)

        # per-patch records (built deterministically from the labeled set)
        self.images = []          # bank image list (labeled items with >=1 patch)
        self._label_to_bank = {}  # labeled-set index -> bank image index
        self.img_idx = []         # [P] bank-image index of each patch
        self.px = []              # [P] patch x0 (original-image frame)
        self.py = []              # [P] patch y0
        self.gt = []              # [P] np[N,2] GT points, patch-local coords
        self._lookup = {}         # (bank_img_idx, gx, gy) -> patch index

        # ASM accumulation state
        self.asm_sum = np.zeros(0, dtype=np.float64)
        self.asm_cnt = np.zeros(0, dtype=np.int64)

    # --------------------------------------------------------------- build
    @classmethod
    def build(cls, dataset, patch_size=64, bank_context=128, log=print):
        """Build the bank over `dataset.label` (the labeled item list).

        Reuses dataset._load so patches live in exactly the same
        (long-side-resized) frame the training crops are cut from.
        """
        bank = cls(patch_size=patch_size,
                   crop_size=dataset.crop_size,
                   bank_context=bank_context)
        P = patch_size
        for label_i, item in enumerate(dataset.label):
            # 确定性缩放后的图 + GT 点（与训练裁剪同一坐标系）
            img, points = dataset._load(item)
            w, h = img.size
            if w < P or h < P:
                continue                          # 连一个 patch 都放不下
            b_i = len(bank.images)
            bank.images.append(item)
            bank._label_to_bank[label_i] = b_i    # 有标注集下标 -> bank 图像下标
            # 不重叠的 P×P 网格切分，每个格子一条 patch 记录
            for y0 in range(0, h - P + 1, P):
                for x0 in range(0, w - P + 1, P):
                    # 落在格子内的 GT 点，平移到 patch 局部坐标
                    inside = ((points[:, 0] >= x0) & (points[:, 0] < x0 + P) &
                              (points[:, 1] >= y0) & (points[:, 1] < y0 + P))
                    gt = points[inside] - np.array([x0, y0], dtype=np.float32)
                    idx = len(bank.img_idx)

                    bank.img_idx.append(b_i)
                    bank.px.append(x0)            # patch 左上角（原图坐标系）
                    bank.py.append(y0)
                    bank.gt.append(gt.astype(np.float32))
                    # 网格坐标 -> patch 下标（query 反查用）
                    bank._lookup[(b_i, x0 // P, y0 // P)] = idx
        n = len(bank.img_idx)
        bank.asm_sum = np.zeros(n, dtype=np.float64)
        bank.asm_cnt = np.zeros(n, dtype=np.int64)
        log('[bank] built: {} images -> {} patches ({}x{}, C={:.2f})'.format(
            len(bank.images), n, P, P, bank.C))
        return bank

    def __len__(self):
        return len(self.img_idx)

    # ----------------------------------------------------- ASM accumulation
    @torch.no_grad()
    def update_asm(self, model, dataset, device, score_thresh=0.5,
                   batch_size=16, max_patches=0, seed=0, log=print):
        """One snapshot: evaluate `model` (the EMA teacher) on every bank
        patch (with context window), accumulate the matching distance."""
        if len(self) == 0:
            return
        was_training = model.training
        model.eval()
        patch_ids = np.arange(len(self))
        if max_patches and max_patches < len(patch_ids):
            # 大 bank 时每轮可只抽部分 patch 做快照（seed 变化轮换覆盖面）
            rng = np.random.RandomState(seed)
            patch_ids = np.sort(rng.choice(patch_ids, size=max_patches,
                                           replace=False))
        # 按图像分组，整图只从磁盘加载/归一化一次
        by_img = {}
        for i in patch_ids:
            by_img.setdefault(self.img_idx[i], []).append(i)

        P, ctx = self.patch_size, self.context
        pad = (ctx - P) // 2          # patch 可能触及图像边缘时的边距
        n_done = 0
        n_imgs_done = 0
        n_imgs = max(1, len(by_img))
        log_every = max(1, n_imgs // 10)
        t0 = time.time()
        log(f"[bank] snapshot 开始：{n_imgs} 张标注图像 / {len(patch_ids)} 个 patch")
        for img_i, ids in by_img.items():
            img, _ = dataset._load(self.images[img_i])
            w, h = img.size
            full = dataset.norm(img)                       # [3, H, W] 归一化张量
            if full.shape[1] < ctx or full.shape[2] < ctx:  # 小图零填充到 ctx
                c = torch.zeros(3, max(ctx, full.shape[1]),
                                max(ctx, full.shape[2]))
                c[:, :full.shape[1], :full.shape[2]] = full
                full = c
            # 每个 patch 取一个以它为中心的 ctx×ctx 窗口（贴边时平移回来）
            windows, origins = [], []
            for i in ids:
                x0, y0 = self.px[i], self.py[i]
                wx0 = max(0, min(x0 - pad, full.shape[2] - ctx))
                wy0 = max(0, min(y0 - pad, full.shape[1] - ctx))
                windows.append(full[:, wy0:wy0 + ctx, wx0:wx0 + ctx])
                origins.append((wx0, wy0))
            for s in range(0, len(windows), batch_size):
                # 窗口整批前向（上下文帮助模型在 patch 边界处预测）
                batch = torch.stack(windows[s:s + batch_size]).to(device)
                out = model(batch)
                probs = out['pred_logits'].softmax(-1)[..., 1]  # [B, M]
                pts = out['pred_points']                          # [B, M, 2]
                for bi in range(pts.shape[0]):
                    i = ids[s + bi]
                    keep = probs[bi] > score_thresh   # 过滤低置信度预测点
                    pred = pts[bi][keep].cpu().numpy()
                    wx0, wy0 = origins[s + bi]
                    # patch 左上角在窗口坐标系中的位置
                    bx, by = self.px[i] - wx0, self.py[i] - wy0
                    # 只留落在 patch 格子内的预测点，再平移到 patch 局部系
                    inbox = ((pred[:, 0] >= bx) & (pred[:, 0] < bx + P) &
                             (pred[:, 1] >= by) & (pred[:, 1] < by + P))
                    pred_local = pred[inbox] - np.array([bx, by])
                    dist = spatial_matching_distance(pred_local, self.gt[i],
                                                     self.C)
                    # 时间累积：sum/cnt 的运行均值即论文的 ASM
                    self.asm_sum[i] += dist
                    self.asm_cnt[i] += 1
                    n_done += 1
            n_imgs_done += 1
            if n_imgs_done % log_every == 0 or n_imgs_done == n_imgs:
                log(f"[bank] snapshot {n_imgs_done}/{n_imgs} 图像 | "
                    f"{n_done} patch | 用时 {time.time() - t0:.1f}s")
        if was_training:
            model.train()
        log('[bank] snapshot: {} patches updated | mean ASM {:.3f}'.format(
            n_done, self.mean_asm()))

    # ------------------------------------------------------------- queries
    def asm(self, i):
        """Current accumulated ASM of patch i (None if never evaluated)."""
        if self.asm_cnt[i] <= 0:
            return None
        return float(self.asm_sum[i] / self.asm_cnt[i])

    def mean_asm(self):
        # 已评估 patch 的 ASM 均值（整体不确定度水位，仅用于日志）
        vals = self.asm_sum[self.asm_cnt > 0]
        return float(vals.mean() / 1.0) if len(vals) else 0.0

    def query(self, img_idx, scale, crop_xy, flip, qu, qv):
        """ASM surrogate for the (qu, qv) sub-patch of a training crop.

        Maps the sub-patch region of the crop back to the original-image
        frame (undo flip -> crop origin -> scale) and returns the
        overlap-area-weighted mean of accumulated ASM over bank patches
        (None when the region has no evaluated overlap).

        `img_idx` is the labeled-image index carried by the crop geometry
        (index into dataset.label).
        """
        b_i = self._label_to_bank.get(img_idx)
        if b_i is None:
            return None                    # 该图没进 bank（太小，建库时跳过）
        cx, cy = crop_xy
        P, Csz = self.patch_size, self.crop_size
        # 裁剪帧内子块区间 [x_lo, x_hi) × [y_lo, y_hi)；翻转时 x 轴镜像，
        # 子块 qu 从右往左数
        if flip:
            x_lo = cx + (Csz - (qu + 1) * P)
            x_hi = cx + (Csz - qu * P)
        else:
            x_lo = cx + qu * P
            x_hi = cx + (qu + 1) * P
        y_lo = cy + qv * P
        y_hi = cy + (qv + 1) * P
        # 缩放帧 -> 原图帧（训练增强先缩放后裁剪，这里按相反顺序逆回去）
        x_lo, x_hi = x_lo / scale, x_hi / scale
        y_lo, y_hi = y_lo / scale, y_hi / scale

        # 随机裁剪的子块边界通常跨 2~4 个 bank 网格格子：
        # 取所有被覆盖的 patch，按重叠面积加权平均它们的累积 ASM
        acc, w_total = 0.0, 0.0
        for gy in range(int(np.floor(y_lo / P)), int(np.ceil(y_hi / P))):
            for gx in range(int(np.floor(x_lo / P)), int(np.ceil(x_hi / P))):
                i = self._lookup.get((b_i, gx, gy))
                if i is None or self.asm_cnt[i] <= 0:
                    continue               # 无 patch 记录，或还没评估过
                ox = min(x_hi, (gx + 1) * P) - max(x_lo, gx * P)   # 重叠宽
                oy = min(y_hi, (gy + 1) * P) - max(y_lo, gy * P)   # 重叠高
                if ox <= 0 or oy <= 0:
                    continue
                acc += (self.asm_sum[i] / self.asm_cnt[i]) * ox * oy
                w_total += ox * oy
        if w_total <= 0:
            return None
        return float(acc / w_total)

    # -------------------------------------------------------- (de)serialize
    def state_dict(self):
        """ASM 累积状态随 checkpoint 保存；patch 记录本身由数据集确定性重建"""
        return {'asm_sum': torch.from_numpy(self.asm_sum).float(),
                'asm_cnt': torch.from_numpy(self.asm_cnt).long()}

    def load_state_dict(self, state):
        n = len(self)
        if state['asm_sum'].numel() != n:
            # patch 数对不上 = 数据集 / 协议 / patch_size 变了，累积值不可复用
            raise ValueError('bank size mismatch: ckpt {} vs rebuilt {} -- '
                             'use the same data/protocol'.format(
                                 state['asm_sum'].numel(), n))
        self.asm_sum = state['asm_sum'].numpy().astype(np.float64)
        self.asm_cnt = state['asm_cnt'].numpy().astype(np.int64)
