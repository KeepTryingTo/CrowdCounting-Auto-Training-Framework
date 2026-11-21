# -*- coding: utf-8 -*-
"""Training / validation loops."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .datasets.crowd import collate_eval, collate_train
from .losses import stage_loss
from .metrics import count_metrics
from .postprocess import extract_peaks, fuse_density_and_points
from .utils import atomic_torch_save


def make_loader(dataset, args, train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size) if train else 1,
        shuffle=bool(train),
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_train if train else collate_eval,
        drop_last=False,
    )


def make_optimizer(model: torch.nn.Module, args):
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr,
                                weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr,
                                 weight_decay=args.weight_decay)
    return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                           weight_decay=args.weight_decay)


def train_one_epoch(model, loader, optimizer, device, stage: str,
                    args, epoch: int, aux_loss=None, aux_w: float = 1.0) -> float:
    model.train()
    total = 0.0
    n = 0
    t0 = time.time()
    for it, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        batch_dev = {
            "density": batch["density"].to(device, non_blocking=True),
            "loc": batch["loc"].to(device, non_blocking=True),
            "attn": batch["attn"].to(device, non_blocking=True),
        }
        outputs = model(images)
        # 只计算当前阶段需要的损失；例如 density 阶段不会优化 loc/attn。
        loss = stage_loss(stage, outputs, batch_dev,
                          gamma=args.gamma,
                          attn_pos_weight=args.attn_pos_weight)
        if aux_loss is not None:
            loss = loss + float(aux_w) * aux_loss(outputs, batch_dev)
        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at epoch {epoch} iter {it}, skip")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 10.0)
        optimizer.step()
        total += float(loss.detach())
        n += 1
        if args.log_every and (it + 1) % int(args.log_every) == 0:
            print(f"  epoch {epoch} | {it + 1}/{len(loader)} | "
                  f"loss {total / max(1, n):.4f} | "
                  f"{time.time() - t0:.1f}s")
    return total / max(1, n)


@torch.no_grad()
def evaluate_counts(model, loader, device, peak_threshold: float = 0.3,
                    fusion_threshold: float = 5.0,
                    has_density: bool = True,
                    collect_records: bool = False):
    # 验证集同时统计三种计数：密度图积分、定位峰值数、4x4 融合。
    # 训练阶段据此选择最佳 checkpoint。
    model.eval()
    density_pred, loc_pred, fused_pred, gt = [], [], [], []
    records = []
    for batch in loader:
        sample = batch[0]
        image = sample["image"].unsqueeze(0).to(device)
        out = model(image)
        H, W = sample["size"]
        # 提取坐标信息
        pts, scores = extract_peaks(out["loc"], threshold=peak_threshold)
        if has_density and "density" in out:
            dens = out["density"]
            dcount = float(dens.sum().item())
            fcount, _, _ = fuse_density_and_points(
                density=dens,
                points=pts,
                scores=scores,
                height=H,
                width=W,
                grid=4,
                theta_f=fusion_threshold
            )
        else:
            dcount = float(len(pts))
            fcount = dcount
        lcount = int(len(pts))
        gcount = int(len(sample["points"]))
        density_pred.append(dcount)
        loc_pred.append(lcount)
        fused_pred.append(fcount)
        gt.append(gcount)
        if collect_records:
            p = pts
            s = scores
            records.append({
                "image_path": sample["image_path"],
                "gt": sample["points"].numpy(),
                "points": p,
                "scores": s,
                "size": (H, W),
                "density": dcount,
                "loc": lcount,
                "fused": fcount,
                "gt_count": gcount,
            })
    metrics = {
        "density": count_metrics(density_pred, gt),
        "loc": count_metrics(loc_pred, gt),
        "fused": count_metrics(fused_pred, gt),
        "n": len(gt),
    }
    if collect_records:
        return metrics, records
    return metrics


def save_checkpoint(path, model, args, stage: str, epoch: int,
                    best: Optional[float] = None) -> None:
    atomic_torch_save({
        "model": model.state_dict(),
        "stage": stage,
        "epoch": int(epoch),
        "best": best,
        "args": vars(args),
    }, path)


def load_model_weights(model, path) -> None:
    # 兼容框架 checkpoint（"model" 键）和裸 state_dict；strict=False 便于
    # RAZ-Net 从 Main-Net 只加载重叠参数。
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(str(path), map_location="cpu")
    sd = ck.get("model") if isinstance(ck, dict) else ck
    if not isinstance(sd, dict):
        raise ValueError(f"无效权重文件: {path}")
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {path} | missing={len(missing)} unexpected={len(unexpected)}")
