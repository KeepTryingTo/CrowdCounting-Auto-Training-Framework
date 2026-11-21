# -*- coding: utf-8 -*-
"""Training / validation loops."""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .datasets.crowd import collate_eval, collate_train
from .lcg_ppd import lcg_ppd
from .metrics import count_metrics, game_metric, localization_metrics
from .utils import atomic_torch_save


def train_one_epoch(pmr, cmr, loader, opt_p, opt_c, device, args,
                    synth=None, epoch: int = 0, aux_loss=None,
                    aux_w: float = 1.0) -> Dict[str, float]:
    pmr.train()
    cmr.train()
    p_loss_sum = c_loss_sum = s_loss_sum = 0.0
    n = 0
    t0 = time.time()
    for it, batch in enumerate(loader):
        imgs = batch["image"].to(device)
        prob_t = batch["prob"].to(device)
        count_t = batch["count"].to(device)

        # ---- PMR：预测概率图 ----
        prob_p = pmr(imgs)
        loss_p = F.mse_loss(prob_p, prob_t)

        # ---- CMR：预测概率图 + 真值概率图一起输入（官方训练方式） ----
        if args.use_gt_prob:
            cmr_in = torch.cat([prob_p, prob_t], dim=0)
            target_c = torch.cat([count_t, count_t], dim=0)
        else:
            cmr_in = prob_p
            target_c = count_t
        count_p = cmr(cmr_in)
        loss_c = F.mse_loss(count_p, target_c)

        loss = loss_p + loss_c
        if aux_loss is not None:
            loss = loss + float(aux_w) * aux_loss(
                {"prob": prob_p, "density": count_p},
                {"prob": prob_t, "density": count_t})
        opt_p.zero_grad(set_to_none=True)
        opt_c.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pmr.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(cmr.parameters(), 10.0)
        opt_p.step()
        opt_c.step()

        p_loss_sum += float(loss_p.detach())
        c_loss_sum += float(loss_c.detach())
        n += 1

        # ---- 合成概率图只更新 CMR ----
        if synth is not None:
            prob_s, count_s = synth.sample_batch(args.synth_batch_size)
            prob_s = prob_s.to(device)
            count_s = count_s.to(device)
            pred_s = cmr(prob_s)
            loss_s = F.mse_loss(pred_s, count_s)
            opt_c.zero_grad(set_to_none=True)
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(cmr.parameters(), 10.0)
            opt_c.step()
            s_loss_sum += float(loss_s.detach())

        if args.log_every and (it + 1) % args.log_every == 0:
            print(f"  epoch {epoch} | {it+1}/{len(loader)} | "
                  f"prob {p_loss_sum/max(1,n):.4f} | "
                  f"count {c_loss_sum/max(1,n):.4f} | "
                  f"synth {s_loss_sum/max(1,n):.4f} | "
                  f"{time.time()-t0:.1f}s")
    return {"prob_loss": p_loss_sum / max(1, n),
            "count_loss": c_loss_sum / max(1, n),
            "synth_loss": s_loss_sum / max(1, n)}


@torch.no_grad()
def validate_count(pmr, cmr, loader, device) -> Dict[str, float]:
    pmr.eval()
    cmr.eval()
    preds, gts = [], []
    for batch in loader:
        sample = batch[0]
        img = sample["image"].unsqueeze(0).to(device)
        prob = pmr(img)
        count = cmr(prob)
        preds.append(float(count.sum().item()))
        gts.append(float(sample["dot"].sum().item()))
    return count_metrics(preds, gts)


@torch.no_grad()
def evaluate_full(pmr, cmr, loader, device, distance: float = 20.0):
    pmr.eval()
    cmr.eval()
    reg_pred, loc_pred, gts = [], [], []
    game = {0: [], 1: [], 2: [], 3: []}
    pred_dots, gt_dots = [], []
    per_image = []
    for batch in loader:
        sample = batch[0]
        img = sample["image"].unsqueeze(0).to(device)
        prob = pmr(img)
        count_map = cmr(prob)
        reg = float(count_map.sum().item())
        dot, points = lcg_ppd(prob, count_map, rf=cmr.rf)
        gt_dot = sample["dot"].numpy()[0]
        # 评估时已按 32 倍数补边，形状一致。
        if dot.shape != gt_dot.shape:
            h = min(dot.shape[0], gt_dot.shape[0])
            w = min(dot.shape[1], gt_dot.shape[1])
            dot = dot[:h, :w]
            gt_dot = gt_dot[:h, :w]
        loc = float(dot.sum())
        gt = float(gt_dot.sum())
        reg_pred.append(reg)
        loc_pred.append(loc)
        gts.append(gt)
        for level in game:
            game[level].append(game_metric(gt_dot, dot, level))
        pred_dots.append(dot)
        gt_dots.append(gt_dot)
        per_image.append({"image": sample["image_path"],
                          "gt": gt, "regression": reg, "localization": loc})
    metrics = {
        "regression": count_metrics(reg_pred, gts),
        "localization_count": count_metrics(loc_pred, gts),
        "game": {f"GAME{k}": float(np.mean(v)) for k, v in game.items()},
        "localization": localization_metrics(pred_dots, gt_dots, distance),
        "per_image": per_image,
    }
    return metrics


def save_checkpoint(path, pmr, cmr, args, epoch: int,
                    best: Optional[float] = None) -> None:
    atomic_torch_save({
        "pmr": pmr.state_dict(),
        "cmr": cmr.state_dict(),
        "epoch": int(epoch),
        "best": best,
        "args": vars(args),
    }, path)


def load_checkpoint(path, pmr, cmr, strict: bool = True):
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(str(path), map_location="cpu")
    if isinstance(ck, dict) and "pmr" in ck:
        pmr.load_state_dict(ck["pmr"], strict=strict)
        cmr.load_state_dict(ck["cmr"], strict=strict)
        return ck
    # 兼容直接保存 state_dict 的情况
    if isinstance(ck, dict) and "state_dict1" in ck:
        pmr.load_state_dict(ck["state_dict1"], strict=strict)
        cmr.load_state_dict(ck["state_dict2"], strict=strict)
        return ck
    raise ValueError(f"无法识别的 checkpoint: {path}")


def make_loader(dataset, args, train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, args.batch_size) if train else 1,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_train if train else collate_eval,
        drop_last=False)
