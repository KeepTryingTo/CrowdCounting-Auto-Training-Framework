# -*- coding: utf-8 -*-
"""D2C (Decoupled Two-Stage Crowd Counting) adapter for Codes.

D2C is vendored under ``nets/d2c``.  PMR predicts a probability map and CMR
turns it into a local count map; LCG-PPD extracts localization points.  This
service maps the implementation onto the framework's task/checkpoint API.
"""
from __future__ import annotations

import base64
import csv
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nets.d2c.datasets import BalancedSyntheticSampler, build_datasets
from nets.d2c.engine import (load_checkpoint, make_loader, save_checkpoint,
                             train_one_epoch, validate_count)
from nets.d2c.lcg_ppd import lcg_ppd
from nets.d2c.models import D2CNet
from nets.d2c.utils import get_device, set_seed

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _make_args(params: dict, save_dir: Path) -> SimpleNamespace:
    sigma_mode = str(params.get("sigma_mode") or "adaptive").lower()
    if sigma_mode not in ("adaptive", "fixed"):
        sigma_mode = "adaptive"
    return SimpleNamespace(
        data_root=str(params["dataset_path"]),
        dataset=str(params.get("d2c_dataset") or "auto"),
        part=params.get("part") or None,
        val_ratio=float(params.get("val_ratio", 0.2) or 0.2),
        seed=int(params.get("seed", 42) or 42),
        manifest=str(params.get("manifest") or ""),
        train_sub=str(params.get("train_sub") or ""),
        test_sub=str(params.get("test_sub") or ""),
        allow_density_gt=bool(params.get("allow_density_gt", True)),
        density_threshold=float(params.get("density_threshold", 0.05) or 0.05),
        patch_size=max(32, int(params.get("img_size") or 384)),
        max_side=max(0, int(params.get("max_side") or 1920)),
        sigma_mode=sigma_mode,
        sigma=float(params.get("sigma") or 15.0),
        no_augment=not bool(params.get("augment", True)),
        max_train=int(params.get("max_train") or 0),
        max_val=int(params.get("max_val") or 0),
        pretrained=bool(params.get("pretrained", True)),
        vgg_weights=str(params.get("vgg_weights") or ""),
        device=str(params.get("device") or "cpu"),
        num_workers=max(0, int(params.get("num_workers") or 0)),
        epochs=max(1, int(params.get("epochs") or 30)),
        batch_size=max(1, int(params.get("batch_size") or 1)),
        lr=float(params.get("lr") or 1e-5),
        weight_decay=float(params.get("weight_decay") or 5e-3),
        optimizer=str(params.get("optimizer") or "adam").lower(),
        patience=10,
        output_dir=str(save_dir),
        resume="",
        use_gt_prob=bool(params.get("use_gt_prob", True)),
        use_synth=bool(params.get("use_synth", True)),
        synth_stat_images=max(1, int(params.get("synth_stat_images") or 100)),
        synth_batch_size=max(1, int(params.get("synth_batch_size") or 1)),
        synth_max_count=max(1, int(params.get("synth_max_count") or 40)),
        log_every=0,
    )


def run_d2c_train(task, params: dict):
    """Training entry for D2CNet on the crowd-point route."""
    from app.services.crowd_counting import clear_inference_cache
    from app.services.train_common import (release_training_memory,
                                           save_loss_curves)

    device = get_device(str(params.get("device") or "cpu"))
    params = dict(params)
    params["device"] = str(device)
    set_seed(int(params.get("seed", 42) or 42))
    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args = _make_args(params, save_dir)
    task.total_epochs = args.epochs
    task.log(f"D2C 训练 | PMR(prob) + CMR(count) | patch={args.patch_size} | "
             f"sigma={args.sigma_mode} | batch={args.batch_size} | "
             f"device={args.device}")
    clear_inference_cache()
    spec, train_ds, val_ds, _test_ds = build_datasets(args)
    task.log(f"数据划分: {spec.summary()} | {spec.name}")
    loss_name = str(params.get("loss") or "d2c")
    aux_loss, aux_w = None, 1.0
    if loss_name.startswith("custom:"):
        from app.core.registry import get_custom_loss, is_custom_loss_for
        if not is_custom_loss_for(loss_name, "crowd_point"):
            raise ValueError(f"自定义损失 {loss_name} 未注册到 crowd_point 路线")
        aux_loss = get_custom_loss(loss_name, task="crowd_point")
        if hasattr(aux_loss, "to"):
            aux_loss = aux_loss.to(device)
        lw = params.get("loss_weights") or {}
        aux_w = float(lw.get("aux_w", params.get("loss_weight", 1.0)) or 1.0)
        task.log(f"自定义附加损失: {loss_name}（权重 {aux_w}）")
    else:
        task.log(f"损失: D2C 内置 PMR/CMR loss（{loss_name or 'd2c'}）")
    train_loader = make_loader(train_ds, args, train=True)
    val_loader = make_loader(val_ds, args, train=False)

    model = D2CNet(pretrained=args.pretrained,
                   weights_path=args.vgg_weights or "")
    if params.get("weight_path"):
        load_checkpoint(params["weight_path"], model.pmr, model.cmr,
                        strict=False)
        task.log(f"加载初始权重: {params['weight_path']}")
    model.to(device)
    pmr, cmr = model.pmr, model.cmr
    task.log(f"PMR {sum(p.numel() for p in pmr.parameters()) / 1e6:.2f}M | "
             f"CMR {sum(p.numel() for p in cmr.parameters()) / 1e6:.2f}M")

    opt_p = torch.optim.Adam(pmr.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    opt_c = torch.optim.Adam(cmr.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    sch_p = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_p, mode="min", factor=0.5, patience=args.patience)
    sch_c = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_c, mode="min", factor=0.5, patience=args.patience)

    synth = None
    if args.use_synth:
        synth = BalancedSyntheticSampler(
            spec.train, patch_size=args.patch_size, rf=32,
            sigma_mode=args.sigma_mode, sigma=args.sigma,
            stat_images=args.synth_stat_images, max_count=args.synth_max_count,
            allow_density_gt=args.allow_density_gt,
            density_threshold=args.density_threshold, seed=args.seed)
        task.log("平衡合成概率/计数图增强: 开启（仅更新 CMR）")

    csv_path = save_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "prob_loss",
                                "count_loss", "synth_loss", "val_mae",
                                "val_rmse"])
    best = float("inf")
    best_path = save_dir / "d2c_best.pth"
    last_path = save_dir / "d2c_last.pth"
    task.best = {}
    for epoch in range(1, args.epochs + 1):
        if task.stop_event.is_set():
            raise StopIteration()
        losses = train_one_epoch(pmr, cmr, train_loader, opt_p, opt_c,
                                 device, args, synth=synth, epoch=epoch,
                                 aux_loss=aux_loss, aux_w=aux_w)
        metrics = validate_count(pmr, cmr, val_loader, device)
        score = float(metrics["mae"])
        sch_p.step(score)
        sch_c.step(score)
        train_loss = losses["prob_loss"] + losses["count_loss"]
        task.epoch = epoch
        task.progress = epoch / max(1, args.epochs) * 100.0
        task.metrics = {
            "train_loss": round(float(train_loss), 5),
            "prob_loss": round(float(losses["prob_loss"]), 5),
            "count_loss": round(float(losses["count_loss"]), 5),
            "synth_loss": round(float(losses["synth_loss"]), 5),
            "val_mae": round(score, 4),
            "val_rmse": round(float(metrics["mse"]), 4),
        }
        task.history.append({"epoch": epoch, **task.metrics})
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.5f}", f"{losses['prob_loss']:.5f}",
                f"{losses['count_loss']:.5f}",
                f"{losses['synth_loss']:.5f}", f"{score:.4f}",
                f"{metrics['mse']:.4f}"])
        task.log(f"epoch {epoch}/{args.epochs} | prob "
                 f"{losses['prob_loss']:.4f} | count {losses['count_loss']:.4f} "
                 f"| synth {losses['synth_loss']:.4f} | val MAE {score:.3f} | "
                 f"RMSE {metrics['mse']:.3f}")
        save_checkpoint(last_path, pmr, cmr, args, epoch, best)
        if score < best:
            best = score
            task.best = {"epoch": epoch, "val_mae": round(score, 4)}
            save_checkpoint(best_path, pmr, cmr, args, epoch, best)
        # 每轮收尾及时回收：gc + CUDA 缓存显存归还（PMR/CMR 双骨干常驻）
        release_training_memory(task.log, reason=f"epoch {epoch} 收尾")
    if not best_path.exists():
        save_checkpoint(best_path, pmr, cmr, args, args.epochs, best)
    try:
        save_loss_curves(save_dir, task.history,
                         ("val_mae", "val_rmse"))
    except Exception:
        pass
    task.progress = 100.0
    task.result = {"best_weight": str(best_path), "last_weight": str(last_path),
                   "best_val_mae": round(best, 4), "save_dir": str(save_dir),
                   "model": "d2c"}
    task.log(f"D2C 训练完成 | best MAE={best:.3f} | 权重: {best_path}")


def _preprocess(img: Image.Image, max_side: int = 1920):
    W, H = img.size
    scale = 1.0
    if max_side and max(W, H) > max_side:
        scale = max_side / max(W, H)
        img = img.resize((max(1, int(W * scale)), max(1, int(H * scale))),
                         Image.BILINEAR)
    W, H = img.size
    nw = int(np.ceil(W / 32) * 32)
    nh = int(np.ceil(H / 32) * 32)
    if (nw, nh) != (W, H):
        canvas = Image.new("RGB", (nw, nh), (0, 0, 0))
        canvas.paste(img, (0, 0))
        img = canvas
    x = np.asarray(img, np.float32) / 255.0
    x = (x - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return img, torch.from_numpy(x.transpose(2, 0, 1)).float()[None]


def _to_b64(img: np.ndarray, rgb: bool = True) -> str:
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("D2C 可视化编码失败")
    return base64.b64encode(buf.tobytes()).decode()


def d2c_infer(image: Image.Image, weight_path=None, device="cpu",
              pretrained=False, conf_threshold=0.5, max_side=1920,
              return_all_points=False):
    """Single-image D2C inference (PMR + CMR + LCG-PPD)."""
    if not weight_path:
        raise ValueError("D2C 推理需要训练产物权重（d2c_best.pth / "
                         "d2c_last.pth）")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    from app.services.crowd_counting import (_model_cache_get,
                                             _model_cache_put,
                                             _model_cache_key)
    key = _model_cache_key("d2c", weight_path, device)
    model = _model_cache_get(key)
    if model is None:
        model = D2CNet(pretrained=False)
        load_checkpoint(str(weight_path), model.pmr, model.cmr, strict=False)
        model.to(dev).eval()
        _model_cache_put(key, model)
    img = image.convert("RGB")
    proc, tensor = _preprocess(img, max_side=int(max_side or 1920))
    with torch.no_grad():
        prob = model.pmr(tensor.to(dev))
        count_map = model.cmr(prob)
    dot, points = lcg_ppd(prob, count_map, rf=model.cmr.rf)
    H, W = prob.shape[-2:]
    count_reg = float(count_map.sum().item())
    count_loc = int(dot.sum())
    p = F.interpolate(prob, size=(H, W), mode="bilinear",
                      align_corners=False)[0, 0].detach().cpu().numpy()
    p = p - p.min()
    if p.max() > 1e-8:
        p = p / p.max()
    base = cv2.cvtColor(np.asarray(proc), cv2.COLOR_RGB2BGR)
    heat = cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    radius = max(2, int(max(H, W) / 400))
    for x, y in points:
        cv2.circle(vis, (int(round(x)), int(round(y))), radius,
                   (0, 0, 255), -1)
    return {
        "count": round(count_reg, 2),
        "regression_count": round(count_reg, 2),
        "localization_count": count_loc,
        "num_points": count_loc,
        "points": [[round(float(x), 1), round(float(y), 1)]
                   for x, y in points[:2000]],
        **({"all_points": [[round(float(x), 2), round(float(y), 2)]
                           for x, y in points]} if return_all_points else {}),
        "visualization": _to_b64(vis, rgb=False),
        "density_map": _to_b64(heat, rgb=False),
        "conf_threshold": float(conf_threshold),
        "model": "d2c",
        "point_based": True,
    }
