# -*- coding: utf-8 -*-
"""PET（Point-Query Quadtree, ICCV 2023）训练与推理服务。

- 模型/损失：nets/pet（vendor 自 PET 官方实现）
- 点标注输入：框架统一布局 Train/Images + GT/*.txt|.mat|.npy（(N,2) 坐标）
  PET 内部使用 (y, x) 坐标，读取后自动交换
- 训练：AdamW 分组学习率 + StepLR（官方设定）；模型内部计算
  “点查询损失 + 四叉树划分损失”；输出 MAE / RMSE
- 推理：模型 test_forward 已按 score>0.5 过滤点，返回最终点集
"""
import copy
import csv
import logging
import math
import time
from pathlib import Path

from app.services.loss_weight_utils import (all_weights, apply_criterion_weights,
                                       apply_matcher_costs, describe)
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from nets.pet import build_pet
from nets.pet.util.misc import collate_fn as pet_collate

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PET_PATCH = 256          # 官方训练/推理适配的 patch 尺寸（四叉树窗口按此设计）


# ------------------------- 数据集 -------------------------

def _crop_patch(img_t, points, patch):
    """随机裁剪 patch×patch 并插值到 patch；点坐标 (y,x) 同步。"""
    H, W = img_t.shape[-2:]
    y0 = np.random.randint(0, H - patch + 1) if H > patch else 0
    x0 = np.random.randint(0, W - patch + 1) if W > patch else 0
    y1, x1 = min(H, y0 + patch), min(W, x0 + patch)
    crop = img_t[:, y0:y1, x0:x1]
    if points.shape[0]:
        inside = ((points[:, 0] >= y0) & (points[:, 0] < y1) &
                  (points[:, 1] >= x0) & (points[:, 1] < x1))
        pts = points[inside].copy()
        pts[:, 0] -= y0
        pts[:, 1] -= x0
    else:
        pts = points
    ch, cw = crop.shape[-2:]
    crop = F.interpolate(crop.unsqueeze(0), size=(patch, patch),
                         mode="bilinear", align_corners=False)[0]
    if len(pts):
        pts[:, 0] *= patch / max(1, ch)
        pts[:, 1] *= patch / max(1, cw)
    return crop, pts


class PETDataset(Dataset):
    """PET 点标注数据集（框架布局；坐标转为 PET 的 (y,x)）。"""

    def __init__(self, root, train=True, crop_size=PET_PATCH, augment=True,
                 augment_strength=1.0):
        from app.services.crowd_counting import _crowd_dirs
        self.root = Path(root)
        self.train = train
        self.crop = max(64, int(crop_size or PET_PATCH))
        self.augment = bool(augment and train)
        self.aug_strength = max(0.0, min(2.0, float(augment_strength)))
        img_dir, gt_dir = _crowd_dirs(self.root, train)
        if img_dir is None:
            raise ValueError(f"未找到图像目录: {root}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        cand_dirs = [gt_dir] if gt_dir is not None else [
            self.root / d for d in ("gt", "npys", "labels")]
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
                f"未找到点标注样本（PET 需要 .txt / .mat 点标注或 (N,2) "
                f"坐标 .npy）：{root}")
        self.items = items
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.items)

    def _load_points(self, gt_path):
        from app.services.crowd_counting import _load_points_gt_any
        pts = _load_points_gt_any(gt_path)       # (x, y)
        return pts[:, ::-1].astype(np.float32)   # (y, x) for PET

    @staticmethod
    def _density(points):
        pts = torch.from_numpy(points.copy())
        if pts.shape[0] > 1:
            dist = torch.cdist(pts, pts, p=2)
            return dist.sort(dim=1)[0][:, 1].mean().reshape(-1)
        return torch.tensor(999.0).reshape(-1)

    def __getitem__(self, idx):
        img_path, gt_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        points = self._load_points(gt_path)
        W, H = img.size

        if self.train:
            # 随机尺度 [1-0.2s, 1+0.2s]（与 PET 官方一致，图像与坐标同步）
            if self.augment:
                s = self.aug_strength
                lo, hi = 1 - 0.2 * s, 1 + 0.2 * s
                scale = np.random.uniform(lo, hi) if hi > lo else 1.0
                if abs(scale - 1.0) > 0.01:
                    nw, nh = max(1, int(W * scale)), max(1, int(H * scale))
                    img = img.resize((nw, nh), Image.BILINEAR)
                    points = points * np.array([nh / H, nw / W],
                                               dtype=np.float32)
                    W, H = nw, nh
            img_t = self.transform(img)
            img_t, points = _crop_patch(img_t, points, self.crop)
            if self.augment and np.random.rand() < 0.5:
                img_t = torch.flip(img_t, dims=[2])
                if len(points):
                    points[:, 1] = self.crop - points[:, 1]
            target = {
                "points": torch.from_numpy(points).float(),
                "labels": torch.ones(len(points), dtype=torch.long),
                "density": self._density(points),
            }
        else:
            img_t = self.transform(img)
            target = {
                "points": torch.from_numpy(points).float(),
                "labels": torch.ones(len(points), dtype=torch.long),
                "image_path": str(img_path),
            }
        return img_t, target


# ------------------------- 模型 / 权重 -------------------------

def _pet_args(device, pretrained=False):
    return SimpleNamespace(
        backbone="vgg16_bn", position_embedding="sine",
        dec_layers=2, dim_feedforward=512, hidden_dim=256,
        dropout=0.0, nheads=8,
        set_cost_class=1.0, set_cost_point=0.05,
        ce_loss_coef=1.0, point_loss_coef=5.0, eos_coef=0.5,
        device=str(device), pretrained=bool(pretrained))


def _extract_state(ck):
    if not isinstance(ck, dict):
        return ck
    for key in ("model", "model_state_dict", "state_dict"):
        sd = ck.get(key)
        if isinstance(sd, dict):
            return sd
    return ck


def _load_pet_weights(model, weight_path):
    """加载 PET 权重（兼容官方 checkpoint['model'] 与框架 checkpoint）"""
    ck = torch.load(weight_path, map_location="cpu")
    sd = _extract_state(ck)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    total = len(model.state_dict())
    res = model.load_state_dict(sd, strict=False)
    matched = len(sd) - len(res.unexpected_keys)
    if matched < total * 0.95:
        raise ValueError(
            f"PET 权重与模型架构不匹配（{matched}/{total} 个参数键命中，"
            f"missing={len(res.missing_keys)} "
            f"unexpected={len(res.unexpected_keys)}）: {weight_path}")
    if res.missing_keys or res.unexpected_keys:
        # 官方 checkpoint 与框架结构可能存在少量命名差异，保留兼容加载，
        # 但必须显式告警，避免“能跑但半加载”的静默错误。
        logging.getLogger(__name__).warning(
            "PET 权重兼容加载: missing=%d unexpected=%d | missing[:10]=%s "
            "unexpected[:10]=%s",
            len(res.missing_keys), len(res.unexpected_keys),
            res.missing_keys[:10], res.unexpected_keys[:10])
    return res


# ------------------------- 训练 -------------------------

@torch.no_grad()
def _validate_pet(model, loader, device, task, criterion=None, epoch=0,
                  loss_size=None):
    """全图逐张评估：候选点集计数 vs GT 点数 (MAE, RMSE) + 训练同口径损失

    criterion 给出时在验证集上计算与训练一致的损失（no_grad），并打印逐张
    验证进度；无法计算时回退为 None（不影响 MAE/RMSE）。

    loss_size：训练输入尺寸（PET 四叉树按固定块工作，非训练尺寸会产生 NaN
    代价矩阵）——计算损失前把验证图等比缩放到该尺寸，GT 点同步缩放。
    """
    model.eval()
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    loss_sum, nl = 0.0, 0
    total = max(1, len(loader))
    log_every = max(1, total // 10)
    if task is not None:
        task.log(f"  开始验证（测试集）：{total} 张图像（含损失）")
    t0 = time.time()
    for samples, targets in loader:
        if task.stop_event.is_set():
            raise StopIteration()
        samples = samples.to(device)
        out = model(samples, test=True, targets=targets)
        pred_cnt = int(out["pred_logits"].shape[1])
        gt_cnt = int(targets[0]["points"].shape[0])
        mae_sum += abs(pred_cnt - gt_cnt)
        mse_sum += (pred_cnt - gt_cnt) ** 2
        n += 1
        if criterion is not None:
            try:
                # 与训练同口径构造 targets：验证集原始 target 只有 points/labels
                # （PET 的 compute_loss 还需要 density，缺失会 KeyError），且
                # 输入需缩放到训练尺寸（PET 四叉树固定块，其他尺寸会有 NaN）。
                # 注意 val loader 经 pet_collate 后 samples 为 NestedTensor。
                src = samples.tensors if hasattr(samples, "tensors") else samples
                H, W = src.shape[-2:]
                S = int(loss_size or max(H, W))
                x_loss = F.interpolate(src, size=(S, S), mode="bilinear",
                                       align_corners=False)
                tgt = []
                for item in targets:
                    pts = item["points"].detach().clone().float()
                    pts[:, 0] *= S / max(1, H)     # (y, x) 坐标同步缩放
                    pts[:, 1] *= S / max(1, W)
                    d = {"points": pts.to(device),
                         "labels": item["labels"].to(device)}
                    d["density"] = PETDataset._density(
                        pts.cpu().numpy()).to(device)
                    tgt.append(d)
                out_tr = model(x_loss, epoch=epoch, train=True,
                               criterion=criterion, targets=tgt)
                lv = out_tr["losses"]
                if torch.is_tensor(lv) and torch.isfinite(lv):
                    loss_sum += float(lv)
                    nl += 1
            except Exception as e:
                task.log(f"  验证损失计算跳过: {e}")
                criterion = None
        if n % log_every == 0 or n >= total:
            task.log(
                f"  验证 {n}/{total} | loss "
                f"{loss_sum / max(1, nl):.5f} | MAE "
                f"{mae_sum / max(1, n):.3f} | RMSE "
                f"{math.sqrt(mse_sum / max(1, n)):.3f} | "
                f"用时 {time.time() - t0:.1f}s")
    model.train()
    n = max(1, n)
    return (mae_sum / n, math.sqrt(mse_sum / n),
            loss_sum / nl if nl else None)


def run_pet_train(task, params: dict):
    """PET 点预测训练（官方管线：AdamW 分组学习率 + StepLR + 模型内损失）"""
    from app.services.train_common import (set_seed, EarlyStopper,
                                           save_checkpoint, save_loss_curves,
                                           report_batch_progress,
                                           is_main_process,
                                           release_training_memory)
    from app.services.crowd_counting import _ckpt_paths, clear_inference_cache

    dataset_path = params["dataset_path"]
    epochs = int(params.get("epochs", 30))
    eval_every = max(1, int(params.get("eval_every") or 1))
    batch_size = int(params.get("batch_size", 8))
    # lr=0 / 缺省 → PET 论文默认 lr=1e-4（骨干 1e-5）
    lr = float(params.get("lr") or 1e-4)
    lr_backbone = float(params.get("lr_backbone") or 1e-5)
    weight_decay = float(params.get("weight_decay", 1e-4))
    clip = float(params.get("clip_max_norm", 0.1))
    pretrained = bool(params.get("pretrained", True))
    weight_path = params.get("weight_path") or None
    resume_weight = params.get("resume_weight") or None
    device_str = params.get("device", "cpu")
    num_workers = int(params.get("num_workers", 0))
    early_stop = int(params.get("early_stopping", 0))
    val_ratio = float(params.get("val_ratio", 0.2))
    augment = bool(params.get("augment", True))
    aug_strength = float(params.get("augment_strength", 1.0))
    crop_size = int(params.get("img_size", 0) or PET_PATCH)
    if crop_size != PET_PATCH:
        task.log(f"PET 官方实现按 {PET_PATCH} patch 训练/推理，"
                 f"本次使用 {PET_PATCH}")
        crop_size = PET_PATCH
    if batch_size < 2:
        raise ValueError(
            "PET 训练要求 Batch Size ≥ 2（稀疏/稠密双监督按 batch 内密度排序）")

    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    task.total_epochs = epochs
    task.log(f"PET 点预测训练 | ICCV 2023 Point-Query Quadtree | "
             f"epochs={epochs} batch={batch_size} patch={crop_size}")
    task.log(f"数据集: {dataset_path} | 设备: {device_str} | lr={lr} "
             f"(backbone {lr_backbone})")
    set_seed(int(params.get("seed", 42)))
    _n_cached = clear_inference_cache()
    if _n_cached:
        task.log(f"已清空推理模型缓存（{_n_cached} 个），为训练让出内存")

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    full = PETDataset(dataset_path, train=True, crop_size=crop_size,
                      augment=augment, augment_strength=aug_strength)
    # 验证集：优先数据集自带 Test 划分；否则按 val_ratio 从训练集切分
    from app.services.crowd_counting import _eval_from_test
    val_eval_ds, val_src = _eval_from_test(
        dataset_path,
        lambda: PETDataset(dataset_path, train=False,
                           crop_size=crop_size, augment=False))
    if val_eval_ds is not None:
        train_set = full
        val_set = val_eval_ds
        n_val = len(val_set)
    else:
        n_val = max(1, int(len(full) * val_ratio)) if len(full) > 1 else 0
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(full), generator=g).tolist()
        val_full = copy.copy(full)
        val_full.train = False
        val_full.augment = False
        train_set = torch.utils.data.Subset(full, idx[n_val:])
        val_set = torch.utils.data.Subset(val_full, idx[:n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              collate_fn=pet_collate)
    if len(train_loader) == 0:
        raise ValueError(
            f"训练批数为 0（训练样本 {len(train_set)} < Batch Size "
            f"{batch_size}，PET 启用 drop_last），请减小 Batch Size 或补充数据")
    val_loader = (DataLoader(val_set, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=pet_collate)
                  if n_val else None)
    task.log(f"训练样本: {len(train_set)} | 验证样本: {len(val_set)}（{val_src}）")

    # 有初始/续训权重时不再预载 ImageNet VGG（checkpoint 已包含骨干）
    use_imagenet = pretrained and not (weight_path or resume_weight)
    model, criterion = build_pet(_pet_args(device, pretrained=use_imagenet))
    if weight_path:
        _load_pet_weights(model, weight_path)
        task.log(f"PET 初始权重: {weight_path}")
    model.to(device)
    criterion.to(device)
    # 损失分项权重（缺省 = PET 源码设置：ce 1 / points 5 / 代价 1、0.05）
    _lw = all_weights(params, {"ce_w": 1.0, "points_w": 5.0,
                               "cost_class": 1.0, "cost_point": 0.05})
    apply_criterion_weights(criterion, _lw)
    apply_matcher_costs(criterion, _lw)
    task.log(f"损失权重: {describe(_lw)}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": lr_backbone},
    ]
    # 优化器 / 调度器可选：默认对齐 PET 论文（AdamW + 恒定学习率）
    from app.core.optim_builders import build_torch_optimizer
    from app.services.crowd_counting import build_crowd_scheduler
    opt_params = dict(params)
    if not opt_params.get("optimizer"):
        opt_params["optimizer"] = "adamw"
    optimizer = build_torch_optimizer(opt_params, param_dicts, lr,
                                      weight_decay=weight_decay)
    scheduler, _is_plateau = build_crowd_scheduler(
        optimizer, params.get("scheduler") or "cosine", epochs, params)
    task.log(f"优化器: {opt_params['optimizer']}（骨干 lr_backbone / 其余 lr）"
             f" | 调度器: {params.get('scheduler') or 'cosine'}")

    csv_path = save_dir / "metrics.csv"
    header = ["epoch", "train_loss", "val_loss", "val_mae", "val_rmse"]
    best_mae, start_epoch = float("inf"), 1
    best_path, last_path = _ckpt_paths(save_dir, "pet")
    if resume_weight:
        ck = torch.load(resume_weight, map_location="cpu")
        sd = _extract_state(ck)
        model.load_state_dict(sd, strict=True)
        if isinstance(ck, dict):
            opt = ck.get("optimizer_state_dict") or ck.get("optimizer")
            if opt:
                optimizer.load_state_dict(opt)
            sch = ck.get("scheduler_state_dict") or ck.get("lr_scheduler")
            if sch:
                scheduler.load_state_dict(sch)
            start_epoch = int(ck.get("epoch", 0)) + 1
            bm = ck.get("best_mae")
            if bm is None:
                bm = (ck.get("best_metric") or {}).get("val_mae")
            if bm is not None:
                best_mae = float(bm)
        task.best = ({"epoch": start_epoch - 1,
                      "val_mae": round(best_mae, 3)}
                     if best_mae < float("inf") else {})
        task.log(f"断点续训：从 epoch {start_epoch} 继续"
                 f"（best_mae={best_mae}）")
        from app.services.train_common import reload_task_history
        reload_task_history(task, csv_path, start_epoch, header)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    stopper = EarlyStopper(early_stop, mode="min")
    for epoch in range(start_epoch, epochs + 1):
        if task.stop_event.is_set():
            raise StopIteration()
        do_eval = ((eval_every <= 1) or (epoch % eval_every == 0)
                   or (epoch == epochs))
        model.train()
        criterion.train()
        total_loss, n_batches = 0.0, 0
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)
        _main = is_main_process(params)
        for batch_idx, (samples, targets) in enumerate(train_loader):
            if task.stop_event.is_set():
                raise StopIteration()
            samples = samples.to(device)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in t.items()} for t in targets]
            out = model(samples, epoch=epoch, train=True,
                        criterion=criterion, targets=targets)
            losses = out["losses"]
            if not torch.isfinite(losses):
                task.log(f"损失非有限值，跳过该 batch: {out.get('loss_dict')}")
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            losses.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += float(losses.detach())
            n_batches += 1
            report_batch_progress(
                task, epoch=epoch, epochs=epochs, batch_idx=batch_idx,
                total_batches=total_batches, running_loss=total_loss,
                n_batches=n_batches, log_every=log_every, is_main=_main)
        train_loss = total_loss / max(1, n_batches)
        val_mae = val_rmse = val_loss = None
        if val_loader is not None and do_eval:
            val_mae, val_rmse, val_loss = _validate_pet(
                model, val_loader, device, task, criterion=criterion,
                epoch=epoch, loss_size=crop_size)
        if _is_plateau:
            if do_eval and val_loader is not None:
                scheduler.step(val_mae)
        else:
            scheduler.step()

        task.epoch = epoch
        task.progress = epoch / epochs * 100.0
        task.metrics = {"train_loss": round(train_loss, 5)}
        if do_eval and val_mae is not None:
            task.metrics["val_mae"] = round(val_mae, 4)
            task.metrics["val_rmse"] = round(val_rmse, 4)
        if val_loss is not None:
            task.metrics["val_loss"] = round(val_loss, 5)
        task.history.append({"epoch": epoch, **task.metrics})
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.5f}",
                 "" if val_loss is None else f"{val_loss:.5f}",
                 "" if val_mae is None else f"{val_mae:.4f}",
                 "" if val_rmse is None else f"{val_rmse:.4f}"])
        task.log(f"Epoch {epoch}/{epochs} | loss {train_loss:.5f}"
                 + (f" | val_loss {val_loss:.5f}"
                    if val_loss is not None else "")
                 + (f" | val_MAE {val_mae:.3f} | val_RMSE {val_rmse:.3f}"
                    if val_mae is not None else "")
                 + ("" if do_eval else "（本轮跳过测试集评估）"))
        save_loss_curves(save_dir, task.history,
                         ("val_mae", "val_rmse", "val_loss"))

        is_best = val_mae is not None and val_mae < best_mae
        if is_best:
            best_mae = val_mae
            task.best = {"epoch": epoch, "val_mae": round(val_mae, 3)}
        save_checkpoint(best_path if is_best else last_path,
                        net=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch, best_metric=task.best, model="pet",
                        val_mae=val_mae)
        if is_best:
            save_checkpoint(last_path, net=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch,
                            best_metric=task.best, model="pet",
                            val_mae=val_mae)
        if val_mae is not None and stopper.step(val_mae):
            task.log(f"早停触发：MAE 已 {stopper.patience} 轮未提升")
            break
        release_training_memory()

    task.result = {"best_weight": str(best_path), "last_weight": str(last_path),
                   "best_val_mae": round(best_mae, 3),
                   "save_dir": str(save_dir)}
    task.log(f"训练结束，最优 val_MAE={best_mae:.3f}")


# ------------------------- 推理 -------------------------

def pet_infer(image: Image.Image, weight_path=None, device="cpu",
              pretrained=True, conf_threshold=0.5,
              return_all_points=False):
    """PET 推理：返回人数 + 预测点可视化（红色圆点）"""
    import base64
    import io

    import cv2

    from app.services.crowd_counting import (_model_cache_key,
                                             _model_cache_get,
                                             _model_cache_put)

    wp = weight_path
    if not wp:
        from config import WEIGHTS_DIR
        cand = Path(WEIGHTS_DIR) / "pet" / "SHA_model.pth"
        if cand.is_file():
            wp = str(cand)
    if not wp:
        raise ValueError(
            "PET 推理需要权重：请选择训练产物 / 上传 .pth，或将 "
            "SHA_model.pth 放到 weights/pretrained/pet/")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    cache_key = _model_cache_key("pet", wp, device)
    model = _model_cache_get(cache_key)
    if model is None:
        model, _ = build_pet(_pet_args(dev, pretrained=False))
        _load_pet_weights(model, wp)
        model.to(dev).eval()
        _model_cache_put(cache_key, model)

    img = image.convert("RGB")
    W0, H0 = img.size
    max_side = 1536
    scale = min(1.0, max_side / max(H0, W0))
    if scale < 1.0:
        img = img.resize((int(W0 * scale), int(H0 * scale)), Image.BILINEAR)
    Wi, Hi = img.size
    x = transforms.functional.to_tensor(img)
    x = transforms.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    samples = _to_nested([x]).to(dev)
    ph, pw = samples.tensors.shape[-2:]
    thr = 0.5 if conf_threshold is None else float(conf_threshold)
    with torch.no_grad():
        out = model(samples, test=True, threshold=thr)
    points = out["pred_points"][0].detach().cpu().numpy()   # normalized (y,x)
    pts = np.stack([points[:, 0] * ph, points[:, 1] * pw], axis=1)
    # padded/resized -> 原图
    pts[:, 0] *= H0 / max(1, Hi)
    pts[:, 1] *= W0 / max(1, Wi)
    pts_xy = pts[:, ::-1]

    canvas = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    size = max(2, int(max(H0, W0) / 300))
    for p in pts_xy:
        canvas = cv2.circle(canvas, (int(p[0]), int(p[1])), size,
                            (0, 0, 255), -1)
    vis = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(vis).save(buf, format="PNG")
    vis_b64 = base64.b64encode(buf.getvalue()).decode()

    return {"count": int(len(pts_xy)), "visualization": vis_b64,
            "num_points": int(len(pts_xy)),
            "points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in pts_xy[:2000]],
            **({"all_points": [[round(float(p[0]), 2), round(float(p[1]), 2)]
                               for p in pts_xy]} if return_all_points else {}),
            "conf_threshold": thr, "model": "pet", "point_based": True}


def _to_nested(tensor_list):
    from nets.pet.util.misc import nested_tensor_from_tensor_list
    return nested_tensor_from_tensor_list(tensor_list)

