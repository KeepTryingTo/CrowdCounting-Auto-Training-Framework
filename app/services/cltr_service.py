# -*- coding: utf-8 -*-
"""CLTR（Crowd Localization TRansformer, ECCV 2022）训练与推理服务。

- 模型：nets/cltr（Conditional DETR + ResNet50 + Transformer，点定位）
- 点标注：框架统一布局 Train/Images + GT/*.txt|.mat|.npy（(N,2) 坐标）
- 训练：256×256 随机裁剪 + 点查询（num_queries=500）+ 匈牙利匹配；
  损失 = 分类 focal + 点 L1（模型内 criterion）
- 验证/推理：整图按 256×256 分块（官方做法），按 score 阈值统计/定位点
- 注意：CLTR 无内置训练权重，推理需选择训练产物 / 上传 .pth
"""
import copy
import csv
import logging
import math
import time
from pathlib import Path

from app.services.loss_weight_utils import (all_weights, apply_criterion_weights,
                                       apply_matcher_costs, aux_weight,
                                       describe)
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from nets.cltr import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLTR_CROP = 256           # 官方训练/分块推理尺寸
CLTR_THRESHOLD = 0.35     # 官方验证阈值


# ------------------------- 数据集 -------------------------

class CLTRDataset(Dataset):
    """CLTR 点标注数据集（训练返回 256 裁剪；验证返回整图 + GT 点）。"""

    def __init__(self, root, train=True, crop_size=CLTR_CROP,
                 num_queries=500, num_knn=4, augment=True,
                 augment_strength=1.0):
        from app.services.crowd_counting import _crowd_dirs
        self.root = Path(root)
        self.train = train
        self.crop = max(64, int(crop_size or CLTR_CROP))
        self.num_queries = max(8, int(num_queries or 500))
        self.num_knn = max(2, int(num_knn or 4))
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
                f"未找到点标注样本（CLTR 需要 .txt / .mat 点标注或 (N,2) "
                f"坐标 .npy）：{root}")
        self.items = items
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.items)

    def _load_points(self, gt_path):
        from app.services.crowd_counting import _load_points_gt_any
        return _load_points_gt_any(gt_path).astype(np.float32)   # (x, y)

    def _knn_distances(self, pts):
        """每点 k 近邻平均距离（官方 compute_knn_distance 等价实现）"""
        import scipy.spatial
        n = len(pts)
        if n >= 4:
            tree = scipy.spatial.cKDTree(pts, leafsize=2048)
            d, _ = tree.query(pts, k=min(self.num_knn, n))
            return d[:, 1:].mean(axis=1).astype(np.float32)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        if n == 1:
            return np.zeros(1, dtype=np.float32)
        tree = scipy.spatial.cKDTree(pts, leafsize=2048)
        d, _ = tree.query(pts, k=n)
        if n == 2:
            return d[:, 1].astype(np.float32)
        return ((d[:, 1] + d[:, 2]) / 2.0).astype(np.float32)

    def _random_crop(self, img_t, pts, crop):
        """随机裁 crop×crop，返回 (crop_t, pts_yx_local)；点为空也允许。"""
        H, W = img_t.shape[-2:]
        best = None
        for _ in range(20):
            y0 = np.random.randint(0, H - crop + 1) if H > crop else 0
            x0 = np.random.randint(0, W - crop + 1) if W > crop else 0
            y1, x1 = min(H, y0 + crop), min(W, x0 + crop)
            crop_t = img_t[:, y0:y1, x0:x1]
            if len(pts):
                inside = ((pts[:, 0] >= y0) & (pts[:, 0] < y1) &
                          (pts[:, 1] >= x0) & (pts[:, 1] < x1))
                p = pts[inside].copy()
                p[:, 0] -= y0
                p[:, 1] -= x0
            else:
                p = pts
            ch, cw = crop_t.shape[-2:]
            if (ch, cw) != (crop, crop):
                crop_t = F.interpolate(crop_t.unsqueeze(0), size=(crop, crop),
                                       mode="bilinear", align_corners=False)[0]
                if len(p):
                    p[:, 0] *= crop / max(1, ch)
                    p[:, 1] *= crop / max(1, cw)
            n = len(p)
            if 1 <= n <= self.num_queries:
                return crop_t, p
            if n > self.num_queries:
                idx = np.random.choice(n, self.num_queries, replace=False)
                return crop_t, p[idx]
            if best is None:
                best = (crop_t, p)
        return best if best is not None else (crop_t, p)

    def __getitem__(self, idx):
        img_path, gt_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        pts = self._load_points(gt_path)                  # (x, y)
        W, H = img.size

        if self.train:
            # 保证短边 >= crop（不足则等比放大）
            if min(W, H) < self.crop:
                s = self.crop / max(1, min(W, H))
                nw, nh = max(1, int(W * s)), max(1, int(H * s))
                img = img.resize((nw, nh), Image.BILINEAR)
                if len(pts):
                    pts = pts * np.array([nw / W, nh / H], dtype=np.float32)
                W, H = nw, nh
            # 随机尺度 [1-0.2s, 1+0.2s]
            if self.augment:
                s = self.aug_strength
                lo, hi = 1 - 0.2 * s, 1 + 0.2 * s
                scale = np.random.uniform(lo, hi) if hi > lo else 1.0
                if scale > 0 and min(W * scale, H * scale) >= self.crop:
                    nw, nh = max(1, int(W * scale)), max(1, int(H * scale))
                    img = img.resize((nw, nh), Image.BILINEAR)
                    if len(pts):
                        pts = pts * np.array([nw / W, nh / H],
                                             dtype=np.float32)
                    W, H = nw, nh
                # 随机水平翻转（x 坐标同步）
                if np.random.rand() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    if len(pts):
                        pts = pts.copy()
                        pts[:, 0] = (W - 1) - pts[:, 0]

            img_t = self.transform(img)
            pts_yx = pts[:, ::-1].copy()                  # (y, x)
            img_t, pts_yx = self._random_crop(img_t, pts_yx, self.crop)
            n = len(pts_yx)
            dist = self._knn_distances(pts_yx).reshape(-1, 1)
            points = np.concatenate([pts_yx, dist], axis=1) / self.crop
            target = {
                "labels": torch.ones(n, dtype=torch.long),
                "points": torch.from_numpy(points).float(),
                "points_macher": torch.from_numpy(points).float(),
            }
            return img_t, target

        # 验证：返回整图 + GT 点（(y,x) 像素坐标，供计数）
        img_t = self.transform(img)
        pts_yx = np.ascontiguousarray(pts[:, ::-1])
        return img_t, torch.from_numpy(pts_yx).float()


def cltr_collate(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)


# ------------------------- 模型 / 权重 -------------------------

def _cltr_args(device, pretrained=False, params=None):
    p = params or {}
    return SimpleNamespace(
        dataset_file="crowd_data", device=str(device),
        masks=False, frozen_weights=None, aux_loss=True,
        dec_layers=6, cls_loss_coef=2.0, point_loss_coef=5.0,
        giou_loss_coef=2.0, mask_loss_coef=1.0, dice_loss_coef=1.0,
        focal_alpha=0.25,
        num_queries=int(p.get("cltr_num_queries", 500)),
        channel_point=3,
        backbone="resnet50", dilation=False,
        position_embedding="sine",
        enc_layers=6, hidden_dim=256, dropout=0.1, nheads=8,
        dim_feedforward=2048, pre_norm=False,
        lr_backbone=float(p.get("lr_backbone", 1e-4) or 1e-4),
        set_cost_class=2.0, set_cost_point=5.0, set_cost_giou=2.0,
        pretrained=bool(pretrained))


def _extract_state(ck):
    if not isinstance(ck, dict):
        return ck
    for key in ("state_dict", "model", "model_state_dict"):
        sd = ck.get(key)
        if isinstance(sd, dict):
            return sd
    return ck


def _strip_module(sd):
    return {k[7:] if k.startswith("module.") else k: v
            for k, v in sd.items()}


def _load_cltr_weights(model, weight_path):
    ck = torch.load(weight_path, map_location="cpu")
    sd = _strip_module(_extract_state(ck))
    total = len(model.state_dict())
    res = model.load_state_dict(sd, strict=False)
    matched = len(sd) - len(res.unexpected_keys)
    if matched < total * 0.95:
        raise ValueError(
            f"CLTR 权重与模型架构不匹配（{matched}/{total} 个参数键命中，"
            f"missing={len(res.missing_keys)} "
            f"unexpected={len(res.unexpected_keys)}）: {weight_path}")
    if res.missing_keys or res.unexpected_keys:
        logging.getLogger(__name__).warning(
            "CLTR 权重兼容加载: missing=%d unexpected=%d | missing[:10]=%s "
            "unexpected[:10]=%s",
            len(res.missing_keys), len(res.unexpected_keys),
            res.missing_keys[:10], res.unexpected_keys[:10])
    return res


# ------------------------- 分块推理 -------------------------

def _tile_patches(img_t, crop):
    """整图 padding 到 crop 倍数并切成 patch：[N,3,crop,crop]"""
    C, H, W = img_t.shape
    ph = (crop - H % crop) % crop
    pw = (crop - W % crop) % crop
    if ph or pw:
        img_t = F.pad(img_t, (0, pw, 0, ph), value=0)
    Hp, Wp = img_t.shape[-2:]
    nh, nw = Hp // crop, Wp // crop
    patches = (img_t.view(C, nh, crop, Wp)
               .view(C, nh, crop, nw, crop)
               .permute(1, 3, 0, 2, 4).contiguous()
               .view(nh * nw, C, crop, crop))
    return patches, (nh, nw, Hp, Wp)


@torch.no_grad()
def _predict_points(model, img_t, device, crop, threshold, chunk=16):
    """返回原图坐标下的预测点 (y, x) 列表。"""
    model.eval()
    H0, W0 = img_t.shape[-2:]
    patches, (nh, nw, Hp, Wp) = _tile_patches(img_t, crop)
    out_pts = []
    n = patches.shape[0]
    for start in range(0, n, chunk):
        batch = patches[start:start + chunk].to(device)
        out = model(batch)
        logits = out["pred_logits"]              # [b,Q,2]
        pnts = out["pred_points"][..., :2]       # [b,Q,2] normalized (y,x)
        scores = logits.sigmoid()[..., 1]        # [b,Q]
        for bi in range(batch.shape[0]):
            idx = start + bi
            pr, pc = idx // nw, idx % nw
            keep = scores[bi] > threshold
            if not bool(keep.any()):
                continue
            p = pnts[bi][keep].cpu().numpy() * crop
            p[:, 0] += pr * crop
            p[:, 1] += pc * crop
            valid = (p[:, 0] < H0) & (p[:, 1] < W0)
            if valid.any():
                out_pts.extend(p[valid].tolist())
    return out_pts


# ------------------------- 训练 -------------------------

@torch.no_grad()
def _validate_cltr(model, loader, device, crop, threshold, task=None):
    """全图分块评估：MAE / RMSE + 验证损失（计数 MSE，与计数指标同口径）

    同时打印逐张验证进度（loss / MAE 滚动均值），便于观察测试进度。
    """
    model.eval()
    mae_sum, mse_sum, loss_sum, n = 0.0, 0.0, 0.0, 0
    total = max(1, len(loader))
    log_every = max(1, total // 10)
    if task is not None:
        task.log(f"  开始验证（测试集）：{total} 张图像（含损失）")
    t0 = time.time()
    for imgs, pts_list in loader:
        if task is not None and task.stop_event.is_set():
            raise StopIteration()
        img_t = imgs[0]
        gt_count = int(pts_list[0].shape[0])
        pred = _predict_points(model, img_t, device, crop, threshold)
        cnt = len(pred)
        mae_sum += abs(cnt - gt_count)
        mse_sum += (cnt - gt_count) ** 2
        loss_sum += (cnt - gt_count) ** 2
        n += 1
        if task is not None and (n % log_every == 0 or n >= total):
            task.log(
                f"  验证 {n}/{total} | loss "
                f"{loss_sum / max(1, n):.5f} | MAE "
                f"{mae_sum / max(1, n):.3f} | RMSE "
                f"{math.sqrt(mse_sum / max(1, n)):.3f} | "
                f"用时 {time.time() - t0:.1f}s")
    model.train()
    n = max(1, n)
    return mae_sum / n, math.sqrt(mse_sum / n), loss_sum / n


def run_cltr_train(task, params: dict):
    """CLTR 点定位训练（官方：Adam + MultiStepLR + 模型内 criterion）"""
    from app.services.train_common import (set_seed, EarlyStopper,
                                           save_checkpoint, save_loss_curves,
                                           report_batch_progress,
                                           is_main_process,
                                           release_training_memory)
    from app.services.crowd_counting import _ckpt_paths, clear_inference_cache

    dataset_path = params["dataset_path"]
    epochs = int(params.get("epochs", 30))
    eval_every = max(1, int(params.get("eval_every") or 1))
    batch_size = int(params.get("batch_size", 16))
    # lr=0 / 缺省 → CLTR 论文默认 lr=1e-4
    lr = float(params.get("lr") or 1e-4)
    lr_backbone = float(params.get("lr_backbone") or 1e-4)
    weight_decay = float(params.get("weight_decay", 5e-4))
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
    num_queries = int(params.get("cltr_num_queries", 500))
    crop_size = int(params.get("img_size", 0) or CLTR_CROP)
    if crop_size != CLTR_CROP:
        task.log(f"CLTR 官方按 {CLTR_CROP} crop 训练/分块推理，"
                 f"本次使用 {CLTR_CROP}")
        crop_size = CLTR_CROP
    threshold = float(params.get("cltr_threshold", CLTR_THRESHOLD))

    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    task.total_epochs = epochs
    task.log(f"CLTR 点定位训练 | ECCV 2022 Conditional DETR | "
             f"epochs={epochs} batch={batch_size} crop={crop_size} "
             f"queries={num_queries}")
    task.log(f"数据集: {dataset_path} | 设备: {device_str} | lr={lr}")
    set_seed(int(params.get("seed", 42)))
    _n_cached = clear_inference_cache()
    if _n_cached:
        task.log(f"已清空推理模型缓存（{_n_cached} 个），为训练让出内存")

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    full = CLTRDataset(dataset_path, train=True, crop_size=crop_size,
                       num_queries=num_queries, augment=augment,
                       augment_strength=aug_strength)
    # 验证集：优先数据集自带 Test 划分；否则按 val_ratio 从训练集切分
    from app.services.crowd_counting import _eval_from_test
    val_eval_ds, val_src = _eval_from_test(
        dataset_path,
        lambda: CLTRDataset(dataset_path, train=False,
                            crop_size=crop_size,
                            num_queries=num_queries, augment=False))
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
                              collate_fn=cltr_collate)
    if len(train_loader) == 0:
        raise ValueError(
            f"训练批数为 0（训练样本 {len(train_set)} < Batch Size "
            f"{batch_size}，CLTR 启用 drop_last），请减小 Batch Size 或补充数据")
    val_loader = (DataLoader(val_set, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=cltr_collate)
                  if n_val else None)
    task.log(f"训练样本: {len(train_set)} | 验证样本: {len(val_set)}（{val_src}）")

    use_imagenet = pretrained and not (weight_path or resume_weight)
    model, criterion, _post = build_model(
        _cltr_args(device, pretrained=use_imagenet, params=params))
    if weight_path:
        _load_cltr_weights(model, weight_path)
        task.log(f"CLTR 初始权重: {weight_path}")
    model.to(device)
    criterion.to(device)
    # 损失分项权重（缺省 = CLTR 源码设置：ce 2 / points 5 / giou 2 / 代价 2、5）
    _lw = all_weights(params, {"ce_w": 2.0, "points_w": 5.0, "giou_w": 2.0,
                               "cost_class": 2.0, "cost_point": 5.0})
    apply_criterion_weights(criterion, _lw)
    apply_matcher_costs(criterion, _lw)
    _aux_w = aux_weight(params)
    task.log(f"损失权重: {describe(_lw)}")

    # 优化器 / 调度器 / 损失可选：默认对齐 CLTR 论文
    # （Adam + MultiStepLR + 点定位损失），可追加 crowd_point 自定义附加损失
    from app.core.optim_builders import build_torch_optimizer
    from app.services.crowd_counting import build_crowd_scheduler
    optimizer = build_torch_optimizer(params, model.parameters(), lr,
                                      weight_decay=weight_decay)
    scheduler, _is_plateau = build_crowd_scheduler(
        optimizer, params.get("scheduler") or "cosine", epochs, params)
    task.log(f"优化器: {params.get('optimizer') or 'adam'}"
             f" | 调度器: {params.get('scheduler') or 'cosine'}")
    loss_name = str(params.get("loss") or "cltr")
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

    csv_path = save_dir / "metrics.csv"
    header = ["epoch", "train_loss", "val_loss", "val_mae", "val_rmse"]
    best_mae, start_epoch = float("inf"), 1
    best_path, last_path = _ckpt_paths(save_dir, "cltr")
    if resume_weight:
        ck = torch.load(resume_weight, map_location="cpu")
        model.load_state_dict(_strip_module(_extract_state(ck)), strict=True)
        if isinstance(ck, dict):
            opt = ck.get("optimizer_state_dict") or ck.get("optimizer")
            if opt:
                optimizer.load_state_dict(opt)
            sch = ck.get("scheduler_state_dict")
            if sch:
                scheduler.load_state_dict(sch)
            start_epoch = int(ck.get("epoch", 0)) + 1
            bm = ck.get("best_mae", ck.get("best_prec1"))
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
        total_loss, n_batches = 0.0, 0
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)
        _main = is_main_process(params)
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            if task.stop_event.is_set():
                raise StopIteration()
            imgs = imgs.to(device)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in t.items()} for t in targets]
            outputs = model(imgs)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k]
                       for k in loss_dict.keys() if k in weight_dict)
            if aux_loss is not None:
                extra = aux_loss(outputs, targets)
                if isinstance(extra, dict):
                    extra = sum(v for v in extra.values())
                loss = loss + _aux_w * extra
            if not torch.isfinite(loss):
                task.log(f"损失非有限值，跳过该 batch: {loss_dict}")
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
            report_batch_progress(
                task, epoch=epoch, epochs=epochs, batch_idx=batch_idx,
                total_batches=total_batches, running_loss=total_loss,
                n_batches=n_batches, log_every=log_every, is_main=_main)

        train_loss = total_loss / max(1, n_batches)
        val_mae = val_rmse = val_loss = None
        if val_loader is not None and do_eval:
            val_mae, val_rmse, val_loss = _validate_cltr(
                model, val_loader, device, crop_size, threshold, task=task)
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
                        epoch=epoch, best_metric=task.best, model="cltr",
                        val_mae=val_mae)
        if is_best:
            save_checkpoint(last_path, net=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch,
                            best_metric=task.best, model="cltr",
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

def cltr_infer(image: Image.Image, weight_path=None, device="cpu",
               pretrained=True, conf_threshold=None):
    """CLTR 推理：整图 256 分块，按 score 阈值取点并可视化。"""
    import base64
    import io

    import cv2

    from app.services.crowd_counting import (_model_cache_key,
                                             _model_cache_get,
                                             _model_cache_put)

    if not weight_path:
        raise ValueError(
            "CLTR 推理需要权重：请选择训练产物 / 上传 .pth（本项目未内置 "
            "CLTR 训练权重）")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    cache_key = _model_cache_key("cltr", weight_path, device)
    model = _model_cache_get(cache_key)
    if model is None:
        model, _, _ = build_model(_cltr_args(dev, pretrained=False))
        _load_cltr_weights(model, weight_path)
        model.to(dev).eval()
        _model_cache_put(cache_key, model)

    img = image.convert("RGB")
    W0, H0 = img.size
    x = transforms.functional.to_tensor(img)
    x = transforms.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    threshold = float(conf_threshold if conf_threshold is not None
                      else CLTR_THRESHOLD)
    pts_yx = _predict_points(model, x, dev, CLTR_CROP, threshold)
    pts_xy = np.asarray(pts_yx, dtype=np.float32)[:, ::-1] if pts_yx else \
        np.zeros((0, 2), dtype=np.float32)

    canvas = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
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
            "conf_threshold": threshold, "model": "cltr",
            "point_based": True}

