# -*- coding: utf-8 -*-
"""单域泛化（Single-Domain Generalization, SDG）人群计数服务

算法（均基于密度图，训练用源域 A，测试/泛化评估用目标域 B）：
- sdg_dccus   DCCUS（AAAI 2023）：在线伪域聚类 + MAML 式元学习
              （语义/风格 memory 重建 CE + 正交 + 前景 mask L1）
- sdg_llb     LLB_Count：DCCUS 元学习 + 局部感知损失 + 边界感知损失
- sdg_mpcount MPCount（CVPR 2024）：自适应 memory bank + patch 分类 +
              双分支一致性（普通 epoch 循环，非元学习）

设计：
- 与既有「密度图 / 点预测 / 头部框」路线完全隔离：独立子任务 crowd_sdg
- 训练可选「源数据集」+「目标数据集」：目标集仅用于跨域验证与最优模型选择，
  不参与梯度更新（严格单域泛化设定）
- 测试模式（独立后台任务）：加载训练权重，在任意测试集上仅评估
  MAE / MSE（+ 逐图 CSV），用于衡量跨域泛化性
- 计数换算：count = density.sum() / log_para（默认 log_para=1000，三份源码一致）
- 推理/测试均采用 clamp 窗口 + mask 平均切块（DCCUS/LLB 同源策略）
"""
import copy
import csv
import io
import random
import time
from pathlib import Path

from app.services.loss_weight_utils import all_weights, describe

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.services.crowd_counting import (_crowd_dirs, _ckpt_paths,
                                         gaussian_kernel_density)

LOG_PARA = 1000
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 模型注册表（前端下拉 / 路由分发）
SDG_MODELS = {
    "sdg_dccus": {
        "label": "DCCUS（AAAI 2023）",
        "desc": "元学习 + 语义/风格记忆（convNeXt_tiny / vgg 骨干，"
                "近似域聚类 + MAML 内外循环）",
        "algo": "dccus", "backbone": "convNeXt_tiny",
    },
    "sdg_llb": {
        "label": "LLB_Count（局部-边界感知）",
        "desc": "DCCUS 元学习框架 + 局部感知损失 + 边界感知损失"
                "（convNeXt_tiny / vgg 骨干）",
        "algo": "llb", "backbone": "convNeXt_tiny",
    },
    "sdg_mpcount": {
        "label": "MPCount（CVPR 2024）",
        "desc": "自适应记忆库 AMB + patch 分类 + 双分支一致性"
                "（VGG16-BN 骨干，普通训练）",
        "algo": "mpcount", "backbone": "vgg16_bn",
    },
}


# ---------------------------------------------------------------------------
# 模型构建 / 权重加载
# ---------------------------------------------------------------------------

def list_models():
    """前端模型清单条目（与 crowd_counting.list_models 同结构）"""
    return [{"name": k, "label": v["label"], "desc": v["desc"],
             "type": "sdg", "pretrained": False, "need_weight": True}
            for k, v in SDG_MODELS.items()]


def _build_model(model_name, backbone=None, num_clusters=4, threshold=0.1,
                 pretrained=False):
    """按注册表构建 SDG 模型（未加载权重）"""
    from nets.sdg import build_metamemnet, build_mpcount
    meta = SDG_MODELS.get(model_name)
    if meta is None:
        raise ValueError(f"未知的单域泛化模型: {model_name}"
                         f"（可选: {list(SDG_MODELS)}）")
    if meta["algo"] == "mpcount":
        return build_mpcount(pretrained=pretrained)
    return build_metamemnet(pretrained=pretrained,
                            backbone_name=backbone or meta["backbone"],
                            variant=meta["algo"],
                            num_clusters=int(num_clusters),
                            threshold=float(threshold))


def _load_weights(model, weight_path):
    """加载权重（兼容 state_dict / checkpoint）；返回 checkpoint 的 sdg_meta"""
    ck = torch.load(str(weight_path), map_location="cpu",
                    weights_only=False)
    sd, meta = None, None
    if isinstance(ck, dict):
        meta = ck.get("sdg_meta")
        sd = (ck.get("state_dict") or ck.get("model_state_dict")
              or (ck.get("model") if isinstance(ck.get("model"), dict) else None))
    if not isinstance(sd, dict):
        sd = ck
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    total = len(model.state_dict())
    matched = total - len(missing)
    if matched < total * 0.9:
        raise ValueError(
            f"权重与模型不匹配（{matched}/{total} 命中，"
            f"missing={len(missing)}, unexpected={len(unexpected)}）")
    return meta or {}


def _resolve_meta(model_name, backbone, num_clusters, ckpt_meta):
    """优先使用 checkpoint 内记录的算法元信息（保证测试/推理可复现）"""
    meta = dict(ckpt_meta or {})
    if meta.get("model_name") in SDG_MODELS:
        model_name = meta["model_name"]
    backbone = meta.get("backbone") or backbone or \
        SDG_MODELS[model_name]["backbone"]
    num_clusters = int(meta.get("num_clusters") or num_clusters or 4)
    return model_name, backbone, num_clusters


# ---------------------------------------------------------------------------
# 数据集（统一：Train|Test/Images + GT/*.npy|.txt；密度图 ×log_para）
# ---------------------------------------------------------------------------

def _split_dirs(root, train=True):
    """定位划分目录（train=True 取 Train，False 取 Test；扁平结构兜底）"""
    root = Path(root)
    img_dir, gt_dir = _crowd_dirs(root, train)
    if img_dir is None:
        img_dir, gt_dir = _crowd_dirs(root, not train)
    return img_dir, gt_dir


def _list_items(img_dir, gt_dir):
    items = []
    if img_dir is None:
        return items
    for p in sorted(Path(img_dir).iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        kind, gt = None, None
        cands = []
        if gt_dir is not None:
            cands = [(gt_dir / (p.stem + ".npy"), "npy"),
                     (gt_dir / (p.stem + ".txt"), "txt"),
                     (gt_dir / (p.stem + ".mat"), "mat")]
        else:
            for d in ("gt", "npys", "GT"):
                cands += [(Path(img_dir).parent / d / (p.stem + ".npy"), "npy"),
                          (Path(img_dir).parent / d / (p.stem + ".txt"), "txt")]
        for g, k in cands:
            if Path(g).exists():
                kind, gt = k, Path(g)
                break
        items.append((p, gt, kind))
    return items


def _load_density(gt_path, kind, h, w, sigma):
    """GT → 密度图（.npy 直接读；.txt/.mat 用高斯核生成；尺寸对齐原图）"""
    from app.services.crowd_counting import _resize_density, load_mat_points
    if kind == "npy":
        arr = np.load(str(gt_path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = arr.astype(np.float32)
        if arr.shape != (h, w):
            arr = _resize_density(arr, w, h)
        return arr
    if kind == "mat":
        pts, _ = load_mat_points(gt_path)
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    else:
        pts = []
        for line in Path(gt_path).read_text(encoding="utf-8-sig").splitlines():
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    pts.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    pass
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return gaussian_kernel_density((h, w), pts, sigma)


def _pad16(t):
    """t [1,3,H,W] → 补到 16 倍数（MPCount 的 /16 patch 网格要求）"""
    h, w = t.shape[-2:]
    nh = (h // 16 + (h % 16 != 0)) * 16
    nw = (w // 16 + (w % 16 != 0)) * 16
    if nh == h and nw == w:
        return t, 0, 0
    return F.pad(t, (0, nw - w, 0, nh - h), mode="constant", value=0), \
        nw - w, nh - h


class SDGDataset(torch.utils.data.Dataset):
    """SDG 训练/评估数据集：返回两视图（img1/den1, img2/den2）

    - 几何变换一次完成（img 与 den 同步 crop/flip）；img2 再做一次随机翻转
      （DCCUS/LLB 源实现语义：两视图几何可不同、各自带对应 GT）
    - mode="mpcount" 时 img2 使用 0.5/0.5 归一化（MPCount 风格偏移）
    """

    def __init__(self, root, crop_size=320, train=True, augment=True,
                 mode="dccus", sigma=8.0, log_para=LOG_PARA, seed=42):
        self.mode = mode
        self.train = train
        self.crop = max(64, int(crop_size or 320))
        self.augment = bool(augment)
        self.sigma = float(sigma)
        self.log_para = float(log_para)
        self.rng = random.Random(seed)
        img_dir, gt_dir = _split_dirs(root, train)
        if img_dir is None:
            raise ValueError(f"未找到图像目录（需 Train|Test/Images + GT）：{root}")
        self.items = _list_items(img_dir, gt_dir)
        if not self.items:
            raise ValueError(f"未找到有效样本：{root}")

    def __len__(self):
        return len(self.items)

    def _aug(self, img, den):
        """随机裁剪（不足则居中补零）+ 水平翻转（图像/密度同步）"""
        w, h = img.size
        c = self.crop
        if h < c or w < c:
            nh, nw = max(h, c), max(w, c)
            canvas = Image.new("RGB", (nw, nh), (0, 0, 0))
            canvas.paste(img, ((nw - w) // 2, (nh - h) // 2))
            img = canvas
            d2 = np.zeros((nh, nw), np.float32)
            d2[(nh - h) // 2:(nh - h) // 2 + h,
               (nw - w) // 2:(nw - w) // 2 + w] = den
            den = d2
            w, h = nw, nh
        i = self.rng.randint(0, h - c)
        j = self.rng.randint(0, w - c)
        img = img.crop((j, i, j + c, i + c))
        den = den[i:i + c, j:j + c].copy()
        if self.rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            den = np.ascontiguousarray(den[:, ::-1])
        return img, den

    def _to_tensor(self, img, mean, std):
        t = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())
        t = t.permute(2, 0, 1).float().div_(255.0)
        m = torch.tensor(mean).view(3, 1, 1)
        s = torch.tensor(std).view(3, 1, 1)
        return (t - m) / s

    def __getitem__(self, idx):
        import torchvision.transforms.functional as TF
        img_p, gt_p, kind = self.items[idx]
        img = Image.open(img_p).convert("RGB")
        W, H = img.size
        if gt_p is None or kind is None:
            den = np.zeros((H, W), np.float32)
        else:
            den = _load_density(gt_p, kind, H, W, self.sigma)
        den = den * self.log_para

        if self.train:
            if self.augment:
                img, den = self._aug(img, den)
            else:
                img, den = self._aug(img, den) if (img.size[0] < self.crop
                                                   or img.size[1] < self.crop) \
                    else (img, den)
            # 训练：几何已由 _aug 统一为 crop×crop（保证同 batch 尺寸一致；
            # 多尺度在 crop 前不可行时由 _aug 的补边/裁剪保证覆盖），这里
            # 直接做光度增强与第二视图派生
            if self.augment:
                # 光度增强（img1：标准；img2：更强风格偏移）
                img1 = TF.adjust_brightness(
                    img, 1 + self.rng.uniform(-0.3, 0.3) * 0.8)
                img1 = TF.adjust_contrast(
                    img1, 1 + self.rng.uniform(-0.3, 0.3) * 0.8)
                img1 = TF.adjust_saturation(
                    img1, 1 + self.rng.uniform(-0.2, 0.2) * 0.8)
                if self.rng.random() < 0.2:
                    img1 = img1.convert("L").convert("RGB")
                img2 = TF.adjust_brightness(
                    img, 1 + self.rng.uniform(-0.4, 0.4))
                img2 = TF.adjust_contrast(
                    img2, 1 + self.rng.uniform(-0.4, 0.4))
                img2 = TF.adjust_saturation(
                    img2, 1 + self.rng.uniform(-0.4, 0.4))
                if self.rng.random() < 0.2:
                    img2 = img2.convert("L").convert("RGB")
                den2 = den
                if self.rng.random() < 0.5:      # 第二视图独立翻转（同源语义）
                    img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
                    den2 = np.ascontiguousarray(den[:, ::-1])
            else:
                img1 = img2 = img
                den2 = den

            mean2 = (0.5, 0.5, 0.5) if self.mode == "mpcount" \
                else IMAGENET_MEAN
            std2 = (0.5, 0.5, 0.5) if self.mode == "mpcount" \
                else IMAGENET_STD
            t1 = self._to_tensor(img1, IMAGENET_MEAN, IMAGENET_STD)
            t2 = self._to_tensor(img2, mean2, std2)
            return (t1, torch.from_numpy(np.asarray(den)).float().unsqueeze(0),
                    t2, torch.from_numpy(np.asarray(den2)).float().unsqueeze(0))
        # 评估：整图（外部负责 pad16），这里只做归一化
        t1 = self._to_tensor(img, IMAGENET_MEAN, IMAGENET_STD)
        return (t1, torch.from_numpy(den).float().unsqueeze(0))


class SDGClusterDataset(torch.utils.data.Dataset):
    """在线聚类用：图像 resize 224 + ImageNet 归一化"""

    def __init__(self, root, train=True):
        img_dir, _ = _split_dirs(root, train)
        if img_dir is None:
            raise ValueError(f"未找到图像目录: {root}")
        self.paths = [p for p in sorted(Path(img_dir).iterdir())
                      if p.is_file() and p.suffix.lower() in IMG_EXTS]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        t = Image.open(self.paths[idx]).convert("RGB").resize((224, 224))
        arr = np.asarray(t, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (t - m) / s, self.paths[idx].stem


# ---------------------------------------------------------------------------
# 推理 / 评估
# ---------------------------------------------------------------------------

def _forward_density(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


@torch.no_grad()
def predict_count(model, img_t, log_para=LOG_PARA, patch_size=0,
                  return_density=False):
    """单图人数：clamp 窗口 + mask 平均切块（DCCUS/LLB myEvalator 同源）

    return_density=True 时返回 (count, 密度图)，供验证损失计算复用。
    """
    model.eval()
    b, _, h, w = img_t.shape
    ps = int(patch_size or 0)
    if ps <= 0 or (h <= ps and w <= ps):
        dens = _forward_density(model, img_t)
        cnt = float(dens.sum().item()) / log_para
        return (cnt, dens) if return_density else cnt
    mask = torch.zeros(1, 1, h, w, device=img_t.device)
    pred = torch.zeros(1, 1, h, w, device=img_t.device)
    for i in range(0, h, ps):
        gis, gie = max(min(h - ps, i), 0), min(h, i + ps)
        for j in range(0, w, ps):
            gjs, gje = max(min(w - ps, j), 0), min(w, j + ps)
            crop = img_t[:, :, gis:gie, gjs:gje]
            out = _forward_density(model, crop)
            if out.shape[-2:] != (gie - gis, gje - gjs):
                out = F.interpolate(out, size=(gie - gis, gje - gjs),
                                    mode="bilinear", align_corners=False)
            pred[:, :, gis:gie, gjs:gje] += out
            mask[:, :, gis:gie, gjs:gje] += 1
    dens = pred / mask.clamp_min(1.0)
    cnt = float(dens.sum().item()) / log_para
    return (cnt, dens) if return_density else cnt


def _image_tensor(img, device):
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = ((t - m) / s).unsqueeze(0)
    t, _, _ = _pad16(t)
    return t.to(device)


@torch.no_grad()
def evaluate_dataset(task, model, root, device, log_para=LOG_PARA,
                     patch_size=0, csv_path=None, tag="test",
                     items=None, epoch=None, epochs=None, pred_dir=None,
                     png_dir=None):
    """跨域评估：MAE / MSE（+ 逐图 CSV）。返回 result dict

    items 给出时直接使用该样本列表（训练期无目标域时由源训练集
    按 val_ratio 确定性切分得到）；否则优先使用数据集 Test/Val 划分，
    无划分时退回全部图像（作为独立测试集使用）。
    """
    if items is None:
        img_dir, _ = _split_dirs(root, train=False)
        if img_dir is None:
            img_dir, _ = _split_dirs(root, train=True)
        if img_dir is None:
            raise ValueError(f"测试集未找到图像目录: {root}")
        items = _list_items(img_dir, None)
        task.log(f"[{tag}] 测试集: {img_dir}（{len(items)} 张）| "
                 f"patch={patch_size or '整图'} | log_para={log_para}")
    else:
        task.log(f"[{tag}] 评估样本: {len(items)} 张（指定子集）| "
                 f"patch={patch_size or '整图'} | log_para={log_para}")
    if not items:
        raise ValueError(f"测试集无图像: {root}")
    model.eval()
    rows, errs = [], []
    loss_sum, nl = 0.0, 0
    t0 = time.time()
    for i, (img_p, gt_p, kind) in enumerate(items, 1):
        if task.stop_event.is_set():
            raise StopIteration()
        if epoch is not None and epochs:
            task.progress = ((epoch - 1) + i / len(items)) / max(1, epochs) * 100.0
        else:
            task.progress = i / len(items) * 100.0
        img = Image.open(img_p).convert("RGB")
        W, H = img.size
        if gt_p is None:
            continue
        den = _load_density(gt_p, kind, H, W, 8.0)
        gt_cnt = float(den.sum())
        x = _image_tensor(img, device)
        pred_cnt, dens_pred = predict_count(
            model, x, log_para=log_para, patch_size=patch_size,
            return_density=True)
        # 验证/测试损失：预测密度 vs GT 密度（×log_para，padding 区 GT=0）
        try:
            gt_d = torch.from_numpy(np.asarray(den, np.float32)).to(device) \
                * float(log_para)
            hh, ww = dens_pred.shape[-2:]
            pb, pr = hh - gt_d.shape[0], ww - gt_d.shape[1]
            if pb > 0 or pr > 0:
                gt_d = F.pad(gt_d, (0, max(0, pr), 0, max(0, pb)))
            gt_d = gt_d[:hh, :ww]
            lv = F.mse_loss(dens_pred[0, 0], gt_d)
            if torch.isfinite(lv):
                loss_sum += float(lv)
                nl += 1
        except Exception:
            pass
        if pred_dir is not None:
            # 保存逐图预测密度图（.npy，sum≈预测人数；统一到原图分辨率）
            dm = dens_pred
            if dm.dim() == 4:
                dm = dm[0, 0]
            dm = (dm.float() / max(float(log_para), 1e-8)).detach().cpu()
            if tuple(dm.shape[-2:]) != (H, W):
                dm = F.interpolate(dm[None, None], size=(H, W),
                                   mode="bilinear", align_corners=False)[0, 0]
            dsum = float(dm.sum())
            if dsum > 1e-8 and pred_cnt > 0:
                dm = dm * (pred_cnt / dsum)
            from app.services.crowd_counting import _save_density_npy
            dm_np = dm.numpy()
            _save_density_npy(Path(pred_dir), img_p.stem, dm_np)
            if png_dir is not None:
                from app.services.crowd_counting import _save_density_png
                _save_density_png(Path(png_dir), img_p.stem, img_p, dm_np,
                                  pred_cnt)
        err = pred_cnt - gt_cnt
        errs.append(err)
        rows.append({"name": img_p.stem, "pred": round(pred_cnt, 4),
                     "gt": round(gt_cnt, 4), "error": round(err, 4)})
        if i % 20 == 0 or i == len(items):
            task.log(f"[{tag}] {i}/{len(items)}  pred={pred_cnt:.2f} "
                     f"gt={gt_cnt:.2f} | loss "
                     f"{loss_sum / max(1, nl):.5f} | "
                     f"用时 {time.time() - t0:.1f}s")
    if not errs:
        raise ValueError("测试集无有效标注样本")
    errs = np.asarray(errs, dtype=np.float64)
    mae = float(np.mean(np.abs(errs)))
    mse = float(np.sqrt(np.mean(errs ** 2)))   # 与三份源码一致：MSE=√mean(err²)
    loss_mean = round(loss_sum / nl, 5) if nl else None
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name", "pred", "gt", "error"])
            w.writeheader()
            w.writerows(rows)
    task.log(f"[{tag}] 完成：MAE={mae:.3f} | RMSE={mse:.3f}"
             + (f" | loss={loss_mean:.5f}" if loss_mean is not None else "")
             + f" | 样本={len(errs)}")
    return {"mae": round(mae, 3), "mse": round(mse, 3),
            "rmse": round(mse, 3),
            "loss": loss_mean,
            "num_images": int(len(errs)), "dataset": str(root),
            "per_image_csv": str(csv_path) if csv_path else "",
            "saved_outputs": int(len(rows)) if pred_dir is not None else 0,
            "saved_pngs": int(len(rows)) if png_dir is not None else 0,
            "pred_dir": str(pred_dir) if pred_dir is not None else "",
            "png_dir": str(png_dir) if png_dir is not None else "",
            "rows": rows}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def _optimizer_scheduler(params, model, epochs):
    from app.core.optim_builders import build_torch_optimizer
    from app.services.crowd_counting import build_crowd_scheduler
    lr = float(params.get("lr") or 1e-3)
    wd = float(params.get("weight_decay", 1e-4))
    optimizer = build_torch_optimizer(params, model.parameters(), lr,
                                      weight_decay=wd)
    default_sched = "cosine"
    scheduler, is_plateau = build_crowd_scheduler(
        optimizer, params.get("scheduler") or default_sched, epochs, params)
    return optimizer, scheduler, is_plateau, lr


def _checkpoint_meta(model_name, backbone, num_clusters, params):
    return {"model_name": model_name, "backbone": backbone,
            "num_clusters": int(num_clusters), "log_para": LOG_PARA,
            "algo": SDG_MODELS[model_name]["algo"],
            "crop_size": int(params.get("img_size") or 320)}


def _val_target_dir(params):
    target = (params.get("target_dataset_path") or "").strip()
    return target or None


def run_sdg_train(task, params: dict):
    """SDG 训练入口：源数据集 + 可选目标数据集（跨域验证）"""
    from app.services.train_common import (set_seed, save_checkpoint,
                                           save_loss_curves,
                                           release_training_memory)
    model_name = params.get("model", "sdg_dccus")
    if model_name not in SDG_MODELS:
        raise ValueError(f"未知的单域泛化模型: {model_name}"
                         f"（可选: {list(SDG_MODELS)}）")
    algo = SDG_MODELS[model_name]["algo"]
    seed = int(params.get("seed", 42))
    device_str = params.get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    epochs = int(params.get("epochs", 100))
    eval_every = max(1, int(params.get("eval_every")
                            or params.get("sdg_eval_every") or 1))
    crop = int(params.get("img_size") or 320)
    batch_size = int(params.get("batch_size", 1))
    num_clusters = max(2, int(params.get("sdg_num_clusters") or 4))
    iters = max(1, int(params.get("sdg_iters") or 20))
    backbone = params.get("sdg_backbone") or SDG_MODELS[model_name]["backbone"]
    pretrained = bool(params.get("pretrained", True))
    augment = bool(params.get("augment", True))
    source = params["dataset_path"]
    target = _val_target_dir(params)

    set_seed(seed)
    torch.manual_seed(seed)
    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    task.total_epochs = epochs
    task.log(f"单域泛化训练 | 模型: {model_name}（{SDG_MODELS[model_name]['label']}）")
    task.log(f"源数据集: {source} | 目标数据集: {target or '（未指定，仅源域）'}")
    task.log(f"骨干: {backbone} | 聚类数: {num_clusters} | "
             f"crop: {crop} | batch: {batch_size} | 设备: {device_str}")
    if algo == "mpcount":
        task.log("训练范式: 双分支一致性（普通 epoch 循环）| "
                 "损失: MSE + 10·BCE(patch) + 10·ACL")
    else:
        task.log(f"训练范式: 在线伪域聚类 + MAML 元学习（每 epoch {iters} 次）"
                 f"| 损失: {'全局 MSE + 局部感知 + 边界感知' if algo == 'llb' else '全局 MSE + 前景 mask L1'} + 语义/风格重建 + 正交")

    # 损失：算法内置损失为主（默认沿用论文设置）；可选注册到 crowd_sdg 的
    # 自定义附加损失（契约：forward(pred_density, gt_density) → 标量）
    loss_name = str(params.get("loss") or "")
    if algo == "mpcount":
        _lw = all_weights(params, {"den_w": 1.0, "cls_w": 10.0,
                                   "con_w": 10.0})
    elif algo == "llb":
        _lw = all_weights(params, {"den_w": 1.0, "local_w": 10.0,
                                   "bound_w": 10.0, "sim_w": 1.0,
                                   "orth_w": 1.0})
    else:   # dccus
        _lw = all_weights(params, {"den_w": 1.0, "mask_w": 100.0,
                                   "sim_w": 0.001, "orth_w": 1.0})
    task.log(f"损失权重: {describe(_lw)}")
    aux_loss, aux_w = None, float(params.get("loss_weight", 1.0) or 1.0)
    if loss_name.startswith("custom:"):
        from app.core.registry import get_custom_loss, is_custom_loss_for
        if not is_custom_loss_for(loss_name, "crowd_sdg"):
            raise ValueError(
                f"自定义损失 {loss_name} 未注册到 crowd_sdg 路线"
                f"（注册时请写 tasks=[\"crowd_sdg\"]）")
        aux_loss = get_custom_loss(loss_name, task="crowd_sdg")
        if hasattr(aux_loss, "to"):
            aux_loss = aux_loss.to(device)
        task.log(f"自定义附加损失: {loss_name}（与算法内置损失相加，权重 "
                 f"{aux_w}；契约 forward(pred_density, gt_density)）")
    else:
        task.log(f"损失: 算法内置默认（{loss_name or '论文默认'}）")

    model = _build_model(model_name, backbone, num_clusters,
                         pretrained=pretrained).to(device)
    optimizer, scheduler, is_plateau, lr = _optimizer_scheduler(
        params, model, epochs)
    task.log(f"优化器: {params.get('optimizer') or ('adamw' if algo == 'mpcount' else 'adam')}"
             f" | lr={lr} | 调度器: {params.get('scheduler') or 'cosine'}")

    ck_meta = _checkpoint_meta(model_name, backbone, num_clusters, params)
    best_path, last_path = _ckpt_paths(save_dir, model_name)
    csv_path = save_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_mae", "val_mse"])

    best_mae = float("inf")
    for epoch in range(1, epochs + 1):
        if task.stop_event.is_set():
            raise StopIteration()
        task.epoch = epoch
        task.progress = (epoch - 1) / max(1, epochs) * 100.0
        if algo == "mpcount":
            train_loss = _epoch_mpcount(task, model, optimizer, scheduler,
                                        source, crop, batch_size, augment,
                                        epochs, lr, seed + epoch, epoch=epoch,
                                        aux_loss=aux_loss, aux_w=aux_w,
                                        weights=_lw)
        else:
            train_loss = _epoch_meta(task, model, optimizer, scheduler,
                                     source, crop, batch_size, augment,
                                     num_clusters, iters, algo, epochs, lr,
                                     seed + epoch, device,
                                     aux_loss=aux_loss, aux_w=aux_w,
                                     weights=_lw)
        # 跨域验证（目标数据集；未指定则跳过；每 eval_every 轮一次）
        val_mae = val_mse = val_loss = None
        do_eval = ((eval_every <= 1) or (epoch % eval_every == 0)
                   or (epoch == epochs))
        if do_eval:
            # Do not write the per-image index into task.epoch here:
            # it would make the UI show "Epoch 111/1" (111 images, 1 epoch).
            task.epoch = epoch
            task.progress = max(task.progress, (epoch - 1) / max(1, epochs) * 100.0)
            res = None
            if target:
                res = evaluate_dataset(task, model, target, device,
                                       log_para=LOG_PARA,
                                       patch_size=int(params.get("sdg_test_patch") or 0),
                                       csv_path=None, tag=f"epoch{epoch}",
                                       epoch=epoch, epochs=epochs)
            else:
                # 无目标域：按 val_ratio 从源训练集确定性切分验证子集
                val_ratio = float(params.get("val_ratio", 0.2) or 0.2)
                src_img, src_gt = _split_dirs(source, train=True)
                all_items = _list_items(src_img, src_gt)
                if all_items:
                    n_val = max(1, int(len(all_items) * val_ratio)) \
                        if len(all_items) > 1 else len(all_items)
                    n_val = min(n_val, len(all_items))
                    rng = random.Random(seed)
                    idx = sorted(rng.sample(range(len(all_items)), n_val))
                    val_items = [all_items[i] for i in idx]
                    task.log(f"[epoch{epoch}] 未指定目标数据集：源训练集 "
                             f"按 {val_ratio:.0%} 确定性切分 "
                             f"（{len(val_items)}/{len(all_items)} 张）")
                    res = evaluate_dataset(
                        task, model, source, device, log_para=LOG_PARA,
                        patch_size=int(params.get("sdg_test_patch") or 0),
                        csv_path=None, tag=f"epoch{epoch}",
                        items=val_items, epoch=epoch, epochs=epochs)
            if res is not None:
                val_mae, val_mse = res["mae"], res["mse"]
                val_loss = res.get("loss")
        if val_mae is not None and is_plateau:
            scheduler.step(val_mae)
        else:
            scheduler.step()

        task.epoch = epoch
        task.progress = epoch / epochs * 100.0
        task.metrics = {"train_loss": round(train_loss, 4)}
        if val_mae is not None:
            task.metrics.update({"val_mae": val_mae, "val_mse": val_mse})
        if val_loss is not None:
            task.metrics["val_loss"] = val_loss
        task.history.append({"epoch": epoch, **task.metrics})
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}",
                                    "" if val_loss is None else val_loss,
                                    "" if val_mae is None else val_mae,
                                    "" if val_mse is None else val_mse])
        task.log(f"Epoch {epoch}/{epochs} | loss {train_loss:.4f}"
                 + (f" | 跨域 loss {val_loss:.5f}" if val_loss is not None
                    else "")
                 + (f" | 跨域 MAE {val_mae:.3f} / RMSE {val_mse:.3f}"
                    if val_mae is not None else ""))
        save_checkpoint(last_path, net=model, optimizer=optimizer,
                        scheduler=scheduler, epoch=epoch,
                        best_metric=task.best or None, sdg_meta=ck_meta)
        is_best = val_mae is not None and val_mae < best_mae
        if is_best:
            best_mae = val_mae
            task.best = {"epoch": epoch, "val_mae": val_mae,
                         "val_mse": val_mse}
            save_checkpoint(best_path, net=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch,
                            best_metric=task.best, sdg_meta=ck_meta)
        save_loss_curves(save_dir, task.history,
                         tuple(k for k in ("val_mae", "val_mse")
                               if val_mae is not None))
        release_training_memory()

    task.progress = 100.0
    task.result = {"best_weight": str(best_path), "last_weight": str(last_path),
                   "best_val_mae": None if best_mae == float("inf") else best_mae,
                   "save_dir": str(save_dir), "model": model_name}
    task.log("训练结束" + (f"，最优跨域 MAE={best_mae:.3f}"
                           if best_mae < float("inf") else "（未指定目标域验证）"))


def _epoch_mpcount(task, model, optimizer, scheduler, source, crop,
                   batch_size, augment, epochs, lr, seed, epoch=1,
                   aux_loss=None, aux_w=1.0, weights=None):
    """MPCount：普通 epoch 循环 + 双分支一致性"""
    ds = SDGDataset(source, crop_size=crop, train=True, augment=augment,
                    mode="mpcount", seed=seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=True, num_workers=0,
                                         drop_last=False)
    model.train()
    total, n = 0.0, 0
    dev = next(model.parameters()).device
    for bi, (img1, den1, img2, den2) in enumerate(loader, 1):
        if task.stop_event.is_set():
            raise StopIteration()
        img1, den1 = img1.to(dev), den1.to(dev)
        img2 = img2.to(dev)
        b, _, h, w = den1.shape
        bmap = F.avg_pool2d(den1, 16) > 0          # /16 patch 分类目标
        d1, d2, c1, c2, c_err, loss_con, _ = model.forward_train(
            img1, img2, bmap.float())
        loss_den = F.mse_loss(d1, den1) + F.mse_loss(d2, den1)
        loss_cls = F.binary_cross_entropy(c1, bmap.float()) + \
            F.binary_cross_entropy(c2, bmap.float())
        w = weights or {"den_w": 1.0, "cls_w": 10.0, "con_w": 10.0}
        loss = (float(w.get("den_w", 1.0)) * loss_den
                + float(w.get("cls_w", 10.0)) * loss_cls
                + float(w.get("con_w", 10.0)) * loss_con)
        if aux_loss is not None:
            loss = loss + float(aux_w) * aux_loss(d1, den1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        n += 1
        task.progress = ((epoch - 1) + bi / max(1, len(loader))) / max(1, epochs) * 100.0
        if bi % 10 == 0 or bi == len(loader):
            task.log(f"  [batch {bi}/{len(loader)}] loss={float(loss):.4f} "
                     f"(den {float(loss_den):.4f} + cls {float(loss_cls):.4f}"
                     f" + con {float(loss_con):.4f})")
    return total / max(1, n)


def _cluster_labels(task, model, source_dir, num_clusters, prev_labels,
                    seed, device):
    """在线伪域聚类：backbone 特征的 空间 mean/std → PCA → KMeans → 匈牙利对齐"""
    from nets.sdg.cluster import cluster_domains
    ds = SDGClusterDataset(source_dir, train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
    feats = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            fmap = model.conv_features(imgs.to(device))  # [B,256,h,w]
            b = fmap.shape[0]
            f = fmap.reshape(b, fmap.shape[1], -1)
            mean = f.mean(-1)
            std = f.var(-1).clamp_min(1e-5).sqrt()
            feats.append(torch.cat([mean, std], 1).cpu().numpy())
    feats = np.concatenate(feats, 0)
    labels = cluster_domains(feats, num_clusters, prev_labels=prev_labels,
                             seed=seed)
    task.log(f"  在线聚类：{len(labels)} 张 → {num_clusters} 个伪域 | 分布 "
             f"{np.bincount(labels, minlength=num_clusters).tolist()}")
    return labels


def _epoch_meta(task, model, optimizer, scheduler, source, crop, batch_size,
                augment, num_clusters, iters, algo, epochs, lr, seed,
                device, aux_loss=None, aux_w=1.0, weights=None):
    """DCCUS / LLB：在线伪域聚类 + MAML 元学习（内/外循环）"""
    from nets.sdg import LocalPerceptionLoss, BoundAwareLoss
    prev = getattr(task, "_sdg_prev_labels", None)
    labels = _cluster_labels(task, model, source, num_clusters, prev, seed,
                             device)
    task._sdg_prev_labels = labels
    ds = SDGDataset(source, crop_size=crop, train=True, augment=augment,
                    mode=algo, seed=seed)
    dl_kwargs = dict(batch_size=batch_size, num_workers=0, drop_last=False)
    loaders = []
    for k in range(num_clusters):
        idx = np.where(labels == k)[0].tolist()
        if not idx:
            continue
        loaders.append(torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), shuffle=True, **dl_kwargs))
    if len(loaders) < 2:
        # 聚类退化（特征高度相似 / 图像过少）：按索引均分兜底，保证元学习可运行
        task.log("  聚类退化（有效伪域 < 2），按索引均分兜底")
        idx_all = list(range(len(ds)))
        random.Random(seed).shuffle(idx_all)
        mid = max(1, len(idx_all) // 2)
        loaders = []
        for part in (idx_all[:mid], idx_all[mid:]):
            if part:
                loaders.append(torch.utils.data.DataLoader(
                    torch.utils.data.Subset(ds, part), shuffle=True,
                    **dl_kwargs))
        num_clusters = len(loaders)
    K = len(loaders)
    iters_dict = {}

    def _next(k):
        if k not in iters_dict:
            iters_dict[k] = iter(loaders[k])
        try:
            return next(iters_dict[k])
        except StopIteration:
            iters_dict[k] = iter(loaders[k])
            return next(iters_dict[k])

    local_fn = LocalPerceptionLoss(grid_size=32, lambda_local=10.0) \
        if algo == "llb" else None
    bound_fn = BoundAwareLoss(lambda_boundary=10.0, extractor="sobel") \
        if algo == "llb" else None
    w = weights or {}
    sim_w = float(w.get("sim_w", 0.001 if algo == "dccus" else 1.0))
    orth_w = float(w.get("orth_w", 1.0))
    l1_w = float(w.get("mask_w", 100.0))
    local_w = float(w.get("local_w", 10.0))
    bound_w = float(w.get("bound_w", 10.0))
    den_w = float(w.get("den_w", 1.0))
    model.train()
    total, n = 0.0, 0
    for it in range(iters):
        if task.stop_event.is_set():
            raise StopIteration()
        order = list(range(K))
        random.shuffle(order)
        batches = {k: _next(k) for k in range(K)}
        optimizer.zero_grad()
        for p in model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        for t in range(K):
            k_tr = order[t]
            k_te = order[(t + 1) % K]
            i1_tr, d1_tr, i2_tr, d2_tr = batches[k_tr]
            i1_te, d1_te, _, _ = batches[k_te]

            inner = copy.deepcopy(model)
            inner_opt = torch.optim.Adam(inner.parameters(), lr=lr,
                                         weight_decay=1e-4)
            label_tr = torch.full((i2_tr.shape[0],), k_tr,
                                  dtype=torch.long, device=device)
            out_tr, binmap, sim, sim2, orth = inner.train_forward(
                i2_tr.to(device), label_tr)
            loss_tr = _meta_loss(algo, out_tr, d2_tr.to(device), binmap,
                                 sim, sim2, orth, l1_w, sim_w, orth_w,
                                 local_fn, bound_fn, aux_loss, aux_w, w, den_w, local_w, bound_w)
            inner_opt.zero_grad()
            loss_tr.backward()
            inner_opt.step()
            for p_tgt, p_src in zip(model.parameters(), inner.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / K)

            label_te = torch.full((i1_te.shape[0],), k_te,
                                  dtype=torch.long, device=device)
            out_te, binmap_te, sim_te, sim2_te, orth_te = inner.train_forward(
                i1_te.to(device), label_te)
            loss_te = _meta_loss(algo, out_te, d1_te.to(device), binmap_te,
                                 sim_te, sim2_te, orth_te, l1_w, sim_w,
                                 orth_w, local_fn, bound_fn, aux_loss, aux_w, w, den_w, local_w, bound_w)
            grads = torch.autograd.grad(loss_te, inner.parameters(),
                                        allow_unused=True)
            for p, g in zip(model.parameters(), grads):
                if g is not None:
                    p.grad.data.add_(g.data / K)
        optimizer.step()
        total += float(loss_tr.detach())
        n += 1
        if (it + 1) % max(1, iters // 5) == 0 or it + 1 == iters:
            task.log(f"  [meta {it + 1}/{iters}] loss={float(loss_tr):.4f}")
    return total / max(1, n)


def _meta_loss(algo, pred, gt, binmap, sim, sim2, orth, l1_w, sim_w,
               orth_w, local_fn, bound_fn, aux_loss=None, aux_w=1.0,
               weights=None, den_w=1.0, local_w=10.0, bound_w=10.0):
    """DCCUS / LLB 单步损失组合（与各自 trainer 同构）"""
    pred = pred.squeeze()
    gt = gt.squeeze()
    loss = float(den_w) * F.mse_loss(pred, gt, reduction="sum")
    gt4 = gt.unsqueeze(0).unsqueeze(0) if gt.dim() == 2 else gt
    pred4 = pred.unsqueeze(0).unsqueeze(0) if pred.dim() == 2 else pred
    if aux_loss is not None:
        loss = loss + float(aux_w) * aux_loss(pred4, gt4)
    if algo == "dccus":
        if binmap is not None:
            gt_mask = (gt > 0).float()
            bm = binmap.squeeze()
            if bm.shape != gt_mask.shape:
                bm = F.interpolate(bm.unsqueeze(0).unsqueeze(0),
                                   size=gt_mask.shape[-2:], mode="nearest"
                                   )[0, 0]
            loss = loss + F.l1_loss(bm, gt_mask) * l1_w
    else:
        loss = loss + float(local_w) * local_fn(pred4, gt4)
        loss = loss + float(bound_w) * bound_fn(pred4, gt4)
    loss = loss + float(sim_w) * (torch.sum(sim) + sim2) + \
        float(orth_w) * orth
    return loss


# ---------------------------------------------------------------------------
# 测试（仅评估泛化性能）
# ---------------------------------------------------------------------------

def run_sdg_test(task, params: dict):
    """SDG 测试：加载权重在指定测试集上评估 MAE / MSE（不训练）"""
    weight_path = params.get("weight_path")
    if not weight_path or not Path(weight_path).is_file():
        raise ValueError("测试需要已训练的权重文件（训练产物 best/last.pth）")
    device_str = params.get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    model_name = params.get("model") or "sdg_dccus"
    ck = torch.load(str(weight_path), map_location="cpu",
                    weights_only=False)
    ck_meta = ck.get("sdg_meta", {}) if isinstance(ck, dict) else {}
    model_name, backbone, num_clusters = _resolve_meta(
        model_name, params.get("sdg_backbone"),
        params.get("sdg_num_clusters"), ck_meta)
    log_para = float(ck_meta.get("log_para") or LOG_PARA)
    model = _build_model(model_name, backbone, num_clusters)
    _load_weights(model, weight_path)
    model.to(device)
    task.log(f"单域泛化测试 | 模型: {model_name} | 骨干: {backbone} | "
             f"权重: {weight_path}")
    root = params["dataset_path"]
    patch = int(params.get("sdg_test_patch") or 0)
    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = None
    png_dir = None
    if params.get("save_outputs"):
        from app.services.crowd_counting import _test_output_dirs
        pred_dir, png_dir = _test_output_dirs(params, save_dir)
        task.log(f"逐图预测密度图将保存为 .npy：{pred_dir}")
        task.log(f"逐图可视化将保存为 .png：{png_dir}")
    res = evaluate_dataset(task, model, root, device, log_para=log_para,
                           patch_size=patch,
                           csv_path=save_dir / "test_results.csv",
                           tag="test", pred_dir=pred_dir, png_dir=png_dir)
    task.progress = 100.0
    task.result = {**res, "weight_path": str(weight_path),
                   "model": model_name, "backbone": backbone,
                   "save_dir": str(save_dir)}
    task.log(f"泛化测试结果：MAE={res['mae']:.3f} | RMSE={res['mse']:.3f} | "
             f"样本 {res['num_images']} 张（逐图结果见 test_results.csv）")
    if pred_dir is not None:
        task.log(f"已保存 {res.get('saved_outputs', 0)} 张预测密度图"
                 f"（.npy）到 {pred_dir}")
        task.log(f"已保存 {res.get('saved_pngs', 0)} 张可视化结果"
                 f"（.png）到 {png_dir}")


# ---------------------------------------------------------------------------
# 推理（单图）
# ---------------------------------------------------------------------------

def sdg_infer(image: Image.Image, weight_path=None, model_name="sdg_dccus",
              backbone=None, device="cpu", pretrained=False,
              patch_size=0, conf_threshold=0.5):
    """单图推理：返回人数 + 密度热力可视化（与框架其它路线返回结构一致）"""
    import base64

    import matplotlib.cm as cm

    if not weight_path:
        raise ValueError("单域泛化推理需要训练产物权重（请选择 best/last.pth）")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    ck = torch.load(str(weight_path), map_location="cpu",
                    weights_only=False)
    ck_meta = ck.get("sdg_meta", {}) if isinstance(ck, dict) else {}
    model_name, backbone, num_clusters = _resolve_meta(
        model_name, backbone, None, ck_meta)
    log_para = float(ck_meta.get("log_para") or LOG_PARA)
    model = _build_model(model_name, backbone, num_clusters)
    _load_weights(model, weight_path)
    model.to(dev).eval()
    img = image.convert("RGB")
    W, H = img.size
    x = _image_tensor(img, dev)
    with torch.no_grad():
        if patch_size and (H > patch_size or W > patch_size):
            count = predict_count(model, x, log_para, patch_size)
            dens = _forward_density(model, x)
        else:
            dens = _forward_density(model, x)
            count = float(dens.sum().item()) / log_para
    dens = F.interpolate(dens, size=(H, W), mode="bilinear",
                         align_corners=False)[0, 0]
    d = (dens - dens.min()).clamp_min(0)
    mx = float(d.max())
    d = (d / mx) if mx > 1e-8 else d
    heat = (cm.jet(d.detach().cpu().numpy())[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat)
    overlay = Image.blend(img, heat_img, 0.5)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(overlay)
    label = f"count = {count:.1f}"
    draw.rectangle([6, 6, 6 + 8 * len(label), 26], fill=(0, 0, 0))
    draw.text((10, 9), label, fill=(255, 255, 0))

    def _b64(im):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return {"count": round(count, 2), "visualization": _b64(overlay),
            "density_map": _b64(heat_img), "model": model_name,
            "point_based": False, "vis_type": "density"}

