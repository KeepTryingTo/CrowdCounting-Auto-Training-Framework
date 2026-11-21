# -*- coding: utf-8 -*-
"""数据读取预处理服务：常用人群计数数据集 → YOLO 头部框格式转换

支持的常用数据集（自动探测标注格式）：
- jhu          JHU-CROWD++      gt/*.txt 每行 "x y w h o b"（含头部框标注）
- nwpu         NWPU-Crowd       mats/*.mat annPoints + jsons/*.json 框标注
- qnrf         UCF-QNRF         *.mat annPoints（纯点标注）
- shha         ShanghaiTech A   ground_truth/*.mat image_info（纯点标注）
- shb          ShanghaiTech B   ground_truth/*.mat image_info（纯点标注）
- ucf50        UCF_CC_50        *.mat annPoints（纯点标注）
- custom_txt   自定义点标注      GT/*.txt 每行 "x y"（纯点标注）
- custom_box   自定义框标注      Labels/*.txt 每行 "x1 y1 x2 y2"（像素坐标框）

无框标注数据集 → 头部框获取策略（三选一）：
- adaptive:    基于 k 近邻自适应推框（从头部坐标点大致推出头部坐标框；
               每点取 k 近邻平均距离 × 系数作为正方形边长）
- yolo_detect: 用已训练的 YOLO 头部框检测模型对图像推理，取检测框
- yolo_pt:     用官方 COCO 预训练 YOLO（检测 person 类），取 person 框

输出（统一 YOLO 格式，ultralytics 布局）：
    <save_dir>/
    ├── train/
    │   ├── images/   *.jpg
    │   └── labels/   *.txt 每行 "0 cx cy w h"（归一化）
    └── test/
        ├── images/
        └── labels/
    └── classes.txt   内容 "head"
"""
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 数据集预设：key → { label, tree, 说明 }
# 前端通过 /api/crowd/preprocess/datasets 获取
# ---------------------------------------------------------------------------
DATASET_PRESETS = {
    "jhu": {
        "label": "JHU-CROWD++",
        "annotation": "box",
        "tree": "jhu/\n├── train/\n│   ├── images/   *.jpg\n│   ├── gt/       *.txt (x y w h o b)\n│   └── image_labels.txt\n├── val/\n│   ├── images/\n│   ├── gt/\n│   └── image_labels.txt\n├── test/\n│   ├── images/\n│   ├── gt/\n│   └── image_labels.txt\n└── (val 自动合并到 train)",
    },
    "nwpu": {
        "label": "NWPU-Crowd",
        "annotation": "box",
        "tree": "nwpu/\n├── images/       0001.jpg ... 5109.jpg\n├── jsons/       0001.json ...（框标注）\n├── mats/        0001.mat ...（annPoints 点标注）\n├── train.txt    每行: ID 亮度 场景\n├── val.txt      每行: ID 亮度 场景\n├── test.txt     每行: ID 亮度 场景（无标注，跳过）\n└── (train+val → train, test 无标注仅保留 COCO 检测可用图像)",
    },
    "qnrf": {
        "label": "UCF-QNRF",
        "annotation": "point",
        "tree": "UCF-QNRF_ECCV18/\n├── Train/\n│   ├── img_0001.jpg\n│   └── img_0001_ann.mat   (annPoints 点标注)\n└── Test/\n    ├── img_0001.jpg\n    └── img_0001_ann.mat",
    },
    "shha": {
        "label": "ShanghaiTech Part A",
        "annotation": "point",
        "tree": "part_A/\n├── train_data/\n│   ├── images/        IMG_1.jpg\n│   └── ground_truth/  GT_IMG_1.mat (image_info 点标注)\n└── test_data/\n    ├── images/\n    └── ground_truth/",
    },
    "shb": {
        "label": "ShanghaiTech Part B",
        "annotation": "point",
        "tree": "part_B/\n├── train_data/\n│   ├── images/        IMG_1.jpg\n│   └── ground_truth/  GT_IMG_1.mat (image_info 点标注)\n└── test_data/\n    ├── images/\n    └── ground_truth/",
    },
    "ucf50": {
        "label": "UCF_CC_50",
        "annotation": "point",
        "tree": "UCF-CC-50/\n├── 1.jpg\n├── 1_ann.mat   (annPoints 点标注)\n├── 2.jpg\n├── 2_ann.mat\n└── ... (50 对图像/标注)",
    },
    "custom_txt": {
        "label": "自定义（.txt 点标注）",
        "annotation": "point",
        "tree": "my_dataset/\n├── Train/\n│   ├── Images/   *.jpg\n│   └── GT/       *.txt 每行 'x y'\n└── Test/\n    ├── Images/\n    └── GT/",
    },
    "custom_box": {
        "label": "自定义（.txt 框标注）",
        "annotation": "box",
        "tree": "my_dataset/\n├── Train/\n│   ├── Images/   *.jpg\n│   └── Labels/    *.txt 每行 'x1 y1 x2 y2'（像素坐标）\n└── Test/\n    ├── Images/\n    └── Labels/",
    },
}


# ---------------------------------------------------------------------------
# 标注读取：各数据集 → [(img_path, boxes Nx4 [x1,y1,x2,y2], points Nx2, has_box)]
# ---------------------------------------------------------------------------

def _iter_images(d, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def _read_jhu(gt_txt: Path):
    """JHU gt: x y w h o b → boxes (x1,y1,x2,y2), points (x,y)"""
    boxes, pts = [], []
    for line in gt_txt.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            x, y, w, h = map(float, parts[:4])
            boxes.append([x, y, x + w, y + h])
            pts.append([x, y])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), \
        np.array(pts, dtype=np.float32).reshape(-1, 2)


def _read_mat_points(mat_path: Path):
    """.mat 点标注（annPoints / image_info），复用 crowd_counting 的解析"""
    from app.services.crowd_counting import load_mat_points
    pts, _ = load_mat_points(mat_path)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def _read_txt_points(txt: Path):
    pts = []
    for line in txt.read_text(encoding="utf-8-sig").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                pts.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    return np.array(pts, dtype=np.float32).reshape(-1, 2)


def _read_txt_boxes(txt: Path):
    boxes = []
    for line in txt.read_text(encoding="utf-8-sig").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 4:
            try:
                x1, y1, x2, y2 = map(float, parts[:4])
                boxes.append([min(x1, x2), min(y1, y2),
                              max(x1, x2), max(y1, y2)])
            except ValueError:
                pass
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def _read_nwpu_json(json_path: Path):
    """NWPU jsons/*.json → points + boxes"""
    import json as _json
    data = _json.loads(json_path.read_text(encoding="utf-8"))
    pts = []
    for p in data.get("points", []):
        if isinstance(p, dict):
            pts.append([p.get("x", 0), p.get("y", 0)])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append([p[0], p[1]])
    boxes = []
    for b in data.get("boxes", []):
        if isinstance(b, dict):
            boxes.append([b.get("xmin", 0), b.get("ymin", 0),
                          b.get("xmax", 0), b.get("ymax", 0)])
        elif isinstance(b, (list, tuple)) and len(b) >= 4:
            boxes.append(list(b[:4]))
    return np.array(pts, dtype=np.float32).reshape(-1, 2), \
        np.array(boxes, dtype=np.float32).reshape(-1, 4)


def _child_dir(d: Path, names):
    """大小写不敏感地查找子目录（Train/train、Images/images、GT/gt…）"""
    dirs = _child_dirs(d, names)
    return dirs[0] if dirs else None


def _child_dirs(d: Path, names):
    """大小写不敏感地查找全部同名子目录（如 train 与 val 同时存在）"""
    out = []
    try:
        for p in d.iterdir():
            if p.is_dir() and p.name.lower() in names:
                out.append(p)
    except OSError:
        pass
    return out


def _unified_layout(root: Path):
    """识别「上传统一结构」或「本模块 _box 产物」，返回 (kind, splits) 或 None。

    kind:
      - "yolo":    <root>/data.yaml + train|test/images + train|test/labels
                   （labels 为 YOLO 归一化 cls cx cy w h）
      - "unified": Train|Test/Images(+GT)（上传统一结构；GT 可为 .txt/.mat/.npy）
    splits: [(split_key, images_dir, annotations_dir), ...]
    """
    img_names = {"images", "img", "imgs"}
    ann_names = {"gt", "gts", "ground_truth", "labels", "annotations"}
    splits = []
    for key, names in (("train", {"train", "training", "val", "validate"}),
                       ("test", {"test", "testing"})):
        # 同一 split 可能对应多个目录（如 JHU 的 train + val 都并入 train）
        for sub in _child_dirs(root, names):
            img_d = _child_dir(sub, img_names)
            ann_d = _child_dir(sub, ann_names)
            if img_d is not None and ann_d is not None:
                splits.append((key, img_d, ann_d))
    if not splits:
        return None
    has_yaml = any((root / n).is_file() for n in
                   ("data.yaml", "data.yml", "dataset.yaml"))
    is_labels = all(ann.name.lower() == "labels" for _, _, ann in splits)
    return ("yolo" if has_yaml and is_labels else "unified"), splits


def _read_yolo_norm(gt_txt: Path, img: Path):
    """YOLO 归一化 txt（cls cx cy w h）→ 像素框 (x1,y1,x2,y2)"""
    from PIL import Image as _Image
    try:
        with _Image.open(img) as im:
            W, H = im.size
    except Exception:
        return np.zeros((0, 4), np.float32)
    boxes = []
    for line in gt_txt.read_text(encoding="utf-8-sig",
                                 errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                _, cx, cy, w, h = map(float, parts[:5])
            except ValueError:
                continue
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def _read_ann_auto(ann: Path, img: Path):
    """按扩展名/内容自动识别单文件标注 → (boxes, points, has_box) 或 None"""
    ext = ann.suffix.lower()
    if ext == ".mat":
        return None, _read_mat_points(ann), False
    if ext == ".npy":
        try:
            arr = np.load(ann)
        except Exception:
            return None
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return None, arr[:, :2].astype(np.float32), False
        return None
    if ext == ".txt":
        raw = ann.read_text(encoding="utf-8-sig", errors="ignore")
        rows = [ln.replace(",", " ").split() for ln in raw.splitlines()]
        rows = [r for r in rows if len(r) >= 2]
        if not rows:
            return None
        ncol = max(len(r) for r in rows)
        if ncol >= 6:            # JHU：x y w h o b
            boxes, pts = _read_jhu(ann)
            return boxes, pts, len(boxes) > 0
        if ncol >= 5:            # YOLO 归一化：cls cx cy w h
            try:
                vals = [float(v) for r in rows for v in r[:5]]
            except ValueError:
                vals = []
            if vals and 0 <= min(vals) and max(vals) <= 1.01:
                boxes = _read_yolo_norm(ann, img)
                return (boxes, None, True) if len(boxes) else None
        if ncol >= 4:            # 像素框：x1 y1 x2 y2
            boxes = _read_txt_boxes(ann)
            return (boxes, None, True) if len(boxes) else None
        return None, _read_txt_points(ann), False
    return None


def _parse_dataset(root: Path, preset: str):
    """按预设解析数据集 → dict: split → [(img_path, boxes, points, has_box)]"""
    items = {"train": [], "test": []}
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")

    def _add(split, img, boxes=None, points=None, has_box=False, gt=None):
        items[split].append({
            "img": img, "boxes": boxes if boxes is not None and len(boxes)
            else np.zeros((0, 4), np.float32),
            "points": points if points is not None and len(points)
            else np.zeros((0, 2), np.float32),
            "has_box": bool(has_box and boxes is not None and len(boxes)),
            # 原始标注文件路径（结构统一时可选保留原格式 .mat/.txt）
            "gt": Path(gt) if gt is not None else None,
        })

    # 优先按「上传/预处理产物」的实际结构解析（与所选预设无关）：
    # 数据集在上传时已被统一为 Train/Test × Images/GT，直接用预设的
    # 原始目录结构（part_A/train_data/ground_truth 等）会找不到样本
    layout = _unified_layout(root)
    if layout is not None:
        kind, splits = layout
        seen = set()
        for key, img_d, ann_d in splits:
            sig = (str(img_d.resolve()), str(ann_d.resolve()))
            if sig in seen:
                continue
            seen.add(sig)
            for img in _iter_images(img_d, img_exts):
                if kind == "yolo":
                    gt = ann_d / f"{img.stem}.txt"
                    boxes = _read_yolo_norm(gt, img) if gt.is_file() else None
                    if boxes is not None and len(boxes):
                        _add(key, img, boxes, None, has_box=True, gt=gt)
                    continue
                # 统一结构：GT 与图像同主名，兼容 .txt/.mat/.npy 与
                # ShanghaiTech 原始命名 GT_<主名>.mat
                cands = [ann_d / f"{img.stem}{e}" for e in (".txt", ".mat", ".npy")]
                cands.append(ann_d / f"GT_{img.stem}.mat")
                for gt in cands:
                    if not gt.is_file():
                        continue
                    parsed = _read_ann_auto(gt, img)
                    if parsed:
                        boxes, pts, has_box = parsed
                        _add(key, img, boxes, pts, has_box=has_box, gt=gt)
                    break
        if items["train"] or items["test"]:
            return items

    if preset == "jhu":
        for split_dir in ("train", "val", "test"):
            sub = root / split_dir
            if not sub.is_dir():
                continue
            img_d = sub / "images"
            gt_d = sub / "gt"
            if not img_d.is_dir() or not gt_d.is_dir():
                continue
            split_key = "train" if split_dir in ("train", "val") else "test"
            for img in _iter_images(img_d, img_exts):
                gt = gt_d / (img.stem + ".txt")
                if gt.exists():
                    boxes, pts = _read_jhu(gt)
                    _add(split_key, img, boxes, pts, has_box=len(boxes),
                         gt=gt)

    elif preset == "nwpu":
        img_d = root / "images"
        json_d = root / "jsons"
        mat_d = root / "mats"
        # 读取 split 列表
        splits = {}
        for name, target in (("train", "train"), ("val", "train"),
                             ("test", "test")):
            f = root / f"{name}.txt"
            if f.exists():
                ids = set()
                for line in f.read_text(encoding="utf-8-sig").splitlines():
                    parts = line.split()
                    if parts:
                        ids.add(parts[0])
                splits[name] = ids
        # NWPU test 无标注 → 只有 COCO person 检测时才处理
        for img in _iter_images(img_d, img_exts) if img_d.is_dir() else []:
            stem = img.stem
            split_key = "train"
            for name, ids in splits.items():
                if stem in ids:
                    split_key = "train" if name in ("train", "val") else "test"
                    break
            boxes, pts = np.zeros((0, 4)), np.zeros((0, 2))
            jf = json_d / (stem + ".json")
            mf = mat_d / (stem + ".mat")
            gt_src = None
            if jf.exists():
                pts, boxes = _read_nwpu_json(jf)
                gt_src = jf
            elif mf.exists():
                pts = _read_mat_points(mf)
                gt_src = mf
            _add(split_key, img, boxes, pts,
                 has_box=len(boxes) if len(boxes) else False, gt=gt_src)

    elif preset == "qnrf":
        for split_dir, key in (("Train", "train"), ("Test", "test")):
            sub = root / split_dir
            if not sub.is_dir():
                continue
            for img in _iter_images(sub, img_exts):
                mat = sub / (img.stem + "_ann.mat")
                if mat.exists():
                    pts = _read_mat_points(mat)
                    _add(key, img, None, pts, has_box=False, gt=mat)

    elif preset in ("shha", "shb"):
        part = "part_A" if preset == "shha" else "part_B"
        for split_dir, key in (("train_data", "train"),
                               ("test_data", "test")):
            sub = root / part / split_dir
            if not sub.is_dir():
                sub = root / split_dir  # 兼容直接传 part_A 目录
            if not sub.is_dir():
                continue
            img_d = sub / "images"
            gt_d = sub / "ground_truth"
            if not img_d.is_dir() or not gt_d.is_dir():
                continue
            for img in _iter_images(img_d, img_exts):
                mat = gt_d / f"GT_{img.stem}.mat"
                if mat.exists():
                    pts = _read_mat_points(mat)
                    _add(key, img, None, pts, has_box=False, gt=mat)

    elif preset == "ucf50":
        for img in _iter_images(root, img_exts):
            mat = root / (img.stem + "_ann.mat")
            if mat.exists():
                pts = _read_mat_points(mat)
                # UCF_CC_50 无 train/test 划分，80/20 自动分
                key = "train" if len(items["train"]) >= len(items["test"]) \
                    else "test"
                _add(key, img, None, pts, has_box=False, gt=mat)

    elif preset == "custom_txt":
        _seen = set()
        for split_dir, key in (("Train", "train"), ("Test", "test")):
            sub = root / split_dir
            if not sub.is_dir() or str(sub.resolve()) in _seen:
                continue
            _seen.add(str(sub.resolve()))
            img_d = next((sub / n for n in ("Images", "images", "imgs")
                          if (sub / n).is_dir()), None)
            gt_d = next((sub / n for n in ("GT", "gt", "labels")
                         if (sub / n).is_dir()), None)
            if img_d is None:
                continue
            for img in _iter_images(img_d, img_exts):
                if gt_d:
                    # 统一格式的 GT 同时支持 .txt（点）/ .mat（点）/
                    # .npy（(N,2) 点坐标；密度图不适用框预处理）
                    hit, pts = None, None
                    txt = gt_d / (img.stem + ".txt")
                    mat = gt_d / (img.stem + ".mat")
                    npy = gt_d / (img.stem + ".npy")
                    if txt.exists():
                        pts, hit = _read_txt_points(txt), txt
                    elif mat.exists():
                        pts, hit = _read_mat_points(mat), mat
                    elif npy.exists():
                        try:
                            arr = np.load(npy)
                            if arr.ndim == 2 and arr.shape[1] >= 2:
                                pts, hit = arr[:, :2].astype(np.float32), npy
                        except Exception:
                            hit = None
                    if hit is not None:
                        _add(key, img, None, pts, has_box=False, gt=hit)

    elif preset == "custom_box":
        _seen = set()
        for split_dir, key in (("Train", "train"), ("Test", "test")):
            sub = root / split_dir
            if not sub.is_dir() or str(sub.resolve()) in _seen:
                continue
            _seen.add(str(sub.resolve()))
            img_d = next((sub / n for n in ("Images", "images", "imgs")
                          if (sub / n).is_dir()), None)
            lbl_d = next((sub / n for n in ("Labels", "labels", "GT")
                          if (sub / n).is_dir()), None)
            if img_d is None:
                continue
            for img in _iter_images(img_d, img_exts):
                if lbl_d:
                    txt = lbl_d / (img.stem + ".txt")
                    if txt.exists():
                        boxes = _read_txt_boxes(txt)
                        _add(key, img, boxes, None, has_box=len(boxes),
                             gt=txt)

    return items


# ---------------------------------------------------------------------------
# 点 → 伪框（k 近邻自适应）
# ---------------------------------------------------------------------------

def points_to_pseudo_boxes(points, k=4, scale=0.4, default_size=20.0):
    """头部坐标点 → 头部伪框。

    每点取 k 近邻平均距离 d，正方形边长 = max(8, d * scale * 2)，
    以点为中心生成 (x1,y1,x2,y2)。点数不足 k 时用 default_size。
    """
    if len(points) == 0:
        return np.zeros((0, 4), np.float32)
    pts = np.asarray(points, np.float32)
    boxes = np.zeros((len(pts), 4), np.float32)
    if len(pts) <= k:
        sizes = np.full(len(pts), default_size, np.float32)
    else:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=min(k + 1, len(pts)))
        # dists[:, 0] 是自身距离 0，取 1:k+1 的均值
        nn = dists[:, 1:] if dists.shape[1] > 1 else dists[:, :1]
        sizes = np.maximum(8.0, nn.mean(axis=1) * scale * 2).astype(np.float32)
    half = sizes / 2
    boxes[:, 0] = pts[:, 0] - half
    boxes[:, 1] = pts[:, 1] - half
    boxes[:, 2] = pts[:, 0] + half
    boxes[:, 3] = pts[:, 1] + half
    return boxes


# ---------------------------------------------------------------------------
# YOLO 检测生成框
# ---------------------------------------------------------------------------

def _yolo_detect_boxes(img_path: Path, model, conf=0.25):
    """用 YOLO 模型对单张图检测，返回 person/head 框"""
    try:
        import numpy as _np
        img = Image.open(img_path).convert("RGB")
        bgr = _np.asarray(img)[:, :, ::-1].copy()
        results = model.predict(source=bgr, conf=conf, verbose=False)
        r = results[0] if results else None
        if r is None or r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4), np.float32)
        return r.boxes.xyxy.cpu().numpy().astype(np.float32)
    except Exception:
        return np.zeros((0, 4), np.float32)


# ---------------------------------------------------------------------------
# YOLO 格式输出
# ---------------------------------------------------------------------------

def _boxes_to_yolo_line(boxes, W, H, cls_id=0):
    """(x1,y1,x2,y2) 像素坐标 → YOLO 归一化 cx cy w h"""
    lines = []
    for b in np.asarray(boxes, np.float32).reshape(-1, 4):
        x1, y1, x2, y2 = b
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


# ---------------------------------------------------------------------------
# 主任务（后台运行）
# ---------------------------------------------------------------------------

def run_preprocess(task, params: dict):
    """数据读取预处理后台任务。

    params:
      dataset_path: 数据集根目录（服务器路径或已上传数据集名）
      preset:       数据集预设 key（jhu/nwpu/qnrf/shha/shb/ucf50/custom_txt/custom_box）
      box_source:   无框标注时的策略 "adaptive" | "yolo_detect" | "skip"
      yolo_weight:  yolo_detect 时的 YOLO .pt 权重路径
      yolo_conf:    yolo_detect 置信度阈值
      adaptive_k:   自适应 k 近邻数
      adaptive_scale: 自适应缩放系数
      adaptive_default: 点数不足时默认框边长
      save_dir:     输出目录（留空 → runs/preprocess/<dataset_name>_yolo/）
    """
    from app.services.train_common import set_seed
    dataset_path = params["dataset_path"]
    preset = params.get("preset", "custom_txt")
    box_source = params.get("box_source", "adaptive")
    save_dir = Path(params["save_dir"])

    set_seed(int(params.get("seed", 42)))
    save_dir.mkdir(parents=True, exist_ok=True)
    # 覆盖式输出：清理上一次生成的 train/test 与 data.yaml，避免残留旧样本
    for _name in ("train", "test"):
        _d = save_dir / _name
        if _d.exists():
            shutil.rmtree(_d)
    (save_dir / "data.yaml").unlink(missing_ok=True)
    task.log(f"数据读取预处理 | 数据集: {dataset_path} | 预设: {preset}")
    task.log(f"框来源: {box_source} | 输出: {save_dir}")

    # ---- 1) 解析数据集 ----
    root = Path(dataset_path)
    if not root.is_dir():
        raise ValueError(f"数据集目录不存在: {dataset_path}")
    task.log("正在解析数据集标注...")
    items = _parse_dataset(root, preset)
    n_train = len(items["train"])
    n_test = len(items["test"])
    n_total = n_train + n_test
    if n_total == 0:
        raise ValueError(f"未找到有效样本。请检查数据集格式是否匹配预设 {preset}。")
    n_has_box = sum(1 for s in items.values() for it in s if it["has_box"])
    n_no_box = n_total - n_has_box
    task.log(f"样本: train={n_train} test={n_test} | "
             f"自带 GT 框标注: {n_has_box}（直接读取框并转换，"
             f"不生成、不跳过） | 缺少框标注: {n_no_box}")
    if n_no_box:
        task.log(f"缺少框标注的 {n_no_box} 个样本按「{box_source}」策略处理"
                 + ("（策略为 skip：这些样本不写入输出）"
                    if box_source == "skip" else ""))

    # ---- 2) 加载 YOLO 检测模型（box_source=yolo_detect 时） ----
    yolo_model = None
    if box_source == "yolo_detect":
        from app.services import yolo_service
        yolo_service._require_ultralytics()
        weight = params.get("yolo_weight")
        if not weight:
            raise ValueError("yolo_detect 模式需要提供 YOLO 权重路径")
        yolo_model = yolo_service._model_class_for("yolov8n")(weight)
        task.log(f"YOLO 检测模型已加载: {weight}")

    # ---- 3) 处理并写出 ----
    out_train_img = save_dir / "train" / "images"
    out_train_lbl = save_dir / "train" / "labels"
    out_test_img = save_dir / "test" / "images"
    out_test_lbl = save_dir / "test" / "labels"
    for d in (out_train_img, out_train_lbl, out_test_img, out_test_lbl):
        d.mkdir(parents=True, exist_ok=True)

    # classes.txt（单类 head）
    (save_dir / "classes.txt").write_text("head\n", encoding="utf-8")

    stats = {"processed": 0, "skipped": 0, "failed": 0,
             "boxes_generated": 0, "boxes_from_gt": 0, "yolo_lines": 0}
    used_stems = {"train": set(), "test": set()}
    written = {"train": 0, "test": 0}
    idx = 0
    for split, split_items in items.items():
        out_img = out_train_img if split == "train" else out_test_img
        out_lbl = out_train_lbl if split == "train" else out_test_lbl
        for item in split_items:
            idx += 1
            task.epoch = idx
            task.progress = idx / n_total * 100.0
            if task.stop_event.is_set():
                raise StopIteration()
            img_path = item["img"]
            try:
                with Image.open(img_path) as im:
                    W, H = im.size

                # 获取框
                if item["has_box"]:
                    boxes = item["boxes"]
                    stats["boxes_from_gt"] += len(boxes)
                elif box_source == "adaptive" and len(item["points"]):
                    boxes = points_to_pseudo_boxes(
                        item["points"],
                        k=int(params.get("adaptive_k", 4)),
                        scale=float(params.get("adaptive_scale", 0.4)),
                        default_size=float(params.get("adaptive_default", 20)))
                    stats["boxes_generated"] += len(boxes)
                elif box_source == "yolo_detect" and yolo_model is not None:
                    boxes = _yolo_detect_boxes(
                        img_path, yolo_model,
                        conf=float(params.get("yolo_conf", 0.25)))
                    stats["boxes_generated"] += len(boxes)
                else:
                    stats["skipped"] += 1
                    continue

                if len(boxes) == 0:
                    stats["skipped"] += 1
                    continue

                # 复制/链接图像：同一 split 内同名文件自动加 _1/_2 后缀，
                # 避免不同来源目录的同名图片/标签互相覆盖
                used = used_stems[split]
                stem = img_path.stem
                if stem in used:
                    i = 1
                    while f"{stem}_{i}" in used:
                        i += 1
                    stem = f"{stem}_{i}"
                used.add(stem)
                dst_img = out_img / f"{stem}{img_path.suffix}"
                if not dst_img.exists():
                    try:
                        os.link(str(img_path), str(dst_img))
                    except OSError:
                        shutil.copy2(str(img_path), str(dst_img))

                # 写 YOLO 标签
                lines = _boxes_to_yolo_line(boxes, W, H, cls_id=0)
                (out_lbl / (stem + ".txt")).write_text(
                    "\n".join(lines) + "\n" if lines else "",
                    encoding="utf-8")
                stats["yolo_lines"] += len(lines)
                stats["processed"] += 1
                written[split] += 1

                if idx % 20 == 0 or idx == n_total:
                    task.log(f"[{idx}/{n_total}] {img_path.name} "
                             f"→ {len(lines)} 框")
            except Exception as e:
                stats["failed"] += 1
                task.log(f"[{idx}/{n_total}] 失败 {img_path.name}: {e}")

    # ---- 4) 生成 data.yaml（兼容 ultralytics 直接训练） ----
    # test 为空（或 test 全部跳过）时把 val 回退到 train，避免
    # YOLO 训练启动即报“验证集无图像/标签”
    if written["train"] == 0:
        raise ValueError("没有生成任何训练样本，请检查数据集/预设/框来源设置")
    val_rel = "test/images" if written["test"] else "train/images"
    if not written["test"]:
        task.log("未生成 test 样本，data.yaml 的 val 回退为 train/images")
    data_yaml = save_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {save_dir.resolve().as_posix()}\n"
        f"train: train/images\n"
        f"val: {val_rel}\n"
        f"nc: 1\n"
        f"names:\n  0: head\n", encoding="utf-8")

    task.result = {
        "save_dir": str(save_dir),
        "total": n_total, "train": n_train, "test": n_test,
        "processed": stats["processed"], "skipped": stats["skipped"],
        "failed": stats["failed"],
        "boxes_from_gt": stats["boxes_from_gt"],
        "boxes_generated": stats["boxes_generated"],
        "yolo_lines": stats["yolo_lines"],
        "data_yaml": str(data_yaml),
    }
    task.log(f"预处理完成: 处理 {stats['processed']} "
             + (f"跳过 {stats['skipped']}（缺少框标注且策略=skip）"
                if stats["skipped"] else "跳过 0")
             + f" 失败 {stats['failed']} | GT框 {stats['boxes_from_gt']} "
             f"生成框 {stats['boxes_generated']} | "
             f"YOLO 标签行 {stats['yolo_lines']}")
    task.log(f"输出目录: {save_dir}")
    task.log(f"data.yaml: {data_yaml}")
