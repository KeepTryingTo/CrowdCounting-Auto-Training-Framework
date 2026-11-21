# -*- coding: utf-8 -*-
"""Dataset layout auto-detection / adapters.

The training code only needs lists of ``(image_path, gt_path)`` items.  This
module supports the common crowd-counting layouts and provides a generic
fallback for custom point-annotated datasets.

Supported presets
-----------------
- ShanghaiTech A/B: part_A/part_B -> train_data/test_data -> images/ground_truth
- UCF-QNRF: Train/Test -> images + *_ann.mat
- UCF-CC-50: images + *_ann.mat (deterministic split)
- NWPU-Crowd: images + jsons/mats + train.txt/val.txt/test.txt
- JHU-Crowd++: train/val/test -> images + gt/*.txt
- Unified custom: Train/Test -> Images + GT (.txt/.mat/.npy/.json)
- Generic custom: images + gt/annotations/ground_truth/mats/jsons, or a manifest
"""
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .points import IMAGE_EXTS, find_gt_for_image, iter_images


@dataclass
class DatasetSpec:
    name: str
    train: List[Tuple[Path, Path]] = field(default_factory=list)
    val: List[Tuple[Path, Path]] = field(default_factory=list)
    test: List[Tuple[Path, Path]] = field(default_factory=list)

    def summary(self) -> Dict[str, int]:
        return {"train": len(self.train), "val": len(self.val),
                "test": len(self.test)}


def _find_ci(base: Path, names: Sequence[str]) -> Optional[Path]:
    if not base.is_dir():
        return None
    lookup = {p.name.lower(): p for p in base.iterdir() if p.is_dir()}
    for n in names:
        p = lookup.get(n.lower())
        if p is not None:
            return p
    return None


def _first_dir(base: Path, names: Sequence[str]) -> Optional[Path]:
    for n in names:
        p = base / n
        if p.is_dir():
            return p
    return None


def _pair(img_dir: Path, gt_dirs: Sequence[Path],
          recursive: bool = False) -> List[Tuple[Path, Path]]:
    items: List[Tuple[Path, Path]] = []
    seen = set()
    for img in iter_images(img_dir, recursive=recursive):
        gt = find_gt_for_image(img, gt_dirs)
        if gt is None:
            continue
        key = (str(img.resolve()), str(gt.resolve()))
        if key not in seen:
            seen.add(key)
            items.append((img, gt))
    return items


def _split_items(items: List[Tuple[Path, Path]], val_ratio: float,
                 seed: int) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    items = list(items)
    random.Random(seed).shuffle(items)
    if len(items) <= 1 or val_ratio <= 0:
        return items, []
    n_val = max(1, int(round(len(items) * val_ratio)))
    n_val = min(n_val, len(items) - 1)
    return items[n_val:], items[:n_val]


def _read_manifest(path: Path) -> List[Tuple[Path, Path]]:
    path = Path(path)
    base = path.parent
    rows: List[Tuple[Path, Path]] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("items") or data.get("data") or data.get("images") or []
        for it in data:
            if isinstance(it, dict):
                img = it.get("image") or it.get("img") or it.get("path")
                gt = it.get("gt") or it.get("label") or it.get("annotation")
            else:
                img, gt = it[0], it[1]
            if img and gt:
                rows.append((base / str(img), base / str(gt)))
    else:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for row in reader:
                    img = row.get("image") or row.get("img") or row.get("path")
                    gt = row.get("gt") or row.get("label") or row.get("annotation")
                    if img and gt:
                        rows.append((base / img, base / gt))
            else:
                f.seek(0)
                for line in f:
                    parts = [x.strip() for x in line.replace(",", " ").split()]
                    if len(parts) >= 2:
                        rows.append((base / parts[0], base / parts[1]))
    return [(p, q) for p, q in rows if p.is_file() and q.is_file()]


def _unified(root: Path) -> Optional[DatasetSpec]:
    """Train/Test + Images/GT (case-insensitive) layout.

    统一格式优先级最高：自定义数据集只要按 Train/Test + Images/GT 组织，
    就会直接走这里，不再尝试其他预设。
    """
    train = _find_ci(root, ["Train", "train"])
    test = _find_ci(root, ["Test", "test"])
    val = _find_ci(root, ["Val", "val", "Validation", "validation"])
    if train is None or test is None:
        return None
    t_img = _find_ci(train, ["Images", "images", "imgs", "img"])
    t_gt = _find_ci(train, ["GT", "gt", "npys", "masks", "labels"])
    s_img = _find_ci(test, ["Images", "images", "imgs", "img"])
    s_gt = _find_ci(test, ["GT", "gt", "npys", "masks", "labels"])
    if not all([t_img, t_gt, s_img, s_gt]):
        return None
    train_items = _pair(t_img, [t_gt])
    test_items = _pair(s_img, [s_gt])
    if not train_items:
        return None
    val_items = []
    if val is not None:
        v_img = _find_ci(val, ["Images", "images", "imgs", "img"])
        v_gt = _find_ci(val, ["GT", "gt", "npys", "masks", "labels"])
        if v_img and v_gt:
            val_items = _pair(v_img, [v_gt])
    return DatasetSpec("unified", train_items, val_items, test_items)


def _shanghaitech(root: Path, part: Optional[str], val_ratio: float,
                  seed: int) -> Optional[DatasetSpec]:
    # 既支持传 ShanghaiTech 根目录（内含 part_A / part_B / part_A_final /
    # part_B_final），也支持直接传 part_* 目录；part 名称大小写与是否带
    # _final 后缀均可（part_A / part_a / part_A_final 都识别）。
    part_pat = re.compile(r"^part_[ab](_final)?$", re.I)
    part_root = root if part_pat.match(root.name) else None
    if part_root is not None:
        root = root.parent
    # 收集候选 part 目录（按 part 参数精确/模糊匹配，缺省取全部）
    dirs = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    parts: List[Path] = []
    if part:
        for n in ([part] if isinstance(part, str) else part):
            key = str(n).strip().lower()
            for cand in (key, key + "_final", key.replace("_final", "")):
                p = dirs.get(cand)
                if p is not None and p not in parts:
                    parts.append(p)
    if part_root is not None:
        parts = [part_root] + [p for p in parts if p != part_root]
        parts = parts[:1]          # 已明确到某个 part_* 目录
    if not parts:
        parts = [p for p in root.iterdir()
                 if p.is_dir() and part_pat.match(p.name)]
    if not parts:
        return None
    part_root = part_root or parts[0]
    tr = _find_ci(part_root, ["train_data", "train"])
    te = _find_ci(part_root, ["test_data", "test"])
    if tr is None or te is None:
        return None
    tr_img = _find_ci(tr, ["images", "imgs"])
    tr_gt = _find_ci(tr, ["ground_truth", "ground-truth", "gt", "GT"])
    te_img = _find_ci(te, ["images", "imgs"])
    te_gt = _find_ci(te, ["ground_truth", "ground-truth", "gt", "GT"])
    if not all([tr_img, tr_gt, te_img, te_gt]):
        return None
    train_items = _pair(tr_img, [tr_gt])
    test_items = _pair(te_img, [te_gt])
    train, val = _split_items(train_items, val_ratio, seed)
    return DatasetSpec("shanghaitech", train, val, test_items)


def _qnrf(root: Path, val_ratio: float, seed: int) -> Optional[DatasetSpec]:
    tr = _find_ci(root, ["Train", "train"])
    te = _find_ci(root, ["Test", "test"])
    if tr is None or te is None:
        return None
    tr_img = _first_dir(tr, ["images", "Images", "imgs"])
    te_img = _first_dir(te, ["images", "Images", "imgs"])
    # QNRF 常见两种：Train/images + Train/annotations，
    # 或图像与 *_ann.mat 直接混放在 Train/Test 下；两种都覆盖。
    tr_items = _pair(tr_img if tr_img else tr, [tr, tr / "annotations"])
    te_items = _pair(te_img if te_img else te, [te, te / "annotations"])
    if not tr_items:
        return None
    train, val = _split_items(tr_items, val_ratio, seed)
    return DatasetSpec("qnrf", train, val, te_items)


def _ucf50(root: Path, val_ratio: float, seed: int) -> Optional[DatasetSpec]:
    img_dir = _first_dir(root, ["images", "Images", "imgs"])
    if img_dir is None:
        # Root itself contains images.
        if not any(p.suffix.lower() in IMAGE_EXTS for p in iter_images(root, False)):
            return None
        img_dir = root
    items = _pair(img_dir, [root, root / "annotations", root / "mats"])
    if not items:
        return None
    # UCF-CC-50 官方没有 train/test 划分，这里用固定 seed 做确定性切分；
    # 默认至少留 20% 作为验证集。
    train, val = _split_items(items, max(val_ratio, 0.2), seed)
    return DatasetSpec("ucf_cc_50", train, val, val)


def _jhu(root: Path, val_ratio: float, seed: int) -> Optional[DatasetSpec]:
    tr = _find_ci(root, ["train"])
    va = _find_ci(root, ["val", "validation"])
    te = _find_ci(root, ["test"])
    if tr is None:
        return None
    def pair_split(split: Path) -> List[Tuple[Path, Path]]:
        img = _first_dir(split, ["images", "Images", "imgs"])
        gt = _first_dir(split, ["gt", "GT", "ground_truth", "annotations"])
        if img is None or gt is None:
            return []
        return _pair(img, [gt])
    train_items = pair_split(tr)
    val_items = pair_split(va) if va else []
    test_items = pair_split(te) if te else []
    if not train_items:
        return None
    if not val_items:
        train_items, val_items = _split_items(train_items, val_ratio, seed)
    return DatasetSpec("jhu", train_items, val_items, test_items or val_items)


def _nwpu(root: Path, val_ratio: float, seed: int) -> Optional[DatasetSpec]:
    img_dir = _first_dir(root, ["images", "Images", "imgs"])
    json_dir = _first_dir(root, ["jsons", "json", "annotations"])
    mat_dir = _first_dir(root, ["mats", "mat", "ground_truth"])
    if img_dir is None or (json_dir is None and mat_dir is None):
        return None
    id_to_img = {p.stem: p for p in iter_images(img_dir)}
    # NWPU 用 train/val/test.txt 指定图像 ID；标注可从 jsons 或 mats 读取。
    def read_list(name: str) -> List[str]:
        f = root / f"{name}.txt"
        if not f.is_file():
            return []
        ids = []
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if parts:
                # Some lists use "0001.jpg" / "0001 123"; index by stem.
                ids.append(Path(parts[0]).stem)
        return ids
    def build(ids: Sequence[str]) -> List[Tuple[Path, Path]]:
        out = []
        for i in ids:
            img = id_to_img.get(i)
            if img is None:
                continue
            gt = None
            for d in (json_dir, mat_dir):
                if d is None:
                    continue
                for ext in (".json", ".mat", ".txt", ".npy"):
                    p = d / f"{i}{ext}"
                    if p.is_file():
                        gt = p
                        break
                if gt is not None:
                    break
            if gt is not None:
                out.append((img, gt))
        return out
    train_ids = read_list("train") or read_list("trainval")
    val_ids = read_list("val")
    test_ids = read_list("test")
    train_items = build(train_ids) if train_ids else []
    val_items = build(val_ids) if val_ids else []
    test_items = build(test_ids) if test_ids else []
    if not train_items:
        items = _pair(img_dir, [d for d in (json_dir, mat_dir) if d])
        train_items, val_items = _split_items(items, val_ratio, seed)
    return DatasetSpec("nwpu", train_items, val_items, test_items or val_items)


def _generic(root: Path, val_ratio: float, seed: int,
             train_sub: str = "", test_sub: str = "") -> DatasetSpec:
    # 显式指定 train/test 子目录时，优先按子目录配对。
    if train_sub:
        tr = Path(train_sub)
        if not tr.is_absolute():
            tr = root / tr
        te = Path(test_sub) if test_sub else None
        if te is not None and not te.is_absolute():
            te = root / te
        tr_img = _first_dir(tr, ["images", "Images", "imgs"]) or tr
        train_items = _pair(tr_img, [tr, tr / "gt", tr / "GT",
                                     tr / "ground_truth", tr / "annotations"])
        test_items = []
        if te is not None:
            te_img = _first_dir(te, ["images", "Images", "imgs"]) or te
            test_items = _pair(te_img, [te, te / "gt", te / "GT",
                                        te / "ground_truth", te / "annotations"])
        if not train_items:
            raise ValueError(f"自定义训练目录未找到图像/标注配对: {tr}")
        train, val = _split_items(train_items, val_ratio, seed)
        return DatasetSpec("custom", train, val, test_items or val)

    # 通用回退：先找根目录 images + 同级 gt/annotations；找不到再递归
    # 扫描全部图像，并用同名规则在邻近目录寻找标注。
    img_dir = _first_dir(root, ["images", "Images", "imgs", "img"])
    gt_dirs = [d for d in (
        _first_dir(root, ["gt", "GT", "ground_truth", "ground-truth",
                          "annotations", "annotation", "mats", "jsons"]),
    ) if d is not None]
    if img_dir is not None:
        items = _pair(img_dir, gt_dirs or [root])
    else:
        # Recursive fallback: pair each image with a nearby same-stem GT.
        items = _pair(root, [root], recursive=True)
    if not items:
        raise ValueError(
            "未能自动识别数据集布局。请提供以下之一：\n"
            "  1) Train/Test + Images/GT 统一格式；\n"
            "  2) ShanghaiTech part_A/part_B；\n"
            "  3) QNRF Train/Test + *_ann.mat；\n"
            "  4) images/ + gt|annotations|ground_truth|mats|jsons；\n"
            "  5) --manifest 指定 image,gt 清单。\n"
            f"数据根目录: {root}")
    train, val = _split_items(items, val_ratio, seed)
    return DatasetSpec("custom", train, val, val)


def build_dataset_spec(data_root: str | Path, dataset: str = "auto",
                       part: Optional[str] = None, val_ratio: float = 0.1,
                       seed: int = 42, manifest: str = "",
                       train_sub: str = "", test_sub: str = "") -> DatasetSpec:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")

    # manifest 是最明确的自定义方式：每行 image,gt，路径相对清单文件目录。
    if manifest:
        items = _read_manifest(Path(manifest))
        if not items:
            raise ValueError(f"manifest 未读取到有效 image/gt 对: {manifest}")
        train, val = _split_items(items, val_ratio, seed)
        return DatasetSpec("manifest", train, val, val)

    # auto 的识别顺序很重要：统一格式 > ShanghaiTech > QNRF > JHU
    # > NWPU > UCF-CC-50 > 通用递归配对；避免宽松规则抢先命中。
    name = (dataset or "auto").lower()
    if name in ("auto", "custom"):
        for fn in (lambda: _unified(root),
                   lambda: _shanghaitech(root, part, val_ratio, seed),
                   lambda: _qnrf(root, val_ratio, seed),
                   lambda: _jhu(root, val_ratio, seed),
                   lambda: _nwpu(root, val_ratio, seed),
                   lambda: _ucf50(root, val_ratio, seed)):
            spec = fn()
            if spec is not None and spec.train:
                return spec
        return _generic(root, val_ratio, seed, train_sub, test_sub)

    if name in ("shanghaitech", "sha", "shha", "shhb"):
        spec = _shanghaitech(root, part or ("part_B" if name == "shhb" else None),
                             val_ratio, seed)
        if spec is None:
            raise ValueError("未识别到 ShanghaiTech part_A/part_B 布局")
        return spec
    if name in ("qnrf", "ucf-qnrf", "ucf_qnrf"):
        spec = _qnrf(root, val_ratio, seed)
        if spec is None:
            raise ValueError("未识别到 UCF-QNRF Train/Test 布局")
        return spec
    if name in ("ucf50", "ucf_cc_50", "ucf-cc-50"):
        spec = _ucf50(root, val_ratio, seed)
        if spec is None:
            raise ValueError("未识别到 UCF-CC-50 images + *_ann.mat 布局")
        return spec
    if name in ("nwpu", "nwpu-crowd"):
        spec = _nwpu(root, val_ratio, seed)
        if spec is None:
            raise ValueError("未识别到 NWPU-Crowd 布局")
        return spec
    if name in ("jhu", "jhu-crowd", "jhu_crowd"):
        spec = _jhu(root, val_ratio, seed)
        if spec is None:
            raise ValueError("未识别到 JHU-Crowd++ 布局")
        return spec
    if name in ("worldexpo", "world-expo"):
        # WorldExpo scene/frame layouts vary; use the generic recursive
        # image/GT pairing and a deterministic train/val split.
        return _generic(root, val_ratio, seed, train_sub, test_sub)
    return _generic(root, val_ratio, seed, train_sub, test_sub)
