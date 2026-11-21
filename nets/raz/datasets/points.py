# -*- coding: utf-8 -*-
"""Point-annotation readers and image/GT pairing.

Supported point formats:
  - .txt : first two numeric columns per line (``x y``; extra columns ignored)
  - .mat : ``annPoints`` (QNRF/UCF-CC-50) or ShanghaiTech ``image_info``
  - .npy : (N, 2) coordinates; optionally a 2-D density map can be converted
           to pseudo points with local-maximum extraction
  - .json: nested ``points`` / ``annotations`` / ``image_info`` structures
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from scipy.io import loadmat
    from scipy.io.matlab import mat_struct as _mat_struct
except Exception:  # pragma: no cover - scipy is a hard dependency in practice
    loadmat = None
    _mat_struct = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GT_EXTS = (".mat", ".txt", ".npy", ".json")

_GT_PREFIXES = ("gt_", "gt-", "groundtruth_", "ground_truth_",
                "ground-truth_", "label_", "labels_", "label-",
                "ann_", "annotation_", "annotations_")
_GT_SUFFIXES = ("_ann", "_gt", "_label", "_labels", "_annotation",
                "_annotations", "_groundtruth", "_points")


def iter_images(root: Path, recursive: bool = True) -> List[Path]:
    root = Path(root)
    it = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _strip_gt_marks(stem: str) -> str:
    """Remove common GT prefix/suffix decorations from a file stem."""
    s = stem
    changed = True
    while changed and s:
        changed = False
        low = s.lower()
        for p in _GT_PREFIXES:
            if low.startswith(p) and len(s) > len(p):
                s = s[len(p):]
                changed = True
                break
        low = s.lower()
        for x in _GT_SUFFIXES:
            if low.endswith(x) and len(s) > len(x):
                s = s[: -len(x)]
                changed = True
                break
    return s


def _orientation_fix(points: np.ndarray, image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(0, 2)
    if pts.ndim != 2:
        raise ValueError(f"点标注数组形状不合法: {pts.shape}")
    # 统一成 (N,2)；兼容 (2,N) 转置格式和带距离的 (N,3)/(N,4)
    if pts.shape[0] == 2 and pts.shape[1] > 2:
        pts = pts.T[:, :2]
    elif pts.shape[1] > 2:
        pts = pts[:, :2]
    elif pts.shape[1] < 2:
        if pts.shape[0] == 2:
            pts = pts.T
        else:
            raise ValueError(f"点标注数组形状不合法: {pts.shape}")
    if image_size is not None:
        w, h = float(image_size[0]), float(image_size[1])
        if w > 0 and h > 0:
            # 部分上海科技 .mat 实际存的是 (y,x)：只有当交换后
            # 落在图像范围内的点明显更多时才交换，避免误伤正常标注。
            in_xy = ((pts[:, 0] >= -1) & (pts[:, 0] <= w + 1) &
                     (pts[:, 1] >= -1) & (pts[:, 1] <= h + 1)).sum()
            in_yx = ((pts[:, 1] >= -1) & (pts[:, 1] <= w + 1) &
                     (pts[:, 0] >= -1) & (pts[:, 0] <= h + 1)).sum()
            if in_yx > in_xy:
                pts = pts[:, ::-1].copy()
            # 少数标注是 0~1 归一化坐标，直接按图像宽高还原
            if pts[:, 0].max(initial=0) <= 1.5 and pts[:, 1].max(initial=0) <= 1.5:
                pts[:, 0] *= w
                pts[:, 1] *= h
    return np.ascontiguousarray(pts, dtype=np.float32)


def _array_candidates(obj, depth: int = 0, out: Optional[list] = None) -> list:
    if out is None:
        out = []
    if depth > 8:
        return out
    # image_info 常见 MATLAB cell/struct 嵌套，递归收集所有 Nx2 数值数组，
    # 最后再由 _choose_candidate 选“行数最多/最像点集”的候选。
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for x in obj.ravel():
                _array_candidates(x, depth + 1, out)
            return out
        if obj.dtype.names:
            for name in obj.dtype.names:
                try:
                    _array_candidates(obj[name], depth + 1, out)
                except Exception:
                    pass
            return out
        if np.issubdtype(obj.dtype, np.number):
            arr = np.asarray(obj)
            if arr.ndim == 2 and min(arr.shape) >= 2:
                out.append(arr)
        return out
    if _mat_struct is not None and isinstance(obj, _mat_struct):
        for name in getattr(obj, _fieldnames, []):
            try:
                _array_candidates(getattr(obj, name), depth + 1, out)
            except Exception:
                pass
        return out
    if isinstance(obj, np.void):
        if obj.dtype.names:
            for name in obj.dtype.names:
                try:
                    _array_candidates(obj[name], depth + 1, out)
                except Exception:
                    pass
    elif isinstance(obj, dict):
        for v in obj.values():
            _array_candidates(v, depth + 1, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _array_candidates(v, depth + 1, out)
    return out


def _choose_candidate(candidates: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    best = None
    best_score = -1
    for arr in candidates:
        arr = np.asarray(arr)
        if arr.ndim != 2 or min(arr.shape) < 2:
            continue
        if arr.shape[1] == 2:
            score = arr.shape[0]
        elif arr.shape[0] == 2:
            score = arr.shape[1] - 0.5
        else:
            score = max(arr.shape) * 0.1
        if score > best_score:
            best_score = score
            best = arr
    return best


def load_mat_points(path: Path, image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if loadmat is None:
        raise ImportError("读取 .mat 需要 scipy: pip install scipy")
    # 使用 scipy 默认参数（struct_as_record=True）：ShanghaiTech 的
    # image_info.location 会以带字段名的结构体返回，便于统一提取
    data = loadmat(str(path))
    if "annPoints" in data:
        arr = np.asarray(data["annPoints"])
        return _orientation_fix(arr, image_size)
    candidates = []
    for k, v in data.items():
        if k.startswith("__"):
            continue
        _array_candidates(v, 0, candidates)
    # Prefer image_info-like candidates with a meaningful number of points.
    arr = _choose_candidate(candidates)
    if arr is None:
        keys = [k for k in data if not k.startswith("__")]
        raise ValueError(f"不支持的 .mat 格式（键: {keys}，未找到 Nx2 点数组）: {path}")
    return _orientation_fix(arr, image_size)


def load_txt_points(path: Path, image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    pts = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").replace(";", " ").split()
        if len(parts) < 2:
            continue
        try:
            pts.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    return _orientation_fix(np.asarray(pts, dtype=np.float32), image_size)


def _json_points(node, depth: int = 0) -> list:
    if depth > 8:
        return []
    if isinstance(node, dict):
        # Direct point object.
        if "x" in node and "y" in node:
            try:
                return [[float(node["x"]), float(node["y"])]]
            except (TypeError, ValueError):
                pass
        out = []
        for key in ("points", "annotations", "image_info", "gt", "labels"):
            if key in node:
                out.extend(_json_points(node[key], depth + 1))
        if out:
            return out
        for v in node.values():
            out.extend(_json_points(v, depth + 1))
        return out
    if isinstance(node, (list, tuple)):
        if len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]):
            return [[float(node[0]), float(node[1])]]
        out = []
        for v in node:
            out.extend(_json_points(v, depth + 1))
        return out
    return []


def load_json_points(path: Path, image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    pts = np.asarray(_json_points(data), dtype=np.float32)
    return _orientation_fix(pts, image_size)


def points_from_density(density: np.ndarray, threshold: float = 0.05,
                        min_distance: int = 3) -> np.ndarray:
    """Fallback for density-only GT: extract local maxima as pseudo points."""
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("密度图转点需要 opencv-python") from exc
    arr = np.asarray(density, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.size == 0:
        return np.zeros((0, 2), np.float32)
    k = max(1, int(min_distance) * 2 + 1)
    dil = cv2.dilate(arr, np.ones((k, k), np.uint8))
    mask = (arr >= dil) & (arr >= float(threshold))
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def load_npy_points(path: Path, image_size: Optional[Tuple[int, int]] = None,
                    allow_density: bool = False,
                    density_threshold: float = 0.05) -> np.ndarray:
    arr = np.load(str(path), allow_pickle=False)
        # 先按坐标数组处理：Nx2/Nx3/Nx4、转置 2xN、单点 (2,)；
        # 若 shape 像密度图（如 512x512）则不能误判为点集。
    if arr.ndim == 1 and arr.size in (2, 3, 4):
        arr = arr[:2].reshape(1, 2)
    if arr.ndim == 2 and (
            arr.shape[1] in (2, 3, 4) or arr.shape[0] == 2):
        return _orientation_fix(arr, image_size)
    if allow_density and arr.ndim in (2, 3):
        return _orientation_fix(
            points_from_density(arr, threshold=density_threshold), image_size)
    raise ValueError(
        f".npy 不是点坐标数组（shape={arr.shape}）。RAZ 需要 (N,2) 坐标；"
        f"若为密度图，请加 --allow-density-gt 或提供 .txt/.mat 点标注: {path}")


def load_points(path: Path, image_size: Optional[Tuple[int, int]] = None,
                allow_density: bool = False,
                density_threshold: float = 0.05) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_txt_points(path, image_size)
    if suffix == ".mat":
        return load_mat_points(path, image_size)
    if suffix == ".npy":
        return load_npy_points(path, image_size, allow_density,
                               density_threshold)
    if suffix == ".json":
        return load_json_points(path, image_size)
    raise ValueError(f"不支持的点标注格式: {path}")


def _candidate_stems(stem: str) -> List[str]:
    stems = [stem, _strip_gt_marks(stem)]
    # GT_IMG_1 -> IMG_1; img_0001 -> img_0001_ann
    for s in list(stems):
        stems += [f"GT_{s}", f"gt_{s}", f"{s}_ann", f"{s}_gt",
                  f"ground_truth_{s}", f"ground-truth_{s}"]
    seen = set()
    out = []
    for s in stems:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def find_gt_for_image(image_path: Path,
                      gt_dirs: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """Find a same-stem GT file in the supplied directories / nearby dirs."""
    image_path = Path(image_path)
    dirs: List[Path] = []
    if gt_dirs is None:
        gt_dirs = []
    for d in gt_dirs:
        if d is not None:
            dirs.append(Path(d))
    # 先搜显式 GT 目录，再搜常见的同级/上级 gt、annotations、mats 等目录
    parent = image_path.parent
    for name in ("gt", "GT", "ground_truth", "ground-truth",
                 "annotations", "annotation", "mats", "jsons",
                 "labels", "npys"):
        dirs.append(parent / name)
        dirs.append(parent.parent / name)
    dirs.append(parent)
    seen_dirs = set()
    for d in dirs:
        try:
            key = str(d.resolve())
        except OSError:
            key = str(d)
        if key in seen_dirs or not d.is_dir():
            continue
        seen_dirs.add(key)
        for stem in _candidate_stems(image_path.stem):
            for ext in GT_EXTS:
                candidate = d / f"{stem}{ext}"
                if candidate.is_file():
                    return candidate
    return None


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size
