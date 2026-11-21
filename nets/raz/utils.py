# -*- coding: utf-8 -*-
"""Small shared utilities."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退到 CPU")
        return torch.device("cpu")
    return torch.device(name)


def save_json(path: os.PathLike | str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_json(path: os.PathLike | str):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def points_to_tensor(points: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2)


def scale_points(points: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2).astype(np.float32)
    out = np.asarray(points, dtype=np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def clip_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2).astype(np.float32)
    out = np.asarray(points, dtype=np.float32).copy()
    out[:, 0] = np.clip(out[:, 0], 0, max(0, width - 1))
    out[:, 1] = np.clip(out[:, 1], 0, max(0, height - 1))
    return out


def atomic_torch_save(obj, path: os.PathLike | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def format_count(x: float) -> str:
    return f"{x:.2f}"


def find_existing(paths) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.is_file():
            return p
    return None


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
