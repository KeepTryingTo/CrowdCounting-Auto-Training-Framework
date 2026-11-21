# -*- coding: utf-8 -*-
"""数据集目录命名：同名数据集不覆盖，追加序号后缀区分。

上传 / 下载数据集时，若目标目录已存在，则依次尝试
``<name>_1``、``<name>_2``……，并用 ``mkdir(exist_ok=False)`` 原子占位，
避免并发上传同名数据集时互相覆盖或写入同一个目录。
"""
import re
from pathlib import Path

# Windows 文件名非法字符
_ILLEGAL = re.compile(r'[\\/:*?"<>|]')


def unique_dataset_name(base_dir, name):
    """在 base_dir 下创建并返回唯一的数据集目录。

    参数：
        base_dir: 数据集根目录（如 uploads/datasets）
        name:     期望的数据集名（不含路径）
    返回：
        已创建的空目录 Path；同名已存在时返回 ``name_1`` / ``name_2`` …
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    stem = _ILLEGAL.sub("_", str(name or "")).strip().strip(".") or "dataset"
    stem = stem[:120]
    for i in range(0, 100000):
        cand = base / (stem if i == 0 else f"{stem}_{i}")
        try:
            cand.mkdir(parents=False, exist_ok=False)
            return cand
        except FileExistsError:
            continue
    raise RuntimeError(f"无法为数据集分配唯一目录（尝试过多）: {stem}")
