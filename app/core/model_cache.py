# -*- coding: utf-8 -*-
"""预训练权重统一下载到项目内 "pretrained model/"

环境变量须在 torch import 前设置（那边 import 时就固化了），
因此 main.py 在 import 路由前调用 init()：

    pretrained model/
    ├── torch/         torchvision / torch.hub（hub/checkpoints/）
    ├── ultralytics/   YOLO 官方 .pt 权重
    ├── yolov13/       YOLOv13 官方 COCO 权重（iMoonLab Release）
    └── clip/          （预留）

weights/pretrained/ 只放随项目内置的离线权重
（GhostNetV2 计数骨干 / P2PNet SHTechA 检查点 / vgg16_bn）。
用户主目录的旧缓存不迁移，首次使用重新下载。
"""
import os
from pathlib import Path

# 项目根下的统一缓存根目录（Codes/pretrained model/）
PRETRAINED_DIR = Path(__file__).resolve().parents[2] / "pretrained model"


def _sub(name: str) -> Path:
    """子目录（不存在则创建）"""
    d = PRETRAINED_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def torch_dir() -> Path:
    """torchvision / torch.hub 权重目录（实际文件在 <dir>/hub/checkpoints/）"""
    return _sub("torch")


def ultralytics_dir() -> Path:
    """ultralytics 官方 .pt 权重目录（YOLO 系列）"""
    return _sub("ultralytics")


def yolov13_dir() -> Path:
    """YOLOv13 官方 COCO 预训练权重目录"""
    return _sub("yolov13")


def clip_dir() -> Path:
    """OpenAI CLIP 权重目录（预留）"""
    return _sub("clip")


# 旧下载位置（weights/pretrained/ 下，便于已下载文件继续复用，避免重复下载）
_LEGACY_ROOT = Path(__file__).resolve().parents[2] / "weights" / "pretrained"


def resolve_legacy(new_path: Path, *legacy_rel) -> Path:
    """新位置优先，旧位置有文件就复用；都没有就返回 new_path 当下载目标"""
    if new_path.is_file():
        return new_path
    legacy = _LEGACY_ROOT.joinpath(*legacy_rel)
    if legacy.is_file():
        return legacy
    return new_path


def init():
    """服务启动早期调用：设置各下载源的环境变量，统一指向 pretrained model/

    幂等；对已存在的同名环境变量执行覆盖（保证缓存位置可控）。
    """
    os.environ["TORCH_HOME"] = str(torch_dir())
    os.environ["CLIP_CACHE_DIR"] = str(clip_dir())
    print(f"[model_cache] 预训练模型统一缓存目录: {PRETRAINED_DIR}")
    # 旧缓存提示（不自动迁移，避免误搬大文件）
    for old in (Path.home() / ".cache" / "torch",
                Path.home() / ".cache" / "clip"):
        if old.exists():
            print(f"[model_cache] 检测到旧缓存目录 {old}（不会自动迁移，"
                  f"如需复用请手动移动到 {PRETRAINED_DIR} 对应子目录）")
            break


def status() -> dict:
    """当前缓存目录布局（供界面 / 调试查看）"""
    return {
        "root": str(PRETRAINED_DIR),
        "torch": str(torch_dir()),
        "ultralytics": str(ultralytics_dir()),
        "yolov13": str(yolov13_dir()),
        "clip": str(clip_dir()),
        "env": {k: os.environ.get(k, "")
                for k in ("TORCH_HOME", "CLIP_CACHE_DIR")},
    }
