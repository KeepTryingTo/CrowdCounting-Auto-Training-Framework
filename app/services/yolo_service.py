# -*- coding: utf-8 -*-
"""YOLO 系列头部定位框检测服务（ultralytics 引擎，搬移自原自动训练框架）

路线：crowd_box（人群计数 · 头部框预测）——用 YOLO 系列检测人头的
bounding box，人头数量 = 检测框数量。COCO 预训练不含 head 类别，
推理需训练产物/上传 .pt 权重；训练为 YOLO 微调（自动下载 COCO 预训练
替换分类头）或从架构 yaml 从头训练。

标注格式（训练，三选一）：
    yolo: root/Train/Images/*.png + root/Train/Labels/*.txt
          （每行 class_id cx cy w h，归一化中心坐标；单类 head 即 class_id=0）
    voc:  root/Train/Images/ + root/Train/Labels/*.xml
    csv:  root/annotations.csv + root/class_list.csv

依赖：pip install ultralytics（未安装时本路线模型不可用，其余功能不受影响）
"""
import base64
import csv
import io
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from app.services.loss_weight_utils import all_weights, describe
from app.services.train_common import (set_seed, save_loss_curves,
                                       release_training_memory)

# ultralytics 官方模型：名称 -> (描述, 最低 ultralytics 版本)
# 首次使用自动下载 COCO 预训练 .pt 权重（微调时自动替换分类头为数据集类别）
YOLO_AUTO = {
    # --- YOLO 老系列（u = Ultralytics 更新版检查点）---
    "yolov3u": ("YOLOv3u（Ultralytics 更新版）", (8, 0, 0)),
    "yolov5nu": ("YOLOv5n-u（Ultralytics 更新版）", (8, 0, 0)),
    "yolov5su": ("YOLOv5s-u", (8, 0, 0)),
    "yolov5mu": ("YOLOv5m-u", (8, 0, 0)),
    "yolov5lu": ("YOLOv5l-u", (8, 0, 0)),
    "yolov5xu": ("YOLOv5x-u", (8, 0, 0)),
    # --- YOLOv6（美团架构：仅 yaml 配置，官方不托管 .pt，从架构从头训练）---
    "yolov6n": ("YOLOv6n（美团，从 yaml 训练，无官方 .pt）", (8, 0, 0)),
    "yolov6s": ("YOLOv6s（美团，从 yaml 训练）", (8, 0, 0)),
    "yolov6m": ("YOLOv6m（美团，从 yaml 训练）", (8, 0, 0)),
    # --- YOLOv8 - YOLO12 ---
    "yolov8n": ("YOLOv8n（nano）", (8, 0, 0)),
    "yolov8s": ("YOLOv8s（small）", (8, 0, 0)),
    "yolov8m": ("YOLOv8m（medium）", (8, 0, 0)),
    "yolov8l": ("YOLOv8l（large）", (8, 0, 0)),
    "yolov8x": ("YOLOv8x（xlarge）", (8, 0, 0)),
    "yolov9t": ("YOLOv9t（tiny）", (8, 2, 0)),
    "yolov9c": ("YOLOv9c", (8, 2, 0)),
    "yolov9e": ("YOLOv9e", (8, 2, 0)),
    "yolov10n": ("YOLOv10n（nano）", (8, 2, 0)),
    "yolov10s": ("YOLOv10s（small）", (8, 2, 0)),
    "yolov10b": ("YOLOv10b（base）", (8, 2, 0)),
    "yolov10l": ("YOLOv10l（large）", (8, 2, 0)),
    "yolov10x": ("YOLOv10x（xlarge）", (8, 2, 0)),
    "yolo11n": ("YOLO11n（nano）", (8, 3, 0)),
    "yolo11s": ("YOLO11s（small）", (8, 3, 0)),
    "yolo11m": ("YOLO11m（medium）", (8, 3, 0)),
    "yolo11l": ("YOLO11l（large）", (8, 3, 0)),
    "yolo11x": ("YOLO11x（xlarge）", (8, 3, 0)),
    "yolo12n": ("YOLO12n（nano）", (8, 3, 78)),
    "yolo12s": ("YOLO12s（small）", (8, 3, 78)),
    "yolo12m": ("YOLO12m（medium）", (8, 3, 78)),
    "yolo12l": ("YOLO12l（large）", (8, 3, 78)),
    "yolo12x": ("YOLO12x（xlarge）", (8, 3, 78)),
    # --- YOLO26（最新：无 NMS 端到端，按安装版本自动过滤）---
    "yolo26n": ("YOLO26n（最新，无 NMS 端到端）", (8, 3, 200)),
    "yolo26s": ("YOLO26s", (8, 3, 200)),
    "yolo26m": ("YOLO26m", (8, 3, 200)),
    "yolo26l": ("YOLO26l", (8, 3, 200)),
    "yolo26x": ("YOLO26x", (8, 3, 200)),
    # --- RT-DETR（百度实时 Transformer 检测器，专用类）---
    "rtdetr_l": ("RT-DETR-L（百度实时 Transformer 检测器）", (8, 1, 0)),
    "rtdetr_x": ("RT-DETR-X", (8, 1, 0)),
}

# 从 yaml 架构配置从头训练的模型（官方不托管 .pt 检查点）
YOLO_YAML_ONLY = {"yolov6n", "yolov6s", "yolov6m"}

# 从 yaml 构建时的架构配置名映射（部分系列 yaml 命名与检查点名不同）
_YAML_NAME = {
    "yolov3u": "yolov3.yaml",
    "rtdetr_l": "rtdetr-l.yaml",
    "rtdetr_x": "rtdetr-x.yaml",
}

# 官方检查点名映射（默认 {name}.pt；RT-DETR 用连字符命名）
_PT_NAME = {
    "rtdetr_l": "rtdetr-l.pt",
    "rtdetr_x": "rtdetr-x.pt",
}

# YOLOv13 官方 COCO 预训练权重（iMoonLab/yolov13 GitHub Release）
# 标准 ultralytics 包不含 YOLOv13 架构，需安装官方分支引擎
# 名称 -> (描述, 下载地址, 字节数, sha256)
YOLOV13_URLS = {
    "yolov13n": ("YOLOv13n（nano，HyperACE 超图增强）",
                 "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt",
                 10570688,
                 "6653035017b0f111f80ec11ed914874ea85699b104aeac1e46e517d16889d6b7"),
    "yolov13s": ("YOLOv13s（small，HyperACE 超图增强）",
                 "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13s.pt",
                 36922944,
                 "cf0fe62e2ef520a63c1e3d380f5771b7ac701b97d07f11a28c314e68618ba072"),
    "yolov13l": ("YOLOv13l（large，HyperACE 超图增强）",
                 "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt",
                 111819790,
                 "f95ad5bbf3aa80a3df2a28ff4c623582ded6e02bb43d7cea9063c2b2448e316cb"),
    "yolov13x": ("YOLOv13x（xlarge，HyperACE 超图增强）",
                 "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13x.pt",
                 257795222,
                 "df5f9af2e1ba506c4b164cd22a8ae6e9eff23295265df3a137e1ce8ccf4c1700d"),
}

# 需上传 .pt 权重的模型（官方不托管权重）与自定义权重；
# 训练时未提供权重不报错：回退到默认架构（yolov8n.yaml）从头训练
YOLO_UPLOAD = {
    "yolo_custom": "自定义 YOLO 权重（任意 ultralytics .pt）",
}

# YOLOv13 分支引擎缺失时的安装引导
_YOLOV13_INSTALL_HINT = (
    "当前环境无法加载 YOLOv13 架构（HyperACE/FullPAD/DS-C3k2）。请确认项目 "
    "third_party/yolov13/yolov13.yaml 存在，且已安装 ultralytics / torch；"
    "或改用 iMoonLab/yolov13 分支引擎。"
)

# 稳定的类别颜色（头部框固定用红色系更醒目，但保留调色板供多类场景）
_PALETTE = [(random.Random(i % 24).randint(50, 255),
             random.Random(i % 24 + 7).randint(50, 255),
             random.Random(i % 24 + 13).randint(50, 255))
            for i in range(24)]
_HEAD_COLOR = (0, 220, 120)   # head 类统一绿色


class _YoloStop(Exception):
    """用户请求停止训练（在 ultralytics 回调中抛出，由训练流程捕获）"""


def is_available() -> bool:
    """检测 ultralytics 引擎是否已安装"""
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def _require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception as e:
        raise RuntimeError(
            f"未安装 ultralytics 引擎（{e}）：请先执行 pip install ultralytics "
            f"后重启服务再使用头部框预测功能") from e


def is_yolo_model(name) -> bool:
    try:
        from app.core.registry import is_custom_model_for
        if is_custom_model_for(name, "crowd_box"):
            return True
    except Exception:
        pass
    return str(name or "").lower().startswith("yolo") or \
        str(name or "").lower().startswith("rtdetr")


def _is_custom_box_model(name) -> bool:
    try:
        from app.core.registry import is_custom_model_for
        return is_custom_model_for(name, "crowd_box")
    except Exception:
        return False


def _install_hint() -> str:
    return "" if is_available() else "（未安装 ultralytics 引擎，请先 pip install ultralytics）"


# 网络类错误关键词（用于判断下载失败是否值得重试）
_NET_ERR_KEYS = ("download", "connection", "timed out", "timeout", "url",
                 "getaddrinfo", "reset", "ssl", "eof", "unreachable",
                 "network", "proxy", "socket", "handshake")


def _looks_like_net_error(e) -> bool:
    """异常是否为网络/下载类错误（值得自动重试）。

    排除「资源不存在」类错误（Entry Not Found / 404 等）：
    其消息常含 url/download 字样，但重试无意义。
    """
    s = str(e).lower()
    if "not found" in s or "404" in s or "no such file" in s:
        return False
    return any(k in s for k in _NET_ERR_KEYS)


def _cached_asset(name: str) -> str:
    """ultralytics 官方权重统一缓存路径（pretrained model/ultralytics/）"""
    p = Path(name)
    if p.suffix.lower() == ".pt" and p.parent == Path("."):
        from app.core.model_cache import ultralytics_dir
        return str(ultralytics_dir() / name)
    return name


def _load_asset_with_retry(build, asset_name: str, log=None, tries: int = 3):
    """加载需联网下载的官方权重：网络类错误自动重试，
    多次失败后抛出含手动下载指引（GitHub 资产页 + 放置路径）的错误。"""
    last = None
    for i in range(max(1, tries)):
        try:
            return build()
        except Exception as e:
            if not _looks_like_net_error(e):
                raise          # 非网络类错误（模型/参数问题）不重试
            last = e
            if log:
                log(f"官方权重 {asset_name} 下载失败"
                    f"（第 {i + 1}/{tries} 次）：{e}")
    target = _cached_asset(asset_name)
    place = (f"下载后放到：{target}\n"
             f"或经界面上传该 .pt 文件后在权重下拉框中选择。"
             if target != asset_name else
             "或经界面上传该 .pt 文件后在权重下拉框中选择。")
    raise OSError(
        f"官方权重 {asset_name} 自动下载失败（已尝试 {tries} 次）：{last}\n"
        f"网络受限可在「代理设置」配置代理后重试；"
        f"或手动下载：https://github.com/ultralytics/assets/releases"
        f"（在 Assets 中搜索 {asset_name}）\n{place}")


def _ultralytics_version():
    """已安装的 ultralytics 版本（(major, minor, patch) 元组），未安装返回 None"""
    try:
        import ultralytics
        parts = []
        for x in str(ultralytics.__version__).split(".")[:3]:
            digits = "".join(ch for ch in x if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)
    except Exception:
        return None


def _model_supported(name) -> bool:
    """当前 ultralytics 版本是否支持该内置模型（未安装时不筛选，由安装提示兜底）"""
    ver = _ultralytics_version()
    if ver is None or name not in YOLO_AUTO:
        return True
    return ver >= YOLO_AUTO[name][1]


def _model_class_for(model_name: str):
    """按模型家族返回 ultralytics 模型类（默认 YOLO；RT-DETR 专用类）"""
    import ultralytics
    if model_name.startswith("rtdetr"):
        return getattr(ultralytics, "RTDETR", None) or ultralytics.YOLO
    return ultralytics.YOLO


def _yolov13_supported() -> bool:
    """当前引擎是否可用 YOLOv13。

    两条路径：
    1. 已安装 iMoonLab 分支（cfg/models/v13 或 yolov13*.yaml）；
    2. 标准 ultralytics + 项目内置 third_party/yolov13 架构适配层。
    """
    try:
        import ultralytics
        root = Path(ultralytics.__file__).resolve().parent
        if (root / "cfg" / "models" / "v13").is_dir():
            return True
        if any(root.glob("cfg/models/**/yolov13*.yaml")):
            return True
    except Exception:
        pass
    try:
        from app.services import yolov13_arch
        return yolov13_arch.available()
    except Exception:
        return False


# GitHub 直连不稳定时的公共加速镜像前缀，依次回退尝试
_YOLOV13_MIRRORS = (
    "",                          # 官方直连（优先）
    "https://ghfast.top/",       # ghfast 加速
    "https://gh-proxy.com/",     # gh-proxy 加速
    "https://ghproxy.net/",      # ghproxy 加速
)

_YOLOV13_DOWNLOAD_ROUNDS = 2


def _download_yolov13_part(part: Path, url: str, size: int, sha: str):
    """从单个地址流式下载到 .part 临时文件，并校验字节数与 sha256"""
    import hashlib
    import urllib.request
    h = hashlib.sha256()
    got = 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(part, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            got += len(chunk)
    if got != size:
        raise OSError(f"长度不符（实际 {got}B / 期望 {size}B）")
    if h.hexdigest() != sha:
        raise OSError("sha256 校验不匹配")


def _yolov13_weight_path(model_name: str, log=None) -> Path:
    """YOLOv13 官方 COCO 预训练权重本地路径（无缓存时从 GitHub Release 下载，
    失败回退加速镜像，成功后原子替换，全部失败给出手动下载指引）"""
    from app.core.model_cache import yolov13_dir, resolve_legacy
    _, url, size, sha = YOLOV13_URLS[model_name]
    cand = resolve_legacy(yolov13_dir() / f"{model_name}.pt",
                          "yolov13", f"{model_name}.pt")
    if cand.is_file() and cand.stat().st_size == size:
        return cand
    dst = yolov13_dir() / f"{model_name}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    if log:
        log(f"下载 YOLOv13 官方 COCO 预训练权重 {model_name}.pt"
            f"（{size / 1048576:.1f} MB，iMoonLab/yolov13 Release）...")
    errors = []
    rounds = max(1, _YOLOV13_DOWNLOAD_ROUNDS)
    for rnd in range(rounds):
        for prefix in _YOLOV13_MIRRORS:
            try:
                _download_yolov13_part(part, prefix + url, size, sha)
                part.replace(dst)
                if log:
                    log(f"权重已缓存: {dst}")
                return dst
            except Exception as e:
                errors.append(f"[轮次{rnd + 1}] {prefix or '官方直连'}: {e}")
        if log and rnd < rounds - 1:
            log(f"下载失败（第 {rnd + 1}/{rounds} 轮），自动重试...")
    part.unlink(missing_ok=True)
    mirror = _YOLOV13_MIRRORS[1] + url if len(_YOLOV13_MIRRORS) > 1 else url
    raise OSError(
        f"YOLOv13 权重自动下载失败（已重试 {rounds} 轮）："
        f"{'；'.join(errors[-len(_YOLOV13_MIRRORS):])}\n"
        f"可手动下载：{url}\n"
        f"（网络受限可用加速镜像：{mirror}）\n"
        f"下载后放到：{dst}\n"
        f"或经界面上传该 .pt 文件后在权重下拉框中选择。")


def list_models():
    """推理模型清单（头部框：COCO 预训练不含 head 类，推理需训练产物/上传权重）"""
    hint = _install_hint()
    out = []
    for n, (d, _) in YOLO_AUTO.items():
        if not _model_supported(n):
            continue
        out.append({"name": n, "desc": f"{d} — 头部框数据集训练后使用{hint}",
                    "type": "yolo", "pretrained": False,
                    "pretrained_local": False, "need_weight": True,
                    "input_size": 640})
    v13_ok = _yolov13_supported()
    for n, (d, _, _, _) in YOLOV13_URLS.items():
        out.append({"name": n,
                    "desc": (f"{d} — 头部框数据集训练后使用"
                             + ("" if v13_ok else "（缺少 third_party/yolov13）")
                             + hint),
                    "type": "yolo", "pretrained": False,
                    "pretrained_local": False, "need_weight": True,
                    "input_size": 640})
    for n, d in YOLO_UPLOAD.items():
        out.append({"name": n, "desc": f"{d} — 头部框数据集训练后使用{hint}",
                    "type": "yolo", "pretrained": False,
                    "pretrained_local": False, "need_weight": True,
                    "input_size": 640})
    return out


def list_train_models():
    """训练模型清单（YOLO 微调/从头训练，ultralytics 引擎）"""
    hint = _install_hint()
    out = []
    for n, (d, _) in YOLO_AUTO.items():
        if not _model_supported(n):
            continue
        out.append({"name": n,
                    "desc": f"{d} — YOLO 微调/从头训练（ultralytics 引擎）{hint}",
                    "pretrained": n not in YOLO_YAML_ONLY,
                    "pretrained_local": False,
                    "need_weight": False})
    v13_ok = _yolov13_supported()
    for n, (d, _, _, _) in YOLOV13_URLS.items():
        out.append({"name": n,
                    "desc": (f"{d} — 官方 COCO 预训练微调（自动下载，内置架构）"
                             if v13_ok else
                             f"{d} — 缺少 third_party/yolov13 架构")
                             + hint,
                    "pretrained": v13_ok, "pretrained_local": False,
                    "need_weight": not v13_ok})
    for n, d in YOLO_UPLOAD.items():
        out.append({"name": n,
                    "desc": f"{d} — 选择 .pt 微调；未选择时回退默认架构"
                            f"(yolov8n)从头训练{hint}",
                    "pretrained": False, "pretrained_local": False,
                    "need_weight": False})
    return out


def _map_device(device_str):
    """框架设备标识 → ultralytics 设备参数（cpu / "0" / [0,1,...]）"""
    d = str(device_str or "cpu").lower()
    if d.startswith("cpu"):
        return "cpu"
    if d.startswith("cuda"):
        if ":" in d:
            return d.split(":", 1)[1]
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                return list(range(torch.cuda.device_count()))
            return "0"
        except Exception:
            return "0"
    return device_str


# ------------------------- 知识蒸馏（ultralytics 引擎原生） -------------------------

def _distill_supported() -> bool:
    """当前 ultralytics 引擎是否原生支持知识蒸馏（8.4+：cfg 含 distill_model/dis）"""
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        return ("distill_model" in DEFAULT_CFG_DICT
                and "dis" in DEFAULT_CFG_DICT)
    except Exception:
        return False


def _resolve_distill_teacher(params: dict, log=print):
    """解析蒸馏教师来源 → (权重路径, 教师参数量, 描述)。

    优先级：教师权重文件（训练产物 / 上传 .pt，推荐同数据集训练产物，
    类别一致蒸馏效果最好）> 官方教师模型名（自动下载 COCO 预训练；
    特征蒸馏与得分加权均与类别数无关，COCO 教师同样可用）。
    返回可直接传给 ultralytics train(distill_model=...) 的本地路径。
    """
    YOLO = _require_ultralytics()
    teacher_weight = params.get("distill_teacher_weight")
    teacher_model = params.get("distill_teacher_model")
    if not (teacher_weight or teacher_model):
        raise ValueError("启用知识蒸馏需要指定教师：请选择教师权重文件"
                         "（推荐同数据集训练产物）或官方教师模型")

    if teacher_weight:
        tw = Path(teacher_weight)
        if not tw.is_file():
            raise ValueError(f"教师权重文件不存在: {teacher_weight}")
        try:
            ty = YOLO(str(tw))
        except Exception as e:
            raise ValueError(f"教师权重加载失败（需 ultralytics 格式 .pt）: {e}")
        if type(ty.model).__name__ == "DistillationModel":
            raise ValueError("所选教师权重是蒸馏训练产物（含学生-教师包装），"
                             "暂不支持再作为教师：请选择普通训练产物 "
                             "best.pt / last.pt 或官方预训练权重")
        n_t = sum(p.numel() for p in ty.model.parameters())
        return str(tw), n_t, tw.name

    name = teacher_model
    if name.startswith("rtdetr"):
        raise ValueError("RT-DETR 检测头结构不同，暂不支持作为蒸馏教师"
                         "（请选择 YOLO 系列教师）")
    if name in YOLO_YAML_ONLY:
        raise ValueError(f"{name} 无官方预训练权重（yaml 从头训练系列），"
                         "不能作为教师模型：请改用「教师权重文件」")
    if name in YOLOV13_URLS:
        if not _yolov13_supported():
            raise ValueError(_YOLOV13_INSTALL_HINT)
        pt = _yolov13_weight_path(name, log=log)
        ty = YOLO(str(pt))
    elif name in YOLO_AUTO:
        if not _model_supported(name):
            need = ".".join(str(x) for x in YOLO_AUTO[name][1])
            raise ValueError(f"当前 ultralytics 版本不支持教师模型 {name}："
                             f"请执行 pip install -U 'ultralytics>={need}'")
        pt_name = _PT_NAME.get(name, f"{name}.pt")
        cached = _cached_asset(pt_name)
        # 预下载官方权重到本地缓存（引擎按路径二次加载）
        ty = _load_asset_with_retry(lambda: YOLO(cached), pt_name, log=log)
        pt = Path(cached)
    else:
        raise ValueError(f"未知教师模型: {name}")
    n_t = sum(p.numel() for p in ty.model.parameters())
    return str(pt), n_t, f"{name}（官方 COCO 预训练）"


# ------------------------- 标注加载（YOLO / VOC / CSV） -------------------------

def _det_split_dir(root: Path, train: bool):
    """定位 Train/Test 子目录（兼容 train/test/val 命名）"""
    names = ["Train", "train"] if train else ["Test", "test", "Val", "val"]
    for n in names:
        d = root / n
        if d.is_dir():
            return d
    return None


def _det_sub_dirs(split_dir: Path):
    """split 目录下定位 Images 与 Labels（兼容多种命名）"""
    img_dir = None
    for n in ["Images", "images", "JPEGImages", "imgs", "img"]:
        if (split_dir / n).is_dir():
            img_dir = split_dir / n
            break
    lbl_dir = None
    for n in ["Labels", "labels", "Annotations", "annotations", "xml", "anno"]:
        if (split_dir / n).is_dir():
            lbl_dir = split_dir / n
            break
    return img_dir, lbl_dir


def load_csv_items(root, csv_file, class_list_file):
    """CSV 格式（yhenon）加载 → (items, class_names)"""
    root = Path(root)
    class_map = {}
    for row in csv.reader(Path(class_list_file)
                          .read_text(encoding="utf-8").splitlines()):
        if len(row) >= 2 and row[0].strip():
            class_map[row[0].strip()] = int(row[1])
    class_names = {v: k for k, v in class_map.items()}
    # 同名图的多框聚合
    per_img = {}
    for row in csv.reader(Path(csv_file)
                          .read_text(encoding="utf-8").splitlines()):
        if not row or not row[0].strip():
            continue
        img_path = row[0].strip()
        if not Path(img_path).is_absolute():
            img_path = str(root / img_path)
        per_img.setdefault(img_path, [])
        if len(row) >= 6 and row[1].strip():
            try:
                box = [float(row[1]), float(row[2]),
                       float(row[3]), float(row[4])]
                cls = class_map.get(row[5].strip())
                if cls is not None:
                    per_img[img_path].append(box + [cls])
            except ValueError:
                continue
    items = []
    for img_path, anns in per_img.items():
        if Path(img_path).exists():
            items.append((img_path,
                          np.array(anns, dtype=np.float32).reshape(-1, 5)))
    if not items:
        raise ValueError(f"CSV 数据集无有效样本: {csv_file}")
    return items, class_names


def load_yolo_items(root, train=True):
    """YOLO 格式加载 → (items, class_names)

    root/Train/Images/*.png + root/Train/Labels/*.txt
    Labels 每行: class_id cx cy w h（归一化中心坐标；单类 head 即 class_id=0）
    类名可选 root/classes.txt（每行类名，行号=id），否则 class_0..class_N
    """
    root = Path(root)
    split = _det_split_dir(root, train)
    if split is None:
        raise ValueError(f"未找到 Train/Test 子目录"
                          f"（YOLO 格式要求 root/Train/Images+Labels）: {root}")
    img_dir, lbl_dir = _det_sub_dirs(split)
    if img_dir is None or lbl_dir is None:
        raise ValueError(f"{split} 下未找到 Images 与 Labels 目录")
    class_names = {}
    cl = None
    for name in ("classes.txt", "class_list.txt"):
        for base in (root, split):
            if (base / name).exists():
                cl = base / name
                break
        if cl is not None:
            break
    if cl is not None:
        for i, line in enumerate(cl.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                class_names[i] = line.strip()
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    items = []
    max_id = 0
    for img_p in sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in exts):
        txt = lbl_dir / (img_p.stem + ".txt")
        anns = []
        if txt.exists():
            with Image.open(img_p) as im:
                W, H = im.size
            for line in txt.read_text(encoding="utf-8-sig").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid, cx, cy, w, h = (int(parts[0]), float(parts[1]),
                                         float(parts[2]), float(parts[3]),
                                         float(parts[4]))
                except ValueError:
                    continue
                x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
                anns.append([x1, y1, x2, y2, cid])
                max_id = max(max_id, cid)
        items.append((str(img_p),
                      np.array(anns, dtype=np.float32).reshape(-1, 5)))
    if not class_names:
        class_names = {i: f"class_{i}" for i in range(max_id + 1)}
    if not items:
        raise ValueError(f"YOLO 目录无图像: {img_dir}")
    return items, class_names


def load_voc_items(root, train=True):
    """PASCAL VOC 格式加载 → (items, class_names)

    root/Train/Images/*.png + root/Train/Labels/*.xml
    xml 内 <object><name>..</name><bndbox><xmin><ymin><xmax><ymax></bndbox></object>
    """
    import xml.etree.ElementTree as ET
    root = Path(root)
    split = _det_split_dir(root, train)
    if split is None:
        raise ValueError(f"未找到 Train/Test 子目录"
                          f"（VOC 格式要求 root/Train/Images+Labels）: {root}")
    img_dir, lbl_dir = _det_sub_dirs(split)
    if img_dir is None or lbl_dir is None:
        raise ValueError(f"{split} 下未找到 Images 与 Labels 目录")
    name_to_id = {}
    cl = None
    for name in ("classes.txt", "class_list.txt"):
        for base in (root, split):
            if (base / name).exists():
                cl = base / name
                break
        if cl is not None:
            break
    if cl is not None:
        for i, line in enumerate(cl.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                name_to_id[line.strip()] = i
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    items = []
    auto_id = 0
    for img_p in sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in exts):
        xml_f = lbl_dir / (img_p.stem + ".xml")
        anns = []
        if xml_f.exists():
            try:
                tree = ET.parse(xml_f)
            except ET.ParseError:
                continue
            for obj in tree.getroot().iter("object"):
                name_el = obj.find("name")
                bb = obj.find("bndbox")
                if name_el is None or bb is None:
                    continue
                cname = (name_el.text or "").strip()
                if not cname:
                    continue
                if cname not in name_to_id:
                    if cl is None:
                        name_to_id[cname] = auto_id
                        auto_id += 1
                    else:
                        continue   # classes.txt 已固定类别表，未知类跳过
                try:
                    x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
                    x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
                except (TypeError, ValueError):
                    continue
                anns.append([x1, y1, x2, y2, name_to_id[cname]])
        items.append((str(img_p),
                      np.array(anns, dtype=np.float32).reshape(-1, 5)))
    class_names = {v: k for k, v in name_to_id.items()}
    if not items:
        raise ValueError(f"VOC 目录无图像: {img_dir}")
    return items, class_names


# ------------------------- 数据集准备 -------------------------

def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(str(a), str(b))
    except OSError:
        return False


def _link_or_copy(src: Path, dst: Path):
    """硬链接优先（省磁盘），跨卷回退复制"""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _yaml_quote(s) -> str:
    s = str(s)
    if s and not s.isalnum() and not (s.startswith('"') and s.endswith('"')):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _available_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1048576.0
    except Exception:
        return None


def _yolo_cpu_gt_limit(imgsz: int, batch_size: int):
    """CPU 训练时单图 GT 框数上限的保守估计。

    YOLO 的 TaskAlignedAssigner 会为每张图分配
    ``n_gt × num_anchors`` 的 mask / 指标张量；JHU 等数据集存在
    2 万+ GT 的极端图像，CPU 上会直接耗尽内存导致原生崩溃。
    这里按可用内存和输入尺寸给出一个上限，超限图像在训练集中跳过。
    """
    avail = _available_ram_mb()
    anchors = sum((imgsz // s) ** 2 for s in (8, 16, 32))
    per_box_mb = anchors * 12 / 1048576.0   # mask + align + overlap
    scale = (imgsz / 640.0) ** 2 * max(1, batch_size) / 2.0
    fixed = 5000 * (2.0 / max(1, batch_size)) * (640.0 / imgsz) ** 2
    if avail is None:
        return int(max(500, min(8000, fixed)))
    reserve = 1200 + 600 * scale
    budget = max(0.0, avail * 0.8 - reserve)
    dynamic = budget / per_box_mb if per_box_mb > 0 else 0
    return int(max(500, min(20000, min(fixed, dynamic))))


def _prepare_yolo_dataset(root: Path, fmt: str, save_dir: Path,
                          val_ratio=0.2, seed=42, log=print,
                          max_gt_per_image=None):
    """将三种检测标注格式转换为 ultralytics 标准布局并生成 data.yaml。

    输出：save_dir/dataset/{images,labels}/{train,val}/ + data.yaml
    返回 (data_yaml_path, names_list)
    """
    root = Path(root)
    fmt = (fmt or "yolo").lower()

    def _random_split(items):
        gg = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(items), generator=gg).tolist()
        n_val = max(1, int(len(items) * val_ratio))
        if len(items) - n_val < 1:
            n_val = max(0, len(items) - 1)
        return [items[i] for i in idx[:n_val]], [items[i] for i in idx[n_val:]]

    if fmt == "csv":
        ann, cls_f = root / "annotations.csv", root / "class_list.csv"
        if not ann.exists() or not cls_f.exists():
            raise ValueError(f"CSV 格式需要 annotations.csv 与 "
                             f"class_list.csv: {root}")
        items, class_names = load_csv_items(root, ann, cls_f)
        if not items:
            raise ValueError("CSV 数据集无有效样本")
        val_items, train_items = _random_split(items)
    elif fmt in ("yolo", "voc"):
        loader = load_yolo_items if fmt == "yolo" else load_voc_items
        train_items, class_names = loader(root, train=True)
        try:
            val_items, _ = loader(root, train=False)
        except ValueError:
            val_items, train_items = _random_split(train_items)
    else:
        raise ValueError(f"未知标注格式: {fmt}（支持 yolo / voc / csv）")

    if not train_items:
        raise ValueError("训练集为空，请检查数据集目录结构")
    if not val_items:
        val_items = list(train_items)

    if max_gt_per_image:
        def _n_gt(ann):
            try:
                return int(np.asarray(ann).reshape(-1, 5).shape[0])
            except Exception:
                return 0
        kept = [it for it in train_items
                if _n_gt(it[1]) <= int(max_gt_per_image)]
        skipped = len(train_items) - len(kept)
        if kept and skipped:
            log(f"[内存保护] 训练集中 {skipped} 张图像的 GT 框数 > "
                f"{int(max_gt_per_image)}（极端密集），已跳过以避免 CPU "
                f"内存崩溃；如需使用请减小输入尺寸/批大小或改用 GPU")
            train_items = kept
        elif not kept:
            log(f"[内存保护] 所有训练图像 GT 框数都超过 "
                f"{int(max_gt_per_image)}，未做过滤（请显著减小输入尺寸）")

    nc = (max(class_names) + 1) if class_names else 1
    names = [class_names.get(i, f"class_{i}") for i in range(nc)]

    out_root = save_dir / "dataset"
    for split, items in (("train", train_items), ("val", val_items)):
        # ultralytics 标准布局：images/<split>/ 与 labels/<split>/ 同级
        # （引擎按路径中的 /images/ → /labels/ 推导标签位置）
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n_img = 0
        for i, (img_path, annot) in enumerate(items):
            img_path = Path(img_path)
            if not img_path.exists():
                continue
            fname = img_path.name
            dst_img = img_dir / fname
            if dst_img.exists() and not _same_file(img_path, dst_img):
                fname = f"{i}_{img_path.name}"
                dst_img = img_dir / fname
            try:
                _link_or_copy(img_path, dst_img)
            except OSError as e:
                log(f"跳过无法访问的图像 {img_path}: {e}")
                continue
            n_img += 1
            # 生成 ultralytics YOLO 标签（归一化 cx cy w h）
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                W, H = 0, 0
            lines = []
            if W and H:
                for x1, y1, x2, y2, cid in np.asarray(annot).reshape(-1, 5):
                    x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
                    x2, y2 = min(float(W), float(x2)), min(float(H), float(y2))
                    if x2 - x1 < 1 or y2 - y1 < 1:
                        continue
                    cx = (x1 + x2) / 2.0 / W
                    cy = (y1 + y2) / 2.0 / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H
                    lines.append(
                        f"{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (lbl_dir / (Path(fname).stem + ".txt")).write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        log(f"YOLO 数据集[{split}]: {n_img} 张图像 | 保存于 {img_dir}")

    data_yaml = out_root / "data.yaml"
    yaml_text = (f"# 自动生成：人群计数自动训练和推理框架（头部框路线）\n"
                 f"path: {out_root.resolve().as_posix()}\n"
                 f"train: images/train\n"
                 f"val: images/val\n"
                 f"nc: {nc}\n"
                 f"names:\n"
                 + "\n".join(f"  {i}: {_yaml_quote(n)}"
                             for i, n in enumerate(names))
                 + "\n")
    data_yaml.write_text(yaml_text, encoding="utf-8")
    return data_yaml, names


# ------------------------- 训练 -------------------------

def run_yolo_train(task, params: dict):
    """YOLO 头部框训练（ultralytics 引擎）

    模型来源优先级：断点续训 > 初始权重微调 > 官方预训练 .pt > yaml 从头训练；
    无内置权重的模型（如 yolo_custom）未上传 .pt 时回退默认架构从头训练。
    """
    YOLO = _require_ultralytics()
    model_name = params["model"]
    dataset_path = params["dataset_path"]
    fmt = (params.get("annotation_format") or "yolo").lower()
    epochs = int(params.get("epochs", 50))
    batch_size = int(params.get("batch_size", 2))
    # lr=0 / 缺省 → YOLO 论文默认 lr0=1e-3
    lr = float(params.get("lr") or 1e-3)
    weight_decay = float(params.get("weight_decay", 5e-4))
    pretrained = bool(params.get("pretrained", True))
    weight_path = params.get("weight_path")
    device_str = params.get("device", "cpu")
    num_workers = int(params.get("num_workers", 0))
    imgsz = int(params.get("img_size", 0)) or 640
    early_stop = int(params.get("early_stopping", 0))
    seed = int(params.get("seed", 42))
    amp = bool(params.get("amp", False))
    val_ratio = float(params.get("val_ratio", 0.2))

    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    task.total_epochs = epochs
    task.log(f"YOLO 头部框训练 | 模型: {model_name} | 引擎: ultralytics | "
             f"标注格式: {fmt}")
    task.log(f"数据集: {dataset_path} | 设备: {device_str} | 输入尺寸: {imgsz}")
    set_seed(seed)

    # CPU 训练内存保护：极端密集图像的 GT 框数会放大 TAL 分配张量，
    # 直接在训练集准备阶段跳过，避免运行时 access violation 带走服务。
    gt_limit = None
    if not device_str.startswith("cuda"):
        gt_limit = _yolo_cpu_gt_limit(imgsz, batch_size)
        avail = _available_ram_mb()
        task.log(f"[内存保护] CPU 训练单图 GT 上限 {gt_limit}"
                 + (f"（可用内存 {avail:.0f}MB）" if avail is not None else ""))

    # 数据准备（YOLO / VOC / CSV → ultralytics 标准布局 + data.yaml）
    task.log("正在准备 ultralytics 数据集（链接图像 + 生成 YOLO 标签与 "
             "data.yaml）...")
    data_yaml, names = _prepare_yolo_dataset(
        Path(dataset_path), fmt, save_dir, val_ratio=val_ratio, seed=seed,
        log=task.log, max_gt_per_image=gt_limit)
    task.log(f"类别数: {len(names)} | 类别: {names}")
    (save_dir / "classes.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")

    # 模型构建（断点续训优先：ultralytics 原生 resume，自动恢复
    # epoch/optimizer/scheduler 与训练参数；RT-DETR 用专用类）
    resume_pt = params.get("resume_weight")
    model_cls = _model_class_for(model_name)
    if model_cls.__name__ != "YOLO":
        task.log(f"模型类: {model_cls.__name__}")
    # YOLO 只能加载 ultralytics 格式的 .pt；.pth 等框架检查点给出明确提示
    if weight_path and str(weight_path).lower().endswith(".pth"):
        raise ValueError(
            f"YOLO 头部框训练需要 ultralytics 格式的 .pt 检查点，当前选择的是 "
            f".pth：{weight_path}\n（.pth 为密度图/点预测训练产物，请改选 .pt "
            f"训练产物或上传 .pt 权重）")

    # 断点续训的可行性检查：所选检查点若已跑完全部 epoch（如正常结束的
    # last.pt），ultralytics 会报 "nothing to resume"；此时自动改为按初始化
    # 权重微调，避免用户拿到看不懂的断言错误
    if resume_pt:
        try:
            # 本地训练产物可信，显式关闭 weights_only（torch>=2.6 默认 True
            # 会因检查点含模型对象而失败）
            _ck = torch.load(resume_pt, map_location="cpu", weights_only=False)
            if isinstance(_ck, dict):
                _args = _ck.get("train_args") or _ck.get("args") or {}
                _ep = int(_ck.get("epoch", 0) or 0)
                _total = int((_args or {}).get("epochs", 0) or 0)
                # 正常结束的检查点：epoch 被 strip 成 -1（且 optimizer=None），
                # 或已跑满总轮数 → ultralytics 会报 nothing to resume
                _finished = _ep < 0 or (_total and _ep + 1 >= _total)
                if _finished:
                    task.log(f"所选检查点已完成训练（epoch 记录 {_ep}"
                             f"{f'/{_total}' if _total else ''}），"
                             f"无法断点续训：改为作为初始化权重微调")
                    weight_path = weight_path or resume_pt
                    resume_pt = None
        except Exception:
            pass
    if resume_pt:
        model = model_cls(resume_pt)
        task.log(f"断点续训：从 {resume_pt} 恢复（ultralytics 原生 resume）")
    elif weight_path:
        model = model_cls(weight_path)
        task.log(f"微调初始权重: {weight_path}")
    elif _is_custom_box_model(model_name):
        # 自定义头部框模型：注册工厂返回 ultralytics 可加载的
        # .yaml / .pt 路径，或直接返回已构建的模型对象
        from app.core.registry import get_custom_model_source
        src = get_custom_model_source(model_name)
        model = src if hasattr(src, "train") else model_cls(str(src))
        task.log(f"自定义头部框模型: {model_name} ← {src}")
    elif model_name in YOLO_AUTO:
        if not _model_supported(model_name):
            need = ".".join(str(x) for x in YOLO_AUTO[model_name][1])
            raise ValueError(
                f"当前 ultralytics 版本 {_ultralytics_version()} 不支持 "
                f"{model_name}：请执行 pip install -U 'ultralytics>={need}' 升级")
        if model_name in YOLO_YAML_ONLY:
            # 官方不托管 .pt 检查点（如 YOLOv6）：从架构配置从头训练
            yaml_name = _YAML_NAME.get(model_name, f"{model_name}.yaml")
            model = model_cls(yaml_name)
            task.log(f"从架构配置从头训练: {yaml_name}"
                     f"（该系列无官方预训练权重，可上传 .pt 微调）")
        elif pretrained:
            pt_name = _PT_NAME.get(model_name, f"{model_name}.pt")
            model = _load_asset_with_retry(
                lambda: model_cls(_cached_asset(pt_name)), pt_name,
                log=task.log)
            task.log(f"使用官方 COCO 预训练权重: {pt_name}"
                     f"（首次使用自动下载，微调时自动替换分类头）")
        else:
            yaml_name = _YAML_NAME.get(model_name, f"{model_name}.yaml")
            model = model_cls(yaml_name)
            task.log(f"从架构配置从头训练: {yaml_name}")
    elif model_name in YOLOV13_URLS:
        if not _yolov13_supported():
            raise ValueError(_YOLOV13_INSTALL_HINT)
        from app.services import yolov13_arch
        if pretrained:
            pt = _yolov13_weight_path(model_name, log=task.log)
            model = yolov13_arch.load_model(model_cls, pt)
            task.log(f"使用官方 COCO 预训练权重微调: {pt.name}"
                     f"（iMoonLab/yolov13 Release + 内置架构适配）")
        else:
            yaml_p = yolov13_arch.yaml_for(model_name)
            model = yolov13_arch.load_model(model_cls, yaml_p)
            task.log(f"从架构配置从头训练: {yaml_p.name}"
                     f"（项目内置 YOLOv13 架构）")
    elif model_name in YOLO_UPLOAD:
        # 无内置预训练权重且未上传 .pt：回退到默认架构从头训练
        yaml_name = "yolov8n.yaml"
        model = model_cls(yaml_name)
        task.log(f"{model_name} 无内置预训练权重且未上传 .pt："
                 f"回退到默认架构 {yaml_name} 从头训练"
                 f"（如需微调自定义权重，请在「初始权重文件」中选择）")
    else:
        raise ValueError(f"未知 YOLO 模型: {model_name}")

    # YOLO 专属增强参数（受「数据增强」开关与「增强强度」统一控制）
    mosaic = float(params.get("yolo_mosaic", 1.0))
    scale = float(params.get("yolo_scale", 0.5))
    mixup = float(params.get("yolo_mixup", 0.0))
    copy_paste = float(params.get("yolo_copy_paste", 0.1))
    extra_aug = {}
    if not params.get("augment", True):
        # 「数据增强」开关关闭：同时关闭翻转 / 色彩抖动 / 平移等基础增强
        mosaic = scale = mixup = copy_paste = 0.0
        extra_aug = dict(fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0,
                         hsv_v=0.0, translate=0.0)
    else:
        strength = float(params.get("augment_strength", 1.0))
        mosaic = min(1.0, mosaic * strength)
        scale = min(1.0, scale * strength)
        mixup = min(1.0, mixup * strength)
        copy_paste = min(1.0, copy_paste * strength)

    patience = early_stop if early_stop > 0 else epochs  # 0=不启用早停
    # 损失分项权重（缺省 = ultralytics 默认：box 7.5 / cls 0.5 / dfl 1.5）
    _lw = all_weights(params, {"box_w": 7.5, "cls_w": 0.5, "dfl_w": 1.5})
    task.log(f"损失权重: {describe(_lw)}")
    train_args = dict(
        data=str(data_yaml), epochs=epochs, batch=batch_size, imgsz=imgsz,
        device=_map_device(device_str), project=str(save_dir), name="yolo",
        exist_ok=True, seed=seed, workers=num_workers, patience=patience,
        amp=amp, mosaic=mosaic, scale=scale, mixup=mixup,
        copy_paste=copy_paste, lr0=lr, weight_decay=weight_decay,
        box=_lw["box_w"], cls=_lw["cls_w"], dfl=_lw["dfl_w"],
        verbose=True, plots=True, **extra_aug,
    )
    if resume_pt:
        # resume 模式下 ultralytics 以检查点内保存的训练参数为准
        train_args["resume"] = True
    # 优化器 / 动量 / 学习率调度：界面可选，映射为 ultralytics 训练参数
    # （引擎内置支持 SGD/Adam/AdamW/Adamax/NAdam/RAdam/RMSProp/auto；
    #   调度器只支持线性衰减与余弦退火，通过 cos_lr + lrf 控制）
    _OPT_MAP = {"adam": "Adam", "sgd": "SGD", "adamw": "AdamW",
                "adamax": "Adamax", "rmsprop": "RMSProp",
                "nadam": "NAdam", "radam": "RAdam", "auto": "auto"}
    opt = (params.get("optimizer") or "").lower()
    if opt in _OPT_MAP:
        train_args["optimizer"] = _OPT_MAP[opt]
        task.log(f"优化器: {_OPT_MAP[opt]} | 学习率 lr0={lr} | "
                 f"权重衰减={weight_decay}")
        if opt in ("adam", "adamw", "adamax", "nadam", "radam") and lr > 5e-3:
            task.log(f"[警告] {_OPT_MAP[opt]} 常用学习率为 1e-3 量级，当前 "
                     f"lr0={lr} 偏大，如不收敛请下调学习率")
    else:
        note = f"（ultralytics 不支持 {opt}）" if opt else ""
        task.log(f"优化器: 自动，由引擎按模型选择{note}")
    momentum = params.get("momentum")
    if momentum not in (None, ""):
        train_args["momentum"] = float(momentum)
    sched = (params.get("scheduler") or "cosine").lower()
    train_args["cos_lr"] = sched == "cosine"
    task.log(f"学习率调度: {'Cosine 退火' if sched == 'cosine' else '线性衰减'}")

    # 知识蒸馏（ultralytics 引擎原生 DistillationModel：训练时自动把
    # self.model 包装为 student_model + 冻结 teacher_model，neck 特征经
    # 投影层对齐后做 score 加权 L2 蒸馏；val 指标按学生单独计算，
    # 训练产物推理时引擎自动解包学生）
    distill = bool(params.get("distill", False))
    distill_info = None
    if distill:
        if not _distill_supported():
            raise ValueError(
                "当前 ultralytics 引擎不支持知识蒸馏（需 8.4.0+ 内置 "
                "distill_model / dis 参数）：请执行 pip install -U "
                "ultralytics 升级后重启服务")
        student_cls = type(getattr(model, "model", None)).__name__
        if student_cls != "DetectionModel":
            raise ValueError(
                f"学生模型检测头为 {student_cls}：知识蒸馏仅支持 3 尺度 "
                f"Detect 头的 YOLO 系列（RT-DETR 暂不支持）")
        teacher_path, n_teacher, teacher_desc = _resolve_distill_teacher(
            params, log=task.log)
        dis = float(params.get("distill_lambda", 6.0))
        train_args["distill_model"] = teacher_path
        train_args["dis"] = dis
        n_student = sum(p.numel() for p in model.model.parameters())
        distill_info = {"teacher": teacher_desc,
                        "teacher_path": teacher_path,
                        "teacher_params_m": round(n_teacher / 1e6, 2),
                        "student_params_m": round(n_student / 1e6, 2),
                        "dis": dis}
        task.log(f"知识蒸馏已启用 | 学生 {model_name} "
                 f"{n_student / 1e6:.2f}M ← 教师 {teacher_desc} "
                 f"{n_teacher / 1e6:.2f}M（冻结，仅前向提供软目标）| "
                 f"蒸馏权重 dis={dis}")
        task.log("引擎将构建 DistillationModel：教师与学生 neck 特征经投影层"
                 "对齐 + score 加权 L2 蒸馏损失（参数量更少的学生在教师指导下"
                 "训练，性能通常优于同规模从零训练）")

    # 回调：进度 / 指标 / 停止（每轮训练+验证后触发）
    csv_path = save_dir / "metrics.csv"
    _csv_header = ["epoch", "train_loss", "val_loss", "map_50", "map_50_95"]
    if resume_pt:
        # 续训：恢复历史指标并截断 csv（epoch 从检查点续算）
        from app.services.train_common import reload_task_history
        try:
            _ck = torch.load(resume_pt, map_location="cpu", weights_only=False)
            _start = int(_ck.get("epoch", 0) or 0) + 1 \
                if isinstance(_ck, dict) else 1
        except Exception:
            _start = 1
        reload_task_history(task, csv_path, _start, _csv_header)
        task.log(f"续训起点: epoch {_start}")
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_csv_header)

    def _on_fit_epoch_end(trainer):
        if task.stop_event.is_set():
            raise _YoloStop()
        m = dict(getattr(trainer, "metrics", None) or {})
        epoch = int(getattr(trainer, "epoch", 0) or 0) + 1
        # 训练结束后的最终验证会再次触发本回调（epoch 越界），跳过以免重复记录
        if epoch > epochs:
            return
        if task.history and task.history[-1]["epoch"] >= epoch:
            return
        tl = sum(float(v) for k, v in m.items()
                 if k.startswith("train/") and k.endswith("_loss"))
        if not tl:
            # 兼容：8.4+ 此阶段 metrics 不含 train/* 损失，从 tloss（epoch
            # 运行均值，dict 或 tensor）兜底汇总（蒸馏时含 dis_loss）
            tloss = getattr(trainer, "tloss", None)
            if tloss is not None:
                try:
                    if isinstance(tloss, dict):
                        tl = float(sum(float(v) for v in tloss.values()))
                    elif hasattr(tloss, "detach"):
                        tl = float(np.sum(np.asarray(
                            tloss.detach().cpu(), dtype=np.float64)))
                    else:
                        tl = float(np.sum(np.asarray(tloss,
                                                      dtype=np.float64)))
                except Exception:
                    tl = 0.0
        vl = sum(float(v) for k, v in m.items()
                 if k.startswith("val/") and k.endswith("_loss"))
        map50 = m.get("metrics/mAP50(B)")
        map95 = m.get("metrics/mAP50-95(B)")
        task.epoch = epoch
        task.progress = min(100.0, epoch / epochs * 100.0)
        task.metrics = {"train_loss": round(tl, 4), "val_loss": round(vl, 4)}
        if map50 is not None:
            task.metrics["val_map50"] = round(float(map50), 4)
        if map95 is not None:
            task.metrics["val_map50_95"] = round(float(map95), 4)
        row = {"epoch": epoch, "train_loss": round(tl, 4),
               "val_loss": round(vl, 4)}
        if map50 is not None:
            row["val_map50"] = round(float(map50), 4)
        if map95 is not None:
            row["val_map50_95"] = round(float(map95), 4)
        task.history.append(row)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{tl:.4f}", f"{vl:.4f}",
                 f"{float(map50):.4f}" if map50 is not None else "",
                 f"{float(map95):.4f}" if map95 is not None else ""])
        task.log(f"Epoch {epoch}/{epochs} | train_loss {tl:.4f} | "
                 f"val_loss {vl:.4f} | mAP@0.5 {map50} | "
                 f"mAP@0.5:0.95 {map95}")
        save_loss_curves(save_dir, task.history)
        fit = getattr(trainer, "fitness", None)
        if fit is not None:
            fit = float(fit)
            if task.best.get("fitness") is None or fit > task.best["fitness"]:
                task.best = {"epoch": epoch, "fitness": round(fit, 5),
                             **({"mAP50": round(float(map50), 4)}
                                if map50 is not None else {})}
        # 每轮收尾及时释放：gc 循环垃圾 + CUDA 缓存显存归还（在引擎自身的
        # val 清理之外再补齐进程级归还，保障同卡并发任务与训练后推理）
        release_training_memory()

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    # 批内进度日志：ultralytics 的 tqdm 进度条只输出到控制台（\r 覆盖），
    # 这里通过回调把每个 epoch 内的迭代损失（约每 10% 记录一次）写入任务
    # 日志，便于在界面观察训练细节；同时在批末响应停止请求
    _batch_log = {"nb": 0, "every": 1, "i": 0}

    def _on_train_epoch_start(trainer):
        try:
            nb = len(trainer.train_loader)
        except Exception:
            nb = int(getattr(trainer, "nb", 0) or 0)
        _batch_log["nb"] = nb
        _batch_log["every"] = max(1, nb // 10) if nb > 0 else 1
        _batch_log["i"] = 0
        epoch = int(getattr(trainer, "epoch", 0) or 0) + 1
        lr_now = None
        try:
            lr_now = float(trainer.optimizer.param_groups[0]["lr"])
        except Exception:
            pass
        task.log(f"Epoch {epoch}/{epochs} 开始 | 训练批次数 {nb} | "
                 f"每 {_batch_log['every']} 批记录一次损失" +
                 (f" | 当前学习率 {lr_now:.6g}" if lr_now is not None else ""))

    def _on_train_batch_end(trainer):
        if task.stop_event.is_set():
            raise _YoloStop()
        # 内存看门狗：训练中内存持续下降时提前安全停止，避免原生 OOM
        avail = _available_ram_mb()
        if avail is not None and avail < 800:
            raise RuntimeError(
                f"可用内存不足（{avail:.0f}MB < 800MB），为避免整个服务进程"
                f"崩溃已安全停止 YOLO 训练；请减小 batch size / 输入尺寸，"
                f"或关闭其他占用内存的程序后重试（已保存的 last.pt 可续训）")
        # 不同版本 trainer 未必暴露 batch 序号，自行计数更稳
        _batch_log["i"] += 1
        i = _batch_log["i"]
        nb = _batch_log["nb"]
        every = _batch_log["every"]
        if i % every != 0 and i != nb:
            return
        epoch = int(getattr(trainer, "epoch", 0) or 0) + 1
        tloss = getattr(trainer, "tloss", None)
        if tloss is None:
            return
        try:
            items = trainer.label_loss_items(tloss, prefix="train")
            parts = " | ".join(f"{str(k).split('/')[-1]} {float(v):.4f}"
                               for k, v in items.items())
        except Exception:
            parts = ""
        task.log(f"  Epoch {epoch}/{epochs} | iter {i}/{nb}（平均）" +
                 (f" | {parts}" if parts else ""))

    model.add_callback("on_train_epoch_start", _on_train_epoch_start)
    model.add_callback("on_train_batch_end", _on_train_batch_end)

    # 关闭外部实验平台集成回调（wandb/clearml/comet 等）：
    # 若用户环境装有 wandb，ultralytics 会在训练时尝试初始化并可能崩溃
    os.environ.setdefault("WANDB_MODE", "disabled")
    try:
        from ultralytics.utils import callbacks as _ul_cbs
        _ul_cbs.add_integration_callbacks = lambda instance: None
    except Exception:
        pass

    # 执行训练
    stopped = False
    try:
        model.train(**train_args)
    except _YoloStop:
        stopped = True
        task.log("收到停止信号，YOLO 训练已停止（保留最近一轮权重）")

    # 训练产物整理与指标图
    from app.services import eval_metrics as em
    trainer = getattr(model, "trainer", None)
    best_path = Path(trainer.best) if getattr(trainer, "best", None) else None
    last_path = Path(trainer.last) if getattr(trainer, "last", None) else None
    # 权重文件名加模型名前缀（如 yolov8n_best.pt），便于区分；保留
    # ultralytics 原始 best.pt/last.pt 以兼容其原生 resume
    import re as _re
    _tag = _re.sub(r"[^\w\-]", "_", str(model_name)).strip("_") or "yolo"
    best_disp, last_disp = best_path, last_path
    try:
        if best_path and best_path.exists():
            best_disp = best_path.parent / f"{_tag}_{best_path.name}"
            shutil.copy2(str(best_path), str(best_disp))
        if last_path and last_path.exists():
            last_disp = last_path.parent / f"{_tag}_{last_path.name}"
            shutil.copy2(str(last_path), str(last_disp))
    except OSError as e:
        task.log(f"权重前缀副本创建失败（沿用原文件）: {e}")
        best_disp, last_disp = best_path, last_path

    final = {}
    if task.history:
        h = task.history[-1]
        if h.get("val_map50") is not None:
            final["mAP@0.5"] = h["val_map50"]
        if h.get("val_map50_95") is not None:
            final["mAP@0.5:0.95"] = h["val_map50_95"]
    if final:
        em.plot_metrics_summary(final, save_dir / "yolo_metrics.png",
                                "YOLO 头部框检测指标（验证集）")
    if task.history:
        em.plot_loss_curves(task.history, save_dir / "loss_curves.png",
                            ("train_loss", "val_loss"))

    task.result = {"best_weight": str(best_disp) if best_disp else None,
                   "last_weight": str(last_disp) if last_disp else None,
                   "final_map": final, "classes": names,
                   "yolo_run_dir": str(save_dir / "yolo"),
                   "save_dir": str(save_dir)}
    if distill_info:
        task.result["distill"] = distill_info
    task.log(f"训练结束，mAP@0.5={final.get('mAP@0.5')}"
             + (f" | 最优权重: {best_disp}" if best_disp else ""))
    if stopped:
        raise StopIteration()  # 沿用框架约定：由任务管理器标记为「已停止」


# ------------------------- 推理 -------------------------

def _try_font(size=16):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_yolo(model_name, weight_path, pretrained):
    _require_ultralytics()
    model_cls = _model_class_for(model_name)
    if weight_path:
        if str(model_name).lower().startswith("yolov13"):
            from app.services import yolov13_arch
            return yolov13_arch.load_model(model_cls, weight_path)
        return model_cls(weight_path)
    if model_name in YOLO_AUTO:
        if not _model_supported(model_name):
            need = ".".join(str(x) for x in YOLO_AUTO[model_name][1])
            raise ValueError(
                f"当前 ultralytics 版本 {_ultralytics_version()} 不支持 "
                f"{model_name}：请执行 pip install -U 'ultralytics>={need}' 升级")
        if pretrained:
            # COCO 预训练不含 head 类：头部框推理须训练产物/上传权重
            raise ValueError(
                "头部框推理需要权重：请选择训练产物权重、上传 .pt 权重，"
                "或先在训练模式用头部框数据集微调模型"
                "（COCO 预训练模型不含 head 类别）")
        raise ValueError("YOLO 推理需要权重：请选择训练产物权重、"
                         "上传 .pt 权重，或勾选「使用内置预训练模型」")
    if model_name in YOLOV13_URLS:
        if not _yolov13_supported():
            raise ValueError(_YOLOV13_INSTALL_HINT)
        if not pretrained:
            raise ValueError("YOLOv13 推理需要权重：请勾选「使用预训练权重」"
                             "（自动下载官方 COCO 权重），或选择训练产物/"
                             "上传 .pt 权重文件")
        from app.services import yolov13_arch
        return yolov13_arch.load_model(
            model_cls, _yolov13_weight_path(model_name))
    if model_name in YOLO_UPLOAD:
        raise ValueError(f"模型「{model_name}」无内置权重："
                         f"请选择训练产物权重或上传 .pt 权重文件")
    raise ValueError(f"未知 YOLO 模型: {model_name}")


def _predict(model, image: Image.Image, conf_threshold, device):
    """ultralytics 推理（输入 PIL → BGR ndarray），
    返回 (boxes, scores, cls_ids, names)"""
    img = image.convert("RGB")
    bgr = np.asarray(img)[:, :, ::-1].copy()  # ultralytics 约定 BGR 输入
    results = model.predict(source=bgr, conf=float(conf_threshold),
                            device=_map_device(device), verbose=False)
    r = results[0] if results else None
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)
    cls_ids = np.zeros((0,), dtype=np.int64)
    if r is not None and r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    return img, boxes, scores, cls_ids


def headbox_infer(image: Image.Image, model_name, weight_path=None,
                  conf_threshold=0.5, device="cpu", pretrained=True):
    """头部框推理：返回人头数 + 框可视化图 + 检测结果列表

    训练产物 classes.json / 检查点类别名自动读取（未训练过的类别按
    模型内建 names 显示）；结果格式与 density / point 路线对齐
    （count 字段 + point_based=False + box_based=True）。
    """
    model = _load_yolo(model_name, weight_path, pretrained)
    img, boxes, scores, cls_ids = _predict(model, image, conf_threshold,
                                           device)

    # 类别名：优先权重同目录 classes.json（训练时保存），否则模型内建
    names = {}
    if weight_path:
        cj = Path(weight_path).parent / "classes.json"
        if cj.exists():
            try:
                names = {i: n for i, n in enumerate(
                    json.loads(cj.read_text(encoding="utf-8")))}
            except Exception:
                names = {}
    if not names:
        try:
            names = dict(model.names or {})
        except Exception:
            names = {}

    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    font = _try_font(16)
    detections = []
    summary = {}
    for box, score, cid in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = [float(v) for v in box]
        cname = names.get(int(cid), str(cid))
        color = _HEAD_COLOR if cname.lower() in ("head", "class_0") \
            else _PALETTE[int(cid) % len(_PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{cname} {score:.2f}"
        tb = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2],
                       fill=color)
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
        detections.append({"class": cname, "label": int(cid),
                           "score": round(float(score), 4),
                           "box": [round(x1, 1), round(y1, 1),
                                   round(x2, 1), round(y2, 1)]})
        summary[cname] = summary.get(cname, 0) + 1

    buf = io.BytesIO()
    draw_img.save(buf, format="PNG")
    vis_b64 = base64.b64encode(buf.getvalue()).decode()
    return {"count": int(len(detections)), "visualization": vis_b64,
            "detections": detections, "num_detected": len(detections),
            "summary": summary, "conf_threshold": conf_threshold,
            "model": model_name, "box_based": True}


def _save_box_txt(out_dir, stem, boxes, cls_ids, W, H):
    """Save detections as YOLO-format .txt (class_id cx cy w h, normalized)."""
    from app.services.crowd_counting import _unique_output_path
    lines = []
    for box, cid in zip(boxes, cls_ids):
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = ((x1 + x2) / 2.0) / max(W, 1)
        cy = ((y1 + y2) / 2.0) / max(H, 1)
        bw = (x2 - x1) / max(W, 1)
        bh = (y2 - y1) / max(H, 1)
        lines.append(f"{int(cid)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path = _unique_output_path(out_dir, stem, ".txt")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _draw_boxes(img, boxes, scores, cls_ids, names):
    """Draw detection boxes on a copy of the image (RGB PIL Image)."""
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    font = _try_font(16)
    for box, score, cid in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = [float(v) for v in box]
        cname = names.get(int(cid), str(cid))
        color = _HEAD_COLOR if cname.lower() in ("head", "class_0") \
            else _PALETTE[int(cid) % len(_PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{cname} {float(score):.2f}"
        tb = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2],
                       fill=color)
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
    return draw_img


def run_box_test(task, params: dict):
    """Head-box route: evaluate a YOLO weight on a dataset test split.

    Predictions are saved as YOLO-format .txt (normalized cx cy w h) and
    visualizations as PNG overlays, matching the density / point test routes.
    """
    model_name = params.get("model") or "yolov8n"
    weight_path = params.get("weight_path")
    if not weight_path or not Path(weight_path).is_file():
        raise ValueError("测试需要训练产物或上传的权重文件（.pt）")
    root = params["dataset_path"]
    split = (params.get("split") or "test").lower()
    device_str = params.get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    conf_threshold = float(params.get("conf_threshold") or 0.5)
    save_dir = Path(task.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "test_results.csv"

    use_train = split == "train"

    def _load(train):
        try:
            return load_yolo_items(root, train=train)
        except ValueError:
            return load_voc_items(root, train=train)

    try:
        items, class_names = _load(use_train)
    except Exception as exc:
        if use_train:
            raise
        task.log(f"未找到 Test/Val 划分（{exc}），回退到 Train 划分")
        use_train = True
        items, class_names = _load(True)
    task.log(f"头部框测试 | 模型: {model_name} | 权重: {weight_path}")
    task.log(f"数据集: {root} | 划分: {'Train' if use_train else 'Test/Val'} "
             f"| 样本: {len(items)}")
    save_outputs = bool(params.get("save_outputs"))
    pred_dir = png_dir = None
    if save_outputs:
        from app.services.crowd_counting import _test_output_dirs
        pred_dir, png_dir = _test_output_dirs(params, save_dir)
        task.log(f"逐图预测框将保存为 .txt：{pred_dir}")
        task.log(f"逐图可视化将保存为 .png：{png_dir}")

    model = _load_yolo(model_name, weight_path, pretrained=False)
    try:
        names = dict(model.names or {}) or class_names
    except Exception:
        names = class_names

    rows, errs = [], []
    saved = saved_png = 0
    t0 = time.time()
    for i, (img_p, anns) in enumerate(items, 1):
        if task.stop_event.is_set():
            raise StopIteration()
        gt_count = float(len(anns))
        img = Image.open(img_p).convert("RGB")
        W, H = img.size
        _, boxes, scores, cls_ids = _predict(model, img, conf_threshold,
                                             device_str)
        pred_count = float(len(boxes))
        stem = Path(img_p).stem
        if pred_dir is not None:
            _save_box_txt(pred_dir, stem, boxes, cls_ids, W, H)
            vis_img = _draw_boxes(img, boxes, scores, cls_ids, names)
            from app.services.crowd_counting import _unique_output_path
            out_png = _unique_output_path(png_dir, stem, ".png")
            vis_img.save(out_png, format="PNG")
            saved += 1
            saved_png += 1
        err = pred_count - gt_count
        errs.append(err)
        rows.append({"name": stem, "pred": round(pred_count, 4),
                     "gt": round(gt_count, 4), "error": round(err, 4)})
        task.progress = i / len(items) * 100.0
        if i % 20 == 0 or i == len(items):
            task.log(f"  [{i}/{len(items)}] pred={pred_count:.0f} "
                     f"gt={gt_count:.0f} | 耗时 {time.time() - t0:.1f}s")
    if not errs:
        raise ValueError("测试集没有有效样本")
    arr = np.asarray(errs, dtype=np.float64)
    mae = round(float(np.mean(np.abs(arr))), 3)
    rmse = round(float(np.sqrt(np.mean(arr ** 2))), 3)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "pred", "gt", "error"])
        w.writeheader()
        w.writerows(rows)
    task.progress = 100.0
    task.result = {"mae": mae, "rmse": rmse, "mse": rmse,
                   "num_images": len(errs), "model": model_name,
                   "dataset": str(root), "patch_size": 0,
                   "per_image_csv": str(csv_path), "save_dir": str(save_dir),
                   "saved_outputs": saved, "saved_pngs": saved_png,
                   "point_format": "",
                   "pred_dir": str(pred_dir) if pred_dir is not None else "",
                   "png_dir": str(png_dir) if png_dir is not None else ""}
    task.log(f"测试完成：MAE={mae:.3f} | RMSE={rmse:.3f} | "
             f"样本={len(errs)} | 逐图结果={csv_path}")
    if pred_dir is not None:
        task.log(f"已保存 {saved} 个预测框文件（.txt）到 {pred_dir}")
        task.log(f"已保存 {saved_png} 张可视化结果（.png）到 {png_dir}")
