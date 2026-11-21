# -*- coding: utf-8 -*-
"""权重格式推断模块（ONNX 部署导出的前置能力，独立设计为一个模块）

给定任意权重文件（.pth / .pt），推断它属于哪条技术路线、哪个模型架构、
推荐输入尺寸等部署所需信息。推断按可靠性优先级依次尝试：

1. 训练产物同目录 params.json（本框架训练任务 100% 命中，task+model 精确）
2. ultralytics 检查点（.pt 内含完整 nn.Module + train_args，YOLO 头部框路线）
3. state_dict 键名指纹（架构级识别：P2R encoders / P2PNet regression+
   classification / 密度网 head.convs / 参考项目 feblock 等）

用途：
- 部署面板选择权重后自动显示模型类型（前端 POST /api/crowd/onnx/format）
- ONNX 导出任务据此选择正确的模型构建器 / 预处理 / 片段代码模板；
  密度网具体骨干无法从指纹唯一定名时标记 model="auto"，由导出任务
  逐个候选架构探测（严格加载校验）补全
"""
import json
from pathlib import Path

import torch

# 各路线默认输入尺寸与对齐约束（P2PNet FPN 三级严格 ×2：16 倍数；
# P2R /4 下采样 + 两次 ×2 上采样：32 倍数；密度网 /8：8 的倍数即可）
ROUTE_DEFAULTS = {
    "crowd_density": {"imgsz": 384, "multiple": 8},
    "crowd_point": {"imgsz": 384, "multiple": 16},   # p2pnet / cp
    "crowd_point_p2r": {"imgsz": 384, "multiple": 32},
    "crowd_box": {"imgsz": 640, "multiple": 32},
    "crowd_sdg": {"imgsz": 320, "multiple": 16},      # 单域泛化
}

ROUTE_LABELS = {
    "crowd_density": "人群计数 · 密度图",
    "crowd_point": "人群计数 · 点预测",
    "crowd_box": "人群计数 · 头部框（YOLO）",
    "crowd_sdg": "人群计数 · 单域泛化",
}


def safe_load_ckpt(path):
    """加载 checkpoint：优先 weights_only=True（纯 state_dict 安全且够用），
    失败（ultralytics .pt 内含完整 pickled 模块）再回退 weights_only=False；
    再兼容不支持 weights_only 关键字的旧版 torch。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # 旧版 torch 无 weights_only 参数
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _extract_state_dict(ck):
    """从 checkpoint 提取 state_dict（兼容 model_state_dict / state_dict /
    "model" 三种键与 DP/DDP 的 module. 前缀），返回键集合或 None。"""
    sd = None
    if isinstance(ck, dict):
        sd = ck.get("model_state_dict") or ck.get("state_dict")
        if sd is None and isinstance(ck.get("model"), dict):
            sd = ck["model"]
    if not isinstance(sd, dict) or not sd:
        return None
    return {k[7:] if k.startswith("module.") else k for k in sd.keys()}


def _from_params_json(weight_path: Path):
    """路径 1：训练产物同目录 params.json（本框架训练任务）"""
    for base in (weight_path.parent, weight_path.parent.parent):
        pf = base / "params.json"
        if pf.is_file():
            try:
                p = json.loads(pf.read_text(encoding="utf-8"))
            except Exception:
                continue
            task = p.get("task")
            model = p.get("model")
            if task == "crowd_sdg":
                return {"source": "params.json", "task": "crowd_sdg",
                        "model": model, "backbone": p.get("sdg_backbone"),
                        "num_clusters": p.get("sdg_num_clusters")}
            if task in ("crowd_density", "crowd_point", "crowd_box",
                        "crowd_counting"):
                if task == "crowd_counting":       # 旧版合并 key：按模型定路线
                    task = ("crowd_point"
                            if model in ("p2pnet", "p2r", "cp", "cu")
                            else "crowd_density")
                return {"source": "params.json", "task": task, "model": model}
    return None


def _from_ultralytics_ckpt(weight_path: Path, ck):
    """路径 2：ultralytics 检查点（'model' 为完整 nn.Module / 含 train_args）"""
    m = ck.get("model") if isinstance(ck, dict) else None
    has_train_args = isinstance(ck, dict) and isinstance(ck.get("train_args"),
                                                          dict)
    if not (isinstance(m, torch.nn.Module) or
            (has_train_args and m is not None)):
        return None
    info = {"source": "ultralytics", "task": "crowd_box"}
    model_name = ""
    try:
        model_name = str((m.args or {}).get("model", "")) if hasattr(
            m, "args") else ""
    except Exception:
        pass
    if not model_name and has_train_args:
        model_name = str(ck["train_args"].get("model", ""))
    info["model"] = Path(model_name).stem if model_name else "yolo"
    try:
        info["nc"] = int(getattr(m, "nc", 0) or 0)
    except Exception:
        pass
    names = getattr(m, "names", None)
    if isinstance(names, dict):
        info["classes"] = [str(v) for _, v in sorted(names.items())]
    elif isinstance(names, (list, tuple)):
        info["classes"] = [str(v) for v in names]
    try:    # DistillationModel 蒸馏产物：解包学生
        if type(m).__name__ == "DistillationModel" and getattr(
                m, "student_model", None) is not None:
            info["distilled"] = True
            info["model"] = info["model"] or "yolo"
    except Exception:
        pass
    return info


def _from_fingerprint(keys):
    """路径 3：state_dict 键名指纹（架构级）"""
    if keys is None:
        return None
    has = lambda *prefixes: any(k.startswith(p) for p in prefixes
                                for k in keys)  # noqa: E731
    # --- 点预测：P2R（encoders）/ CU（uncertainty 分支为独有键）/
    #     P2PNet 家族（p2pnet 与 Consistent-Point 架构完全相同，IUC 为基于
    #     分类得分的损失加权，无额外分支）---
    if has("encoders."):
        return {"source": "state-dict", "task": "crowd_point", "model": "p2r"}
    if has("uncertainty."):
        return {"source": "state-dict", "task": "crowd_point", "model": "cu"}
    if has("regression.", "classification.") and has("fpn.", "backbone."):
        return {"source": "state-dict", "task": "crowd_point",
                "model": "p2pnet"}
    # --- 头部框：YOLO 家族扁平层编号（model.0. / model.22. + cv2/cv3 头）---
    if has("model.22.") or (has("model.0.") and
                            any(".cv2." in k or ".cv3." in k for k in keys)):
        return {"source": "state-dict", "task": "crowd_box", "model": "yolo"}
    # --- 密度图：参考项目（feblock / conv1+layer 直挂 backbone）---
    if has("backbone.feblock"):
        return {"source": "state-dict", "task": "crowd_density",
                "model": "crowdvgg"}
    if has("backbone.conv1.") and has("backbone.layer"):
        return {"source": "state-dict", "task": "crowd_density",
                "model": "crowdresnet"}
    # --- 密度图：框架内置 CrowdNet（backbone 为 nn.Sequential + head.convs）；
    #     具体骨干（ghostnet 系 / torchvision）无法由前缀唯一定名 -> auto ---
    if has("head.convs.") and has("backbone."):
        return {"source": "state-dict", "task": "crowd_density",
                "model": "auto"}
    return None


def infer_weight_format(weight_path):
    """推断权重的模型格式 → 部署信息 dict。

    Returns:
        {"route": crowd_density|crowd_point|crowd_box,
         "model": str,           # "auto" 表示需在导出任务中探测补全
         "source": str,          # params.json / ultralytics / state-dict
         "imgsz": int, "multiple": int,   # 建议输入尺寸与对齐约束
         "nc"/"classes"/"distilled"/"model_label": ...,
         "detail": str}          # 人读结论
    Raises:
        ValueError: 文件不存在 / 无法识别（附排查提示）
    """
    p = Path(weight_path)
    if not p.is_file():
        raise ValueError(f"权重文件不存在: {weight_path}")

    info = _from_params_json(p)
    if info is None:
        try:
            ck = safe_load_ckpt(p)
        except Exception as e:
            raise ValueError(
                f"权重加载失败（可能是非 PyTorch checkpoint 或文件损坏）: {e}")
        # 单域泛化：训练产物 checkpoint 内嵌 sdg_meta（优先于架构指纹）
        if isinstance(ck, dict) and isinstance(ck.get("sdg_meta"), dict):
            m = ck["sdg_meta"]
            info = {"source": "sdg_meta", "task": "crowd_sdg",
                    "model": m.get("model_name", "sdg_dccus"),
                    "backbone": m.get("backbone"),
                    "num_clusters": m.get("num_clusters"),
                    "log_para": m.get("log_para")}
        if info is None:
            info = _from_ultralytics_ckpt(p, ck)
        if info is None:
            info = _from_fingerprint(_extract_state_dict(ck))

    if info is None:
        raise ValueError(
            "无法识别该权重的模型格式（三条路线的指纹均未命中）：\n"
            "  - 本框架训练产物（同目录应有 params.json）\n"
            "  - ultralytics YOLO .pt 检查点\n"
            "  - 密度网 / P2PNet / P2R / Consistent-Point 的 state_dict\n"
            "若为手工导出的权重，请确认 state_dict 结构未被改写")

    route = info["task"]
    defaults = (ROUTE_DEFAULTS["crowd_point_p2r"]
                if (route == "crowd_point" and info["model"] == "p2r")
                else ROUTE_DEFAULTS[route])
    out = {"route": route, "model": info["model"],
           "source": info["source"],
           "imgsz": defaults["imgsz"], "multiple": defaults["multiple"],
           "route_label": ROUTE_LABELS[route]}
    for k in ("nc", "classes", "distilled", "backbone", "num_clusters",
              "log_para"):
        if k in info and info[k] is not None:
            out[k] = info[k]
    if route == "crowd_sdg":
        out["sdg_meta"] = {"model_name": info["model"],
                           "backbone": info.get("backbone"),
                           "num_clusters": info.get("num_clusters") or 4,
                           "log_para": info.get("log_para") or 1000}
    if route == "crowd_point" and info["model"] == "p2pnet":
        out["model_label"] = "P2PNet / Consistent-Point（架构相同）"
    elif info["model"] == "auto":
        out["model_label"] = "密度网（具体骨干导出时自动探测）"
    else:
        out["model_label"] = str(info["model"])
    out["detail"] = "{route} | 模型 {model} | 依据 {source}".format(
        route=out["route_label"], model=out["model_label"],
        source={"params.json": "同目录 params.json",
                "ultralytics": "ultralytics 检查点",
                "sdg_meta": "checkpoint 内 sdg_meta",
                "state-dict": "state_dict 指纹"}[info["source"]])
    return out


# ------------------------- 密度网骨干探测（导出任务用） -------------------------

def probe_density_model(weight_path, log=None):
    """密度网具体骨干探测：逐个候选架构构建（不加载预训练）并做 ≥95% 键
    匹配校验，返回第一个命中的模型名（本框架密度网骨干种类多且键名
    前缀雷同（nn.Sequential backbone），指纹无法唯一定名）。"""
    from app.services.crowd_counting import (_build_crowd_model,
                                             _load_weights_compat,
                                             TV_CROWD_BACKBONES,
                                             REFERENCE_CROWD_MODELS)
    from app.core.registry import custom_models_for
    candidates = (["ghostnetv2_10", "ghostnetv2_13", "ghostnetv2_16",
                   "ghostnet"] + list(REFERENCE_CROWD_MODELS) +
                  list(TV_CROWD_BACKBONES) +
                  [m["name"] for m in custom_models_for("crowd_density")])
    log = log or (lambda *_: None)
    last_err = None
    for name in candidates:
        log(f"骨干探测: 尝试 {name} ...")
        try:
            model = _build_crowd_model(name, pretrained=False)
            _load_weights_compat(model, weight_path,
                                 context=f"骨干探测[{name}]")
            log(f"骨干探测命中: {name}")
            return name
        except Exception as e:
            last_err = e
            continue
    raise ValueError(
        "密度网骨干探测失败：候选架构均无法匹配该权重"
        f"（{len(candidates)} 个候选，最后错误: {last_err}）。"
        "请确认权重为本框架训练产物或 CrowdNet 结构")
