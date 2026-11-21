# -*- coding: utf-8 -*-
"""torch 优化器统一构建（含自定义优化器分发）

内置优化器：adam / sgd / adamw / rmsprop / adagrad / adadelta / adamax / asgd
自定义优化器：注册于 nets/custom_optimizers.py（@register_custom_optimizer）

params 认键：optimizer（优化器名）、momentum、weight_decay；
界面各训练表单都有这三项输入。框架 import 放在函数里。
"""

# 内置优化器清单（界面下拉用）
_BUILTIN_OPTIMIZERS = ["adam", "sgd", "adamw", "rmsprop", "adagrad",
                       "adadelta", "adamax", "asgd"]

# 各优化器的推荐默认超参（前端切换优化器时联动填 momentum / weight_decay）
OPTIMIZER_DEFAULTS = {
    "adam":     {"momentum": 0.9,  "weight_decay": 0.0},
    "sgd":      {"momentum": 0.9,  "weight_decay": 5e-4},
    "adamw":    {"momentum": 0.9,  "weight_decay": 1e-2},
    "rmsprop":  {"momentum": 0.9,  "weight_decay": 0.0},
    "adagrad":  {"momentum": 0.0,  "weight_decay": 0.0},
    "adadelta": {"momentum": 0.0,  "weight_decay": 0.0},
    "adamax":   {"momentum": 0.9,  "weight_decay": 0.0},
    "asgd":     {"momentum": 0.0,  "weight_decay": 1e-4},
}


def _pick(params: dict, key: str, default):
    """从 params 取值；None / 空串视为未设置（回退默认）"""
    if not isinstance(params, dict):
        return default
    v = params.get(key, default)
    return default if v is None or v == "" else v


def _pick_opt_default(params, name, key, call_default):
    """优先级：params 显式值 > 优化器推荐默认 > 调用点默认"""
    if isinstance(params, dict):
        v = params.get(key)
        if v is not None and v != "":
            return v
    return OPTIMIZER_DEFAULTS.get(name, {}).get(key, call_default)


def torch_optimizers() -> list:
    """PyTorch 可选优化器（界面下拉用，含自定义注册的）"""
    from app.core.registry import OPTIMIZER_REGISTRY
    return _BUILTIN_OPTIMIZERS + [f"custom:{k}" for k in OPTIMIZER_REGISTRY]


def build_torch_optimizer(params, parameters, lr, weight_decay=0.0,
                          momentum=0.9, adam_kwargs=None):
    """构建优化器。

    支持 8 种内置 PyTorch 优化器 + custom:* 自定义优化器。
    adam_kwargs 并进 Adam/AdamW 构造参数。
    """
    import torch
    name = str(_pick(params, "optimizer", "adam")).lower()

    # 自定义优化器（custom: 前缀或注册名）
    if name.startswith("custom:") or (
            not name.startswith(("adam", "sgd", "rmsprop", "adagrad",
                                "adadelta", "adamax", "asgd"))
            and name != "adamw"):
        from app.core.registry import OPTIMIZER_REGISTRY
        key = name.split(":", 1)[1] if ":" in name else name
        if key in OPTIMIZER_REGISTRY:
            kw = {"momentum": float(_pick_opt_default(params, key, "momentum",
                                                      momentum)),
                  "weight_decay": float(_pick_opt_default(
                      params, key, "weight_decay", weight_decay))}
            return OPTIMIZER_REGISTRY[key]["factory"](
                parameters, lr, **kw)
        # 未注册 → 回退 adam
        name = "adam"

    mom = float(_pick_opt_default(params, name, "momentum", momentum))
    wd = float(_pick_opt_default(params, name, "weight_decay", weight_decay))
    kw = dict(adam_kwargs or {})

    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=mom,
                               weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd, **kw)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, momentum=mom,
                                    weight_decay=wd)
    if name == "adagrad":
        return torch.optim.Adagrad(parameters, lr=lr,
                                   weight_decay=wd)
    if name == "adadelta":
        return torch.optim.Adadelta(parameters, lr=lr,
                                     weight_decay=wd)
    if name == "adamax":
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=wd, **kw)
    if name == "asgd":
        return torch.optim.ASGD(parameters, lr=lr, weight_decay=wd)
    # 默认 adam
    return torch.optim.Adam(parameters, lr=lr, weight_decay=wd, **kw)
