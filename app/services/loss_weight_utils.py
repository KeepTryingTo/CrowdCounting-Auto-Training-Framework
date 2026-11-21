# -*- coding: utf-8 -*-
"""损失分项权重工具。

训练界面可按算法展示“损失权重”输入（缺省值 = 原论文/源代码设置），前端以
``loss_weights`` 字典提交；后端在构建 criterion 后调用本模块把权重写入
criterion.weight_dict / matcher 代价，或在各自训练循环中直接使用。
"""

# 界面 key → criterion.weight_dict 中的候选键名
ALIASES = {
    "ce_w": ("loss_ce",),
    "points_w": ("loss_points", "loss_point"),
    "bbox_w": ("loss_bbox",),
    "giou_w": ("loss_giou",),
}


def all_weights(params, defaults: dict) -> dict:
    """合并默认权重与用户提交的 loss_weights（非法值回退默认）"""
    out = {k: float(v) for k, v in defaults.items()}
    given = (params or {}).get("loss_weights") or {}
    if not isinstance(given, dict):
        return out
    for key, dflt in defaults.items():
        v = given.get(key)
        if v in (None, ""):
            continue
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            out[key] = float(dflt)
    return out


def get_weight(params, key: str, default: float) -> float:
    """读取单个权重（缺省返回算法默认值）"""
    try:
        v = ((params or {}).get("loss_weights") or {}).get(key)
        return float(default if v in (None, "") else v)
    except (TypeError, ValueError):
        return float(default)


def aux_weight(params) -> float:
    """自定义附加损失的整体权重（默认 1.0）"""
    return get_weight(params, "aux_w", 1.0)


def apply_criterion_weights(criterion, weights: dict) -> int:
    """把 ce_w/points_w/bbox_w/giou_w 覆盖到 criterion.weight_dict

    返回实际覆盖的键数量；criterion 无 weight_dict 时返回 0。
    """
    wd = getattr(criterion, "weight_dict", None)
    if not isinstance(wd, dict):
        return 0
    n = 0
    for key, aliases in ALIASES.items():
        if key not in weights:
            continue
        for alias in aliases:
            if alias in wd:
                wd[alias] = float(weights[key])
                n += 1
    return n


def apply_matcher_costs(criterion, weights: dict) -> int:
    """覆盖匈牙利匹配代价权重（cost_class / cost_point / cost_bbox / cost_giou）"""
    matcher = getattr(criterion, "matcher", None)
    if matcher is None:
        return 0
    n = 0
    for key in ("cost_class", "cost_point", "cost_bbox", "cost_giou"):
        if key in weights and hasattr(matcher, key):
            setattr(matcher, key, float(weights[key]))
            n += 1
    return n


def describe(weights: dict) -> str:
    """权重摘要（写日志用）"""
    return " · ".join(f"{k}={v:g}" for k, v in weights.items())
