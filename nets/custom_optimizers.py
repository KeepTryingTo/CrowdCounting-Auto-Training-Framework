# -*- coding: utf-8 -*-
"""自定义优化器示例与注册入口。

训练「优化器」下拉框中 custom: 开头的即来自此处。
新增：在 nets/custom_optimizers/ 目录建 .py，或直接在此文件追加。

注册约定: factory(parameters, lr, momentum=0.9, weight_decay=0.0, **kw)
          → torch.optim.Optimizer 实例

界面传入的 momentum / weight_decay 会作为关键字参数传给 factory。
"""
from app.core.registry import register_custom_optimizer


# ---------------------------------------------------------------------------
# 示例 1：带 Nesterov 动量的 SGD（torch 内置但界面未单列，演示注册方式）
# ---------------------------------------------------------------------------
@register_custom_optimizer("sgd_nesterov",
                           desc="SGD + Nesterov 动量（前瞻梯度）")
def sgd_nesterov_factory(parameters, lr, momentum=0.9, weight_decay=0.0, **kw):
    import torch
    return torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                           weight_decay=weight_decay, nesterov=True)


# ---------------------------------------------------------------------------
# 示例 2：带差异化学习率的 Adam（第一层参数用 lr×0.1，其余用 lr）
# ---------------------------------------------------------------------------
@register_custom_optimizer("adam_layerwise",
                           desc="Adam 分层学习率（第一层 ×0.1，骨干常用技巧）")
def adam_layerwise_factory(parameters, lr, momentum=0.9, weight_decay=0.0, **kw):
    import torch

    class LayerwiseAdam(torch.optim.Adam):
        """按参数名前缀分组：第一层（backbone.0 / features.0）lr×0.1"""

        def __init__(self, params, lr, weight_decay=0.0):
            groups = []
            for name, p in params:
                group_lr = lr * 0.1 if name.split(".")[0] in (
                    "backbone", "features", "encoder") else lr
                # 按 lr 去重成组
                for g in groups:
                    if g["lr"] == group_lr:
                        g["params"].append(p)
                        break
                else:
                    groups.append({"params": [p], "lr": group_lr,
                                  "weight_decay": weight_decay})
            super().__init__(groups, lr=lr)

    # parameters 可能是 generator 或 list of (name, param)
    if hasattr(parameters, "__next__"):
        parameters = list(parameters)
    # 判断是否为 named_parameters 的结果
    if parameters and isinstance(parameters[0], tuple):
        return LayerwiseAdam(parameters, lr, weight_decay)
    # 普通 parameters（无名）：退化为标准 Adam
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# 示例 3：RAdam（Rectified Adam，自适应方差校正，防初期发散）
# ---------------------------------------------------------------------------
@register_custom_optimizer("radam",
                           desc="RAdam（方差校正 Adam，训练初期更稳定）")
def radam_factory(parameters, lr, momentum=0.9, weight_decay=0.0, **kw):
    import torch
    # 如果环境有 torch.optim.RAdam 就直接用（PyTorch 2.x 已内置）
    if hasattr(torch.optim, "RAdam"):
        return torch.optim.RAdam(parameters, lr=lr,
                                  weight_decay=weight_decay)
    # 否则退退化为标准 Adam（加提示到参数名）
    import warnings
    warnings.warn("当前 torch 无 RAdam，退化为 Adam；"
                  "PyTorch ≥ 2.0 自带 torch.optim.RAdam")
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
