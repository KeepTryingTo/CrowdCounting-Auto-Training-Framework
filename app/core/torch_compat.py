# -*- coding: utf-8 -*-
"""torch.load 旧版本兼容补丁（跨 conda 环境运行）。

本框架会在多个 conda 环境间切换（例如 YOLOv3 环境为 Python 3.8 +
torch 1.12），而业务代码统一使用 torch>=2.x 引入的
``torch.load(..., weights_only=...)``。torch 1.12 的 ``load`` 不认识该
关键字（会转发给 Unpickler 并抛
``TypeError: 'weights_only' is an invalid keyword argument for Unpickler()``），
导致加载权重 / 断点续训直接失败。

这里做一次幂等包装：仅当当前 torch 的 ``load`` 不支持 ``weights_only``
时，剥掉该关键字后再调用。旧版 torch 的默认语义即为
``weights_only=False``，因此与显式传 ``False`` 行为一致；新版 torch 原生
支持该参数，不做任何改动。
"""
import inspect

import torch


def _patch() -> bool:
    """给不支持 weights_only 的旧版 torch 打补丁；返回是否打了补丁。"""
    try:
        params = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return False
    if "weights_only" in params:
        return False                      # 新版 torch：原生支持，无需补丁
    orig = torch.load
    if getattr(orig, "_crowd_torch_compat", False):
        return False                      # 已打过补丁（热重载等场景）

    def load_compat(*args, **kwargs):
        kwargs.pop("weights_only", None)
        return orig(*args, **kwargs)

    load_compat.__name__ = getattr(orig, "__name__", "load")
    load_compat.__doc__ = getattr(orig, "__doc__", None)
    load_compat._crowd_torch_compat = True
    torch.load = load_compat
    return True


TORCH_LOAD_WEIGHTS_ONLY_PATCHED = _patch()
