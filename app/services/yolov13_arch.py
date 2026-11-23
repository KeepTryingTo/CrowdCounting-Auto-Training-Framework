# -*- coding: utf-8 -*-
"""YOLOv13 架构适配层（参考 iMoonLab/yolov13，供新版 ultralytics 使用）。

官方 YOLOv13 依赖超图增强 HyperACE / FullPAD / DS-C3k2 等自定义模块，
标准 ultralytics（如 8.4.x）未内置。本模块把这些模块注册到
``ultralytics.nn.modules.block`` / ``conv``，并提供一个只针对
yolov13.yaml 的 parse_model，从而复用已安装的 ultralytics 引擎。
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Bottleneck, C2f, C3, Conv

YOLOV13_YAML = Path(__file__).resolve().parents[2] / "third_party" / "yolov13" / "yolov13.yaml"


def yaml_for(model_name: str) -> Path:
    """按模型名取 scale 对应的 yaml（yolov13n/s/l/x.yaml）。"""
    p = YOLOV13_YAML.with_name(f"{model_name}.yaml")
    return p if p.is_file() else YOLOV13_YAML

_installed = False
_originals = {}


class DSConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=k, stride=s,
                            padding=p, dilation=d, groups=c_in, bias=bias)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0)
              for _ in range(n)))


class AAttn(nn.Module):
    """Area-attention module used by ABlock."""

    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)
        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)
        q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        max_attn = attn.max(dim=-1, keepdim=True).values
        exp_attn = torch.exp(attn - max_attn)
        attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
        x = (v @ attn.transpose(-2, -1)).permute(0, 3, 1, 2)
        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.proj(x + pp)


class ABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()
        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1),
                                 Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False,
                 mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        num_heads = c_ // 32
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones(c2),
                                  requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area)
                            for _ in range(2)))
            if a2 else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n))

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))


class DSBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
        self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,
                 k1=3, k2=5, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(DSBottleneck(c_, c_, shortcut=shortcut, e=1.0,
                           k1=k1, k2=k2, d2=d2) for _ in range(n)))


class DSC3k2(C2f):
    def __init__(self, c1, c2, n=1, dsc3k=False, e=0.5, g=1,
                 shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0,
                      k1=k1, k2=k2, d2=d2) for _ in range(n))
        else:
            self.m = nn.ModuleList(
                DSBottleneck(self.c, self.c, shortcut=shortcut, e=1.0,
                             k1=k1, k2=k2, d2=d2) for _ in range(n))


class AdaHyperedgeGen(nn.Module):
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1,
                 context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context
        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim,
                                         num_hyperedges * node_dim)
        else:
            raise ValueError(f"Unsupported context '{context}'.")
        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)
        else:
            context_cat = torch.cat([X.mean(dim=1), X.max(dim=1).values], dim=-1)
        prototype_offsets = self.context_net(context_cat).view(
            B, self.num_hyperedges, D)
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets
        X_proj = self.pre_head_proj(X)
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads,
                                      self.head_dim).permute(0, 2, 1, 3)
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(
            B * self.num_heads, self.num_hyperedges,
            self.head_dim).transpose(1, 2)
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N,
                             self.num_hyperedges).mean(dim=1)
        return F.softmax(self.dropout(logits), dim=1)


class AdaHGConv(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4,
                 dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges,
                                              num_heads, dropout, context)
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                       nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                       nn.GELU())

    def forward(self, X):
        A = self.edge_generator(X)
        He = self.edge_proj(torch.bmm(A.transpose(1, 2), X))
        X_new = self.node_proj(torch.bmm(A, He))
        return X_new + X


class AdaHGComputation(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8,
                 dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(embed_dim=embed_dim,
                              num_hyperedges=num_hyperedges,
                              num_heads=num_heads, dropout=dropout,
                              context=context)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = self.hgnn(x.flatten(2).transpose(1, 2))
        return tokens.transpose(1, 2).view(B, C, H, W)


class C3AH(nn.Module):
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(embed_dim=c_,
                                  num_hyperedges=num_hyperedges,
                                  num_heads=num_heads, dropout=0.1,
                                  context=context)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FuseModule(nn.Module):
    def __init__(self, c_in, channel_adjust):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_out = Conv(4 * c_in if channel_adjust else 3 * c_in,
                             c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        return self.conv_out(torch.cat([x1_ds, x[1], x3_up], dim=1))


class HyperACE(nn.Module):
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True,
                 shortcut=False, e1=0.5, e2=1, context="both",
                 channel_adjust=True):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k
            else DSBottleneck(self.c, self.c, shortcut=shortcut)
            for _ in range(n))
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, channel_adjust=True):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.channel_adjust = (Conv(in_channels, in_channels * 2, 1)
                               if channel_adjust else nn.Identity())

    def forward(self, x):
        return self.channel_adjust(self.downsample(x))


class FullPAD_Tunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x[0] + self.gate * x[1]


_CUSTOM = ("AAttn", "ABlock", "A2C2f", "DSBottleneck", "DSC3k", "DSC3k2",
           "AdaHyperedgeGen", "AdaHGConv", "AdaHGComputation", "C3AH",
           "FuseModule", "HyperACE", "DownsampleConv", "FullPAD_Tunnel", "C3k")


def install():
    """把自定义模块临时注入 ultralytics 定义模块（供官方 .pt 反序列化）。"""
    global _installed
    if _installed:
        return
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.modules.conv as conv
    g = globals()
    for mod, name in ([(block, n) for n in _CUSTOM]
                      + [(modules, n) for n in _CUSTOM]
                      + [(conv, "DSConv"), (modules, "DSConv")]):
        _originals[(mod.__name__, name)] = getattr(mod, name, _MISSING)
        setattr(mod, name, g[name])
    _installed = True


_MISSING = object()


def uninstall():
    """恢复 ultralytics 原有类，避免影响标准 YOLO 模型的反序列化。"""
    global _installed
    if not _installed:
        return
    import ultralytics.nn.modules as modules  # noqa: F401
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.modules.conv as conv
    mods = {"ultralytics.nn.modules": modules,
            "ultralytics.nn.modules.block": block,
            "ultralytics.nn.modules.conv": conv}
    for (mod_name, name), old in list(_originals.items()):
        mod = mods.get(mod_name)
        if mod is None:
            continue
        if old is _MISSING:
            if hasattr(mod, name):
                delattr(mod, name)
        else:
            setattr(mod, name, old)
    _originals.clear()
    _installed = False


def available():
    try:
        import ultralytics  # noqa: F401
        return YOLOV13_YAML.is_file()
    except Exception:
        return False


def parse_model_v13(d, ch, verbose=False):
    """只处理 yolov13.yaml 的模块表（避免影响其它 ultralytics 模型）。"""
    import ast
    from ultralytics.nn.modules import Concat, Detect
    from ultralytics.nn.tasks import make_divisible

    max_channels = float("inf")
    nc, scales = d.get("nc"), d.get("scales")
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales))
        depth, width, max_channels = scales[scale]
    else:
        depth = width = 1.0
    scale = str(scale or "")

    module_map = {c.__name__: c for c in (
        Conv, Concat, Detect, A2C2f, DSC3k2, DSConv, HyperACE,
        DownsampleConv, FullPAD_Tunnel)}
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    legacy = True

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        args = list(args)
        m = getattr(torch.nn, m[3:]) if m.startswith("nn.") else module_map[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = ast.literal_eval(a)
                except Exception:
                    pass
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        kwargs = {}

        if m in (Conv, DSC3k2, DSConv):
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m is DSC3k2:
                args.insert(2, n)
                n = 1
                legacy = False
                if scale in "lx":
                    args[3] = True
        elif m is A2C2f:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            args.insert(2, n)
            n = 1
            legacy = False
            if scale in "lx":
                args.extend([True, 1.5])
        elif m is HyperACE:
            legacy = False
            c1 = ch[f[1]]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            he = args[1]
            if scale == "n":
                he = int(he * 0.5)
            elif scale == "x":
                he = int(he * 1.5)
            args = [c1, c2, n, he, *args[2:]]
            n = 1
            if scale in "lx":
                args.append(False)
        elif m is DownsampleConv:
            c1 = ch[f]
            c2 = c1 * 2
            args = [c1]
            if scale in "lx":
                args.append(False)
                c2 = c1
        elif m is FullPAD_Tunnel:
            c2 = ch[f[0]]
            args = []
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            Detect.legacy = legacy
            args = [nc]
            kwargs = {"ch": [ch[x] for x in f]}
            c2 = ch[f[0]]
        else:
            c2 = ch[f]

        m_ = (nn.Sequential(*(m(*args, **kwargs) for _ in range(n)))
              if n > 1 else m(*args, **kwargs))
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        save.extend(x % i for x in ([f] if isinstance(f, int) else f)
                    if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


_orig_parse_model = None


def _dispatch_parse_model(d, ch, verbose=True):
    """按 yaml 来源分派：yolov13 用自定义解析器，其它模型保持原逻辑。"""
    src = str(d.get("yaml_file", "")).lower()
    if "yolov13" in src:
        return parse_model_v13(d, ch, verbose)
    return _orig_parse_model(d, ch, verbose)


def _ensure_dispatch():
    global _orig_parse_model
    import ultralytics.nn.tasks as tasks
    if _orig_parse_model is None and tasks.parse_model is not _dispatch_parse_model:
        _orig_parse_model = tasks.parse_model
    if tasks.parse_model is not _dispatch_parse_model:
        tasks.parse_model = _dispatch_parse_model


def load_model(model_cls, path):
    """用自定义 parse_model 构建 YOLOv13 模型（YAML 或官方 .pt）。"""
    install()
    _ensure_dispatch()
    try:
        return model_cls(str(path))
    finally:
        uninstall()
