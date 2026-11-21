# -*- coding: utf-8 -*-
"""人群计数损失函数库（移植自 CrowdCounting-framework-PyTorch/losses，纯 PyTorch 零依赖）

- MAE / MSE：密度图对的 L1 / L2
- SSIM：结构相似性损失 1-SSIM(密度图对)
- MMD：多带宽高斯核最大均值差异（特征对，域适应惩罚）
- OT：基于 Sinkhorn 最优传输的计数损失（GT 点坐标 + 预测密度图）
- DMCount：DM-Count（NeurIPS 2020, Distribution Matching for Crowd Counting）
  完整损失 = 计数损失 L_C + λ_ot·OT 分布匹配损失 L_OT + λ_tv·TV 损失 L_TV
  （Sinkhorn 最优传输 + 包络定理梯度写回，权重/正则/迭代数对齐官方 train.py）
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class MAELoss(nn.Module):
    """L1 密度图损失"""

    def __init__(self, lambda_c=1.0, reduction="mean"):
        super().__init__()
        self.lambda_c = lambda_c
        # 必须用关键字：nn.L1Loss 的首个位置参数是已弃用的 size_average，
        # 传 "sum" 会被当成 True 而错误地退化为 mean
        self.mae_loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred, gt):
        return self.lambda_c * self.mae_loss(pred, gt)


class MSELoss(nn.Module):
    """L2 密度图损失"""

    def __init__(self, lambda_global=1.0, reduction="mean"):
        super().__init__()
        self.lambda_global = lambda_global
        # 同上：reduction 必须关键字传参，避免落到 size_average
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, gt):
        return self.lambda_global * self.mse_loss(pred, gt)


class SSIM_Loss(_Loss):
    """结构相似性损失 1 - SSIM(pred, gt)"""

    def __init__(self, in_channels=1, size=11, sigma=1.5, size_average=True):
        # 不再向 _Loss 传已弃用的 size_average（会触发 deprecation warning）；
        # SSIM 恒按 mean 计算，size_average 仅为兼容旧调用签名保留
        super().__init__()
        self.in_channels = in_channels
        self.size = int(size)
        self.sigma = sigma
        kernel = self._gaussian_kernel(self.size, self.sigma)
        weight = np.tile(kernel, (in_channels, 1, 1, 1))
        self.weight = nn.Parameter(
            torch.from_numpy(weight).float(), requires_grad=False)

    @staticmethod
    def _gaussian_kernel(size, sigma):
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        k = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
        return k / k.sum()

    def forward(self, input, target):
        mean1 = F.conv2d(input, self.weight, padding=self.size,
                         groups=self.in_channels)
        mean2 = F.conv2d(target, self.weight, padding=self.size,
                         groups=self.in_channels)
        mean1_sq, mean2_sq, mean_12 = mean1 * mean1, mean2 * mean2, mean1 * mean2
        sigma1_sq = F.conv2d(input * input, self.weight, padding=self.size,
                             groups=self.in_channels) - mean1_sq
        sigma2_sq = F.conv2d(target * target, self.weight, padding=self.size,
                             groups=self.in_channels) - mean2_sq
        sigma_12 = F.conv2d(input * target, self.weight, padding=self.size,
                            groups=self.in_channels) - mean_12
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mean_12 + C1) * (2 * sigma_12 + C2)) / \
               ((mean1_sq + mean2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim.mean()


class MMD(nn.Module):
    """多带宽高斯核最大均值差异（域适应惩罚，输入为特征对）"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _cdist(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def _gaussian_kernel(self, x, y,
                         gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self._cdist(x, y)
        K = torch.zeros_like(D)
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
        return K

    def mmd(self, x, y):
        Kxx = self._gaussian_kernel(x, x).mean()
        Kyy = self._gaussian_kernel(y, y).mean()
        Kxy = self._gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def forward(self, s_features, t_features):
        return self.mmd(s_features, t_features)


# ------------------------- Sinkhorn（OT 底层，POT sinkhorn 的纯 torch 重写） -------------------------

M_EPS = 1e-16


def sinkhorn_knopp(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9,
                   log=False, eval_freq=10, warm_start=None, verbose=False,
                   print_freq=200):
    device = a.device
    na, nb = C.shape
    assert na == a.shape[0], "a 与 C 行数不一致"
    assert nb == b.shape[0], "b 与 C 列数不一致"
    if warm_start is not None:
        u = warm_start["u"]
        v = warm_start["v"]
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb
    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    torch.div(C, -reg, out=K)
    torch.exp(K, out=K)
    log_dict = {"err": []}
    err = 1.0
    it = 1
    while err > stopThr and it <= maxIter:
        upre, vpre = u, v
        # 注：不用 matmul(out=) —— 1D×2D 在新版 torch 会按 [1,nb] 视作目标
        # 形状并触发 resize 警告（官方实现对同一写法也有该警告）
        KTu = torch.matmul(u, K)
        v = torch.div(b, KTu + M_EPS)
        Kv = torch.matmul(K, v)
        u = torch.div(a, Kv + M_EPS)
        # 数值保护（同 DM-Count 官方实现）：NaN/Inf 时回退上一步并终止
        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            if verbose:
                print(f"Warning: numerical errors at iteration {it}")
            u, v = upre, vpre
            break
        if log and it % eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            log_dict["err"].append(err)
        if verbose and it % print_freq == 0:
            print(f"iteration {it:5d}, constraint error {err:5e}")
        it += 1
    if log:
        log_dict["u"] = u
        log_dict["v"] = v
        log_dict["alpha"] = reg * torch.log(u + M_EPS)
        log_dict["beta"] = reg * torch.log(v + M_EPS)
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    return P, log_dict


def sinkhorn(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9, log=True, **kw):
    return sinkhorn_knopp(a, b, C, reg=reg, maxIter=maxIter,
                          stopThr=stopThr, log=log, **kw)


class OT_Loss(nn.Module):
    """基于 Sinkhorn 最优传输的计数损失（"One-shot" OT）

    输入：normed_density（softmax 归一化的预测密度）、
         unnormed_density（原始预测密度）、points（每图 GT 点坐标 list）
    返回：(loss, wd Wasserstein 距离, ot_obj_values)
    """

    def __init__(self, c_size, stride, norm_cood, device,
                 num_of_iter_in_ot=100, reg=10.0):
        super().__init__()
        assert c_size % stride == 0
        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.density_size = self.cood.size(0)
        self.cood.unsqueeze_(0)
        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1
        self.output_size = self.cood.size(1)

    def forward(self, normed_density, unnormed_density, points):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)
        loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0
        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                if self.norm_cood:
                    im_points = im_points / self.c_size * 2 - 1
                x = im_points[:, 0].unsqueeze_(1)
                y = im_points[:, 1].unsqueeze_(1)
                x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
                y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
                y_dis.unsqueeze_(2)
                x_dis.unsqueeze_(1)
                dis = y_dis + x_dis
                dis = dis.view((dis.size(0), -1))
                source_prob = normed_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(im_points)]) /
                               len(im_points)).to(self.device)
                P, lg = sinkhorn(target_prob, source_prob, dis, self.reg,
                                 maxIter=self.num_of_iter_in_ot, log=True)
                beta = lg["beta"]
                ot_obj_values += torch.sum(
                    normed_density[idx] * beta.view([1, self.output_size, self.output_size]))
                source_density = unnormed_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                im_grad_1 = (source_count) / (source_count * source_count + 1e-8) * beta
                im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8)
                im_grad = (im_grad_1 - im_grad_2).detach().view(
                    [1, self.output_size, self.output_size])
                loss += torch.sum(unnormed_density[idx] * im_grad)
                wd += torch.sum(dis * P).item()
        return loss, wd, ot_obj_values


# ------------------------- DM-Count（NeurIPS 2020） -------------------------

class DMCountLoss(nn.Module):
    """DM-Count 损失（NeurIPS 2020, Distribution Matching for Crowd Counting）

        L = w_count·L_count + w_ot·L_OT + w_tv·L_TV

    - L_count：预测人数与真实人数之差的绝对值（nn.L1Loss，按 batch 平均）
    - L_OT：Sinkhorn 熵正则最优传输的分布匹配损失（预测归一化密度 vs GT
      归一化密度，代价矩阵 = 网格像素坐标平方距离）。按官方 ot_loss.py 的
      包络定理写法：Sinkhorn 在 detach 后的分布上求解，梯度写回为 beta 项
        im_grad = beta/ŝ - <d,β>/ŝ²,  loss += Σ(d · im_grad)
      （d 为未归一化预测密度，ŝ 为其总和；梯度链保留到预测密度）
    - L_TV：Σ|D̂ - d̂| × 真实人数，再按 batch 平均（官方 train_helper.py 写法）

    参数默认值对齐 DM-Count 官方 train.py：
      w_ot=0.1, w_tv=0.01, reg=10.0, num_of_iter_in_ot=100
    stride：密度图相对输入图像的下采样倍数（本框架密度图恒为 1/8 → 8），
            用于把输出网格坐标还原为输入像素坐标（与官方 cood 完全一致）。
    max_ot_cells：OT 匹配网格的单元数上限（默认 64×64=4096）。输出网格超过时
            按整数倍面积池化（保质量）降采样后再做 OT —— 训练裁剪 384/512
            时为 48×48 / 64×64，与官方分辨率一致；验证阶段的整图（长边可达
            数千像素）自动降采样，避免 O(P²) 代价矩阵爆炸。
    norm_cood：是否把坐标归一化到 [-1, 1]（官方 norm_cood，默认 0=像素坐标）。
    eps：归一化/除零保护（官方 1e-6）。

    说明：
    - 官方实现的 OT 目标测度为「GT 点集」；本框架训练管线只向下游传密度图，
      故目标测度取「归一化 GT 密度」（同为目标测度的离散形式，.npy 与 .txt
      GT 均可用）。两者在 OT 目标上等价（点集即狄拉克形式的密度）。
    - 验证阶段（torch.no_grad）跳过 L_OT，仅统计 L_count + L_TV，避免整图
      Sinkhorn 的可观耗时；验证指标 val_MAE 不受影响。
    """

    def __init__(self, wot=0.1, wtv=0.01, count_weight=1.0, reg=10.0,
                 num_of_iter_in_ot=100, stride=8, norm_cood=False,
                 eps=1e-6, max_ot_cells=4096):
        super().__init__()
        self.wot = float(wot)
        self.wtv = float(wtv)
        self.count_weight = float(count_weight)
        self.reg = float(reg)
        self.iters = max(1, int(num_of_iter_in_ot))
        self.stride = max(1, int(stride))
        self.norm_cood = bool(norm_cood)
        self.eps = float(eps)
        self.max_ot_cells = max(1, int(max_ot_cells))
        self._cost_cache = {}
        self.last_terms = {}

    # ---- 代价矩阵（网格单元中心像素坐标平方距离，按 (H,W,降采样,设备) 缓存） ----
    def _cost(self, h, w, ds, device, dtype):
        key = (h, w, ds, str(device), dtype)
        C = self._cost_cache.get(key)
        if C is None:
            pix = self.stride * ds
            ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * pix
            xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * pix
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
            if self.norm_cood:
                size = float(max(h, w) * pix)
                coords = coords / size * 2 - 1
            C = torch.cdist(coords, coords, p=2).pow_(2)
            self._cost_cache[key] = C
        return C

    def forward(self, pred, gt):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        if pred.shape[1] != 1 or gt.shape[1] != 1:
            raise ValueError(f"DMCount 需要单通道密度图，输入 "
                             f"{tuple(pred.shape)} / {tuple(gt.shape)}")
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear",
                                 align_corners=False)
        pred = pred.float()
        gt = gt.float()
        B = pred.shape[0]
        p_cnt = pred.sum(dim=(1, 2, 3))          # [B] 预测人数（可导）
        g_cnt = gt.sum(dim=(1, 2, 3))            # [B] 真实人数
        # ---- 计数损失（官方：nn.L1Loss 作用于每图总人数） ----
        count_loss = F.l1_loss(p_cnt, g_cnt)
        # ---- 归一化密度（官方：mu / (mu_sum + 1e-6)） ----
        p_norm = pred / (p_cnt.view(B, 1, 1, 1) + self.eps)
        g_norm = gt / (g_cnt.view(B, 1, 1, 1) + self.eps)
        # ---- TV 损失（官方：Σ|p̂-ĝ| 空间求和 × 真实人数 → batch 平均） ----
        tv_loss = ((p_norm - g_norm).abs().flatten(1).sum(1) * g_cnt).mean()
        # ---- OT 损失（验证阶段跳过；需梯度才参与训练） ----
        ot_loss = pred.new_zeros(())
        if self.wot > 0 and torch.is_grad_enabled() and B > 0:
            h, w = pred.shape[-2:]
            cells = h * w
            ds = 1
            while cells // (ds * ds) > self.max_ot_cells:
                ds += 1
            if ds > 1:                            # 面积池化（保总量）
                p_ds = F.avg_pool2d(pred, ds) * float(ds * ds)
                g_ds = F.avg_pool2d(gt, ds) * float(ds * ds)
            else:
                p_ds, g_ds = pred, gt
            hh, ww = p_ds.shape[-2:]
            C = self._cost(hh, ww, ds, pred.device, pred.dtype)   # [P,P]
            p_ds_flat = p_ds.detach().reshape(B, -1)
            src_all = (p_ds / (p_ds.sum(dim=(1, 2, 3)).view(B, 1, 1, 1)
                               + self.eps)).detach().reshape(B, -1)
            tgt_all = (g_ds / (g_ds.sum(dim=(1, 2, 3)).view(B, 1, 1, 1)
                               + self.eps)).detach().reshape(B, -1)
            for i in range(B):
                if float(g_cnt[i]) <= self.eps:
                    continue                      # 0 人图像：无目标测度
                src = src_all[i].clamp_min(0)
                tgt = tgt_all[i].clamp_min(0)
                if float(src.sum()) <= 0 or float(tgt.sum()) <= 0:
                    continue
                # Sinkhorn：a=目标（GT 分布），b=源（预测分布）→ beta 对应源
                _, lg = sinkhorn_knopp(tgt, src, C, reg=self.reg,
                                       maxIter=self.iters, log=True)
                beta = lg["beta"]
                d = p_ds_flat[i]
                s = d.sum()
                # 包络定理：OT 对未归一化预测密度的梯度（官方 im_grad）
                im_grad = beta / (s + 1e-8) - (d * beta).sum() / (s * s + 1e-8)
                ot_loss = ot_loss + p_ds[i].reshape(-1).mul(
                    im_grad.detach()).sum()
        loss = (self.count_weight * count_loss
                + self.wot * ot_loss
                + self.wtv * tv_loss)
        self.last_terms = {"count": float(count_loss.detach()),
                           "ot": float(ot_loss.detach()),
                           "tv": float(tv_loss.detach()),
                           "loss": float(loss.detach())}
        return loss



# 计数损失注册表：名称 -> (构建函数, 输入类型)
# 输入类型: "density"（密度图对）/ "points"（OT，需点标注）
CROWD_LOSSES = {
    "mse": lambda **kw: MSELoss(),
    "mae": lambda **kw: MAELoss(),
    "ssim": lambda **kw: SSIM_Loss(in_channels=1),
    "mse_ssim": lambda **kw: MSESsimCombo(**kw),
    # DM-Count（NeurIPS 2020）：计数 + Sinkhorn OT 分布匹配 + TV
    "dmcount": lambda **kw: DMCountLoss(**kw),
    "ot": None,  # 需要点标注与特定参数，在训练函数内单独构建
    "mmd": None,  # 需要特征对（域适应），作为可选组合项
}


class MSESsimCombo(nn.Module):
    """MSE + SSIM 组合损失（mse_w / ssim_w 可配置，缺省 0.5 / 0.5）"""

    def __init__(self, alpha=None, mse_w=None, ssim_w=None):
        super().__init__()
        if mse_w is None and ssim_w is None:
            a = 0.5 if alpha is None else float(alpha)
            mse_w, ssim_w = a, 1.0 - a
        self.mse_w = float(0.5 if mse_w is None else mse_w)
        self.ssim_w = float(0.5 if ssim_w is None else ssim_w)
        self.mse = MSELoss()
        self.ssim = SSIM_Loss(in_channels=1)

    def forward(self, pred, gt):
        return (self.mse_w * self.mse(pred, gt)
                + self.ssim_w * self.ssim(pred, gt))


def build_crowd_loss(name, **kwargs):
    if name not in CROWD_LOSSES:
        raise ValueError(f"未知计数损失: {name}，可用: {list(CROWD_LOSSES)}")
    fn = CROWD_LOSSES[name]
    if fn is None:
        raise ValueError(f"损失 {name} 需要专用构建参数，不支持此入口")
    return fn(**kwargs)
