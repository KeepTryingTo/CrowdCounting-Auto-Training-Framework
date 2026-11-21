# -*- coding: utf-8 -*-
"""Point-to-Region (P2R) 损失（vendor 自 P2RLoss-main/losses/p2rloss.py，
CVPR 2025，逻辑未改动，仅补充类型注解与注释格式）

- 有标签数据：P2R 匹配替代匈牙利一对一匹配——每个 GT 点以 Voronoi
  最近邻分派像素形成「点→区域」，区域内按成本（距离项 τ·l2 - 前景
  logit 项）选最优像素作为学习目标，其余区域像素为背景，加权 BCE
- 无标签数据：教师置信度 ζ 传播到区域（可靠前景）与区域外（可靠
  背景），不可靠位置权重置零
"""
import torch


def _meshgrid_ij(a, b):
    """torch.meshgrid（'ij' 语义与源实现默认一致；新版本 torch 需显式
    indexing 消除 FutureWarning，旧版本无该参数则回退）"""
    try:
        return torch.meshgrid(a, b, indexing="ij")
    except TypeError:
        return torch.meshgrid(a, b)


class L2DIS:
    """两组点的成对欧氏距离（论文 l2(i,j) = ||x_i - x'_j||_2）"""

    def __init__(self, factor=512) -> None:
        self.factor = factor

    def __call__(self, X, Y):
        """X: [1,HW,2] 像素网格坐标，Y: [1,n,2] GT/伪标签点 → [1,HW,n]"""
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        C = torch.norm(x_col - y_row, dim=-1)
        C = C / self.factor
        return C


class P2RLoss(torch.nn.modules.loss._Loss):
    """Point-to-Region 损失（对应论文第 5 节）"""

    def __init__(self, factor=1, reduction='mean') -> None:
        super().__init__()
        self.factor = factor
        self.cost = L2DIS(1)

        # P2R 区域半径上下界（论文中的 μ，自适应并截断）
        self.min_radis = 8
        self.max_radis = 96

        # 成本矩阵权重（论文公式 21 的 τ 与 -S(p)）
        self.cost_class = 1
        self.cost_point = 8

    def forward(self, dens, seqs, down, masks=None, crop_den_masks=None):
        """
        dens: [B,1,H,W] 学生模型 logits 图
        seqs: list of [n,2+] 每图 GT/伪标签点坐标（(y,x) 序）
        down: int 像素坐标 → 原图坐标的下采样映射因子
        masks: list of [n,1] 教师对伪标签点的置信度 ζ（半监督）
        crop_den_masks: [B,1,H,W] 强增强随机裁剪掩码（0=被裁剪区域）
        """
        import torch.nn.functional as tF

        bs = len(seqs)
        oot_loss, cnt_loss = 0, 0

        for i in range(bs):
            den = dens[i]                       # [1,H,W]
            seq = seqs[i]                       # [n,2+]

            if crop_den_masks is not None:
                crop_den_mask = crop_den_masks[i]

            if masks is not None:
                mask = masks[i]                 # [n,1]

            den = den.permute(1, 2, 0)          # [H,W,1]
            H, W = den.shape[:2]

            # ---- 情况 1：空场景（无 GT 点）→ 全图压向 0 ----
            if seq.size(0) < 1:
                cnt_loss = cnt_loss + tF.binary_cross_entropy_with_logits(
                    input=den,
                    target=torch.zeros_like(den),
                    weight=torch.ones_like(den) * 0.5
                )
            else:
                # ---- 情况 2：P2R 匹配 ----
                # 像素网格坐标 (row,col)，映射回原图坐标（像素中心对齐）
                A_coord = torch.stack(
                    _meshgrid_ij(torch.arange(H), torch.arange(W)),
                    dim=-1
                ).view(1, -1, 2) * down + (down - 1) / 2  # [1,HW,2]

                A = den.view(1, -1, 1)          # [1,HW,1] logits
                A_coord = A_coord.to(seq).float().view(1, -1, 2)

                B_coord = seq[None, :, :2].float()   # [1,n,2]
                B = torch.ones(seq.size(0)).float().to(A).view(1, -1, 1)
                if masks is not None:
                    MB = mask.view_as(B).to(B)       # [1,n,1] ζ

                with torch.no_grad():
                    # [1,HW,n] 每像素到每 GT 点的距离
                    C = self.cost(A_coord, B_coord)

                    # Voronoi 分割：每像素分给最近 GT 点，超出 max_radis 不分派
                    minC, mcidx = C.min(dim=-1, keepdim=True)
                    M = torch.zeros_like(C).scatter_(-1, mcidx, 1.0) * \
                        (C < self.max_radis)

                    # 自适应区域半径：各 GT 点分派像素的最远距离，
                    # 截断到 [min_radis, max_radis]
                    maxC = (minC.view_as(A) * M).amax(dim=1, keepdim=True)
                    maxC = torch.clip(maxC, min=self.min_radis,
                                      max=self.max_radis)
                    C = C / maxC

                    # 成本 = τ·归一化距离 - 前景 logit
                    C = C * self.cost_point - A * self.cost_class

                    # 过滤没有任何像素分派的 GT 点
                    vid = (M.sum(dim=1) > 0).view(-1)
                    C, M = C[..., vid], M[..., vid]
                    B, B_coord = B[:, vid, :], B_coord[:, vid, :]

                    # 区域内选成本最低像素作为该 GT 点的学习目标
                    C2 = M * C + (1 - M) * (C.max() + 1)
                    minC2, mcidx2 = C2.min(dim=1, keepdim=True)
                    T = torch.zeros_like(C2).scatter_(1, mcidx2, 1.0) \
                        .sum(dim=-1).view(1, -1, 1)
                    T = (T > 0.5).to(A).view_as(A)

                    # 权重 W = T + 1（前景 2 / 背景 1）
                    W = T + 1

                    # 半监督置信度传播：区域内可靠前景 + 区域外可靠背景
                    if masks is not None:
                        foreground_conf = M @ MB[:, vid, :]
                        background_conf = 1 - M.sum(dim=-1).view_as(A)
                        M = foreground_conf + background_conf
                        W = W * M

                if crop_den_masks is not None:
                    W = W * crop_den_mask.view_as(W)

                cnt_loss = cnt_loss + tF.binary_cross_entropy_with_logits(
                    A, T, weight=W)

        loss = (oot_loss + cnt_loss) / bs
        return loss
