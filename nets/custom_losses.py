# -*- coding: utf-8 -*-
"""自定义损失示例：训练「损失函数」下拉框中 custom: 开头的即来自此处。

新增：在 nets/custom_losses/ 目录建 .py，@register_custom_loss 注册。

按路线分区（tasks）：
  密度图 crowd_density: forward(pred[N,1,h,w], gt[N,1,h,w]) -> 标量
  点预测 crowd_point:   forward(outputs, targets) -> 标量
                        （作为附加项叠加到算法内置损失上：P2PNet / CP / CU /
                         CLTR 的 outputs 为 {pred_logits, pred_points} 字典；
                         P2R 为 forward(pred_density, target_seq)；PET 的损失
                         在模型内计算，不支持附加）
  坐标框 crowd_box:     ultralytics 引擎托管，暂不支持自定义损失
"""
from app.core.registry import register_custom_loss


@register_custom_loss("huber_density",
                      desc="Huber（SmoothL1）密度图损失（计数，对异常值鲁棒）",
                      tasks=["crowd_counting"])
def huber_density_factory(beta=1.0, **kw):
    import torch.nn as nn
    return nn.SmoothL1Loss(beta=beta)


@register_custom_loss("mae_ssim",
                      desc="MAE + SSIM 组合损失（计数）",
                      tasks=["crowd_counting"])
def mae_ssim_factory(alpha=0.5, **kw):
    import torch.nn as nn

    class MAESsimCombo(nn.Module):
        def __init__(self):
            super().__init__()
            from app.services.crowd_losses import MAELoss, SSIM_Loss
            self.alpha = alpha
            self.mae = MAELoss()
            self.ssim = SSIM_Loss(in_channels=1)

        def forward(self, pred, gt):
            return (self.alpha * self.mae(pred, gt)
                    + (1 - self.alpha) * self.ssim(pred, gt))

    return MAESsimCombo()


@register_custom_loss("point_count_l1",
                      desc="点预测人数一致性附加损失（P2PNet 输出）",
                      tasks=["crowd_point"])
def point_count_l1_factory(weight=0.001, **kw):
    """示例：|预测总人数 - GT 点数| 均值，叠加在内置匹配损失上。

    点预测模型 forward(x) 需返回 {"pred_logits":[B,M,2], ...}；
    targets 为 list[{point:[N,2], labels:[N]}]。
    """
    import torch
    import torch.nn as nn

    class PointCountL1(nn.Module):
        def forward(self, outputs, targets):
            probs = outputs["pred_logits"].softmax(-1)[..., 1]
            pred_cnt = probs.sum(dim=1)
            gt_cnt = torch.tensor(
                [len(t["point"]) for t in targets],
                dtype=pred_cnt.dtype, device=pred_cnt.device)
            return weight * (pred_cnt - gt_cnt).abs().mean()

    return PointCountL1()
