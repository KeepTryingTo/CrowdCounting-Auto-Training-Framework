# -*- coding: utf-8 -*-
"""训练过程图表（matplotlib 保存 PNG，经 /runs 下载）

使用 Figure 对象 API 而非 pyplot：pyplot 维护全局图窗状态、非线程安全，
并发训练（任务管理器默认并发 2）会互相踩踏；Figure 自建实例无共享状态。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

# 中文字体（Windows）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_loss_curves(history, save_path, metrics_keys=("train_loss", "val_loss")):
    """训练/验证损失曲线 PNG（每 epoch 更新）

    保存后 fig.clf() 及时释放渲染缓冲（dpi=150 的像素阵列每轮数 MB，
    等惰性 GC 会随训练轮数堆积；配合训练循环的逐轮 gc 双保险）
    """
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    labels = {"train_loss": "训练损失", "val_loss": "验证损失",
              "val_mae": "验证 MAE"}
    for k in metrics_keys:
        # 逐点过滤 None：续训/部分轮次缺验证值时仍能画出已有段，
        # 而不是整条曲线被丢弃
        pts = [(h["epoch"], h.get(k)) for h in history
               if h.get(k) is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", markersize=3, label=labels.get(k, k))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("值")
    ax.set_title("训练过程曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    fig.clf()


def plot_metrics_summary(metrics, save_path, title="验证集指标"):
    """指标柱状图 PNG（dict of name->value；YOLO 头部框路线 mAP 汇总用）"""
    items = [(k, v) for k, v in metrics.items()
             if isinstance(v, (int, float))]
    if not items:
        return
    names = [k for k, _ in items]
    vals = [float(v) for _, v in items]
    fig = Figure(figsize=(max(6, len(names) * 1.2), 5))
    ax = fig.subplots()
    bars = ax.bar(names, vals, color="#3b82f6", alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.4g}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    fig.clf()
