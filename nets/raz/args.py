# -*- coding: utf-8 -*-
"""Shared command-line arguments."""
from __future__ import annotations

import argparse


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", type=str, default=r'C:/Users/张旭波/Desktop/ShanghaiTech_Crowd_Counting_Dataset/part_B_final',
                   required=False,
                   help="数据集根目录")
    p.add_argument("--dataset", type=str, default="shhb",
                   choices=["auto", "shanghaitech", "sha", "shha", "shhb",
                            "qnrf", "ucf-qnrf", "ucf_qnrf", "ucf50",
                            "ucf_cc_50", "ucf-cc-50", "nwpu", "nwpu-crowd",
                            "jhu", "jhu-crowd", "jhu_crowd", "worldexpo",
                            "world-expo", "custom"],
                   help="数据集预设；auto 自动识别布局")
    p.add_argument("--part", type=str,
                   help="ShanghaiTech part_A / part_B（可选）")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="无官方 val 划分时从 train 切分的比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", type=str, default="",
                   help="可选 CSV/JSON 清单：image,gt（相对清单目录）")
    p.add_argument("--train-sub", type=str, default="",
                   help="自定义训练子目录（相对 data-root）")
    p.add_argument("--test-sub", type=str, default="",
                   help="自定义测试子目录（相对 data-root）")
    p.add_argument("--allow-density-gt", action="store_true",
                   help="允许把 2-D .npy 密度图通过局部峰值转成伪点")
    p.add_argument("--density-threshold", type=float, default=0.05,
                   help="密度图转伪点的响应阈值")
    p.add_argument("--patch-size", type=int, default=256,
                   help="训练随机裁剪尺寸")
    p.add_argument("--max-side", type=int, default=1536,
                   help="验证/推理长边上限；0=不缩放")
    p.add_argument("--attn-stride", type=int, default=4,
                   help="注意力真值计算步长（越小越精确、越慢）")
    p.add_argument("--beta", type=float, default=0.1,
                   help="几何自适应高斯 beta（论文 0.1）")
    p.add_argument("--sigma-z", type=float, default=3.0,
                   help="缩放注意力真值 sigma_z（论文 3）")
    p.add_argument("--no-augment", action="store_true",
                   help="关闭训练随机缩放/翻转")
    p.add_argument("--max-train", type=int, default=0,
                   help="仅使用前 N 张训练图（0=全部，便于快速调试）")
    p.add_argument("--max-val", type=int, default=0,
                   help="仅使用前 N 张验证图（0=全部）")


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                   default=True, help="不加载 VGG16_bn 预训练权重")
    p.add_argument("--vgg-weights", type=str, default="",
                   help="VGG16_bn 权重路径（可选）")
    p.add_argument("--device", type=str, default="cpu",
                   help="auto / cpu / cuda / cuda:0")
    p.add_argument("--num-workers", type=int, default=0)


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", type=str, default="all",
                   choices=["density", "localization", "loc", "attention",
                            "attn", "proposal", "raz", "all"],
                   help="训练阶段；all 依次执行 density→loc→attention→RAZ")
    p.add_argument("--epochs", type=int, default=30,
                   help="Main-Net 每个阶段训练轮数")
    p.add_argument("--raz-epochs", type=int, default=0,
                   help="RAZ 训练轮数；0 使用 --epochs")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--momentum", type=float, default=0.95,
                   help="SGD 动量（论文 0.95）")
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "adam", "adamw"])
    p.add_argument("--patience", type=int, default=4,
                   help="ReduceLROnPlateau 耐心值")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--init-checkpoint", type=str, default="",
                   help="加载 Main-Net 初始权重（loc/attention 阶段）")
    p.add_argument("--main-checkpoint", type=str, default="",
                   help="RAZ 阶段使用的 Main-Net 权重")
    p.add_argument("--zoom-cache", type=str, default="",
                   help="RAZ 子图缓存目录；留空 output-dir/zoom_cache")
    p.add_argument("--regenerate-zoom", action="store_true",
                   help="即使缓存已存在也重新生成 RAZ 子图")
    p.add_argument("--fusion-threshold", type=float, default=5.0,
                   help="论文式 4x4 融合阈值 theta_f（建议在训练集上网格搜索）")
    p.add_argument("--peak-threshold", type=float, default=0.3,
                   help="定位图峰值阈值")
    p.add_argument("--gamma", type=float, default=100.0,
                   help="定位 BCE 正样本权重（论文 100）")
    p.add_argument("--attn-pos-weight", type=float, default=1.0,
                   help="注意力 BCE 正样本权重")
    p.add_argument("--log-every", type=int, default=20)


def add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=str, default="",
                   help="Main-Net 权重")
    p.add_argument("--raz-checkpoint", type=str, default="",
                   help="可选 RAZ-Net 权重")
    p.add_argument("--split", type=str, default="test",
                   choices=["test", "val"], help="评估划分")
    p.add_argument("--peak-threshold", type=float, default=0.3)
    p.add_argument("--fusion-threshold", type=float, default=5.0)
    p.add_argument("--sigma", type=float, default=40.0,
                   help="定位 AP/AR 高斯 sigma")
    p.add_argument("--no-ap", action="store_true",
                   help="跳过 AP/AR 计算")
    p.add_argument("--raz-rounds", type=int, default=1,
                   help="RAZ 循环次数（0=不用 RAZ）")
    p.add_argument("--output", type=str, default="",
                   help="评估结果 JSON 输出路径")
