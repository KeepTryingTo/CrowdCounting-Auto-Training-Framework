# -*- coding: utf-8 -*-
"""Command-line arguments shared by D2C scripts."""
from __future__ import annotations

import argparse


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", required=True, help="数据集根目录")
    p.add_argument("--dataset", default="auto",
                   choices=["auto", "shanghaitech", "sha", "shha", "shhb",
                            "qnrf", "ucf-qnrf", "ucf_qnrf", "ucf50",
                            "ucf_cc_50", "ucf-cc-50", "nwpu", "nwpu-crowd",
                            "jhu", "jhu-crowd", "jhu_crowd", "worldexpo",
                            "world-expo", "custom"],
                   help="数据集预设；auto 自动识别")
    p.add_argument("--part", default="", help="ShanghaiTech part_A/part_B")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", default="", help="image,gt CSV/JSON 清单")
    p.add_argument("--train-sub", default="")
    p.add_argument("--test-sub", default="")
    p.add_argument("--allow-density-gt", action="store_true",
                   help="允许 .npy 密度图用局部峰值转伪点")
    p.add_argument("--density-threshold", type=float, default=0.05)
    p.add_argument("--patch-size", type=int, default=384)
    p.add_argument("--max-side", type=int, default=1920)
    p.add_argument("--sigma-mode", default="adaptive",
                   choices=["adaptive", "fixed"])
    p.add_argument("--sigma", type=float, default=15.0,
                   help="fixed 模式的 sigma")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                   default=True)
    p.add_argument("--vgg-weights", default="",
                   help="VGG16_bn 权重路径（可选）")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-3)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--resume", default="")
    p.add_argument("--use-gt-prob", action="store_true", default=True,
                   help="CMR 同时输入预测概率图与真值概率图（官方做法）")
    p.add_argument("--no-gt-prob", dest="use_gt_prob", action="store_false")
    p.add_argument("--use-synth", action="store_true", default=True,
                   help="用平衡合成概率图增强 CMR")
    p.add_argument("--no-synth", dest="use_synth", action="store_false")
    p.add_argument("--synth-stat-images", type=int, default=100)
    p.add_argument("--synth-batch-size", type=int, default=1)
    p.add_argument("--synth-max-count", type=int, default=40,
                   help="合成 32x32 单元的最大计数")
    p.add_argument("--log-every", type=int, default=20)


def add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--mode", default="both",
                   choices=["both", "count", "loc"])
    p.add_argument("--match-distance", type=float, default=20.0)
    p.add_argument("--output", default="")
