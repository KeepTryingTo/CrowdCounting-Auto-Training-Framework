# -*- coding: utf-8 -*-
"""Counting (MAE/RMSE) and localization (AP/AR) metrics."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def count_metrics(pred: Sequence[float], gt: Sequence[float]) -> Dict[str, float]:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if len(p) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    err = p - g
    return {"mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "n": int(len(p))}


def _precision_recall_ap(tp_scores: List[Tuple[float, int]], n_gt: int):
    """按置信度全局排序后计算 AP/AR。

    tp_scores 是所有图像累计的 (score, 是否 TP)；用 101 点插值
    （COCO 风格）计算 AP。
    """
    if n_gt == 0:
        return float("nan"), float("nan")
    if not tp_scores:
        return 0.0, 0.0
    tp_scores.sort(key=lambda x: x[0], reverse=True)
    tp = np.cumsum([x[1] for x in tp_scores])
    fp = np.cumsum([1 - x[1] for x in tp_scores])
    recalls = tp / max(1, n_gt)
    precisions = tp / np.maximum(1, tp + fp)
    # 101-point interpolated AP (COCO style).
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = precisions[recalls >= r]
        ap += (p.max() if p.size else 0.0) / 101.0
    ar = float(tp[-1] / max(1, n_gt))
    return float(ap), ar


def compute_ap_ar(preds: Sequence[np.ndarray],
                  gts: Sequence[np.ndarray],
                  sigma: float = 40.0,
                  thresholds: Sequence[float] = tuple(np.arange(0.5, 1.0, 0.05)),
                  ) -> Dict[str, float]:
    """Paper/COCO-style point AP/AR with a Gaussian similarity.

    Each prediction is matched to the nearest *unmatched* GT point; it is a
    true positive only if the unnormalised Gaussian response
    ``exp(-d^2/(2*sigma^2))`` exceeds the threshold.  Duplicate predictions
    therefore count as false positives.
    """
    num_gt = int(sum(len(g) for g in gts))
    results = {}
    ap_values = []
    for thr in thresholds:
        records: List[Tuple[float, int]] = []
        for pred, gt in zip(preds, gts):
            p = np.asarray(pred, dtype=np.float32).reshape(-1, 3)
            g = np.asarray(gt, dtype=np.float32).reshape(-1, 2)
            if len(p) == 0:
                continue
            # 每张图内按预测分数从高到低匹配。
            order = np.argsort(-p[:, 2])
            p = p[order]
            if len(g) == 0:
                records.extend((float(s), 0) for s in p[:, 2])
                continue
            used = np.zeros(len(g), dtype=bool)
            for x, y, score in p:
                # 只和“尚未被更高分预测占用”的最近 GT 匹配；
                # 高斯响应超过阈值才算 TP，重复检测自然成为 FP。
                d2 = ((g[:, 0] - x) ** 2 + (g[:, 1] - y) ** 2)
                d2[used] = np.inf
                j = int(np.argmin(d2))
                response = math.exp(-float(d2[j]) / (2.0 * float(sigma) ** 2))
                if np.isfinite(d2[j]) and response > float(thr):
                    used[j] = True
                    records.append((float(score), 1))
                else:
                    records.append((float(score), 0))
        ap, ar = _precision_recall_ap(records, num_gt)
        results[f"AP{int(round(thr * 100))}"] = ap
        results[f"AR{int(round(thr * 100))}"] = ar
        ap_values.append(ap)
    valid = [x for x in ap_values if not math.isnan(x)]
    results["mAP"] = float(np.mean(valid)) if valid else float("nan")
    ar_keys = [k for k in results if k.startswith("AR")]
    ars = [results[k] for k in ar_keys if not math.isnan(results[k])]
    results["mAR"] = float(np.mean(ars)) if ars else float("nan")
    return results
