# -*- coding: utf-8 -*-
"""Counting (MAE/MSE), GAME and localization AP/AR/F1 metrics."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def count_metrics(pred, gt) -> Dict[str, float]:
    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)
    if len(p) == 0:
        return {"mae": float("nan"), "mse": float("nan"), "n": 0}
    err = p - g
    return {"mae": float(np.abs(err).mean()),
            "mse": float(np.sqrt((err ** 2).mean())), "n": int(len(p))}


def _dot_points(dot: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(dot)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def game_metric(gt_dot: np.ndarray, pred_dot: np.ndarray,
                level: int) -> float:
    """GAME(level): split into 2^level x 2^level cells, sum abs count error."""
    assert gt_dot.shape == pred_dot.shape
    cuts = 2 ** int(level)
    h, w = gt_dot.shape
    ch, cw = h // cuts, w // cuts
    if ch == 0 or cw == 0:
        return float(abs(gt_dot.sum() - pred_dot.sum()))
    g = gt_dot[:ch * cuts, :cw * cuts].reshape(cuts, ch, cuts, cw).sum((1, 3))
    p = pred_dot[:ch * cuts, :cw * cuts].reshape(cuts, ch, cuts, cw).sum((1, 3))
    return float(np.abs(g - p).sum())


def match_tp(pred_dot: np.ndarray, gt_dot: np.ndarray,
             distance: float = 20.0) -> int:
    """Hungarian matching with distance<=20 (official AP/AR protocol)."""
    pred = _dot_points(pred_dot)
    gt = _dot_points(gt_dot)
    if len(pred) == 0 or len(gt) == 0:
        return 0
    dist = cdist(pred, gt, metric="euclidean")
    big = distance + 1.0
    cost = np.where(dist <= distance, dist, big)
    rows, cols = linear_sum_assignment(cost)
    return int(sum(dist[r, c] <= distance for r, c in zip(rows, cols)))


def localization_metrics(pred_dots: Sequence[np.ndarray],
                         gt_dots: Sequence[np.ndarray],
                         distance: float = 20.0) -> Dict[str, float]:
    tp = pred = gt = 0
    for p, g in zip(pred_dots, gt_dots):
        tp += match_tp(p, g, distance)
        pred += int(p.sum())
        gt += int(g.sum())
    ap = tp / max(1, pred)
    ar = tp / max(1, gt)
    f1 = 2 * ap * ar / max(1e-9, ap + ar)
    return {"AP": float(ap), "AR": float(ar), "F1": float(f1),
            "TP": int(tp), "pred": int(pred), "gt": int(gt)}
