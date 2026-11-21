# Spatial Matching Distance (CU paper Sec. 3.2, "dist(P, Q)").
#
# dist(P, Q) = (M(P, Q) + H(P, Q)) / N_Q ,  N_P <= N_Q
#   M : optimal one-to-one (Hungarian) matching distance between the two
#       point sets -- min_gamma sum ||p - gamma(p)||_2
#   H : unmatched penalty, one full constant C per surplus point:
#       H = (N_max - N_min) * C,  C = patch diagonal sqrt(h^2 + w^2)
#
# Unlike the Hungarian loss (which discards unmatched points) and discrete
# Wasserstein (which treats points as mass and dilutes count errors), BOTH
# over- and under-prediction pay the FULL per-point penalty here.
#
# We implement the symmetric generalization: the smaller set is matched
# one-to-one into the larger set and the distance is normalized by
# max(N_P, N_G), which preserves the paper's formula for N_P <= N_G and
# gives dist in [0, C] for every case (dist = C when either set is empty).
import numpy as np
from scipy.optimize import linear_sum_assignment


def spatial_matching_distance(pred_points, gt_points, C):
    """Spatial matching distance between two point sets.

    Args:
        pred_points: np.ndarray [P, 2] (x, y) in patch-local coordinates
        gt_points:   np.ndarray [G, 2] (x, y) in the same coordinates
        C:           unmatched-point penalty (patch diagonal, e.g. 64*sqrt(2))
    Returns:
        float distance in [0, C].
    """
    pred = np.asarray(pred_points, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt_points, dtype=np.float64).reshape(-1, 2)
    n_p, n_g = len(pred), len(gt)

    if n_p == 0 and n_g == 0:
        return 0.0
    if n_p == 0 or n_g == 0:
        # all points of the non-empty set are unmatched -> full penalty
        return float(C)

    # one-to-one matching of the SMALLER set into the larger set
    if n_p <= n_g:
        small, large = pred, gt
    else:
        small, large = gt, pred
    # 小集合一一配入大集合（对称推广：paper 公式固定 N_P <= N_Q）
    if n_p <= n_g:
        small, large = pred, gt
    else:
        small, large = gt, pred
    # 成对欧式距离矩阵 [S, L]
    d = np.linalg.norm(small[:, None, :] - large[None, :, :], axis=2)
    # 最小总代价指派（返回配对的行列索引）
    row, col = linear_sum_assignment(d)
    match_cost = float(d[row, col].sum())   # M 项：已配对点的距离和

    # H 项：多出的点每个按满额 C 罚；分母取较大集合规模
    unmatched = abs(n_p - n_g)
    dist = (match_cost + unmatched * C) / max(n_p, n_g)
    # 格子内任意两点距离 <= 对角线 C，理论上界即 C，clamp 只防数值毛刺
    return float(min(dist, C))
