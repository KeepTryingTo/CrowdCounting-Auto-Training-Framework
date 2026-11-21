# -*- coding: utf-8 -*-
"""在线伪域聚类（DCCUS / LLB 元学习使用）

PCA 降维 + KMeans 聚类 + 匈牙利匹配标签对齐（同 DCCUS
utils/clustering/{clustering,domain_split}.py）。sklearn 可选：
有则用 sklearn（与官方一致），无则用 numpy 实现兜底。
"""
import numpy as np


def _pca(X, dim):
    X = np.asarray(X, dtype=np.float64)
    try:
        from sklearn.decomposition import PCA
        dim = min(dim, X.shape[0], X.shape[1])
        return PCA(dim).fit_transform(X)
    except Exception:
        Xc = X - X.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        dim = min(dim, Vt.shape[0])
        return (Xc @ Vt[:dim].T).astype(np.float32)


def _kmeans(X, k, seed=0, n_iter=100):
    try:
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, n_init=10, random_state=seed) \
            .fit_predict(X).astype(np.int64)
    except Exception:
        rng = np.random.default_rng(seed)
        n = len(X)
        k = max(1, min(k, n))
        centers = [X[rng.integers(n)]]
        for _ in range(1, k):   # k-means++ 初始化
            d = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2)
                       .sum(-1), axis=1)
            p = d / max(d.sum(), 1e-12)
            centers.append(X[rng.choice(n, p=p)])
        centers = np.array(centers, dtype=np.float64)
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(n_iter):
            d = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                m = labels == c
                if m.any():
                    centers[c] = X[m].mean(0)
        return labels


def reassign(y_before, y_pred):
    """用匈牙利匹配把本次聚类标签对齐到上次，避免子域标签抖动"""
    y_before = np.asarray(y_before, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    D = int(max(y_before.max(), y_pred.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for a, b in zip(y_before, y_pred):
        w[a, b] += 1
    try:
        from scipy.optimize import linear_sum_assignment
        _, mapping = linear_sum_assignment(w.max() - w)
        remap = {int(b): int(a) for a, b in enumerate(mapping)}
    except Exception:
        remap = {i: i for i in range(D)}
    return np.array([remap.get(int(v), int(v)) for v in y_pred],
                    dtype=np.int64)


def cluster_domains(features, k, prev_labels=None, pca_dim=256, seed=0):
    """特征 → 伪域标签（先 PCA 后 KMeans，可选匈牙利对齐）

    features: (N, D) 每图特征（DCCUS 用 backbone 空间 mean/std 拼接）
    """
    X = _pca(features, pca_dim)
    labels = _kmeans(X, k, seed=seed)
    if prev_labels is not None and len(prev_labels) == len(labels):
        labels = reassign(prev_labels, labels)
    return labels
