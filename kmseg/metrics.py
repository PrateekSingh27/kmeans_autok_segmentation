"""Quantitative cluster-validity criteria, implemented from scratch.

* inertia / WCSS                    (elbow method input)
* silhouette coefficient            (Rousseeuw, 1987)         - maximise
* Davies-Bouldin index              (Davies & Bouldin, 1979)  - minimise
* Calinski-Harabasz index           (Calinski & Harabasz, 1974) - maximise
* Gap statistic                     (Tibshirani et al., 2001) - 1-SE rule
* Adjusted Rand Index               (Hubert & Arabie, 1985)   - external, vs ground truth
* MSE / PSNR                        - image reconstruction quality
"""
from __future__ import annotations

import numpy as np

from .kmeans import KMeans, sq_dists


# ----------------------------------------------------------------------
# internal validity indices
# ----------------------------------------------------------------------
def silhouette_score(X, labels, sample_size=2000, random_state=0) -> float:
    """Mean silhouette coefficient. O(m^2) so it is computed on a random
    subsample of at most ``sample_size`` points."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(X)
    if n > sample_size:
        idx = np.random.default_rng(random_state).choice(n, sample_size, replace=False)
        X, labels = X[idx], labels[idx]
        n = sample_size
    uniq, inv = np.unique(labels, return_inverse=True)
    k = len(uniq)
    if k < 2 or k >= n:
        return 0.0
    D = np.sqrt(sq_dists(X, X))
    onehot = np.zeros((n, k))
    onehot[np.arange(n), inv] = 1.0
    sums = D @ onehot                      # (n,k) sum of distances to every cluster
    counts = onehot.sum(0)
    ar = np.arange(n)
    a = sums[ar, inv] / np.maximum(counts[inv] - 1, 1)
    mean_other = sums / counts[None, :]
    mean_other[ar, inv] = np.inf
    b = mean_other.min(axis=1)
    denom = np.maximum(a, b)
    s = np.divide(b - a, denom, out=np.zeros(n), where=denom > 0)
    s[counts[inv] == 1] = 0.0
    return float(s.mean())


def _centroids(X, labels):
    uniq, inv = np.unique(labels, return_inverse=True)
    k = len(uniq)
    C = np.zeros((k, X.shape[1]))
    for j in range(X.shape[1]):
        C[:, j] = np.bincount(inv, weights=X[:, j], minlength=k)
    counts = np.bincount(inv, minlength=k)
    return C / counts[:, None], inv, counts


def davies_bouldin_score(X, labels) -> float:
    X = np.asarray(X, dtype=np.float64)
    C, inv, counts = _centroids(X, labels)
    k = len(C)
    if k < 2:
        return 0.0
    dist_to_c = np.sqrt(((X - C[inv]) ** 2).sum(1))
    S = np.bincount(inv, weights=dist_to_c, minlength=k) / counts
    M = np.sqrt(sq_dists(C, C))
    np.fill_diagonal(M, np.inf)
    R = (S[:, None] + S[None, :]) / M
    np.fill_diagonal(R, -np.inf)
    return float(R.max(axis=1).mean())


def calinski_harabasz_score(X, labels) -> float:
    X = np.asarray(X, dtype=np.float64)
    C, inv, counts = _centroids(X, labels)
    k, n = len(C), len(X)
    if k < 2 or k >= n:
        return 0.0
    mean = X.mean(0)
    B = float((counts * ((C - mean) ** 2).sum(1)).sum())
    W = float(((X - C[inv]) ** 2).sum())
    return 0.0 if W == 0 else (B / (k - 1)) / (W / (n - k))


# ----------------------------------------------------------------------
# gap statistic
# ----------------------------------------------------------------------
def gap_statistic(X, k_values, log_wk, B=10, random_state=0, max_iter=50):
    """Gap statistic with PCA-aligned uniform reference sets.

    Parameters
    ----------
    log_wk : log(inertia) of the real data for each k in ``k_values``
    Returns (gap, s_k) arrays aligned with ``k_values``.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Z = (X - mu) @ Vt.T
    lo, hi = Z.min(0), Z.max(0)
    ref = np.zeros((B, len(k_values)))
    for b in range(B):
        Xb = (rng.uniform(lo, hi, size=Z.shape)) @ Vt + mu
        for i, k in enumerate(k_values):
            km = KMeans(k, n_init=1, max_iter=max_iter,
                        random_state=int(rng.integers(1 << 31))).fit(Xb)
            ref[b, i] = np.log(max(km.inertia_, 1e-12))
    gap = ref.mean(0) - np.asarray(log_wk)
    s = ref.std(0) * np.sqrt(1.0 + 1.0 / B)
    return gap, s


def gap_choose_k(k_values, gap, s) -> int:
    """Tibshirani's rule: smallest k with Gap(k) >= Gap(k+1) - s(k+1)."""
    for i in range(len(k_values) - 1):
        if gap[i] >= gap[i + 1] - s[i + 1]:
            return int(k_values[i])
    return int(k_values[int(np.argmax(gap))])


# ----------------------------------------------------------------------
# elbow
# ----------------------------------------------------------------------
def elbow_k(k_values, inertia) -> int:
    """Knee detection: point of the (normalised) inertia curve that is
    farthest below the chord joining its end points (Kneedle-style)."""
    k = np.asarray(k_values, dtype=float)
    y = np.asarray(inertia, dtype=float)
    if len(k) < 3 or y.max() == y.min():
        return int(k[0])
    xn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    return int(k[int(np.argmax(1.0 - xn - yn))])


# ----------------------------------------------------------------------
# external / reconstruction metrics
# ----------------------------------------------------------------------
def adjusted_rand_index(a, b) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    cont = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(cont, (ia, ib), 1)
    c2 = lambda x: x * (x - 1) / 2.0
    sum_ij = c2(cont).sum()
    sum_a, sum_b = c2(cont.sum(1)).sum(), c2(cont.sum(0)).sum()
    expected = sum_a * sum_b / c2(len(a))
    max_idx = 0.5 * (sum_a + sum_b)
    return 1.0 if max_idx == expected else float((sum_ij - expected) / (max_idx - expected))


def mse(img_a, img_b) -> float:
    d = np.asarray(img_a, dtype=np.float64) - np.asarray(img_b, dtype=np.float64)
    return float((d ** 2).mean())


def psnr(img_a, img_b, peak=255.0) -> float:
    m = mse(img_a, img_b)
    return float("inf") if m == 0 else float(10 * np.log10(peak ** 2 / m))
