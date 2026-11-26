"""K-Means clustering implemented from scratch with NumPy.

Features
--------
* k-means++ (D^2 weighted) or random initialisation
* several restarts (``n_init``) - the run with the lowest inertia wins
* vectorised, chunked distance computation (memory-safe for large images)
* empty-cluster repair (re-seed with the points farthest from their centroid)
* relative tolerance convergence test (scaled by data variance)
"""
from __future__ import annotations

import numpy as np

_CHUNK = 200_000


def sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Pair-wise squared Euclidean distances, shape (n, k).

    Uses ||x||^2 - 2 x.c + ||c||^2 which is much faster than broadcasting.
    """
    x_sq = np.einsum("ij,ij->i", X, X)
    c_sq = np.einsum("ij,ij->i", C, C)
    d = x_sq[:, None] - 2.0 * (X @ C.T) + c_sq[None, :]
    np.maximum(d, 0.0, out=d)
    return d


def assign(X: np.ndarray, C: np.ndarray, chunk: int = _CHUNK):
    """Assign each row of X to its nearest centre.

    Returns (labels, min_sq_dist), both of length n.
    """
    n = X.shape[0]
    labels = np.empty(n, dtype=np.int32)
    mind = np.empty(n, dtype=X.dtype)
    for s in range(0, n, chunk):
        d = sq_dists(X[s:s + chunk], C)
        l = d.argmin(axis=1)
        labels[s:s + chunk] = l
        mind[s:s + chunk] = d[np.arange(len(l)), l]
    return labels, mind


def kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (Arthur & Vassilvitskii, 2007)."""
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    centers[0] = X[rng.integers(n)]
    closest = sq_dists(X, centers[:1])[:, 0].astype(np.float64)
    for c in range(1, k):
        total = closest.sum()
        if total <= 0:                      # all points identical to a centre
            idx = rng.integers(n)
        else:
            idx = rng.choice(n, p=closest / total)
        centers[c] = X[idx]
        closest = np.minimum(closest, sq_dists(X, centers[c:c + 1])[:, 0])
    return centers


def _random_init(X, k, rng):
    return X[rng.choice(X.shape[0], size=k, replace=False)].copy()


class KMeans:
    """Lloyd's algorithm K-Means.

    Parameters
    ----------
    n_clusters : int
    init : {"k-means++", "random"}
    n_init : int            number of restarts
    max_iter : int          max Lloyd iterations per restart
    tol : float             relative tolerance on centre movement
    random_state : int | None
    """

    def __init__(self, n_clusters=3, init="k-means++", n_init=5, max_iter=100,
                 tol=1e-4, random_state=None):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def _single_run(self, X, rng, tol_abs):
        k = self.n_clusters
        centers = (kmeans_plus_plus(X, k, rng) if self.init == "k-means++"
                   else _random_init(X, k, rng))
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            labels, mind = assign(X, centers)
            counts = np.bincount(labels, minlength=k)
            new = np.empty_like(centers)
            for j in range(X.shape[1]):
                new[:, j] = np.bincount(labels, weights=X[:, j], minlength=k)
            nz = counts > 0
            new[nz] /= counts[nz, None]
            if not nz.all():                # repair empty clusters
                far = np.argsort(mind)[::-1][: (~nz).sum()]
                new[~nz] = X[far]
            shift = float(((new - centers) ** 2).sum())
            centers = new
            if shift <= tol_abs:
                break
        labels, mind = assign(X, centers)
        return centers, labels, float(mind.sum(dtype=np.float64)), n_iter

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")
        rng = np.random.default_rng(self.random_state)
        tol_abs = self.tol * float(np.mean(np.var(X, axis=0))) + 1e-12
        best = None
        for _ in range(max(1, self.n_init)):
            res = self._single_run(X, rng, tol_abs)
            if best is None or res[2] < best[2]:
                best = res
        (self.cluster_centers_, self.labels_,
         self.inertia_, self.n_iter_) = best
        return self

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return assign(X, self.cluster_centers_)[0]

    def fit_predict(self, X):
        return self.fit(X).labels_
