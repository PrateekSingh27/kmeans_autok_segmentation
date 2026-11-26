"""End-to-end image segmentation with K-Means and automatic K selection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .colorspace import rgb_to_lab
from .kmeans import KMeans, assign
from .metrics import (adjusted_rand_index, calinski_harabasz_score,
                      davies_bouldin_score, mse, psnr, silhouette_score)
from .selection import ALL_METHODS, SelectionResult, select_k


# ----------------------------------------------------------------------
def load_image(path, max_side: int | None = 256) -> np.ndarray:
    """Load an image as uint8 RGB, optionally down-scaling its longest side."""
    img = Image.open(path).convert("RGB")
    return resize_image(np.asarray(img), max_side)


def resize_image(arr: np.ndarray, max_side: int | None) -> np.ndarray:
    h, w = arr.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = Image.fromarray(arr).resize((max(1, round(w * s)), max(1, round(h * s))),
                                          Image.LANCZOS)
        return np.asarray(img)
    return arr


def build_features(image: np.ndarray, color_space="lab", spatial_weight=0.0) -> np.ndarray:
    """Per-pixel feature matrix (H*W, d).

    color_space    : "rgb" (0..1) or "lab" (L/100, a/100, b/100)
    spatial_weight : if > 0 append (x, y) normalised to [0,1] times this weight,
                     which encourages spatially compact segments.
    """
    h, w = image.shape[:2]
    rgb = image.astype(np.float64) / 255.0
    if color_space == "lab":
        feat = rgb_to_lab(rgb) / 100.0
    elif color_space == "rgb":
        feat = rgb
    else:
        raise ValueError("color_space must be 'rgb' or 'lab'")
    feat = feat.reshape(-1, 3)
    if spatial_weight > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        pos = np.stack([xx / max(w - 1, 1), yy / max(h - 1, 1)], -1).reshape(-1, 2)
        feat = np.hstack([feat, spatial_weight * pos])
    return feat.astype(np.float32)


# ----------------------------------------------------------------------
@dataclass
class SegmentationResult:
    image: np.ndarray               # original (H,W,3) uint8
    labels: np.ndarray              # (H,W) cluster ids, 0 = largest cluster
    segmented: np.ndarray           # (H,W,3) uint8, every pixel = its cluster mean colour
    palette: np.ndarray             # (k,3) uint8
    proportions: np.ndarray         # (k,) fraction of pixels per cluster
    k: int
    selection: SelectionResult | None
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"k": int(self.k), "metrics": self.metrics,
             "palette": self.palette.tolist(),
             "proportions": [round(float(p), 4) for p in self.proportions]}
        if self.selection is not None:
            d["selection"] = self.selection.to_dict()
        return d


class AutoKSegmenter:
    """K-Means colour segmentation with optional automatic K selection.

    Parameters
    ----------
    k_min, k_max   : search range for K
    color_space    : "lab" | "rgb"
    spatial_weight : weight of (x,y) features (0 = colour only)
    max_side       : down-scale longest image side before processing (None = keep)
    sample_size    : #pixels used to fit / select (the full image is labelled afterwards)
    strategy       : "vote" | "borda" | single criterion name
    """

    def __init__(self, k_min=2, k_max=10, color_space="lab", spatial_weight=0.0,
                 max_side=256, sample_size=20000, strategy="borda", methods=ALL_METHODS,
                 n_init=3, final_n_init=10, random_state=0):
        self.k_min, self.k_max = k_min, k_max
        self.color_space, self.spatial_weight = color_space, spatial_weight
        self.max_side, self.sample_size = max_side, sample_size
        self.strategy, self.methods = strategy, tuple(methods)
        self.n_init, self.final_n_init = n_init, final_n_init
        self.random_state = random_state

    def _sample(self, X):
        if len(X) <= self.sample_size:
            return X
        idx = np.random.default_rng(self.random_state).choice(len(X), self.sample_size,
                                                              replace=False)
        return X[idx]

    def select(self, image: np.ndarray) -> SelectionResult:
        image = resize_image(image, self.max_side)
        X = build_features(image, self.color_space, self.spatial_weight)
        return select_k(self._sample(X), self.k_min, self.k_max, self.strategy,
                        self.methods, self.n_init, random_state=self.random_state)

    def segment(self, image: np.ndarray, k: int | None = None,
                selection: SelectionResult | None = None, ground_truth=None) -> SegmentationResult:
        """Segment ``image``. If ``k`` is None it is chosen automatically."""
        t0 = time.time()
        image = resize_image(np.asarray(image, dtype=np.uint8), self.max_side)
        h, w = image.shape[:2]
        X = build_features(image, self.color_space, self.spatial_weight)
        sample = self._sample(X)
        if k is None:
            selection = selection or select_k(sample, self.k_min, self.k_max, self.strategy,
                                              self.methods, self.n_init,
                                              random_state=self.random_state)
            k = selection.best_k
        km = KMeans(k, n_init=self.final_n_init, random_state=self.random_state).fit(sample)
        labels = assign(X, km.cluster_centers_)[0]

        # relabel by decreasing cluster size so colours are stable / reproducible
        order = np.argsort(-np.bincount(labels, minlength=k))
        remap = np.empty(k, dtype=np.int32)
        remap[order] = np.arange(k)
        labels = remap[labels]
        counts = np.bincount(labels, minlength=k)

        flat = image.reshape(-1, 3).astype(np.float64)
        palette = np.stack([np.bincount(labels, weights=flat[:, c], minlength=k)
                            for c in range(3)], 1) / np.maximum(counts, 1)[:, None]
        palette = np.clip(np.round(palette), 0, 255).astype(np.uint8)
        segmented = palette[labels].reshape(h, w, 3)

        # --- quantitative quality ------------------------------------
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), min(len(X), 20000), replace=False)
        m = {
            "k": int(k),
            "mse": round(mse(image, segmented), 3),
            "psnr_db": round(psnr(image, segmented), 3),
            "silhouette": round(silhouette_score(X[idx], labels[idx], 2000), 4),
            "davies_bouldin": round(davies_bouldin_score(X[idx], labels[idx]), 4),
            "calinski_harabasz": round(calinski_harabasz_score(X[idx], labels[idx]), 2),
            "compression_ratio": round(float(24.0 / (np.log2(max(k, 2))
                                                     + 24.0 * k / (h * w))), 2),
            "image_size": [int(h), int(w)],
            "seconds": round(time.time() - t0, 3),
        }
        if ground_truth is not None:
            gt = resize_image_labels(ground_truth, (h, w))
            m["ari_vs_ground_truth"] = round(adjusted_rand_index(labels.ravel(), gt.ravel()), 4)
        return SegmentationResult(image, labels.reshape(h, w), segmented, palette,
                                  counts / counts.sum(), int(k), selection, m)


def resize_image_labels(gt, shape):
    gt = np.asarray(gt)
    if gt.shape == tuple(shape):
        return gt
    img = Image.fromarray(gt.astype(np.int32), mode="I").resize((shape[1], shape[0]),
                                                               Image.NEAREST)
    return np.asarray(img)
