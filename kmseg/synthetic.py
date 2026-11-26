"""Synthetic test images with a *known* number of segments.

Having ground truth lets us measure how often each criterion recovers the
true K, and the segmentation accuracy (Adjusted Rand Index).
"""
import colorsys

import numpy as np


def make_synthetic_image(k=4, size=128, noise=10.0, seed=0, blur_edges=True):
    """Return (image uint8 (size,size,3), ground-truth labels (size,size)).

    Regions are a warped Voronoi partition of ``k`` random sites; every
    region gets a well-separated colour; Gaussian noise (std = ``noise``
    grey-levels) is added to every channel.
    """
    rng = np.random.default_rng(seed)
    # well separated hues, shuffled; random saturation / value
    hues = (np.arange(k) / k + rng.uniform(0, 1 / k * 0.3)) % 1.0
    rng.shuffle(hues)
    colors = np.array([colorsys.hsv_to_rgb(h, rng.uniform(0.55, 0.95),
                                           rng.uniform(0.55, 0.98)) for h in hues]) * 255

    # Poisson-like site placement: reject sites that are too close
    sites = []
    min_d = 0.9 / np.sqrt(k + 1) * 0.6
    while len(sites) < k:
        p = rng.uniform(0.05, 0.95, 2)
        if all(np.hypot(*(p - q)) > min_d for q in sites):
            sites.append(p)
    sites = np.array(sites)

    yy, xx = np.mgrid[0:size, 0:size] / (size - 1.0)
    # smooth sinusoidal warp so boundaries are curved
    wx = xx + 0.03 * np.sin(2 * np.pi * (3 * yy + rng.uniform()))
    wy = yy + 0.03 * np.sin(2 * np.pi * (3 * xx + rng.uniform()))
    d = (wx[..., None] - sites[:, 0]) ** 2 + (wy[..., None] - sites[:, 1]) ** 2
    gt = d.argmin(-1)
    img = colors[gt] + rng.normal(0, noise, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8), gt.astype(np.int32)
