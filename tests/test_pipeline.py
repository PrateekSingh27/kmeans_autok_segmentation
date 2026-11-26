import numpy as np
import pytest

from kmseg import AutoKSegmenter, make_synthetic_image, select_k, build_features


@pytest.mark.parametrize("k_true", [3, 4])
def test_auto_k_recovers_truth_on_clean_synthetic(k_true):
    img, gt = make_synthetic_image(k_true, 96, noise=6, seed=k_true)
    res = AutoKSegmenter(max_side=None).segment(img, ground_truth=gt)
    assert res.k == k_true
    assert res.metrics["ari_vs_ground_truth"] > 0.97


def test_fixed_k_and_output_shapes():
    img, _ = make_synthetic_image(4, 64, 10, seed=0)
    res = AutoKSegmenter(max_side=None).segment(img, k=5)
    assert res.k == 5 and res.selection is None
    assert res.labels.shape == (64, 64) and res.segmented.shape == (64, 64, 3)
    assert len(np.unique(res.labels)) <= 5
    assert res.palette.shape == (5, 3) and abs(res.proportions.sum() - 1) < 1e-9
    # largest cluster gets label 0
    assert res.proportions[0] == res.proportions.max()


def test_select_k_strategies():
    img, _ = make_synthetic_image(3, 64, 6, seed=1)
    X = build_features(img)
    for s in ("vote", "borda", "silhouette", "elbow", "gap"):
        r = select_k(X, 2, 7, strategy=s, gap_B=4)
        assert 2 <= r.best_k <= 7
    with pytest.raises(ValueError):
        select_k(X, 2, 6, strategy="nonsense")


def test_uniform_image_raises():
    img = np.full((20, 20, 3), 120, np.uint8)
    with pytest.raises(ValueError):
        select_k(build_features(img))


def test_features_spatial_and_spaces():
    img = np.random.default_rng(0).integers(0, 255, (10, 12, 3), dtype=np.uint8)
    assert build_features(img, "rgb").shape == (120, 3)
    assert build_features(img, "lab", 0.5).shape == (120, 5)
    with pytest.raises(ValueError):
        build_features(img, "hsv")


def test_rgb_and_lab_both_work():
    img, gt = make_synthetic_image(3, 64, 6, seed=2)
    for cs in ("rgb", "lab"):
        res = AutoKSegmenter(color_space=cs, max_side=None).segment(img, k=3, ground_truth=gt)
        assert res.metrics["ari_vs_ground_truth"] > 0.95
