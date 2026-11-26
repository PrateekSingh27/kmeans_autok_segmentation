import numpy as np
import pytest

from kmseg.colorspace import lab_to_rgb, rgb_to_lab
from kmseg.kmeans import KMeans
from kmseg.metrics import (adjusted_rand_index, calinski_harabasz_score,
                           davies_bouldin_score, elbow_k, psnr, silhouette_score)


def blobs(n=300, k=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [6, 0], [3, 6], [9, 7]])[:k]
    X = np.vstack([c + rng.normal(0, .5, (n // k, 2)) for c in centers])
    y = np.repeat(np.arange(k), n // k)
    return X, y


def test_kmeans_recovers_blobs():
    X, y = blobs()
    km = KMeans(3, n_init=5, random_state=0).fit(X)
    assert adjusted_rand_index(km.labels_, y) > 0.99
    assert km.inertia_ > 0 and km.cluster_centers_.shape == (3, 2)


def test_predict_matches_labels():
    X, _ = blobs()
    km = KMeans(3, random_state=1).fit(X)
    assert np.array_equal(km.predict(X), km.labels_)


def test_inertia_decreases_with_k():
    X, _ = blobs(k=4)
    w = [KMeans(k, random_state=0).fit(X).inertia_ for k in range(1, 7)]
    assert all(a >= b - 1e-6 for a, b in zip(w, w[1:]))


def test_deterministic_with_seed():
    X, _ = blobs()
    a = KMeans(3, random_state=5).fit(X).cluster_centers_
    b = KMeans(3, random_state=5).fit(X).cluster_centers_
    assert np.allclose(a, b)


def test_empty_cluster_handling():
    X = np.vstack([np.zeros((50, 2)), np.ones((50, 2))])
    km = KMeans(4, init="random", n_init=1, random_state=0).fit(X)
    assert km.cluster_centers_.shape == (4, 2) and np.isfinite(km.cluster_centers_).all()


def test_matches_sklearn_inertia():
    sk = pytest.importorskip("sklearn.cluster")
    X, _ = blobs(k=4)
    ours = KMeans(4, n_init=10, random_state=0).fit(X).inertia_
    ref = sk.KMeans(4, n_init=10, random_state=0).fit(X).inertia_
    assert ours == pytest.approx(ref, rel=1e-3)


def test_metrics_match_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    X, _ = blobs(k=3)
    lab = KMeans(3, random_state=0).fit(X).labels_
    assert silhouette_score(X, lab, sample_size=10_000) == pytest.approx(skm.silhouette_score(X, lab), abs=1e-6)
    assert davies_bouldin_score(X, lab) == pytest.approx(skm.davies_bouldin_score(X, lab), abs=1e-6)
    assert calinski_harabasz_score(X, lab) == pytest.approx(skm.calinski_harabasz_score(X, lab), rel=1e-6)


def test_ari():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_index(a, a) == 1.0
    assert adjusted_rand_index(a, np.array([2, 2, 0, 0, 1, 1])) == 1.0   # label permutation invariant
    sk = pytest.importorskip("sklearn.metrics")
    b = np.array([0, 1, 1, 1, 2, 0])
    assert adjusted_rand_index(a, b) == pytest.approx(sk.adjusted_rand_score(a, b))


def test_elbow_detects_knee():
    ks = np.arange(2, 11)
    inertia = np.array([1000, 400, 100, 90, 82, 76, 71, 67, 64.])
    assert elbow_k(ks, inertia) == 4


def test_psnr():
    a = np.zeros((4, 4, 3), np.uint8)
    assert psnr(a, a) == float("inf")
    assert psnr(a, a + 1) == pytest.approx(48.13, abs=0.01)


def test_lab_roundtrip():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 1, (50, 3))
    assert np.allclose(lab_to_rgb(rgb_to_lab(rgb)), rgb, atol=1e-4)
    assert rgb_to_lab(np.array([1.0, 1, 1]))[0] == pytest.approx(100, abs=0.1)   # white
    assert rgb_to_lab(np.array([0.0, 0, 0]))[0] == pytest.approx(0, abs=0.1)     # black
