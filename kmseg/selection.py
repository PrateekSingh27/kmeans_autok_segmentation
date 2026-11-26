"""Automatic selection of the number of clusters K.

Five quantitative criteria are evaluated over a range of K:

=================  ==========================================  =========
criterion          rule                                        direction
=================  ==========================================  =========
elbow (WCSS)       knee of the inertia curve                   knee
silhouette         largest mean silhouette                     maximise
davies_bouldin     smallest DB index                           minimise
calinski_harabasz  largest CH (variance-ratio) index           maximise
gap                Tibshirani 1-SE rule                        see paper
=================  ==========================================  =========

The per-criterion choices are combined by a *strategy*:

* ``"vote"``  - majority vote (ties broken by the silhouette score)
* ``"borda"`` - Borda rank aggregation of silhouette / DB / CH
* or the name of a single criterion, e.g. ``"silhouette"``
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .kmeans import KMeans
from .metrics import (calinski_harabasz_score, davies_bouldin_score, elbow_k,
                      gap_choose_k, gap_statistic, silhouette_score)

ALL_METHODS = ("elbow", "silhouette", "davies_bouldin", "calinski_harabasz", "gap")


@dataclass
class SelectionResult:
    k_values: np.ndarray
    inertia: np.ndarray
    silhouette: np.ndarray
    davies_bouldin: np.ndarray
    calinski_harabasz: np.ndarray
    gap: np.ndarray | None
    gap_se: np.ndarray | None
    chosen: dict = field(default_factory=dict)   # criterion -> K
    best_k: int = 2
    strategy: str = "vote"
    seconds: float = 0.0

    def table(self) -> str:
        head = f"{'K':>3} {'inertia':>11} {'silhouette':>10} {'DB':>8} {'CH':>10} {'gap':>8}"
        rows = [head, "-" * len(head)]
        for i, k in enumerate(self.k_values):
            g = f"{self.gap[i]:8.3f}" if self.gap is not None else f"{'-':>8}"
            mark = " <== best" if k == self.best_k else ""
            rows.append(f"{k:>3} {self.inertia[i]:11.2f} {self.silhouette[i]:10.4f} "
                        f"{self.davies_bouldin[i]:8.4f} {self.calinski_harabasz[i]:10.1f} {g}{mark}")
        return "\n".join(rows)

    def to_dict(self) -> dict:
        d = {"k_values": self.k_values.tolist(), "inertia": self.inertia.tolist(),
             "silhouette": self.silhouette.tolist(),
             "davies_bouldin": self.davies_bouldin.tolist(),
             "calinski_harabasz": self.calinski_harabasz.tolist(),
             "chosen": {k: int(v) for k, v in self.chosen.items()},
             "best_k": int(self.best_k), "strategy": self.strategy,
             "seconds": round(self.seconds, 3)}
        if self.gap is not None:
            d["gap"], d["gap_se"] = self.gap.tolist(), self.gap_se.tolist()
        return d


def _combine(strategy, chosen, k_values, sil, db, ch):
    if strategy in chosen:
        return int(chosen[strategy])
    if strategy == "vote":
        counts = Counter(chosen.values())
        top = max(counts.values())
        tied = [k for k, c in counts.items() if c == top]
        if len(tied) == 1:
            return int(tied[0])
        idx = {int(k): i for i, k in enumerate(k_values)}
        return int(max(tied, key=lambda k: sil[idx[int(k)]]))
    if strategy == "borda":
        # rank 0 = best; sum ranks over the three internal indices
        r = (np.argsort(np.argsort(-sil)) + np.argsort(np.argsort(db))
             + np.argsort(np.argsort(-ch)))
        return int(k_values[int(np.argmin(r))])
    raise ValueError(f"unknown strategy '{strategy}'")


def select_k(X, k_min=2, k_max=10, strategy="vote", methods=ALL_METHODS,
             n_init=3, sil_sample=2000, gap_B=10, gap_sample=3000,
             random_state=0) -> SelectionResult:
    """Evaluate K in [k_min, k_max] on feature matrix ``X`` and pick the best."""
    t0 = time.time()
    X = np.ascontiguousarray(X, dtype=np.float32)
    n_unique = len(np.unique(np.round(X, 4), axis=0))
    if n_unique < 3:
        raise ValueError("Image has fewer than 3 distinct colours - nothing to select.")
    k_max = min(k_max, n_unique)
    k_min = max(2, min(k_min, k_max - 1))
    ks = np.arange(k_min, k_max + 1)

    inertia, sil, db, ch = [], [], [], []
    for k in ks:
        km = KMeans(int(k), n_init=n_init, random_state=random_state).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_, sil_sample, random_state))
        db.append(davies_bouldin_score(X, km.labels_))
        ch.append(calinski_harabasz_score(X, km.labels_))
    inertia, sil, db, ch = map(np.array, (inertia, sil, db, ch))

    gap = gap_se = None
    chosen = {
        "elbow": elbow_k(ks, inertia),
        "silhouette": int(ks[np.argmax(sil)]),
        "davies_bouldin": int(ks[np.argmin(db)]),
        "calinski_harabasz": int(ks[np.argmax(ch)]),
    }
    if "gap" in methods:
        # gap statistic is evaluated on a (smaller) subsample for speed
        rng = np.random.default_rng(random_state)
        sub = X if len(X) <= gap_sample else X[rng.choice(len(X), gap_sample, replace=False)]
        log_w = [np.log(max(KMeans(int(k), n_init=n_init, random_state=random_state)
                            .fit(sub).inertia_, 1e-12)) for k in ks]
        gap, gap_se = gap_statistic(sub, ks, log_w, B=gap_B, random_state=random_state)
        chosen["gap"] = gap_choose_k(ks, gap, gap_se)
    chosen = {m: chosen[m] for m in methods if m in chosen}
    if strategy not in ("vote", "borda") and strategy not in chosen:
        raise ValueError(f"strategy '{strategy}' not among methods {list(chosen)}")
    best = _combine(strategy, chosen, ks, sil, db, ch)
    return SelectionResult(ks, inertia, sil, db, ch, gap, gap_se, chosen, best,
                           strategy, time.time() - t0)
