#!/usr/bin/env python3
"""Benchmark the K-selection criteria on synthetic images with KNOWN K.

For every (K_true, noise, seed) a synthetic image is generated, every
criterion proposes a K, and we report exact-match accuracy, mean absolute
error and the segmentation ARI obtained with the finally chosen K.

  python benchmark.py                 # full run (~3-5 min)
  python benchmark.py --quick         # small run (~1 min)
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kmseg import AutoKSegmenter, make_synthetic_image
from kmseg.selection import ALL_METHODS, _combine

NAVY = "#1F3864"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/benchmark")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    k_true = [2, 3, 4, 5, 6, 7, 8] if not a.quick else [3, 5]
    noises = [8, 20, 35] if not a.quick else [10, 30]
    seeds = range(5) if not a.quick else range(2)
    seg = AutoKSegmenter(2, 10, "lab", max_side=None, sample_size=6000, strategy="borda")
    rows = []
    t0 = time.time()
    for kt in k_true:
        for nz in noises:
            for sd in seeds:
                img, gt = make_synthetic_image(kt, 96, nz, seed=100 * kt + sd)
                sel = seg.select(img)
                row = dict(k_true=kt, noise=nz, seed=sd, **{m: sel.chosen[m] for m in ALL_METHODS})
                for strat in ("vote", "borda"):
                    row[strat] = _combine(strat, sel.chosen, sel.k_values, sel.silhouette,
                                          sel.davies_bouldin, sel.calinski_harabasz)
                r = seg.segment(img, k=row["vote"], ground_truth=gt)
                row["ari_vote"] = r.metrics["ari_vs_ground_truth"]
                rows.append(row)
        print(f"K_true={kt} done ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(a.out, "benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    methods = list(ALL_METHODS) + ["vote", "borda"]
    kt = np.array([r["k_true"] for r in rows])
    stats = {}
    for m in methods:
        pred = np.array([r[m] for r in rows])
        stats[m] = (np.mean(pred == kt) * 100, np.mean(np.abs(pred - kt)))
    lines = ["| Method | Exact-match accuracy (%) | Mean abs. error |", "|---|---|---|"]
    for m in methods:
        lines.append(f"| {m} | {stats[m][0]:.1f} | {stats[m][1]:.2f} |")
    noise_arr = np.array([r["noise"] for r in rows])
    lines += ["", "Accuracy (%) by noise level (noise std in grey-levels):", "",
              "| Method | " + " | ".join(f"noise {n}" for n in noises) + " |",
              "|---|" + "---|" * len(noises)]
    for m in methods:
        pred = np.array([r[m] for r in rows])
        lines.append(f"| {m} | " + " | ".join(
            f"{np.mean(pred[noise_arr == n] == kt[noise_arr == n]) * 100:.1f}" for n in noises) + " |")
    lines += ["", f"Mean ARI of segmentation using the voted K: "
                  f"{np.mean([r['ari_vote'] for r in rows]):.4f}",
              f"Runs: {len(rows)}  ({time.time() - t0:.0f}s)"]
    open(os.path.join(a.out, "benchmark.md"), "w").write("\n".join(lines))
    print("\n".join(lines))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].bar(methods, [stats[m][0] for m in methods], color=NAVY)
    ax[0].set_ylabel("exact-match accuracy (%)"); ax[0].set_title("Recovering the true K")
    ax[0].tick_params(axis="x", rotation=30)
    for i, m in enumerate(methods):
        ax[0].text(i, stats[m][0] + 1, f"{stats[m][0]:.0f}", ha="center")
    for m in ("elbow", "silhouette", "gap", "vote"):
        acc = [np.mean(np.array([r[m] for r in rows if r["k_true"] == k]) == k) * 100 for k in k_true]
        ax[1].plot(k_true, acc, "o-", label=m)
    ax[1].set_xlabel("true K"); ax[1].set_ylabel("accuracy (%)")
    ax[1].set_title("Accuracy vs. number of true segments"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "benchmark.png"), dpi=130)


if __name__ == "__main__":
    main()
