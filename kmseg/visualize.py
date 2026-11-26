"""Matplotlib figures for the report / README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY = "#1F3864"
ACC = "#C0392B"


def plot_selection(sel, path=None, title="Automatic selection of K"):
    ks = sel.k_values
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("Elbow method (WCSS / inertia)", sel.inertia, "elbow", "inertia"),
        ("Silhouette score (maximise)", sel.silhouette, "silhouette", "score"),
        ("Davies-Bouldin index (minimise)", sel.davies_bouldin, "davies_bouldin", "index"),
        ("Calinski-Harabasz index (maximise)", sel.calinski_harabasz, "calinski_harabasz", "index"),
    ]
    for a, (t, y, key, yl) in zip(ax.ravel()[:4], panels):
        a.plot(ks, y, "o-", color=NAVY)
        if key in sel.chosen:
            a.axvline(sel.chosen[key], color=ACC, ls="--", label=f"chosen K = {sel.chosen[key]}")
            a.legend()
        a.set_title(t); a.set_xlabel("K"); a.set_ylabel(yl); a.grid(alpha=.3)
        a.set_xticks(ks)
    a = ax[1, 1]
    if sel.gap is not None:
        a.errorbar(ks, sel.gap, yerr=sel.gap_se, fmt="o-", color=NAVY, capsize=3)
        if "gap" in sel.chosen:
            a.axvline(sel.chosen["gap"], color=ACC, ls="--", label=f"chosen K = {sel.chosen['gap']}")
            a.legend()
        a.set_title("Gap statistic (1-SE rule)"); a.set_xlabel("K"); a.set_ylabel("gap")
        a.set_xticks(ks); a.grid(alpha=.3)
    else:
        a.axis("off")
    a = ax[1, 2]
    names = list(sel.chosen)
    vals = [sel.chosen[n] for n in names]
    bars = a.bar(range(len(names)), vals, color=NAVY)
    a.axhline(sel.best_k, color=ACC, ls="--", label=f"final K = {sel.best_k} ({sel.strategy})")
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v + .1, str(v), ha="center")
    a.set_xticks(range(len(names)))
    a.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    a.set_title("K proposed by each criterion"); a.set_ylabel("K"); a.legend()
    fig.suptitle(title, fontsize=15, color=NAVY, fontweight="bold")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130); plt.close(fig)
    return fig


def plot_segmentation(res, path=None, title=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, .7]})
    ax[0].imshow(res.image); ax[0].set_title("Original")
    ax[1].imshow(res.segmented)
    ax[1].set_title(f"K-Means segmentation (K = {res.k}, PSNR {res.metrics['psnr_db']:.1f} dB)")
    for a in ax[:2]:
        a.axis("off")
    pal = res.palette / 255.0
    y = np.arange(res.k)
    ax[2].barh(y, res.proportions * 100, color=pal, edgecolor="k")
    ax[2].set_yticks(y); ax[2].set_yticklabels([f"C{i}" for i in y]); ax[2].invert_yaxis()
    ax[2].set_xlabel("% of pixels"); ax[2].set_title("Cluster palette")
    if title:
        fig.suptitle(title, fontsize=14, color=NAVY, fontweight="bold")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130); plt.close(fig)
    return fig


def plot_k_gallery(image, results, best_k=None, path=None):
    n = len(results) + 1
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    ax[0].imshow(image); ax[0].set_title("Original"); ax[0].axis("off")
    for a, r in zip(ax[1:], results):
        a.imshow(r.segmented)
        star = "  *" if r.k == best_k else ""
        a.set_title(f"K={r.k}{star}\nPSNR {r.metrics['psnr_db']:.1f} dB", fontsize=10,
                    color=ACC if r.k == best_k else "black")
        a.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120); plt.close(fig)
    return fig
