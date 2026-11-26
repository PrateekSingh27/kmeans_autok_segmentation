#!/usr/bin/env python3
"""Command-line interface: K-Means image segmentation with automatic K selection.

Examples
--------
  python main.py --demo                          # run on all bundled samples
  python main.py --image samples/astronaut.jpg   # auto-select K
  python main.py --image my.jpg --kmax 12 --color-space lab --strategy borda
  python main.py --image my.jpg --k 5            # force K
  python main.py --synthetic 5 --noise 12        # synthetic image with known K
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

from kmseg import AutoKSegmenter, ALL_METHODS, load_image, make_synthetic_image
from kmseg.visualize import plot_k_gallery, plot_segmentation, plot_selection


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", nargs="+", help="path(s) to input image(s)")
    src.add_argument("--synthetic", type=int, metavar="K_TRUE", help="generate a synthetic image with K_TRUE regions")
    src.add_argument("--demo", action="store_true", help="run on every image in ./samples")
    p.add_argument("--noise", type=float, default=10.0, help="noise std for --synthetic (default 10)")
    p.add_argument("--k", type=int, default=None, help="fixed K (skip automatic selection)")
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=10)
    p.add_argument("--color-space", choices=["lab", "rgb"], default="lab")
    p.add_argument("--spatial-weight", type=float, default=0.0, help="weight of (x,y) features, 0 = colour only")
    p.add_argument("--strategy", default="borda", help="borda | vote | elbow | silhouette | davies_bouldin | calinski_harabasz | gap")
    p.add_argument("--max-side", type=int, default=256, help="resize longest side (0 = keep original)")
    p.add_argument("--sample-size", type=int, default=20000)
    p.add_argument("--no-gap", action="store_true", help="skip the (slower) gap statistic")
    p.add_argument("--gallery", type=int, nargs="*", default=None, help="also save a gallery for these K values (default: kmin..kmax step)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results", help="output directory")
    return p.parse_args()


def process(name, image, gt, args, seg):
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    print(f"\n=== {name}  {image.shape[1]}x{image.shape[0]} ===")
    res = seg.segment(image, k=args.k, ground_truth=gt)
    if res.selection is not None:
        print(res.selection.table())
        print("criteria choices:", res.selection.chosen)
        plot_selection(res.selection, os.path.join(out, "k_selection.png"),
                       title=f"Automatic K selection - {name}")
        json.dump(res.selection.to_dict(), open(os.path.join(out, "selection.json"), "w"), indent=2)
        with open(os.path.join(out, "selection.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["K", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz", "gap"])
            s = res.selection
            for i, k in enumerate(s.k_values):
                w.writerow([k, s.inertia[i], s.silhouette[i], s.davies_bouldin[i],
                            s.calinski_harabasz[i], s.gap[i] if s.gap is not None else ""])
    print(f"-> K = {res.k}   metrics: {res.metrics}")
    plot_segmentation(res, os.path.join(out, "segmentation.png"),
                      title=f"{name}: K = {res.k}")
    Image.fromarray(res.image).save(os.path.join(out, "original.png"))
    Image.fromarray(res.segmented).save(os.path.join(out, "segmented.png"))
    np.save(os.path.join(out, "labels.npy"), res.labels)
    json.dump(res.to_dict(), open(os.path.join(out, "report.json"), "w"), indent=2)

    ks = args.gallery if args.gallery else list(range(args.kmin, args.kmax + 1, 2))
    if args.gallery is not None:
        gal = [seg.segment(image, k=k) for k in ks]
        plot_k_gallery(res.image, gal, res.k, os.path.join(out, "k_gallery.png"))
    print(f"saved to {out}/")
    return res


def main():
    args = parse()
    seg = AutoKSegmenter(args.kmin, args.kmax, args.color_space, args.spatial_weight,
                         args.max_side or None, args.sample_size, args.strategy,
                         tuple(m for m in ALL_METHODS if not (args.no_gap and m == "gap")),
                         random_state=args.seed)
    jobs = []
    if args.synthetic:
        img, gt = make_synthetic_image(args.synthetic, 160, args.noise, args.seed)
        jobs.append((f"synthetic_k{args.synthetic}", img, gt))
    else:
        paths = args.image or sorted(glob.glob("samples/*.*"))
        if not paths:
            sys.exit("No input. Use --image PATH, --synthetic K or --demo.")
        for p in paths:
            jobs.append((os.path.splitext(os.path.basename(p))[0],
                         load_image(p, args.max_side or None), None))
    for name, img, gt in jobs:
        process(name, img, gt, args, seg)


if __name__ == "__main__":
    main()
