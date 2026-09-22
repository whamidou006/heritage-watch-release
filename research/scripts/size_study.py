#!/usr/bin/env python3
"""Does class SIZE or class CONFUSABILITY explain the per-class spread?

balance_study.py showed reweighting is a no-op (-0.001 macro-F1), and that
Temporary Structure (n=52) outscores Destruction (n=126) by +0.22 F1. That
already suggests size is not the driver. This is the direct test:

  downsample New Construction 352 -> 126 (Destruction's exact size) and remeasure.

If New Construction holds well above Destruction's 0.455 at identical support,
the bottleneck is confusability, not imbalance, and collecting more Destruction
labels will not fix it.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score

from balance_study import cv
from compare_gpu import BLOCKS, CACHE, OUT, SEEDS, set_pca, spatial_blocks
from cachesel import cname, oname

TARGET = "New Construction"
SIZES = [352, 250, 180, 126, 90, 52]


def main() -> None:
    set_pca(0)
    dino = np.load(CACHE / cname("features.npz"), allow_pickle=True)
    sat = np.load(CACHE / cname("features_satlas.npz"), allow_pickle=True)
    y = dino["y"].astype(str)
    labs = sorted(set(y))
    X = np.hstack([dino["f1"], dino["f2"], dino["f2"] - dino["f1"],
                   sat["f1"], sat["f2"], sat["f2"] - sat["f1"]])
    blk = spatial_blocks(dino["x"], dino["yy"], BLOCKS)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ti = labs.index(TARGET)
    di = labs.index("Destruction")
    idx_t = np.flatnonzero(y == TARGET)

    print(f"device={dev}  downsampling '{TARGET}' (n={len(idx_t)}) "
          f"to match smaller classes; spatial CV, {SEEDS} seeds\n")
    print(f"{'n(' + TARGET + ')':<22}{'F1 target':>11}{'F1 Destruction':>17}{'macro-F1':>11}")
    print("-" * 61)

    out = {}
    for k in SIZES:
        f_t, f_d, mac = [], [], []
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 + s)
            keep = np.ones(len(y), bool)
            drop = rng.permutation(idx_t)[k:]
            keep[drop] = False
            ysub = y[keep]
            yi = np.searchsorted(labs, ysub)
            oof = cv(X[keep], yi, blk[keep], s, len(labs), dev, True)
            per = f1_score(yi, oof, average=None, labels=range(len(labs)))
            f_t.append(per[ti]); f_d.append(per[di]); mac.append(per.mean())
        out[k] = dict(target_f1=float(np.mean(f_t)), destr_f1=float(np.mean(f_d)),
                      macro_f1=float(np.mean(mac)))
        print(f"{k:<22}{np.mean(f_t):>8.3f} +/-{np.std(f_t):.3f}"
              f"{np.mean(f_d):>12.3f} +/-{np.std(f_d):.3f}{np.mean(mac):>11.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / oname("size_study.json")).write_text(json.dumps(out, indent=1))
    print(f"\nsaved {OUT / oname('size_study.json')}")


if __name__ == "__main__":
    main()
