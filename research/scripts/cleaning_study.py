#!/usr/bin/env python3
"""What does each dataset-cleaning decision actually cost or buy?

The manifest applies four filters (footprint, endpoint imagery, class exclusion,
edge clipping). Two of them are recoverable and therefore testable:

  A. cleaned      4 classes, n=838                          -- the reported system
  B. raw          6 classes, n=864 ('Other' 25 + 'Reconstruction' 1 retained)
                  scored two ways: naive 6-class macro-F1, and macro-F1 restricted
                  to the 4 target classes so it is comparable to A
  C. dedup        4 classes, one label per 32 m cluster      -- chips are 32 m wide,
                  so points closer than that share pixels; this asks whether the
                  headline number leans on overlapping chips

The footprint and missing-imagery filters are not recoverable (no pixels exist),
so they are reported as hard constraints rather than choices.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score

from balance_study import cv
from compare_gpu import BLOCKS, CACHE, OUT, SEEDS, set_pca, spatial_blocks
from cachesel import cname, oname

TARGETS = ["Destruction", "New Construction", "Solar Panel", "Temporary Structure"]


def load(dino_f: str, sat_f: str):
    d = np.load(CACHE / cname(dino_f), allow_pickle=True)
    s = np.load(CACHE / cname(sat_f), allow_pickle=True)
    assert (d["fid"] == s["fid"]).all()
    X = np.hstack([d["f1"], d["f2"], d["f2"] - d["f1"],
                   s["f1"], s["f2"], s["f2"] - s["f1"]])
    return X, d["y"].astype(str), d["x"], d["yy"]


def score(X, y, x, yy, dev, restrict=None):
    """Spatial-CV macro-F1, optionally restricted to a subset of classes."""
    labs = sorted(set(y))
    yi = np.searchsorted(labs, y)
    blk = spatial_blocks(x, yy, BLOCKS)
    vals = []
    for s in range(SEEDS):
        oof = cv(X, yi, blk, s, len(labs), dev, True)
        keep = [labs.index(c) for c in (restrict or labs) if c in labs]
        vals.append(f1_score(yi, oof, average="macro", labels=keep, zero_division=0))
    return float(np.mean(vals)), float(np.std(vals))


def dedup_mask(x, yy, radius_m=32.0):
    """Greedy: keep a label only if no already-kept label lies within radius_m."""
    from scipy.spatial import cKDTree
    m = np.c_[x * np.cos(np.radians(34.34)) * 111320, yy * 111320]
    tree = cKDTree(m)
    keep = np.zeros(len(m), bool)
    taken = np.zeros(len(m), bool)
    for i in range(len(m)):
        if taken[i]:
            continue
        keep[i] = True
        for j in tree.query_ball_point(m[i], radius_m):
            taken[j] = True
    return keep


def main() -> None:
    set_pca(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}

    Xc, yc, xc, yyc = load("features.npz", "features_satlas.npz")
    Xr, yr, xr, yyr = load("features_raw.npz", "features_satlas_raw.npz")

    print(f"device={dev}  spatial CV, {SEEDS} seeds, merged representation\n")
    print(f"{'condition':<34}{'n':>6}{'cls':>5}{'macro-F1':>18}{'vs cleaned':>12}")
    print("-" * 75)

    m, sd = score(Xc, yc, xc, yyc, dev)
    base = m
    out["cleaned_4class"] = dict(n=len(yc), classes=4, macro_f1=m, std=sd)
    print(f"{'A. cleaned (reported)':<34}{len(yc):>6}{4:>5}{m:>12.4f} +/-{sd:.3f}{'--':>12}")

    m6, sd6 = score(Xr, yr, xr, yyr, dev)
    out["raw_6class_all"] = dict(n=len(yr), classes=6, macro_f1=m6, std=sd6)
    print(f"{'B1. raw, 6-class macro':<34}{len(yr):>6}{6:>5}{m6:>12.4f} +/-{sd6:.3f}"
          f"{m6 - base:>+12.4f}")

    m4, sd4 = score(Xr, yr, xr, yyr, dev, restrict=TARGETS)
    out["raw_6class_on_targets"] = dict(n=len(yr), classes=6, macro_f1=m4, std=sd4)
    print(f"{'B2. raw, scored on 4 targets':<34}{len(yr):>6}{6:>5}{m4:>12.4f} +/-{sd4:.3f}"
          f"{m4 - base:>+12.4f}")

    k = dedup_mask(xc, yyc)
    md, sdd = score(Xc[k], yc[k], xc[k], yyc[k], dev)
    out["dedup_32m"] = dict(n=int(k.sum()), classes=4, macro_f1=md, std=sdd)
    print(f"{'C. dedup, 1 label / 32 m':<34}{k.sum():>6}{4:>5}{md:>12.4f} +/-{sdd:.3f}"
          f"{md - base:>+12.4f}")

    print("\nper-class F1, cleaned vs raw-6class (scored on the 4 target classes)")
    labs_r = sorted(set(yr))
    yir = np.searchsorted(labs_r, yr)
    blkr = spatial_blocks(xr, yyr, BLOCKS)
    per_r = np.mean([f1_score(yir, cv(Xr, yir, blkr, s, len(labs_r), dev, True),
                              average=None, labels=range(len(labs_r)), zero_division=0)
                     for s in range(SEEDS)], 0)
    labs_c = sorted(set(yc))
    yic = np.searchsorted(labs_c, yc)
    blkc = spatial_blocks(xc, yyc, BLOCKS)
    per_c = np.mean([f1_score(yic, cv(Xc, yic, blkc, s, len(labs_c), dev, True),
                              average=None, labels=range(len(labs_c)), zero_division=0)
                     for s in range(SEEDS)], 0)
    print(f"{'class':<24}{'cleaned':>10}{'raw':>10}{'delta':>10}")
    pc = {}
    for c in TARGETS:
        a, b = per_c[labs_c.index(c)], per_r[labs_r.index(c)]
        pc[c] = dict(cleaned=float(a), raw=float(b))
        print(f"{c:<24}{a:>10.3f}{b:>10.3f}{b - a:>+10.3f}")
    for c in ("Other", "Reconstruction"):
        if c in labs_r:
            print(f"{c:<24}{'--':>10}{per_r[labs_r.index(c)]:>10.3f}")
            pc[c] = dict(cleaned=None, raw=float(per_r[labs_r.index(c)]))
    out["per_class"] = pc

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / oname("cleaning_study.json")).write_text(json.dumps(out, indent=1))
    print(f"\nsaved {OUT / oname('cleaning_study.json')}")


if __name__ == "__main__":
    main()
