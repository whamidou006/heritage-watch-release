#!/usr/bin/env python3
"""Re-run the main comparison under the corrected (jittered-grid) protocol.

The original table used 3 StratifiedGroupKFold seeds, which we discovered vary the
partition only through tie-breaking -- on one dataset not at all. This re-measures
every representation over 8 jittered spatial grids, *paired*: replicate r uses the
same grid for all representations, so differences can be tested per replicate rather
than compared across independent noisy means.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score

from balance_study import cv
from compare_gpu import BLOCKS, CACHE, OUT, set_pca
from noise_floor import REPS, blocks_jittered
import decision as D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", default="features.npz")
    ap.add_argument("--satlas", default="features_satlas.npz")
    ap.add_argument("--out", default="compare_jitter.json")
    args = ap.parse_args()

    set_pca(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(CACHE / args.dino, allow_pickle=True)
    s = np.load(CACHE / args.satlas, allow_pickle=True)
    assert (d["fid"] == s["fid"]).all()
    y = d["y"].astype(str)
    labs = sorted(set(y))
    yi = np.searchsorted(labs, y)
    d1, d2, s1, s2, smi = d["f1"], d["f2"], s["f1"], s["f2"], s["fmi"]

    cands = {
        "DINOv2 post-event":     d2,
        "DINOv2 both+diff":      np.hstack([d1, d2, d2 - d1]),
        "Satlas-SI post-event":  s2,
        "Satlas-SI both+diff":   np.hstack([s1, s2, s2 - s1]),
        "Satlas-MI fused":       smi,
        "Satlas-MI + SI diff":   np.hstack([smi, s2 - s1]),
        "Merged DINOv2+Satlas":  np.hstack([d1, d2, d2 - d1, s1, s2, s2 - s1]),
    }

    grids = []
    for r in range(REPS):
        rng = np.random.default_rng(100 + r)
        grids.append(blocks_jittered(d["x"], d["yy"], BLOCKS, rng.random(), rng.random()))

    print(f"device={dev}  {REPS} paired jittered grids, n={len(y)}\n")
    print(f"{'representation':<24}{'dim':>6}{'mean':>9}{'sd':>8}{'min':>8}{'max':>8}")
    print("-" * 63)
    res = {}
    for name, X in cands.items():
        v = np.array([f1_score(yi, cv(X, yi, grids[r], r, len(labs), dev, True), average="macro")
                      for r in range(REPS)])
        res[name] = v
        print(f"{name:<24}{X.shape[1]:>6}{v.mean():>9.4f}{v.std():>8.4f}{v.min():>8.4f}{v.max():>8.4f}")

    best = max(res, key=lambda k: res[k].mean())
    print(f"\nbest: {best}  {res[best].mean():.4f}")
    print(f"\npaired vs {best} (same grid per replicate):")
    print(f"{'representation':<24}{'delta':>9}{'sd':>8}{'wins':>7}   verdict")
    print("-" * 70)
    out = {"reps": REPS, "best": best, "per_config": {}, "paired_vs_best": {}}
    for name, v in res.items():
        out["per_config"][name] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                       vals=[float(t) for t in v])
        if name == best:
            continue
        dd = res[best] - v
        vd = D.verdict(dd)
        out["paired_vs_best"][name] = vd
        print(D.line(name, vd, 24))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(out, indent=1))
    print(f"\nsaved {OUT / args.out}")


if __name__ == "__main__":
    main()
