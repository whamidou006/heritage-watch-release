#!/usr/bin/env python3
"""Corrected noise floor: jitter the spatial grid instead of reseeding the splitter.

Problem found while reviewing: StratifiedGroupKFold is a greedy assignment whose
`shuffle`/`random_state` only breaks ties. On the 64 px set it returns the *identical*
partition for every seed (0 disagreements), so the reported '+/-0.000' measured nothing.
Even at 128 px only 27/838 predictions changed across 3 seeds, so the usual +/-0.004
badly understates uncertainty.

Fix: make the *grouping* itself stochastic. Offsetting the 6x6 grid origin by a random
fraction of a block moves every block boundary, producing genuinely different -- and
still spatially blocked -- partitions. This is the honest noise floor for comparing
configurations.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score

from balance_study import cv
from cleaning_study import load
from compare_gpu import BLOCKS, OUT, set_pca

REPS = 8


def blocks_jittered(x, y, n, ox, oy):
    """6x6 equal-area grid whose origin is shifted by (ox, oy) blocks."""
    def ax(v, o):
        span = np.ptp(v) + 1e-12
        return np.floor((v - v.min()) / span * n + o).astype(int) % (n + 1)
    return ax(x, ox) * (n + 1) + ax(y, oy)


def main() -> None:
    set_pca(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"64 px": ("features_c64.npz", "features_satlas_c64.npz"),
           "128 px": ("features.npz", "features_satlas.npz")}

    print(f"device={dev}  {REPS} jittered grid replicates, merged representation\n")
    print(f"{'config':<12}{'n':>6}{'mean':>9}{'sd':>8}{'min':>9}{'max':>9}   per-replicate")
    print("-" * 96)
    out, keep = {}, {}
    for tag, (a, b) in cfg.items():
        X, y, x, yy = load(a, b)
        labs = sorted(set(y))
        yi = np.searchsorted(labs, y)
        vals = []
        for r in range(REPS):
            rng = np.random.default_rng(100 + r)
            g = blocks_jittered(x, yy, BLOCKS, rng.random(), rng.random())
            vals.append(f1_score(yi, cv(X, yi, g, r, len(labs), dev, True), average="macro"))
        v = np.array(vals)
        keep[tag] = v
        out[tag] = dict(n=len(y), mean=float(v.mean()), sd=float(v.std()),
                        lo=float(v.min()), hi=float(v.max()), vals=[float(t) for t in v])
        print(f"{tag:<12}{len(y):>6}{v.mean():>9.4f}{v.std():>8.4f}{v.min():>9.4f}{v.max():>9.4f}   "
              + " ".join(f"{t:.3f}" for t in v))

    a, b = keep["64 px"], keep["128 px"]
    d = a - b  # paired: replicate r uses the same grid for both configs
    print(f"\npaired 64px - 128px: mean {d.mean():+.4f}  sd {d.std():.4f}  "
          f"wins {int((d > 0).sum())}/{REPS}")
    print(f"noise floor (sd of a single config): ~{max(a.std(), b.std()):.4f} macro-F1")
    out["paired_64_minus_128"] = dict(mean=float(d.mean()), sd=float(d.std()),
                                      wins=int((d > 0).sum()), reps=REPS)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "noise_floor.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {OUT / 'noise_floor.json'}")


if __name__ == "__main__":
    main()
