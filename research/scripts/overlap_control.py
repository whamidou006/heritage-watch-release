#!/usr/bin/env python3
"""How much of the headline score comes from chips that physically overlap?

The protocol blocks SPACE by assigning each label to a cell of a 6x6 grid and
keeping whole cells inside a fold.  That stops a fold from being scored on the
same neighbourhood it trained on, but it does not stop two labels that sit a few
metres apart from landing in adjacent cells on opposite sides of a fold
boundary.  When that happens their 128 px (32 m) windows overlap, so the model
has literally seen some of the test pixels during training.

This measures the cost of that, rather than arguing about it.  Three arms, the
same folds and the same classifier throughout:

  keep    the shipped protocol, nothing removed                 (upper bound)
  purge   drop every TRAINING row whose window overlaps a test
          row's window AND shares an acquisition scene with it   (the honest number)
  purge_all  drop every TRAINING row whose window overlaps a
          test row's window, whatever the dates                  (most conservative)

Only training rows are dropped, so all three arms are scored on exactly the same
test rows and the difference is attributable to the removed pixels alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from cachesel import cname, oname
from balance_study import fit_logreg  # same estimator, with the `balanced` switch
from compare_gpu import (BLOCKS, CACHE, FOLDS, OUT, fit_transform,
                         set_pca, spatial_blocks)

REPS = 8
CHIP_PX = 128
PX_M = 0.25
LAT = 34.34                      # site latitude, for degrees -> metres


def to_metres(x, yy):
    return np.c_[x * np.cos(np.radians(LAT)) * 111320.0, yy * 111320.0]


def overlap_sets(m, scenes, chip_m):
    """For each row, the set of rows whose window overlaps it.

    Windows are axis-aligned and `chip_m` across, so two overlap iff both
    coordinate separations are strictly below chip_m.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(m)
    # Chebyshev ball of radius chip_m contains every possible overlap
    cand = tree.query_ball_point(m, chip_m * np.sqrt(2))
    out = []
    for i, cs in enumerate(cand):
        hit = []
        for j in cs:
            if j == i:
                continue
            if abs(m[j, 0] - m[i, 0]) < chip_m and abs(m[j, 1] - m[i, 1]) < chip_m:
                hit.append(j)
        out.append(hit)
    return out


def main() -> None:
    set_pca(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(CACHE / cname("features.npz"), allow_pickle=True)
    s = np.load(CACHE / cname("features_satlas.npz"), allow_pickle=True)
    assert (d["fid"] == s["fid"]).all()

    y = d["y"].astype(str)
    labs = sorted(set(y))
    yi = np.searchsorted(labs, y)
    X = np.hstack([d["f1"], d["f2"], d["f2"] - d["f1"],
                   s["f1"], s["f2"], s["f2"] - s["f1"]])
    x, yy, pair = d["x"], d["yy"], d["pair"].astype(str)
    blk = spatial_blocks(x, yy, BLOCKS)
    m = to_metres(x, yy)
    chip_m = CHIP_PX * PX_M

    ov = overlap_sets(m, pair, chip_m)
    n_with = sum(1 for h in ov if h)
    print(f"device={dev}  n={len(y)}  chip {CHIP_PX}px = {chip_m:.0f} m")
    print(f"rows whose window overlaps at least one other row: {n_with} "
          f"({n_with/len(y):.1%})\n")

    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(yi, dtype=torch.long, device=dev)

    arms = {"keep": None, "purge": "scene", "purge_all": "any"}
    scores = {k: [] for k in arms}
    dropped = {k: [] for k in arms}
    affected = []

    for seed in range(REPS):
        sp = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed)
        splits = list(sp.split(X, yi, blk))
        # how many test rows have an overlapping partner in their training fold
        aff = 0
        for tr, te in splits:
            trs = set(tr.tolist())
            for i in te:
                if any(j in trs and pair[j] == pair[i] for j in ov[i]):
                    aff += 1
        affected.append(aff)

        for arm, mode in arms.items():
            oof = np.empty(len(yi), np.int64)
            drop_tot = 0
            for tr, te in splits:
                if mode is None:
                    keep = tr
                else:
                    tes = set(te.tolist())
                    bad = set()
                    for i in te:
                        for j in ov[i]:
                            if j in tes:
                                continue
                            if mode == "any" or pair[j] == pair[i]:
                                bad.add(j)
                    keep = np.array([t for t in tr if t not in bad])
                    drop_tot += len(tr) - len(keep)
                tr_t = torch.as_tensor(keep, device=dev)
                te_t = torch.as_tensor(te, device=dev)
                Xtr, Xte = fit_transform(Xt[tr_t], Xt[te_t], X.shape[1])
                W, b = fit_logreg(Xtr, yt[tr_t], len(labs), True)
                oof[te] = (Xte @ W + b).argmax(1).cpu().numpy()
            scores[arm].append(f1_score(yi, oof, average="macro", zero_division=0))
            dropped[arm].append(drop_tot / FOLDS)

    print(f"test rows sharing pixels with their own training fold: "
          f"mean {np.mean(affected):.1f}/{len(y)} ({np.mean(affected)/len(y):.1%})\n")
    print(f"{'arm':<12}{'macro-F1':>12}{'sd':>9}{'vs keep':>10}{'2*SE':>9}"
          f"{'wins':>8}{'mean dropped/fold':>20}")
    print("-" * 80)
    base = np.array(scores["keep"])
    out = {"reps": REPS, "chip_px": CHIP_PX, "chip_m": chip_m,
           "n": len(y), "rows_with_overlap": n_with,
           "test_rows_leaking_mean": float(np.mean(affected)), "arms": {}}
    for arm in arms:
        v = np.array(scores[arm])
        dd = v - base
        se = dd.std(ddof=1) / np.sqrt(len(dd)) if arm != "keep" else 0.0
        wins = int((dd < 0).sum())
        print(f"{arm:<12}{v.mean():>12.4f}{v.std(ddof=1):>9.4f}{dd.mean():>+10.4f}"
              f"{2*se:>9.4f}{wins:>5}/{REPS}{np.mean(dropped[arm]):>20.1f}")
        out["arms"][arm] = dict(macro_f1=float(v.mean()), sd=float(v.std(ddof=1)),
                                vals=[float(t) for t in v],
                                delta_vs_keep=float(dd.mean()), two_se=float(2 * se),
                                lower_in=wins,
                                mean_dropped_per_fold=float(np.mean(dropped[arm])))

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / oname("overlap_control.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
