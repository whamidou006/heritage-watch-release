#!/usr/bin/env python3
"""Chip size, compared the way the report's own rule requires: strictly paired.

`chip_study.py` scores each chip size on whatever points survive extraction at
that size (871-883 of 879-ish).  The populations therefore differ, and because
the spatial grid is normalised to each population's own extent, so do the grid
cells and the folds.  A difference between two such runs mixes three changes:
chip size, population and partition.

This restricts every arm to the FIDs that survive at ALL chip sizes, builds the
grid once from those shared coordinates, and reuses the identical fold
assignment for every arm and replicate.  The only thing that varies is the
pixels.  Differences are then judged by the shared three-bar rule.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import decision as D
from cachesel import cname, oname
from balance_study import fit_logreg  # identical to compare_gpu's when balanced=True;
                                      # imported from here so the estimator is provably
                                      # the same object chip_study.py scores with
from compare_gpu import (BLOCKS, CACHE, FOLDS, OUT, fit_transform, set_pca,
                         spatial_blocks)

REPS = 8
SIZES = (32, 64, 128, 256)


def cache_names(c: int) -> tuple[str, str]:
    if c == 128:
        return cname("features.npz"), cname("features_satlas.npz")
    return cname(f"features_c{c}.npz"), cname(f"features_satlas_c{c}.npz")


def main() -> None:
    set_pca(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    loaded = {}
    for c in SIZES:
        dn, sn = cache_names(c)
        d = np.load(CACHE / dn, allow_pickle=True)
        s = np.load(CACHE / sn, allow_pickle=True)
        assert (d["fid"] == s["fid"]).all(), c
        loaded[c] = (d, s)
        print(f"chip {c:>3}: n={len(d['fid'])}  ({dn})")

    common = set(loaded[SIZES[0]][0]["fid"].tolist())
    for c in SIZES[1:]:
        common &= set(loaded[c][0]["fid"].tolist())
    common = np.array(sorted(common))
    print(f"\nshared across all {len(SIZES)} chip sizes: n={len(common)}")

    arms, ref = {}, None
    for c in SIZES:
        d, s = loaded[c]
        order = np.argsort(d["fid"])
        sel = order[np.searchsorted(d["fid"][order], common)]
        assert (d["fid"][sel] == common).all()
        X = np.hstack([d["f1"][sel], d["f2"][sel], d["f2"][sel] - d["f1"][sel],
                       s["f1"][sel], s["f2"][sel], s["f2"][sel] - s["f1"][sel]])
        meta = (d["y"][sel].astype(str), d["x"][sel], d["yy"][sel])
        if ref is None:
            ref = meta
        else:  # labels and coordinates must be identical, or it is not paired
            assert (meta[0] == ref[0]).all() and np.allclose(meta[1], ref[1])
        arms[c] = X

    y, x, yy = ref
    labs = sorted(set(y))
    yi = np.searchsorted(labs, y)
    blk = spatial_blocks(x, yy, BLOCKS)          # ONE grid, shared by every arm
    splits = [list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=s)
                   .split(np.zeros((len(yi), 1)), yi, blk)) for s in range(REPS)]

    yt = torch.as_tensor(yi, dtype=torch.long, device=dev)
    scores = {}
    for c in SIZES:
        Xt = torch.as_tensor(arms[c], dtype=torch.float32, device=dev)
        vals = []
        for sp in splits:
            oof = np.empty(len(yi), np.int64)
            for tr, te in sp:
                tr_t = torch.as_tensor(tr, device=dev)
                te_t = torch.as_tensor(te, device=dev)
                Xtr, Xte = fit_transform(Xt[tr_t], Xt[te_t], Xt.shape[1])
                W, b = fit_logreg(Xtr, yt[tr_t], len(labs), True)
                oof[te] = (Xte @ W + b).argmax(1).cpu().numpy()
            vals.append(f1_score(yi, oof, average="macro", zero_division=0))
        scores[c] = np.array(vals)
        print(f"chip {c:>3} ({c*0.25:>5.1f} m): {scores[c].mean():.4f} "
              f"+/- {scores[c].std(ddof=1):.4f}")

    best = max(SIZES, key=lambda c: scores[c].mean())
    print(f"\nbest = {best} px; paired differences against it "
          f"({REPS} shared partitions, n={len(common)})\n")
    print(f"{'vs':<12}{'delta':>9}{'sd':>8}{'2*SE':>9}{'wins':>7}   verdict")
    print("-" * 62)
    out = {"reps": REPS, "n_shared": int(len(common)), "best_px": best,
           "per_size": {str(c): dict(macro_f1=float(scores[c].mean()),
                                     sd=float(scores[c].std(ddof=1)),
                                     vals=[float(v) for v in scores[c]])
                        for c in SIZES},
           "paired_vs_best": {}}
    for c in SIZES:
        if c == best:
            continue
        vd = D.verdict(scores[best] - scores[c])
        out["paired_vs_best"][str(c)] = vd
        print(D.line(f"{c} px", vd, 12))

    # the specific contrast the report makes a claim about
    if 64 in SIZES and 128 in SIZES:
        vd = D.verdict(scores[64] - scores[128])
        out["64_vs_128"] = vd
        print("\n64 px vs the 128 px default:")
        print(D.line("64 - 128", vd, 12))

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / oname("chip_paired.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
