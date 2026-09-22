#!/usr/bin/env python3
"""Per-class breakdown and confusion matrix of the selected model on a given cache pair.

This is part 2 of `perclass_shortcut.py` factored out so it can be pointed at the
month-resolved feature caches. Part 1 of that script (the date-shortcut control)
needs the raw 6-class caches and is not reproduced here.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from balance_study import cv
from cleaning_study import load
from compare_gpu import BLOCKS, OUT, SEEDS, set_pca, spatial_blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", default="features_month.npz")
    ap.add_argument("--satlas", default="features_satlas_month.npz")
    ap.add_argument("--rep", default="merged", choices=("merged", "mi_sidiff"),
                    help="merged = DINOv2+Satlas-SI; mi_sidiff = Satlas-MI + SI diff")
    ap.add_argument("--out", default="perclass_month.json")
    args = ap.parse_args()

    set_pca(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.rep == "merged":
        X, y, x, yy = load(args.dino, args.satlas)
    else:
        import numpy as _np
        from compare_gpu import CACHE as _C
        d = _np.load(_C / args.dino, allow_pickle=True)
        s = _np.load(_C / args.satlas, allow_pickle=True)
        assert (d["fid"] == s["fid"]).all()
        X = _np.hstack([s["fmi"], s["f2"] - s["f1"]])
        y, x, yy = d["y"].astype(str), d["x"], d["yy"]
    labs = sorted(set(y))
    yi = np.searchsorted(labs, y)
    blk = spatial_blocks(x, yy, BLOCKS)

    oofs = [cv(X, yi, blk, s, len(labs), dev, True) for s in range(SEEDS)]
    P, R, F = [], [], []
    for o in oofs:
        p_, r_, f_, _ = precision_recall_fscore_support(
            yi, o, labels=range(len(labs)), zero_division=0)
        P.append(p_); R.append(r_); F.append(f_)
    P, R, F = np.array(P), np.array(R), np.array(F)

    print(f"n={len(y)}  seeds={SEEDS}  dim={X.shape[1]}")
    print(f"\n{'class':<22}{'P':>8}{'R':>8}{'F1':>8}{'+/-':>7}{'n':>6}{'share':>8}")
    rec = {}
    for i, c in enumerate(labs):
        n = int((y == c).sum())
        print(f"{c:<22}{P[:,i].mean():>8.3f}{R[:,i].mean():>8.3f}"
              f"{F[:,i].mean():>8.3f}{F[:,i].std():>7.3f}{n:>6}{100*n/len(y):>7.1f}%")
        rec[c] = dict(precision=float(P[:, i].mean()), recall=float(R[:, i].mean()),
                      f1=float(F[:, i].mean()), f1_std=float(F[:, i].std()), n=n,
                      share=float(n / len(y)))
    mac = [f1_score(yi, o, average="macro") for o in oofs]
    print(f"{'macro':<22}{P.mean():>8.3f}{R.mean():>8.3f}"
          f"{np.mean(mac):>8.3f}{np.std(mac):>7.3f}{len(y):>6}")

    cm = confusion_matrix(yi, oofs[0], labels=range(len(labs)))
    print("\nrow-normalised confusion (seed 0, %):")
    print(f"{'':<22}" + "".join(f"{c[:11]:>13}" for c in labs))
    for i, c in enumerate(labs):
        print(f"{c:<22}" + "".join(f"{100*v/cm[i].sum():>12.1f}%" for v in cm[i]))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(
        dict(n=int(len(y)), per_class=rec, macro_f1=float(np.mean(mac)),
             confusion_rownorm={labs[i]: {labs[j]: float(cm[i][j] / cm[i].sum())
                                          for j in range(len(labs))}
                                for i in range(len(labs))}), indent=1))
    print(f"\nsaved {OUT / args.out}")


if __name__ == "__main__":
    main()
