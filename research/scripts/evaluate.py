#!/usr/bin/env python3
"""Evaluate frozen DINOv2 bi-temporal embeddings for Herat change classification.

Reports two numbers deliberately:

  random CV   - stratified k-fold ignoring geography. Optimistic, because change
                labels are spatially clustered (one construction site yields
                several neighbouring points), so near-duplicates straddle the
                train/test boundary.
  spatial CV  - grouped k-fold over a coarse spatial grid, so every point from a
                given block is held out together. This is the number to quote.

Both are run over several seeds so each effect can be read against its own
seed-to-seed spread rather than eyeballed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path(__file__).resolve().parents[1] / "cache"
OUT = Path(__file__).resolve().parents[1] / "out"


def spatial_blocks(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Assign each point to one of n*n equal-area blocks over the footprint."""
    xb = np.clip(((x - x.min()) / (np.ptp(x) + 1e-12) * n).astype(int), 0, n - 1)
    yb = np.clip(((y - y.min()) / (np.ptp(y) + 1e-12) * n).astype(int), 0, n - 1)
    return xb * n + yb


def features(f1: np.ndarray, f2: np.ndarray, mode: str) -> np.ndarray:
    if mode == "concat_diff":
        return np.hstack([f1, f2, f2 - f1])
    if mode == "diff":
        return f2 - f1
    if mode == "t2":
        return f2
    if mode == "concat":
        return np.hstack([f1, f2])
    raise ValueError(mode)


def make_clf(kind: str, seed: int):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                               random_state=seed),
        )
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed,
    )


def run_cv(X, y, groups, seed: int, kind: str, folds: int):
    """Return (macro_f1, out-of-fold predictions)."""
    if groups is None:
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
        it = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        it = splitter.split(X, y, groups)

    oof = np.empty(len(y), object)
    for tr, te in it:
        clf = make_clf(kind, seed).fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    return f1_score(y, list(oof), average="macro"), oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=6, help="spatial grid is blocks x blocks")
    ap.add_argument("--clfs", default="logreg",
                    help="comma list of: logreg,hgb. HGB is very slow on this "
                         "shared host and does not suit n=838 / d=768-2304.")
    args = ap.parse_args()

    d = np.load(CACHE / "features.npz", allow_pickle=True)
    f1, f2, y = d["f1"], d["f2"], d["y"].astype(str)
    blk = spatial_blocks(d["x"], d["yy"], args.blocks)

    print(f"n={len(y)}  classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"spatial blocks occupied: {len(np.unique(blk))} of {args.blocks ** 2}")
    print()

    # Majority-class floor: what you get for free by always guessing the
    # most common class. Any model must clear this to be worth anything.
    maj = np.full(len(y), max(set(y), key=list(y).count))
    print(f"majority-class baseline macro-F1 : {f1_score(y, maj, average='macro'):.4f}")
    print()

    seeds = list(range(args.seeds))
    results = {}
    print(f"{'features':<12} {'clf':<7} {'random CV':>18} {'spatial CV':>18}   gap")
    print("-" * 70)
    for mode in ["t2", "diff", "concat", "concat_diff"]:
        X = features(f1, f2, mode)
        for kind in args.clfs.split(","):
            r = [run_cv(X, y, None, s, kind, args.folds)[0] for s in seeds]
            g = [run_cv(X, y, blk, s, kind, args.folds)[0] for s in seeds]
            results[f"{mode}|{kind}"] = dict(
                random_mean=float(np.mean(r)), random_std=float(np.std(r)),
                spatial_mean=float(np.mean(g)), spatial_std=float(np.std(g)),
            )
            print(f"{mode:<12} {kind:<7} "
                  f"{np.mean(r):.4f} +/- {np.std(r):.4f} "
                  f"{np.mean(g):.4f} +/- {np.std(g):.4f}   "
                  f"{np.mean(r) - np.mean(g):+.4f}")

    best = max(results, key=lambda k: results[k]["spatial_mean"])
    mode, kind = best.split("|")
    print()
    print(f"best on spatial CV: {best}  macro-F1={results[best]['spatial_mean']:.4f}")
    print()

    X = features(f1, f2, mode)
    _, oof = run_cv(X, y, blk, 0, kind, args.folds)
    oof = list(oof)
    print("per-class report (spatial CV, seed 0):")
    print(classification_report(y, oof, digits=3, zero_division=0))
    labs = sorted(set(y))
    print("confusion matrix (rows = true):")
    print(f"{'':<22}" + "".join(f"{l[:11]:>13}" for l in labs))
    for l, row in zip(labs, confusion_matrix(y, oof, labels=labs)):
        print(f"{l:<22}" + "".join(f"{v:>13}" for v in row))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        dict(n=len(y), majority_macro_f1=f1_score(y, maj, average="macro"),
             best=best, results=results), indent=1))
    print(f"\nsaved {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
