#!/usr/bin/env python3
"""Head-to-head: DINOv2 vs SatlasPretrain Aerial (SI and MI) on Herat.

Every candidate is scored under one identical protocol - same spatial blocks,
same folds, same seeds, same linear classifier - so the only thing varying is
the representation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path(__file__).resolve().parents[1] / "cache"
OUT = Path(__file__).resolve().parents[1] / "out"
FOLDS, SEEDS, BLOCKS = 5, 3, 6
# These representations run to 8k dimensions against n=838. PCA is both the
# statistically sensible move at d >> n and what makes the sweep tractable.
# It sits inside the pipeline so it is re-fitted on the training half of every
# fold - fitting it once over the whole matrix would leak the test set.
PCA_DIM = 256


def spatial_blocks(x, y, n):
    xb = np.clip(((x - x.min()) / (np.ptp(x) + 1e-12) * n).astype(int), 0, n - 1)
    yb = np.clip(((y - y.min()) / (np.ptp(y) + 1e-12) * n).astype(int), 0, n - 1)
    return xb * n + yb


def clf(seed, dim):
    steps = [StandardScaler()]
    if dim > PCA_DIM:
        steps.append(PCA(n_components=PCA_DIM, random_state=seed))
    steps.append(LogisticRegression(max_iter=3000, C=1.0,
                                    class_weight="balanced", random_state=seed))
    return make_pipeline(*steps)


def cv(X, y, groups, seed):
    sp = (StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed)
          if groups is not None else StratifiedKFold(FOLDS, shuffle=True, random_state=seed))
    it = sp.split(X, y, groups) if groups is not None else sp.split(X, y)
    oof = np.empty(len(y), object)
    for tr, te in it:
        oof[te] = clf(seed, X.shape[1]).fit(X[tr], y[tr]).predict(X[te])
    return f1_score(y, list(oof), average="macro"), oof


def main() -> None:
    dino = np.load(CACHE / "features.npz", allow_pickle=True)
    sat = np.load(CACHE / "features_satlas.npz", allow_pickle=True)
    assert (dino["fid"] == sat["fid"]).all(), "sample order differs between caches"

    y = dino["y"].astype(str)
    blk = spatial_blocks(dino["x"], dino["yy"], BLOCKS)
    d1, d2 = dino["f1"], dino["f2"]
    s1, s2, smi = sat["f1"], sat["f2"], sat["fmi"]

    cands = {
        "DINOv2 t2":            d2,
        "DINOv2 concat_diff":   np.hstack([d1, d2, d2 - d1]),
        "Satlas-SI t2":         s2,
        "Satlas-SI concat_diff": np.hstack([s1, s2, s2 - s1]),
        "Satlas-MI fused":      smi,
        "Satlas-MI + SI diff":  np.hstack([smi, s2 - s1]),
        "DINOv2 + Satlas-SI":   np.hstack([d1, d2, d2 - d1, s1, s2, s2 - s1]),
    }

    print(f"n={len(y)}  folds={FOLDS} seeds={SEEDS} blocks={BLOCKS}x{BLOCKS} pca={PCA_DIM}")
    print(f"majority-class floor macro-F1: {f1_score(y, np.full(len(y), 'New Construction'), average='macro'):.4f}\n")
    print(f"{'representation':<24}{'dim':>6}{'random CV':>18}{'spatial CV':>18}")
    print("-" * 66)

    res, oofs = {}, {}
    for name, X in cands.items():
        r = [cv(X, y, None, s)[0] for s in range(SEEDS)]
        g = [cv(X, y, blk, s)[0] for s in range(SEEDS)]
        oofs[name] = cv(X, y, blk, 0)[1]
        res[name] = dict(dim=X.shape[1], random_mean=float(np.mean(r)),
                         spatial_mean=float(np.mean(g)), spatial_std=float(np.std(g)))
        print(f"{name:<24}{X.shape[1]:>6}"
              f"{np.mean(r):>11.4f} +/-{np.std(r):.3f}"
              f"{np.mean(g):>11.4f} +/-{np.std(g):.3f}")

    best = max(res, key=lambda k: res[k]["spatial_mean"])
    print(f"\nbest on spatial CV: {best}  {res[best]['spatial_mean']:.4f}\n")

    labs = sorted(set(y))
    # Destruction vs New Construction is the direction-sensitive pair, so show
    # it explicitly for the order-invariant MI model against the best model.
    for name in [best, "Satlas-MI fused"]:
        print(f"=== {name} (spatial CV, seed 0) ===")
        print(classification_report(y, list(oofs[name]), digits=3, zero_division=0))
        cm = confusion_matrix(y, list(oofs[name]), labels=labs)
        print(f"{'':<22}" + "".join(f"{l[:11]:>13}" for l in labs))
        for l, row in zip(labs, cm):
            print(f"{l:<22}" + "".join(f"{v:>13}" for v in row))
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(json.dumps(dict(n=len(y), best=best, results=res), indent=1))
    print(f"saved {OUT / 'compare.json'}")


if __name__ == "__main__":
    main()
