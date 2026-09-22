#!/usr/bin/env python3
"""Paired test: does resolving annotation layers to the MONTH change the score?

The annotation layer names carry YYYYMM ("change_between_202205_and_202310").
The original manifest builder threw the month away and kept only the year, then
picked "latest scene of year a, earliest of year b". That selects the wrong
image in 6 of 9 pairs, so a patch is labelled with a change class whose pixels
do not show that change.

This script holds EVERYTHING fixed except which image was read:

  * same points (intersection of fids present in both arms)
  * same labels        - asserted identical, not assumed
  * same coordinates   - so the same 6x6 spatial blocks
  * same folds         - StratifiedGroupKFold is deterministic given
                         (y, groups, seed), and all three are shared
  * same classifier    - the report's merged encoder, PCA 256 + logreg

so the per-seed difference is paired and the only varying input is the pixels.

Two populations are reported:

  shared   - every fid present in both arms. This is the honest "what does
             fixing the bug buy on the dataset as a whole" number, diluted by
             the points whose imagery never moved.
  moved    - the subset whose t1/t2 scene actually changed. If the fix does
             anything, it has to show up here; if it does not move here it
             does not move anywhere.

Decision rule is the report's: unanimous across seeds, |delta| > 2*SE, and
|delta| >= 0.02. Anything smaller is inside the measured noise floor and is
reported as "not resolved", not as a win.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import decision as D

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = ROOT / "out"

FOLDS, BLOCKS, PCA_DIM = 5, 6, 256
SEEDS = 8
MDE = 0.02


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
    sp = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed)
    oof = np.empty(len(y), object)
    for tr, te in sp.split(X, y, groups):
        oof[te] = clf(seed, X.shape[1]).fit(X[tr], y[tr]).predict(X[te])
    return f1_score(y, list(oof), average="macro")


def merged(dino, sat, idx):
    d1, d2 = dino["f1"][idx], dino["f2"][idx]
    s1, s2 = sat["f1"][idx], sat["f2"][idx]
    return np.hstack([d1, d2, d2 - d1, s1, s2, s2 - s1])


def load_arm(dino_name, sat_name):
    d = np.load(CACHE / dino_name, allow_pickle=True)
    s = np.load(CACHE / sat_name, allow_pickle=True)
    assert (d["fid"] == s["fid"]).all(), f"fid order differs: {dino_name} vs {sat_name}"
    return d, s


def scene_dates(manifest_name):
    """fid -> (t1_path, t2_path) so we can tell which points actually moved."""
    recs = json.loads((CACHE / manifest_name).read_text())
    if isinstance(recs, dict):
        recs = recs.get("records", recs.get("samples"))
    out = {}
    for r in recs:
        fid = str(r.get("fid", r.get("id")))
        out[fid] = (str(r.get("t1_date", r.get("date1", ""))),
                    str(r.get("t2_date", r.get("date2", ""))))
    return out


def report(tag, Xy, Xm, y, blk):
    ry = np.array([cv(Xy, y, blk, s) for s in range(SEEDS)])
    rm = np.array([cv(Xm, y, blk, s) for s in range(SEEDS)])
    d = rm - ry
    vd = D.verdict(d, MDE)
    se, wins = vd["two_se"] / 2, vd["positive"]
    unanimous, resolved = vd["unanimous"], vd["resolved"]

    print(f"\n=== {tag}  (n={len(y)}, {SEEDS} seeds) ===")
    print(f"  year  macro-F1  {ry.mean():.4f} +/- {ry.std(ddof=1):.4f}")
    print(f"  month macro-F1  {rm.mean():.4f} +/- {rm.std(ddof=1):.4f}")
    print(f"  delta           {d.mean():+.4f}  SE {se:.4f}  "
          f"2*SE {2*se:.4f}  MDE {MDE:.3f}")
    print(f"  month better in {wins}/{SEEDS} seeds")
    print(f"  -> {'RESOLVED' if resolved else 'NOT RESOLVED (inside noise floor)'}")
    return dict(n=int(len(y)), year_mean=float(ry.mean()), month_mean=float(rm.mean()),
                delta=float(d.mean()), se=float(se), wins=wins,
                unanimous=bool(unanimous), resolved=bool(resolved),
                year_per_seed=ry.tolist(), month_per_seed=rm.tolist())


def main() -> None:
    dy, sy = load_arm("features.npz", "features_satlas.npz")
    dm, sm = load_arm("features_month.npz", "features_satlas_month.npz")

    fy = {f: i for i, f in enumerate(dy["fid"].astype(str))}
    fm = {f: i for i, f in enumerate(dm["fid"].astype(str))}
    shared = sorted(set(fy) & set(fm))
    iy = np.array([fy[f] for f in shared])
    im = np.array([fm[f] for f in shared])
    print(f"year arm {len(fy)}   month arm {len(fm)}   shared {len(shared)}")

    ly = dy["y"].astype(str)[iy]
    lm = dm["y"].astype(str)[im]
    assert (ly == lm).all(), "labels differ on shared fids - not a paired comparison"
    y = ly

    # coordinates come from the point, not the image, so they must agree
    assert np.allclose(dy["x"][iy], dm["x"][im]), "x coords differ"
    blk = spatial_blocks(dy["x"][iy], dy["yy"][iy], BLOCKS)

    Xy = merged(dy, sy, iy)
    Xm = merged(dm, sm, im)

    # which points actually had their imagery swapped
    dates_y = scene_dates("manifest_year.json")
    dates_m = scene_dates("manifest_month.json")
    moved = np.array([dates_y.get(f) != dates_m.get(f) for f in shared])
    print(f"imagery changed for {int(moved.sum())}/{len(shared)} shared points "
          f"({100*moved.mean():.1f}%)")

    results = {"shared": report("shared (all paired points)", Xy, Xm, y, blk)}

    if moved.sum() >= 150:
        ym = y[moved]
        # a block must survive the subset, and every class needs enough support
        results["moved"] = report("moved (imagery actually changed)",
                                  Xy[moved], Xm[moved], ym, blk[moved])
    else:
        print(f"\nmoved subset too small to score ({int(moved.sum())})")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "month_vs_year.json").write_text(json.dumps(results, indent=1))
    print(f"\nsaved {OUT / 'month_vs_year.json'}")


if __name__ == "__main__":
    main()
