#!/usr/bin/env python3
"""Date-shortcut control on the month-resolved raw (6-class) set.

Part 1 of `perclass_shortcut.py`, pointed at month-resolved caches. If class
labels correlate with the acquisition pair, a classifier can key on scene
radiometry instead of change semantics; the control predicts the class from the
one-hot acquisition pair alone, with no imagery at all.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from balance_study import cv
from cleaning_study import load
from compare_gpu import BLOCKS, CACHE, OUT, SEEDS, set_pca, spatial_blocks


def pairs_for(fids: np.ndarray, manifest: str) -> np.ndarray:
    recs = {r["fid"]: r["pair"] for r in json.loads((CACHE / manifest).read_text())["records"]}
    return np.array([recs[int(f)] for f in fids])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", default="features_raw_month.npz")
    ap.add_argument("--satlas", default="features_satlas_raw_month.npz")
    ap.add_argument("--manifest", default="manifest_raw_month.json")
    ap.add_argument("--out", default="date_shortcut_month.json")
    args = ap.parse_args()

    set_pca(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(CACHE / args.dino, allow_pickle=True)
    y, x, yy, fid = d["y"].astype(str), d["x"], d["yy"], d["fid"]
    pr = pairs_for(fid, args.manifest)

    print(f"=== class x acquisition pair (raw {len(set(y))}-class set, n={len(y)}) ===")
    pl, cl = sorted(set(pr)), sorted(set(y))
    print(f"{'class':<22}" + "".join(f"{p[:4]+'/'+p[4:]:>10}" for p in pl) + f"{'n':>7}")
    for c in cl:
        row = [int(((y == c) & (pr == p)).sum()) for p in pl]
        print(f"{c:<22}" + "".join(f"{v:>10}" for v in row) + f"{sum(row):>7}")

    labs = sorted(set(y))
    yi = np.searchsorted(labs, y)
    blk = spatial_blocks(x, yy, BLOCKS)
    Xdate = np.eye(len(pl))[np.searchsorted(pl, pr)]
    Ximg, _, _, _ = load(args.dino, args.satlas)

    print(f"\n=== date-shortcut control (spatial CV, {len(labs)} classes) ===")
    out = {}
    for tag, X in (("date one-hot only (no imagery)", Xdate),
                   ("imagery (merged)", Ximg),
                   ("imagery + date", np.hstack([Ximg, Xdate]))):
        v = [f1_score(yi, cv(X, yi, blk, s, len(labs), dev, True),
                      average="macro", zero_division=0) for s in range(SEEDS)]
        out[tag] = dict(macro_f1=float(np.mean(v)), std=float(np.std(v)))
        print(f"  {tag:<34}{np.mean(v):>8.4f} +/-{np.std(v):.3f}")
    floor = f1_score(y, np.full(len(y), Counter(y).most_common(1)[0][0]), average="macro")
    out["majority_floor"] = float(floor)
    print(f"  {'majority-class floor':<34}{floor:>8.4f}")

    is_o = (y == "Other").astype(int)
    if is_o.sum():
        best = Counter(pr[is_o == 1]).most_common(1)[0][0]
        p, r, f, _ = precision_recall_fscore_support(
            is_o, (pr == best).astype(int), average="binary", zero_division=0)
        print(f"\n  rule 'pair=={best[:4]}/{best[4:]} => Other':  P {p:.3f}  R {r:.3f}  F1 {f:.3f}")
        out["other_date_rule"] = dict(pair=best, precision=float(p),
                                      recall=float(r), f1=float(f))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(out, indent=1))
    print(f"\nsaved {OUT / args.out}")


if __name__ == "__main__":
    main()
