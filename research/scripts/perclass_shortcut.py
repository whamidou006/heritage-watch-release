#!/usr/bin/env python3
"""Is there a date shortcut, and what is the selected model's per-class behaviour?

Two questions:

  1. 'Other' is 20/25 from a single annotation layer (2023->2024). If class labels
     correlate with the acquisition pair, a classifier can exploit scene radiometry
     instead of change semantics. We test this with an imagery-free control: predict
     the class from the one-hot year-pair alone. Anything well above the 0.148
     majority floor means a date shortcut exists and must be reported.

  2. Full per-class breakdown of the selected configuration, with seed spread and
     the row-normalised confusion matrix.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from balance_study import cv
from cleaning_study import load
from compare_gpu import BLOCKS, CACHE, OUT, SEEDS, set_pca, spatial_blocks


def pairs_for(fids: np.ndarray, manifest: str) -> np.ndarray:
    recs = {r["fid"]: r["pair"] for r in json.loads((CACHE / manifest).read_text())["records"]}
    return np.array([recs[int(f)] for f in fids])


def main() -> None:
    set_pca(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 1. date-shortcut control (uses the raw 6-class set) ----------
    d = np.load(CACHE / "features_raw.npz", allow_pickle=True)
    yr, xr, yyr, fr = d["y"].astype(str), d["x"], d["yy"], d["fid"]
    pr = pairs_for(fr, "manifest_raw.json")

    print("=== class x acquisition pair (raw 6-class set) ===")
    pl = sorted(set(pr))
    cl = sorted(set(yr))
    print(f"{'class':<22}" + "".join(f"{p[:4]+'/'+p[4:]:>10}" for p in pl) + f"{'n':>7}")
    for c in cl:
        row = [int(((yr == c) & (pr == p)).sum()) for p in pl]
        print(f"{c:<22}" + "".join(f"{v:>10}" for v in row) + f"{sum(row):>7}")

    labs = sorted(set(yr))
    yi = np.searchsorted(labs, yr)
    blk = spatial_blocks(xr, yyr, BLOCKS)
    Xdate = np.eye(len(pl))[np.searchsorted(pl, pr)]          # one-hot date only
    Ximg, _, _, _ = load("features_raw.npz", "features_satlas_raw.npz")

    print("\n=== date-shortcut control (spatial CV, 6 classes) ===")
    out = {}
    for tag, X in (("date one-hot only (no imagery)", Xdate),
                   ("imagery (merged)", Ximg),
                   ("imagery + date", np.hstack([Ximg, Xdate]))):
        v = [f1_score(yi, cv(X, yi, blk, s, len(labs), dev, True),
                      average="macro", zero_division=0) for s in range(SEEDS)]
        out[tag] = dict(macro_f1=float(np.mean(v)), std=float(np.std(v)))
        print(f"  {tag:<34}{np.mean(v):>8.4f} +/-{np.std(v):.3f}")
    print(f"  {'majority-class floor':<34}"
          f"{f1_score(yr, np.full(len(yr), Counter(yr).most_common(1)[0][0]), average='macro'):>8.4f}")

    # how well can 'Other' alone be found from the date?
    is_o = (yr == "Other").astype(int)
    pred_o = (pr == "20232024").astype(int)
    p, r, f, _ = precision_recall_fscore_support(is_o, pred_o, average="binary", zero_division=0)
    print(f"\n  rule 'pair==2023/2024 => Other':  P {p:.3f}  R {r:.3f}  F1 {f:.3f}")
    out["other_date_rule"] = dict(precision=float(p), recall=float(r), f1=float(f))

    # ---------- 2. per-class breakdown of the selected configuration ----------
    for tag, (a, b) in (("128 px (reported)", ("features.npz", "features_satlas.npz")),
                        ("64 px (tuned)", ("features_c64.npz", "features_satlas_c64.npz"))):
        X, y, x, yy = load(a, b)
        labs4 = sorted(set(y))
        yi4 = np.searchsorted(labs4, y)
        blk4 = spatial_blocks(x, yy, BLOCKS)
        oofs = [cv(X, yi4, blk4, s, len(labs4), dev, True) for s in range(SEEDS)]
        P, R, F = [], [], []
        for o in oofs:
            p_, r_, f_, s_ = precision_recall_fscore_support(
                yi4, o, labels=range(len(labs4)), zero_division=0)
            P.append(p_); R.append(r_); F.append(f_)
        P, R, F = np.array(P), np.array(R), np.array(F)
        print(f"\n=== selected model, per class — {tag} ===")
        print(f"{'class':<22}{'P':>8}{'R':>8}{'F1':>8}{'+/-':>7}{'n':>6}")
        rec = {}
        for i, c in enumerate(labs4):
            n = int((y == c).sum())
            print(f"{c:<22}{P[:,i].mean():>8.3f}{R[:,i].mean():>8.3f}"
                  f"{F[:,i].mean():>8.3f}{F[:,i].std():>7.3f}{n:>6}")
            rec[c] = dict(precision=float(P[:,i].mean()), recall=float(R[:,i].mean()),
                          f1=float(F[:,i].mean()), f1_std=float(F[:,i].std()), n=n)
        mac = [f1_score(yi4, o, average="macro") for o in oofs]
        print(f"{'macro':<22}{P.mean():>8.3f}{R.mean():>8.3f}"
              f"{np.mean(mac):>8.3f}{np.std(mac):>7.3f}{len(y):>6}")
        cm = confusion_matrix(yi4, oofs[0], labels=range(len(labs4)))
        print("row-normalised confusion (seed 0, %):")
        print(f"{'':<22}" + "".join(f"{c[:11]:>13}" for c in labs4))
        for i, c in enumerate(labs4):
            print(f"{c:<22}" + "".join(f"{100*v/cm[i].sum():>12.1f}%" for v in cm[i]))
        out[tag] = dict(per_class=rec, macro_f1=float(np.mean(mac)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "perclass_shortcut.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {OUT / 'perclass_shortcut.json'}")


if __name__ == "__main__":
    main()
