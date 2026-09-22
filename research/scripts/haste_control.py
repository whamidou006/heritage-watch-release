#!/usr/bin/env python3
"""HASTE control: does a single-date (post-event) encoder work on the damage class?

Why this experiment exists
--------------------------
microsoft/haste is the platform we would deliver into. Its assessment model is
trained on ONE date: `docker/training/code/configs/config.yml` sets
`num_channels: 3  # (3 for RGB)` and `bda/datamodules.py:60` appends a
single-element list per sample, so pre-event imagery only ever reaches the
side-by-side visualiser, never the network. TRANSPARENCY.md agrees ("applies a
... model to ... post-disaster imagery").

That predicts a specific failure: Destruction and New Construction are the SAME
two images in the opposite ORDER, so a post-event-only model should be unable to
tell them apart, while a bi-temporal model should. We asserted this earlier.
Here we measure it, using the identical frozen-encoder features so the only
thing that varies is which dates the classifier sees.

Representations compared (all from the same DINOv2 / Satlas embeddings):
    haste_post  : f2 only            <- faithful HASTE analogue
    pre_only    : f1 only            <- control for "is it just image quality?"
    diff        : f2 - f1            <- direction, but no absolute appearance
    bitemporal  : [f1, f2, f2-f1]    <- our system

Tasks:
    A. Destruction vs New Construction (the order-confusable pair, n=478)
    B. Full 4-class, reporting Destruction (the damage class) specifically

Run:  cd work/scripts && python haste_control.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HACK = Path(__file__).resolve().parents[2] / "hackathon"
sys.path.insert(0, str(_HACK))

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

import herat_eval as H  # noqa: E402

from cachesel import cname, oname  # noqa: E402
import decision as D  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "cache"
OUT = Path(__file__).resolve().parents[1] / "out" / oname("haste_control.json")
REPS = H.REPLICATES


def reps_from(d):
    """Build the four representations from a loaded npz."""
    f1, f2 = d["f1"], d["f2"]
    return {
        "haste_post": f2,
        "pre_only": f1,
        "diff": f2 - f1,
        "bitemporal": np.hstack([f1, f2, f2 - f1]),
    }


def binary_cv(X, ypos, x, yy, reps=REPS):
    """Paired jittered spatial CV for a binary task. Returns (f1s, aps)."""
    X = np.asarray(X, float)
    f1s, aps = [], []
    with H._limit_threads():
        for r in range(reps):
            rng = np.random.default_rng(H._SEED + r)
            g = H.spatial_blocks(x, yy, H.BLOCKS, rng.random(), rng.random())
            oof = np.zeros(len(ypos), int)
            prob = np.zeros(len(ypos), float)
            for tr, te in StratifiedGroupKFold(
                    H.FOLDS, shuffle=True, random_state=r).split(X, ypos, g):
                clf = H.default_classifier()
                clf.fit(X[tr], ypos[tr])
                oof[te] = clf.predict(X[te])
                prob[te] = clf.predict_proba(X[te])[:, 1]
            f1s.append(f1_score(ypos, oof))
            aps.append(average_precision_score(ypos, prob))
    return np.array(f1s), np.array(aps)


def paired(a, b):
    return D.verdict(np.asarray(a) - np.asarray(b))


def main():
    res = {"replicates": REPS, "encoders": {}}
    for enc, path in [("dinov2", "features_dinov2.npz"),
                      ("satlas", "features_satlas.npz")]:
        d = np.load(CACHE / cname(path), allow_pickle=True)
        y = d["y"].astype(str)
        x, yy = d["x"], d["yy"]
        R = reps_from(d)
        block = {}

        # ---- Task A: Destruction vs New Construction ------------------
        m = (y == "Destruction") | (y == "New Construction")
        ypos = (y[m] == "Destruction").astype(int)
        print(f"\n[{enc}] Task A  Destruction vs New Construction  "
              f"n={m.sum()} (pos={ypos.sum()})")
        base = float(ypos.mean())
        print(f"  prevalence baseline AP = {base:.3f}")
        taskA = {"n": int(m.sum()), "n_pos": int(ypos.sum()),
                 "ap_baseline": base, "reps": {}}
        for name, X in R.items():
            f1s, aps = binary_cv(X[m], ypos, x[m], yy[m])
            taskA["reps"][name] = {"f1": float(f1s.mean()), "f1_sd": float(f1s.std()),
                                   "ap": float(aps.mean()), "ap_sd": float(aps.std()),
                                   "f1_all": f1s.tolist(), "ap_all": aps.tolist()}
            print(f"  {name:<12} F1 {f1s.mean():.3f}+/-{f1s.std():.3f}   "
                  f"AP {aps.mean():.3f}+/-{aps.std():.3f}")
        ta = taskA["reps"]
        taskA["bitemporal_vs_haste_f1"] = paired(
            np.array(ta["bitemporal"]["f1_all"]), np.array(ta["haste_post"]["f1_all"]))
        taskA["bitemporal_vs_haste_ap"] = paired(
            np.array(ta["bitemporal"]["ap_all"]), np.array(ta["haste_post"]["ap_all"]))
        print("  paired bitemporal - haste_post")
        print("  " + D.line("    F1", taskA["bitemporal_vs_haste_f1"], 12))
        print("  " + D.line("    AP", taskA["bitemporal_vs_haste_ap"], 12))
        block["taskA_destruction_vs_newconstruction"] = taskA

        # ---- Task B: full 4-class, focus on the damage class ----------
        print(f"[{enc}] Task B  4-class (reporting Destruction)")
        meta = {"x": x, "yy": yy, "fid": d["fid"]}
        taskB = {}
        for name, X in R.items():
            r = H.evaluate(X, y, meta, replicates=REPS)
            taskB[name] = {"macro_f1": r.macro_f1, "sd": r.sd,
                           "per_class": dict(r.per_class_f1),
                           "per_replicate": r.per_replicate}
            print(f"  {name:<12} macro {r.macro_f1:.3f}   "
                  f"Destruction {r.per_class_f1['Destruction']:.3f}   "
                  f"NewConstr {r.per_class_f1['New Construction']:.3f}")
        block["taskB_4class"] = taskB
        res["encoders"][enc] = block

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
