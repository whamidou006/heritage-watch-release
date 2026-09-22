#!/usr/bin/env python3
"""Does the score depend on having seen the same acquisition pair during training?

The headline protocol blocks *space* but not *time*: all 10 acquisition pairs appear on
both sides of every fold. §A4 shows the date carries real signal (date-only macro-F1
0.272 vs a 0.097 floor), so the question is how much of the 0.711 survives on a pair
the model has not seen.

A naive leave-one-interval-out answers the wrong question, in three ways that all
inflate the apparent effect:

  1. Seven of the ten intervals contain only 3 of the 4 classes (there is no Solar
     Panel label before 2018). Scoring macro-F1 over all 4 labels gives the absent
     class F1=0 and still averages it in, capping those folds at 0.75.
  2. A size-matched control drawn at random from all points is not spatially
     blocked, so it enjoys exactly the leakage the protocol exists to remove.
  3. Splitting an interval's own points at random puts training rows metres from
     test rows, so the "seen" arm is inflated by spatial leakage and the contrast
     measures proximity rather than acquisition-pair novelty.

This version fixes all three. For each interval the test rows are whole spatial grid
cells, held FIXED across the two arms; those cells are excluded from the training pool
of BOTH arms; and only the training composition changes:

  seen    train = the rest of this interval + rows from other intervals
  unseen  train = rows from other intervals only

Both arms have the same training-set size and the same test rows, and neither can see
inside a test cell, so the paired difference isolates one thing: was this acquisition
pair represented in training. Macro-F1 is computed over the classes actually present
in the test cells.

Caveat measured with the result: under the month rule the resolved scenes form a chain
(each pair's second image is the next pair's first), so EVERY interval shares an endpoint
scene with a neighbour. The "unseen" arm has therefore always seen one of the two images
through an adjacent pair, and a fully endpoint-clean leave-one-interval-out is impossible
on this site. The sharing set is derived from the manifest rather than hardcoded, so this
is reported rather than assumed.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from compare_gpu import (BLOCKS, CACHE, OUT, fit_transform, fit_logreg,
                         set_pca, spatial_blocks)
from cachesel import cname, oname

REPS = 6        # paired replicates per interval
N_TRAIN = 500   # training-set size CAP; the usable pool is smaller in some folds


def shared_endpoint_pairs(manifest: str) -> set[str]:
    """Intervals that share at least one acquisition scene with another interval."""
    recs = json.loads((CACHE / manifest).read_text())["records"]
    ends: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        ends[r["pair"]].update((r["t1"], r["t2"]))
    out = set()
    for p, e in ends.items():
        for q, f in ends.items():
            if p != q and e & f:
                out.add(p)
                break
    return out


def run(X, yi, tr_idx, te_idx, n_cls, dev):
    Xtr = torch.as_tensor(X[tr_idx], dtype=torch.float32, device=dev)
    Xte = torch.as_tensor(X[te_idx], dtype=torch.float32, device=dev)
    ytr = torch.as_tensor(yi[tr_idx], dtype=torch.long, device=dev)
    a, b = fit_transform(Xtr, Xte, X.shape[1])
    W, bb = fit_logreg(a, ytr, n_cls)
    pred = (b @ W + bb).argmax(1).cpu().numpy()
    present = np.unique(yi[te_idx])          # score only classes really in the test half
    return f1_score(yi[te_idx], pred, average="macro", labels=present, zero_division=0)


def main() -> None:
    set_pca(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(CACHE / cname("features.npz"), allow_pickle=True)
    s = np.load(CACHE / cname("features_satlas.npz"), allow_pickle=True)
    assert (d["fid"] == s["fid"]).all() and (d["pair"] == s["pair"]).all()

    y = d["y"].astype(str)
    labs = sorted(set(y))
    yi = np.searchsorted(labs, y)
    n_cls = len(labs)
    pair = d["pair"].astype(str)
    X = np.hstack([d["f1"], d["f2"], d["f2"] - d["f1"],
                   s["f1"], s["f2"], s["f2"] - s["f1"]])
    blk = spatial_blocks(d["x"], d["yy"], BLOCKS)

    # intervals whose "unseen" arm is only partial: they share an endpoint scene
    man = "manifest_month.json" if cname("features.npz").endswith("_month.npz") \
        else "manifest.json"
    SHARED = shared_endpoint_pairs(man)

    pairs = sorted(set(pair))
    print(f"device={dev}  n={len(y)}  dim={X.shape[1]}  "
          f"{REPS} paired replicates, train size {N_TRAIN}\n")
    print(f"{'pair':<10}{'n':>5}{'n_te':>6}{'seen':>9}{'unseen':>9}{'delta':>9}"
          f"{'wins':>8}  endpoint")
    print("-" * 70)

    rows, deltas = {}, []
    for p in pairs:
        idx_p = np.flatnonzero(pair == p)
        seen_v, unseen_v, te_sizes, tr_sizes = [], [], [], []
        for r in range(REPS):
            rng = np.random.default_rng(7000 + 13 * r)
            # --- spatially blocked split of THIS interval -------------------
            # Whole grid cells go to test, so no training row of either arm
            # sits inside a test cell.  Without this the "seen" arm trains on
            # points metres from its own test points and the contrast measures
            # spatial leakage rather than acquisition-pair novelty.
            cells = rng.permutation(np.unique(blk[idx_p]))
            take, want = [], len(idx_p) // 2
            for c in cells:
                if sum(int((blk[idx_p] == t).sum()) for t in take) >= want:
                    break
                take.append(c)
            te_cells = set(int(c) for c in take)
            in_te = np.array([blk[i] in te_cells for i in idx_p])
            te, rest_p = idx_p[in_te], idx_p[~in_te]
            if len(te) == 0 or len(rest_p) == 0:
                continue
            # both arms draw from other intervals, with test cells removed
            pool_o = np.flatnonzero((pair != p) & ~np.isin(blk, list(te_cells)))
            n_tr = min(N_TRAIN, len(pool_o))
            tr_unseen = rng.choice(pool_o, size=n_tr, replace=False)
            k = min(len(rest_p), n_tr)
            tr_seen = np.concatenate([rest_p[:k],
                                      rng.choice(pool_o, size=n_tr - k, replace=False)])
            unseen_v.append(run(X, yi, tr_unseen, te, n_cls, dev))
            seen_v.append(run(X, yi, tr_seen, te, n_cls, dev))
            te_sizes.append(len(te))
            tr_sizes.append(int(n_tr))
        if not seen_v:
            continue
        sv, uv = np.array(seen_v), np.array(unseen_v)
        dd = uv - sv
        wins = int((dd < 0).sum())
        deltas.append(dd.mean())
        rows[p] = dict(n=len(idx_p), n_test=int(np.mean(te_sizes)),
                       seen=float(sv.mean()), seen_sd=float(sv.std()),
                       unseen=float(uv.mean()), unseen_sd=float(uv.std()),
                       delta=float(dd.mean()), delta_sd=float(dd.std()),
                       unseen_worse_in=wins, reps=len(sv),
                       n_train_mean=float(np.mean(tr_sizes)),
                       n_train_min=int(min(tr_sizes)), n_train_max=int(max(tr_sizes)),
                       shares_endpoint=p in SHARED)
        print(f"{p:<10}{len(idx_p):>5}{int(np.mean(te_sizes)):>6}{sv.mean():>9.4f}"
              f"{uv.mean():>9.4f}{dd.mean():>+9.4f}{wins:>5}/{len(sv)}"
              f"  {'shared' if p in SHARED else 'clean'}")

    pairs = [p for p in pairs if p in rows]
    deltas = np.array(deltas)
    # sample SD, not population: n is 10 intervals, so ddof matters
    se = deltas.std(ddof=1) / np.sqrt(len(deltas))
    worse = int((deltas < 0).sum())
    print("-" * 70)
    print(f"{'mean':<10}{'':>5}{'':>6}{'':>9}{'':>9}{deltas.mean():>+9.4f}")
    print(f"\nunseen worse in {worse}/{len(pairs)} intervals; "
          f"paired delta {deltas.mean():+.4f}  sd {deltas.std(ddof=1):.4f}  "
          f"2*SE {2*se:.4f}  (spread is over the {len(deltas)} interval means)")
    resolved = (worse == len(pairs) and abs(deltas.mean()) > 2 * se
                and abs(deltas.mean()) >= 0.02)
    print(f"verdict under the report's rule (unanimous, >2*SE, >=0.02): "
          f"{'RESOLVED' if resolved else 'not resolved'}")

    clean = np.array([rows[p]["delta"] for p in pairs if not rows[p]["shares_endpoint"]])
    if len(clean):
        print(f"restricted to the {len(clean)} intervals with no shared endpoint: "
              f"{clean.mean():+.4f}")
    else:
        print("no interval is endpoint-clean: under the month rule the scenes chain, "
              "so every 'unseen' arm has seen one endpoint image via a neighbouring pair")

    out = dict(reps=REPS, n_train_cap=N_TRAIN,
               shared_endpoint_intervals=sorted(SHARED), per_interval=rows,
               mean_delta=float(deltas.mean()), sd=float(deltas.std(ddof=1)),
               two_se=float(2 * se), se_over="interval means",
               unseen_worse_in=worse, n_intervals=len(pairs),
               clean_only_delta=float(clean.mean()) if len(clean) else None,
               resolved=bool(resolved))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / oname("loio.json")).write_text(json.dumps(out, indent=1))
    print(f"\nsaved {OUT / oname('loio.json')}")


if __name__ == "__main__":
    main()
