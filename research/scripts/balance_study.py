#!/usr/bin/env python3
"""Impact of class imbalance on the selected (merged) representation.

The label set is heavily skewed (New Construction 42% ... Temporary Structure 6%),
so two questions matter for the report:

  1. How much does the balanced class weighting actually buy, and who pays for it?
     -> refit the identical head with sw_i = 1 (unweighted) and compare per class.
  2. Is the per-class ranking explained by class size, or by class difficulty?
     -> report F1 against support, plus recall on the two directional classes.

Everything else (spatial blocking, folds, seeds, in-fold scaling) is identical to
compare_gpu.py so the numbers drop straight into the same table.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from compare_gpu import BLOCKS, CACHE, FOLDS, OUT, SEEDS, fit_transform, set_pca, spatial_blocks
from cachesel import cname, oname

C = 1.0


def fit_logreg(X, y, n_cls, balanced: bool):
    n, d = X.shape
    if balanced:
        cnt = torch.bincount(y, minlength=n_cls).float().clamp_min(1)
        sw = (n / (n_cls * cnt))[y]
    else:
        sw = torch.ones(n, device=X.device, dtype=X.dtype)

    W = torch.zeros(d, n_cls, device=X.device, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(n_cls, device=X.device, dtype=X.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=500, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        ce = torch.nn.functional.cross_entropy(X @ W + b, y, reduction="none")
        loss = 0.5 * (W * W).sum() + C * (sw * ce).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def cv(Xn, y_idx, groups, seed, n_cls, dev, balanced):
    sp = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed)
    X = torch.as_tensor(Xn, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y_idx, dtype=torch.long, device=dev)
    oof = np.empty(len(y_idx), np.int64)
    for tr, te in sp.split(Xn, y_idx, groups):
        tr_t = torch.as_tensor(tr, device=dev)
        te_t = torch.as_tensor(te, device=dev)
        Xtr, Xte = fit_transform(X[tr_t], X[te_t], X.shape[1])
        W, b = fit_logreg(Xtr, yt[tr_t], n_cls, balanced)
        oof[te] = (Xte @ W + b).argmax(1).cpu().numpy()
    return oof


def main() -> None:
    set_pca(0)
    dino = np.load(CACHE / cname("features.npz"), allow_pickle=True)
    sat = np.load(CACHE / cname("features_satlas.npz"), allow_pickle=True)
    assert (dino["fid"] == sat["fid"]).all()

    y = dino["y"].astype(str)
    labs = sorted(set(y))
    y_idx = np.searchsorted(labs, y)
    blk = spatial_blocks(dino["x"], dino["yy"], BLOCKS)
    d1, d2, s1, s2 = dino["f1"], dino["f2"], sat["f1"], sat["f2"]
    X = np.hstack([d1, d2, d2 - d1, s1, s2, s2 - s1])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    supp = {l: int((y == l).sum()) for l in labs}
    print(f"device={dev}  n={len(y)}  merged dim={X.shape[1]}  spatial CV {FOLDS}x{SEEDS} seeds\n")

    out = {"support": supp, "settings": {}}
    store = {}
    for tag, bal in (("balanced", True), ("unweighted", False)):
        macro, per, rec = [], [], []
        for s in range(SEEDS):
            oof = cv(X, y_idx, blk, s, len(labs), dev, bal)
            macro.append(f1_score(y_idx, oof, average="macro"))
            per.append(f1_score(y_idx, oof, average=None, labels=range(len(labs))))
            rec.append(recall_score(y_idx, oof, average=None, labels=range(len(labs))))
        store[tag] = (np.mean(macro), np.std(macro), np.mean(per, 0), np.mean(rec, 0))
        out["settings"][tag] = dict(
            macro_f1=float(np.mean(macro)), macro_std=float(np.std(macro)),
            per_class_f1={l: float(v) for l, v in zip(labs, np.mean(per, 0))},
            per_class_recall={l: float(v) for l, v in zip(labs, np.mean(rec, 0))})

    print(f"{'class':<22}{'n':>5}{'share':>8}"
          f"{'F1 bal':>9}{'F1 unw':>9}{'dF1':>8}{'R bal':>8}{'R unw':>8}{'dR':>8}")
    print("-" * 85)
    order = sorted(labs, key=lambda l: -supp[l])
    for l in order:
        i = labs.index(l)
        fb, fu = store["balanced"][2][i], store["unweighted"][2][i]
        rb, ru = store["balanced"][3][i], store["unweighted"][3][i]
        print(f"{l:<22}{supp[l]:>5}{supp[l]/len(y):>7.1%}"
              f"{fb:>9.3f}{fu:>9.3f}{fb-fu:>+8.3f}{rb:>8.3f}{ru:>8.3f}{rb-ru:>+8.3f}")
    print("-" * 85)
    mb, sb = store["balanced"][0], store["balanced"][1]
    mu, su = store["unweighted"][0], store["unweighted"][1]
    print(f"{'macro-F1':<22}{'':>5}{'':>8}{mb:>9.3f}{mu:>9.3f}{mb-mu:>+8.3f}")
    print(f"\nbalanced   macro-F1 {mb:.4f} +/-{sb:.3f}")
    print(f"unweighted macro-F1 {mu:.4f} +/-{su:.3f}")

    # Is per-class F1 explained by class size?
    n = np.array([supp[l] for l in labs], float)
    for tag in ("balanced", "unweighted"):
        f = store[tag][2]
        r = np.corrcoef(n, f)[0, 1]
        out["settings"][tag]["corr_f1_vs_support"] = float(r)
        print(f"corr(F1, class size) [{tag}]: {r:+.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / oname("balance_study.json")).write_text(json.dumps(out, indent=1))
    print(f"\nsaved {OUT / oname('balance_study.json')}")


if __name__ == "__main__":
    main()
