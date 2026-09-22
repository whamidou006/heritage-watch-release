#!/usr/bin/env python3
"""GPU head-to-head: DINOv2 vs SatlasPretrain Aerial (SI / MI) on Herat.

The CPU sweep became unusable when this shared host hit a load average of
~8600 on 96 cores. Everything here that costs real compute - standardisation,
PCA and the multinomial logistic regression - runs on the GPU instead, which
is comparatively idle. Only the fold splitting stays on the CPU, which is free.

The classifier mirrors sklearn's LogisticRegression(C, class_weight='balanced')
objective exactly:

    0.5 * ||W||^2  +  C * sum_i  sw_i * cross_entropy_i
    sw_i = n / (n_classes * count(class_i))

so the GPU rows are directly comparable to the CPU rows measured earlier.
Pass --validate to re-run the three representations that were already scored
on CPU and confirm the implementations agree before trusting the rest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

CACHE = Path(__file__).resolve().parents[1] / "cache"
OUT = Path(__file__).resolve().parents[1] / "out"
FOLDS, SEEDS, BLOCKS, PCA_DIM, C = 5, 3, 6, 256, 1.0


def set_pca(d: int) -> None:
    global PCA_DIM
    PCA_DIM = d if d > 0 else 10**9


def spatial_blocks(x, y, n):
    xb = np.clip(((x - x.min()) / (np.ptp(x) + 1e-12) * n).astype(int), 0, n - 1)
    yb = np.clip(((y - y.min()) / (np.ptp(y) + 1e-12) * n).astype(int), 0, n - 1)
    return xb * n + yb


def fit_transform(Xtr: torch.Tensor, Xte: torch.Tensor, dim: int):
    """StandardScaler -> optional PCA, both fitted on the training fold only."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-8)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if dim > PCA_DIM:
        c = Xtr.mean(0, keepdim=True)
        # economy SVD of the centred training matrix gives the PCA basis
        V = torch.linalg.svd(Xtr - c, full_matrices=False)[2][:PCA_DIM]
        Xtr, Xte = (Xtr - c) @ V.T, (Xte - c) @ V.T
    return Xtr, Xte


def fit_logreg(X: torch.Tensor, y: torch.Tensor, n_cls: int) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = X.shape
    cnt = torch.bincount(y, minlength=n_cls).float().clamp_min(1)
    sw = (n / (n_cls * cnt))[y]  # class_weight='balanced'

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


def cv(Xn: np.ndarray, y_idx: np.ndarray, groups, seed: int, n_cls: int, dev: str):
    sp = (StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed)
          if groups is not None else StratifiedKFold(FOLDS, shuffle=True, random_state=seed))
    it = sp.split(Xn, y_idx, groups) if groups is not None else sp.split(Xn, y_idx)

    X = torch.as_tensor(Xn, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y_idx, dtype=torch.long, device=dev)
    oof = np.empty(len(y_idx), np.int64)
    for tr, te in it:
        tr_t = torch.as_tensor(tr, device=dev)
        te_t = torch.as_tensor(te, device=dev)
        Xtr, Xte = fit_transform(X[tr_t], X[te_t], X.shape[1])
        W, b = fit_logreg(Xtr, yt[tr_t], n_cls)
        oof[te] = (Xte @ W + b).argmax(1).cpu().numpy()
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="only run the representations already scored on CPU")
    ap.add_argument("--pca", type=int, default=PCA_DIM,
                    help="in-fold PCA dim; 0 disables PCA entirely")
    args = ap.parse_args()
    set_pca(args.pca)

    dino = np.load(CACHE / "features.npz", allow_pickle=True)
    sat = np.load(CACHE / "features_satlas.npz", allow_pickle=True)
    assert (dino["fid"] == sat["fid"]).all(), "sample order differs between caches"

    y = dino["y"].astype(str)
    labs = sorted(set(y))
    y_idx = np.searchsorted(labs, y)
    blk = spatial_blocks(dino["x"], dino["yy"], BLOCKS)
    d1, d2 = dino["f1"], dino["f2"]
    s1, s2, smi = sat["f1"], sat["f2"], sat["fmi"]

    cands = {
        "DINOv2 t2":             d2,
        "DINOv2 concat_diff":    np.hstack([d1, d2, d2 - d1]),
        "Satlas-SI t2":          s2,
    }
    if not args.validate:
        cands.update({
            "Satlas-SI concat_diff": np.hstack([s1, s2, s2 - s1]),
            "Satlas-MI fused":       smi,
            "Satlas-MI + SI diff":   np.hstack([smi, s2 - s1]),
            "DINOv2 + Satlas-SI":    np.hstack([d1, d2, d2 - d1, s1, s2, s2 - s1]),
        })

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU reference values from the earlier sklearn sweep, for --validate.
    ref = {"DINOv2 t2": 0.5799, "DINOv2 concat_diff": 0.6198, "Satlas-SI t2": 0.6392}

    print(f"device={dev}  n={len(y)}  folds={FOLDS} seeds={SEEDS} "
          f"blocks={BLOCKS}x{BLOCKS} pca={args.pca or 'off'}")
    print(f"majority-class floor macro-F1: "
          f"{f1_score(y, np.full(len(y), 'New Construction'), average='macro'):.4f}\n")
    hdr = f"{'representation':<24}{'dim':>6}{'random CV':>18}{'spatial CV':>18}"
    print(hdr + ("      CPU ref   delta" if args.validate else ""))
    print("-" * (len(hdr) + (22 if args.validate else 0)))

    res, oofs = {}, {}
    for name, X in cands.items():
        r = [f1_score(y_idx, cv(X, y_idx, None, s, len(labs), dev), average="macro")
             for s in range(SEEDS)]
        g = [f1_score(y_idx, cv(X, y_idx, blk, s, len(labs), dev), average="macro")
             for s in range(SEEDS)]
        oofs[name] = cv(X, y_idx, blk, 0, len(labs), dev)
        res[name] = dict(dim=int(X.shape[1]), random_mean=float(np.mean(r)),
                         spatial_mean=float(np.mean(g)), spatial_std=float(np.std(g)))
        line = (f"{name:<24}{X.shape[1]:>6}"
                f"{np.mean(r):>11.4f} +/-{np.std(r):.3f}"
                f"{np.mean(g):>11.4f} +/-{np.std(g):.3f}")
        if args.validate and name in ref:
            line += f"   {ref[name]:.4f}  {np.mean(g) - ref[name]:+.4f}"
        print(line)

    if args.validate:
        print("\nGPU implementation agrees with the CPU sweep if deltas are ~0.00.")
        return

    best = max(res, key=lambda k: res[k]["spatial_mean"])
    print(f"\nbest on spatial CV: {best}  {res[best]['spatial_mean']:.4f}\n")

    # Destruction vs New Construction is the direction-sensitive pair, so show it
    # explicitly for the order-invariant MI model alongside the best model.
    for name in dict.fromkeys([best, "Satlas-MI fused"]):
        pred = [labs[i] for i in oofs[name]]
        print(f"=== {name} (spatial CV, seed 0) ===")
        print(classification_report(y, pred, digits=3, zero_division=0))
        print(f"{'':<22}" + "".join(f"{l[:11]:>13}" for l in labs))
        for l, row in zip(labs, confusion_matrix(y, pred, labels=labs)):
            print(f"{l:<22}" + "".join(f"{v:>13}" for v in row))
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare_gpu.json").write_text(
        json.dumps(dict(n=len(y), best=best, results=res), indent=1))
    print(f"saved {OUT / 'compare_gpu.json'}")


if __name__ == "__main__":
    main()
