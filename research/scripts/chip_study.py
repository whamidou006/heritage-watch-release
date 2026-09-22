#!/usr/bin/env python3
"""Sensitivity to chip size - the one free hyperparameter in the pipeline.

128 px (~32 m) was chosen a priori. This checks whether that choice matters,
using the same merged representation and spatial CV protocol as everything else.
Note n varies slightly: larger chips clip against the raster edge more often.
"""
from __future__ import annotations
import json
import numpy as np, torch
from cleaning_study import load, score
from compare_gpu import OUT, SEEDS, set_pca
from cachesel import cname, oname

def main() -> None:
    set_pca(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {32: ("features_c32.npz", "features_satlas_c32.npz"),
           64: ("features_c64.npz", "features_satlas_c64.npz"),
           128: ("features.npz", "features_satlas.npz"),
           256: ("features_c256.npz", "features_satlas_c256.npz")}
    print(f"device={dev}  spatial CV, {SEEDS} seeds, merged representation\n")
    print(f"{'chip px':>8}{'ground m':>10}{'n':>6}{'macro-F1':>18}")
    print("-" * 42)
    out = {}
    for c, (a, b) in cfg.items():
        X, y, x, yy = load(a, b)
        m, sd = score(X, y, x, yy, dev)
        out[c] = dict(n=len(y), ground_m=round(c * 0.25, 1), macro_f1=m, std=sd)
        print(f"{c:>8}{c*0.25:>10.0f}{len(y):>6}{m:>12.4f} +/-{sd:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / oname("chip_study.json")).write_text(json.dumps(out, indent=1))
    print(f"\nsaved {OUT / oname('chip_study.json')}")

if __name__ == "__main__":
    main()
