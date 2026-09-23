"""Score the four reference baselines under both official protocol scores.

These are the numbers in docs/RESULTS.md and hackathon/README.md. Re-run this
if you change the chip cache, the encoder, or the protocol; never hand-edit
the tables.

    python scripts/baselines.py --cache-dir <dir> --out out/baselines.json

Every baseline is scored with the same classifier, the same jittered grids and
the same seed. Only the feature matrix changes, so any gap between two rows is
a property of the representation and nothing else.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from heritage_watch.protocol import (CLASSES, MIN_EFFECT, evaluate,
                                     evaluate_interval)


def one_hot(values):
    vals = sorted(set(values))
    return np.array([[1.0 if v == c else 0.0 for c in vals] for v in values])


def majority_classifier():
    """Predicts the most frequent training class, ignoring the features.

    This is the floor. A macro-F1 above it means only that the model has
    noticed the classes are not all the same size.
    """
    from sklearn.dummy import DummyClassifier
    return DummyClassifier(strategy="most_frequent")


def build(dino_path: Path, satlas_path: Path):
    dino = np.load(dino_path, allow_pickle=True)
    sat = np.load(satlas_path, allow_pickle=True)
    if not np.array_equal(dino["fid"], sat["fid"]):
        raise SystemExit("the two caches are not row-aligned; regenerate them together")

    y = dino["y"].astype(str)
    meta = {"x": dino["x"], "yy": dino["yy"], "fid": dino["fid"],
            "pair": dino["pair"].astype(str)}

    return y, meta, [
        ("majority floor", np.zeros((len(y), 1)), majority_classifier,
         "predicts the biggest class; no imagery is read"),
        ("date only", one_hot(meta["pair"]), None,
         "the acquisition interval, one-hot; no imagery is read"),
        ("DINOv2 post-event", dino["f2"], None,
         "the after-image only, through a general-purpose encoder"),
        ("Satlas-MI + SI diff", np.hstack([sat["fmi"], sat["f2"] - sat["f1"]]), None,
         "reference: fused bi-temporal features plus the single-image difference"),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"),
                    help="Directory holding both encoder caches (default: cache).")
    ap.add_argument("--dino", type=Path, help="Override the DINOv2 cache path.")
    ap.add_argument("--satlas", type=Path, help="Override the Satlas cache path.")
    ap.add_argument("--out", type=Path, default=Path("out/baselines.json"))
    ap.add_argument("--skip-interval", action="store_true",
                    help="spatial score only; roughly 4x faster")
    args = ap.parse_args()

    y, meta, rows = build(args.dino or args.cache_dir / "features_month.npz",
                          args.satlas or args.cache_dir / "features_satlas_month.npz")
    print(f"n={len(y)}  classes={len(CLASSES)}  intervals={len(set(meta['pair']))}\n")

    head = f"{'baseline':<22} {'spatial':>9} {'sd':>7}"
    if not args.skip_interval:
        head += f" {'interval':>9} {'sd':>7}"
    print(head)
    print("-" * len(head))

    out = {"n": int(len(y)), "min_effect": MIN_EFFECT, "baselines": []}
    for name, X, clf, note in rows:
        kw = {"clf_factory": clf} if clf else {}
        sp = evaluate(X, y, meta, **kw)
        rec = {"name": name, "note": note, "dim": int(X.shape[1]),
               "spatial": sp.macro_f1, "spatial_sd": sp.sd}
        line = f"{name:<22} {sp.macro_f1:>9.4f} {sp.sd:>7.4f}"
        if not args.skip_interval:
            iv = evaluate_interval(X, y, meta, **kw)
            rec |= {"interval": iv.macro_f1, "interval_sd": iv.sd}
            line += f" {iv.macro_f1:>9.4f} {iv.sd:>7.4f}"
        print(line, flush=True)
        out["baselines"].append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
