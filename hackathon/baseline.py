#!/usr/bin/env python3
"""The reference baseline: raw pixels, numpy and scikit-learn, nothing else.

No pretrained model, on purpose. This file exists to show you the official
protocol end to end and to give you a number to beat -- not to suggest a
direction.

    cd hackathon
    python baseline.py

It cuts 64 px chips on the first run (~25 s) and then scores them. CPU only,
about two minutes in total. Each score is 8 replicates of a full
cross-validation and prints only when it has finished, so a quiet minute is
normal, not a hang.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

try:
    from heritage_watch.protocol import (compare, evaluate, evaluate_interval,
                                         load_chips, report)
except ImportError:
    raise SystemExit("Run 'pip install -e .' from the repository root first.")

FLOOR_INTERVAL = 0.1112  # majority-class floor under score 2, measured


def colour_hist(a: np.ndarray, bins: int = 16) -> np.ndarray:
    """Per-channel normalised intensity histogram. (N, 3*bins)"""
    out = []
    for ch in range(3):
        h = np.stack([np.histogram(im[:, :, ch], bins=bins, range=(0, 1))[0]
                      for im in a])
        out.append(h / (a.shape[1] * a.shape[2]))
    return np.concatenate(out, 1).astype(np.float32)


def one_hot(values) -> np.ndarray:
    """Membership indicator for each distinct value. (N, n_distinct)"""
    distinct = sorted(set(values))
    return np.array([[float(v == d) for d in distinct] for v in values])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chips", default=Path("chips_64.npz"), type=Path,
                    help="chip file; cut automatically if missing (default chips_64.npz)")
    ap.add_argument("--images", default=Path("../raw/dataset"), type=Path,
                    help="unzipped dataset root, only needed the first time")
    ap.add_argument("--skip-interval", action="store_true",
                    help="score 1 only. You must still report score 2 in a submission.")
    args = ap.parse_args()

    if not args.chips.exists():
        print(f"{args.chips} not found -- cutting chips from {args.images} (~25 s)",
              flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "make_chips.py"),
                        "--images", str(args.images), "--chip", "64",
                        "--out", str(args.chips)], check=True)

    t1, t2, y, meta = load_chips(args.chips)
    print(f"\nloaded {len(y)} chips of {t1.shape[1]}x{t1.shape[2]} px", flush=True)
    for c in sorted(set(y)):
        print(f"  {c:<22}{(y == c).sum():>5}")
    floor = f1_score(y, np.full(len(y), "New Construction"), average="macro")
    print(f"  majority-class floor: {floor:.4f}")

    # Two feature sets: colour only -- no shape, no texture, no pretrained
    # weights -- and one that contains no pixels whatsoever.
    h1, h2 = colour_hist(t1), colour_hist(t2)
    colour = np.concatenate([h1, h2, h2 - h1], 1)
    date_only = one_hot(meta["pair"])

    print("\n" + "=" * 67)
    print("SCORE 1 of 2: another building, same flight")
    print("=" * 67)
    print("8 replicates x 5 folds each; roughly 20 s per block.", flush=True)

    s_date = evaluate(date_only, y, meta)
    report(s_date, "date only -- the acquisition, NO PIXELS AT ALL")
    s_after = evaluate(h2, y, meta)
    report(s_after, "colour histogram, after-image only")
    s_colour = evaluate(colour, y, meta)
    report(s_colour, "colour histogram, both dates + difference  <- beat this")

    # How to claim an improvement honestly. Both+diff wins every replicate and
    # clears 2*SE, yet the effect is under 0.02, so the protocol still refuses
    # to call it a win. "My mean is higher" is not evidence.
    compare(s_colour, s_after, "both+diff", "after-only")

    if args.skip_interval:
        print("\nSkipped score 2. A submission must report it.")
        return

    print("\n" + "=" * 67)
    print("SCORE 2 of 2: an acquisition you have never seen")
    print("=" * 67)
    print("Each interval is held out in turn; expect this to be much lower.", flush=True)

    i_date = evaluate_interval(date_only, y, meta)
    report(i_date, "date only -- NO PIXELS AT ALL")
    i_after = evaluate_interval(h2, y, meta)
    report(i_after, "colour histogram, after-image only")
    i_colour = evaluate_interval(colour, y, meta)
    report(i_colour, "colour histogram, both dates + difference")

    # The verdict flips: on score 1 the before-image was a (too small) gain,
    # here it is a resolved LOSS. Score 2 asks a different question.
    compare(i_colour, i_after, "both+diff", "after-only")

    print(f"""
{"=" * 67}
Read the date-only rows again.

                     score 1    score 2
  majority floor      {floor:.4f}     {FLOOR_INTERVAL:.4f}
  date only           {s_date.macro_f1:.4f}     {i_date.macro_f1:.4f}
  colour histogram    {s_colour.macro_f1:.4f}     {i_colour.macro_f1:.4f}

A feature matrix containing NO PIXELS scores {s_date.macro_f1:.3f} on score 1, most of
the way to a colour model, because different kinds of change happened in
different years. Held out from its own acquisition it collapses to {i_date.macro_f1:.3f}.

That is why you report two numbers: score 1 alone lets a model bank the
calendar without ever looking at a building.

Report both, and run compare() before believing any result -- including
your own. See README.md for what to try next.
{"=" * 67}""")


if __name__ == "__main__":
    main()
