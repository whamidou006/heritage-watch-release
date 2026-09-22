#!/usr/bin/env python3
"""The report's decision rule, implemented once.

An effect counts as RESOLVED only if all three bars are cleared:

  1. unanimous   every paired replicate difference has the same sign.  A tie
                 (exact zero) breaks unanimity - it is not evidence of an effect.
  2. > 2*SE      |mean| exceeds twice the standard error of the paired
                 differences, using the SAMPLE standard deviation (ddof=1).
                 With 6-8 replicates the population SD understates the spread
                 by ~7-9 %, which is enough to flip a borderline verdict.
  3. >= MDE      |mean| is at least the minimum effect the protocol can
                 detect (0.02 macro-F1).  An effect can be perfectly consistent
                 and still be too small to act on.

Every study imports `verdict` so the published JSON and the prose cannot drift
apart.  Three earlier copies of this logic disagreed with each other and with
the report: one allowed a dissenting replicate and omitted the MDE bar, one
compared against SD instead of 2*SE, and one counted ties as unanimous.
"""
from __future__ import annotations

import numpy as np

MDE = 0.02


def verdict(diffs, mde: float = MDE) -> dict:
    """Judge a set of paired differences. `diffs` is array-like, one per replicate."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    unanimous = (pos == n) or (neg == n)
    over_se = n > 1 and abs(mean) > 2 * se
    over_mde = abs(mean) >= mde
    res = bool(unanimous and over_se and over_mde)
    if res:
        label = "resolved"
    elif unanimous and over_mde:
        label = "consistent, under 2*SE"
    elif unanimous and over_se:
        label = "consistent but below MDE"
    elif abs(mean) < sd:
        label = "within noise"
    else:
        label = "suggestive"
    return dict(delta=mean, sd=sd, two_se=float(2 * se), n=n,
                positive=pos, negative=neg, ties=n - pos - neg,
                unanimous=bool(unanimous), over_two_se=bool(over_se),
                over_mde=bool(over_mde), mde=mde,
                resolved=res, verdict=label)


def line(name: str, v: dict, width: int = 26) -> str:
    return (f"{name:<{width}}{v['delta']:>+9.4f}{v['sd']:>8.4f}"
            f"{v['two_se']:>9.4f}{max(v['positive'], v['negative']):>4}/{v['n']}"
            f"   {v['verdict']}")


if __name__ == "__main__":
    # self-check: the three bars, and the cases the old copies got wrong
    import sys
    ok = True

    def chk(name, got, want):
        global ok
        if got != want:
            print(f"FAIL {name}: got {got}, want {want}")
            ok = False

    chk("tie breaks unanimity",
        verdict([0.03] * 7 + [0.0])["resolved"], False)
    chk("one dissent breaks unanimity",
        verdict([0.03] * 7 + [-0.01])["resolved"], False)
    chk("consistent but below MDE",
        verdict([0.0197] * 8)["resolved"], False)
    chk("unanimous, large, tight",
        verdict([0.03, 0.028, 0.031, 0.029, 0.03, 0.027, 0.032, 0.03])["resolved"], True)
    chk("sample SD is used",
        round(verdict([0.0063, 0.0357])["two_se"], 4), 0.0294)
    chk("negative effects resolve too",
        verdict([-0.05] * 8)["resolved"], True)
    print("decision.py self-check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
