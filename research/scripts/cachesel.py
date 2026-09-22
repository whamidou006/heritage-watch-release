"""Select which feature-cache generation a study reads.

The pipeline exists in two generations that differ only in how a label's two
acquisition dates are resolved:

  ""      legacy, year-resolved pairs   (kept so old numbers can be reproduced)
  "_month" current, month-resolved pairs (the reported system)

Set HW_SUFFIX=_month to run any study against the current data.  Output JSON is
renamed in step so the two generations never overwrite each other.
"""
from __future__ import annotations

import os

# features_dinov2.npz is a byte-identical copy of features.npz kept for
# readability in haste_control; the month generation has only one file.
_ALIAS = {"features_dinov2.npz": "features.npz"}


def suffix() -> str:
    return os.environ.get("HW_SUFFIX", "")


def cname(base: str) -> str:
    """Map a legacy cache filename onto the selected generation.

    Idempotent: a name that already carries the generation suffix is returned
    unchanged, so passing an explicit `features_month.npz` on the command line
    does not produce `features_month_month.npz`.
    """
    suf = suffix()
    if not suf:
        return base
    base = _ALIAS.get(base, base)
    if base.endswith(f"{suf}.npz"):
        return base
    return base.replace(".npz", f"{suf}.npz")


def oname(base: str) -> str:
    """Map an output filename onto the selected generation. Idempotent."""
    suf = suffix()
    if not suf:
        return base
    if base.endswith(f"{suf}.json"):
        return base
    return base.replace(".json", f"{suf}.json")
