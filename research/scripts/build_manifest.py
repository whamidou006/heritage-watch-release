#!/usr/bin/env python3
"""Build the usable-label manifest for Herat bi-temporal change classification.

A label is *usable* when:
  1. its point falls inside the footprint of the two scenes it will actually use, and
  2. both endpoints of its change-layer can be matched to a scene.

PAIRING (see --pairing)
----------------------
The CSV carries two descriptions of the same interval:

    layer   change_between_202205_and_202310      <- year AND month
    year    20222023                              <- lossy, year only

`year` is a coarsened derivation of `layer`. Pairing on it loses the month, which
matters because a single year can hold three scenes ten months apart. Measured on
this archive, the year-based rule picked a different scene from the one the layer
names in 6 of the 9 month-specified layers, and the resulting image window covered
as little as 43% of the annotated window:

    2022->2023   annotated 2022-05 -> 2023-10, imaged 2022-05-11 -> 2023-04-22
                 the after-image is 189 days early, over 197 labels

Both error directions corrupt a label the same way: the pair is tagged with a change
class while the pixels show no change, because the change happens outside the window
that was actually imaged. 476 of 843 labels sat in a pair off by more than 45 days.

So the default is `--pairing month`: parse YYYYMM from the layer name and take the
scene nearest each stated month, subject to --tol days. `--pairing year` restores the
old latest-in-year/earliest-in-year behaviour for comparison.

FOOTPRINT
---------
The footprint test uses the intersection of the two scenes the label will actually be
cut from, rather than one arbitrary scene. On this archive all 21 scenes share one
grid so the two rules agree exactly (measured: 1003 kept either way), but keying off
`next(glob(...))` is only correct by coincidence and would silently mis-filter a site
with irregular coverage.
"""
from __future__ import annotations

import csv
import argparse
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

import rasterio

DATA = Path(__file__).resolve().parents[2] / "extracted" / "dataset"
TM_DIR = DATA / "herat_site_sat_images" / "herat_site_TM_z19"
OUT = Path(__file__).resolve().parents[1] / "cache"

# Classes kept for modelling. 'Other' (25) is a semantic grab-bag and
# 'Reconstruction' (1) is a single sample, so neither can be learned or scored.
KEEP = ["New Construction", "Solar Panel", "Destruction", "Temporary Structure"]

_DATE_RE = re.compile(r"z19_(\d{4})-(\d{2})-(\d{2})\.tif$")
_MONTH_RE = re.compile(r"(\d{6})")


def all_scenes() -> list[tuple[dt.date, Path]]:
    out = []
    for p in sorted(TM_DIR.glob("*.tif")):
        m = _DATE_RE.search(p.name)
        if m:
            y, mo, d = m.groups()
            out.append((dt.date(int(y), int(mo), int(d)), p))
    return sorted(out)


def scenes_by_year(scenes) -> dict[str, list[tuple[dt.date, Path]]]:
    out: dict[str, list[tuple[dt.date, Path]]] = {}
    for d, p in scenes:
        out.setdefault(f"{d.year:04d}", []).append((d, p))
    return out


def nearest(scenes, target: dt.date, tol: int):
    """Scene closest to `target`, or None if nothing lands within `tol` days."""
    d, p = min(scenes, key=lambda s: abs((s[0] - target).days))
    return (d, p) if abs((d - target).days) <= tol else None


def resolve(layer: str, year: str, scenes, by_year, tol: int, mode: str):
    """Return (t1, t2, how) or (None, None, reason)."""
    months = _MONTH_RE.findall(layer)
    if mode == "month" and len(months) == 2:
        # mid-month is the least-biased reading of a YYYYMM label
        tg = [dt.date(int(m[:4]), int(m[4:]), 15) for m in months]
        a, b = nearest(scenes, tg[0], tol), nearest(scenes, tg[1], tol)
        if a is None or b is None:
            return None, None, "no_scene_near_stated_month"
        if a[0] >= b[0]:
            return None, None, "degenerate_pair"
        return a, b, "month"

    ya, yb = year[:4], year[4:]
    if ya not in by_year or yb not in by_year:
        return None, None, "missing_endpoint_image"
    return by_year[ya][-1], by_year[yb][0], "year"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-all", action="store_true",
                    help="retain 'Other'/'Reconstruction' instead of dropping them")
    ap.add_argument("--pairing", choices=("month", "year"), default="month",
                    help="month: use YYYYMM from the layer name (default). "
                         "year: legacy latest-in-year/earliest-in-year.")
    ap.add_argument("--tol", type=int, default=75,
                    help="max days between a stated month and the chosen scene")
    ap.add_argument("--out", default="manifest.json")
    args = ap.parse_args()

    scenes = all_scenes()
    by_year = scenes_by_year(scenes)
    bounds, crs = {}, None
    for d, p in scenes:
        with rasterio.open(p) as src:
            bounds[p] = (src.bounds.left, src.bounds.bottom,
                         src.bounds.right, src.bounds.top)
            crs = str(src.crs)

    rows = list(csv.DictReader(open(DATA / "Herat_all_changes.csv")))

    recs, drop, how = [], Counter(), Counter()
    for r in rows:
        cat = r["category"].strip()
        # NB: the X column has a trailing space in the header.
        try:
            x = float(r["X "].replace(",", "."))
            y = float(r["Y"].replace(",", "."))
        except (ValueError, KeyError):
            drop["bad_coord"] += 1
            continue

        a, b, tag = resolve(r["layer"], r["year"], scenes, by_year, args.tol, args.pairing)
        if a is None:
            drop[tag] += 1
            continue

        # footprint of the two scenes this label will actually be cut from
        W1, S1, E1, N1 = bounds[a[1]]
        W2, S2, E2, N2 = bounds[b[1]]
        W, S, E, N = max(W1, W2), max(S1, S2), min(E1, E2), min(N1, N2)
        if not (W <= x <= E and S <= y <= N):
            drop["outside_footprint"] += 1
            continue

        if not args.keep_all and cat not in KEEP:
            drop[f"class_excluded:{cat}"] += 1
            continue

        how[tag] += 1
        recs.append(
            dict(fid=int(r["fid"]), category=cat, x=x, y=y,
                 pair=r["year"], year_a=r["year"][:4], year_b=r["year"][4:],
                 t1_date=a[0].isoformat(), t2_date=b[0].isoformat(),
                 t1=str(a[1]), t2=str(b[1]))
        )

    OUT.mkdir(parents=True, exist_ok=True)
    classes = sorted({r["category"] for r in recs}) if args.keep_all else KEEP
    with open(OUT / args.out, "w") as f:
        json.dump(dict(crs=crs, pairing=args.pairing, tol_days=args.tol,
                       classes=classes, records=recs), f, indent=1)

    print(f"pairing={args.pairing}  tol={args.tol}d  scenes={len(scenes)}")
    print(f"CSV rows            : {len(rows)}")
    for k, v in drop.most_common():
        print(f"  dropped {k:<32}: {v}")
    print(f"usable labels       : {len(recs)}   resolved by {dict(how)}")
    print()
    for k, v in Counter(r["category"] for r in recs).most_common():
        print(f"  {v:5d}  {k}")
    print()
    print("per change-pair:")
    used = set()
    for k, v in sorted(Counter(r["pair"] for r in recs).items()):
        ex = next(r for r in recs if r["pair"] == k)
        gap = (dt.date.fromisoformat(ex["t2_date"]) - dt.date.fromisoformat(ex["t1_date"])).days
        used |= {ex["t1_date"], ex["t2_date"]}
        print(f"  {k}  n={v:4d}   {ex['t1_date']} -> {ex['t2_date']}  ({gap}d)")
    print(f"\ndistinct scenes used: {len(used)} of {len(scenes)}")


if __name__ == "__main__":
    main()
