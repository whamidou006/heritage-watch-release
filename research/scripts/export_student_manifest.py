#!/usr/bin/env python3
"""Emit the student-facing manifest from a work manifest.

The work manifest carries absolute paths to the imagery; the shipped one is
relative to whatever root the student passes as --images. Everything else
(fid, class, coordinates, pair, dates) is copied through unchanged, so the
shipped file and the one used for the report's numbers describe the same
samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio

ROOT = Path(__file__).resolve().parents[2]
ANCHOR = "herat_site_sat_images"
NOTE = "t1/t2 are relative to the imagery root passed as --images"


def relative(p: str) -> str:
    parts = Path(p).parts
    if ANCHOR not in parts:
        raise ValueError(f"{ANCHOR!r} not in {p}")
    return str(Path(*parts[parts.index(ANCHOR):]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=ROOT / "work" / "cache" / "manifest_month.json")
    ap.add_argument("--out", default=ROOT / "hackathon" / "data" / "manifest.json")
    args = ap.parse_args()

    src = json.loads(Path(args.src).read_text())
    recs = src["records"]

    # The footprint the points were filtered against. Every scene in this
    # archive carries an identical bounds tuple, so the per-pair intersection
    # collapses to one rectangle; assert that rather than trust it.
    scenes = sorted({r[k] for r in recs for k in ("t1", "t2")})
    tuples = set()
    for s in scenes:
        with rasterio.open(s) as ds:
            tuples.add((ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top))
    assert len(tuples) == 1, f"scenes disagree on bounds: {tuples}"
    bounds = list(next(iter(tuples)))

    out = {
        "crs": src["crs"],
        "bounds": bounds,
        "classes": src["classes"],
        "records": [
            {**{k: r[k] for k in ("fid", "category", "x", "y", "pair",
                                  "year_a", "year_b", "t1_date", "t2_date")},
             "t1": relative(r["t1"]), "t2": relative(r["t2"])}
            for r in recs
        ],
        "note": NOTE,
    }

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(out['records'])} records -> {args.out}")
    print(f"pairing={src.get('pairing')} tol_days={src.get('tol_days')} "
          f"scenes_used={len(scenes)}")
    print(f"bounds={bounds}")


if __name__ == "__main__":
    main()
