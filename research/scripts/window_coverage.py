#!/usr/bin/env python3
"""How well does each pair's imagery cover the interval the analyst annotated?

The layer names state an annotated interval in months (e.g. `change_between_201711
_and_201802`).  The manifest then picks a real scene for each endpoint.  This script
measures the gap between the two, under both pairing rules, and archives the result so
the table in the report is reproducible rather than quoted from a scratch session.

Convention (stated because the answer depends on it): a `YYYYMM` token is read as the
15th of that month, the least-biased reading of a month label.  Coverage is

    |[i1, i2] intersect [a1, a2]| / |[a1, a2]|

i.e. the fraction of the ANNOTATED window that the IMAGED window actually spans.  A
pair can therefore score below 100 % either by starting late or by ending early.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from build_manifest import OUT as CACHE, all_scenes, resolve, scenes_by_year
from cachesel import oname

_OUT = Path(__file__).resolve().parents[1] / "out"
TOL = 75


def mid(month_token: str) -> dt.date:
    return dt.date(int(month_token[:4]), int(month_token[4:]), 15)


def coverage(i1: dt.date, i2: dt.date, a1: dt.date, a2: dt.date) -> float:
    lo, hi = max(i1, a1), min(i2, a2)
    span = (a2 - a1).days
    if span <= 0:
        return float("nan")
    return max(0, (hi - lo).days) / span


def main() -> None:
    import re
    scenes = all_scenes()
    by_year = scenes_by_year(scenes)
    rows = []

    # the layer -> (layer name, YYYYYYYY key) pairs actually present in the labels
    man = json.loads((CACHE / "manifest_month.json").read_text())
    recs = man["records"] if isinstance(man, dict) else man
    layers = sorted({(r["layer"], r["pair"]) for r in recs if "layer" in r})
    if not layers:                      # manifest does not carry layer names
        import csv
        src = Path(__file__).resolve().parents[2] / "extracted" / "dataset"
        csvp = next(src.rglob("Herat_all_changes.csv"))
        with open(csvp) as fh:
            layers = sorted({(r["layer"], None) for r in csv.DictReader(fh)})

    for layer, _ in layers:
        months = re.findall(r"(\d{6})", layer)
        if len(months) != 2:
            continue                    # year-only layer: no annotated window to score
        a1, a2 = mid(months[0]), mid(months[1])
        year = months[0][:4] + months[1][:4]
        row = {"layer": layer, "annotated": [a1.isoformat(), a2.isoformat()]}
        for mode in ("year", "month"):
            a, b, how = resolve(layer, year, scenes, by_year, TOL, mode)
            if a is None:
                row[mode] = {"status": how}
                continue
            off = [(a[0] - a1).days, (b[0] - a2).days]
            row[mode] = {"imaged": [a[0].isoformat(), b[0].isoformat()],
                         "coverage": round(coverage(a[0], b[0], a1, a2), 4),
                         "offset_days": off, "max_abs_offset": max(abs(o) for o in off),
                         "how": how}
        rows.append(row)

    hdr = f"{'layer':<34}{'year-rule imaged':<26}{'cov':>7}{'off':>6}" \
          f"  {'month-rule imaged':<26}{'cov':>7}{'off':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        a = " -> ".join(r["annotated"])
        y, m = r["year"], r["month"]
        ys = " -> ".join(y["imaged"]) if "imaged" in y else y["status"]
        ms = " -> ".join(m["imaged"]) if "imaged" in m else m["status"]
        yc = f"{y['coverage']:.1%}" if "coverage" in y else "--"
        mc = f"{m['coverage']:.1%}" if "coverage" in m else "--"
        yo = f"{y['max_abs_offset']}d" if "coverage" in y else "--"
        mo = f"{m['max_abs_offset']}d" if "coverage" in m else "--"
        print(f"{r['layer'][:33]:<34}{ys:<26}{yc:>7}{yo:>6}  {ms:<26}{mc:>7}{mo:>6}")

    kept = [r for r in rows if "coverage" in r["month"]]
    moved = [r for r in kept if r["year"].get("imaged") != r["month"]["imaged"]]
    print(f"\npairs with a month-specified window: {len(rows)}")
    print(f"  surviving the month rule:          {len(kept)}")
    print(f"  endpoint imagery changed:          {len(moved)}")
    if kept:
        ym = [r["year"]["coverage"] for r in kept if "coverage" in r["year"]]
        mm = [r["month"]["coverage"] for r in kept]
        yo = [r["year"]["max_abs_offset"] for r in kept if "coverage" in r["year"]]
        mo = [r["month"]["max_abs_offset"] for r in kept]
        print(f"  mean coverage, year rule:  {sum(ym)/len(ym):.1%}  "
              f"worst-endpoint offset median {sorted(yo)[len(yo)//2]}d, max {max(yo)}d  (n={len(ym)})")
        print(f"  mean coverage, month rule: {sum(mm)/len(mm):.1%}  "
              f"worst-endpoint offset median {sorted(mo)[len(mo)//2]}d, max {max(mo)}d  (n={len(mm)})")

    _OUT.mkdir(parents=True, exist_ok=True)
    p = _OUT / "window_coverage.json"
    p.write_text(json.dumps({"tol_days": TOL, "month_token_read_as": "15th",
                             "rows": rows}, indent=1))
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
