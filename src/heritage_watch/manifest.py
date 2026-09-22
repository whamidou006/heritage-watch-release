"""Resolve annotation endpoints to scenes and filter against their intersection.

Layer names encode YYYYMM, not merely YYYY. Month pairing targets the 15th
and refuses scenes outside the tolerance. Year-only layers fall back to the
latest scene of the first year and earliest scene of the second. This is a
provenance correction, not a claimed accuracy improvement.
"""
from collections import Counter
from contextlib import ExitStack
import csv
import datetime as dt
import json
import math
from pathlib import Path
import re

import rasterio

from .chips import chip_window, validate_raster


def all_scenes(config):
    pattern = re.compile(config.scene_date_regex)
    scenes = []
    for path in sorted(config.imagery_dir.glob("*.tif")):
        match = pattern.search(path.name)
        if match:
            try:
                date = dt.date(*map(int, match.groups()))
            except ValueError as exc:
                raise ValueError(f"invalid scene date in {path.name}") from exc
            scenes.append((date, path))
    if not scenes:
        raise ValueError(f"no dated .tif scenes match scene_date_regex in {config.imagery_dir}")
    return sorted(scenes)


def nearest(scenes, target, tolerance):
    date, path = min(scenes, key=lambda s: abs((s[0] - target).days))
    return (date, path) if abs((date - target).days) <= tolerance else None


def resolve(layer, year, scenes, by_year, tolerance=75, mode="month"):
    """Return (before, after, provenance), or (None, None, drop reason)."""
    if not re.fullmatch(r"\d{8}", year):
        return None, None, "invalid_year_interval"
    months = re.findall(r"(\d{6})", layer)
    if mode == "month" and len(months) == 2:
        try:
            targets = [dt.date(int(m[:4]), int(m[4:]), 15) for m in months]
        except ValueError:
            return None, None, "invalid_layer_month"
        a, b = [nearest(scenes, t, tolerance) for t in targets]
        if a is None or b is None:
            return None, None, "no_scene_near_stated_month"
        how = "month"
    else:
        ya, yb = year[:4], year[4:]
        if ya not in by_year or yb not in by_year:
            return None, None, "missing_endpoint_image"
        a, b, how = by_year[ya][-1], by_year[yb][0], "year"
    if a[0] >= b[0]:
        return None, None, "degenerate_pair"
    return a, b, how


def build_manifest(config):
    """Build final usable records, including the full-chip edge rejection.

    Coordinates must already be in the raster CRS. The mutually-exclusive
    ledger follows pairing -> actual pair footprint -> class -> chip edge;
    therefore its reason counts need not match a footprint-first report ledger.
    """
    scenes = all_scenes(config)
    by_year = {}
    for date, path in scenes:
        by_year.setdefault(str(date.year), []).append((date, path))
    with config.annotation_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(config.columns.values()) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing exact CSV headers {sorted(missing)!r}; "
                             "check whitespace, especially 'X '")
        rows = list(reader)
    recs, drop, how = [], Counter(), Counter()
    ids = set()
    with ExitStack() as stack:
        rasters, reference = {}, None
        for _, path in scenes:
            src = stack.enter_context(rasterio.open(path))
            reference = validate_raster(src, reference)
            rasters[path] = src
        for line, raw in enumerate(rows, start=2):
            r = {k: raw[v] for k, v in config.columns.items()}
            if any(not isinstance(v, str) for v in r.values()) or None in raw:
                raise ValueError(f"CSV line {line}: wrong field count; check quoting/delimiter")
            try:
                fid = int(r["fid"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid annotation fid: {r['fid']!r}") from exc
            if fid in ids:
                raise ValueError(f"duplicate annotation fid: {fid}")
            ids.add(fid)
            try:
                x, y = (float(r[k].replace(",", ".")) for k in ("x", "y"))
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("nonfinite")
            except (ValueError, AttributeError):
                drop["bad_coord"] += 1
                continue
            year = r["year"].strip()
            a, b, tag = resolve(r["layer"], year, scenes, by_year,
                                config.tolerance_days, config.pairing)
            if a is None:
                drop[tag] += 1
                continue
            sa, sb = rasters[a[1]], rasters[b[1]]
            wa, so, ea, no = sa.bounds
            wb, ss, eb, ns = sb.bounds
            if not (max(wa, wb) <= x <= min(ea, eb) and max(so, ss) <= y <= min(no, ns)):
                drop["outside_footprint"] += 1
                continue
            cat = r["category"].strip()
            if cat in config.drop_categories:
                drop[f"class_excluded:{cat}"] += 1
                continue
            if cat not in config.classes:
                drop[f"class_not_kept:{cat}"] += 1
                continue
            if any(chip_window(s, x, y, config.chip_size) is None for s in (sa, sb)):
                drop["chip_edge"] += 1
                continue
            how[tag] += 1
            recs.append(dict(fid=fid, category=cat, x=x, y=y, pair=year,
                             t1_date=a[0].isoformat(), t2_date=b[0].isoformat(),
                             t1=str(a[1].relative_to(config.imagery_dir)),
                             t2=str(b[1].relative_to(config.imagery_dir))))
    return dict(crs=str(reference[0]), pairing=config.pairing,
                tolerance_days=config.tolerance_days, chip_size=config.chip_size,
                classes=config.classes, raw_count=len(rows), scene_count=len(scenes),
                drop_ledger=dict(drop), resolved_by=dict(how), records=recs)


def print_manifest(man):
    print(f"pairing={man['pairing']} tolerance={man['tolerance_days']}d scenes={man['scene_count']}")
    print(f"CSV rows: {man['raw_count']}")
    for reason, n in man["drop_ledger"].items():
        print(f"  dropped {reason}: {n}")
    recs = man["records"]
    print(f"usable labels: {len(recs)}")
    for label in man["classes"]:
        print(f"  {label}: {sum(r['category'] == label for r in recs)}")
    print("per change-pair: before -> after, days, n, class counts")
    for pair in sorted({r["pair"] for r in recs}):
        subset = [r for r in recs if r["pair"] == pair]
        r = subset[0]
        gap = (dt.date.fromisoformat(r["t2_date"]) - dt.date.fromisoformat(r["t1_date"])).days
        print(f"  {pair}: {r['t1_date']} -> {r['t2_date']} ({gap}d), "
              f"n={len(subset)}, {dict(Counter(r['category'] for r in subset))}")


def write_manifest(man, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=2) + "\n")
