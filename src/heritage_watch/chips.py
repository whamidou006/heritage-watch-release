"""Georeferenced RGB chip extraction; incomplete windows are never padded."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def chip_window(src, x, y, chip):
    row, col = src.index(x, y)
    c0, r0 = col - chip // 2, row - chip // 2
    if c0 < 0 or r0 < 0 or c0 + chip > src.width or r0 + chip > src.height:
        return None
    return Window(c0, r0, chip, chip)


def validate_raster(src, reference=None):
    if src.crs is None:
        raise ValueError(f"{src.name}: raster has no CRS")
    if src.count != 3 or any(d != "uint8" for d in src.dtypes):
        raise ValueError(f"{src.name}: requires three-band uint8 RGB imagery")
    t = src.transform
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
        raise ValueError(f"{src.name}: reproject rotated/south-up imagery to a north-up grid")
    if reference is not None:
        crs, ref = reference
        if src.crs != crs or not np.allclose([t.a, t.e], [ref.a, ref.e], rtol=1e-8, atol=0):
            raise ValueError("scenes must share a CRS and pixel resolution; co-register first")
        offset = [(t.c - ref.c) / ref.a, (t.f - ref.f) / ref.e]
        if not np.allclose(offset, np.round(offset), atol=1e-5, rtol=0):
            raise ValueError("scenes must share an aligned pixel grid; co-register first")
    return src.crs, t


def extract_chips(records, imagery_dir, chip=64):
    """Return (t1, t2, keep) with only full-window rows in t1/t2.

    `keep` maps the returned arrays back to the input records. A missing window
    at EITHER date drops both chips. Scene paths are relative to imagery_dir.
    """
    if type(chip) is not int or chip < 1:
        raise ValueError("chip must be a positive integer")
    n = len(records)
    t1 = np.empty((n, chip, chip, 3), np.uint8)
    t2 = np.empty_like(t1)
    keep = np.ones(n, dtype=bool)
    by_scene = defaultdict(list)
    for i, r in enumerate(records):
        for slot, key in enumerate(("t1", "t2")):
            by_scene[r[key]].append((i, slot))
    reference = None
    for scene, items in by_scene.items():
        with rasterio.open(Path(imagery_dir) / scene) as src:
            reference = validate_raster(src, reference)
            for i, slot in items:
                r = records[i]
                win = chip_window(src, r["x"], r["y"], chip)
                if win is None:
                    keep[i] = False
                    continue
                arr = src.read(window=win, boundless=False)
                (t1 if slot == 0 else t2)[i] = arr.transpose(1, 2, 0)
    return t1[keep], t2[keep], keep
