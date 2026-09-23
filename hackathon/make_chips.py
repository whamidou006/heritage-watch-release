#!/usr/bin/env python3
"""Cut raw bi-temporal pixel chips at every labelled point.

This pack ships PIXELS, not embeddings -- no precomputed feature caches are
included, on purpose, because they would lock you into the two encoders we
happened to pick. Use this script to cut the pixels, then build whatever
representation you like on top: frozen embeddings, a fine-tuned backbone, a
segmentation or change-detection network.

All 21 TM scenes share one grid (same CRS, bounds, size), so the T1 and T2
chips are pixel-aligned by construction -- no co-registration needed.

    python make_chips.py --images ../raw/dataset --chip 64 --out chips_64.npz

You normally do not need to run this yourself: baseline.py cuts the default
64 px chips automatically. Run it directly to try a different --chip size.

Output npz:
    t1   (N, chip, chip, 3) uint8   before-image chip
    t2   (N, chip, chip, 3) uint8   after-image chip
    y    (N,)  str    class label
    fid  (N,)  int64  stable join key -- always join on this, never row order
    pair (N,)  str    change pair, e.g. "20242025"
    x,yy (N,)  float64  lon/lat (EPSG:4326), needed for spatial blocking

Points whose window falls off the raster edge are dropped, not zero-padded:
at chip=64 that costs 2 of 884 labels, leaving 882. A 128 px window costs 5,
leaving 879 -- which is why n depends on the chip size.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def extract_chips(records: list[dict], images: Path, chip: int):
    """Cut a chip*chip window centred on each point from its T1 and T2 scene."""
    half = chip // 2
    n = len(records)
    t1 = np.zeros((n, chip, chip, 3), np.uint8)
    t2 = np.zeros((n, chip, chip, 3), np.uint8)
    ok = np.ones(n, bool)

    by_scene: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, r in enumerate(records):
        by_scene[r["t1"]].append((i, 0))
        by_scene[r["t2"]].append((i, 1))

    for scene, items in by_scene.items():
        path = images / scene
        if not path.exists():
            raise SystemExit(
                f"missing scene: {path}\n"
                "--images must point at the folder CONTAINING "
                "herat_site_sat_images/ (the unzipped dataset root)."
            )
        with rasterio.open(path) as src:
            H, W = src.height, src.width
            for i, slot in items:
                r = records[i]
                col, row = src.index(r["x"], r["y"])[::-1]
                c0, r0 = col - half, row - half
                # Reject points whose full window would fall off the raster,
                # rather than silently zero-padding them into the training set.
                if c0 < 0 or r0 < 0 or c0 + chip > W or r0 + chip > H:
                    ok[i] = False
                    continue
                arr = src.read(window=Window(c0, r0, chip, chip))  # (3,H,W)
                (t1 if slot == 0 else t2)[i] = np.transpose(arr, (1, 2, 0))
    return t1, t2, ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path,
                    help="unzipped dataset root (contains herat_site_sat_images/)")
    ap.add_argument("--manifest", default=Path(__file__).parent / "data" / "manifest.json",
                    type=Path)
    ap.add_argument("--chip", type=int, default=64,
                    help="chip size in pixels at ~0.25 m/px (64 px = 16 m, about "
                         "one building). Default 64; a real hyperparameter.")
    ap.add_argument("--out", default=Path("chips_64.npz"), type=Path)
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"manifest not found: {args.manifest}\n"
            "It ships with this pack at hackathon/data/manifest.json -- "
            "if it is missing, your clone is incomplete.")
    if not args.images.exists():
        raise SystemExit(
            f"imagery root not found: {args.images}\n"
            "Unzip the dataset first:  unzip dataset.zip -d raw\n"
            "then pass its 'dataset' directory, e.g. --images ../raw/dataset")

    man = json.loads(args.manifest.read_text())
    recs = man["records"]
    print(f"manifest: {len(recs)} labels   chip={args.chip}px (~{args.chip * 0.25:.0f} m)")

    t1, t2, ok = extract_chips(recs, args.images, args.chip)
    n_drop = int((~ok).sum())
    print(f"dropped {n_drop} label(s) whose window fell off the raster edge")

    keep = np.where(ok)[0]
    np.savez_compressed(
        args.out,
        t1=t1[keep], t2=t2[keep],
        y=np.array([recs[i]["category"] for i in keep]),
        fid=np.array([recs[i]["fid"] for i in keep], np.int64),
        pair=np.array([recs[i]["pair"] for i in keep]),
        x=np.array([recs[i]["x"] for i in keep]),
        yy=np.array([recs[i]["y"] for i in keep]),
    )
    mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}  ({len(keep)} chips, {mb:.0f} MB)")
    print("\nnext: build any feature matrix X (row order = this file's fid order),")
    print("      then score it with BOTH official protocol scores:")
    print("        from heritage_watch.protocol import evaluate, evaluate_interval, report")
    print("        report(evaluate(X, y, meta))            # another building")
    print("        report(evaluate_interval(X, y, meta))   # an unseen acquisition")


if __name__ == "__main__":
    main()
