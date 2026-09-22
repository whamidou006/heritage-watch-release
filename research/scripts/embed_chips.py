#!/usr/bin/env python3
"""Extract bi-temporal chips at each labelled point and embed them with DINOv2.

Chips are cut from the T1 and T2 TM scenes at the same geographic location.
All 21 TM scenes share an identical grid (same CRS, bounds and size), so the
two chips are pixel-aligned by construction - no co-registration needed.

Outputs cache/features.npz with X_t1, X_t2 (N, 768) plus label metadata.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
import timm
from rasterio.windows import Window

CACHE = Path(__file__).resolve().parents[1] / "cache"


def extract_chips(records: list[dict], chip: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        with rasterio.open(scene) as src:
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


@torch.no_grad()
def embed(chips: np.ndarray, model, size: int, device: str, bs: int) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(chips), bs):
        b = torch.from_numpy(chips[i:i + bs]).to(device)
        b = b.permute(0, 3, 1, 2).float().div_(255.0)
        b = torch.nn.functional.interpolate(b, size=(size, size),
                                            mode="bicubic", align_corners=False)
        b = (b - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            f = model(b)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--out", default="features.npz")
    ap.add_argument("--chip", type=int, default=128, help="chip size in pixels (~0.25 m/px)")
    ap.add_argument("--size", type=int, default=224, help="model input size (multiple of 14)")
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    man = json.loads((CACHE / args.manifest).read_text())
    recs = man["records"]
    print(f"records: {len(recs)}  chip={args.chip}px (~{args.chip * 0.25:.0f} m)")

    t1, t2, ok = extract_chips(recs, args.chip)
    print(f"chips extracted; {int((~ok).sum())} dropped for falling off the raster edge")
    recs = [r for r, k in zip(recs, ok) if k]
    t1, t2 = t1[ok], t2[ok]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(args.model, pretrained=True, num_classes=0,
                              img_size=args.size).to(device).eval()
    print(f"{args.model} on {device}, embed_dim={model.num_features}")

    f1 = embed(t1, model, args.size, device, args.batch)
    f2 = embed(t2, model, args.size, device, args.batch)
    print("features", f1.shape, f2.shape)

    np.savez_compressed(
        CACHE / args.out,
        f1=f1, f2=f2,
        y=np.array([r["category"] for r in recs]),
        pair=np.array([r["pair"] for r in recs]),
        x=np.array([r["x"] for r in recs]),
        yy=np.array([r["y"] for r in recs]),
        fid=np.array([r["fid"] for r in recs]),
    )
    print(f"saved {CACHE / args.out}")


if __name__ == "__main__":
    main()
