#!/usr/bin/env python3
"""Embed Herat bi-temporal chips with SatlasPretrain Aerial (Swin-v2-B).

Two variants, because they differ in a way that matters for this taxonomy:

  SI  - single-image encoder run separately on T1 and T2, giving [e1, e2, e2-e1].
        Order-sensitive, and directly comparable to the DINOv2 baseline.
  MI  - native multi-image encoder, both dates stacked on the channel axis.
        Fuses time by max-pooling, which is *exactly* order-invariant
        (verified: swapping T1/T2 changes the output by 0.0), so it cannot
        represent the direction of a change.

SatlasPretrain expects RGB scaled to [0,1] with no ImageNet mean/std, and is
pretrained at 0.5-2 m/px, so chips are fed at their native size (--chip, default
128) rather than being resized up to a ViT grid.  Note the contrast with
embed_chips.py, which interpolates every chip to a fixed model input: here the
chip size changes both the ground extent and the input resolution, there it
changes only the ground extent.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import satlaspretrain_models
import torch
from rasterio.windows import Window

CACHE = Path(__file__).resolve().parents[1] / "cache"


def extract_chips(records, chip):
    half = chip // 2
    n = len(records)
    t1 = np.zeros((n, chip, chip, 3), np.uint8)
    t2 = np.zeros((n, chip, chip, 3), np.uint8)
    ok = np.ones(n, bool)
    by_scene = defaultdict(list)
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
                if c0 < 0 or r0 < 0 or c0 + chip > W or r0 + chip > H:
                    ok[i] = False
                    continue
                arr = src.read(window=Window(c0, r0, chip, chip))
                (t1 if slot == 0 else t2)[i] = np.transpose(arr, (1, 2, 0))
    return t1, t2, ok


def pool(feats) -> torch.Tensor:
    """Global-average-pool every FPN level and concatenate -> 1920-d."""
    return torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1)


@torch.no_grad()
def embed_si(chips, model, device, bs):
    out = []
    for i in range(0, len(chips), bs):
        b = torch.from_numpy(chips[i:i + bs]).to(device)
        b = b.permute(0, 3, 1, 2).float().div_(255.0)
        out.append(pool(model(b)).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_mi(c1, c2, model, device, bs):
    out = []
    for i in range(0, len(c1), bs):
        a = torch.from_numpy(c1[i:i + bs]).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        b = torch.from_numpy(c2[i:i + bs]).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        out.append(pool(model(torch.cat([a, b], dim=1))).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--out", default="features_satlas.npz")
    ap.add_argument("--chip", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    man = json.loads((CACHE / args.manifest).read_text())
    recs = man["records"]
    t1, t2, ok = extract_chips(recs, args.chip)
    recs = [r for r, k in zip(recs, ok) if k]
    t1, t2 = t1[ok], t2[ok]
    print(f"chips {t1.shape}, {int((~ok).sum())} dropped at raster edge")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = satlaspretrain_models.Weights()

    si = w.get_pretrained_model("Aerial_SwinB_SI").to(device).eval()
    f1 = embed_si(t1, si, device, args.batch)
    f2 = embed_si(t2, si, device, args.batch)
    print("SI features", f1.shape)
    del si
    torch.cuda.empty_cache()

    mi = w.get_pretrained_model("Aerial_SwinB_MI").to(device).eval()
    fm = embed_mi(t1, t2, mi, device, args.batch)
    print("MI features", fm.shape)

    np.savez_compressed(
        CACHE / args.out,
        f1=f1, f2=f2, fmi=fm,
        y=np.array([r["category"] for r in recs]),
        pair=np.array([r["pair"] for r in recs]),
        x=np.array([r["x"] for r in recs]),
        yy=np.array([r["y"] for r in recs]),
        fid=np.array([r["fid"] for r in recs]),
    )
    print(f"saved {CACHE / args.out}")


if __name__ == "__main__":
    main()
