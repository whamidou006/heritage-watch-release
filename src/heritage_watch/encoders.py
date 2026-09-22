"""Frozen encoders with the exact preprocessing used for the published caches.

Imports and weight downloads are lazy: cache evaluation never loads torch,
timm, Satlas, or a GPU. Explicit device selection defaults to CPU.
"""
from pathlib import Path

import numpy as np

from .chips import extract_chips


def _tensor(chips, device):
    import torch
    return torch.from_numpy(chips).to(device).permute(0, 3, 1, 2).float().div_(255)


def dinov2(t1, t2, device="cpu", batch_size=32):
    import torch
    import timm
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0, img_size=224).to(device).eval()
    model.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    blocks = {}
    with torch.inference_mode():
        for key, chips in (("f1", t1), ("f2", t2)):
            out = []
            for i in range(0, len(chips), batch_size):
                b = _tensor(chips[i:i + batch_size], device)
                b = torch.nn.functional.interpolate(b, size=(224, 224), mode="bicubic",
                                                     align_corners=False)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=str(device).startswith("cuda")):
                    f = model((b - mean) / std)
                out.append(f.float().cpu().numpy())
            blocks[key] = np.concatenate(out)
    return blocks


def satlas(t1, t2, device="cpu", batch_size=32):
    import torch
    import satlaspretrain_models
    weights = satlaspretrain_models.Weights()

    def run(model, a, b=None):
        out = []
        for i in range(0, len(a), batch_size):
            x = _tensor(a[i:i + batch_size], device)
            if b is not None:
                x = torch.cat([x, _tensor(b[i:i + batch_size], device)], dim=1)
            feats = model(x)
            out.append(torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1).float().cpu().numpy())
        return np.concatenate(out)

    with torch.inference_mode():
        si = weights.get_pretrained_model("Aerial_SwinB_SI").to(device).eval()
        si.requires_grad_(False)
        blocks = {"f1": run(si, t1), "f2": run(si, t2)}
        del si
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        mi = weights.get_pretrained_model("Aerial_SwinB_MI").to(device).eval()
        mi.requires_grad_(False)
        blocks["fmi"] = run(mi, t1, t2)
    return blocks


ENCODERS = {"dinov2": dinov2, "satlas": satlas}


def embed_manifest(config, manifest, encoder, out=None, device="cpu", batch_size=32):
    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder: {encoder}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if manifest["chip_size"] != config.chip_size or manifest["classes"] != config.classes:
        raise ValueError("manifest chip size/classes differ from configuration")
    records = manifest["records"]
    if not records:
        raise ValueError("no usable records to embed")
    t1, t2, keep = extract_chips(records, config.imagery_dir, config.chip_size)
    records = [r for r, ok in zip(records, keep) if ok]
    print(f"chips: {len(records)}; rejected at edge: {int((~keep).sum())}", flush=True)
    if not records:
        raise ValueError("no full-window chips remain")
    blocks = ENCODERS[encoder](t1, t2, device=device, batch_size=batch_size)
    blocks.update(y=np.array([r["category"] for r in records]),
                  pair=np.array([r["pair"] for r in records]),
                  fid=np.array([r["fid"] for r in records]),
                  x=np.array([r["x"] for r in records]),
                  yy=np.array([r["y"] for r in records]),
                  encoder=np.array(encoder), chip_size=np.array(config.chip_size))
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as stream:
            np.savez_compressed(stream, **blocks)
    return blocks
