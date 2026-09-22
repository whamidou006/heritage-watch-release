"""Named frozen feature blocks and the seven published representations."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Representation:
    title: str
    encoders: tuple[str, ...]
    dimension: int
    terms: tuple[str, ...]


REPRESENTATIONS = {
    "dinov2_post": Representation("DINOv2, post-event only", ("dinov2",), 768, ("dinov2.f2",)),
    "dinov2_both_diff": Representation("DINOv2, both + diff", ("dinov2",), 2304,
                                      ("dinov2.f1", "dinov2.f2", "dinov2.diff")),
    "satlas_si_post": Representation("Satlas-SI, post-event only", ("satlas",), 1920, ("satlas.f2",)),
    "satlas_si_both_diff": Representation("Satlas-SI, both + diff", ("satlas",), 5760,
                                         ("satlas.f1", "satlas.f2", "satlas.diff")),
    "satlas_mi": Representation("Satlas-MI (order-invariant)", ("satlas",), 1920, ("satlas.fmi",)),
    "satlas_mi_si_diff": Representation("Satlas-MI + SI diff (selected)", ("satlas",), 3840,
                                       ("satlas.fmi", "satlas.diff")),
    "merged": Representation("Merged DINOv2 + Satlas", ("dinov2", "satlas"), 8064,
                              ("dinov2.f1", "dinov2.f2", "dinov2.diff",
                               "satlas.f1", "satlas.f2", "satlas.diff")),
}
SELECTED = "satlas_mi_si_diff"
META_KEYS = ("y", "pair", "x", "yy", "fid")


def representation(name):
    try:
        return REPRESENTATIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown representation {name!r}; choose {', '.join(REPRESENTATIONS)}") from exc


def assemble(blocks, name):
    rep = representation(name)
    arrays = []
    for term in rep.terms:
        enc, key = term.split(".")
        try:
            block = blocks[enc]
            a = block["f2"] - block["f1"] if key == "diff" else block[key]
        except KeyError as exc:
            raise ValueError(f"representation {name} requires block {term}") from exc
        dim = 768 if enc == "dinov2" else 1920
        if a.ndim != 2 or a.shape[1] != dim or not np.isfinite(a).all():
            raise ValueError(f"{term} must be a finite (n, {dim}) array")
        arrays.append(a)
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("feature blocks have different row counts")
    return np.hstack(arrays)


def load_features(paths, config=None):
    """Load numeric NPZ caches, rejecting misaligned populations, not sorting silently.

    Legacy caches identify the encoder by f1 width. New caches additionally record
    encoder and chip size. Legacy chip size is unknown and is explicitly warned.
    """
    import warnings
    blocks, meta = {}, None
    for path in paths:
        with np.load(Path(path), allow_pickle=False) as loaded:
            d = {key: loaded[key] for key in loaded.files}
        if any(key not in d for key in (*META_KEYS, "f1", "f2")):
            raise ValueError(f"{path}: missing feature or metadata keys")
        enc = {768: "dinov2", 1920: "satlas"}.get(d["f1"].shape[-1])
        if enc is None or enc in blocks:
            raise ValueError(f"{path}: unknown or duplicate encoder")
        if "encoder" in d and str(d["encoder"]) != enc:
            raise ValueError(f"{path}: encoder metadata disagrees with feature width")
        if "chip_size" not in d:
            warnings.warn(f"{Path(path).name}: legacy cache has no chip-size provenance; "
                          "caller must verify it was extracted at the configured size", stacklevel=2)
        elif config is not None and int(d["chip_size"]) != config.chip_size:
            raise ValueError(f"{path}: chip size differs from config")
        n = len(d["fid"])
        if not n or len(set(d["fid"].tolist())) != n:
            raise ValueError(f"{path}: empty cache or duplicate fid")
        for key in META_KEYS:
            if d[key].shape != (n,):
                raise ValueError(f"{path}: {key} must have one value per sample")
        for key in ("f1", "f2", "fmi"):
            if key in d and (d[key].shape != (n, 768 if enc == "dinov2" else 1920)
                             or not np.isfinite(d[key]).all()):
                raise ValueError(f"{path}: invalid {key} shape or nonfinite features")
        if not np.isfinite(d["x"]).all() or not np.isfinite(d["yy"]).all():
            raise ValueError(f"{path}: nonfinite coordinates")
        if meta is None:
            meta = {key: d[key] for key in META_KEYS}
        else:
            for key in META_KEYS:
                if not np.array_equal(meta[key], d[key]):
                    raise ValueError(f"sample order/metadata differs between caches: {key}")
        if config is not None and not set(d["y"]).issubset(config.classes):
            raise ValueError(f"{path}: labels are not in configured classes")
        blocks[enc] = d
    if meta is None:
        raise ValueError("provide at least one feature cache")
    return blocks, meta


def fingerprint(X, meta):
    """SHA-256 over ordered metadata and actual representation values."""
    h = hashlib.sha256()
    h.update(json.dumps({k: np.asarray(meta[k]).tolist() for k in META_KEYS},
                        sort_keys=True, separators=(",", ":")).encode())
    h.update(np.ascontiguousarray(X, dtype="<f8").tobytes())
    return h.hexdigest()
