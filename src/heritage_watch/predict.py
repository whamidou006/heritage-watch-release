"""Apply a trusted model bundle to new-site points, refusing incompatible inputs."""
import csv
from pathlib import Path

import joblib
import numpy as np

from .encoders import embed_manifest
from .features import assemble, representation
from .manifest import build_manifest, print_manifest
from .protocol import _limit_threads


def validate_bundle(bundle, config):
    required = {"format_version", "classifier", "classes", "representation", "encoders",
                "chip_size", "feature_dimension", "training_fingerprint"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("invalid model bundle: missing required metadata")
    if bundle["format_version"] != 1:
        raise ValueError("unsupported model bundle format version")
    if bundle["classes"] != config.classes:
        raise ValueError("model/config class list mismatch (including order)")
    if bundle["chip_size"] != config.chip_size:
        raise ValueError("model/config chip size mismatch")
    rep = representation(bundle["representation"])
    if bundle["encoders"] != list(rep.encoders):
        raise ValueError("bundle encoder set does not match its representation")
    if bundle["feature_dimension"] != rep.dimension:
        raise ValueError("bundle feature dimension does not match its representation")
    clf = bundle["classifier"]
    if getattr(clf, "n_features_in_", None) != rep.dimension:
        raise ValueError("classifier feature dimension does not match bundle")
    if set(getattr(clf, "classes_", [])) != set(bundle["classes"]):
        raise ValueError("classifier classes do not match bundle")
    return rep


def score_features(bundle, config, blocks):
    validate_bundle(bundle, config)
    if set(blocks) != set(bundle["encoders"]):
        raise ValueError("prediction encoder set differs from bundle")
    X = assemble(blocks, bundle["representation"])
    if X.shape[1] != bundle["feature_dimension"]:
        raise ValueError("prediction feature dimension differs from bundle")
    with _limit_threads():
        return (bundle["classifier"].predict(X),
                bundle["classifier"].predict_proba(X))


def predict(model, config, out, device="cpu", batch_size=32):
    """Only load trusted joblib files: deserialization can execute arbitrary code."""
    bundle = joblib.load(model)
    validate_bundle(bundle, config)
    manifest = build_manifest(config)
    print_manifest(manifest)
    if not manifest["records"]:
        raise ValueError("new site has no usable full-window points")
    blocks = {enc: embed_manifest(config, manifest, enc, device=device, batch_size=batch_size)
              for enc in bundle["encoders"]}
    first = next(iter(blocks.values()))
    for b in blocks.values():
        if not np.array_equal(first["fid"], b["fid"]):
            raise ValueError("prediction sample order differs between encoders")
    pred, prob = score_features(bundle, config, blocks)
    records = {r["fid"]: r for r in manifest["records"]}
    classes = list(bundle["classifier"].classes_)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["fid", "pair", "x", "y", "t1_date", "t2_date", "truth", "prediction"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + [f"p:{c}" for c in classes])
        writer.writeheader()
        for i, fid in enumerate(first["fid"]):
            r = records[int(fid)]
            row = {k: r[k] for k in fields if k in r}
            row.update(truth=r["category"], prediction=pred[i])
            row.update({f"p:{c}": float(prob[i, j]) for j, c in enumerate(classes)})
            writer.writerow(row)
    print(f"wrote {len(pred)} predictions to {path}; probabilities are not calibrated")
    return pred
