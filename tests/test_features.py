from dataclasses import replace

import joblib
import numpy as np
import pytest

from heritage_watch.features import REPRESENTATIONS, assemble, fingerprint, load_features
from heritage_watch.predict import score_features, validate_bundle
from heritage_watch.train import train


def test_seven_representations_and_direction(feature_data):
    blocks, meta = feature_data
    for name, rep in REPRESENTATIONS.items():
        assert assemble(blocks, name).shape == (len(meta["y"]), rep.dimension)
    selected = assemble(blocks, "satlas_mi_si_diff")
    assert np.array_equal(selected[:, :1920], blocks["satlas"]["fmi"])
    assert np.array_equal(selected[:, 1920:], blocks["satlas"]["f2"] - blocks["satlas"]["f1"])
    merged = assemble(blocks, "merged")
    assert np.array_equal(merged[:, :2304], assemble(blocks, "dinov2_both_diff"))
    assert np.array_equal(merged[:, 2304:], assemble(blocks, "satlas_si_both_diff"))


def test_cache_join_and_provenance(tmp_path, feature_data, site):
    blocks, meta = feature_data
    paths = []
    for enc, b in blocks.items():
        p = tmp_path / f"{enc}.npz"
        np.savez(p, **b, **meta, encoder=enc, chip_size=8)
        paths.append(p)
    loaded, m = load_features(paths, site)
    assert np.array_equal(assemble(loaded, "merged"), assemble(blocks, "merged"))
    assert np.array_equal(m["fid"], meta["fid"])
    wrong = dict(meta, fid=meta["fid"][::-1])
    np.savez(paths[1], **blocks["satlas"], **wrong, encoder="satlas", chip_size=8)
    with pytest.raises(ValueError, match="fid"):
        load_features(paths, site)
    with pytest.raises(ValueError, match="chip size"):
        load_features(paths[:1], replace(site, chip_size=128))


def test_train_bundle_and_prediction_validation(tmp_path, site, feature_data):
    blocks, meta = feature_data
    path = tmp_path / "model.joblib"
    train(site, blocks, meta, "satlas_mi_si_diff", path)
    bundle = joblib.load(path)
    pred, prob = score_features(bundle, site, {"satlas": blocks["satlas"]})
    assert len(pred) == len(meta["y"])
    assert np.allclose(prob.sum(axis=1), 1)
    assert bundle["training_fingerprint"] == fingerprint(assemble(blocks, "satlas_mi_si_diff"), meta)
    for config, match in ((replace(site, classes=["A", "C"]), "class list"),
                           (replace(site, chip_size=16), "chip size")):
        with pytest.raises(ValueError, match=match):
            validate_bundle(bundle, config)
    for key, value, match in (("encoders", ["dinov2"], "encoder"),
                               ("feature_dimension", 1920, "dimension"),
                               ("representation", "unknown", "representation"),
                               ("representation", "merged", "encoder")):
        with pytest.raises(ValueError, match=match):
            validate_bundle(dict(bundle, **{key: value}), site)
    with pytest.raises(ValueError, match="encoder set"):
        score_features(bundle, site, blocks)
    with pytest.raises(ValueError, match="finite"):
        assemble({"satlas": {"fmi": np.ones((1, 1920)), "f1": np.ones((1, 1920)),
                            "f2": np.full((1, 1920), np.nan)}}, "satlas_mi_si_diff")


def test_predict_workflow_uses_bundle_encoders(tmp_path, site, feature_data, monkeypatch):
    """Exercise real raster/manifest + model persistence, replacing costly inference only."""
    import heritage_watch.predict as module
    blocks, meta = feature_data
    model = tmp_path / "model.joblib"
    train(site, blocks, meta, "dinov2_post", model)
    calls = []
    def embed(config, manifest, encoder, **kwargs):
        from heritage_watch.chips import extract_chips
        t1, t2, keep = extract_chips(manifest["records"], config.imagery_dir, config.chip_size)
        assert t1.shape == t2.shape == (2, 8, 8, 3) and keep.all()
        calls.append(encoder)
        return {**{k: v[:2] for k, v in blocks[encoder].items()},
                "fid": np.array([r["fid"] for r in manifest["records"]])}
    monkeypatch.setattr(module, "embed_manifest", embed)
    out = tmp_path / "predictions.csv"
    module.predict(model, site, out)
    assert calls == ["dinov2"]
    assert len(out.read_text().splitlines()) == 3
    assert "p:A" in out.read_text()
