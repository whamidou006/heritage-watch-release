from dataclasses import replace

import pytest
import yaml

from heritage_watch.config import load_config


def test_exact_headers_and_relative_paths(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump(dict(imagery_dir="images", annotation_csv="points.csv",
                                     scene_date_regex=r"(\d{4})-(\d{2})-(\d{2})",
                                     classes=["A", "B"])))
    cfg = load_config(p)
    assert cfg.imagery_dir == tmp_path / "images"
    assert cfg.columns["x"] == "X "
    assert (cfg.chip_size, cfg.blocks, cfg.folds, cfg.replicates) == (64, 6, 5, 8)


@pytest.mark.parametrize("field,value", [
    ("pairing", "guess"), ("folds", 1), ("blocks", 0), ("chip_size", True),
    ("replicates", 1), ("tolerance_days", -1), ("min_effect", 0),
    ("min_effect", float("nan")), ("classes", ["A", "A"]),
    ("drop_categories", ["A"]), ("scene_date_regex", "("),
    ("scene_date_regex", "(one)"), ("columns", {"x": "X"})])
def test_validation(site, field, value):
    with pytest.raises(ValueError):
        replace(site, **{field: value})


def test_unknown_and_missing_config(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("typo: 1")
    with pytest.raises(ValueError, match="unknown"):
        load_config(p)
    p.write_text("{}")
    with pytest.raises(ValueError, match="invalid site config"):
        load_config(p)


def test_environment_path(tmp_path, monkeypatch):
    from heritage_watch.config import _expand
    monkeypatch.setenv("TEST_SITE_ROOT", str(tmp_path))
    assert _expand("${TEST_SITE_ROOT}/images") == str(tmp_path / "images")
    monkeypatch.delenv("MISSING_SITE_VAR", raising=False)
    assert _expand("${MISSING_SITE_VAR:-data}") == "data"
    with pytest.raises(ValueError, match="MISSING_SITE_VAR"):
        _expand("${MISSING_SITE_VAR}")
