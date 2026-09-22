from dataclasses import replace
import datetime as dt

import numpy as np
import pytest

from heritage_watch.chips import extract_chips
from heritage_watch.manifest import all_scenes, build_manifest, resolve


def test_pair_intersection_ledger_and_chip_edges(site):
    man = build_manifest(site)
    assert [r["fid"] for r in man["records"]] == [1, 2]
    assert man["drop_ledger"] == {
        "outside_footprint": 1, "chip_edge": 1, "class_excluded:Other": 1,
        "bad_coord": 1, "no_scene_near_stated_month": 1}
    assert sum(man["drop_ledger"].values()) + len(man["records"]) == man["raw_count"]
    assert man["crs"] == "EPSG:3857"
    r = man["records"][0]
    assert r["t1_date"] == "2020-01-15"
    assert not r["t1"].startswith("/")
    a, b, keep = extract_chips(man["records"], site.imagery_dir, site.chip_size)
    assert a.shape == b.shape == (2, 8, 8, 3)
    assert keep.all() and np.all(a == 20) and np.all(b == 90)


def test_extractor_rejects_both_dates_not_padding(site):
    r = build_manifest(site)["records"][0]
    # Inside the first raster but at the second raster's left edge.
    bad = dict(r, x=8)
    a, b, keep = extract_chips([r, bad], site.imagery_dir, 8)
    assert keep.tolist() == [True, False]
    assert len(a) == len(b) == 1
    # Right edge is valid only when the entire before window fits.
    _, _, keep = extract_chips([dict(r, x=28), dict(r, x=29)], site.imagery_dir, 8)
    assert keep.tolist() == [True, False]


def test_month_vs_year_and_fallback(site):
    scenes = all_scenes(site)
    by = {}
    for date, path in scenes:
        by.setdefault(str(date.year), []).append((date, path))
    month = resolve("change_202001_and_202101", "20202021", scenes, by)
    year = resolve("change_202001_and_202101", "20202021", scenes, by, mode="year")
    assert month[0][0] == dt.date(2020, 1, 15)
    assert year[0][0] == dt.date(2020, 12, 1)
    assert resolve("change_2020_and_2021", "20202021", scenes, by) == year
    assert resolve("change_202013_and_202101", "20202021", scenes, by)[2] == "invalid_layer_month"
    assert resolve("change_202001_and_202001", "20202020", scenes, by)[2] == "degenerate_pair"


def test_header_error_is_explicit(site):
    site.annotation_csv.write_text(site.annotation_csv.read_text().replace("X ", "X"))
    with pytest.raises(ValueError, match="exact CSV headers"):
        build_manifest(site)


def test_raster_alignment_refused(site):
    import rasterio
    from rasterio.transform import from_origin
    p = site.imagery_dir / "a_2021-01-15.tif"
    with rasterio.open(p, "r+") as src:
        src.transform = from_origin(8.25, 32, 1, 1)
    with pytest.raises(ValueError, match="aligned pixel grid"):
        build_manifest(site)


def test_unexpected_category_counted(site):
    config = replace(site, drop_categories=[])
    assert build_manifest(config)["drop_ledger"]["class_not_kept:Other"] == 1
