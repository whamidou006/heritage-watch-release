import csv

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from heritage_watch.config import SiteConfig


@pytest.fixture
def site(tmp_path):
    """Small real GeoTIFFs in pytest's project-local base directory."""
    images = tmp_path / "images"
    images.mkdir()
    for name, left, value in (("a_2020-01-15.tif", 0, 20),
                              ("a_2021-01-15.tif", 8, 90),
                              ("a_2020-12-01.tif", 0, 40)):
        with rasterio.open(images / name, "w", driver="GTiff", width=32, height=32,
                           count=3, dtype="uint8", crs="EPSG:3857",
                           transform=from_origin(left, 32, 1, 1)) as ds:
            ds.write(np.full((3, 32, 32), value, np.uint8))
    csv_path = tmp_path / "points.csv"
    rows = [
        (1, "A", "16,0", "16,0", "change_202001_and_202101", "20202021"),
        (2, "B", "18", "17", "change_202001_and_202101", "20202021"),
        (3, "A", "4", "16", "change_202001_and_202101", "20202021"),
        (4, "A", "8", "16", "change_202001_and_202101", "20202021"),
        (5, "Other", "16", "16", "change_202001_and_202101", "20202021"),
        (6, "A", "bad", "16", "change_202001_and_202101", "20202021"),
        (7, "A", "16", "16", "change_201801_and_202101", "20182021"),
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fid", "category", "X ", "Y", "layer", "year"])
        writer.writerows(rows)
    return SiteConfig(images, csv_path, r"a_(\d{4})-(\d{2})-(\d{2})\.tif$",
                      ["A", "B"], ["Other"], chip_size=8, folds=2, replicates=2, blocks=2)


@pytest.fixture
def feature_data():
    rng = np.random.default_rng(52)
    n = 32
    blocks = {"dinov2": {k: rng.normal(size=(n, 768)).astype("float32") for k in ("f1", "f2")},
              "satlas": {k: rng.normal(size=(n, 1920)).astype("float32") for k in ("f1", "f2", "fmi")}}
    meta = dict(y=np.array(["A", "B"] * (n // 2)), fid=np.arange(n),
                pair=np.array(["20202021"] * n), x=rng.random(n), yy=rng.random(n))
    return blocks, meta
