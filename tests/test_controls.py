from pathlib import Path
import runpy

import numpy as np


def test_loio_same_size_and_endpoint_metadata(site):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "controls.py"))
    rng = np.random.default_rng(8)
    n = 80
    pairs = np.repeat(["20202021", "20212022", "20232024", "20252026"], 20)
    meta = dict(pair=pairs, y=np.array(["A", "B"] * 40), fid=np.arange(n),
                x=rng.random(n), yy=rng.random(n))
    scenes = {"20202021": ("a", "b"), "20212022": ("b", "c"),
              "20232024": ("d", "e"), "20252026": ("f", "g")}
    manifest = {"records": [dict(fid=i, pair=p, t1=scenes[p][0], t2=scenes[p][1])
                             for i, p in enumerate(pairs)]}
    out = module["loio"](site, rng.normal(size=(n, 5)), meta, manifest, reps=2, n_train=24)
    assert out["n_train"] == 24
    assert out["per_interval"]["20202021"]["shares_endpoint"]
    assert not out["per_interval"]["20232024"]["shares_endpoint"]
    assert out["clean_only_delta"] is not None
    assert all(len(r["seen"]) == len(r["unseen"]) == 2
               for r in out["per_interval"].values())


def test_date_control_uses_official_protocol(site, feature_data):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "controls.py"))
    _, meta = feature_data
    meta["pair"] = np.array(["20202021", "20212022"] * 16)
    result = module["date_only"](site, meta)
    assert result["n"] == 32
    assert result["macro_f1"] == 1
    assert len(result["per_replicate"]) == site.replicates
