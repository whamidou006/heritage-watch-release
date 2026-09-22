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
    cells = (rng.random(n) * 4).astype(int)
    out = module["loio"](site, rng.normal(size=(n, 5)), meta, manifest, cells,
                         reps=2, n_train=24)
    assert out["n_train_cap"] == 24
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


def test_loio_excludes_test_cells_from_both_training_arms(site):
    """The spatial guarantee, asserted on the rows actually fitted.

    Without it the "seen" arm trains on points metres from its own test points,
    and the seen/unseen contrast measures proximity rather than date novelty.
    """
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "controls.py"))
    rng = np.random.default_rng(1)
    n = 120
    pairs = np.repeat(["20202021", "20212022", "20232024"], 40)
    meta = dict(pair=pairs, y=np.array(["A", "B"] * 60), fid=np.arange(n),
                x=rng.random(n), yy=rng.random(n))
    scenes = {"20202021": ("a", "b"), "20212022": ("b", "c"), "20232024": ("d", "e")}
    manifest = {"records": [dict(fid=i, pair=p, t1=scenes[p][0], t2=scenes[p][1])
                            for i, p in enumerate(pairs)]}
    cells = (rng.random(n) * 6).astype(int)

    seen_fits = []
    real_fit = module["default_classifier"]

    class Spy:
        def fit(self, X, y):
            seen_fits.append(len(X))
            self._c = real_fit().fit(X, y)
            return self
        def predict(self, X):
            return self._c.predict(X)

    module["default_classifier"] = Spy
    try:
        out = module["loio"](site, rng.normal(size=(n, 5)), meta, manifest, cells,
                             reps=2, n_train=30)
    finally:
        module["default_classifier"] = real_fit

    # both arms of every replicate fitted the same number of rows
    assert len(seen_fits) % 2 == 0
    assert all(a == b for a, b in zip(seen_fits[::2], seen_fits[1::2]))
    # and the recorded minimum training size is the truth, not the cap
    for r in out["per_interval"].values():
        assert r["n_train_min"] <= r["n_train_mean"] <= 30


def test_loio_reports_no_clean_subset_when_every_interval_chains(site):
    """A chained archive has no endpoint-clean interval; saying otherwise invents one."""
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "controls.py"))
    rng = np.random.default_rng(2)
    n = 120
    pairs = np.repeat(["20202021", "20212022", "20222023"], 40)
    meta = dict(pair=pairs, y=np.array(["A", "B"] * 60), fid=np.arange(n),
                x=rng.random(n), yy=rng.random(n))
    chain = {"20202021": ("a", "b"), "20212022": ("b", "c"), "20222023": ("c", "d")}
    manifest = {"records": [dict(fid=i, pair=p, t1=chain[p][0], t2=chain[p][1])
                            for i, p in enumerate(pairs)]}
    cells = (rng.random(n) * 6).astype(int)
    out = module["loio"](site, rng.normal(size=(n, 5)), meta, manifest, cells,
                         reps=2, n_train=30)
    assert all(r["shares_endpoint"] for r in out["per_interval"].values())
    assert out["clean_only_delta"] is None
