import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from heritage_watch.protocol import Result, compare, default_classifier, evaluate, spatial_blocks


def test_jitter_changes_groups_and_fold_disjointness():
    rng = np.random.default_rng(14)
    x, y = rng.random(200), rng.random(200)
    a = spatial_blocks(x, y, 6, 0.1, 0.1)
    b = spatial_blocks(x, y, 6, 0.7, 0.7)
    assert not np.array_equal(a, b)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(x, np.arange(200) % 4, a):
        assert not set(a[tr]) & set(a[te])


def test_fold_only_scaling_and_pca():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(80, 10))
    X[:, 0] += np.arange(80) * 20
    y = np.array(["A", "B"] * 40)
    meta = dict(x=rng.random(80), yy=rng.random(80), fid=np.arange(80))
    fitted = []
    def factory():
        clf = default_classifier(pca_components=3)
        original = clf.fit
        def fit(a, b):
            original(a, b)
            assert len(a) < len(X)
            assert np.allclose(clf[0].mean_, a.mean(axis=0))
            assert not np.allclose(clf[0].mean_, X.mean(axis=0))
            assert clf[1].n_samples_ == len(a)
            fitted.append(len(a))
            return clf
        clf.fit = fit
        return clf
    result = evaluate(X, y, meta, factory, classes=["A", "B"], folds=2, blocks=3, replicates=2)
    assert len(fitted) == 4 and len(result.per_replicate) == 2
    assert 0 <= result.macro_f1 <= 1


@pytest.mark.parametrize("deltas,verdict", [
    ([0.03] * 8, "resolved:"),
    ([-0.03] * 8, "resolved:"),
    ([0.0197] * 8, "consistent but SMALL"),
    ([0.03] * 7 + [-0.001], "suggestive"),
    ([0.0] + [0.03] * 7, "suggestive"),
    ([0.001] * 7 + [0.4], "WITHIN NOISE"),
])
def test_three_bars(deltas, verdict, capsys):
    a = Result(0, 0, (np.array(deltas) + 0.5).tolist(), pairing_id="same")
    b = Result(0, 0, [0.5] * 8, pairing_id="same")
    compare(a, b)
    assert verdict in capsys.readouterr().out


def test_pairing_metadata_required_to_match():
    a = Result(0, 0, [0.5, 0.6], pairing_id="one")
    b = Result(0, 0, [0.5, 0.6], pairing_id="two")
    with pytest.raises(ValueError, match="not paired"):
        compare(a, b)
    with pytest.raises(ValueError, match="not paired"):
        compare(Result(0, 0, [0.5, 0.6]), Result(0, 0, [0.5, 0.6]))


def test_repeatability_and_fingerprint(feature_data):
    blocks, meta = feature_data
    X = blocks["dinov2"]["f1"][:, :5]
    kwargs = dict(classes=["A", "B"], folds=2, replicates=2, blocks=2)
    a = evaluate(X, meta["y"], dict(meta), **kwargs)
    b = evaluate(X, meta["y"], dict(meta), **kwargs)
    assert a.per_replicate == b.per_replicate
    assert a.pairing_id == b.pairing_id
    c = evaluate(X, meta["y"], dict(meta), seed=200, **kwargs)
    assert a.pairing_id != c.pairing_id
