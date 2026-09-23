import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from heritage_watch.protocol import (Result, compare, default_classifier, evaluate,
                                     evaluate_interval, spatial_blocks)


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


def test_compare_uses_sample_standard_error(capsys):
    """2*SE must use the sample SD (ddof=1).

    With the population SD a two-replicate difference of [0.0063, 0.0357] gets
    2*SE = 0.0208 and is declared resolved; the sample SD gives 0.0294 and it is
    not. Borderline verdicts flip on this choice, so it is pinned here.
    """
    a = Result(macro_f1=0.021, sd=0.0, per_replicate=[0.0063, 0.0357], pairing_id="p")
    b = Result(macro_f1=0.0, sd=0.0, per_replicate=[0.0, 0.0], pairing_id="p")
    compare(a, b, "a", "b", min_effect=0.02)
    out = capsys.readouterr().out
    assert "resolved:" not in out


def test_per_class_precision_recall_average_over_all_replicates():
    """P, R and F1 in the per-class table must describe the same experiment."""
    rng = np.random.default_rng(0)
    n = 160
    y = np.array(["A", "B", "C", "D"] * (n // 4))
    X = rng.normal(size=(n, 6)) + (y == "A")[:, None]
    meta = {"x": rng.random(n), "yy": rng.random(n)}
    res = evaluate(X, y, meta, replicates=3, classes=["A", "B", "C", "D"])
    assert len(res.per_replicate) == 3
    # replicate-0-only P/R would make these disagree with the averaged F1 harmonically
    for c in res.per_class_f1:
        p, r = res.per_class_pr[c]
        assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0
    assert res.sd == pytest.approx(np.std(res.per_replicate, ddof=1))


def _interval_data(seed=3, n=160):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = np.array(["A", "B"] * (n // 2))
    X[y == "B", 0] += 3.0
    meta = dict(x=rng.random(n), yy=rng.random(n), fid=np.arange(n),
                pair=np.array([f"p{i % 4}" for i in range(n)]))
    return X, y, meta


def test_evaluate_interval_needs_pair_metadata():
    X, y, meta = _interval_data()
    with pytest.raises(ValueError, match="pair"):
        evaluate_interval(X, y, {k: v for k, v in meta.items() if k != "pair"})


def test_evaluate_interval_never_trains_on_the_held_out_interval():
    """Every training row must come from a different interval AND a different cell."""
    X, y, meta = _interval_data()
    seen = []

    def factory():
        clf = default_classifier()
        original = clf.fit
        def fit(a, b):
            seen.append(len(a))
            return original(a, b)
        clf.fit = fit
        return clf

    r = evaluate_interval(X, y, meta, factory, classes=["A", "B"], blocks=3,
                          replicates=2, n_train=60)
    assert len(r.per_replicate) == 2 and 0 <= r.macro_f1 <= 1
    assert seen and max(seen) <= 60


def test_evaluate_interval_rejects_a_single_interval():
    X, y, meta = _interval_data()
    meta["pair"] = np.array(["only"] * len(y))
    with pytest.raises(ValueError, match="two acquisition intervals"):
        evaluate_interval(X, y, meta, classes=["A", "B"])


def test_evaluate_interval_is_harder_than_the_spatial_score_when_dates_carry_the_label():
    """A date shortcut must inflate evaluate() and collapse under evaluate_interval()."""
    rng = np.random.default_rng(11)
    n = 200
    pair = np.array([f"p{i % 4}" for i in range(n)])
    y = np.where(np.isin(pair, ["p0", "p1"]), "A", "B")     # label is a function of date
    X = np.zeros((n, 4))
    for i, p in enumerate(sorted(set(pair))):               # features encode ONLY the date
        X[pair == p, i] = 1.0
    X += rng.normal(scale=0.01, size=X.shape)
    meta = dict(x=rng.random(n), yy=rng.random(n), fid=np.arange(n), pair=pair)
    spatial = evaluate(X, y, meta, classes=["A", "B"], blocks=3, folds=2, replicates=2)
    interval = evaluate_interval(X, y, meta, classes=["A", "B"], blocks=3, replicates=2)
    assert spatial.macro_f1 > 0.9
    assert interval.macro_f1 < 0.6


def test_interval_results_compare_and_refuse_mismatched_protocols(capsys):
    """compare() must work on two interval arms and refuse anything not paired."""
    X, y, meta = _interval_data()
    kw = dict(classes=["A", "B"], blocks=3, replicates=2, n_train=60)
    a = evaluate_interval(X, y, meta, **kw)
    b = evaluate_interval(X + 0.5, y, meta, **kw)
    compare(a, b, "a", "b")                       # same rows, same protocol: allowed
    assert "a - b" in capsys.readouterr().out

    for changed in (dict(seed=99), dict(blocks=4), dict(n_train=40)):
        other = evaluate_interval(X, y, meta, **{**kw, **changed})
        with pytest.raises(ValueError, match="not paired"):
            compare(a, other)

    meta2 = {**meta, "pair": np.where(meta["pair"] == "p0", "pX", meta["pair"])}
    with pytest.raises(ValueError, match="not paired"):
        compare(a, evaluate_interval(X, y, meta2, **kw))


def test_interval_and_spatial_scores_are_never_compared():
    """They answer different questions; mixing them would be a false claim."""
    X, y, meta = _interval_data()
    spatial = evaluate(X, y, meta, classes=["A", "B"], blocks=3, folds=2, replicates=2)
    interval = evaluate_interval(X, y, meta, classes=["A", "B"], blocks=3, replicates=2)
    with pytest.raises(ValueError, match="not paired"):
        compare(spatial, interval)


def test_interval_reports_population_n_and_held_out_range():
    X, y, meta = _interval_data()
    r = evaluate_interval(X, y, meta, classes=["A", "B"], blocks=3, replicates=2,
                          n_train=60)
    assert r.n == len(y)                          # population, matching evaluate()
    lo, hi = r.held_out_per_replicate
    assert 0 < lo <= hi < len(y)                  # the split, reported separately


def test_interval_fails_loudly_on_an_unusable_interval():
    """A silent skip would score a subset while reporting the full population."""
    X, y, meta = _interval_data()
    meta = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in meta.items()}
    meta["pair"] = np.asarray(meta["pair"]).astype(object)
    meta["pair"][0] = "singleton"
    with pytest.raises(ValueError, match="singleton"):
        evaluate_interval(X, y, meta, classes=["A", "B"], blocks=3, replicates=1)
