"""Official evaluation harness for the Herat Heritage Watch challenge.

Import this instead of writing your own cross-validation. It enforces the two
rules that make scores comparable between teams:

  1. SPATIAL BLOCKING. Labels are clustered by construction site, so ordinary
     random CV puts neighbouring pixels in both train and test and inflates
     macro-F1 by 2.5-6.3 points. Folds are grouped by cells of a grid laid over
     the site footprint.

  2. PAIRED, REPLICATED SCORING. The grid origin is jittered per replicate, so
     every replicate is a genuinely different partition. A single run has a
     standard deviation of about 0.016 macro-F1, so differences smaller than
     ~0.03 CANNOT be judged by comparing two means -- compare per replicate
     instead, using the same grid for both systems (see compare()).

Typical use:

    from heritage_watch.protocol import load_chips, evaluate, report

    t1, t2, y, meta = load_chips("chips_128.npz")
    X = my_model_embeddings(t1, t2)   # anything you like
    report(evaluate(X, y, meta))      # your features, our protocol
"""
from __future__ import annotations

import os

# Cap BLAS threads BEFORE numpy/sklearn import. These problems are small and
# high-dimensional, so the linear-algebra backend happily spawns one thread per
# core and then spends all its time in contention. Uncapped on a 96-core shared
# machine this file took 40 minutes; capped at 4 threads it takes about 3.
# Override with HERAT_THREADS if you know what you are doing.
_THREADS = os.environ.get("HERITAGE_THREADS", os.environ.get("HERAT_THREADS", "4"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["Destruction", "New Construction", "Solar Panel", "Temporary Structure"]
FOLDS = 5
BLOCKS = 6
REPLICATES = 8
_SEED = 100  # fixed so every team gets identical grids -- do not change


@dataclass
class Result:
    macro_f1: float
    sd: float
    per_replicate: list[float]
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_pr: dict[str, tuple[float, float]] = field(default_factory=dict)
    confusion: np.ndarray | None = None
    n: int = 0
    pairing_id: str | None = None


def load_chips(path: str):
    """Load the output of make_chips.py. Returns (t1, t2, y, meta).

    t1, t2 are float32 in [0, 1] with shape (n, chip, chip, 3) -- the before and
    after image at each labelled point, pixel-aligned by construction. meta
    carries the coordinates that evaluate() needs for spatial blocking.

    Build whatever feature matrix X you like from t1/t2 -- embeddings from any
    pretrained model, a fine-tuned backbone, hand-made statistics -- then call
    evaluate(X, y, meta). The protocol does not care where X came from.
    """
    d = np.load(path, allow_pickle=True)
    meta = {"x": d["x"], "yy": d["yy"], "fid": d["fid"]}
    t1 = d["t1"].astype(np.float32) / 255.0
    t2 = d["t2"].astype(np.float32) / 255.0
    return t1, t2, d["y"].astype(str), meta


def spatial_blocks(x, y, n=BLOCKS, ox=0.0, oy=0.0):
    """Grid-cell id per label. (ox, oy) shift the origin by a fraction of a cell."""
    def axis(v, o):
        span = np.ptp(v) + 1e-12
        return np.floor((v - v.min()) / span * n + o).astype(int) % (n + 1)
    return axis(x, ox) * (n + 1) + axis(y, oy)


def _fit_predict(X, y_idx, groups, seed, clf_factory, folds=FOLDS):
    oof = np.empty(len(y_idx), np.int64)
    if len(np.unique(groups)) < folds:
        raise ValueError(f"need at least {folds} populated spatial groups")
    sp = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    for tr, te in sp.split(X, y_idx, groups):
        clf = clf_factory()
        clf.fit(X[tr], y_idx[tr])                 # NB: fitted inside the fold
        oof[te] = clf.predict(X[te])
    return oof


def default_classifier(pca_components=None):
    """Standardise, optional in-fold PCA, then balanced multinomial regression.

    Tight convergence is important at d >> n: sklearn's default tol=1e-4
    stopped early on the reference cache (0.7215 versus the published 0.7259).
    tol=1e-9 reproduces it within 0.002 without changing the folds or objective.
    The original measurement solver used float32 torch LBFGS and sample standard
    deviations; this reference uses sklearn float64 and population deviations,
    so bit-for-bit identical scores are not promised.
    """
    steps = [StandardScaler()]
    if pca_components is not None:
        from sklearn.decomposition import PCA
        steps.append(PCA(n_components=pca_components, svd_solver="full"))
    steps.append(LogisticRegression(max_iter=3000, tol=1e-9, C=1.0, class_weight="balanced"))
    return make_pipeline(*steps)


def _limit_threads():
    """Cap BLAS threads at call time.

    The environment variables at the top of this module only work if nothing
    imported numpy first, which is fragile -- `import numpy` above `import
    herat_eval` is enough to defeat them, and that costs ~20x wall time on a
    many-core machine. threadpoolctl reaches into the already-loaded BLAS, so
    it works regardless of import order. It ships with scikit-learn.
    """
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=int(_THREADS))
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def evaluate(X, y, meta, clf_factory=default_classifier, replicates=REPLICATES,
             seed=_SEED, *, blocks=BLOCKS, folds=FOLDS, classes=None):
    """Run the official protocol. Returns a Result.

    X    : (n, d) features you built
    y    : (n,)   string labels
    meta : dict from load_chips (must contain 'x' and 'yy')
    seed : grid-jitter seed. Leave it alone. Development uses the public seed
           so every team sees identical partitions; final scoring is run by the
           organisers under an UNPUBLISHED seed, so tuning against these exact
           8 partitions buys you nothing.
    """
    with _limit_threads():
        return _evaluate(X, y, meta, clf_factory, replicates, seed, blocks, folds, classes)


def _evaluate(X, y, meta, clf_factory, replicates, seed=_SEED,
              blocks=BLOCKS, folds=FOLDS, classes=None):
    X = np.asarray(X, dtype=np.float64)
    labs = sorted(classes if classes is not None else CLASSES)
    if X.ndim != 2 or not len(X) or len(X) != len(y) or not np.isfinite(X).all():
        raise ValueError("features must be a finite, nonempty (n, d) matrix matching labels")
    if replicates < 2 or folds < 2 or blocks < 1:
        raise ValueError("require replicates >= 2, folds >= 2 and blocks >= 1")
    if len(set(labs)) != len(labs) or set(y) != set(labs):
        raise ValueError("labels must contain exactly the configured classes")
    for key in ("x", "yy"):
        meta[key] = np.asarray(meta[key], dtype=float)
        if meta[key].shape != (len(y),) or not np.isfinite(meta[key]).all():
            raise ValueError(f"metadata {key} must contain one finite coordinate per row")
    y_idx = np.array([labs.index(v) for v in y])
    pairing = hashlib.sha256(json.dumps(
        [labs, list(map(str, y)), seed, replicates, folds, blocks,
         meta["x"].tolist(), meta["yy"].tolist(),
         list(map(str, meta.get("fid", range(len(y)))))],
        separators=(",", ":")).encode()).hexdigest()

    scores, oofs, grids = [], [], []
    for r in range(replicates):
        rng = np.random.default_rng(seed + r)
        g = spatial_blocks(meta["x"], meta["yy"], blocks, rng.random(), rng.random())
        oof = _fit_predict(X, y_idx, g, r, clf_factory, folds)
        scores.append(f1_score(y_idx, oof, average="macro"))
        oofs.append(oof)
        grids.append(g)

    res = Result(macro_f1=float(np.mean(scores)), sd=float(np.std(scores, ddof=1)),
                 per_replicate=[float(s) for s in scores], n=len(y_idx), pairing_id=pairing)
    f1s = np.array([f1_score(y_idx, o, average=None, labels=range(len(labs)))
                    for o in oofs])
    # precision/recall are averaged over the SAME replicates as F1; taking them
    # from replicate 0 while averaging F1 produces a table whose columns do not
    # describe the same experiment.
    prs = np.array([precision_recall_fscore_support(
        y_idx, o, labels=range(len(labs)), zero_division=0)[:2] for o in oofs])
    for i, c in enumerate(labs):
        res.per_class_f1[c] = float(f1s[:, i].mean())
        res.per_class_pr[c] = (float(prs[:, 0, i].mean()), float(prs[:, 1, i].mean()))
    res.confusion = confusion_matrix(y_idx, oofs[0], labels=range(len(labs)))
    res._oofs, res._y = oofs, y_idx  # kept for compare()
    return res


MIN_EFFECT = 0.02  # smallest macro-F1 gap the protocol will call real


def compare(res_a, res_b, name_a="A", name_b="B", min_effect=MIN_EFFECT):
    """Paired comparison. Use this, NOT a difference of two means.

    Both results must come from evaluate() with the same replicate count, so
    replicate r used the identical grid for both systems.

    A win must clear three bars: it must hold in EVERY replicate, exceed twice
    its own standard error, and be at least MIN_EFFECT in size. The last two
    matter because paired deltas are strongly correlated -- without a minimum
    effect size, a change of 0.002 can look perfectly "consistent". Measured
    against a null of two equally-good representations, requiring only 7/8 and
    2*SE called 20% of non-differences "resolved"; these three bars cut that
    to 1.7%.
    """
    a = np.array(res_a.per_replicate)
    b = np.array(res_b.per_replicate)
    if not res_a.pairing_id or res_a.pairing_id != res_b.pairing_id:
        raise ValueError("results are not paired: population, coordinates or protocol differs")
    if len(a) != len(b):
        raise ValueError("results must have the same number of replicates")
    if len(a) < 2 or not np.isfinite([a, b]).all():
        raise ValueError("comparison requires at least two finite replicate scores")
    if not 0 < min_effect <= 1:
        raise ValueError("min_effect must be in (0, 1]")
    d = a - b
    n = len(d)
    wins = int((d > 0).sum())
    # sample SD (ddof=1): with 6-8 replicates the population SD understates the
    # spread by ~7-9 %, which is enough to flip a borderline verdict.
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    print(f"\npaired {name_a} - {name_b}: {d.mean():+.4f} (sd {sd:.4f}), "
          f"{name_a} wins {wins}/{n}")
    unanimous = bool(np.all(d > 0) or np.all(d < 0))
    if unanimous and abs(d.mean()) > 2 * se and abs(d.mean()) >= min_effect:
        print("  -> resolved: unanimous across replicates and above the "
              f"{min_effect:.2f} minimum effect size")
    elif unanimous and abs(d.mean()) > 2 * se:
        print(f"  -> consistent but SMALL ({abs(d.mean()):.4f} < {min_effect:.2f}): "
              "not separated under the decision rule")
    elif abs(d.mean()) < d.std():
        print("  -> WITHIN NOISE: do not claim this as an improvement")
    else:
        print("  -> suggestive only: needs more replicates")
    return d


def report(res: Result, title="result"):
    labs = list(res.per_class_f1)
    print(f"\n=== {title} ===")
    print(f"n = {res.n}   macro-F1 = {res.macro_f1:.4f} +/- {res.sd:.4f} "
          f"over {len(res.per_replicate)} replicates")
    print(f"  replicates: " + " ".join(f"{s:.3f}" for s in res.per_replicate))
    print(f"\n{'class':<22}{'P':>8}{'R':>8}{'F1':>8}    (all averaged over "
          f"{len(res.per_replicate)} replicates)")
    for c in labs:
        p, r = res.per_class_pr[c]
        print(f"{c:<22}{p:>8.3f}{r:>8.3f}{res.per_class_f1[c]:>8.3f}")
    print(f"{'macro':<22}{'':>8}{'':>8}{res.macro_f1:>8.3f}")
    if res.confusion is not None:
        print("\nconfusion, row-normalised % (replicate 0):")
        print(f"{'true / pred':<22}" + "".join(f"{c[:11]:>13}" for c in labs))
        for i, c in enumerate(labs):
            row = res.confusion[i]
            print(f"{c:<22}" + "".join(f"{100*v/max(row.sum(),1):>12.1f}%" for v in row))
