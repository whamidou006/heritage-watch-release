"""Fit the frozen-feature head on all available training data and persist it."""
from pathlib import Path

import joblib
import numpy as np
import sklearn

from . import __version__
from .features import assemble, fingerprint, representation
from .protocol import _limit_threads, default_classifier


def train(config, blocks, meta, name, out):
    rep = representation(name)
    X = assemble(blocks, name)
    y = np.asarray(meta["y"]).astype(str)
    if len(X) != len(y) or set(y) != set(config.classes):
        raise ValueError("training data must contain every configured class and matching feature rows")
    with _limit_threads():
        clf = default_classifier().fit(X, y)
    bundle = dict(format_version=1, package_version=__version__,
                  sklearn_version=sklearn.__version__, classifier=clf,
                  classes=list(config.classes), representation=name,
                  encoders=list(rep.encoders), chip_size=config.chip_size,
                  feature_dimension=X.shape[1], training_fingerprint=fingerprint(X, meta),
                  n_training=len(y))
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle
