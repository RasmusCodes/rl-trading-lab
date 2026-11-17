"""Supervised-learning baselines: forecast the return the next decision will earn, then size by the forecast.

Label for decision row t (filled at the open of t+1): y_t = log(O_t+2 / O_t+1), the open-to-open return of
the position the decision creates. Training rows come from the fold's training window only; the window
already ends `embargo` sessions before validation, so no label straddles the boundary.

Models (sklearn): ridge regression, logistic regression on sign(y), shallow gradient boosting.
Position: w = clip(yhat / (k * sd(yhat_train)), -1, 1)  (k tuned on validation), or the long-only clip to [0, 1].
Data are pooled across instruments in the pool, i.e. one model per fold, as for the pooled RL agents.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

GRIDS = {
    "ridge": {"alpha": [1.0, 100.0, 10000.0], "k": [0.5, 1.0, 2.0], "long_only": [False, True]},
    "logistic": {"C": [0.001, 0.1], "k": [0.5, 1.0, 2.0], "long_only": [False, True]},
    "gbm": {"max_depth": [2, 3], "k": [0.5, 1.0, 2.0], "long_only": [False, True]},
}


def labels(md) -> np.ndarray:
    O = md.open
    y = np.full_like(O, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        y[:-2] = np.log(O[2:] / O[1:-1])
    return y


def _xy(X, y, ready, rows, assets):
    xs, ys = [], []
    for a in assets:
        r = rows[ready[rows, a] & np.isfinite(y[rows, a])]
        xs.append(X[r, a, :])
        ys.append(y[r, a])
    return np.concatenate(xs), np.concatenate(ys)


def fit_predict(kind: str, X: np.ndarray, y: np.ndarray, ready: np.ndarray, train: tuple, assets, params: dict,
                seed: int = 0) -> tuple[np.ndarray, float]:
    """Fit on training rows; return raw forecasts for every row/asset and the training forecast sd."""
    rows = np.arange(train[0], train[1] + 1)
    Xtr, ytr = _xy(X, y, ready, rows, assets)
    T, N, F = X.shape
    flat = X.reshape(T * N, F)
    if kind == "ridge":
        m = Ridge(alpha=params["alpha"]).fit(Xtr, ytr)
        pred, ptr = m.predict(flat), m.predict(Xtr)
    elif kind == "logistic":
        m = LogisticRegression(C=params["C"], max_iter=2000).fit(Xtr, (ytr > 0).astype(int))
        pred, ptr = m.predict_proba(flat)[:, 1] - 0.5, m.predict_proba(Xtr)[:, 1] - 0.5
    elif kind == "gbm":
        m = HistGradientBoostingRegressor(max_depth=params["max_depth"], learning_rate=0.05, max_iter=150,
                                          l2_regularization=1.0, min_samples_leaf=200, random_state=seed)
        m.fit(Xtr, ytr)
        pred, ptr = m.predict(flat), m.predict(Xtr)
    else:
        raise ValueError(kind)
    return pred.reshape(T, N), float(np.std(ptr) + 1e-12)


def to_weights(pred: np.ndarray, sd: float, k: float, long_only: bool) -> np.ndarray:
    w = np.clip(pred / (k * sd), -1.0, 1.0)
    return np.clip(w, 0.0, 1.0) if long_only else w


def configs(kind: str):
    g = GRIDS[kind]
    keys = list(g)
    for vals in itertools.product(*(g[k] for k in keys)):
        yield dict(zip(keys, vals))
