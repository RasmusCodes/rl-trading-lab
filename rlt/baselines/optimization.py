"""Traditional portfolio construction baselines (monthly rebalanced, hold in between).

Estimates use the trailing `window` sessions of close-to-close log returns ending at the decision row.
Covariances use Ledoit-Wolf shrinkage. Rows that are not rebalance rows hold NaN, which the environment
reads as "keep the current position", so drift is not traded away daily.

Portfolio (N instruments):  equal_weight, inverse_vol, risk_parity (equal risk contribution), min_variance,
mean_variance (gamma = 5, momentum-shrunk means), sixty_forty (SPY/IEF), tsmom_port (vol-scaled trend).
Single instrument:          mv_kelly  w = clip(mu / (gamma sigma^2), 0, 1), mu = expanding mean excess return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from ..data.market import MarketData


def month_end_rows(md: MarketData) -> np.ndarray:
    m = md.dates.month.to_numpy()
    rows = np.where(m[:-1] != m[1:])[0]
    return rows


def _logret(md):
    return np.diff(np.log(md.close), axis=0, prepend=np.nan)


def _solve_long_only(obj, N, max_w=1.0, budget=1.0):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - budget}]
    res = minimize(obj, np.full(N, budget / N), bounds=[(0.0, max_w)] * N, constraints=cons, method="SLSQP",
                   options={"maxiter": 200, "ftol": 1e-10})
    w = np.clip(res.x, 0, max_w)
    return w / w.sum() * budget if w.sum() > 0 else np.full(N, budget / N)


def erc_weights(S: np.ndarray, iters: int = 500) -> np.ndarray:
    """Equal-risk-contribution weights by fixed-point iteration (Spinu-style)."""
    N = len(S)
    w = 1.0 / np.sqrt(np.diag(S))
    w /= w.sum()
    for _ in range(iters):
        rc = w * (S @ w)
        w_new = w * (rc.mean() / np.maximum(rc, 1e-16)) ** 0.5
        w_new /= w_new.sum()
        if np.abs(w_new - w).max() < 1e-10:
            w = w_new
            break
        w = w_new
    return w


def portfolio_weights(md: MarketData, kind: str, window: int = 252, gamma: float = 5.0, max_w: float = 1.0,
                      vol_target: float = 0.10) -> np.ndarray:
    lr = _logret(md)
    T, N = md.T, md.N
    W = np.full((T, N), np.nan)
    rows = month_end_rows(md)
    for t in rows:
        if t < window:
            continue
        R = lr[t - window + 1: t + 1]
        if not np.isfinite(R).all():
            continue
        if kind == "equal_weight":
            w = np.full(N, 1.0 / N)
        elif kind == "inverse_vol":
            iv = 1.0 / R.std(0, ddof=1)
            w = iv / iv.sum()
        elif kind in ("risk_parity", "min_variance", "mean_variance"):
            S = LedoitWolf().fit(R).covariance_
            if kind == "risk_parity":
                w = erc_weights(S)
            elif kind == "min_variance":
                w = _solve_long_only(lambda w: w @ S @ w, N, max_w)
            else:
                mom = np.log(md.close[t - 21] / md.close[t - window]) * (252 / (window - 21))
                mu = 0.5 * mom + 0.5 * mom.mean()  # shrink toward the cross-sectional mean
                w = _solve_long_only(lambda w: -(w @ mu * 1.0 / 252 * 252) + 0.5 * gamma * (w @ (S * 252) @ w), N, max_w)
        elif kind == "sixty_forty":
            w = np.zeros(N)
            if "SPY" in md.symbols and "IEF" in md.symbols:
                w[md.symbols.index("SPY")], w[md.symbols.index("IEF")] = 0.6, 0.4
            else:
                w[:] = 1.0 / N
        elif kind == "tsmom_port":
            sig = np.sign(np.log(md.close[t] / md.close[t - window]))
            vol = R[-63:].std(0, ddof=1) * np.sqrt(252)
            w = sig * (vol_target / np.maximum(vol, 1e-4)) / N
            g = np.abs(w).sum()
            if g > 1.0:
                w = w / g
        else:
            raise ValueError(kind)
        W[t] = w
    return W


def mv_kelly(md: MarketData, gamma: float = 3.0, vol_window: int = 63, min_obs: int = 252) -> np.ndarray:
    """Single-instrument mean-variance (fractional Kelly) weight with an expanding-mean expected return."""
    ex = pd.DataFrame(_logret(md) - md.rf[:, None])
    mu = ex.expanding(min_periods=min_obs).mean()
    var = ex.rolling(vol_window, min_periods=vol_window).var()
    w = (mu / (gamma * var)).clip(0.0, 1.0)
    return w.fillna(0.0).to_numpy()


PORTFOLIO_BASELINES = ["equal_weight", "inverse_vol", "risk_parity", "min_variance", "mean_variance", "sixty_forty",
                       "tsmom_port"]
