"""Synthetic markets: controls and randomised training environments.

Negative controls (no exploitable direction):
    gbm       i.i.d. Gaussian returns
    garch     GARCH(1,1) volatility clustering, still unpredictable in sign (vol timing can help Sharpe)
Positive controls (planted, known edge):
    ar1       AR(1) daily returns with autocorrelation phi (momentum at lag 1 if phi > 0)
    regime    two-state Markov-switching drift and volatility (bull/bear), persistent states
Randomised environments:
    bootstrap stationary block bootstrap of a real instrument's daily bars (gap, intraday, range, volume)
              drawn from a *training window only*; keeps fat tails and short-range dependence, breaks long-range
              structure; used for domain randomisation.

If an agent finds alpha after costs on a negative control, something leaks. If it cannot find the edge on a
positive control, its failure on real data says nothing about the market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market import MarketData
from .universe import InstrumentSpec

SYN_SPEC = InstrumentSpec("SYN", "etf", "synthetic", commission_bps=0.3, min_half_spread_bps=0.5, impact_coef=0.5,
                          borrow_bps_annual=25.0)


def _bars_from_returns(r: np.ndarray, rng, sigma: np.ndarray, start_price=100.0, overnight_share=0.3,
                       base_volume=5e7):
    """Split close-to-close log returns into overnight/intraday parts and build O/H/L/C/V."""
    T, N = r.shape
    nu = rng.normal(0.0, 0.3, (T, N)) * sigma
    r_on = overnight_share * r + nu
    r_id = r - r_on
    logc = np.log(start_price) + np.cumsum(r, axis=0)
    logc_prev = np.vstack([np.full((1, N), np.log(start_price)), logc[:-1]])
    logo = logc_prev + r_on
    hi = np.maximum(logo, logc) + np.abs(rng.normal(0.0, 0.5, (T, N))) * sigma
    lo = np.minimum(logo, logc) - np.abs(rng.normal(0.0, 0.5, (T, N))) * sigma
    vol = base_volume * np.exp(0.5 * np.abs(r) / np.maximum(sigma, 1e-6) + rng.normal(0.0, 0.3, (T, N)))
    del r_id
    return np.exp(logo), np.exp(hi), np.exp(lo), np.exp(logc), vol


def make_synthetic(kind: str = "gbm", T: int = 6000, n_assets: int = 1, seed: int = 0, rf_annual: float = 0.02,
                   **p) -> MarketData:
    rng = np.random.default_rng(seed)
    N = n_assets
    if kind == "gbm":
        mu, sig = p.get("mu", 0.07) / 252, p.get("sigma", 0.16) / np.sqrt(252)
        r = rng.normal(mu - 0.5 * sig**2, sig, (T, N))
        sigma = np.full((T, N), sig)
    elif kind == "garch":
        omega, alpha, beta = p.get("omega", 1e-6), p.get("alpha", 0.08), p.get("beta", 0.9)
        mu = p.get("mu", 0.07) / 252
        h = np.full(N, omega / (1 - alpha - beta))
        r = np.zeros((T, N))
        sigma = np.zeros((T, N))
        e_prev = np.zeros(N)
        for t in range(T):
            h = omega + alpha * e_prev**2 + beta * h
            s = np.sqrt(h)
            e = s * rng.standard_t(6, N) / np.sqrt(6 / 4)
            r[t] = mu + e
            sigma[t] = s
            e_prev = e
    elif kind == "ar1":
        phi, sig = p.get("phi", 0.1), p.get("sigma", 0.16) / np.sqrt(252)
        mu = p.get("mu", 0.05) / 252
        eps = rng.normal(0.0, sig * np.sqrt(1 - phi**2), (T, N))
        r = np.zeros((T, N))
        for t in range(1, T):
            r[t] = mu + phi * (r[t - 1] - mu) + eps[t]
        sigma = np.full((T, N), sig)
    elif kind == "regime":
        mus = np.array(p.get("mus", (0.20, -0.25))) / 252
        sigs = np.array(p.get("sigmas", (0.12, 0.30))) / np.sqrt(252)
        stay = np.array(p.get("stay", (0.99, 0.98)))
        s = np.zeros((T, N), dtype=int)
        for t in range(1, T):
            sw = rng.random(N) > stay[s[t - 1]]
            s[t] = np.where(sw, 1 - s[t - 1], s[t - 1])
        r = rng.normal(mus[s], sigs[s])
        sigma = sigs[s]
    else:
        raise ValueError(kind)
    O, H, L, C, V = _bars_from_returns(r, rng, sigma)
    dates = pd.bdate_range("2000-01-03", periods=T)
    syms = [f"SYN{j}" for j in range(N)] if N > 1 else ["SYN"]
    specs = [SYN_SPEC] * N
    return MarketData(dates=dates, symbols=syms, specs=specs, open=O, high=H, low=L, close=C, close_raw=C.copy(),
                      volume=V, rf=np.full(T, rf_annual / 252), valid=np.ones((T, N), bool),
                      roll=np.zeros((T, N), bool), name=f"synthetic:{kind}", meta={"kind": kind, "seed": seed, **p})


def stack_markets(mds: list) -> MarketData:
    """Side-by-side stack of same-length single-instrument markets (bootstrap paths -> one pool)."""
    cat = lambda k: np.hstack([getattr(m, k) for m in mds])  # noqa: E731
    syms = [f"{m.symbols[0]}_bs{k}" for k, m in enumerate(mds)]
    return MarketData(dates=mds[0].dates, symbols=syms, specs=[m.specs[0] for m in mds], open=cat("open"),
                      high=cat("high"), low=cat("low"), close=cat("close"), close_raw=cat("close_raw"),
                      volume=cat("volume"), rf=np.mean([m.rf for m in mds], axis=0), valid=cat("valid"),
                      roll=cat("roll"), name="bootstrap_pool", meta={"paths": len(mds)})


def block_bootstrap(md: MarketData, i0: int, i1: int, T: int, seed: int = 0, mean_block: float = 20.0) -> MarketData:
    """New single-instrument history assembled from blocks of real daily bars in rows i0..i1 (inclusive)."""
    rng = np.random.default_rng(seed)
    j = 0
    O, H, L, C, V = (getattr(md, k)[:, j] for k in ("open", "high", "low", "close", "volume"))
    rows = np.arange(max(i0, 1), i1 + 1)
    gap = np.log(O[rows] / C[rows - 1])
    intr = np.log(C[rows] / O[rows])
    up = np.log(H[rows] / np.maximum(O[rows], C[rows]))
    dn = np.log(L[rows] / np.minimum(O[rows], C[rows]))
    lv = np.log(np.maximum(V[rows], 1.0))
    raw_ratio = md.close_raw[rows, j] / C[rows]
    n = len(rows)
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, T):
        idx[t] = rng.integers(n) if rng.random() < 1.0 / mean_block else (idx[t - 1] + 1) % n
    logc = np.log(C[rows[0]]) + np.cumsum(gap[idx] + intr[idx])
    logc_prev = np.concatenate([[np.log(C[rows[0]])], logc[:-1]])
    logo = logc_prev + gap[idx]
    hi = np.maximum(logo, logc) + up[idx]
    lo = np.minimum(logo, logc) + dn[idx]
    dates = pd.bdate_range("2000-01-03", periods=T)
    col = lambda x: x[:, None]  # noqa: E731
    Cn = np.exp(logc)
    return MarketData(dates=dates, symbols=[md.symbols[j]], specs=[md.specs[j]], open=col(np.exp(logo)),
                      high=col(np.exp(hi)), low=col(np.exp(lo)), close=col(Cn), close_raw=col(Cn * raw_ratio[idx]),
                      volume=col(np.exp(lv[idx])), rf=md.rf[rows][idx], valid=np.ones((T, 1), bool),
                      roll=np.zeros((T, 1), bool), name=f"bootstrap:{md.symbols[j]}",
                      meta={"source_rows": [int(i0), int(i1)], "seed": seed, "mean_block": mean_block})
