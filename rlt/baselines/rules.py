"""Rule-based baseline strategies as causal (T, N) target-weight tables.

Row t of every table uses bars 0..t only and is executed at the next open by the environment, exactly like an
agent's action. Parameters are the textbook ones, fixed a priori (no tuning on the evaluation data):

buy_hold         w = 1
random_markov    positions in {-1, 0, 1}, switching with probability p per day (mean holding 1/p days)
tsmom_252        sign of the 12-month return, long/short (Moskowitz, Ooi & Pedersen 2012)
tsmom_252_lo     long when the 12-month return is positive, else cash
tsmom_vs         tsmom_252 scaled by 15% / realised vol (63d), capped at |w| <= 1
ma_cross         long when SMA50 > SMA200, else cash (trend filter)
meanrev_5        contrarian: w = -clip(z5 / 2, -1, 1), z5 = 5-day return in units of its volatility
rsi2_dip         long-only: buy when RSI(2) < 10 and price > SMA200, exit when RSI(2) > 70 or price < SMA200
vol_target       w = min(1, 15% / realised vol (21d)): de-risk in turbulence (Moreira & Muir 2017, uncapped there)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.market import MarketData


def _frames(md: MarketData):
    C = pd.DataFrame(md.close, index=md.dates, columns=md.symbols)
    return C, np.log(C).diff()


def buy_hold(md: MarketData) -> np.ndarray:
    return np.ones((md.T, md.N))


def random_markov(md: MarketData, p_switch: float = 0.05, seed: int = 0, levels=(-1.0, 0.0, 1.0)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T, N = md.T, md.N
    lv = np.asarray(levels)
    W = np.zeros((T, N))
    cur = rng.choice(lv, N)
    for t in range(T):
        sw = rng.random(N) < p_switch
        cur = np.where(sw, rng.choice(lv, N), cur)
        W[t] = cur
    return W


def tsmom(md: MarketData, lookback: int = 252, long_only: bool = False, vol_target: float | None = None) -> np.ndarray:
    C, lr = _frames(md)
    sig = np.sign(np.log(C / C.shift(lookback)))
    w = sig.clip(lower=0) if long_only else sig
    if vol_target:
        vol = lr.rolling(63, min_periods=63).std() * np.sqrt(252)
        w = w * (vol_target / vol).clip(upper=1.0)
    return w.fillna(0.0).to_numpy()


def ma_cross(md: MarketData, fast: int = 50, slow: int = 200, short: bool = False) -> np.ndarray:
    C, _ = _frames(md)
    f, s = C.rolling(fast, min_periods=fast).mean(), C.rolling(slow, min_periods=slow).mean()
    w = np.where(f > s, 1.0, -1.0 if short else 0.0)
    w[(f.isna() | s.isna()).to_numpy()] = 0.0
    return w


def meanrev(md: MarketData, lookback: int = 5, z_scale: float = 2.0) -> np.ndarray:
    C, lr = _frames(md)
    z = np.log(C / C.shift(lookback)) / (lr.rolling(21, min_periods=21).std() * np.sqrt(lookback))
    return (-(z / z_scale).clip(-1, 1)).fillna(0.0).to_numpy()


def rsi2_dip(md: MarketData, entry: float = 10.0, exit_: float = 70.0, trend: int = 200) -> np.ndarray:
    C, _ = _frames(md)
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()
    sma = C.rolling(trend, min_periods=trend).mean().to_numpy()
    c = C.to_numpy()
    W = np.zeros((md.T, md.N))
    pos = np.zeros(md.N)
    for t in range(md.T):
        ok = np.isfinite(sma[t]) & np.isfinite(rsi[t])
        above = ok & (c[t] > sma[t])
        enter = above & (rsi[t] < entry)
        leave = ~above | (ok & (rsi[t] > exit_))
        pos = np.where(enter, 1.0, np.where(leave, 0.0, pos))
        W[t] = pos
    return W


def vol_target(md: MarketData, target: float = 0.15, window: int = 21, cap: float = 1.0) -> np.ndarray:
    _, lr = _frames(md)
    vol = lr.rolling(window, min_periods=window).std() * np.sqrt(252)
    return (target / vol).clip(upper=cap).fillna(0.0).to_numpy()


def all_rule_weights(md: MarketData, random_seeds=(0,)) -> dict[str, np.ndarray]:
    out = {
        "buy_hold": buy_hold(md),
        "tsmom_252": tsmom(md),
        "tsmom_252_lo": tsmom(md, long_only=True),
        "tsmom_vs": tsmom(md, vol_target=0.15),
        "ma_cross": ma_cross(md),
        "meanrev_5": meanrev(md),
        "rsi2_dip": rsi2_dip(md),
        "vol_target": vol_target(md),
    }
    for s in random_seeds:
        out[f"random_{s}"] = random_markov(md, seed=s)
    return out
