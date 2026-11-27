"""Causal feature engineering from daily bars.

Every feature at row t uses bars 0..t only (trailing windows, recursive EMAs, filtered regimes).
`tests/test_causality.py` enforces this by recomputing features on truncated histories and checking that
row t never changes when rows > t are deleted.

Groups (the state-representation study switches these on and off):

returns     multi-horizon log returns, 12-1 momentum, overnight gap, intraday move
volatility  realised vol (5/21/63d), Garman-Klass range vol, vol ratio, vol-normalised returns
volume      log-volume z-score, dollar-volume trend
technical   RSI, MACD histogram, Bollinger %b, distance to 50/200d averages, stochastic, ATR
orderflow   daily-bar order-flow proxies: close-location value, bulk-volume-classification imbalance
            (Easley, Lopez de Prado & O'Hara 2012), signed volume, Amihud illiquidity, range z-score.
            True order flow (signed trades, book imbalance) needs TAQ/L2 data that is not available here.
regime      HMM filtered high-vol probability (fit on training data only), vol percentile, trend sign,
            drawdown from 52-week high

Portfolio-state features (positions, exposure, cash, drawdown) are added by the environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .market import MarketData
from .regimes import GaussianHMM

GROUPS = {
    "returns": ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d", "mom_12_1", "gap", "intraday"],
    "volatility": ["vol_5d", "vol_21d", "vol_63d", "gk_vol_21d", "vol_ratio", "ret_5d_vn", "ret_21d_vn"],
    "volume": ["logvol_z", "dvol_trend"],
    "technical": ["rsi_14", "macd_hist", "bb_pctb", "ma_dist_50", "ma_dist_200", "ma_cross", "stoch_14", "atr_pct"],
    "orderflow": ["clv", "clv_5", "bvc_imb_5", "signed_vol_21", "amihud_z", "range_z"],
    "regime": ["hmm_p_high", "vol_pct", "trend_200", "dd_252"],
}
ALL_FEATURES = [f for g in GROUPS.values() for f in g]
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}

FEATURE_SETS = {
    "minimal": ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_252d"],
    "returns": GROUPS["returns"],
    "returns_vol": GROUPS["returns"] + GROUPS["volatility"],
    "rv_volume": GROUPS["returns"] + GROUPS["volatility"] + GROUPS["volume"],
    "rv_technical": GROUPS["returns"] + GROUPS["volatility"] + GROUPS["technical"],
    "rv_orderflow": GROUPS["returns"] + GROUPS["volatility"] + GROUPS["orderflow"],
    "rv_regime": GROUPS["returns"] + GROUPS["volatility"] + GROUPS["regime"],
    "full": ALL_FEATURES,
    # compact default for RL: one or two of each kind, low redundancy
    "compact": ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "mom_12_1", "vol_21d", "vol_ratio", "logvol_z", "rsi_14",
                "ma_dist_200", "clv_5", "bvc_imb_5", "hmm_p_high", "dd_252"],
}

# leakage switches used ONLY by the leakage experiments (rlt.experiments.leakage)
LEAK_FEATURES = {
    "future_return": "next session's open-to-open return (blatant look-ahead; agents must exploit it)",
    "global_zscore": "price level z-scored with full-sample mean/std (subtle look-ahead)",
    "smoothed_regime": "HMM forward-backward (smoothed) probability instead of filtered (subtle look-ahead)",
    "centered_ma": "distance to a centred 21-day moving average (subtle look-ahead)",
}


@dataclass
class FeatureSet:
    values: np.ndarray  # (T, N, F) raw (unscaled) feature values
    names: list
    dates: pd.DatetimeIndex
    symbols: list
    ready: np.ndarray  # (T, N) all features finite
    meta: dict = field(default_factory=dict)

    @property
    def F(self) -> int:
        return len(self.names)

    def get(self, name: str) -> np.ndarray:
        return self.values[:, :, self.names.index(name)]

    def select(self, names) -> "FeatureSet":
        idx = [self.names.index(n) for n in names]
        v = self.values[:, :, idx]
        return FeatureSet(v, list(names), self.dates, self.symbols, np.isfinite(v).all(axis=2), dict(self.meta))


def _df(a, md):
    return pd.DataFrame(a, index=md.dates, columns=md.symbols)


def compute_features(md: MarketData, names=None, fit_end: int | None = None, leak: tuple = (),
                     hmm_states: int = 2) -> FeatureSet:
    """Compute features for every row of `md`.

    fit_end: last row index (inclusive) of the training window. Anything *fitted* (the HMM) only sees rows
             <= fit_end; it is then applied causally (filtered) to the full history.
    leak:    names from LEAK_FEATURES to append (leakage experiments only).
    """
    if names is None:
        names = ALL_FEATURES
    elif isinstance(names, str):
        names = FEATURE_SETS[names]
    names = list(names)
    C, O, H, L = (_df(getattr(md, k), md) for k in ("close", "open", "high", "low"))
    V = _df(np.where(md.volume > 0, md.volume, np.nan), md)
    DV = _df(np.where(md.volume > 0, md.dollar_volume, np.nan), md)
    lr = np.log(C).diff()
    out: dict[str, pd.DataFrame] = {}
    need = set(names)

    def want(*fs):
        return any(f in need for f in fs)

    vol = {n: lr.rolling(n, min_periods=n).std() for n in (5, 21, 63)}
    sd_d63 = vol[63]  # daily units

    # ---- returns
    for n in (5, 21, 63, 126, 252):
        out[f"ret_{n}d"] = np.log(C / C.shift(n))
    out["ret_1d"] = lr
    out["mom_12_1"] = np.log(C.shift(21) / C.shift(252))
    out["gap"] = np.log(O / C.shift(1))
    out["intraday"] = np.log(C / O)
    # ---- volatility (annualised)
    for n in (5, 21, 63):
        out[f"vol_{n}d"] = vol[n] * np.sqrt(252)
    gk = 0.5 * np.log(H / L) ** 2 - (2 * np.log(2) - 1) * np.log(C / O) ** 2
    out["gk_vol_21d"] = np.sqrt(gk.rolling(21, min_periods=21).mean().clip(lower=0) * 252)
    out["vol_ratio"] = np.log(vol[5] / vol[63])
    out["ret_5d_vn"] = out["ret_5d"] / (vol[21] * np.sqrt(5))
    out["ret_21d_vn"] = out["ret_21d"] / (sd_d63 * np.sqrt(21))
    # ---- volume
    if want("logvol_z", "dvol_trend"):
        lv = np.log(V)
        out["logvol_z"] = (lv - lv.rolling(63, min_periods=40).mean()) / lv.rolling(63, min_periods=40).std()
        out["dvol_trend"] = np.log(DV.rolling(5, min_periods=3).mean() / DV.rolling(63, min_periods=40).mean())
    # ---- technical
    if want(*GROUPS["technical"]):
        d = C.diff()
        up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        out["rsi_14"] = (rsi - 50) / 50
        macd = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
        out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (C * sd_d63)
        m20, s20 = C.rolling(20, min_periods=20).mean(), C.rolling(20, min_periods=20).std()
        out["bb_pctb"] = (C - (m20 - 2 * s20)) / (4 * s20) - 0.5
        sma50, sma200 = C.rolling(50, min_periods=50).mean(), C.rolling(200, min_periods=200).mean()
        out["ma_dist_50"] = np.log(C / sma50) / sd_d63
        out["ma_dist_200"] = np.log(C / sma200) / sd_d63
        out["ma_cross"] = np.log(sma50 / sma200) / sd_d63
        hh, ll = H.rolling(14, min_periods=14).max(), L.rolling(14, min_periods=14).min()
        out["stoch_14"] = (C - ll) / (hh - ll).replace(0, np.nan) - 0.5
        tr = pd.concat([H - L, (H - C.shift(1)).abs(), (L - C.shift(1)).abs()]).groupby(level=0).max()
        tr = tr.reindex(C.index)
        out["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / C
    # ---- order-flow proxies
    if want(*GROUPS["orderflow"]):
        rng = (H - L)
        clv = ((C - L) - (H - C)) / rng.replace(0, np.nan)
        clv = clv.fillna(0.0)
        out["clv"] = clv
        Vf = V.fillna(0.0)
        out["clv_5"] = (clv * Vf).rolling(5, min_periods=5).sum() / Vf.rolling(5, min_periods=5).sum().replace(0, np.nan)
        z = lr / vol[21].shift(1)
        buy = pd.DataFrame(ndtr(z.to_numpy()), index=C.index, columns=C.columns)
        imb = (2 * buy - 1) * Vf
        out["bvc_imb_5"] = imb.rolling(5, min_periods=5).sum() / Vf.rolling(5, min_periods=5).sum().replace(0, np.nan)
        sv = np.sign(lr) * Vf
        out["signed_vol_21"] = sv.rolling(21, min_periods=21).sum() / Vf.rolling(21, min_periods=21).sum().replace(0, np.nan)
        am = np.log((lr.abs() / DV).rolling(21, min_periods=15).mean() + 1e-18)
        out["amihud_z"] = (am - am.rolling(252, min_periods=126).mean()) / am.rolling(252, min_periods=126).std()
        lhl = np.log(H / L)
        out["range_z"] = lhl / lhl.rolling(63, min_periods=40).mean().replace(0, np.nan) - 1
    # ---- regimes
    if want(*GROUPS["regime"]) or "smoothed_regime" in leak:
        p_hi = np.full((md.T, md.N), np.nan)
        p_sm = np.full((md.T, md.N), np.nan)
        fe = md.T - 1 if fit_end is None else int(fit_end)
        models = {}
        for j, s in enumerate(md.symbols):
            x = lr.iloc[:, j].to_numpy()
            train = x[: fe + 1]
            train = train[np.isfinite(train)]
            if len(train) < 250:
                continue
            hmm = GaussianHMM(hmm_states).fit(train)
            models[s] = hmm.summary()
            first = np.argmax(np.isfinite(x))
            p_hi[first:, j] = hmm.filter(x[first:])[:, -1]
            if "smoothed_regime" in leak:
                p_sm[first:, j] = hmm.smooth(x[first:])[:, -1]
        out["hmm_p_high"] = _df(p_hi, md)
        if "smoothed_regime" in leak:
            out["leak_smoothed_regime"] = _df(p_sm, md)
        out["vol_pct"] = vol[21].rolling(504, min_periods=252).rank(pct=True) - 0.5
        sma200 = C.rolling(200, min_periods=200).mean()
        out["trend_200"] = np.sign(C - sma200)
        out["dd_252"] = np.log(C / C.rolling(252, min_periods=252).max())
        out["_hmm"] = models  # type: ignore[assignment]
    # ---- deliberate leaks (experiments only)
    if "future_return" in leak:
        out["leak_future_return"] = np.log(O.shift(-2) / O.shift(-1))
    if "global_zscore" in leak:
        lp = np.log(C)
        out["leak_global_zscore"] = (lp - lp.mean()) / lp.std()
    if "centered_ma" in leak:
        out["leak_centered_ma"] = np.log(C / C.rolling(21, center=True, min_periods=21).mean()) / sd_d63

    hmm_meta = out.pop("_hmm", None)
    final_names = names + [f"leak_{k}" for k in leak]
    vals = np.stack([out[n].to_numpy(dtype=float) for n in final_names], axis=2)
    vals[~np.isfinite(vals)] = np.nan
    ready = np.isfinite(vals).all(axis=2) & md.valid
    meta = {"fit_end": None if fit_end is None else md.dates[int(fit_end)].date().isoformat(), "leak": list(leak)}
    if hmm_meta:
        meta["hmm"] = hmm_meta
    return FeatureSet(vals, final_names, md.dates, list(md.symbols), ready, meta)


class FeatureScaler:
    """Robust per-feature standardisation fitted on training rows only (pooled across instruments)."""

    def __init__(self, clip: float = 5.0):
        self.clip = clip
        self.center = self.scale = None

    def fit(self, fs: FeatureSet, i0: int, i1: int) -> "FeatureScaler":
        block = fs.values[i0: i1 + 1]
        mask = fs.ready[i0: i1 + 1]
        X = block[mask]  # (n, F)
        if len(X) < 10:
            raise ValueError("not enough training rows to fit the scaler")
        self.center = np.nanmedian(X, axis=0)
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        sd = np.nanstd(X, axis=0)
        scale = np.where(scale > 1e-12, scale, sd)
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        z = (values - self.center) / self.scale
        z = np.clip(z, -self.clip, self.clip)
        z[~np.isfinite(z)] = 0.0
        return z.astype(np.float32)
