"""Aligned multi-instrument market data.

`MarketData` holds (T, N) arrays on one trading calendar (NYSE days, taken from SPY):

open/high/low/close   total-return prices: dividends reinvested for equities/ETFs (Yahoo adjclose/close factor
                      applied to every field), roll-adjusted for futures. All P&L uses these.
close_raw             prices that actually traded (splits undone, futures not back-adjusted). Used only for
                      tick-size spreads and futures contract sizing.
volume                shares (contracts for futures), split-adjusted.
dollar_volume         close_raw * volume * multiplier.
valid                 a genuine bar exists on that date (otherwise prices are carried and trading is blocked).
roll                  futures roll dates (the environment charges a calendar-spread trade on held contracts).
rf                    daily simple T-bill rate from close t to close t+1, known at close t.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .sources import DATA_END, fetch_splits, fetch_yahoo, risk_free_daily, unsplit_factor
from .universe import UNIVERSES, InstrumentSpec, get_spec


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    symbols: list
    specs: list
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    close_raw: np.ndarray
    volume: np.ndarray
    rf: np.ndarray
    valid: np.ndarray
    roll: np.ndarray
    name: str = "market"
    meta: dict = field(default_factory=dict)
    dvol: np.ndarray | None = None  # traded dollar volume; see dollar_volume

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.symbols)

    @property
    def multipliers(self) -> np.ndarray:
        return np.array([s.multiplier for s in self.specs], dtype=float)

    @property
    def is_future(self) -> np.ndarray:
        return np.array([s.is_future for s in self.specs], dtype=bool)

    @property
    def dollar_volume(self) -> np.ndarray:
        """Traded dollar volume = split-adjusted price x split-adjusted volume (both on the same basis).

        Never close_raw x volume: close_raw has splits undone while Yahoo's volume is split-adjusted, which
        multiplies ADV by the cumulative FUTURE split ratio (x0.006 for AIG before its 2009 reverse split).
        """
        if self.dvol is not None:
            return self.dvol
        return self.close_raw * self.volume * self.multipliers[None, :]

    def index_of(self, date, side="left") -> int:
        return int(self.dates.searchsorted(pd.Timestamp(date), side=side))

    def first_valid_index(self) -> int:
        """First date on which every instrument has a genuine bar."""
        ok = np.where(self.valid.all(axis=1))[0]
        return int(ok[0]) if len(ok) else self.T

    def subset(self, symbols) -> "MarketData":
        idx = [self.symbols.index(s) for s in symbols]
        return replace(self, symbols=[self.symbols[i] for i in idx], specs=[self.specs[i] for i in idx],
                       open=self.open[:, idx], high=self.high[:, idx], low=self.low[:, idx],
                       close=self.close[:, idx], close_raw=self.close_raw[:, idx], volume=self.volume[:, idx],
                       valid=self.valid[:, idx], roll=self.roll[:, idx], name=f"{self.name}[{','.join(symbols)}]",
                       dvol=None if self.dvol is None else self.dvol[:, idx])

    def slice(self, i0: int, i1: int) -> "MarketData":
        """Rows i0..i1-1 (used by tests that truncate history)."""
        sl = slice(i0, i1)
        return replace(self, dates=self.dates[sl], open=self.open[sl], high=self.high[sl], low=self.low[sl],
                       close=self.close[sl], close_raw=self.close_raw[sl], volume=self.volume[sl],
                       rf=self.rf[sl], valid=self.valid[sl], roll=self.roll[sl],
                       dvol=None if self.dvol is None else self.dvol[sl])

    def frame(self, field_name: str) -> pd.DataFrame:
        return pd.DataFrame(getattr(self, field_name), index=self.dates, columns=self.symbols)

    def liquidity(self, window: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """(ADV in $, daily return sigma), both known at close t (windows end at t inclusive).

        ADV is a rolling *median* of dollar volume over genuine bars: Yahoo's futures volume has
        occasional zeros and 10^6x spikes around expiries that would wreck a mean.
        """
        dv = pd.DataFrame(np.where(self.valid & (self.volume > 0), self.dollar_volume, np.nan))
        adv = dv.rolling(window, min_periods=5).median().ffill().to_numpy()
        lr = pd.DataFrame(np.log(self.close)).diff()
        sig = lr.rolling(window, min_periods=5).std().to_numpy()
        sig = np.where(np.isfinite(sig), np.maximum(sig, 1e-4), 0.02)
        adv = np.where(np.isfinite(adv) & (adv > 0), adv, np.nan)
        return adv, sig


# --------------------------------------------------------------------------------------- futures roll handling
def _third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - d.weekday()) % 7  # Friday = 4
    return d + pd.Timedelta(days=offset + 14)


def rebuild_futures_rolls(fut: pd.DataFrame, spot_close: pd.Series | None, roll_months: tuple,
                          beta_mode: str = "one"):
    """Remove roll artefacts from a Yahoo continuous front-month series.

    Yahoo splices contracts without adjustment, so around each quarterly expiry the series jumps by the
    calendar spread (roughly the 3-month carry, ~1% for ES at 5% rates) and the expiry day itself is often
    distorted. For days inside an expiry window whose futures move departs abnormally from beta x the proxy's
    move, the futures log return is replaced by beta x the proxy return plus the normal daily carry (rolling
    median, past only). Prices are then rebuilt backwards from the last close (ratio back-adjustment).

    beta_mode "one": the proxy is the futures' own cash index (ES/^GSPC, NQ/^NDX, YM/^DJI).
    beta_mode "rolling": the proxy is a related instrument (ZN/IEF); beta is a trailing 252-session regression
    slope estimated on past data only.

    Returns (adjusted frame with O/H/L/C scaled per day, boolean Series of patched days, roll-date Series).
    """
    fut = fut.copy()
    lf = np.log(fut["close"])
    rf_ = lf.diff()
    patched = pd.Series(False, index=fut.index)
    if spot_close is not None:
        ls = np.log(spot_close.reindex(fut.index))
        rs = ls.diff()
        if beta_mode == "rolling":
            beta = (rf_.rolling(252, min_periods=60).cov(rs) / rs.rolling(252, min_periods=60).var()).shift(1)
            beta = beta.clip(0.2, 3.0)
        else:
            beta = pd.Series(1.0, index=fut.index)
        db = rf_ - beta * rs
        carry = db.rolling(63, min_periods=20).median().shift(1)
        resid = db - carry
        scale = 1.4826 * resid.abs().rolling(63, min_periods=20).median().shift(1)
        in_window = pd.Series(False, index=fut.index)
        for y in range(fut.index[0].year, fut.index[-1].year + 1):
            for m in roll_months:
                tf = _third_friday(y, m)
                in_window |= (fut.index >= tf - pd.Timedelta(days=12)) & (fut.index <= tf + pd.Timedelta(days=6))
        # a proxy-ETF basis is much quieter than an index-futures basis (ZN-IEF ~5bp/day), so smaller switch
        # jumps are detectable; patching a pure-noise day merely swaps it for beta x proxy + carry (harmless)
        k_sd, min_jump = (5.0, 0.0025) if beta_mode == "one" else (3.0, 0.0008)
        thr = np.maximum(k_sd * scale.fillna(0.002), min_jump)
        patched = in_window & (resid.abs() > thr) & rs.notna() & carry.notna() & beta.notna()
        rf_ = rf_.where(~patched, beta * rs + carry)
    rf_ = rf_.fillna(0.0)
    log_adj = lf.iloc[-1] - rf_.iloc[::-1].cumsum().iloc[::-1].shift(-1).fillna(0.0)
    k = np.exp(log_adj - lf)
    for c in ("open", "high", "low", "close"):
        fut[c] = fut[c] * k
    # standard roll date: 8 calendar days before the third Friday (next trading day if not a session)
    rolls = pd.Series(False, index=fut.index)
    for y in range(fut.index[0].year, fut.index[-1].year + 1):
        for m in roll_months:
            rd = _third_friday(y, m) - pd.Timedelta(days=8)
            pos = fut.index.searchsorted(rd)
            if 0 < pos < len(fut.index):
                rolls.iloc[pos] = True
    return fut, patched, rolls


# --------------------------------------------------------------------------------------------- loading
def _prepare(spec: InstrumentSpec, refresh: bool) -> tuple[pd.DataFrame, pd.Series, dict]:
    raw = fetch_yahoo(spec.source, refresh=refresh)
    info: dict = {}
    if spec.is_future:
        spot = None
        if spec.spot_proxy:
            sp = fetch_yahoo(spec.spot_proxy, refresh=refresh)
            # an ETF proxy must be on a total-return basis (IEF pays coupons monthly)
            spot = sp["adjclose"] if spec.proxy_beta == "rolling" else sp["close"]
        adj, patched, rolls = rebuild_futures_rolls(raw, spot, spec.roll_months, spec.proxy_beta)
        out = adj[["open", "high", "low", "close"]].copy()
        out["close_raw"] = raw["close"]
        out["volume"] = raw["volume"]
        out["dvol"] = raw["close"] * raw["volume"] * spec.multiplier
        info["roll_patched_days"] = int(patched.sum())
        info["roll_adjusted"] = spec.spot_proxy is not None
        info["roll_proxy"] = spec.spot_proxy
        return out, rolls, info
    k = (raw["adjclose"] / raw["close"]).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    out = raw[["open", "high", "low", "close"]].mul(k, axis=0)
    out["close_raw"] = raw["close"] * unsplit_factor(raw.index, fetch_splits(spec.source))
    out["volume"] = raw["volume"]
    out["dvol"] = raw["close"] * raw["volume"]  # split-adjusted price x split-adjusted volume
    return out, pd.Series(False, index=raw.index), info


def load_market(symbols, start=None, end: str = DATA_END, refresh: bool = False, name: str | None = None) -> MarketData:
    """Load instruments (list of symbols or a universe name) onto the NYSE calendar."""
    if isinstance(symbols, str):
        name = name or symbols
        symbols = UNIVERSES[symbols]
    symbols = list(symbols)
    cal = fetch_yahoo("SPY", refresh=refresh).index
    cal = cal[(cal <= pd.Timestamp(end))]
    if start is not None:
        cal = cal[cal >= pd.Timestamp(start)]
    T, N = len(cal), len(symbols)
    arr = {k: np.full((T, N), np.nan) for k in ("open", "high", "low", "close", "close_raw", "volume", "dvol")}
    valid = np.zeros((T, N), dtype=bool)
    roll = np.zeros((T, N), dtype=bool)
    specs, meta = [], {"symbols": {}}
    for j, sym in enumerate(symbols):
        spec = get_spec(sym)
        specs.append(spec)
        df, rolls, info = _prepare(spec, refresh)
        df = df[~df.index.duplicated(keep="last")]
        on_cal = df.reindex(cal)
        has = on_cal["close"].notna().to_numpy()
        # carry the last close across missing sessions; open=high=low=close, volume 0, not tradable
        close_ff = on_cal["close"].ffill()
        for c in ("open", "high", "low"):
            arr[c][:, j] = np.where(has, on_cal[c], close_ff)
        arr["close"][:, j] = close_ff.to_numpy()
        arr["close_raw"][:, j] = on_cal["close_raw"].ffill().to_numpy()
        arr["volume"][:, j] = np.where(has, on_cal["volume"].fillna(0.0), 0.0)
        arr["dvol"][:, j] = np.where(has, on_cal["dvol"].fillna(0.0), 0.0)
        valid[:, j] = has
        roll[:, j] = rolls.reindex(cal).fillna(False).to_numpy(dtype=bool)
        first = df.index[0].date().isoformat()
        meta["symbols"][sym] = {"source": spec.source, "first_bar": first, "bars_on_calendar": int(has.sum()), **info}
    rf = risk_free_daily(cal, refresh=refresh)
    md = MarketData(dates=cal, symbols=symbols, specs=specs, rf=rf, valid=valid, roll=roll,
                    name=name or ",".join(symbols), meta=meta, **arr)
    return md
