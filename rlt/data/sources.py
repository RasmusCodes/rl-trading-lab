"""Daily bar downloads with an on-disk cache.

Source: Yahoo Finance chart API (v8, `query1.finance.yahoo.com/v8/finance/chart/<symbol>`).
It returns split-adjusted open/high/low/close, split-adjusted volume and a split+dividend adjusted close.

Data hygiene applied here (and nowhere else):
* bars are dated in the exchange's own timezone;
* bars after DATA_END are dropped, so a partial intraday bar never enters a backtest;
* rows with a missing close are dropped and duplicated dates keep the last row;
* high/low are widened to contain open and close (Yahoo occasionally violates this);
* the latest session's adjclose is often null, so it is filled from close.

The raw CSVs in data/raw/ are the single source of truth for every experiment; `--refresh` re-downloads them.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import DATA_DIR

DATA_END = "2026-11-12"  # last complete US session when the lab was built (2026-11-13)
RAW_DIR = DATA_DIR / "raw"
_UA = {"User-Agent": "Mozilla/5.0 (rl-trading-lab research)"}


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("^", "IDX_").replace("=", "_").replace("/", "_")
    return RAW_DIR / f"{safe}.csv"


def fetch_yahoo(symbol: str, refresh: bool = False, end: str = DATA_END) -> pd.DataFrame:
    path = _cache_path(symbol)
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return df.loc[: pd.Timestamp(end)]

    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": 0, "period2": int(time.time()), "interval": "1d", "events": "div,split",
              "includeAdjustedClose": "true"}
    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=_UA, timeout=40)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 + 3 * attempt)
    else:
        raise RuntimeError(f"Yahoo download failed for {symbol}: {last_err}")

    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    df = pd.DataFrame(
        {
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
            "adjclose": adj if adj is not None else q["close"],
        },
        index=pd.Index(idx, name="date"),
        dtype="float64",
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df["adjclose"] = df["adjclose"].fillna(df["close"])
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df = df[(df["close"] > 0) | symbol.startswith("^")]
    df = df.loc[: pd.Timestamp(end)]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, float_format="%.8g")

    splits = (res.get("events") or {}).get("splits") or {}
    rows = []
    for ev in splits.values():
        d = pd.to_datetime(ev["date"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
        rows.append({"date": d, "numerator": float(ev["numerator"]), "denominator": float(ev["denominator"])})
    pd.DataFrame(rows, columns=["date", "numerator", "denominator"]).sort_values("date").to_csv(
        _cache_path(symbol).with_name(_cache_path(symbol).stem + "_splits.csv"), index=False)
    return df


def fetch_splits(symbol: str) -> pd.DataFrame:
    """Split events saved alongside the price cache (numerator:denominator, e.g. 2:1)."""
    p = _cache_path(symbol).with_name(_cache_path(symbol).stem + "_splits.csv")
    if not p.exists():
        fetch_yahoo(symbol, refresh=True)
    if not p.exists():
        return pd.DataFrame(columns=["date", "numerator", "denominator"])
    return pd.read_csv(p, parse_dates=["date"])


def unsplit_factor(index: pd.DatetimeIndex, splits: pd.DataFrame) -> np.ndarray:
    """Multiplier turning Yahoo's split-adjusted prices back into the prices that actually traded.

    Needed only for tick-size spreads: a 1/16 $ tick is 14 bp on a $45 stock but 1.4 bp on a $450 one.
    """
    f = np.ones(len(index))
    for _, ev in splits.iterrows():
        if ev["denominator"] > 0:
            f[index < ev["date"]] *= ev["numerator"] / ev["denominator"]
    return f


def risk_free_daily(dates: pd.DatetimeIndex, refresh: bool = False) -> np.ndarray:
    """Daily simple risk-free rate from the 13-week T-bill (^IRX, % p.a.), as known at each close.

    rf[t] is the interest earned on cash held from close t to close t+1. Missing days are forward-filled.
    """
    irx = fetch_yahoo("^IRX", refresh=refresh)["close"].clip(lower=0.0)
    s = irx.reindex(irx.index.union(dates)).sort_index().ffill().reindex(dates)
    s = s.fillna(irx.iloc[0] if len(irx) else 0.0)
    return (s.to_numpy() / 100.0) / 252.0
