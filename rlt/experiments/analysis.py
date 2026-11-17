"""Aggregation of run results into walk-forward out-of-sample records and comparison tables.

Conventions (post-audit):
* Development experiments never include dates on or after the holdout start (2019-01-01). The last test
  decision of fold 6 is filled on the first holdout session; that day is trimmed here.
* A strategy is summarised only if it has every fold for every seed in the experiment; incomplete groups are
  reported, not silently averaged.
* The headline statistic for an RL configuration is the Sharpe of its *seed ensemble* (daily returns averaged
  across seeds), with a stationary-bootstrap 95% CI - the same statistic and interval as for a baseline. The
  spread across individual seeds is reported alongside.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..evaluation.metrics import bootstrap_sharpe_ci, paired_sharpe_test, perf_stats
from ..utils.io import RUNS_DIR, load_json, safe_name

HOLDOUT_START = pd.Timestamp("2019-01-01")


def load_runs(experiment: str, ok_only: bool = True) -> pd.DataFrame:
    rows = []
    for p in sorted((RUNS_DIR / experiment).glob("*/eval.json")):
        r = load_json(p)
        r["dir"] = str(p.parent)
        rows.append(r)
    df = pd.DataFrame(rows)
    if ok_only and "error" in df:
        df = df[df["error"].isna()]
    if "tag" in df:
        df["tag"] = df["tag"].fillna("")
    return df


def _is_synthetic(universe) -> bool:
    return isinstance(universe, str) and universe.startswith("syn")


def _read_ledger(d, holdout: bool, synthetic: bool = False) -> pd.DataFrame | None:
    """Test ledger of one run. Real-data development records stop before the holdout starts; synthetic markets
    have a fabricated calendar (their 2018-2021 test block is not 'the holdout'), so they are never trimmed."""
    f = f"{d}/ledger_test.csv"
    try:
        led = pd.read_csv(f, parse_dates=["date"])
    except FileNotFoundError:
        return None
    if not holdout and not synthetic:
        led = led[led["date"] < HOLDOUT_START]
    return led


def _synthetic_runs(runs: pd.DataFrame) -> bool:
    return "universe" in runs and runs["universe"].map(_is_synthetic).all()


def daily_book(led: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight across pooled instruments per date (a single-instrument ledger is returned as is)."""
    cols = [c for c in ("ret", "rf", "turnover", "cost", "gross", "net") if c in led.columns]
    return led.groupby("date")[cols].mean().sort_index()


def completeness(runs: pd.DataFrame, keys=("algo", "tag")) -> pd.DataFrame:
    """Folds per (group, seed) against the experiment's full fold set."""
    all_folds = set(runs["fold"])
    rows = []
    for key, g in runs.groupby(list(keys) + ["seed"], dropna=False):
        rows.append({**dict(zip(list(keys) + ["seed"], key)), "folds": g["fold"].nunique(),
                     "complete": set(g["fold"]) == all_folds})
    return pd.DataFrame(rows)


def stitched(experiment: str, keys=("algo", "tag", "seed"), runs: pd.DataFrame | None = None,
             require_complete: bool = True) -> pd.DataFrame:
    """One OOS record per (keys...): test ledgers of all folds concatenated in time."""
    runs = load_runs(experiment) if runs is None else runs
    holdout = experiment.startswith("holdout")
    synthetic = _synthetic_runs(runs)
    all_folds = set(runs["fold"])
    out = []
    for key, g in runs.groupby(list(keys), dropna=False):
        if require_complete and set(g["fold"]) != all_folds:
            warnings.warn(f"{experiment}: {key} has folds {sorted(set(g['fold']))}, expected {sorted(all_folds)}; skipped")
            continue
        parts = [x for x in (_read_ledger(d, holdout, synthetic) for d in g["dir"]) if x is not None]
        if not parts:
            continue
        book = daily_book(pd.concat(parts))
        st = perf_stats(book["ret"].to_numpy(), book["rf"].to_numpy(), book["turnover"].to_numpy(),
                        book["cost"].to_numpy(), book["gross"].to_numpy(), book["net"].to_numpy())
        st.update(dict(zip(keys, key if isinstance(key, tuple) else (key,))))
        st["folds"] = int(g["fold"].nunique())
        st["start"], st["end"] = str(book.index[0].date()), str(book.index[-1].date())
        out.append(st)
    return pd.DataFrame(out)


def ensemble_book(runs: pd.DataFrame, holdout: bool = False) -> pd.DataFrame:
    """Seed ensemble: per date, the average daily return of all seeds (and pooled instruments)."""
    parts = []
    synthetic = _synthetic_runs(runs)
    for d in runs["dir"]:
        led = _read_ledger(d, holdout, synthetic)
        if led is not None:
            parts.append(daily_book(led))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).groupby(level=0).mean().sort_index()


def config_table(experiment: str, group=("algo", "tag"), runs: pd.DataFrame | None = None,
                 bench: pd.DataFrame | None = None) -> pd.DataFrame:
    """Headline table: ensemble Sharpe + bootstrap CI, seed spread, behaviour, paired test vs a benchmark."""
    runs = load_runs(experiment) if runs is None else runs
    holdout = experiment.startswith("holdout")
    per_seed = stitched(experiment, keys=tuple(group) + ("seed",), runs=runs)
    rows = []
    for key, g in runs.groupby(list(group), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        sel = per_seed
        for k, v in zip(group, key):
            sel = sel[sel[k] == v]
        if sel.empty:
            continue  # incomplete
        book = ensemble_book(g, holdout)
        ex = (book["ret"] - book["rf"]).to_numpy()
        st = perf_stats(book["ret"].to_numpy(), book["rf"].to_numpy(), book["turnover"].to_numpy(),
                        book["cost"].to_numpy(), book["gross"].to_numpy(), book["net"].to_numpy())
        lo, hi = bootstrap_sharpe_ci(ex, n_boot=1000)
        row = {**dict(zip(group, key)), "seeds": int(len(sel)), "sharpe": st["sharpe"], "ci_lo": lo, "ci_hi": hi,
               "seed_mean": float(sel["sharpe"].mean()), "seed_min": float(sel["sharpe"].min()),
               "seed_max": float(sel["sharpe"].max()), "cagr": st["cagr"], "vol": st["vol"],
               "max_drawdown": st["max_drawdown"], "turnover": float(sel["turnover"].mean()),
               "cost_drag": float(sel["cost_drag"].mean()), "avg_gross": float(sel["avg_gross"].mean()),
               "psr": st.get("psr")}
        if bench is not None:
            j = book.join(bench, how="inner", rsuffix="_b")
            t = paired_sharpe_test((j["ret"] - j["rf"]).to_numpy(), (j["ret_b"] - j["rf_b"]).to_numpy())
            row.update(diff_vs_bench=t["sr_diff"], p_vs_bench=t["p_value"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def baseline_book(universe: str, strategy: str, holdout: bool = False) -> pd.DataFrame:
    led = pd.read_csv(RUNS_DIR / "baselines" / safe_name(universe) / "ledgers" / f"{strategy}.csv",
                      parse_dates=["date"])
    if not holdout and not _is_synthetic(universe):
        led = led[led["date"] < HOLDOUT_START]
    return daily_book(led)


def seed_summary(st: pd.DataFrame, group=("algo", "tag"), metrics=("sharpe", "cagr", "vol", "max_drawdown",
                                                                    "turnover", "cost_drag", "avg_gross")) -> pd.DataFrame:
    rows = []
    for key, g in st.groupby(list(group), dropna=False):
        row = dict(zip(group, key if isinstance(key, tuple) else (key,)))
        row["seeds"] = len(g)
        for m in metrics:
            if m in g:
                v = g[m].astype(float)
                row[f"{m}_mean"] = v.mean()
                row[f"{m}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
        row["sharpe_min"] = g["sharpe"].min()
        row["sharpe_max"] = g["sharpe"].max()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sharpe_mean", ascending=False)


def compare_to(book: pd.DataFrame, ref: pd.DataFrame, seed: int = 0) -> dict:
    j = book.join(ref, how="inner", lsuffix="_a", rsuffix="_b")
    return paired_sharpe_test((j["ret_a"] - j["rf_a"]).to_numpy(), (j["ret_b"] - j["rf_b"]).to_numpy(), seed=seed)
