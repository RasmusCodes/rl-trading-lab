"""Print a quick summary table for one or more experiments (stitched OOS per seed, then across seeds).

    python scripts/summarize.py main_spy [--by algo tag] [--bench spy]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from rlt.experiments.analysis import load_runs, seed_summary, stitched  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--by", nargs="+", default=["algo", "tag"])
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    for e in args.experiments:
        runs = load_runs(e)
        if runs.empty:
            print(e, "no runs")
            continue
        if "error" in runs:
            err = runs[runs["error"].notna()]
            if len(err):
                print(f"{e}: {len(err)} failed runs, e.g. {err['error'].iloc[0]}")
            runs = runs[runs["error"].isna()] if "error" in runs else runs
        st = stitched(e, keys=tuple(args.by) + ("seed",), runs=runs)
        tab = seed_summary(st, group=tuple(args.by))
        cols = [c for c in tab.columns if c in args.by or c in ("seeds", "sharpe_mean", "sharpe_sd", "sharpe_min",
                                                                 "sharpe_max", "cagr_mean", "max_drawdown_mean",
                                                                 "turnover_mean", "avg_gross_mean", "cost_drag_mean")]
        print(f"\n=== {e}: {len(runs)} runs; stitched test record per seed, summarised across seeds ===")
        print(tab[cols].round(3).to_string(index=False))
        v = runs.groupby(list(args.by))[["val_sharpe", "test_sharpe", "test_last_sharpe"]].mean().round(3)
        print("\nmean per-run Sharpe: validation (selected ckpt) / test (selected) / test (last ckpt)")
        print(v.to_string())
