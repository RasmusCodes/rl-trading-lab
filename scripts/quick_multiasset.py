"""Headline tables for the multi-asset studies: RL seed ensembles vs the universe's baselines."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from rlt.evaluation.metrics import bootstrap_sharpe_ci, perf_stats  # noqa: E402
from rlt.experiments.analysis import baseline_book, config_table  # noqa: E402
from rlt.utils.io import RUNS_DIR, safe_name  # noqa: E402

STUDIES = [("portfolio_etf9", "etf9", "equal_weight"), ("futures_fut4", "fut4", "equal_weight"),
           ("equities_eq10", "eq10_2000", "buy_hold")]

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for exp, uni, bname in STUDIES:
        bench = baseline_book(uni, bname)
        tab = config_table(exp, group=("algo",), bench=bench)
        print(f"\n=== {exp} (benchmark {bname}) ===")
        print(tab[["algo", "seeds", "sharpe", "ci_lo", "ci_hi", "seed_min", "seed_max", "diff_vs_bench", "p_vs_bench",
                   "cagr", "max_drawdown", "turnover", "avg_gross"]].round(3).to_string(index=False))
        rows = []
        for f in sorted((RUNS_DIR / "baselines" / safe_name(uni) / "ledgers").glob("*.csv")):
            if f.stem.startswith("random_"):
                continue
            b = baseline_book(uni, f.stem)
            st = perf_stats(b["ret"].values, b["rf"].values, b["turnover"].values, b["cost"].values, b["gross"].values)
            lo, hi = bootstrap_sharpe_ci((b["ret"] - b["rf"]).values, n_boot=500)
            rows.append({"baseline": f.stem, "sharpe": st["sharpe"], "ci_lo": lo, "ci_hi": hi, "cagr": st["cagr"],
                         "max_drawdown": st["max_drawdown"], "turnover": st.get("turnover")})
        print(pd.DataFrame(rows).sort_values("sharpe", ascending=False).round(3).to_string(index=False))
