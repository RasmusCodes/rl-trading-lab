"""Print headline tables (seed-ensemble Sharpe + bootstrap CI + paired test vs buy & hold) for experiments.

    python scripts/quick_tables.py main_spy:algo rewards_spy:tag hacking_spy:tag
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from rlt.experiments.analysis import baseline_book, config_table  # noqa: E402

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    for arg in sys.argv[1:]:
        exp, key = arg.split(":")
        bench = None
        try:
            bench = baseline_book("spy", "buy_hold")
        except FileNotFoundError:
            pass
        tab = config_table(exp, group=(key,), bench=bench)
        cols = [key, "seeds", "sharpe", "ci_lo", "ci_hi", "seed_min", "seed_max", "diff_vs_bench", "p_vs_bench",
                "cagr", "max_drawdown", "turnover", "cost_drag", "avg_gross"]
        print(f"\n=== {exp} ===")
        print(tab[[c for c in cols if c in tab]].round(3).to_string(index=False))
