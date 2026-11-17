"""Run the baseline benchmark suite on one or more universes.

    python scripts/run_baselines.py --universe spy
    python scripts/run_baselines.py --universe etf9 eq10_2000 fut4 syn:gbm syn:ar1 syn:regime
"""
from __future__ import annotations

import os

# one thread per process: sklearn's OpenMP pools thrash when RL workers already occupy every core
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from rlt.experiments.baselines import random_null, run_baseline_suite  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", nargs="+", default=["spy"])
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--features", default="compact")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    for u in args.universe:
        res = run_baseline_suite(u, feature_set=args.features, n_random=args.n_random)
        oos = res["oos"]
        cols = ["family", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "cagr", "vol", "max_drawdown", "turnover",
                "cost_drag", "avg_gross"]
        show = oos[~oos.index.str.startswith("random_")][cols].astype({c: float for c in cols if c != "family"})
        print(f"\n=== {u}: walk-forward OOS (stitched test blocks) ===")
        print(show.round(3).to_string())
        print("random null:", {k: round(v, 3) if isinstance(v, float) else v for k, v in random_null(oos).items()})


if __name__ == "__main__":
    main()
