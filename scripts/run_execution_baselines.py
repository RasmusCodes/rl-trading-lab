"""How much do fill assumptions and costs flatter the SPY baselines?

Runs the baseline suite on the development folds with: fills at the decision close (unrealistic), fills at the
next close, and zero trading costs. Output: runs/baselines/spy_<variant>/ (compare with runs/baselines/spy/).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

from rlt.env.costs import CostConfig  # noqa: E402
from rlt.experiments.baselines import run_baseline_suite  # noqa: E402

VARIANTS = {
    "same_close": {"execution": "same_close"},
    "next_close": {"execution": "next_close"},
    "nocost": {"costs": replace(CostConfig(), multiplier=0.0, borrow=False, financing_spread_bps=0.0)},
    "cost4x": {"costs": replace(CostConfig(), multiplier=4.0)},
}

if __name__ == "__main__":
    for name, ov in VARIANTS.items():
        res = run_baseline_suite("spy", out_name=f"spy_{name}", cfg_overrides=ov, n_random=5, verbose=False,
                                 supervised_kinds=("ridge", "logistic"))
        print(name, res["oos"][["sharpe", "turnover", "cost_drag"]].astype(float).round(3).head(12).to_string(),
              flush=True)
