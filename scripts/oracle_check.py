"""Capability check: can DQN / SAC exploit an oracle feature (next session's return) on a random walk?

Same specs as the oracle part of the leakage experiment, so results are reused there.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt.training.runner import RunSpec, run_many  # noqa: E402

if __name__ == "__main__":
    algos = sys.argv[1:] or ["dqn", "sac"]
    specs = [RunSpec(experiment="leakage", universe="syn:gbm", algo=a, fold="wf1_2018", seed=s, total_steps=200_000,
                     tag="gbm_future_return", leak=("future_return",)) for a in algos for s in range(2)]
    df = run_many(specs, workers=4)
    print(df[["run_id", "val_sharpe", "test_sharpe", "test_turnover", "test_avg_gross"]].to_string())
