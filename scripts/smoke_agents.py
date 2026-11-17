"""Short runs of every algorithm to check the plumbing and measure speed.

Runs on a synthetic random walk, never on real data: a smoke test must not print real test-window numbers.
(The first version ran on SPY fold 1 and so displayed 2007-08 test Sharpes once; see the report's caveats.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt.training.runner import RunSpec, run_many  # noqa: E402

if __name__ == "__main__":
    algos = sys.argv[1:] or ["qlearning", "dqn", "ddqn", "a2c", "ppo", "sac", "td3", "rrl", "ppo_lstm"]
    specs = [RunSpec("smoke_synthetic", "syn:gbm", a, "wf1_2018", seed=0, total_steps=20_000, eval_every=10_000,
                     algo_params={"updates": 100} if a == "rrl" else {}) for a in algos]
    df = run_many(specs, workers=5, skip_done=False)
    cols = [c for c in ["algo", "train_seconds", "val_sharpe", "test_sharpe", "test_turnover", "test_avg_gross",
                        "error"] if c in df.columns]
    print(df[cols].to_string())
