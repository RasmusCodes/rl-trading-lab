"""Final, one-time evaluation on the locked holdout (2019-01-01 to 2026-11-12).

Configurations evaluated here are fixed before this script is run (defaults + the HPO-tuned PPO + the best
training variant from the development folds). Training: 1994-2016, validation: 2017-2018, test: 2019-2026.
Every run appends to runs/holdout_ledger.jsonl.

    python scripts/run_holdout.py rl          # RL agents
    python scripts/run_holdout.py baselines   # baseline suite on the holdout fold
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt.data.splits import HoldoutLedger  # noqa: E402
from rlt.training.runner import RunSpec, run_many  # noqa: E402
from rlt.utils.io import RUNS_DIR  # noqa: E402

ALGOS = ["qlearning", "dqn", "ddqn", "a2c", "ppo", "sac", "td3", "rrl", "ppo_lstm"]


def rl_specs(extra: list[dict]) -> list[RunSpec]:
    specs = [RunSpec(experiment="holdout_spy", universe="spy", algo=a, fold="holdout", seed=s, total_steps=200_000,
                     eval_every=10_000, holdout=True) for a in ALGOS for s in range(3)]
    for e in extra:
        for s in range(3):
            specs.append(RunSpec(experiment="holdout_spy", fold="holdout", seed=s, total_steps=200_000,
                                 eval_every=10_000, holdout=True, **e))
    return specs


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "rl"
    if what == "rl":
        extra = []
        best = RUNS_DIR / "hpo_ppo_spy" / "best.json"
        if best.exists():
            cfg = json.loads(best.read_text())["config"]
            extra.append(dict(universe="spy", algo="ppo", tag="tuned", algo_params=cfg["algo_params"], env=cfg["env"],
                              features=cfg["features"]))
        variant = os.environ.get("HOLDOUT_VARIANT")  # e.g. pooled19, chosen on the development folds
        if variant == "pooled19":
            extra.append(dict(universe="pool19", algo="ppo", tag="pooled19", eval_assets=["SPY"]))
        run_many(rl_specs(extra), workers=14)
    elif what == "baselines":
        from rlt.experiments.baselines import run_baseline_suite
        from rlt.training.context import get_holdout, get_market

        HoldoutLedger.unlock("baselines/spy_holdout", {"universe": "spy"})
        md = get_market("spy")
        run_baseline_suite("spy", folds=[get_holdout(md, "spy")], out_name="spy_holdout")
