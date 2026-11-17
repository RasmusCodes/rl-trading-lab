"""PPO hyperparameter search on SPY validation windows 2005-2010, then evaluation of the tuned config.

    python scripts/run_hpo.py search      # 24 trials, TPE, objective = mean validation Sharpe over folds 1-3
    python scripts/run_hpo.py evaluate    # tuned vs default PPO on folds 4-6 (test 2013-2018), 3 seeds each
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt.training.hpo import best_config, run_hpo  # noqa: E402
from rlt.training.runner import RunSpec, run_many  # noqa: E402

CLEAN_FOLDS = ["wf4_2013", "wf5_2015", "wf6_2017"]
# v1 = original search (its objective was corrupted by float-noise Sharpe ratios of exactly-flat policies);
# v2 = re-run after the FLAT_SD fix in rlt.evaluation.metrics. Set HPO_EXP=hpo_ppo_spy_v2 for the corrected run.
HPO_EXP = os.environ.get("HPO_EXP", "hpo_ppo_spy")
EVAL_EXP = HPO_EXP.replace("hpo_ppo_spy", "hpo_eval_spy")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "search"
    if cmd == "search":
        run_hpo(experiment=HPO_EXP, n_trials=int(os.environ.get("HPO_TRIALS", 24)), workers=12)
    elif cmd == "evaluate":
        cfg = best_config(HPO_EXP)
        specs = []
        for f in CLEAN_FOLDS:
            for s in range(3):
                specs.append(RunSpec(experiment=EVAL_EXP, universe="spy", algo="ppo", fold=f, seed=s,
                                     total_steps=200_000, eval_every=10_000, tag="tuned",
                                     algo_params=cfg["algo_params"], env=cfg["env"], features=cfg["features"]))
                specs.append(RunSpec(experiment=EVAL_EXP, universe="spy", algo="ppo", fold=f, seed=s,
                                     total_steps=200_000, eval_every=10_000, tag="default"))
        run_many(specs, workers=12)
