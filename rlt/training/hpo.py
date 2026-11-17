"""Hyperparameter optimisation with Optuna (TPE), scored on validation windows only.

Leakage-aware protocol: in an expanding walk-forward, fold k+1's validation window can be fold k's test
window. Tuning on the validation windows of folds 1..K therefore contaminates the test windows of those
folds. We tune on folds 1..3 (validation 2005-2010) and report the tuned configuration only on folds whose
test window starts after the last tuning window (folds 4..6: 2013-2018) and on the 2019+ holdout.

Each trial = one seed on each tuning fold; objective = mean validation Sharpe of the selected checkpoints.
Trials run in parallel batches through the ordinary experiment runner (ask/tell interface).
"""
from __future__ import annotations

import json

import optuna

from ..utils.io import RUNS_DIR, save_json
from .runner import RunSpec, run_many

HIDDEN = {"32x32": (32, 32), "64x64": (64, 64), "128x128": (128, 128)}


def ppo_space(trial: optuna.Trial) -> dict:
    return {
        "algo_params": {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "gamma": trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.99]),
            "n_steps": trial.suggest_categorical("n_steps", [32, 64, 128, 256]),
            "epochs": trial.suggest_int("epochs", 2, 10),
            "ent_coef": trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True),
            "log_std_init": trial.suggest_float("log_std_init", -2.0, 0.0),
            "hidden": HIDDEN[trial.suggest_categorical("hidden", list(HIDDEN))],
            "clip": trial.suggest_categorical("clip", [0.1, 0.2, 0.3]),
        },
        "env": {
            "episode_length": trial.suggest_categorical("episode_length", [63, 126, 252, 504]),
            "reward": {"type": trial.suggest_categorical("reward", ["log_return", "diff_sharpe", "vol_scaled",
                                                                    "mv_utility"])},
            "trade_band": trial.suggest_categorical("trade_band", [0.0, 0.05, 0.1, 0.2]),
        },
        "features": trial.suggest_categorical("features", ["compact", "minimal", "returns_vol", "full"]),
    }


def run_hpo(experiment: str = "hpo_ppo_spy", n_trials: int = 32, batch: int = 4, steps: int = 150_000,
            tune_folds=("wf1_2007", "wf2_2009", "wf3_2011"), workers: int = 12, seed: int = 0) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=8, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=experiment)
    log = []
    done = 0
    while done < n_trials:
        trials = [study.ask() for _ in range(min(batch, n_trials - done))]
        cfgs = [ppo_space(t) for t in trials]
        specs = []
        for t, c in zip(trials, cfgs):
            for f in tune_folds:
                specs.append(RunSpec(experiment=experiment, universe="spy", algo="ppo", fold=f, seed=0,
                                     total_steps=steps, eval_every=10_000, tag=f"t{t.number:03d}",
                                     algo_params=c["algo_params"], env=c["env"], features=c["features"]))
        res = run_many(specs, workers=workers, verbose=False)
        for t, c in zip(trials, cfgs):
            r = res[res["tag"] == f"t{t.number:03d}"]
            vals = r["val_sharpe"].astype(float) if "val_sharpe" in r else []
            score = float(vals.mean()) if len(vals) and vals.notna().all() else float("-inf")
            study.tell(t, score if score > float("-inf") else -10.0)
            log.append({"trial": t.number, "score": score, "params": t.params,
                        "val_by_fold": dict(zip(r["fold"], vals)) if len(vals) else {},
                        "test_by_fold_NOT_USED": dict(zip(r["fold"], r.get("test_sharpe", [])))})
            print(f"trial {t.number:3d}: mean val Sharpe {score:+.3f}  {t.params}", flush=True)
        done += len(trials)
        save_json(RUNS_DIR / experiment / "hpo_log.json", log)
    best = study.best_trial
    save_json(RUNS_DIR / experiment / "best.json", {"trial": best.number, "value": best.value, "params": best.params,
                                                    "config": ppo_space(optuna.trial.FixedTrial(best.params)),
                                                    "n_trials": len(study.trials)})
    return study


def best_config(experiment: str = "hpo_ppo_spy") -> dict:
    return json.loads((RUNS_DIR / experiment / "best.json").read_text())["config"]
