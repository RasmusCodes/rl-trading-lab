# RL Trading Lab

A research platform for training and evaluating reinforcement-learning trading agents on daily data with
realistic execution. It covers a vectorised trading environment, nine RL algorithms, a baseline suite,
walk-forward evaluation with a locked holdout, a training dashboard and a research report
([`reports/REPORT.md`](reports/REPORT.md)).

## Results in brief

On daily data with realistic execution, none of the nine RL algorithms beat simple strategies out of sample,
and nothing, rule or agent, beat buy & hold SPY on returns.

* **Controls pass.** On synthetic markets with a planted edge, PPO, A2C and Moody-Saffell recurrent RL find it
  (test Sharpe 0.64-1.19). No agent makes money on a pure random walk. All nine algorithms exploit an oracle
  feature (Sharpe 12-14), so the implementations work.
* **SPY 2007-2018, walk-forward.** Agents score 0.19-0.53 against buy & hold's 0.41, with no significant
  difference. All trail a 50/200-day trend filter (0.61). Across the 72 strategies tried, the best information
  ratio against buy & hold is -0.03.
* **Holdout 2019-2026, evaluated once.** Buy & hold scores 0.77, the agents 0.67 down to -0.52. The development
  ranking of the agents reversed.
* **Method findings:**
  * validation Sharpe does not predict test Sharpe (rank correlation -0.08 over 162 runs);
  * hyperparameter search bought nothing out of sample;
  * one subtly leaky feature produces a Sharpe of 7 on a random walk;
  * high-turnover agents lose half their Sharpe by about $20M of capital.

The report details the environment, controls, reward-hacking experiments, an independent code audit with its
fixes, and every caveat. Research code, not investment advice.

## Setup

Python 3.12, CPU only.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # torch comes from the CPU wheel index listed in the file
.venv/Scripts/python scripts/download_data.py             # caches Yahoo Finance daily bars into data/raw/
.venv/Scripts/python -m pytest tests -q                   # accounting, timing, cost and look-ahead tests
```

Market data and experiment outputs are not part of the repository. `scripts/download_data.py` fetches the daily
bars from Yahoo Finance's public chart endpoint for personal research use, and the experiment scripts below
regenerate `runs/`. The report's figures and tables (`reports/figures`, `reports/tables`) are included.

## Layout

| Path | What it is |
|---|---|
| `rlt/data/` | Download + cache, futures roll repair, NYSE-calendar alignment, synthetic markets (controls, block bootstrap), causal features, filtered HMM regimes, walk-forward splits with embargo and a holdout ledger |
| `rlt/env/` | Vectorised trading environment (pool or portfolio mode); cost model with tick-size spread, slippage, square-root impact, commission, ADV cap, borrow, short rebate, financing and short-sale bans; reward functions |
| `rlt/agents/` | Tabular Q-learning, DQN, Double DQN, A2C, PPO, SAC, TD3, Moody-Saffell RRL, PPO-LSTM (PyTorch) |
| `rlt/baselines/` | Buy & hold, random, momentum, mean reversion, supervised (ridge/logistic/GBM), portfolio optimisation |
| `rlt/evaluation/` | Backtester (same env for every policy); metrics (Sharpe, PSR/DSR, bootstrap CIs, drawdown, turnover); capacity and cost robustness |
| `rlt/training/` | Fold contexts; experiment runner (parallel, reproducible seeds, checkpoint selection on validation, environment versioning); Optuna HPO |
| `rlt/experiments/` | Baseline suite and result aggregation |
| `rlt/dashboard/` | Local training dashboard (`server.py` + static page) |
| `rlt/report/` | Figure helpers for the report |
| `scripts/` | Entry points for every experiment, analysis and report build |
| `tests/` | Accounting identities, execution timing, costs, bankruptcy, short-sale bans, vectorisation, metrics, causality (incl. planted-leak detection) |
| `reports/` | `REPORT.md`, figures and tables |

## Running experiments

```bash
.venv/Scripts/python scripts/run_baselines.py --universe spy etf9 eq10_2000 fut4 syn:gbm syn:ar1 syn:regime
.venv/Scripts/python scripts/sanity_check_agents.py              # CartPole / Pendulum
.venv/Scripts/python scripts/run_experiments.py controls gamma main rewards hacking leakage features execution randomized portfolio futures equities
.venv/Scripts/python scripts/run_hpo.py search
.venv/Scripts/python scripts/run_hpo.py evaluate
.venv/Scripts/python scripts/run_robustness.py main_spy          # costs x0-x4, AUM $1M-$10B, fill timing
.venv/Scripts/python scripts/run_holdout.py rl                   # touches the 2019+ holdout; logged in runs/holdout_ledger.jsonl
.venv/Scripts/python scripts/run_holdout.py baselines
.venv/Scripts/python scripts/build_report.py                     # figures + tables for the report
.venv/Scripts/python scripts/build_report_html.py                # reports/report.html (self-contained)
```

The full set of experiments takes several hours on an 8-core laptop.

A finished run is reused only if its stored spec matches exactly and it was scored under the current environment
version (`rlt.ENV_VERSION`). Stale runs are moved to `runs/_archive/`, never overwritten.

After a change to simulated P&L, `scripts/reevaluate_runs.py <experiment>...` re-scores saved checkpoints without
retraining. `scripts/summarize.py`, `quick_tables.py` and `quick_multiasset.py` print headline tables for any
experiment. `scripts/scan_flat.py` finds evaluations of policies that never trade.

## Dashboard

```bash
.venv/Scripts/python rlt/dashboard/server.py --port 8795
```

Open http://127.0.0.1:8795.

* **Experiment view:** the out-of-sample Sharpe of every configuration (the seed ensemble, with a 95% bootstrap
  interval) against buy & hold, a fold-by-algorithm heatmap, and a runs table.
* **Comparing runs:** select up to 8 runs to see their training curves (any logged metric) and test equity curves.
* **Live updates:** the page refreshes every 5 seconds while runs are training.

## Conventions that matter

* **Timing.** Decision at the close of day t, fill at the open of day t+1, mark at the close of t+1. Baselines go
  through the same environment and pay the same costs.
* **No look-ahead.** Features at row t use bars 0..t only, and anything fitted (HMM, scaler) is fitted on the
  training window. `tests/test_causality.py` deletes the future and checks that nothing in the past changes.
* **Walk-forward folds.** Expanding training window, 2-year validation, 2-year test, 5-session embargo. The
  2019-01-01 onward holdout is used once, at the end, and every access is logged.
* **Provenance.** Every run's directory records the exact config, derived seed, environment version, code
  fingerprint and package versions.
