"""Research experiments (each writes runs/<experiment>/...; re-running skips finished runs).

    python scripts/run_experiments.py controls     # RL on synthetic negative/positive controls
    python scripts/run_experiments.py main         # 9 algorithms x 6 walk-forward folds x 3 seeds on SPY
    python scripts/run_experiments.py features     # state-representation ablation (PPO)
    python scripts/run_experiments.py rewards      # reward-function study (PPO)
    python scripts/run_experiments.py hacking      # reward-hacking / unstable-objective demonstrations
    python scripts/run_experiments.py leakage      # planted look-ahead: does RL find it, does the test catch it
    python scripts/run_experiments.py execution    # same-close fills vs next-open (PPO)
    python scripts/run_experiments.py randomized   # domain randomisation / pooling / bootstrap training
    python scripts/run_experiments.py portfolio    # 9-ETF portfolio, continuous agents
    python scripts/run_experiments.py futures      # 4 index/bond futures, continuous agents
    python scripts/run_experiments.py equities     # 10 single stocks (end-2000 largest), pooled agents
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt.training.runner import RunSpec, run_many  # noqa: E402

SPY_FOLDS = [f"wf{k}_{y}" for k, y in zip(range(1, 7), range(2007, 2019, 2))]
PORT_FOLDS = [f"wf{k}_{y}" for k, y in zip(range(1, 5), range(2011, 2019, 2))]
SYN_FOLD = "wf1_2018"
ALL_ALGOS = ["qlearning", "dqn", "ddqn", "a2c", "ppo", "sac", "td3", "rrl", "ppo_lstm"]
CONT_ALGOS = ["ppo", "a2c", "sac", "td3", "rrl", "ppo_lstm"]


def S(**kw):
    return RunSpec(**kw)


def controls(steps):
    out = []
    for u in ["syn:gbm", "syn:ar1", "syn:regime"]:
        for a in ["ppo", "dqn", "sac", "rrl", "a2c", "ppo_lstm"]:
            for s in range(3):
                out.append(S(experiment="controls", universe=u, algo=a, fold=SYN_FOLD, seed=s, total_steps=steps,
                             tag=u.split(":")[1]))
    return out


def gamma(steps):
    """Discount-factor sweep on the planted-edge controls (decides the default gamma, never touches real data)."""
    return [S(experiment="gamma_controls", universe=u, algo=a, fold=SYN_FOLD, seed=s, total_steps=steps,
              eval_every=10_000, tag=f"{u.split(':')[1]}_g{g:g}", algo_params={"gamma": g})
            for u in ["syn:ar1", "syn:regime"] for a in ["dqn", "sac", "ppo"] for g in (0.0, 0.5, 0.9)
            for s in range(2)]


def main_exp(steps):
    return [S(experiment="main_spy", universe="spy", algo=a, fold=f, seed=s, total_steps=steps, eval_every=10_000)
            for a in ALL_ALGOS for f in SPY_FOLDS for s in range(3)]


def features(steps):
    from rlt.data.features import FEATURE_SETS

    out = []
    # base = returns + volatility; each rv_* adds one group; plus minimal / full / compact and two layout variants
    variants = {k: {"features": k} for k in FEATURE_SETS if k != "returns"}
    variants["compact_noposition"] = {"features": "compact", "env": {"portfolio_state": False}}
    variants["compact_window5"] = {"features": "compact", "env": {"obs_window": 5}}
    for tag, v in variants.items():
        for f in SPY_FOLDS:
            for s in range(2):
                out.append(S(experiment="features_spy", universe="spy", algo="ppo", fold=f, seed=s,
                             total_steps=steps, tag=tag, features=v["features"], env=v.get("env", {})))
    return out


REWARDS = {
    "pnl": {"type": "pnl"},
    "log_return": {"type": "log_return"},
    "excess_return": {"type": "excess_return"},
    "vol_scaled": {"type": "vol_scaled"},
    "risk_adjusted": {"type": "risk_adjusted"},
    "diff_sharpe": {"type": "diff_sharpe"},
    "diff_downside": {"type": "diff_downside"},
    "mv_utility": {"type": "mv_utility", "kappa": 5.0},
    "log+cost_pen": {"type": "log_return", "penalties": {"cost": 1.0}},
    "log+turnover_pen": {"type": "log_return", "penalties": {"turnover": 2e-4}},
    "log+inventory_pen": {"type": "log_return", "penalties": {"inventory": 2e-4}},
    "log+drawdown_pen": {"type": "log_return", "penalties": {"drawdown": 1.0}},
}


def rewards(steps):
    return [S(experiment="rewards_spy", universe="spy", algo="ppo", fold=f, seed=s, total_steps=steps, tag=tag,
              env={"reward": r}) for tag, r in REWARDS.items() for f in SPY_FOLDS for s in range(2)]


def hacking(steps):
    folds = ["wf1_2007", "wf3_2011", "wf5_2015"]
    V = {}
    for lam in (0.0, 1.0, 5.0, 20.0):
        V[f"dd_pen_{lam:g}"] = {"reward": {"type": "log_return", "penalties": {"drawdown": lam}}}
    for lam in (2e-4, 5e-4, 2e-3):
        V[f"inv_pen_{lam:g}"] = {"reward": {"type": "log_return", "penalties": {"inventory": lam}}}
    V["dsr_raw_nofloor"] = {"reward": {"type": "diff_sharpe", "use_excess": False, "var_floor": 0.0}}
    V["dsr_excess_floor"] = {"reward": {"type": "diff_sharpe"}}
    base_dd = {"type": "log_return", "penalties": {"drawdown": 0.05, "drawdown_mode": "level"}}
    V["ddlevel_term_nopen"] = {"reward": base_dd, "terminate_drawdown": 0.10, "terminal_penalty": 0.0}
    V["ddlevel_term_pen"] = {"reward": base_dd, "terminate_drawdown": 0.10, "terminal_penalty": -50.0}
    V["ddlevel_noterm"] = {"reward": base_dd}
    V["lev3_pnl"] = {"reward": {"type": "pnl"}, "max_position": 3.0, "max_gross": 3.0, "max_net": 3.0}
    V["lev3_log"] = {"reward": {"type": "log_return"}, "max_position": 3.0, "max_gross": 3.0, "max_net": 3.0}
    V["riskadj_nofloor"] = {"reward": {"type": "risk_adjusted", "var_floor": 0.0}}
    V["riskadj_floor"] = {"reward": {"type": "risk_adjusted"}}
    return [S(experiment="hacking_spy", universe="spy", algo="ppo", fold=f, seed=s, total_steps=steps, tag=tag,
              env=env) for tag, env in V.items() for f in folds for s in range(2)]


def leakage(steps):
    out = []
    for leak in [(), ("future_return",), ("global_zscore",), ("smoothed_regime",), ("centered_ma",)]:
        tag = leak[0] if leak else "none"
        for s in range(2):
            out.append(S(experiment="leakage", universe="syn:gbm", algo="ppo", fold=SYN_FOLD, seed=s,
                         total_steps=steps, tag=f"gbm_{tag}", leak=leak))
            for f in ["wf1_2007", "wf3_2011", "wf5_2015"]:
                out.append(S(experiment="leakage", universe="spy", algo="ppo", fold=f, seed=s, total_steps=steps,
                             tag=f"spy_{tag}", leak=leak))
    # capability check: can each algorithm family exploit a blatant oracle feature at all?
    for a in ["dqn", "ddqn", "sac", "td3", "a2c", "rrl", "ppo_lstm", "qlearning"]:
        for s in range(2):
            out.append(S(experiment="leakage", universe="syn:gbm", algo=a, fold=SYN_FOLD, seed=s, total_steps=steps,
                         tag="gbm_future_return", leak=("future_return",),
                         algo_params={"features": ("leak_future_return", "ret_21d", "vol_ratio")} if a == "qlearning" else {}))
    return out


def execution(steps):
    out = []
    for ex in ("next_open", "same_close", "next_close"):
        for f in SPY_FOLDS:
            for s in range(2):
                out.append(S(experiment="execution_spy", universe="spy", algo="ppo", fold=f, seed=s,
                             total_steps=steps, tag=ex, env={"execution": ex}))
    return out


def randomized(steps):
    V = {
        "base": dict(),
        # pooled19 can only train from 2001 (selection date of its stocks); this is the like-for-like SPY control
        "base_from2001": dict(train_start="2001-01-02"),
        "cost_rand": dict(env={"randomize": {"cost_mult": (0.5, 3.0), "slippage_bps": (0.0, 3.0)}}),
        "obs_noise": dict(env={"randomize": {"obs_noise": 0.1}}),
        "bootstrap": dict(train_bootstrap={"paths": 8, "length": 3000, "mean_block": 20}),
        "pooled19": dict(universe="pool19", eval_assets=["SPY"]),
        "pooled19_costrand": dict(universe="pool19", eval_assets=["SPY"],
                                  env={"randomize": {"cost_mult": (0.5, 3.0), "slippage_bps": (0.0, 3.0)}}),
    }
    out = []
    for tag, v in V.items():
        for f in SPY_FOLDS:
            for s in range(2):
                kw = dict(experiment="randomized_spy", universe=v.get("universe", "spy"), algo="ppo", fold=f,
                          seed=s, total_steps=steps, tag=tag, env=v.get("env", {}),
                          eval_assets=v.get("eval_assets"), train_bootstrap=v.get("train_bootstrap"),
                          train_start=v.get("train_start"))
                out.append(S(**kw))
    return out


def portfolio(steps):
    return [S(experiment="portfolio_etf9", universe="etf9", algo=a, fold=f, seed=s, total_steps=steps, num_envs=16)
            for a in CONT_ALGOS for f in PORT_FOLDS for s in range(2)]


def futures(steps):
    return [S(experiment="futures_fut4", universe="fut4", algo=a, fold=f, seed=s, total_steps=steps,
              env={"initial_capital": 5_000_000.0})
            for a in ["ppo", "sac", "rrl", "a2c"] for f in PORT_FOLDS for s in range(2)]


def equities(steps):
    return [S(experiment="equities_eq10", universe="eq10_2000", algo=a, fold=f, seed=s, total_steps=steps)
            for a in ["ppo", "dqn", "rrl"] for f in SPY_FOLDS for s in range(2)]


EXPERIMENTS = {"controls": controls, "gamma": gamma, "main": main_exp, "features": features, "rewards": rewards,
               "hacking": hacking, "leakage": leakage, "execution": execution, "randomized": randomized,
               "portfolio": portfolio, "futures": futures, "equities": equities}

DEFAULT_STEPS = {"controls": 200_000, "gamma": 100_000, "main": 200_000, "features": 100_000, "rewards": 100_000,
                 "hacking": 100_000, "leakage": 100_000, "execution": 100_000, "randomized": 150_000,
                 "portfolio": 200_000, "futures": 200_000, "equities": 200_000}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="+", choices=list(EXPERIMENTS))
    ap.add_argument("--steps", type=int, default=None, help="override every experiment's default budget")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    specs = []
    for e in args.experiment:  # submission order = priority order
        specs += EXPERIMENTS[e](args.steps or DEFAULT_STEPS[e])
    run_many(specs, workers=args.workers)
