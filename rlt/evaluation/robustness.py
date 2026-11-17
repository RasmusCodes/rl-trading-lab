"""Robustness re-evaluation of trained checkpoints and baselines.

* capacity     re-run the policy on its test window at AUM = 1e6 ... 1e10. Impact grows with sqrt(trade / ADV)
               and fills are capped at 10% of ADV, so returns decay with size. Capacity is reported as the
               AUM at which the strategy's Sharpe falls to half its $1M value, and as the AUM at which it stops
               beating buy & hold at the same AUM.
* costs        cost multipliers 0x ... 4x (spread, slippage, impact and commission scaled together)
* execution    next_open (default) vs next_close (half a session later) vs same_close (unrealistic)
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..agents import EnvSpec, make_agent
from ..evaluation.backtest import WeightPolicy, ledger_stats, run_policy
from ..utils.io import load_json
from ..training.runner import RunSpec, build_env_config, get_context

AUMS = [1e6, 1e7, 1e8, 1e9, 1e10]
COST_MULTS = [0.0, 0.5, 1.0, 2.0, 4.0]


def load_agent(run_dir: Path, which: str = "best"):
    cfg = load_json(Path(run_dir) / "config.json")
    spec = RunSpec(**{k: (tuple(v) if k == "leak" else v) for k, v in cfg["spec"].items()})
    ctx = get_context(spec.universe, spec.fold, spec.features, tuple(spec.leak), spec.holdout, spec.train_start,
                      purpose=f"reload:{Path(run_dir).parent.name}/{Path(run_dir).name}")
    ecfg = build_env_config(spec)
    env = ctx.eval_env(ecfg, "test", assets=spec.eval_assets)
    espec = EnvSpec.from_env(env, ctx.obs_names(ecfg))
    ck = torch.load(Path(run_dir) / "model.pt", weights_only=False)
    agent = make_agent(spec.algo, espec, ck["params"], seed=0)
    if spec.algo == "rrl":
        agent._build(env)
    agent.set_state(ck[which] if ck.get(which) is not None else ck["last"])
    return agent, spec, ctx, ecfg


def _stats(ctx, ecfg, policy, assets, split="test", **env_over):
    env = ctx.eval_env(replace(ecfg, **env_over), split, assets=assets)
    leds = run_policy(env, policy)
    per = [ledger_stats(led) for led in leds]
    return {k: float(np.mean([p[k] for p in per if p.get(k) is not None])) for k in
            ("sharpe", "cagr", "turnover", "cost_drag", "max_drawdown", "avg_gross")}


def sweep_run(run_dir: Path) -> pd.DataFrame:
    agent, spec, ctx, ecfg = load_agent(run_dir)
    rows = []
    for aum in AUMS:
        rows.append({"kind": "aum", "x": aum, **_stats(ctx, ecfg, agent, spec.eval_assets, initial_capital=aum)})
    for m in COST_MULTS:
        rows.append({"kind": "cost_mult", "x": m,
                     **_stats(ctx, ecfg, agent, spec.eval_assets, costs=replace(ecfg.costs, multiplier=m))})
    for ex in ("next_open", "next_close", "same_close"):
        rows.append({"kind": "execution", "x": ex, **_stats(ctx, ecfg, agent, spec.eval_assets, execution=ex)})
    df = pd.DataFrame(rows)
    df["run"] = Path(run_dir).name
    df["algo"], df["fold"], df["seed"], df["tag"] = spec.algo, spec.fold, spec.seed, spec.tag
    return df


def sweep_weights(universe: str, fold_name: str, name: str, W: np.ndarray, features: str = "compact",
                  mode: str | None = None) -> pd.DataFrame:
    from ..training.context import env_config

    ctx = get_context(universe, fold_name, features, purpose=f"robustness:{name}")
    ecfg = env_config(universe, mode)
    pol = WeightPolicy(W, name)
    rows = []
    for aum in AUMS:
        rows.append({"kind": "aum", "x": aum, **_stats(ctx, ecfg, pol, None, initial_capital=aum)})
    for m in COST_MULTS:
        rows.append({"kind": "cost_mult", "x": m, **_stats(ctx, ecfg, pol, None, costs=replace(ecfg.costs, multiplier=m))})
    for ex in ("next_open", "next_close", "same_close"):
        rows.append({"kind": "execution", "x": ex, **_stats(ctx, ecfg, pol, None, execution=ex)})
    df = pd.DataFrame(rows)
    df["run"], df["algo"], df["fold"] = name, name, fold_name
    return df


def capacity(df_aum: pd.DataFrame, bench_aum: pd.DataFrame | None = None) -> dict:
    """AUM where Sharpe halves vs $1M (log-interpolated), and where it drops below the benchmark."""
    d = df_aum.sort_values("x")
    s0 = d["sharpe"].iloc[0]
    out = {"sharpe_1m": float(s0)}
    half = None
    if s0 > 0:
        for (a0, v0), (a1, v1) in zip(d[["x", "sharpe"]].values[:-1], d[["x", "sharpe"]].values[1:]):
            if v0 >= s0 / 2 > v1:
                t = (v0 - s0 / 2) / (v0 - v1)
                half = float(10 ** (np.log10(a0) + t * (np.log10(a1) - np.log10(a0))))
                break
        if half is None and d["sharpe"].iloc[-1] >= s0 / 2:
            half = float("inf")
    out["aum_half_sharpe"] = half
    if bench_aum is not None:
        b = bench_aum.sort_values("x")
        diff = d["sharpe"].values - b["sharpe"].values
        below = np.where(diff < 0)[0]
        out["aum_below_bench"] = float(d["x"].values[below[0]]) if len(below) else float("inf")
    return out
