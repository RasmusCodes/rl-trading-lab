"""Run any policy (RL agent or baseline) through the environment and collect a daily ledger.

Baselines and agents go through the *same* environment, so they pay the same spread, impact, borrow and
financing and fill at the same next-open prices. A baseline that is only evaluated on close-to-close returns
without costs is not a baseline.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..env.trading_env import EnvConfig, TradingEnv
from .metrics import perf_stats


class WeightPolicy:
    """Policy defined by a precomputed (T, N) table of target weights; row t is used at decision row t.

    NaN means "hold the current position" (lets calendar rebalancers avoid daily drift trades).
    """

    def __init__(self, weights: np.ndarray, name: str = "weights"):
        self.W = np.asarray(weights, float)
        self.name = name
        # latest target known at each row (forward fill uses past rows only): lets a calendar rebalancer that
        # starts an evaluation flat adopt its current target on day one instead of idling until month-end
        self.W_ff = pd.DataFrame(self.W).ffill().to_numpy()

    def reset(self, num_envs: int) -> None:
        pass

    def act(self, obs, env: TradingEnv, deterministic: bool = True):
        w = self.W[env.t[:, None], env.assets]
        if np.isnan(w).any():
            flat = (np.abs(env.q).sum(1) == 0)[:, None]
            w = np.where(np.isnan(w) & flat, self.W_ff[env.t[:, None], env.assets], w)
        return env.weights_to_action(w)


def evaluation_env(md, feats, cfg: EnvConfig, row_range, assets=None, ready=None, **overrides) -> TradingEnv:
    """Deterministic, full-window evaluation env: one episode per instrument (pool) or one portfolio.

    Risk-limit stops and bankruptcy terminations are training devices. In evaluation the window always runs to
    its end: a wiped-out book is liquidated and books zero returns afterwards, so a strategy that blows up is
    charged for every remaining day instead of having its record truncated (which would inflate |Sharpe|).
    """
    base = dict(episode_length=None, randomize={}, terminate_drawdown=None, bankruptcy_level=-1.0, terminal_penalty=0.0)
    base.update(overrides)
    ecfg = replace(cfg, **base)
    n_envs = len(assets) if (assets is not None and cfg.mode == "pool") else (md.N if cfg.mode == "pool" else 1)
    return TradingEnv(md, feats, ecfg, row_range, num_envs=n_envs, seed=0, random_start=False, assets=assets,
                      ready=ready, auto_reset=False, record=True)


def run_policy(env: TradingEnv, policy, deterministic: bool = True) -> list[pd.DataFrame]:
    """Step until every episode ends; return one ledger DataFrame per env (indexed by fill date)."""
    obs = env.reset()
    if hasattr(policy, "reset"):
        policy.reset(env.B)
    steps = 0
    while env.active.any():
        act = policy.act(obs, env=env, deterministic=deterministic)
        obs, _, term, trunc, _ = env.step(act)
        steps += 1
        if steps > env.T + 5:  # pragma: no cover
            raise RuntimeError("evaluation did not terminate")
    return ledgers_from_env(env)


def ledgers_from_env(env: TradingEnv) -> list[pd.DataFrame]:
    md = env.md
    out = []
    L = env.ledger
    for b in range(env.B):
        rows = np.array([s["row"][b] for s in L])
        sym = [md.symbols[a] for a in env.assets[b]]
        df = pd.DataFrame({
            "date": md.dates[rows],
            "equity": [s["equity"][b] for s in L],
            "ret": [s["ret"][b] for s in L],
            "rf": [s["rf"][b] for s in L],
            "cost": [s["cost"][b] / (s["equity"][b] / (1 + s["ret"][b])) for s in L],
            "turnover": [s["turnover"][b] for s in L],
            "gross": [s["gross"][b] for s in L],
            "reward": [s["reward"][b] for s in L],
        })
        W = np.array([s["weights"][b] for s in L])
        TW = np.array([s["target"][b] for s in L])
        for j, s_ in enumerate(sym):
            df[f"w_{s_}"] = W[:, j]
            df[f"tw_{s_}"] = TW[:, j]
        df["net"] = W.sum(1)
        df.attrs["symbols"] = sym
        out.append(df.set_index("date"))
    return out


def ledger_stats(df: pd.DataFrame, bench: pd.DataFrame | None = None, ci: bool = False) -> dict:
    bench_ret = None
    if bench is not None:
        bench_ret = bench["ret"].reindex(df.index).fillna(0.0).to_numpy()
    return perf_stats(df["ret"].to_numpy(), df["rf"].to_numpy(), df["turnover"].to_numpy(), df["cost"].to_numpy(),
                      df["gross"].to_numpy(), df["net"].to_numpy(), bench_ret, ci=ci)
