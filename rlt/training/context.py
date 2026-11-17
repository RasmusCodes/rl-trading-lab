"""Per-fold data preparation shared by agents and baselines.

A `FoldContext` owns everything that must be fitted on the fold's training window: the regime model inside
the features (fit_end = last training row) and the feature scaler. Validation/test rows are transformed with
those training-only statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from ..data.features import FeatureScaler, compute_features
from ..data.market import MarketData, load_market
from ..data.splits import Fold, holdout_fold, walk_forward
from ..data.synthetic import make_synthetic
from ..env.trading_env import EnvConfig, TradingEnv
from ..evaluation.backtest import evaluation_env

# development walk-forward plans (test blocks); the holdout starts 2019-01-01 for every real universe
FOLD_PLANS = {
    "spy": dict(first_test_year=2007, last_test_year=2018, test_years=2, val_years=2),
    "eq10_2000": dict(first_test_year=2007, last_test_year=2018, test_years=2, val_years=2),
    "pool19": dict(first_test_year=2007, last_test_year=2018, test_years=2, val_years=2),
    "etf9": dict(first_test_year=2011, last_test_year=2018, test_years=2, val_years=2),
    "fut4": dict(first_test_year=2011, last_test_year=2018, test_years=2, val_years=2),
}
SYNTH_PLAN = dict(first_test_year=2018, last_test_year=2021, test_years=4, val_years=3)
WARMUP_ROWS = 260  # longest feature window (252) plus slack


@lru_cache(maxsize=16)
def get_market(universe: str, seed: int = 0) -> MarketData:
    if universe.startswith("syn:"):
        kind = universe.split(":", 1)[1]
        return make_synthetic(kind, T=6000, seed=seed)
    return load_market(universe)


PORTFOLIO_UNIVERSES = {"etf9", "fut4"}
# universes whose members were chosen by a 31-Dec-2000 criterion: no training before the selection date
SELECTION_DATE_START = {"eq10_2000": "2001-01-02", "pool19": "2001-01-02"}


def _min_row(md: MarketData, universe: str) -> int:
    """Portfolio universes need every instrument; pooled ones start when the first instrument is live
    (later listings simply join the episode sampler once their own features are ready)."""
    if universe in PORTFOLIO_UNIVERSES:
        row = md.first_valid_index() + WARMUP_ROWS
    else:
        any_valid = np.where(md.valid.any(axis=1))[0]
        row = int(any_valid[0]) + WARMUP_ROWS
    if universe in SELECTION_DATE_START:
        row = max(row, md.index_of(SELECTION_DATE_START[universe]))
    return row


def get_folds(md: MarketData, universe: str) -> list[Fold]:
    plan = SYNTH_PLAN if universe.startswith("syn:") else FOLD_PLANS[universe]
    return walk_forward(md.dates, min_row=_min_row(md, universe), **plan)


def get_holdout(md: MarketData, universe: str) -> Fold:
    return holdout_fold(md.dates, "2019-01-01", val_years=2, min_row=_min_row(md, universe))


@dataclass
class FoldContext:
    md: MarketData
    fold: Fold
    feature_names: list
    leak: tuple = ()

    def __post_init__(self):
        fs = compute_features(self.md, self.feature_names, fit_end=self.fold.train[1], leak=self.leak)
        self.fs = fs
        self.scaler = FeatureScaler().fit(fs, self.fold.train[0], self.fold.train[1])
        self.X = self.scaler.transform(fs.values)
        self.ready = fs.ready

    def rows(self, split: str) -> tuple:
        return getattr(self.fold, split)

    def train_env(self, cfg: EnvConfig, num_envs: int, seed: int, assets=None) -> TradingEnv:
        return TradingEnv(self.md, self.X, cfg, self.fold.train, num_envs=num_envs, seed=seed, random_start=True,
                          assets=assets, ready=self.ready, auto_reset=True)

    def eval_env(self, cfg: EnvConfig, split: str, assets=None, **overrides) -> TradingEnv:
        return evaluation_env(self.md, self.X, cfg, self.rows(split), assets=assets, ready=self.ready, **overrides)

    def obs_names(self, cfg: EnvConfig) -> list:
        n = self.md.N if cfg.mode == "portfolio" else 1
        names = []
        for k in range(cfg.obs_window):
            for a in range(n):
                for f in self.fs.names:
                    names.append(f"{f}@a{a}@lag{cfg.obs_window - 1 - k}")
        if cfg.portfolio_state:
            names += [f"w@a{a}" for a in range(n)] + ["gross", "net", "cash", "drawdown"]
        return names


def env_config(universe: str, mode: str | None = None, **kw) -> EnvConfig:
    """Default environment per universe (overridable)."""
    if mode is None:
        mode = "portfolio" if universe in PORTFOLIO_UNIVERSES else "pool"
    base = EnvConfig(mode=mode)
    if mode == "portfolio":
        base = replace(base, max_position=1.0, max_gross=1.0, max_net=1.0)
    return replace(base, **kw)
