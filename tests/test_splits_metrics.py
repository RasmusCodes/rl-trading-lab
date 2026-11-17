"""Fold boundaries, episode sampling bounds, metrics and reward formulas."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt.data.splits import holdout_fold, walk_forward  # noqa: E402
from rlt.data.synthetic import make_synthetic  # noqa: E402
from rlt.env import EnvConfig, RewardFn, StepInfo, TradingEnv  # noqa: E402
from rlt.evaluation.metrics import deflated_sharpe, drawdown_stats, perf_stats, psr  # noqa: E402


@pytest.fixture(scope="module")
def dates():
    return pd.bdate_range("1993-01-29", "2026-11-12")


def test_folds_ordered_with_embargo_and_never_touch_holdout(dates):
    folds = walk_forward(dates, 2007, 2018, test_years=2, val_years=2, embargo=5, min_row=260)
    assert [f.name for f in folds] == ["wf1_2007", "wf2_2009", "wf3_2011", "wf4_2013", "wf5_2015", "wf6_2017"]
    hold = dates.searchsorted(pd.Timestamp("2019-01-01"))
    for f in folds:
        (a0, a1), (v0, v1), (t0, t1) = f.train, f.val, f.test
        assert a0 >= 260 and a0 < a1 < v0 <= v1 < t0 <= t1
        assert v0 - a1 >= 6 and t0 - v1 >= 6  # 5-session embargo
        assert t1 < hold
        assert dates[t0].year == int(f.name.split("_")[1])
    # consecutive test blocks tile 2007-2018 without gaps
    for f, g in zip(folds[:-1], folds[1:]):
        assert g.test[0] == f.test[1] + 1
    h = holdout_fold(dates, "2019-01-01", min_row=260)
    assert dates[h.test[0]] >= pd.Timestamp("2019-01-01") and h.val[1] < h.test[0] - 5


def test_training_env_never_reads_beyond_its_window():
    md = make_synthetic("gbm", T=800, seed=1)
    feats = np.random.default_rng(0).normal(size=(md.T, 1, 3)).astype(np.float32)
    i0, i1 = 300, 500
    env = TradingEnv(md, feats, EnvConfig(episode_length=40), (i0, i1), num_envs=8, seed=0)
    env.reset()
    max_obs_row = max_fill_row = 0
    for _ in range(400):
        max_obs_row = max(max_obs_row, int(env.t.max()))
        env.step(np.random.default_rng(1).uniform(-1, 1, (8, 1)))
        max_fill_row = max(max_fill_row, int(env.t.max()))  # rows whose prices entered the step
    assert max_obs_row <= i1  # observations come from decision rows inside the window
    assert max_fill_row <= i1 + 1  # the last decision is filled/marked one bar later, still before any embargo


def test_metrics_on_known_series():
    r = np.array([0.01, -0.02, 0.03, -0.01] * 63)
    st = perf_stats(r, np.zeros_like(r))
    assert st["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    mdd, dur = drawdown_stats(np.array([0.1, -0.5, 0.2, 0.0]))
    assert mdd == pytest.approx(-0.5)
    assert dur == 3
    assert psr(0.1, 500, 0.0, 3.0) > psr(0.05, 500, 0.0, 3.0) > 0.5
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, 2000)
    assert deflated_sharpe(x, 50, rng.normal(0, 0.5, 50)) <= psr(x.mean() / x.std(ddof=1), len(x), 0.0, 3.0)


def test_flat_book_has_zero_sharpe_not_float_noise():
    rf = np.full(500, 0.04 / 252)
    ret = rf * (1 + np.random.default_rng(0).normal(0, 1e-12, 500))  # cash book: ret == rf up to rounding
    st = perf_stats(ret, rf)
    assert st["flat"] and st["sharpe"] == 0.0 and st["sortino"] == 0.0


def _info(ret, rf=0.0, dd=0.0, dd_prev=0.0, w=1.0, cost=0.0, to=0.0):
    a = lambda v: np.array([v], float)  # noqa: E731
    return StepInfo(ret=a(ret), log_ret=a(np.log1p(ret)), rf=a(rf), pnl=a(ret * 100.0), initial=a(100.0),
                    cost_frac=a(cost), turnover=a(to), gross=a(abs(w)), weights=a(w)[:, None], dd=a(dd),
                    dd_prev=a(dd_prev), mkt_vol=a(0.01))


def test_differential_sharpe_matches_formula():
    fn = RewardFn({"type": "diff_sharpe", "eta": 0.1, "var_floor": 0.0, "scale": 1.0}, 1)
    A, B = 0.0, 1e-4
    for R in (0.01, -0.02, 0.005):
        d = fn(_info(R))[0]
        dA, dB = R - A, R**2 - B
        assert d == pytest.approx((B * dA - 0.5 * A * dB) / (B - A**2) ** 1.5)
        A, B = A + 0.1 * dA, B + 0.1 * dB


def test_penalties_and_pnl_reward():
    fn = RewardFn({"type": "pnl", "scale": 1.0, "penalties": {"drawdown": 2.0, "inventory": 0.5, "cost": 1.0}}, 1)
    r = fn(_info(0.01, dd=0.05, dd_prev=0.02, w=0.8, cost=0.001))[0]
    assert r == pytest.approx(0.01 - 2.0 * 0.03 - 0.5 * 0.8 - 1.0 * 0.001)
