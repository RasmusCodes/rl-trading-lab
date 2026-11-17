"""Accounting, timing and cost tests for the trading environment."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt.data.market import MarketData  # noqa: E402
from rlt.data.universe import InstrumentSpec  # noqa: E402
from rlt.env import CostConfig, EnvConfig, TradingEnv  # noqa: E402
from rlt.evaluation import WeightPolicy, evaluation_env, run_policy  # noqa: E402

FREE = CostConfig(multiplier=1.0, slippage_bps=0.0, spread=False, impact=False, commissions=False, borrow=False,
                  participation_cap=0.0, financing_spread_bps=0.0)


def toy_market(T=60, seed=0, future=False, rf=0.0):
    rng = np.random.default_rng(seed)
    lc = np.log(100) + np.cumsum(rng.normal(0, 0.01, T))
    C = np.exp(lc)
    O = C * np.exp(rng.normal(0, 0.003, T))
    H = np.maximum(O, C) * 1.002
    L = np.minimum(O, C) * 0.998
    spec = (InstrumentSpec("FUT", "future", "x", multiplier=50.0, tick_size=0.25, commission_per_contract=2.0,
                           min_half_spread_bps=0.2) if future else
            InstrumentSpec("ETF", "etf", "x", commission_bps=0.3, min_half_spread_bps=0.5))
    col = lambda x: x[:, None]  # noqa: E731
    return MarketData(dates=pd.bdate_range("2020-01-01", periods=T), symbols=[spec.symbol], specs=[spec],
                      open=col(O), high=col(H), low=col(L), close=col(C), close_raw=col(C.copy()),
                      volume=col(np.full(T, 1e7)), rf=np.full(T, rf), valid=np.ones((T, 1), bool),
                      roll=np.zeros((T, 1), bool))


def run_weights(md, W, cfg, i0=5, i1=None):
    i1 = i1 or md.T - 2
    feats = np.zeros((md.T, md.N, 1), np.float32)
    env = evaluation_env(md, feats, cfg, (i0, i1))
    return run_policy(env, WeightPolicy(W))[0]


def test_long_only_frictionless_equals_price_return():
    md = toy_market()
    cfg = EnvConfig(costs=FREE)
    led = run_weights(md, np.ones((md.T, 1)), cfg)
    # decision at row 5 fills at open of row 6; held to the close of the last row
    first_fill = 6
    expected = md.close[-1, 0] / md.open[first_fill, 0]
    assert led["equity"].iloc[-1] / cfg.initial_capital == pytest.approx(expected, rel=1e-10)


def test_short_frictionless():
    md = toy_market(seed=3)
    cfg = EnvConfig(costs=FREE)
    led = run_weights(md, -np.ones((md.T, 1)), cfg, i0=5, i1=5)  # one decision, one day
    P, Cn = md.open[6, 0], md.close[6, 0]
    assert led["equity"].iloc[-1] / cfg.initial_capital == pytest.approx(1 - (Cn / P - 1), rel=1e-10)


def test_overnight_gap_belongs_to_old_position():
    md = toy_market(seed=5)
    cfg = EnvConfig(costs=FREE)
    W = np.zeros((md.T, 1))
    W[5] = 1.0  # long decided at row 5 (filled open 6), flat decided at row 6 (filled open 7)
    led = run_weights(md, W, cfg, i0=5, i1=7)
    e6 = md.close[6, 0] / md.open[6, 0]
    e7 = e6 * md.open[7, 0] / md.close[6, 0]  # carried the overnight gap 6->7 then sold at the open
    assert led["equity"].iloc[0] / cfg.initial_capital == pytest.approx(e6, rel=1e-10)
    assert led["equity"].iloc[1] / cfg.initial_capital == pytest.approx(e7, rel=1e-10)


def test_same_close_execution_uses_decision_close():
    md = toy_market(seed=7)
    cfg = EnvConfig(costs=FREE, execution="same_close")
    led = run_weights(md, np.ones((md.T, 1)), cfg, i0=5, i1=6)
    assert led["equity"].iloc[0] / cfg.initial_capital == pytest.approx(md.close[6, 0] / md.close[5, 0], rel=1e-10)


def test_spread_and_commission_charged_once_per_trade():
    md = toy_market(seed=9)
    costs = replace(FREE, spread=True, commissions=True)
    cfg = EnvConfig(costs=costs)
    led = run_weights(md, np.ones((md.T, 1)), cfg, i0=5, i1=6)
    P, Cn = md.open[6, 0], md.close[6, 0]
    E0 = cfg.initial_capital
    hs = max(0.5e-4, 0.5 * 0.01 / md.close_raw[5, 0])
    cost = E0 * (hs + 0.3e-4)
    expected = E0 - cost + (E0 / P) * (Cn - P)
    assert led["equity"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_cash_interest_and_borrow():
    md = toy_market(seed=11, rf=0.0001)
    cfg = EnvConfig(costs=replace(FREE, borrow=True))
    # flat: equity grows at rf exactly
    led = run_weights(md, np.zeros((md.T, 1)), cfg, i0=5, i1=15)
    assert led["equity"].iloc[-1] / cfg.initial_capital == pytest.approx(1.0001**11, rel=1e-10)
    # short 100%: cash = 2E earns rf, borrow charged on the short notional
    led = run_weights(md, -np.ones((md.T, 1)), cfg, i0=5, i1=5)
    E0, P, Cn = cfg.initial_capital, md.open[6, 0], md.close[6, 0]
    borrow = E0 * md.specs[0].borrow_bps_annual / 1e4 / 252
    haircut = E0 * cfg.costs.short_rebate_haircut_bps / 1e4 / 252  # short proceeds earn rf minus the haircut
    expected = E0 + 2 * E0 * 0.0001 - borrow - haircut - (E0 / P) * (Cn - P)
    assert led["equity"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_futures_contract_rounding_and_no_cash_outlay():
    md = toy_market(seed=13, future=True)
    cfg = EnvConfig(costs=FREE, initial_capital=12_000.0)  # one contract is ~$5,000 of notional
    led = run_weights(md, np.ones((md.T, 1)), cfg, i0=5, i1=5)
    contracts = round(12_000.0 / (md.open[6, 0] * 50))
    assert contracts == 2
    pnl = contracts * 50 * (md.close[6, 0] - md.open[6, 0])
    assert led["equity"].iloc[0] == pytest.approx(12_000.0 + pnl, rel=1e-9)


def test_untradable_bar_blocks_trading():
    md = toy_market(seed=15)
    md.valid[6, 0] = False
    cfg = EnvConfig(costs=FREE)
    led = run_weights(md, np.ones((md.T, 1)), cfg, i0=5, i1=5)
    assert led["equity"].iloc[0] == pytest.approx(cfg.initial_capital)  # no fill, stayed in cash (rf=0)


def test_short_squeeze_bankruptcy_is_bounded():
    md = toy_market(seed=21)
    md.open[6, 0] = md.close[5, 0] * 4.0  # overnight gap +300% against a 100% short
    md.high[6, 0] = max(md.high[6, 0], md.open[6, 0])
    md.close[6, 0] = md.open[6, 0]
    cfg = EnvConfig(costs=FREE)
    feats = np.zeros((md.T, 1, 1), np.float32)
    env = evaluation_env(md, feats, cfg, (4, md.T - 2))
    led = run_policy(env, WeightPolicy(-np.ones((md.T, 1))))[0]
    assert np.isfinite(led[["ret", "turnover", "gross"]].to_numpy()).all()
    assert led["ret"].min() >= -1.0 - 1e-12  # limited liability
    # the evaluation window is NOT truncated: the wiped-out book is liquidated and books flat days to the end
    assert len(led) == (md.T - 2) - 4 + 1  # decision rows 4..T-2 inclusive
    assert led["equity"].iloc[1] == 0.0 and (led["equity"].iloc[1:] == 0.0).all()
    assert (led["ret"].iloc[2:] == 0.0).all() and (led["gross"].iloc[1:] == 0.0).all()


def test_short_ban_blocks_new_shorts():
    md = toy_market(seed=23)
    md.symbols = ["C"]  # Citigroup was under the Sep-Oct 2008 SEC short-sale ban
    md.dates = pd.bdate_range("2008-09-10", periods=md.T)
    cfg = EnvConfig(costs=FREE)
    feats = np.zeros((md.T, 1, 1), np.float32)
    env = evaluation_env(md, feats, cfg, (2, 30))
    led = run_policy(env, WeightPolicy(-np.ones((md.T, 1))))[0]
    ban = (led.index >= "2008-09-19") & (led.index <= "2008-10-08")
    first_ban = int(np.argmax(ban))
    assert (led["w_C"].iloc[:first_ban] < -0.9).all()  # short before the ban
    assert (led["w_C"][ban] < 0).all()  # an existing short may be held through the ban (not added to)


def test_vectorised_matches_single():
    md = toy_market(T=300, seed=17)
    feats = np.zeros((md.T, 1, 1), np.float32)
    cfg = EnvConfig(costs=CostConfig(), episode_length=50)
    env = TradingEnv(md, feats, cfg, (5, md.T - 2), num_envs=4, seed=1)
    env.reset()
    starts = env.t.copy()
    rng = np.random.default_rng(0)
    acts = rng.uniform(-1, 1, (50, 4, 1))
    eq = []
    for k in range(50):
        _, _, _, _, info = env.step(acts[k])
        eq.append(info["equity"].copy())
    for b in range(4):
        e1 = TradingEnv(md, feats, replace(cfg, episode_length=None), (int(starts[b]), int(starts[b]) + 49),
                        num_envs=1, random_start=False, auto_reset=False)
        e1.reset()
        for k in range(50):
            _, _, _, _, info = e1.step(acts[k, b:b + 1])
            if k < 49:
                assert info["equity"][0] == pytest.approx(eq[k][b], rel=1e-12)
