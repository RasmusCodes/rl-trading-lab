"""Transaction-cost and financing model (see rlt.data.universe for the parameter rationale)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.market import MarketData

TICK_EIGHTHS_END = pd.Timestamp("1997-06-24")  # US equities moved from 1/8 to 1/16 $
DECIMALISATION = pd.Timestamp("2001-04-09")  # all US equity markets quoting in cents


@dataclass
class CostConfig:
    multiplier: float = 1.0  # scales every trading cost (sensitivity analysis)
    slippage_bps: float = 1.0  # execution noise vs the reference print (open auction, close, VWAP proxy)
    spread: bool = True
    impact: bool = True
    commissions: bool = True
    borrow: bool = True
    participation_cap: float = 0.10  # at most this fraction of ADV per session; 0 disables
    financing_spread_bps: float = 50.0  # p.a. over the T-bill rate on negative cash (margin loan)
    short_rebate_haircut_bps: float = 50.0  # p.a.: short proceeds earn T-bill minus this (on top of the borrow fee)


class CostModel:
    def __init__(self, md: MarketData, cfg: CostConfig):
        self.cfg = cfg
        T, N = md.T, md.N
        d = md.dates.to_numpy()
        tick = np.zeros((T, N))
        for j, s in enumerate(md.specs):
            if s.is_future:
                tick[:, j] = s.tick_size
            else:
                tick[:, j] = np.where(d < TICK_EIGHTHS_END.to_datetime64(), 1 / 8,
                                      np.where(d < DECIMALISATION.to_datetime64(), 1 / 16, 0.01))
        hs_min = np.array([s.min_half_spread_bps for s in md.specs]) / 1e4
        with np.errstate(divide="ignore", invalid="ignore"):
            hs = 0.5 * tick / md.close_raw
        hs = np.where(np.isfinite(hs), hs, hs_min[None, :])
        self.half_spread = np.maximum(hs, hs_min[None, :])
        self.adv, self.sigma = md.liquidity()
        fut = md.is_future
        self.is_future = fut
        self.comm_frac = np.array([0.0 if s.is_future else s.commission_bps / 1e4 for s in md.specs])
        self.comm_contract = np.array([s.commission_per_contract if s.is_future else 0.0 for s in md.specs])
        self.impact_coef = np.array([s.impact_coef for s in md.specs])
        self.borrow_daily = np.array([0.0 if s.is_future else s.borrow_bps_annual / 1e4 / 252 for s in md.specs])
        self.shortable = np.array([s.shortable for s in md.specs])
        self.contract_value = md.close_raw * md.multipliers[None, :]
        self.fin_spread_daily = cfg.financing_spread_bps / 1e4 / 252
        self.short_haircut_daily = cfg.short_rebate_haircut_bps / 1e4 / 252
        from ..data.universe import SHORT_BANS

        self.short_ban = np.zeros((T, N), dtype=bool)
        for j, s in enumerate(md.symbols):
            for start, end in SHORT_BANS.get(s, []):
                self.short_ban[:, j] |= (d >= pd.Timestamp(start).to_datetime64()) & (d <= pd.Timestamp(end).to_datetime64())

    def rate(self, t: np.ndarray, a: np.ndarray, notional: np.ndarray, slip_bps: np.ndarray) -> np.ndarray:
        """Proportional cost rate (fraction of notional) for trades decided at rows t on assets a."""
        cfg = self.cfg
        tt = t[:, None]
        r = np.zeros_like(notional)
        if cfg.spread:
            r = r + self.half_spread[tt, a]
        r = r + slip_bps / 1e4
        if cfg.impact:
            adv = self.adv[tt, a]
            with np.errstate(divide="ignore", invalid="ignore"):
                part = np.where(np.isfinite(adv) & (adv > 0), notional / adv, 0.0)
            r = r + self.impact_coef[a] * self.sigma[tt, a] * np.sqrt(part)
        return r

    def trade_cost(self, t, a, notional, cost_mult, slip_bps) -> np.ndarray:
        """$ cost of trading |notional| (B, n) at the next fill."""
        c = notional * self.rate(t, a, notional, slip_bps)
        if self.cfg.commissions:
            tt = t[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                contracts = np.where(self.is_future[a], notional / self.contract_value[tt, a], 0.0)
            c = c + self.comm_frac[a] * notional + self.comm_contract[a] * np.nan_to_num(contracts)
        return cost_mult * c

    def roll_cost(self, t, a, held_notional, cost_mult) -> np.ndarray:
        """Calendar-spread trade for futures held through a roll date: two legs of spread + commission."""
        tt = t[:, None]
        hs = self.half_spread[tt, a] if self.cfg.spread else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            contracts = np.nan_to_num(held_notional / self.contract_value[tt, a])
        comm = 2 * self.comm_contract[a] * contracts if self.cfg.commissions else 0.0
        return cost_mult * (2 * hs * held_notional + comm)
