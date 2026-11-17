"""Reward functions (vectorised over parallel episodes).

reward_t = scale * ( base_t  -  sum_k lambda_k * penalty_k,t )

Base rewards
------------
pnl            (E_t+1 - E_t) / E_0           absolute P&L in units of starting capital (risk-neutral in $)
log_return     log(E_t+1 / E_t)               growth-optimal (Kelly) objective; penalises variance ~ sigma^2/2
excess_return  r_t - rf_t
vol_scaled     (r_t - rf_t) / sigma_mkt,t     return per unit of the traded instruments' own (known) volatility
risk_adjusted  (r_t - rf_t) / sigma_hat_t-1   return per unit of the agent's own EWMA return volatility
diff_sharpe    Moody & Saffell (1998/2001) differential Sharpe ratio: first-order effect of r_t on an
               exponentially weighted Sharpe ratio. Summed over time ~ gradient of Sharpe.
diff_downside  Moody & Saffell differential downside-deviation ratio (Sortino analogue)
mv_utility     r_t - (kappa/2) r_t^2           per-step mean-variance utility

Penalties (all in return units)
-------------------------------
drawdown       increase of drawdown (mode "increment") or its level (mode "level")
cost           transaction costs paid this step / equity (on top of their effect on P&L)
inventory      gross exposure sum|w| (mode "abs") or sum w^2 (mode "quadratic")
turnover       sum |delta w|

Known failure modes (studied in experiments/reward_hacking): `pnl` with leverage over-bets; Sharpe-type rewards on
raw returns (use_excess=False) can be maximised by holding interest-bearing cash with ~zero variance; heavy
drawdown/inventory penalties collapse the policy to flat; `risk_adjusted` explodes when the agent's own volatility
estimate is near zero (no floor); penalty-dominated rewards plus a risk-limit termination teach the agent to end
episodes early.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SCALE = {"pnl": 100.0, "log_return": 100.0, "excess_return": 100.0, "vol_scaled": 1.0, "risk_adjusted": 1.0,
                 "diff_sharpe": 1.0, "diff_downside": 1.0, "mv_utility": 100.0}


@dataclass
class StepInfo:
    ret: np.ndarray  # simple net return on equity
    log_ret: np.ndarray
    rf: np.ndarray
    pnl: np.ndarray  # $
    initial: np.ndarray  # $
    cost_frac: np.ndarray  # costs / equity at start of step
    turnover: np.ndarray  # sum |dw|
    gross: np.ndarray
    weights: np.ndarray  # (B, n)
    dd: np.ndarray  # drawdown after step, >= 0
    dd_prev: np.ndarray
    mkt_vol: np.ndarray  # daily sigma of the traded book at unit gross, known at decision


class RewardFn:
    def __init__(self, spec: dict | str, num_envs: int):
        if isinstance(spec, str):
            spec = {"type": spec}
        self.spec = dict(spec)
        self.kind = self.spec.get("type", "log_return")
        if self.kind not in DEFAULT_SCALE:
            raise ValueError(f"unknown reward type {self.kind}")
        self.scale = float(self.spec.get("scale", DEFAULT_SCALE[self.kind]))
        self.eta = float(self.spec.get("eta", 1.0 / 63))  # EMA rate for Sharpe-type moments (~3 months)
        self.use_excess = bool(self.spec.get("use_excess", True))
        self.var_floor = float(self.spec.get("var_floor", (0.002) ** 2))  # ~0.2%/day; set 0 to see the blow-up
        self.kappa = float(self.spec.get("kappa", 5.0))
        self.pen = dict(self.spec.get("penalties", {}))
        self.B = num_envs
        self.A = np.zeros(num_envs)
        self.Bm = np.full(num_envs, 1e-4)
        self.DD2 = np.full(num_envs, 1e-4)
        self.ewvar = np.full(num_envs, 1e-4)

    def reset(self, mask=None) -> None:
        m = slice(None) if mask is None else mask
        self.A[m] = 0.0
        self.Bm[m] = 1e-4  # prior: ~1% daily vol
        self.DD2[m] = 1e-4
        self.ewvar[m] = 1e-4

    def __call__(self, s: StepInfo) -> np.ndarray:
        R = s.ret - s.rf if self.use_excess else s.ret
        k = self.kind
        if k == "pnl":
            base = s.pnl / s.initial
        elif k == "log_return":
            base = s.log_ret
        elif k == "excess_return":
            base = s.ret - s.rf
        elif k == "vol_scaled":
            base = (s.ret - s.rf) / np.maximum(s.mkt_vol, 1e-4)
        elif k == "risk_adjusted":
            base = R / np.sqrt(np.maximum(self.ewvar, self.var_floor) if self.var_floor > 0 else np.maximum(self.ewvar, 1e-20))
            self.ewvar = self.ewvar + self.eta * (R**2 - self.ewvar)
        elif k == "diff_sharpe":
            dA = R - self.A
            dB = R**2 - self.Bm
            var = self.Bm - self.A**2
            var = np.maximum(var, self.var_floor) if self.var_floor > 0 else np.maximum(var, 1e-20)
            base = (self.Bm * dA - 0.5 * self.A * dB) / var**1.5
            self.A = self.A + self.eta * dA
            self.Bm = self.Bm + self.eta * dB
        elif k == "diff_downside":
            dd = np.sqrt(np.maximum(self.DD2, self.var_floor if self.var_floor > 0 else 1e-20))
            pos = (R - 0.5 * self.A) / dd
            neg = (self.DD2 * (R - 0.5 * self.A) - 0.5 * self.A * R**2) / dd**3
            base = np.where(R > 0, pos, neg)
            self.A = self.A + self.eta * (R - self.A)
            self.DD2 = self.DD2 + self.eta * (np.minimum(R, 0.0) ** 2 - self.DD2)
        elif k == "mv_utility":
            base = R - 0.5 * self.kappa * R**2
        else:  # pragma: no cover
            raise ValueError(k)

        pen = np.zeros_like(base)
        if self.pen.get("drawdown", 0.0):
            if self.pen.get("drawdown_mode", "increment") == "level":
                pen += self.pen["drawdown"] * s.dd
            else:
                pen += self.pen["drawdown"] * np.maximum(s.dd - s.dd_prev, 0.0)
        if self.pen.get("cost", 0.0):
            pen += self.pen["cost"] * s.cost_frac
        if self.pen.get("inventory", 0.0):
            if self.pen.get("inventory_mode", "abs") == "quadratic":
                pen += self.pen["inventory"] * (s.weights**2).sum(1)
            else:
                pen += self.pen["inventory"] * s.gross
        if self.pen.get("turnover", 0.0):
            pen += self.pen["turnover"] * s.turnover
        return self.scale * (base - pen)
