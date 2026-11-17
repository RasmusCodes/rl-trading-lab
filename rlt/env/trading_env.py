"""Vectorised trading environment.

One `TradingEnv` simulates B independent episodes in lock-step with numpy (no Python loop over episodes),
which is what makes on-policy RL on daily data affordable on a CPU. `GymTradingEnv` wraps a single episode
in the standard gymnasium API for use with external libraries.

Timing (no frictionless prices)
-------------------------------
    close t            the agent observes features built from bars <= t and its portfolio marked at close t,
                       and chooses target weights w*.
    open t+1           the old position has carried the overnight gap close t -> open t+1;
    (execution)        trades to w* are filled at the open of t+1 (default "next_open") paying half-spread +
                       slippage + square-root impact + commission, capped at a fraction of ADV;
    close t+1          the new position is marked at the close; cash earns the T-bill rate, shorts pay borrow,
                       negative cash pays rf + spread; the reward for the step is computed.

`execution="same_close"` (fill at the very close the decision was based on) exists only to measure how
much a backtest is flattered by that common shortcut.

Modes
-----
portfolio   all N instruments are traded jointly; the action is a vector of N target weights.
pool        each episode trades ONE instrument drawn from a pool (domain randomisation / cross-sectional
            pooling for single-asset agents); the action is one target weight.

Accounting identity (checked in tests/test_env_accounting.py):
    E_t+1 = E_t + q_old.(P_exec - C_t) + q_new.(C_t+1 - P_exec) - costs + interest - borrow
where q are units of the total-return price series and P_exec the fill price.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.market import MarketData
from .costs import CostConfig, CostModel
from .rewards import RewardFn, StepInfo


@dataclass
class EnvConfig:
    mode: str = "pool"  # "pool" (single instrument per episode) | "portfolio"
    episode_length: int | None = 252  # None = run to the end of the row range (evaluation)
    initial_capital: float = 1_000_000.0
    max_position: float = 1.0  # cap on |w_i| (fraction of equity)
    max_gross: float = 1.0  # cap on sum |w_i|
    max_net: float = 1.0  # cap on |sum w_i|
    long_only: bool = False
    action: str = "continuous"  # "continuous" | "discrete"
    discrete_levels: tuple = (-1.0, 0.0, 1.0)
    execution: str = "next_open"  # "next_open" | "next_close" | "next_vwap" | "same_close"
    trade_band: float = 0.0  # ignore target changes smaller than this (weight units)
    futures_rounding: bool = True  # trade whole futures contracts
    margin_call_gross: float = 3.0  # forced pro-rata deleveraging above this gross leverage
    bankruptcy_level: float = 0.05  # terminate if equity < 5% of initial
    terminate_drawdown: float | None = None  # optional risk-limit termination (reward-hacking experiment)
    terminal_penalty: float = 0.0  # reward added on any termination (bankruptcy / risk limit)
    obs_window: int = 1  # stack the last k feature rows
    portfolio_state: bool = True  # positions, exposure, cash, drawdown in the observation
    costs: CostConfig = field(default_factory=CostConfig)
    reward: dict = field(default_factory=lambda: {"type": "log_return"})
    randomize: dict = field(default_factory=dict)  # {"cost_mult": (lo, hi), "slippage_bps": (lo, hi), "obs_noise": s}


class TradingEnv:
    def __init__(self, md: MarketData, feats: np.ndarray, cfg: EnvConfig, row_range: tuple, num_envs: int = 1,
                 seed: int = 0, random_start: bool = True, assets=None, ready: np.ndarray | None = None,
                 auto_reset: bool = True, record: bool = False):
        if cfg.mode not in ("pool", "portfolio"):
            raise ValueError(cfg.mode)
        self.md, self.cfg = md, cfg
        self.T, self.N = md.T, md.N
        self.O, self.H, self.L, self.C = md.open, md.high, md.low, md.close
        self.valid, self.roll, self.rf = md.valid, md.roll, md.rf
        self.is_fut = md.is_future
        self.mult = md.multipliers
        with np.errstate(divide="ignore", invalid="ignore"):
            self.kappa = np.where(md.close > 0, md.close_raw / md.close, 1.0)  # raw $ per total-return unit
        self.cost = CostModel(md, cfg.costs)
        self.feats = feats.astype(np.float32, copy=False)
        self.F = feats.shape[2]
        self.ready = ready if ready is not None else np.isfinite(feats).all(axis=2)
        self.i0, self.i1 = int(row_range[0]), int(row_range[1])
        if self.i1 + 1 >= self.T:
            raise ValueError("row_range must leave one bar after the last decision for its fill")
        self.B = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.random_start = random_start
        self.auto_reset = auto_reset
        self.record = record
        pool = list(range(self.N)) if assets is None else [md.symbols.index(a) if isinstance(a, str) else int(a)
                                                            for a in assets]
        self.pool = np.array(pool, dtype=int)
        self.n = len(self.pool) if cfg.mode == "portfolio" else 1
        if cfg.action == "discrete" and self.n != 1:
            raise ValueError("discrete actions are implemented for single-instrument episodes only")
        self.levels = np.asarray(cfg.discrete_levels, dtype=float)
        k = cfg.obs_window
        self.obs_dim = k * self.n * self.F + ((self.n + 4) if cfg.portfolio_state else 0)
        self.action_dim = len(self.levels) if cfg.action == "discrete" else self.n
        self.reward_fn = RewardFn(cfg.reward, self.B)
        # admissible start rows per pool asset (portfolio mode: all assets ready)
        L = cfg.episode_length
        last_start = self.i1 - (L - 1) if L else self.i1
        self._starts = {}
        if cfg.mode == "portfolio":
            ok = self.ready[:, self.pool].all(axis=1)
            rows = np.arange(self.T)
            cand = rows[(rows >= max(self.i0, k - 1)) & (rows <= last_start) & ok]
            self._starts[-1] = cand
        else:
            rows = np.arange(self.T)
            for a in self.pool:
                cand = rows[(rows >= max(self.i0, k - 1)) & (rows <= last_start) & self.ready[:, a]]
                self._starts[int(a)] = cand
        if all(len(v) == 0 for v in self._starts.values()):
            raise ValueError("no admissible episode start in row_range (warm-up too long or range too short)")
        # state
        B, n = self.B, self.n
        self.t = np.zeros(B, dtype=int)
        self.end = np.zeros(B, dtype=int)
        self.assets = np.zeros((B, n), dtype=int)
        self.q = np.zeros((B, n))
        self.equity = np.full(B, cfg.initial_capital)
        self.initial = np.full(B, cfg.initial_capital)
        self.peak = self.equity.copy()
        self.dd = np.zeros(B)
        self.cost_mult = np.ones(B)
        self.slip = np.full(B, cfg.costs.slippage_bps)
        self.obs_noise = float(cfg.randomize.get("obs_noise", 0.0))
        self.active = np.ones(B, dtype=bool)
        self.k = np.zeros(B, dtype=int)
        self._ep = {k_: np.zeros(B) for k_ in ("logret", "s1", "s2", "turn", "cost", "gross", "rew")}
        self.ledger: list = []

    # ------------------------------------------------------------------------------------------- reset
    def _common_start(self) -> int:
        """Evaluation: every env starts on the first row admissible for all of them (identical windows)."""
        if self.cfg.mode == "portfolio":
            return int(self._starts[-1][0])
        firsts = [int(self._starts[int(a)][0]) for a in self.pool if len(self._starts[int(a)])]
        return max(firsts)

    def _reset_envs(self, mask: np.ndarray) -> None:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        cfg = self.cfg
        common = None if self.random_start else self._common_start()
        for b in idx:
            if cfg.mode == "portfolio":
                self.assets[b] = self.pool
                cand = self._starts[-1]
            else:
                if self.random_start:
                    choices = [a for a in self.pool if len(self._starts[int(a)])]
                    a = int(self.rng.choice(choices))
                else:
                    a = int(self.pool[b % len(self.pool)])
                self.assets[b, 0] = a
                cand = self._starts[a]
            if len(cand) == 0:
                raise ValueError(f"no admissible start for env {b}")
            s = int(self.rng.choice(cand)) if self.random_start else common
            self.t[b] = s
            self.end[b] = s + cfg.episode_length - 1 if cfg.episode_length else self.i1
        self.q[idx] = 0.0
        self.equity[idx] = cfg.initial_capital
        self.initial[idx] = cfg.initial_capital
        self.peak[idx] = cfg.initial_capital
        self.dd[idx] = 0.0
        self.k[idx] = 0
        for v in self._ep.values():
            v[idx] = 0.0
        rz = cfg.randomize
        if "cost_mult" in rz:
            lo, hi = rz["cost_mult"]
            self.cost_mult[idx] = np.exp(self.rng.uniform(np.log(lo), np.log(hi), len(idx)))
        else:
            self.cost_mult[idx] = cfg.costs.multiplier
        if "slippage_bps" in rz:
            lo, hi = rz["slippage_bps"]
            self.slip[idx] = self.rng.uniform(lo, hi, len(idx))
        else:
            self.slip[idx] = cfg.costs.slippage_bps
        self.reward_fn.reset(mask)
        self.active[idx] = True

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.ledger = []
        self._reset_envs(np.ones(self.B, dtype=bool))
        return self._obs()

    # ------------------------------------------------------------------------------------- observation
    def weights(self) -> np.ndarray:
        C = self.C[self.t[:, None], self.assets]
        return self.q * C / self.equity[:, None]

    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        k = cfg.obs_window
        rows = self.t[:, None] - np.arange(k)[::-1][None, :]  # (B, k)
        f = self.feats[rows[:, :, None], self.assets[:, None, :]]  # (B, k, n, F)
        f = f.reshape(self.B, -1)
        if self.obs_noise > 0:
            f = f + self.rng.normal(0.0, self.obs_noise, f.shape).astype(np.float32)
        if not cfg.portfolio_state:
            return f
        w = self.weights()
        fut = self.is_fut[self.assets]
        gross = np.abs(w).sum(1)
        net = w.sum(1)
        cash = 1.0 - (w * ~fut).sum(1)
        extra = np.column_stack([w, gross, net, cash, -self.dd]).astype(np.float32)
        return np.concatenate([f, extra], axis=1)

    # ------------------------------------------------------------------------------------------ action
    def weights_to_action(self, w: np.ndarray) -> np.ndarray:
        """Inverse of target_weights for policies that think in weights (baselines). NaN = hold."""
        cfg = self.cfg
        w = np.asarray(w, dtype=float).reshape(self.B, self.n)
        if cfg.action == "discrete":
            lv = self.levels * cfg.max_position
            idx = np.abs(np.nan_to_num(w[:, :1]) - lv[None, :]).argmin(1)
            return idx
        a = (2.0 * w / cfg.max_position - 1.0) if cfg.long_only else w / cfg.max_position
        return np.clip(a, -1.0, 1.0)  # NaN passes through clip

    def target_weights(self, action) -> np.ndarray:
        cfg = self.cfg
        if cfg.action == "discrete":
            a = np.asarray(action, dtype=int).reshape(self.B)
            w = self.levels[a][:, None] * cfg.max_position
            if cfg.long_only:
                w = np.maximum(w, 0.0)
        else:
            a = np.clip(np.asarray(action, dtype=float).reshape(self.B, self.n), -1.0, 1.0)
            w = (a + 1.0) / 2.0 * cfg.max_position if cfg.long_only else a * cfg.max_position
            if np.isnan(w).any():  # NaN = "hold": keep the current (drifted) weight for that instrument
                w = np.where(np.isnan(w), self.weights(), w)
        w = np.where(self.cost.shortable[self.assets], w, np.maximum(w, 0.0))
        gross = np.abs(w).sum(1, keepdims=True)
        w = np.where(gross > cfg.max_gross, w * cfg.max_gross / np.maximum(gross, 1e-12), w)
        net = w.sum(1, keepdims=True)
        w = np.where(np.abs(net) > cfg.max_net, w * cfg.max_net / np.maximum(np.abs(net), 1e-12), w)
        return w

    # -------------------------------------------------------------------------------------------- step
    def step(self, action):
        cfg = self.cfg
        B = self.B
        t = self.t.copy()
        a = self.assets
        tt, t1 = t[:, None], (t + 1)[:, None]
        w_tgt = self.target_weights(action)

        C_t = self.C[tt, a]
        C_n = self.C[t1, a]
        ex = cfg.execution
        if ex == "next_open":
            P = self.O[t1, a]
        elif ex == "next_close":
            P = C_n
        elif ex == "next_vwap":
            P = (self.O[t1, a] + self.H[t1, a] + self.L[t1, a] + C_n) / 4.0
        elif ex == "same_close":
            P = C_t
        else:
            raise ValueError(ex)

        q_old = self.q
        E_old = self.equity
        floor = 0.01 * self.initial  # denominators never shrink below 1% of starting capital
        pnl_pre = (q_old * (P - C_t)).sum(1)
        E_pre = E_old + pnl_pre
        w_pre = q_old * P / np.maximum(E_pre, floor)[:, None]

        # short-sale bans (e.g. SEC emergency order Sep-Oct 2008 on financials): shorts may be covered, not added
        banned = self.cost.short_ban[t1, a]
        if banned.any():
            w_tgt = np.where(banned, np.maximum(w_tgt, np.minimum(w_pre, 0.0)), w_tgt)

        # margin call: if pre-trade leverage is excessive, the target is cut pro rata
        gross_pre = np.abs(w_pre).sum(1)
        mc = gross_pre > cfg.margin_call_gross
        if mc.any():
            g_tgt = np.abs(w_tgt).sum(1)
            scale = np.where(mc & (g_tgt > cfg.max_gross), cfg.max_gross / np.maximum(g_tgt, 1e-12), 1.0)
            w_tgt = w_tgt * scale[:, None]

        fut = self.is_fut[a]
        q_tgt = w_tgt * np.maximum(E_pre, 0.0)[:, None] / P
        if cfg.futures_rounding and fut.any():
            unit = self.mult[a] * self.kappa[tt, a]  # total-return units per contract
            q_tgt = np.where(fut, np.round(q_tgt / unit) * unit, q_tgt)
        if cfg.trade_band > 0:
            q_tgt = np.where(np.abs(w_tgt - w_pre) < cfg.trade_band, q_old, q_tgt)
        if cfg.action == "continuous":
            hold = np.isnan(np.asarray(action, dtype=float).reshape(B, self.n))
            if hold.any():
                q_tgt = np.where(hold, q_old, q_tgt)
        tradable = self.valid[t1, a] & np.isfinite(P) & (P > 0)
        q_tgt = np.where(tradable, q_tgt, q_old)
        dq = q_tgt - q_old
        notional = np.abs(dq) * P
        if cfg.costs.participation_cap > 0:
            cap = cfg.costs.participation_cap * self.cost.adv[tt, a]
            over = np.isfinite(cap) & (notional > cap)
            if over.any():
                shrink = np.where(over, cap / np.maximum(notional, 1e-12), 1.0)
                dq = dq * shrink
                if cfg.futures_rounding and fut.any():
                    unit = self.mult[a] * self.kappa[tt, a]
                    dq = np.where(fut, np.trunc(dq / unit) * unit, dq)
                notional = np.abs(dq) * P
        q_new = q_old + dq

        cost = self.cost.trade_cost(t, a, notional, self.cost_mult[:, None], self.slip[:, None])
        if fut.any():
            rolling = self.roll[t1, a] & fut & (q_new != 0)
            if rolling.any():
                held = np.abs(q_new) * P
                cost = cost + np.where(rolling, self.cost.roll_cost(t, a, held, self.cost_mult[:, None]), 0.0)
        cost_total = cost.sum(1)

        pnl_post = (q_new * (C_n - P)).sum(1)
        nf = ~fut
        cash = E_pre - (q_new * P * nf).sum(1) - cost_total
        rf = self.rf[t]
        interest = np.where(cash >= 0, cash * rf, cash * (rf + self.cost.fin_spread_daily))
        short_val = (np.maximum(-q_new, 0.0) * P * nf)
        if cfg.costs.borrow:
            # borrow fee, plus a haircut on the interest earned by short proceeds (a $1M account does not get the
            # full T-bill rebate on cash collateral)
            borrow = (short_val * self.cost.borrow_daily[a]).sum(1) + short_val.sum(1) * self.cost.short_haircut_daily
        else:
            borrow = 0.0
        E_new = E_pre - cost_total + pnl_post + interest - borrow
        # limited liability: a book (or a pooled single-instrument sleeve) cannot lose more than its capital.
        # Without the floor, a 100% short through a gap (e.g. AIG or C in spring 2009) drives equity negative and
        # every ratio downstream (weights, turnover, returns) explodes. A wiped-out book is liquidated.
        E_new = np.maximum(E_new, 0.0)
        dead = E_new <= 0.0
        if dead.any():
            q_new = np.where(dead[:, None], 0.0, q_new)

        # ----- bookkeeping
        self.q = q_new
        self.t = t + 1
        self.k += 1
        self.equity = E_new
        dd_prev = self.dd.copy()
        self.peak = np.maximum(self.peak, E_new)
        self.dd = np.clip(1.0 - E_new / self.peak, 0.0, None)
        alive = E_old > 0  # a liquidated book (evaluation keeps stepping it) earns nothing
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where(alive, E_new / np.where(alive, E_old, 1.0) - 1.0, 0.0)
            log_ret = np.where(alive, np.maximum(np.log(np.maximum(E_new, 1e-12) / np.where(alive, E_old, 1.0)),
                                                 np.log(0.01)), 0.0)  # bankruptcy counts as ~-99%
        turnover = notional.sum(1) / np.maximum(E_pre, floor)
        w_new = q_new * C_n / np.maximum(E_new, floor)[:, None]
        gross = np.abs(w_new).sum(1)
        mkt_vol = (np.abs(w_tgt) * self.cost.sigma[tt, a]).sum(1) / np.maximum(np.abs(w_tgt).sum(1), 1e-12)
        mkt_vol = np.where(np.abs(w_tgt).sum(1) > 0, mkt_vol, self.cost.sigma[tt, a].mean(1))
        info_s = StepInfo(ret=ret, log_ret=log_ret, rf=rf, pnl=E_new - E_old, initial=self.initial,
                          cost_frac=cost_total / np.maximum(E_old, floor), turnover=turnover, gross=gross,
                          weights=w_new, dd=self.dd, dd_prev=dd_prev, mkt_vol=mkt_vol)
        reward = self.reward_fn(info_s)

        bankrupt = E_new <= cfg.bankruptcy_level * self.initial
        risk_stop = (self.dd >= cfg.terminate_drawdown) if cfg.terminate_drawdown else np.zeros(B, dtype=bool)
        terminated = bankrupt | risk_stop
        truncated = (self.t > self.end) & ~terminated
        reward = reward + np.where(terminated, cfg.terminal_penalty, 0.0)

        ep = self._ep
        ex_r = ret - rf
        ep["logret"] += log_ret
        ep["s1"] += ex_r
        ep["s2"] += ex_r**2
        ep["turn"] += turnover
        ep["cost"] += info_s.cost_frac
        ep["gross"] += gross
        ep["rew"] += reward

        if self.record:
            self.ledger.append({
                "row": self.t.copy(), "asset": a[:, 0].copy() if self.n == 1 else None, "equity": E_new.copy(),
                "ret": ret.copy(), "rf": rf.copy(), "cost": cost_total.copy(), "turnover": turnover.copy(),
                "weights": w_new.copy(), "target": w_tgt.copy(), "gross": gross.copy(), "reward": reward.copy(),
                "borrow": np.asarray(borrow, dtype=float).copy() if np.ndim(borrow) else np.full(B, borrow),
                "interest": interest.copy()})

        done = terminated | truncated
        info = {"equity": E_new.copy(), "ret": ret, "cost": cost_total, "turnover": turnover, "gross": gross,
                "terminated": terminated, "truncated": truncated}
        if done.any():
            n_ = np.maximum(self.k, 1)
            mu = ep["s1"] / n_
            sd = np.sqrt(np.maximum(ep["s2"] / n_ - mu**2, 1e-16))
            info["episode"] = {"mask": done.copy(), "log_return": ep["logret"].copy(), "length": self.k.copy(),
                               "sharpe": np.where(n_ > 5, mu / sd * np.sqrt(252), 0.0),
                               "turnover": ep["turn"] / n_ * 252, "cost": ep["cost"].copy(),
                               "gross": ep["gross"] / n_, "reward": ep["rew"].copy(),
                               "terminated": terminated.copy()}
        obs = self._obs()
        if done.any():
            info["final_obs"] = obs.copy()
            if self.auto_reset:
                self._reset_envs(done)
                obs = np.where(done[:, None], self._obs(), obs)
            else:
                self.active &= ~done
        return obs, reward.astype(np.float32), terminated, truncated, info


class GymTradingEnv:
    """Single-episode gymnasium.Env adapter (for external RL libraries)."""

    def __init__(self, *args, **kwargs):
        import gymnasium as gym

        kwargs["num_envs"] = 1
        self._env = TradingEnv(*args, **kwargs)
        e = self._env
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (e.obs_dim,), np.float32)
        if e.cfg.action == "discrete":
            self.action_space = gym.spaces.Discrete(e.action_dim)
        else:
            self.action_space = gym.spaces.Box(-1.0, 1.0, (e.action_dim,), np.float32)
        self.metadata = {}
        self.render_mode = None
        self.spec = None

    def reset(self, seed=None, options=None):
        obs = self._env.reset(seed)
        return obs[0], {}

    def step(self, action):
        obs, r, term, trunc, info = self._env.step(np.asarray(action)[None])
        flat = {k: (v[0] if isinstance(v, np.ndarray) and v.shape[:1] == (1,) else v) for k, v in info.items()
                if k not in ("episode", "final_obs")}
        return (info.get("final_obs", obs)[0], float(r[0]), bool(term[0]), bool(trunc[0]), flat)

    def close(self):
        pass
