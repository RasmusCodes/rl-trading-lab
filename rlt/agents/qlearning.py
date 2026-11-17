"""Tabular Q-learning on a coarse discretisation of the state.

State = (tercile of each chosen market feature) x (current position level). With 3 features x 3 bins x 3
position levels that is 81 states, i.e. something a table can actually learn from ~6,000 daily bars - and
small enough that every Q-value can be printed and read, which makes it the interpretable reference agent.
Bin edges are quantiles of the scaled features over the TRAINING rows only.
"""
from __future__ import annotations

import numpy as np

from .base import Agent, EpisodeStats


class TabularQ(Agent):
    name = "qlearning"
    defaults = dict(features=("ret_21d", "ma_dist_200", "vol_ratio"), bins=3, alpha=0.02, gamma=0.9,
                    eps_start=1.0, eps_end=0.05, eps_frac=0.5, eval_every=20_000)

    def __init__(self, spec, params=None, seed=0):
        super().__init__(spec, params, seed)
        if not spec.discrete:
            raise ValueError("tabular Q-learning needs a discrete action space")
        names = spec.obs_names
        self.f_idx = []
        for f in self.p["features"]:
            cand = [i for i, n in enumerate(names) if n.startswith(f + "@a0@lag0")]
            if not cand:
                raise ValueError(f"feature {f} not in observation")
            self.f_idx.append(cand[0])
        self.w_idx = names.index("w@a0") if "w@a0" in names else None
        self.levels = np.asarray(spec.levels if spec.levels else (-1.0, 0.0, 1.0))
        self.nb = int(self.p["bins"])
        self.n_pos = len(self.levels) if self.w_idx is not None else 1
        self.n_states = self.nb ** len(self.f_idx) * self.n_pos
        self.Q = np.zeros((self.n_states, spec.action_dim))
        self.N = np.zeros((self.n_states, spec.action_dim))
        self.edges = None

    def _fit_edges(self, env) -> None:
        rows = np.arange(env.i0, env.i1 + 1)
        X = env.feats[rows][:, env.pool, :]  # (rows, pool, F)
        F = env.F
        qs = np.linspace(0, 1, self.nb + 1)[1:-1]
        self.edges = []
        for i in self.f_idx:
            fi = i % F
            v = X[:, :, fi].ravel()
            self.edges.append(np.quantile(v, qs))

    def _state(self, obs: np.ndarray) -> np.ndarray:
        s = np.zeros(len(obs), dtype=int)
        for k, i in enumerate(self.f_idx):
            s = s * self.nb + np.searchsorted(self.edges[k], obs[:, i])
        if self.w_idx is not None:
            pos = np.abs(obs[:, self.w_idx][:, None] - self.levels[None, :]).argmin(1)
            s = s * self.n_pos + pos
        return s

    def act(self, obs, env=None, deterministic=True):
        s = self._state(np.asarray(obs))
        q = self.Q[s]
        # unvisited states: stay flat-ish -> prefer the action closest to 0 on ties
        return q.argmax(1)

    def learn(self, env, total_steps, callback=None, log=None):
        p = self.p
        if self.edges is None:
            self._fit_edges(env)
        obs = env.reset(self.seed)
        stats = EpisodeStats()
        B = env.B
        n_iter = total_steps // B
        for it in range(n_iter):
            frac = min(1.0, it / max(1, p["eps_frac"] * n_iter))
            eps = p["eps_start"] + frac * (p["eps_end"] - p["eps_start"])
            s = self._state(obs)
            greedy = self.Q[s].argmax(1)
            rand = self.rng.integers(0, self.spec.action_dim, B)
            a = np.where(self.rng.random(B) < eps, rand, greedy)
            obs2, r, term, trunc, info = env.step(a)
            nxt = np.where((term | trunc)[:, None], info.get("final_obs", obs2), obs2)
            s2 = self._state(nxt)
            target = r + p["gamma"] * (1.0 - term) * self.Q[s2].max(1)
            np.add.at(self.N, (s, a), 1.0)
            delta = np.zeros_like(self.Q)
            np.add.at(delta, (s, a), target - self.Q[s, a])
            cnt = np.zeros_like(self.Q)
            np.add.at(cnt, (s, a), 1.0)
            self.Q += p["alpha"] * np.where(cnt > 0, delta / np.maximum(cnt, 1), 0.0)
            obs = obs2
            stats.update(info)
            if log is not None and (it + 1) % max(1, 5000 // B) == 0:
                log({"step": self.env_steps + B, "eps": eps, "q_mean": float(self.Q.mean()),
                     "visited_states": int((self.N.sum(1) > 0).sum()), **stats.pop()})
            if not self._tick(B, callback):
                break
        return self

    def get_state(self):
        return {"Q": self.Q.copy(), "N": self.N.copy(), "edges": [e.copy() for e in (self.edges or [])]}

    def set_state(self, st):
        self.Q, self.N, self.edges = st["Q"].copy(), st["N"].copy(), [e.copy() for e in st["edges"]]

    def policy_table(self) -> np.ndarray:
        return self.Q.argmax(1)
