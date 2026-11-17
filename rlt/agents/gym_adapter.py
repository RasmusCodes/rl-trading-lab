"""Vectorised adapter around gymnasium control tasks with the TradingEnv interface.

Used only to sanity-check the agent implementations on problems with a known answer (CartPole, Pendulum)
before trusting what they do on market data.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class _Cfg:
    obs_window = 1
    action = "continuous"


class GymVecAdapter:
    def __init__(self, env_id: str, num_envs: int = 8, seed: int = 0):
        self.envs = [gym.make(env_id) for _ in range(num_envs)]
        self.B = num_envs
        sp = self.envs[0].observation_space
        self.obs_dim = int(np.prod(sp.shape))
        asp = self.envs[0].action_space
        self.discrete = isinstance(asp, gym.spaces.Discrete)
        self.action_dim = int(asp.n) if self.discrete else int(np.prod(asp.shape))
        self.n = 1 if self.discrete else self.action_dim
        self.high = None if self.discrete else asp.high.astype(np.float32)
        self.seed = seed
        self.cfg = _Cfg()
        self.cfg.action = "discrete" if self.discrete else "continuous"
        self._ret = np.zeros(num_envs)
        self._len = np.zeros(num_envs, dtype=int)

    def reset(self, seed=None):
        s = self.seed if seed is None else seed
        obs = [e.reset(seed=s + i)[0] for i, e in enumerate(self.envs)]
        self._ret[:] = 0
        self._len[:] = 0
        return np.asarray(obs, np.float32)

    def step(self, actions):
        obs, rew, term, trunc, final = [], [], [], [], []
        for i, e in enumerate(self.envs):
            a = int(actions[i]) if self.discrete else np.clip(actions[i], -1, 1) * self.high
            o, r, te, tr, _ = e.step(a)
            self._ret[i] += r
            self._len[i] += 1
            final.append(o)
            if te or tr:
                o, _ = e.reset()
            obs.append(o)
            rew.append(r)
            term.append(te)
            trunc.append(tr)
        term, trunc = np.array(term), np.array(trunc)
        done = term | trunc
        info = {"final_obs": np.asarray(final, np.float32)}
        if done.any():
            info["episode"] = {"mask": done.copy(), "log_return": self._ret.copy(), "length": self._len.copy()}
            self._ret[done] = 0
            self._len[done] = 0
        return np.asarray(obs, np.float32), np.asarray(rew, np.float32), term, trunc, info
