"""Replay buffer for off-policy agents (DQN family, SAC, TD3)."""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, discrete: bool):
        self.cap = int(capacity)
        self.o = np.zeros((self.cap, obs_dim), np.float32)
        self.o2 = np.zeros((self.cap, obs_dim), np.float32)
        self.a = np.zeros((self.cap,), np.int64) if discrete else np.zeros((self.cap, act_dim), np.float32)
        self.r = np.zeros((self.cap,), np.float32)
        self.d = np.zeros((self.cap,), np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, o2, d) -> None:
        n = len(o)
        idx = (self.ptr + np.arange(n)) % self.cap
        self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx] = o, a, r, o2, d
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, n: int, rng: np.random.Generator):
        i = rng.integers(0, self.size, n)
        t = torch.as_tensor
        return t(self.o[i]), t(self.a[i]), t(self.r[i]), t(self.o2[i]), t(self.d[i])
