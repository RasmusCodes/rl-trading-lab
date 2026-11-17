"""Deep Q-Network (Mnih et al. 2015) and Double DQN (van Hasselt et al. 2016), optional dueling head.

Double DQN decouples action selection (online net) from evaluation (target net) in the bootstrap target,
which removes the max-operator's upward bias - relevant in trading, where Q-values of all actions are close
and noise dominates, so max over noisy estimates systematically overstates the value of trading.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent, EpisodeStats, mlp, polyak
from .buffers import ReplayBuffer


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden, dueling):
        super().__init__()
        self.dueling = dueling
        if dueling:
            self.body = mlp(obs_dim, hidden[-1], hidden[:-1], act=nn.ReLU)
            self.v = nn.Linear(hidden[-1], 1)
            self.a = nn.Linear(hidden[-1], n_actions)
        else:
            self.q = mlp(obs_dim, n_actions, hidden, act=nn.ReLU)

    def forward(self, x):
        if not self.dueling:
            return self.q(x)
        h = F.relu(self.body(x))
        a = self.a(h)
        return self.v(h) + a - a.mean(1, keepdim=True)


class DQN(Agent):
    name = "dqn"
    defaults = dict(hidden=(64, 64), lr=5e-4, gamma=0.95, buffer=100_000, batch=256, learning_starts=5_000,
                    grad_steps=2, tau=0.005, eps_start=1.0, eps_end=0.05, eps_frac=0.4, double=False,
                    dueling=False, max_grad=10.0, eval_every=20_000)

    def __init__(self, spec, params=None, seed=0):
        super().__init__(spec, params, seed)
        if not spec.discrete:
            raise ValueError("DQN needs a discrete action space")
        p = self.p
        self.q = QNet(spec.obs_dim, spec.action_dim, tuple(p["hidden"]), p["dueling"])
        self.q_t = QNet(spec.obs_dim, spec.action_dim, tuple(p["hidden"]), p["dueling"])
        self.q_t.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=p["lr"])
        self.updates = 0

    def modules(self):
        return {"q": self.q, "q_t": self.q_t}

    def act(self, obs, env=None, deterministic=True):
        with torch.no_grad():
            return self.q(self.t(obs)).argmax(1).numpy()

    def _update(self, buf):
        p = self.p
        o, a, r, o2, d = buf.sample(p["batch"], self.rng)
        with torch.no_grad():
            if p["double"]:
                a2 = self.q(o2).argmax(1, keepdim=True)
                qn = self.q_t(o2).gather(1, a2).squeeze(1)
            else:
                qn = self.q_t(o2).max(1).values
            y = r + p["gamma"] * (1 - d) * qn
        qsa = self.q(o).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(qsa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), p["max_grad"])
        self.opt.step()
        polyak(self.q_t, self.q, p["tau"])
        self.updates += 1
        return float(loss), float(qsa.mean())

    def learn(self, env, total_steps, callback=None, log=None):
        p = self.p
        buf = ReplayBuffer(p["buffer"], self.spec.obs_dim, 1, True)
        obs = env.reset(self.seed)
        B = env.B
        stats = EpisodeStats()
        losses, qs = [], []
        n_iter = total_steps // B
        for it in range(n_iter):
            frac = min(1.0, it / max(1, p["eps_frac"] * n_iter))
            eps = p["eps_start"] + frac * (p["eps_end"] - p["eps_start"])
            greedy = self.act(obs)
            a = np.where(self.rng.random(B) < eps, self.rng.integers(0, self.spec.action_dim, B), greedy)
            obs2, r, term, trunc, info = env.step(a)
            nxt = np.where((term | trunc)[:, None], info.get("final_obs", obs2), obs2)
            buf.add(obs, a, r, nxt, term.astype(np.float32))
            obs = obs2
            stats.update(info)
            if buf.size >= p["learning_starts"]:
                for _ in range(p["grad_steps"]):
                    l_, q_ = self._update(buf)
                    losses.append(l_)
                    qs.append(q_)
            if log is not None and (it + 1) % max(1, 5000 // B) == 0:
                row = {"step": self.env_steps + B, "eps": eps, **stats.pop()}
                if losses:
                    row.update(loss=float(np.mean(losses)), q_mean=float(np.mean(qs)))
                losses, qs = [], []
                log(row)
            if not self._tick(B, callback):
                break
        return self


class DoubleDQN(DQN):
    name = "ddqn"
    defaults = {**DQN.defaults, "double": True}
