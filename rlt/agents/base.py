"""Agent interface and shared training utilities.

Every agent implements
    learn(env, total_steps, callback=None, log=None)   train on a vectorised env (TradingEnv or GymVecAdapter)
    act(obs, env=None, deterministic=True)              actions for a batch of observations
    reset(num_envs)                                     clear recurrent state before an evaluation
    get_state() / set_state(state)                      in-memory snapshots (checkpoint selection)

`callback(agent, env_steps) -> bool` is called every `eval_every` environment transitions; the experiment
runner uses it to evaluate on the validation window, keep the best snapshot and stop early. `log(dict)`
receives training diagnostics (episode returns, losses, entropy, exposure) for the dashboard.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EnvSpec:
    obs_dim: int
    action_dim: int
    discrete: bool
    n_assets: int = 1
    feat_dim: int = 0  # leading part of the observation that is market features
    obs_names: list = field(default_factory=list)
    levels: tuple = ()

    @classmethod
    def from_env(cls, env, obs_names=None) -> "EnvSpec":
        discrete = getattr(env, "discrete", None)
        if discrete is None:
            discrete = env.cfg.action == "discrete"
        n = getattr(env, "n", env.action_dim)
        feat = env.cfg.obs_window * n * env.F if hasattr(env, "F") else env.obs_dim
        levels = tuple(getattr(env, "levels", ())) if discrete else ()
        return cls(env.obs_dim, env.action_dim, bool(discrete), n, feat, list(obs_names or []), levels)


def mlp(inp: int, out: int, hidden=(64, 64), act=nn.Tanh, out_gain: float = 1.0) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, act()]
        d = h
    last = nn.Linear(d, out)
    nn.init.orthogonal_(last.weight, gain=out_gain)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


def polyak(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


class EpisodeStats:
    KEYS = ("log_return", "sharpe", "turnover", "gross", "cost", "length", "reward", "terminated")

    def __init__(self):
        self.buf = defaultdict(list)

    def update(self, info: dict) -> None:
        ep = info.get("episode")
        if ep is None:
            return
        m = ep["mask"]
        for k in self.KEYS:
            if k in ep:
                self.buf[k].extend(np.asarray(ep[k])[m].tolist())

    def pop(self) -> dict:
        out = {f"ep_{k}": float(np.mean(v)) for k, v in self.buf.items() if len(v)}
        if self.buf:
            out["episodes"] = len(next(iter(self.buf.values())))
        self.buf = defaultdict(list)
        return out


class Agent:
    name = "agent"
    defaults: dict = {}
    recurrent = False

    def __init__(self, spec: EnvSpec, params: dict | None = None, seed: int = 0):
        self.spec = spec
        self.p = {**self.defaults, **(params or {})}
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.eval_every = int(self.p.get("eval_every", 20_000))
        self._next_eval = self.eval_every
        self.env_steps = 0

    # -- to implement
    def learn(self, env, total_steps: int, callback=None, log=None):  # pragma: no cover
        raise NotImplementedError

    def act(self, obs, env=None, deterministic: bool = True):  # pragma: no cover
        raise NotImplementedError

    def reset(self, num_envs: int) -> None:
        pass

    def modules(self) -> dict:
        return {}

    # -- shared
    def get_state(self) -> dict:
        return {k: copy.deepcopy(m.state_dict()) for k, m in self.modules().items()}

    def set_state(self, state: dict) -> None:
        for k, m in self.modules().items():
            m.load_state_dict(state[k])

    def save(self, path) -> None:
        torch.save({"name": self.name, "params": self.p, "state": self.get_state()}, path)

    def _tick(self, n: int, callback) -> bool:
        """Advance the step counter; run the callback when due. Returns False to stop training."""
        self.env_steps += n
        if callback is not None and self.env_steps >= self._next_eval:
            self._next_eval += self.eval_every
            return bool(callback(self, self.env_steps))
        return True

    @staticmethod
    def t(x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)
