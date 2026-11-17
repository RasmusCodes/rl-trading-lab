"""Off-policy continuous-control agents: SAC (Haarnoja et al. 2018) and TD3 (Fujimoto et al. 2018).

Both use twin critics and the minimum of the two in the target (clipped double-Q), which matters for the
same reason Double DQN does: with a near-zero true edge, a single critic's optimism is the main thing an
actor can learn to exploit.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .base import Agent, EpisodeStats, mlp, polyak
from .buffers import ReplayBuffer

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden, act=nn.ReLU)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden, act=nn.ReLU)

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SquashedGaussian(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden, act=nn.ReLU, out_gain=0.01)
        self.act_dim = act_dim

    def forward(self, o, deterministic=False, with_logp=True):
        mu, log_std = self.net(o).chunk(2, -1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            return torch.tanh(mu), None
        d = Normal(mu, log_std.exp())
        u = d.rsample()
        a = torch.tanh(u)
        if not with_logp:
            return a, None
        logp = d.log_prob(u).sum(-1) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class OffPolicyAgent(Agent):
    defaults = dict(hidden=(128, 128), lr=3e-4, gamma=0.99, tau=0.005, buffer=200_000, batch=256,
                    learning_starts=5_000, grad_steps=4, eval_every=20_000)

    def learn(self, env, total_steps, callback=None, log=None):
        p = self.p
        buf = ReplayBuffer(p["buffer"], self.spec.obs_dim, self.spec.action_dim, False)
        obs = env.reset(self.seed)
        B = env.B
        stats = EpisodeStats()
        diags = []
        for it in range(total_steps // B):
            if buf.size < p["learning_starts"]:
                a = self.rng.uniform(-1, 1, (B, self.spec.action_dim)).astype(np.float32)
            else:
                a = self.explore(obs)
            obs2, r, term, trunc, info = env.step(a)
            nxt = np.where((term | trunc)[:, None], info.get("final_obs", obs2), obs2)
            buf.add(obs, a, r, nxt, term.astype(np.float32))
            obs = obs2
            stats.update(info)
            if buf.size >= p["learning_starts"]:
                for _ in range(p["grad_steps"]):
                    diags.append(self.update(buf))
            if log is not None and (it + 1) % max(1, 5000 // B) == 0:
                row = {"step": self.env_steps + B, **stats.pop()}
                if diags:
                    keys = diags[0].keys()
                    row.update({k: float(np.mean([d[k] for d in diags])) for k in keys})
                diags = []
                log(row)
            if not self._tick(B, callback):
                break
        return self


class SAC(OffPolicyAgent):
    name = "sac"
    defaults = {**OffPolicyAgent.defaults, "alpha_init": 0.1, "target_entropy": None}

    def __init__(self, spec, params=None, seed=0):
        super().__init__(spec, params, seed)
        if spec.discrete:
            raise ValueError("SAC here is for continuous actions")
        p, h = self.p, tuple(self.p["hidden"])
        self.actor = SquashedGaussian(spec.obs_dim, spec.action_dim, h)
        self.critic = Critic(spec.obs_dim, spec.action_dim, h)
        self.critic_t = copy.deepcopy(self.critic)
        self.log_alpha = torch.tensor(np.log(p["alpha_init"]), dtype=torch.float32, requires_grad=True)
        self.target_ent = p["target_entropy"] if p["target_entropy"] is not None else -float(spec.action_dim)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=p["lr"])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=p["lr"])
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=p["lr"])

    def modules(self):
        return {"actor": self.actor, "critic": self.critic, "critic_t": self.critic_t}

    def get_state(self):
        st = super().get_state()
        st["log_alpha"] = float(self.log_alpha.detach())
        return st

    def set_state(self, st):
        super().set_state({k: v for k, v in st.items() if k != "log_alpha"})
        with torch.no_grad():
            self.log_alpha.fill_(st["log_alpha"])

    def act(self, obs, env=None, deterministic=True):
        with torch.no_grad():
            a, _ = self.actor(self.t(obs), deterministic=deterministic, with_logp=False)
        return a.numpy()

    def explore(self, obs):
        return self.act(obs, deterministic=False)

    def update(self, buf):
        p = self.p
        o, a, r, o2, d = buf.sample(p["batch"], self.rng)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2 = self.actor(o2)
            q1t, q2t = self.critic_t(o2, a2)
            y = r + p["gamma"] * (1 - d) * (torch.min(q1t, q2t) - alpha * lp2)
        q1, q2 = self.critic(o, a)
        lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad()
        lc.backward()
        self.opt_c.step()
        an, lp = self.actor(o)
        q1n, q2n = self.critic(o, an)
        la = (alpha * lp - torch.min(q1n, q2n)).mean()
        self.opt_a.zero_grad()
        la.backward()
        self.opt_a.step()
        lal = -(self.log_alpha * (lp.detach() + self.target_ent)).mean()
        self.opt_al.zero_grad()
        lal.backward()
        self.opt_al.step()
        polyak(self.critic_t, self.critic, p["tau"])
        return {"critic_loss": float(lc), "actor_loss": float(la), "alpha": float(alpha), "entropy": float(-lp.mean()),
                "q_mean": float(q1.mean())}


class TD3(OffPolicyAgent):
    name = "td3"
    defaults = {**OffPolicyAgent.defaults, "lr": 1e-3, "policy_delay": 2, "target_noise": 0.2, "noise_clip": 0.5,
                "explore_noise": 0.2}

    def __init__(self, spec, params=None, seed=0):
        super().__init__(spec, params, seed)
        if spec.discrete:
            raise ValueError("TD3 is for continuous actions")
        h = tuple(self.p["hidden"])
        self.actor = nn.Sequential(mlp(spec.obs_dim, spec.action_dim, h, act=nn.ReLU, out_gain=0.01), nn.Tanh())
        self.actor_t = copy.deepcopy(self.actor)
        self.critic = Critic(spec.obs_dim, spec.action_dim, h)
        self.critic_t = copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.p["lr"])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.p["lr"])
        self.n_upd = 0
        self._last_la = 0.0

    def modules(self):
        return {"actor": self.actor, "actor_t": self.actor_t, "critic": self.critic, "critic_t": self.critic_t}

    def act(self, obs, env=None, deterministic=True):
        with torch.no_grad():
            return self.actor(self.t(obs)).numpy()

    def explore(self, obs):
        a = self.act(obs) + self.rng.normal(0, self.p["explore_noise"], (len(obs), self.spec.action_dim))
        return np.clip(a, -1, 1).astype(np.float32)

    def update(self, buf):
        p = self.p
        o, a, r, o2, d = buf.sample(p["batch"], self.rng)
        with torch.no_grad():
            noise = (torch.randn_like(a) * p["target_noise"]).clamp(-p["noise_clip"], p["noise_clip"])
            a2 = (self.actor_t(o2) + noise).clamp(-1, 1)
            q1t, q2t = self.critic_t(o2, a2)
            y = r + p["gamma"] * (1 - d) * torch.min(q1t, q2t)
        q1, q2 = self.critic(o, a)
        lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad()
        lc.backward()
        self.opt_c.step()
        self.n_upd += 1
        if self.n_upd % p["policy_delay"] == 0:
            la = -self.critic(o, self.actor(o))[0].mean()
            self.opt_a.zero_grad()
            la.backward()
            self.opt_a.step()
            polyak(self.actor_t, self.actor, p["tau"])
            polyak(self.critic_t, self.critic, p["tau"])
            self._last_la = float(la)
        return {"critic_loss": float(lc), "actor_loss": self._last_la, "q_mean": float(q1.mean())}
