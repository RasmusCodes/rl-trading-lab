"""On-policy actor-critic agents: A2C (Mnih et al. 2016) and PPO (Schulman et al. 2017).

Continuous actions use a diagonal Gaussian with a state-independent log-std; samples are clipped to [-1, 1]
by the environment (log-probabilities are of the unclipped sample, as in SB3). Discrete actions use a
categorical head. Advantages: GAE(lambda). Time-limit truncation is not a terminal state: the value of the
final observation is bootstrapped into the last reward (Pardo et al. 2018), otherwise the critic learns that
the world ends every 252 days.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .base import Agent, EpisodeStats, mlp


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, discrete, hidden=(64, 64), log_std_init=-0.5):
        super().__init__()
        self.discrete = discrete
        self.pi = mlp(obs_dim, act_dim, hidden, out_gain=0.01)
        self.v = mlp(obs_dim, 1, hidden, out_gain=1.0)
        if not discrete:
            self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def dist(self, x):
        out = self.pi(x)
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, self.log_std.exp().expand_as(out))

    def value(self, x):
        return self.v(x).squeeze(-1)


def logp_ent(dist, a, discrete):
    if discrete:
        return dist.log_prob(a), dist.entropy()
    return dist.log_prob(a).sum(-1), dist.entropy().sum(-1)


class OnPolicyAgent(Agent):
    defaults = dict(hidden=(64, 64), lr=3e-4, gamma=0.99, lam=0.95, n_steps=128, ent_coef=0.0, vf_coef=0.5,
                    max_grad=0.5, log_std_init=-0.5, eval_every=20_000)

    def __init__(self, spec, params=None, seed=0):
        super().__init__(spec, params, seed)
        p = self.p
        self.ac = ActorCritic(spec.obs_dim, spec.action_dim, spec.discrete, tuple(p["hidden"]), p["log_std_init"])
        self.opt = self._make_opt()

    def _make_opt(self):
        return torch.optim.Adam(self.ac.parameters(), lr=self.p["lr"], eps=1e-5)

    def modules(self):
        return {"ac": self.ac}

    def act(self, obs, env=None, deterministic=True):
        with torch.no_grad():
            d = self.ac.dist(self.t(obs))
            if self.spec.discrete:
                a = d.probs.argmax(-1) if deterministic else d.sample()
                return a.numpy()
            a = d.mean if deterministic else d.sample()
            return np.clip(a.numpy(), -1.0, 1.0)

    def collect(self, env, obs, stats):
        p = self.p
        n, B = p["n_steps"], env.B
        D = self.spec.obs_dim
        O = np.zeros((n, B, D), np.float32)
        A = np.zeros((n, B) if self.spec.discrete else (n, B, self.spec.action_dim), np.float32)
        LP, R, V, DN = (np.zeros((n, B), np.float32) for _ in range(4))
        for k in range(n):
            ot = self.t(obs)
            with torch.no_grad():
                d = self.ac.dist(ot)
                a = d.sample()
                lp, _ = logp_ent(d, a, self.spec.discrete)
                v = self.ac.value(ot)
            a_np = a.numpy()
            obs2, r, term, trunc, info = env.step(a_np if self.spec.discrete else np.clip(a_np, -1, 1))
            r = r.astype(np.float32).copy()
            if trunc.any():
                with torch.no_grad():
                    vf = self.ac.value(self.t(info["final_obs"][trunc])).numpy()
                r[trunc] += p["gamma"] * vf
            O[k], A[k], LP[k], R[k], V[k], DN[k] = obs, a_np, lp.numpy(), r, v.numpy(), (term | trunc)
            obs = obs2
            stats.update(info)
            self._last_info = info
        with torch.no_grad():
            last_v = self.ac.value(self.t(obs)).numpy()
        adv = np.zeros((n, B), np.float32)
        gae = np.zeros(B, np.float32)
        for k in reversed(range(n)):
            nv = last_v if k == n - 1 else V[k + 1]
            nonterm = 1.0 - DN[k]
            delta = R[k] + p["gamma"] * nv * nonterm - V[k]
            gae = delta + p["gamma"] * p["lam"] * nonterm * gae
            adv[k] = gae
        ret = adv + V
        return obs, dict(o=O.reshape(n * B, D), a=A.reshape(n * B, -1).squeeze(-1) if self.spec.discrete
                         else A.reshape(n * B, -1), lp=LP.ravel(), adv=adv.ravel(), ret=ret.ravel(), v=V.ravel())

    def learn(self, env, total_steps, callback=None, log=None):
        p = self.p
        obs = env.reset(self.seed)
        stats = EpisodeStats()
        per_iter = p["n_steps"] * env.B
        n_iter = max(1, total_steps // per_iter)
        for it in range(n_iter):
            obs, batch = self.collect(env, obs, stats)
            diag = self.update(batch, it, n_iter)
            if log is not None:
                row = {"step": self.env_steps + per_iter, **diag, **stats.pop()}
                if not self.spec.discrete:
                    row["policy_std"] = float(self.ac.log_std.exp().mean())
                log(row)
            if not self._tick(per_iter, callback):
                break
        return self


class A2C(OnPolicyAgent):
    name = "a2c"
    defaults = {**OnPolicyAgent.defaults, "lr": 7e-4, "n_steps": 16, "lam": 1.0, "ent_coef": 0.0}

    def _make_opt(self):
        return torch.optim.RMSprop(self.ac.parameters(), lr=self.p["lr"], alpha=0.99, eps=1e-5)

    def update(self, b, it, n_iter):
        p = self.p
        o, a = self.t(b["o"]), torch.as_tensor(b["a"])
        if self.spec.discrete:
            a = a.long()
        adv, ret = self.t(b["adv"]), self.t(b["ret"])
        d = self.ac.dist(o)
        lp, ent = logp_ent(d, a, self.spec.discrete)
        v = self.ac.value(o)
        pg = -(lp * adv).mean()
        vl = ((ret - v) ** 2).mean()
        loss = pg + p["vf_coef"] * vl - p["ent_coef"] * ent.mean()
        self.opt.zero_grad()
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.ac.parameters(), p["max_grad"])
        self.opt.step()
        return {"pg_loss": float(pg), "v_loss": float(vl), "entropy": float(ent.mean()), "grad_norm": float(gn)}


class PPO(OnPolicyAgent):
    name = "ppo"
    defaults = {**OnPolicyAgent.defaults, "epochs": 10, "minibatch": 512, "clip": 0.2, "target_kl": 0.03,
                "lr_decay": True}

    def update(self, b, it, n_iter):
        p = self.p
        if p["lr_decay"]:
            for g in self.opt.param_groups:
                g["lr"] = p["lr"] * (1.0 - it / n_iter)
        N = len(b["lp"])
        o_all, a_all = self.t(b["o"]), torch.as_tensor(b["a"])
        if self.spec.discrete:
            a_all = a_all.long()
        lp_old, adv_all, ret_all = self.t(b["lp"]), self.t(b["adv"]), self.t(b["ret"])
        diag = {"pg_loss": [], "v_loss": [], "entropy": [], "kl": [], "clipfrac": []}
        stop = False
        for _ in range(p["epochs"]):
            perm = self.rng.permutation(N)
            for s in range(0, N, p["minibatch"]):
                i = torch.as_tensor(perm[s: s + p["minibatch"]])
                d = self.ac.dist(o_all[i])
                lp, ent = logp_ent(d, a_all[i], self.spec.discrete)
                adv = adv_all[i]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = (lp - lp_old[i]).exp()
                pg = -torch.min(ratio * adv, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * adv).mean()
                vl = ((ret_all[i] - self.ac.value(o_all[i])) ** 2).mean()
                loss = pg + p["vf_coef"] * vl - p["ent_coef"] * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), p["max_grad"])
                self.opt.step()
                with torch.no_grad():
                    kl = float(((ratio - 1) - (lp - lp_old[i])).mean())
                diag["pg_loss"].append(float(pg))
                diag["v_loss"].append(float(vl))
                diag["entropy"].append(float(ent.mean()))
                diag["kl"].append(kl)
                diag["clipfrac"].append(float(((ratio - 1).abs() > p["clip"]).float().mean()))
                if p["target_kl"] and kl > 1.5 * p["target_kl"]:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) for k, v in diag.items() if v}
        y, v = b["ret"], b["v"]
        out["explained_var"] = float(1 - np.var(y - v) / (np.var(y) + 1e-12))
        return out
