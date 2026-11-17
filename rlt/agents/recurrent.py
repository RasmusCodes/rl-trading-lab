"""Recurrent reinforcement learning.

RRL      Moody & Wu (1997), Moody & Saffell (1998, 2001) "direct reinforcement": the position is a recurrent
         function of the features and the agent's own previous position,
             F_t = tanh(theta . x_t + u * F_t-1 + b),
         trained by gradient ascent (back-propagation through time over F_t-1) on the Sharpe ratio of
             R_t = F_t * (r_t+1 - rf_t) - c_t * |F_t - F_t-1|,
         where r_t+1 is the open-to-open return the decision earns and c_t the cost per unit traded (spread +
         slippage + commission from the environment's cost model at row t). The recurrence on F_t-1 is what
         lets the policy internalise transaction costs. No value function, no bootstrapping.

PPO-LSTM PPO with an LSTM actor-critic (truncated BPTT over 32-step chunks, hidden state reset at episode
         starts): the deep-RL reading of "recurrent RL", for partially observed (regime-switching) markets.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .base import Agent, EpisodeStats


# ------------------------------------------------------------------------------------------------- RRL
class RRLNet(nn.Module):
    def __init__(self, feat_dim: int, n: int, hidden: int = 0, long_only: bool = False):
        super().__init__()
        if hidden:
            self.f = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Tanh(), nn.Linear(hidden, n))
        else:
            self.f = nn.Linear(feat_dim, n)
            nn.init.normal_(self.f.weight, std=0.01)
            nn.init.zeros_(self.f.bias)
        self.u = nn.Parameter(torch.zeros(n))
        self.long_only = long_only

    def step(self, x, F_prev):
        z = self.f(x) + self.u * F_prev
        return torch.sigmoid(z) if self.long_only else torch.tanh(z)


class RRL(Agent):
    name = "rrl"
    recurrent = True
    defaults = dict(lr=3e-3, updates=1500, batch=32, seq_len=252, weight_decay=1e-3, hidden=0, cost_scale=1.0,
                    eval_every=40_000)

    def __init__(self, spec, params=None, seed=0):
        super().__init__(spec, params, seed)
        self.net = None
        self.F_prev = None
        self.cfg = None

    def modules(self):
        return {"net": self.net}

    def _build(self, env):
        self.cfg = env.cfg
        if env.cfg.obs_window != 1:
            raise ValueError("RRL uses obs_window=1 (its memory is the recurrent position)")
        n = env.n
        self.net = RRLNet(n * env.F, n, int(self.p["hidden"]), env.cfg.long_only)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.p["lr"], weight_decay=self.p["weight_decay"])

    def _weights(self, F):
        cfg = self.cfg
        w = F * cfg.max_position
        g = w.abs().sum(-1, keepdim=True)
        return w * torch.clamp(cfg.max_gross / torch.clamp(g, min=1e-8), max=1.0)

    def _data(self, env):
        """Arrays for differentiable training on the env's training rows (decision rows i0..i1-1)."""
        cm = env.cost
        O = env.O
        T = env.T
        r = np.full(O.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            r[:-2] = np.log(O[2:] / O[1:-1])
        r = np.nan_to_num(r)
        slip = env.cfg.costs.slippage_bps / 1e4
        with np.errstate(divide="ignore", invalid="ignore"):
            comm = cm.comm_frac[None, :] + np.where(cm.is_future[None, :],
                                                    cm.comm_contract[None, :] / cm.contract_value, 0.0)
        c = (cm.half_spread + slip + np.nan_to_num(comm)) * env.cfg.costs.multiplier * self.p["cost_scale"]
        return r, c, env.rf, T

    def learn(self, env, total_steps=None, callback=None, log=None):
        if self.net is None:
            self._build(env)
        p = self.p
        r, c, rf, T = self._data(env)
        L, Bt = int(p["seq_len"]), int(p["batch"])
        feats = env.feats
        portfolio = env.cfg.mode == "portfolio"
        pool = env.pool
        starts_by_asset = {}
        for a in (pool if not portfolio else [-1]):
            cand = env._starts[int(a)] if not portfolio else env._starts[-1]
            cand = cand[cand <= env.i1 - 1 - L]
            if len(cand):
                starts_by_asset[int(a)] = cand
        keys = list(starts_by_asset)
        n_updates = int(p["updates"]) if total_steps is None else max(1, int(total_steps) // (L * Bt))
        for upd in range(n_updates):
            if portfolio:
                s = self.rng.choice(starts_by_asset[-1], Bt)
                rows = s[:, None] + np.arange(L)[None, :]  # (B, L)
                X = torch.as_tensor(feats[rows][:, :, pool, :].reshape(Bt, L, -1))
                Rr = torch.as_tensor(r[rows][:, :, pool], dtype=torch.float32)
                Cc = torch.as_tensor(c[rows][:, :, pool], dtype=torch.float32)
            else:
                a = np.array([keys[i] for i in self.rng.integers(0, len(keys), Bt)])
                s = np.array([self.rng.choice(starts_by_asset[k]) for k in a])
                rows = s[:, None] + np.arange(L)[None, :]
                X = torch.as_tensor(feats[rows, a[:, None], :])  # (B, L, F)
                Rr = torch.as_tensor(r[rows, a[:, None]][:, :, None], dtype=torch.float32)
                Cc = torch.as_tensor(c[rows, a[:, None]][:, :, None], dtype=torch.float32)
            RF = torch.as_tensor(rf[rows], dtype=torch.float32)  # (B, L)
            n = Rr.shape[-1]
            Z = self.net.f(X)  # (B, L, n): the non-recurrent part in one matmul
            F_prev = torch.zeros(Bt, n)
            Fs = []
            squash = torch.sigmoid if self.net.long_only else torch.tanh
            for j in range(L):  # only the recurrence is sequential
                F_prev = squash(Z[:, j] + self.net.u * F_prev)
                Fs.append(F_prev)
            W = self._weights(torch.stack(Fs, 1))  # (B, L, n)
            W_prev = torch.cat([torch.zeros(Bt, 1, n), W[:, :-1]], 1)
            R = (W * (Rr - RF[:, :, None])).sum(-1) - (Cc * (W - W_prev).abs()).sum(-1)  # (B, L)
            w_prev = W[:, -1]
            sharpe = R.mean(1) / (R.std(1) + 1e-6) * np.sqrt(252)
            loss = -sharpe.mean()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            if log is not None and (upd + 1) % 25 == 0:
                log({"step": self.env_steps + L * Bt, "train_sharpe": float(sharpe.mean()),
                     "u_recurrence": float(self.net.u.mean()), "avg_abs_position": float(w_prev.abs().mean())})
            if not self._tick(L * Bt, callback):
                break
        return self

    def reset(self, num_envs):
        self.F_prev = torch.zeros(num_envs, self.net.u.shape[0])

    def act(self, obs, env=None, deterministic=True):
        x = self.t(obs)[:, : self.spec.feat_dim]
        if self.F_prev is None or self.F_prev.shape[0] != len(x):
            self.reset(len(x))
        with torch.no_grad():
            F = self.net.step(x, self.F_prev)
        self.F_prev = F
        a = F.numpy()
        return (2 * a - 1) if self.cfg.long_only else a


# --------------------------------------------------------------------------------------------- PPO-LSTM
class RecurrentAC(nn.Module):
    """Separate actor and critic LSTMs (as in SB3's RecurrentPPO). A shared trunk let the value loss, which
    is orders of magnitude larger early in training, dominate the representation and stall the policy."""

    def __init__(self, obs_dim, act_dim, discrete, hidden=64, log_std_init=-0.5):
        super().__init__()
        self.discrete = discrete
        H = hidden
        self.enc_pi = nn.Sequential(nn.Linear(obs_dim, H), nn.Tanh())
        self.enc_v = nn.Sequential(nn.Linear(obs_dim, H), nn.Tanh())
        for enc in (self.enc_pi, self.enc_v):
            nn.init.orthogonal_(enc[0].weight, np.sqrt(2))
            nn.init.zeros_(enc[0].bias)
        self.lstm_pi = nn.LSTM(H, H)
        self.lstm_v = nn.LSTM(H, H)
        for lstm in (self.lstm_pi, self.lstm_v):
            for name, prm in lstm.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(prm, 1.0)
                else:
                    nn.init.zeros_(prm)
        self.pi = nn.Linear(H, act_dim)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        self.v_head = nn.Linear(H, 1)
        nn.init.orthogonal_(self.v_head.weight, 1.0)
        nn.init.zeros_(self.v_head.bias)
        if not discrete:
            self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.H = H
        self.state_size = 2 * H  # hidden state = [actor | critic]

    def seq(self, obs, h, c, starts):
        """obs (L, B, D), h/c (1, B, 2H), starts (L, B) bool -> features (L, B, 2H), (h, c)."""
        H = self.H
        zp, zv = self.enc_pi(obs), self.enc_v(obs)
        hp, hv, cp, cv = h[..., :H], h[..., H:], c[..., :H], c[..., H:]
        outs = []
        for j in range(obs.shape[0]):
            m = (1.0 - starts[j].float()).view(1, -1, 1)
            hp, hv, cp, cv = hp * m, hv * m, cp * m, cv * m
            op, (hp, cp) = self.lstm_pi(zp[j: j + 1], (hp, cp))
            ov, (hv, cv) = self.lstm_v(zv[j: j + 1], (hv, cv))
            outs.append(torch.cat([op, ov], -1))
        return torch.cat(outs, 0), (torch.cat([hp, hv], -1), torch.cat([cp, cv], -1))

    def dist(self, feat):
        out = self.pi(feat[..., : self.H])
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, self.log_std.exp().expand_as(out))

    def v(self, feat):
        return self.v_head(feat[..., self.H:])


class PPOLSTM(Agent):
    name = "ppo_lstm"
    recurrent = True
    defaults = dict(hidden=64, lr=3e-4, gamma=0.99, lam=0.95, n_steps=128, chunk=32, epochs=6, n_minibatch=4,
                    clip=0.2, ent_coef=0.0, vf_coef=0.5, max_grad=0.5, log_std_init=-0.5, target_kl=0.03,
                    eval_every=20_000)

    def __init__(self, spec, params=None, seed=0):
        super().__init__(spec, params, seed)
        p = self.p
        self.ac = RecurrentAC(spec.obs_dim, spec.action_dim, spec.discrete, int(p["hidden"]), p["log_std_init"])
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=p["lr"], eps=1e-5)
        self.h = self.c = None
        self.start = None

    def modules(self):
        return {"ac": self.ac}

    def reset(self, num_envs):
        S = self.ac.state_size
        self.h, self.c = torch.zeros(1, num_envs, S), torch.zeros(1, num_envs, S)
        self.start = torch.ones(num_envs, dtype=torch.bool)

    def act(self, obs, env=None, deterministic=True):
        if self.h is None or self.h.shape[1] != len(obs):
            self.reset(len(obs))
        with torch.no_grad():
            f, (self.h, self.c) = self.ac.seq(self.t(obs)[None], self.h, self.c, self.start[None])
            self.start = torch.zeros_like(self.start)
            d = self.ac.dist(f[0])
            if self.spec.discrete:
                return (d.probs.argmax(-1) if deterministic else d.sample()).numpy()
            a = d.mean if deterministic else d.sample()
            return np.clip(a.numpy(), -1, 1)

    def _lp(self, d, a):
        if self.spec.discrete:
            return d.log_prob(a), d.entropy()
        return d.log_prob(a).sum(-1), d.entropy().sum(-1)

    def learn(self, env, total_steps, callback=None, log=None):
        p = self.p
        B, n, D, H = env.B, p["n_steps"], self.spec.obs_dim, self.ac.state_size
        obs = env.reset(self.seed)
        h, c = torch.zeros(1, B, H), torch.zeros(1, B, H)
        start = np.ones(B, dtype=bool)
        stats = EpisodeStats()
        per_iter = n * B
        n_iter = max(1, total_steps // per_iter)
        adim = () if self.spec.discrete else (self.spec.action_dim,)
        for it in range(n_iter):
            O = np.zeros((n, B, D), np.float32)
            A = np.zeros((n, B) + adim, np.float32)
            LP, R, V, DN = (np.zeros((n, B), np.float32) for _ in range(4))
            ST = np.zeros((n, B), bool)
            H0 = torch.zeros(n, B, H)
            C0 = torch.zeros(n, B, H)
            for k in range(n):
                H0[k], C0[k] = h[0], c[0]
                ST[k] = start
                with torch.no_grad():
                    f, (h, c) = self.ac.seq(self.t(obs)[None], h, c, torch.as_tensor(start)[None])
                    d = self.ac.dist(f[0])
                    a = d.sample()
                    lp, _ = self._lp(d, a)
                    v = self.ac.v(f[0]).squeeze(-1)
                a_np = a.numpy()
                obs2, r, term, trunc, info = env.step(a_np if self.spec.discrete else np.clip(a_np, -1, 1))
                r = r.astype(np.float32).copy()
                if trunc.any():
                    with torch.no_grad():
                        idx = np.where(trunc)[0]
                        f2, _ = self.ac.seq(self.t(info["final_obs"][idx])[None], h[:, idx], c[:, idx],
                                            torch.zeros(1, len(idx), dtype=torch.bool))
                        r[idx] += p["gamma"] * self.ac.v(f2[0]).squeeze(-1).numpy()
                O[k], A[k], LP[k], R[k], V[k] = obs, a_np, lp.numpy(), r, v.numpy()
                done = term | trunc
                DN[k] = done
                start = done.copy()
                obs = obs2
                stats.update(info)
            with torch.no_grad():
                f, _ = self.ac.seq(self.t(obs)[None], h, c, torch.as_tensor(start)[None])
                last_v = self.ac.v(f[0]).squeeze(-1).numpy()
            adv = np.zeros((n, B), np.float32)
            gae = np.zeros(B, np.float32)
            for k in reversed(range(n)):
                nv = last_v if k == n - 1 else V[k + 1]
                nt = 1.0 - DN[k]
                delta = R[k] + p["gamma"] * nv * nt - V[k]
                gae = delta + p["gamma"] * p["lam"] * nt * gae
                adv[k] = gae
            ret = adv + V
            diag = self._update(O, A, LP, adv, ret, ST, H0, C0, it, n_iter)
            diag["explained_var"] = float(1 - np.var(ret - V) / (np.var(ret) + 1e-12))
            if log is not None:
                row = {"step": self.env_steps + per_iter, **diag, **stats.pop()}
                if not self.spec.discrete:
                    row["policy_std"] = float(self.ac.log_std.exp().mean())
                log(row)
            if not self._tick(per_iter, callback):
                break
        return self

    def _update(self, O, A, LP, ADV, RET, ST, H0, C0, it, n_iter):
        p = self.p
        for g in self.opt.param_groups:
            g["lr"] = p["lr"] * (1.0 - it / n_iter)
        n, B = LP.shape
        L = p["chunk"]
        chunks = [(k0, b) for k0 in range(0, n, L) for b in range(B)]
        mb = max(1, len(chunks) // p["n_minibatch"])
        diag = {"pg_loss": [], "v_loss": [], "entropy": [], "kl": []}
        stop = False
        for _ in range(p["epochs"]):
            perm = self.rng.permutation(len(chunks))
            for s in range(0, len(chunks), mb):
                sel = [chunks[i] for i in perm[s: s + mb]]
                k0s = np.array([x[0] for x in sel])
                bs = np.array([x[1] for x in sel])
                rows = k0s[None, :] + np.arange(L)[:, None]  # (L, m)
                o = self.t(O[rows, bs[None, :]])
                st = torch.as_tensor(ST[rows, bs[None, :]])
                h0 = H0[k0s, bs][None]
                c0 = C0[k0s, bs][None]
                f, _ = self.ac.seq(o, h0, c0, st)
                d = self.ac.dist(f)
                a = torch.as_tensor(A[rows, bs[None, :]])
                if self.spec.discrete:
                    a = a.long()
                lp, ent = self._lp(d, a)
                lp_old = self.t(LP[rows, bs[None, :]])
                adv = self.t(ADV[rows, bs[None, :]])
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = (lp - lp_old).exp()
                pg = -torch.min(ratio * adv, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * adv).mean()
                v = self.ac.v(f).squeeze(-1)
                vl = ((self.t(RET[rows, bs[None, :]]) - v) ** 2).mean()
                loss = pg + p["vf_coef"] * vl - p["ent_coef"] * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), p["max_grad"])
                self.opt.step()
                with torch.no_grad():
                    kl = float(((ratio - 1) - (lp - lp_old)).mean())
                diag["pg_loss"].append(float(pg))
                diag["v_loss"].append(float(vl))
                diag["entropy"].append(float(ent.mean()))
                diag["kl"].append(kl)
                if p["target_kl"] and kl > 1.5 * p["target_kl"]:
                    stop = True
                    break
            if stop:
                break
        return {k: float(np.mean(v)) for k, v in diag.items() if v}
