"""Causal market-regime detection.

A K-state Gaussian hidden Markov model on daily log returns, fitted by EM on the TRAINING window only.
Downstream features use the *filtered* state probabilities P(s_t | x_1..x_t) from the forward pass.

Why this matters: the usual library call returns *smoothed* probabilities P(s_t | x_1..x_T) (forward-backward)
or a Viterbi path, both of which condition on future returns. A regime feature built that way is look-ahead,
and an RL agent will find it (see `leak_smoothed_regime` in the leakage experiments).
"""
from __future__ import annotations

import numpy as np

_LOG2PI = np.log(2 * np.pi)


class GaussianHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 200, tol: float = 1e-6, seed: int = 0):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.mu = self.var = self.A = self.pi = None

    # --- emissions
    def _logb(self, x):
        return -0.5 * (_LOG2PI + np.log(self.var)[None, :] + (x[:, None] - self.mu[None, :]) ** 2 / self.var[None, :])

    def fit(self, x: np.ndarray) -> "GaussianHMM":
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        K = self.K
        qs = np.quantile(np.abs(x - x.mean()), np.linspace(0.3, 0.9, K))
        self.mu = np.full(K, x.mean())
        self.var = np.maximum(qs**2, 1e-8)
        self.A = np.full((K, K), 0.02 / max(K - 1, 1))
        np.fill_diagonal(self.A, 0.98)
        self.pi = np.full(K, 1.0 / K)
        prev = -np.inf
        for _ in range(self.n_iter):
            logb = self._logb(x)
            alpha, c = self._forward(logb)
            beta = self._backward(logb, c)
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            b = np.exp(logb - logb.max(1, keepdims=True))
            # expected transition counts, vectorised over t
            m = alpha[:-1, :, None] * self.A[None] * (b[1:] * beta[1:])[:, None, :]
            xi = (m / m.sum(axis=(1, 2), keepdims=True)).sum(0)
            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi / xi.sum(1, keepdims=True)
            w = gamma / gamma.sum(0, keepdims=True)
            self.mu = (w * x[:, None]).sum(0)
            self.var = np.maximum((w * (x[:, None] - self.mu[None, :]) ** 2).sum(0), 1e-8)
            ll = np.log(c).sum()
            if abs(ll - prev) < self.tol * max(1.0, abs(ll)):
                break
            prev = ll
        order = np.argsort(self.var)  # state 0 = calmest
        self.mu, self.var = self.mu[order], self.var[order]
        self.A = self.A[np.ix_(order, order)]
        self.pi = self.pi[order]
        self.loglik_ = prev
        return self

    def _forward(self, logb):
        T, K = logb.shape
        alpha = np.zeros((T, K))
        c = np.zeros(T)
        m = logb.max(1, keepdims=True)
        b = np.exp(logb - m)
        a = self.pi * b[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * b[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        c = c * np.exp(m[:, 0])
        return alpha, c

    def _backward(self, logb, c):
        T, K = logb.shape
        m = logb.max(1, keepdims=True)
        b = np.exp(logb - m)
        cs = c / np.exp(m[:, 0])
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (b[t + 1] * beta[t + 1])) / cs[t + 1]
        return beta

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Filtered probabilities P(s_t | x_1..x_t); NaN observations propagate the prior one step."""
        x = np.asarray(x, float)
        T = len(x)
        out = np.zeros((T, self.K))
        stat = self.stationary()
        p = stat.copy()
        started = False
        for t in range(T):
            prior = p @ self.A if started else stat
            if np.isfinite(x[t]):
                lb = -0.5 * (_LOG2PI + np.log(self.var) + (x[t] - self.mu) ** 2 / self.var)
                post = prior * np.exp(lb - lb.max())
                s = post.sum()
                p = post / s if s > 0 else prior
                started = True
            else:
                p = prior
            out[t] = p
        return out

    def smooth(self, x: np.ndarray) -> np.ndarray:
        """Smoothed probabilities P(s_t | x_1..x_T). LOOK-AHEAD: provided only for the leakage experiments."""
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        xs = np.where(ok, x, 0.0)
        logb = self._logb(xs)
        logb[~ok] = 0.0
        alpha, c = self._forward(logb)
        beta = self._backward(logb, c)
        g = alpha * beta
        return g / g.sum(1, keepdims=True)

    def stationary(self) -> np.ndarray:
        w, v = np.linalg.eig(self.A.T)
        p = np.real(v[:, np.argmin(np.abs(w - 1))])
        p = np.abs(p)
        return p / p.sum()

    def summary(self) -> dict:
        return {"mu_ann": (self.mu * 252).round(4).tolist(), "vol_ann": (np.sqrt(self.var * 252)).round(4).tolist(),
                "persistence": np.diag(self.A).round(4).tolist()}
