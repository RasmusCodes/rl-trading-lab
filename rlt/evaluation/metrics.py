"""Performance and significance metrics.

Sharpe ratios are always on returns in excess of the T-bill rate. Statistical tools:
* Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012): P(true SR > SR*) given skew/kurtosis and length.
* Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): PSR against the expected maximum Sharpe of N trials
  under the null, which is how many configurations were tried before this one was picked.
* Stationary-bootstrap (Politis & Romano 1994) confidence intervals and paired Sharpe-difference tests.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

ANN = 252
EULER_GAMMA = 0.5772156649


FLAT_SD = 1e-9  # daily excess-return sd below this is a flat book: its "Sharpe" is 0/0 plus float residue


def _sr(x):
    """Per-period Sharpe. A flat book (cash only: ret == rf up to ~1e-17 float error) has no Sharpe ratio; return
    0 rather than the ratio of two rounding errors, which can be any number (it was 0.4-1.2 for flat PPO seeds,
    and validation-time checkpoint selection and the HPO objective both consume this statistic)."""
    sd = x.std(ddof=1)
    return x.mean() / sd if sd > FLAT_SD else 0.0


def drawdown_stats(ret: np.ndarray) -> tuple[float, int]:
    eq = np.cumprod(1.0 + ret)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = eq / peak - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0
    # longest stretch under water (sessions)
    longest = cur = 0
    for v in dd:
        cur = cur + 1 if v < -1e-12 else 0
        longest = max(longest, cur)
    return mdd, longest


def psr(sr_per_period: float, n: int, sk: float, ku: float, sr_star: float = 0.0) -> float:
    """ku = non-excess kurtosis."""
    if n < 3:
        return float("nan")
    den = 1.0 - sk * sr_per_period + (ku - 1.0) / 4.0 * sr_per_period**2
    if den <= 0:
        return float("nan")
    return float(norm.cdf((sr_per_period - sr_star) * np.sqrt(n - 1) / np.sqrt(den)))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """Expected maximum per-period Sharpe among n_trials zero-skill strategies with SR variance var_sr."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
                                    + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * np.e))))


def deflated_sharpe(ex_ret: np.ndarray, n_trials: int, trial_srs_annual) -> float:
    x = np.asarray(ex_ret, float)
    sr = _sr(x)
    var = np.var(np.asarray(trial_srs_annual, float) / np.sqrt(ANN), ddof=1) if len(trial_srs_annual) > 1 else 0.0
    sr0 = expected_max_sr(n_trials, var)
    return psr(sr, len(x), float(skew(x)), float(kurtosis(x, fisher=False)), sr0)


def stationary_bootstrap_idx(n: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    jumps = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(jumps[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_sharpe_ci(ex_ret, n_boot: int = 1000, mean_block: float = 20.0, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(ex_ret, float)
    if len(x) < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(len(x), n_boot, mean_block, rng)
    s = x[idx]
    sd = s.std(1, ddof=1)
    srs = np.where(sd > 0, s.mean(1) / np.where(sd > 0, sd, 1), 0.0) * np.sqrt(ANN)
    return (float(np.quantile(srs, alpha / 2)), float(np.quantile(srs, 1 - alpha / 2)))


def paired_sharpe_test(ex_a, ex_b, n_boot: int = 2000, mean_block: float = 20.0, seed: int = 0) -> dict:
    """Bootstrap test of SR(a) - SR(b) on the same dates. Returns the difference, CI and one-sided p (a <= b)."""
    a, b = np.asarray(ex_a, float), np.asarray(ex_b, float)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(n, n_boot, mean_block, rng)

    def srs(x):
        s = x[idx]
        sd = s.std(1, ddof=1)
        return np.where(sd > 0, s.mean(1) / np.where(sd > 0, sd, 1), 0.0) * np.sqrt(ANN)

    d = srs(a) - srs(b)
    obs = (_sr(a) - _sr(b)) * np.sqrt(ANN)
    centred = d - d.mean()  # null-centred distribution
    p = float((centred >= obs).mean())
    return {"sr_diff": float(obs), "ci": (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))), "p_value": p}


def perf_stats(ret, rf=None, turnover=None, cost=None, gross=None, net=None, bench_ret=None, ci: bool = False) -> dict:
    """Summary statistics for a daily net-return series (simple returns on equity)."""
    r = np.asarray(ret, float)
    n = len(r)
    rf = np.zeros(n) if rf is None else np.asarray(rf, float)
    ex = r - rf
    out: dict = {"days": n}
    if n < 2:
        return out
    years = n / ANN
    growth = float(np.prod(1.0 + r))
    out["total_return"] = growth - 1.0
    out["cagr"] = growth ** (1.0 / years) - 1.0 if growth > 0 else -1.0
    out["vol"] = float(r.std(ddof=1) * np.sqrt(ANN))
    sr = _sr(ex)
    out["sharpe"] = float(sr * np.sqrt(ANN))
    out["flat"] = bool(ex.std(ddof=1) <= FLAT_SD)
    dn = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    out["sortino"] = float(ex.mean() / dn * np.sqrt(ANN)) if dn > FLAT_SD else 0.0
    mdd, dur = drawdown_stats(r)
    out["max_drawdown"] = mdd
    out["max_dd_days"] = dur
    out["calmar"] = out["cagr"] / abs(mdd) if mdd < 0 else float("nan")
    out["skew"] = float(skew(ex)) if ex.std() > 0 else 0.0
    out["kurtosis"] = float(kurtosis(ex, fisher=True)) if ex.std() > 0 else 0.0
    out["hit_rate"] = float((ex > 0).mean())
    q = np.quantile(r, 0.05)
    out["var95"] = float(q)
    out["cvar95"] = float(r[r <= q].mean()) if (r <= q).any() else float(q)
    out["psr"] = psr(sr, n, out["skew"], out["kurtosis"] + 3.0) if ex.std() > 0 else float("nan")
    if turnover is not None:
        out["turnover"] = float(np.mean(turnover) * ANN)  # sum|dw| per year (two-sided)
    if cost is not None:
        out["cost_drag"] = float(np.mean(cost) * ANN)  # fraction of equity per year
    if gross is not None:
        out["avg_gross"] = float(np.mean(gross))
    if net is not None:
        out["avg_net"] = float(np.mean(net))
    if turnover is not None and gross is not None and out["turnover"] > 1e-9:
        out["holding_days"] = float(2 * out["avg_gross"] / out["turnover"] * ANN)
    if bench_ret is not None:
        b = np.asarray(bench_ret, float)[-n:] - rf[-n:]
        if b.std() > 0:
            beta = float(np.cov(ex, b, ddof=1)[0, 1] / b.var(ddof=1))
            out["beta"] = beta
            out["alpha"] = float((ex.mean() - beta * b.mean()) * ANN)
            act = ex - b
            out["info_ratio"] = float(act.mean() / act.std(ddof=1) * np.sqrt(ANN)) if act.std() > 0 else 0.0
            out["corr"] = float(np.corrcoef(ex, b)[0, 1])
    if ci:
        lo, hi = bootstrap_sharpe_ci(ex)
        out["sharpe_ci_lo"], out["sharpe_ci_hi"] = lo, hi
    return out
