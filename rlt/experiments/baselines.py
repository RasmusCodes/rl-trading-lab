"""Baseline benchmark suite: every baseline through the same environment, fold by fold.

Outputs (runs/baselines/<universe>/):
    summary.csv          one row per (strategy, fold, split, asset) with all metrics
    oos.csv              stitched walk-forward test record per strategy (all test blocks concatenated)
    ledgers/<name>.csv   daily test ledgers (date, asset, ret, rf, turnover, cost, gross, w)
    supervised.json      chosen hyper-parameters per fold and how many configurations were tried
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from ..baselines import optimization as opt
from ..baselines import rules, supervised
from ..data.features import FEATURE_SETS
from ..evaluation.backtest import WeightPolicy, ledger_stats, run_policy
from ..evaluation.metrics import perf_stats
from ..training.context import FoldContext, env_config, get_folds, get_market
from ..utils.io import RUNS_DIR, safe_name, save_json

FAMILY = {
    "buy_hold": "passive", "equal_weight": "passive", "sixty_forty": "passive",
    "tsmom_252": "momentum", "tsmom_252_lo": "momentum", "tsmom_vs": "momentum", "ma_cross": "momentum",
    "tsmom_port": "momentum",
    "meanrev_5": "mean_reversion", "rsi2_dip": "mean_reversion",
    "vol_target": "optimization", "mv_kelly": "optimization", "inverse_vol": "optimization",
    "risk_parity": "optimization", "min_variance": "optimization", "mean_variance": "optimization",
    "sup_ridge": "supervised", "sup_logistic": "supervised", "sup_gbm": "supervised",
}


def family(name: str) -> str:
    return "random" if name.startswith("random") else FAMILY.get(name, "other")


def strategy_tables(md, mode: str, n_random: int) -> dict:
    if mode == "pool":
        W = rules.all_rule_weights(md, random_seeds=range(n_random))
        W["mv_kelly"] = opt.mv_kelly(md)
        return W
    W = {k: opt.portfolio_weights(md, k) for k in opt.PORTFOLIO_BASELINES}
    N = md.N
    for s in range(n_random):
        W[f"random_{s}"] = rules.random_markov(md, seed=s) / N
    W["tsmom_vs_ew"] = rules.tsmom(md, vol_target=0.15) / N
    W["vol_target_ew"] = rules.vol_target(md) / N
    return W


def _ledger_rows(led_list, name, fold, split):
    rows = []
    for led in led_list:
        sym = led.attrs["symbols"]
        df = led[["ret", "rf", "turnover", "cost", "gross", "net"]].copy()
        df["asset"] = sym[0] if len(sym) == 1 else "portfolio"
        df["strategy"], df["fold"], df["split"] = name, fold, split
        rows.append(df.reset_index())
    return rows


def _eval(env, W, name, fold, split, bench=None):
    leds = run_policy(env, WeightPolicy(W, name))
    stats = []
    for led in leds:
        st = ledger_stats(led)
        sym = led.attrs["symbols"]
        st.update(strategy=name, family=family(name), fold=fold, split=split,
                  asset=sym[0] if len(sym) == 1 else "portfolio")
        stats.append(st)
    return leds, stats


def run_baseline_suite(universe: str, mode: str | None = None, feature_set: str = "compact", n_random: int = 20,
                       folds=None, out_name: str | None = None, cfg_overrides: dict | None = None,
                       supervised_kinds=("ridge", "logistic", "gbm"), verbose: bool = True) -> dict:
    t0 = time.time()
    md = get_market(universe)
    cfg = env_config(universe, mode, **(cfg_overrides or {}))
    mode = cfg.mode
    folds = folds or get_folds(md, universe)
    tables = strategy_tables(md, mode, n_random)
    out = RUNS_DIR / "baselines" / safe_name(out_name or universe)
    (out / "ledgers").mkdir(parents=True, exist_ok=True)
    all_stats, ledger_rows, sup_log = [], [], []
    y = supervised.labels(md)
    for fold in folds:
        ctx = FoldContext(md, fold, FEATURE_SETS[feature_set])
        envs = {s: ctx.eval_env(cfg, s) for s in ("val", "test")}
        for name, W in tables.items():
            for split, env in envs.items():
                leds, st = _eval(env, W, name, fold.name, split)
                all_stats += st
                if split == "test":
                    ledger_rows += _ledger_rows(leds, name, fold.name, split)
        # supervised: fit on train, choose config on validation, report test
        assets = list(range(md.N))
        for kind in supervised_kinds:
            best, tried = None, 0
            cache = {}
            for p in supervised.configs(kind):
                key = tuple((k, v) for k, v in p.items() if k not in ("k", "long_only"))
                if key not in cache:
                    cache[key] = supervised.fit_predict(kind, ctx.X, y, ctx.ready, fold.train, assets, p)
                pred, sd = cache[key]
                W = supervised.to_weights(pred, sd, p["k"], p["long_only"])
                if mode == "portfolio":
                    W = W / md.N
                _, st = _eval(envs["val"], W, f"sup_{kind}", fold.name, "val")
                tried += 1
                score = float(np.mean([s.get("sharpe", 0.0) for s in st]))
                if best is None or score > best[0]:
                    best = (score, p, W, st)
            score, p, W, st = best
            all_stats += st
            leds, st_te = _eval(envs["test"], W, f"sup_{kind}", fold.name, "test")
            all_stats += st_te
            ledger_rows += _ledger_rows(leds, f"sup_{kind}", fold.name, "test")
            sup_log.append({"fold": fold.name, "kind": kind, "params": p, "val_sharpe": score, "configs_tried": tried})
        if verbose:
            print(f"[baselines:{universe}] {fold.name} done ({time.time() - t0:.0f}s)", flush=True)

    summary = pd.DataFrame(all_stats)
    summary.to_csv(out / "summary.csv", index=False)
    led = pd.concat(ledger_rows, ignore_index=True)
    for name, g in led.groupby("strategy"):
        g.drop(columns=["strategy", "split"]).to_csv(out / "ledgers" / f"{name}.csv", index=False)
    if not any(f.name == "holdout" for f in folds) and not universe.startswith("syn"):
        # real-data development records stop before the holdout starts (synthetic calendars are fabricated)
        led = led[pd.to_datetime(led["date"]) < pd.Timestamp("2019-01-01")]
    oos = stitched_oos(led)
    oos.to_csv(out / "oos.csv")
    save_json(out / "supervised.json", sup_log)
    save_json(out / "meta.json", {"universe": universe, "mode": mode, "feature_set": feature_set,
                                  "folds": [f.to_dict() for f in folds], "env": cfg, "n_random": n_random,
                                  "seconds": round(time.time() - t0, 1)})
    return {"summary": summary, "oos": oos, "ledgers": led}


def stitched_oos(led: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward out-of-sample record: test blocks concatenated, averaged across pool instruments."""
    rows = {}
    for name, g in led.groupby("strategy"):
        daily = g.groupby("date")[["ret", "rf", "turnover", "cost", "gross", "net"]].mean().sort_index()
        st = perf_stats(daily["ret"].to_numpy(), daily["rf"].to_numpy(), daily["turnover"].to_numpy(),
                        daily["cost"].to_numpy(), daily["gross"].to_numpy(), daily["net"].to_numpy(), ci=True)
        per_asset = [perf_stats(a["ret"].to_numpy(), a["rf"].to_numpy()).get("sharpe", np.nan)
                     for _, a in g.groupby("asset")]
        st["mean_asset_sharpe"] = float(np.nanmean(per_asset))
        st["family"] = family(name)
        rows[name] = st
    return pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)


def random_null(oos: pd.DataFrame) -> dict:
    r = oos[oos.index.str.startswith("random_")]["sharpe"].astype(float)
    return {"n": int(len(r)), "mean": float(r.mean()), "p05": float(r.quantile(0.05)), "p95": float(r.quantile(0.95))}


def cost_free(cfg_overrides: dict | None = None) -> dict:
    from ..env.costs import CostConfig

    d = dict(cfg_overrides or {})
    d["costs"] = replace(CostConfig(), multiplier=0.0, borrow=False, financing_spread_bps=0.0)
    return d
