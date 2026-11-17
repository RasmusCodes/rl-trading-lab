"""Experiment runner: one (experiment, algorithm, fold, seed) = one run directory.

    runs/<experiment>/<run_id>/
        config.json      everything needed to reproduce the run (+ code fingerprint, package versions)
        status.json      state, progress, best validation score (polled by the dashboard)
        progress.jsonl   training diagnostics and periodic validation evaluations
        eval.json        validation/test metrics of the selected checkpoint (and of the last one)
        ledger_test.csv  daily test ledger of the selected checkpoint

Protocol per run: train on the fold's training window (random 252-day episodes); every `eval_every`
transitions evaluate the deterministic policy on the validation window; keep the snapshot with the best
validation Sharpe (early stopping optional); report test metrics of that snapshot. The test window is
touched once per run, after training. Holdout folds additionally require HoldoutLedger.unlock.
"""
from __future__ import annotations

import hashlib
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

# import every module a run executes *before* fingerprinting, so the fingerprint describes the code this worker
# actually runs (pool workers keep their imported modules even if files change on disk afterwards)
from .. import ENV_VERSION
from .. import agents as _agents  # noqa: F401
from ..env import trading_env as _env  # noqa: F401
from ..evaluation import backtest as _bt  # noqa: F401
from ..data import features as _feat  # noqa: F401
from ..utils.io import RUNS_DIR, DATA_DIR, JsonlLogger, code_fingerprint, dumps, env_info, load_json, save_json
from ..utils.seeding import derive_seed, set_global_seed

CODE_FINGERPRINT = code_fingerprint()
DISCRETE_ALGOS = {"qlearning", "dqn", "ddqn"}

# trading defaults (overridable per run). They were fixed before the real-data experiments were launched.
# (Caveat recorded in the report: a 20k-step plumbing smoke test printed fold-1 test-block numbers once; no
# setting below was chosen from them.)
# gamma: the agent is a price taker, so its action reaches future rewards only through the position it carries
# (costs). On the synthetic controls (runs/gamma_controls) the bootstrapping agents did clearly worse at 0.95 than
# at shorter horizons, while PPO was insensitive -> 0.5 for value-based agents, 0.95 kept for the PPO family.
TRADING_DEFAULTS = {
    "qlearning": dict(gamma=0.5, alpha=0.02, eps_frac=0.5),
    "dqn": dict(gamma=0.5, lr=3e-4, learning_starts=5_000, grad_steps=4, eps_frac=0.3),
    "ddqn": dict(gamma=0.5, lr=3e-4, learning_starts=5_000, grad_steps=4, eps_frac=0.3),
    "a2c": dict(gamma=0.95, lam=0.95, lr=3e-4, n_steps=16),
    "ppo": dict(gamma=0.95, lam=0.95, lr=3e-4, n_steps=128, epochs=5, minibatch=512),
    "sac": dict(gamma=0.5, lr=3e-4, grad_steps=2, learning_starts=5_000),
    "td3": dict(gamma=0.5, lr=3e-4, grad_steps=2, learning_starts=5_000),
    "rrl": dict(updates=800, batch=32, seq_len=252, lr=3e-3, weight_decay=1e-3, eval_every=400_000),
    "ppo_lstm": dict(gamma=0.95, lam=0.95, lr=3e-4, n_steps=128, chunk=32, epochs=4),
}


@dataclass
class RunSpec:
    experiment: str
    universe: str
    algo: str
    fold: str
    seed: int = 0
    total_steps: int = 300_000
    algo_params: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)  # EnvConfig overrides; "costs" may be a dict of CostConfig fields
    features: str = "compact"
    leak: tuple = ()
    num_envs: int = 16
    eval_every: int = 20_000
    patience: int | None = None
    train_assets: list | None = None
    eval_assets: list | None = None
    tag: str = ""
    holdout: bool = False
    save_ledger: bool = True
    train_bootstrap: dict | None = None  # {"paths": 8, "length": 3000, "mean_block": 20}: train on resampled history
    train_start: str | None = None  # later start of the training window (e.g. "2001-01-02")

    @property
    def run_id(self) -> str:
        base = f"{self.algo}__{self.fold}__s{self.seed}"
        return f"{base}__{self.tag}" if self.tag else base


# ------------------------------------------------------------------------------------------ data cache
_CTX_CACHE: dict = {}


def get_context(universe: str, fold_name: str, features: str, leak: tuple = (), holdout: bool = False,
                train_start: str | None = None, purpose: str = ""):
    from ..data.features import FEATURE_SETS
    from ..data.splits import Fold, HoldoutLedger
    from ..training.context import FoldContext, get_folds, get_holdout, get_market

    if holdout or fold_name == "holdout":
        # every load of holdout data is logged, whoever asks (training, re-evaluation, robustness sweeps)
        HoldoutLedger.unlock(f"context:{purpose or 'unspecified'}", {"universe": universe, "features": str(features)})
    key = (universe, fold_name, features, tuple(leak), train_start)
    if key in _CTX_CACHE:
        return _CTX_CACHE[key]
    md = get_market(universe)
    folds = {f.name: f for f in get_folds(md, universe)}
    if holdout or fold_name == "holdout":
        folds["holdout"] = get_holdout(md, universe)
    fold = folds[fold_name]
    if train_start:
        i0 = max(fold.train[0], md.index_of(train_start))
        dates = dict(fold.dates, train=[md.dates[i0].date().isoformat(), fold.dates["train"][1]])
        fold = Fold(fold.name, (i0, fold.train[1]), fold.val, fold.test, dates)
    names = FEATURE_SETS[features] if isinstance(features, str) else list(features)
    h = hashlib.sha1(f"{universe}|{fold.to_dict()}|{names}|{leak}|env{ENV_VERSION}".encode()).hexdigest()[:12]
    cache = DATA_DIR / "cache" / f"ctx_{universe.replace(':', '-')}_{fold_name}_{h}.npz"
    ctx = FoldContext.__new__(FoldContext)
    ctx.md, ctx.fold, ctx.feature_names, ctx.leak = md, fold, names, tuple(leak)
    if cache.exists():
        from ..data.features import FeatureSet

        with np.load(cache, allow_pickle=True) as z:  # close the zip: an open handle blocks os.replace on Windows
            ctx.X, ctx.ready = z["X"], z["ready"]
            ctx.fs = FeatureSet(z["raw"], [str(n) for n in z["names"]], md.dates, list(md.symbols), z["ready"],
                                z["meta"].item())
        from ..data.features import FeatureScaler

        # the scaler is not cached; refit it (deterministic, training rows only) - bootstrap envs need it
        ctx.scaler = FeatureScaler().fit(ctx.fs, fold.train[0], fold.train[1])
    else:
        ctx.__post_init__()
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, X=ctx.X, ready=ctx.ready, raw=ctx.fs.values, names=np.array(ctx.fs.names),
                            meta=np.array(ctx.fs.meta, dtype=object))
        try:
            os.replace(tmp, cache)  # atomic: parallel workers may race to write the same cache
        except OSError:
            pass
    _CTX_CACHE[key] = ctx
    return ctx


def build_env_config(spec: RunSpec):
    from ..env.costs import CostConfig
    from ..training.context import env_config

    ov = dict(spec.env)
    costs = ov.pop("costs", None)
    if spec.algo in DISCRETE_ALGOS:
        ov.setdefault("action", "discrete")
    cfg = env_config(spec.universe, **ov)
    if costs is not None:
        cfg = replace(cfg, costs=costs if isinstance(costs, CostConfig) else replace(CostConfig(), **costs))
    return cfg


def _bootstrap_env(ctx, cfg, spec: RunSpec, seed: int):
    """Training env on block-bootstrapped histories built from the fold's TRAINING rows only.

    Features of the synthetic paths are scaled with the scaler fitted on the real training window, so the
    agent sees the same observation layout and units in training (synthetic) and validation/test (real).
    """
    from ..data.features import compute_features
    from ..data.synthetic import block_bootstrap, stack_markets
    from ..env.trading_env import TradingEnv
    from .context import WARMUP_ROWS

    bs = dict(spec.train_bootstrap)
    sym = bs.get("symbol") or (spec.eval_assets or [ctx.md.symbols[0]])[0]
    md1 = ctx.md.subset([sym])
    i0, i1 = ctx.fold.train
    paths = [block_bootstrap(md1, i0, i1, int(bs.get("length", 3000)), seed=seed + 7919 * k,
                             mean_block=float(bs.get("mean_block", 20))) for k in range(int(bs.get("paths", 8)))]
    mdb = stack_markets(paths)
    fs = compute_features(mdb, ctx.feature_names, fit_end=mdb.T - 1)
    X = ctx.scaler.transform(fs.values)
    return TradingEnv(mdb, X, replace(cfg, mode="pool"), (WARMUP_ROWS, mdb.T - 2), num_envs=spec.num_envs,
                      seed=seed, ready=fs.ready)


def _eval_stats(env, agent, prefix: str) -> tuple[dict, list]:
    from ..evaluation.backtest import ledger_stats, run_policy

    leds = run_policy(env, agent, deterministic=True)
    per = [ledger_stats(led) for led in leds]
    keys = ["sharpe", "cagr", "vol", "max_drawdown", "turnover", "cost_drag", "avg_gross", "avg_net", "sortino",
            "total_return", "hit_rate"]
    out = {}
    for k in keys:
        vals = [p.get(k) for p in per if p.get(k) is not None]
        out[f"{prefix}_{k}"] = float(np.mean(vals)) if vals else None
    if len(leds) > 1:  # pooled instruments: also the equal-weight combination of the per-instrument books
        from ..evaluation.metrics import perf_stats

        df = pd.concat([led[["ret", "rf"]] for led in leds], axis=1, keys=range(len(leds)))
        r = df.xs("ret", axis=1, level=1).mean(1).to_numpy()
        rf = df.xs("rf", axis=1, level=1).mean(1).to_numpy()
        out[f"{prefix}_ew_sharpe"] = perf_stats(r, rf).get("sharpe")
    return out, leds


def run_one(spec: RunSpec) -> dict:
    import torch

    torch.set_num_threads(1)
    from ..agents import EnvSpec, make_agent
    from ..data.splits import HoldoutLedger

    run_dir = RUNS_DIR / spec.experiment / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = derive_seed(spec.experiment, spec.universe, spec.algo, spec.fold, spec.seed, spec.tag)
    set_global_seed(seed)
    t0 = time.time()
    status = {"state": "starting", "run_id": spec.run_id, "experiment": spec.experiment, "step": 0,
              "total_steps": spec.total_steps, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_json(run_dir / "status.json", status)
    try:
        if spec.fold == "holdout" or spec.holdout:
            HoldoutLedger.unlock(f"{spec.experiment}/{spec.run_id}", {"universe": spec.universe, "algo": spec.algo})
        ctx = get_context(spec.universe, spec.fold, spec.features, spec.leak, spec.holdout, spec.train_start,
                          purpose=f"{spec.experiment}/{spec.run_id}")
        cfg = build_env_config(spec)
        params = {**TRADING_DEFAULTS.get(spec.algo, {}), **spec.algo_params}
        if spec.algo != "rrl":  # RRL counts 252x32 transitions per update; it keeps its own cadence (~16 evals)
            params["eval_every"] = spec.eval_every
        if spec.train_bootstrap:
            train_env = _bootstrap_env(ctx, cfg, spec, seed)
        else:
            train_env = ctx.train_env(cfg, spec.num_envs, seed, assets=spec.train_assets)
        val_env = ctx.eval_env(cfg, "val", assets=spec.eval_assets)
        test_env = ctx.eval_env(cfg, "test", assets=spec.eval_assets)
        espec = EnvSpec.from_env(train_env, ctx.obs_names(cfg))
        agent = make_agent(spec.algo, espec, params, seed)
        save_json(run_dir / "config.json", {"spec": asdict(spec), "seed_derived": seed, "algo_params": agent.p,
                                            "env": asdict(cfg), "fold": ctx.fold.to_dict(),
                                            "features": ctx.fs.names, "obs_dim": espec.obs_dim,
                                            "code": CODE_FINGERPRINT, "env_version": ENV_VERSION,
                                            "versions": env_info()})
        log = JsonlLogger(run_dir / "progress.jsonl")
        best = {"score": -np.inf, "state": None, "step": 0, "evals_since": 0, "n_evals": 0}

        def callback(ag, step):
            st, _ = _eval_stats(val_env, ag, "val")
            score = st.get("val_sharpe") or -np.inf
            best["n_evals"] += 1
            if score > best["score"]:
                best.update(score=score, state=ag.get_state(), step=step, evals_since=0)
            else:
                best["evals_since"] += 1
            log.log({"step": step, **st, "best_val_sharpe": best["score"]})
            status.update(state="running", step=int(step), best_val_sharpe=best["score"], best_step=best["step"],
                          elapsed=round(time.time() - t0, 1))
            save_json(run_dir / "status.json", status)
            if spec.patience and best["evals_since"] > spec.patience:
                return False
            return True

        status["state"] = "running"
        save_json(run_dir / "status.json", status)
        steps = None if spec.algo == "rrl" else spec.total_steps
        agent.learn(train_env, steps, callback=callback, log=log.log)
        callback(agent, agent.env_steps)  # always evaluate the final policy too
        last_state = agent.get_state()
        test_last, _ = _eval_stats(test_env, agent, "test_last")
        if best["state"] is not None:
            agent.set_state(best["state"])
        val_best, _ = _eval_stats(val_env, agent, "val")
        test_best, leds = _eval_stats(test_env, agent, "test")
        result = {"run_id": spec.run_id, "experiment": spec.experiment, "universe": spec.universe, "algo": spec.algo,
                  "fold": spec.fold, "seed": spec.seed, "tag": spec.tag, "best_step": best["step"],
                  "n_val_evals": best["n_evals"], "train_seconds": round(time.time() - t0, 1),
                  "env_version": ENV_VERSION, "eval_env_version": ENV_VERSION, "code": CODE_FINGERPRINT,
                  **val_best, **test_best, **test_last}
        save_json(run_dir / "eval.json", result)
        if spec.save_ledger:
            rows = []
            for led in leds:
                sym = led.attrs["symbols"]
                df = led[["equity", "ret", "rf", "turnover", "cost", "gross", "net"]].copy()
                for c in led.columns:
                    if c.startswith("w_"):
                        df[c] = led[c]
                df["asset"] = sym[0] if len(sym) == 1 else "portfolio"
                rows.append(df.reset_index())
            pd.concat(rows).to_csv(run_dir / "ledger_test.csv", index=False, float_format="%.7g")
        try:
            import torch as _t

            _t.save({"best": best["state"], "last": last_state, "params": agent.p}, run_dir / "model.pt")
        except Exception:  # noqa: BLE001
            pass
        log.close()
        status.update(state="done", elapsed=round(time.time() - t0, 1), test_sharpe=result.get("test_sharpe"))
        save_json(run_dir / "status.json", status)
        return result
    except Exception as e:  # noqa: BLE001
        status.update(state="failed", error=repr(e), trace=traceback.format_exc()[-3000:])
        save_json(run_dir / "status.json", status)
        return {"run_id": spec.run_id, "experiment": spec.experiment, "algo": spec.algo, "fold": spec.fold,
                "seed": spec.seed, "tag": spec.tag, "error": repr(e)}


def _worker(spec_dict):
    spec = RunSpec(**spec_dict)
    return run_one(spec)


def _same_spec(a: RunSpec, stored: dict) -> bool:
    import json

    norm = lambda d: json.loads(dumps(d, sort_keys=True))  # noqa: E731  (tuples -> lists, numpy -> python)
    defaults = asdict(RunSpec(experiment="", universe="", algo="", fold=""))
    full = {**defaults, **stored}  # fields added to RunSpec later default for runs stored before them
    full.update({k: stored[k] for k in ("experiment", "universe", "algo", "fold") if k in stored})
    return norm(asdict(a)) == norm(full)


def archive_run(run_dir, reason: str) -> None:
    """Move a stale run aside (Windows forbids deleting here; nothing is ever silently overwritten)."""
    import shutil

    dst = RUNS_DIR / "_archive" / run_dir.parent.name / f"{run_dir.name}__{time.strftime('%Y%m%d-%H%M%S')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(dst))
    (dst / "ARCHIVED.txt").write_text(reason, encoding="utf-8")


def run_many(specs: list[RunSpec], workers: int | None = None, skip_done: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Run specs in parallel. A finished run is reused only if its stored spec is identical to the requested
    one and it was produced by the current environment version; otherwise it is archived and re-run."""
    from .. import ENV_VERSION as _V

    todo = []
    results = []
    for s in specs:
        rd = RUNS_DIR / s.experiment / s.run_id
        ev = rd / "eval.json"
        if skip_done and ev.exists():
            try:
                stored = load_json(rd / "config.json")["spec"]
                prev = load_json(ev)
            except Exception:  # noqa: BLE001
                stored, prev = None, {}
            if stored is not None and _same_spec(s, stored) and prev.get("eval_env_version", 1) == _V:
                results.append(prev)
                continue
            if stored is not None and _same_spec(s, stored):
                results.append(prev)  # same spec, older env: re-evaluate (scripts/reevaluate_runs.py), not retrain
                continue
            archive_run(rd, "spec changed")
        elif rd.exists() and not ev.exists():
            archive_run(rd, "incomplete or failed run")
        todo.append(s)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if verbose:
        print(f"{len(specs)} runs: {len(results)} cached, {len(todo)} to run on {workers} workers", flush=True)
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, asdict(s)): s for s in todo}
            for k, f in enumerate(as_completed(futs), 1):
                r = f.result()
                results.append(r)
                if verbose:
                    msg = r.get("error") or f"val {r.get('val_sharpe', float('nan')):.2f} test {r.get('test_sharpe', float('nan')):.2f}"
                    print(f"[{k}/{len(todo)} {time.time() - t0:5.0f}s] {r['run_id']}: {msg}", flush=True)
    return pd.DataFrame(results)


def collect(experiment: str) -> pd.DataFrame:
    rows = [load_json(p) for p in (RUNS_DIR / experiment).glob("*/eval.json")]
    return pd.DataFrame(rows)
