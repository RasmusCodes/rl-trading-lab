"""Re-evaluate trained checkpoints and baselines under different AUM, cost multipliers and fill timing.

    python scripts/run_robustness.py            -> runs/robustness/sweeps.csv, capacity.csv
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from rlt.utils.io import RUNS_DIR  # noqa: E402

FOLDS = [f"wf{k}_{y}" for k, y in zip(range(1, 7), range(2007, 2019, 2))]
BASELINES = ["buy_hold", "ma_cross", "vol_target", "meanrev_5", "tsmom_252_lo"]


def _run_job(job):
    import torch

    torch.set_num_threads(1)
    from rlt.evaluation.robustness import sweep_run, sweep_weights

    kind, arg = job
    try:
        if kind == "run":
            df = sweep_run(Path(arg))
            df["label"] = df["algo"]
            return df
        name, fold = arg
        from rlt.baselines import rules
        from rlt.training.context import get_market

        md = get_market("spy")
        W = {"buy_hold": rules.buy_hold, "ma_cross": rules.ma_cross, "vol_target": rules.vol_target,
             "meanrev_5": rules.meanrev, "tsmom_252_lo": lambda m: rules.tsmom(m, long_only=True)}[name](md)
        df = sweep_weights("spy", fold, name, W)
        df["label"] = name
        return df
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame([{"error": repr(e), "job": str(job)}])


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "main_spy"
    jobs = [("run", str(p.parent)) for p in sorted((RUNS_DIR / exp).glob("*/model.pt"))]
    jobs += [("base", (b, f)) for b in BASELINES for f in FOLDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 10))) as ex:
        dfs = list(ex.map(_run_job, jobs, chunksize=2))
    df = pd.concat(dfs, ignore_index=True)
    out = RUNS_DIR / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweeps.csv", index=False)
    errs = df[df["error"].notna()] if "error" in df else []
    print(f"{len(jobs)} jobs, {len(errs)} errors")
    from rlt.evaluation.robustness import capacity

    rows = []
    good = df[df["error"].isna()] if "error" in df else df
    bh = good[(good["label"] == "buy_hold") & (good["kind"] == "aum")].groupby("x")["sharpe"].mean().reset_index()
    for label, g in good[good["kind"] == "aum"].groupby("label"):
        a = g.groupby("x")["sharpe"].mean().reset_index()
        rows.append({"label": label, **capacity(a, bh)})
    pd.DataFrame(rows).to_csv(out / "capacity.csv", index=False)
    print(pd.DataFrame(rows).to_string())
