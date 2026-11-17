"""Re-score finished runs under the current environment version (no retraining).

Used after the 2026-11-14 audit fixes (ENV_VERSION 2): the checkpoints chosen during training are kept, and
their validation/test P&L is recomputed with the corrected accounting (limited liability, untruncated
evaluation windows, short-rebate haircut, short-sale bans, split-consistent ADV). The previous evaluation is
kept as eval_v1.json / ledger_test_v1.csv.

    python scripts/reevaluate_runs.py main_spy rewards_spy hacking_spy leakage controls gamma_controls
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt.utils.io import RUNS_DIR, load_json  # noqa: E402


def reevaluate(run_dir: str) -> str:
    import shutil

    import pandas as pd
    import torch

    torch.set_num_threads(1)
    from rlt import ENV_VERSION
    from rlt.evaluation.robustness import load_agent
    from rlt.training.runner import CODE_FINGERPRINT, _eval_stats
    from rlt.utils.io import save_json

    rd = Path(run_dir)
    ev = load_json(rd / "eval.json")
    if ev.get("eval_env_version", 1) >= ENV_VERSION:
        return f"{rd.name}: already v{ENV_VERSION}"
    agent, spec, ctx, ecfg = load_agent(rd, "last")
    val_env = ctx.eval_env(ecfg, "val", assets=spec.eval_assets)
    test_env = ctx.eval_env(ecfg, "test", assets=spec.eval_assets)
    test_last, _ = _eval_stats(test_env, agent, "test_last")
    agent, *_ = load_agent(rd, "best")
    val_best, _ = _eval_stats(val_env, agent, "val")
    test_best, leds = _eval_stats(test_env, agent, "test")
    if not (rd / "eval_v1.json").exists():
        shutil.copy(rd / "eval.json", rd / "eval_v1.json")
        if (rd / "ledger_test.csv").exists():
            shutil.copy(rd / "ledger_test.csv", rd / "ledger_test_v1.csv")
    new = {**ev, **val_best, **test_best, **test_last, "eval_env_version": ENV_VERSION,
           "env_version": ev.get("env_version", 1), "reevaluated_with_code": CODE_FINGERPRINT}
    rows = []
    for led in leds:
        sym = led.attrs["symbols"]
        df = led[["equity", "ret", "rf", "turnover", "cost", "gross", "net"]].copy()
        for c in led.columns:
            if c.startswith("w_"):
                df[c] = led[c]
        df["asset"] = sym[0] if len(sym) == 1 else "portfolio"
        rows.append(df.reset_index())
    pd.concat(rows).to_csv(rd / "ledger_test.csv", index=False, float_format="%.7g")
    save_json(rd / "eval.json", new)
    return f"{rd.name}: test {ev.get('test_sharpe', float('nan')):.3f} -> {new.get('test_sharpe', float('nan')):.3f}"


if __name__ == "__main__":
    exps = sys.argv[1:]
    dirs = [str(p.parent) for e in exps for p in sorted((RUNS_DIR / e).glob("*/model.pt"))
            if (p.parent / "eval.json").exists()]
    print(f"{len(dirs)} runs to re-evaluate", flush=True)
    changed = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 12))) as ex:
        futs = {ex.submit(reevaluate, d): d for d in dirs}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                msg = f.result()
            except Exception as e:  # noqa: BLE001
                msg = f"{Path(futs[f]).name}: ERROR {e!r}"
            if k % 25 == 0 or "ERROR" in msg:
                print(f"[{k}/{len(dirs)}] {msg}", flush=True)
            changed.append(msg)
    big = [m for m in changed if "->" in m and abs(float(m.split("-> ")[1]) - float(m.split("test ")[1].split(" ->")[0])) > 0.05]
    print(f"done; {len(big)} runs moved by more than 0.05 Sharpe:")
    for m in big[:40]:
        print("  ", m)
