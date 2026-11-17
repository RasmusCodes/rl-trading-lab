"""Training-side diagnostics for the reward-hacking runs (what the agent was optimising vs what it earned).

    python scripts/hacking_diagnostics.py   -> reports/tables/hacking_training.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlt.env.rewards import RewardFn, StepInfo  # noqa: E402
from rlt.utils.io import REPORTS_DIR, RUNS_DIR, read_jsonl  # noqa: E402


def training_stats(exp="hacking_spy"):
    rows = []
    for d in sorted((RUNS_DIR / exp).iterdir()):
        if not d.is_dir():
            continue
        tag = d.name.split("__", 3)[-1]
        prog = [r for r in read_jsonl(d / "progress.jsonl") if r.get("ep_length") is not None]
        if not prog:
            continue
        early, late = prog[: max(1, len(prog) // 5)], prog[-max(1, len(prog) // 5):]
        f = lambda rs, k: float(np.nanmean([r.get(k, np.nan) for r in rs]))  # noqa: E731
        rows.append({"tag": tag, "len_early": f(early, "ep_length"), "len_late": f(late, "ep_length"),
                     "term_early": f(early, "ep_terminated"), "term_late": f(late, "ep_terminated"),
                     "reward_late": f(late, "ep_reward"), "reward_max": float(np.nanmax([r.get("ep_reward", np.nan) for r in prog])),
                     "logret_late": f(late, "ep_log_return"), "gross_late": f(late, "ep_gross")})
    return pd.DataFrame(rows).groupby("tag").mean()


def dsr_constant_cash_demo(days=252, rf_daily=0.02 / 252):
    """Differential Sharpe reward collected by a policy that only holds cash earning a constant T-bill rate."""
    out = {}
    for use_excess, floor in ((False, 0.0), (True, (0.002) ** 2)):
        fn = RewardFn({"type": "diff_sharpe", "use_excess": use_excess, "var_floor": floor, "scale": 1.0}, 1)
        rs = []
        for _ in range(days):
            a = lambda v: np.array([v], float)  # noqa: E731
            info = StepInfo(ret=a(rf_daily), log_ret=a(np.log1p(rf_daily)), rf=a(rf_daily), pnl=a(0.0),
                            initial=a(1.0), cost_frac=a(0.0), turnover=a(0.0), gross=a(0.0),
                            weights=np.zeros((1, 1)), dd=a(0.0), dd_prev=a(0.0), mkt_vol=a(0.01))
            rs.append(float(fn(info)[0]))
        out[f"excess={use_excess}, floor={'yes' if floor else 'no'}"] = {
            "reward_day_21": rs[20], "reward_day_126": rs[125], "reward_day_252": rs[-1], "sum": float(np.sum(rs))}
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t = training_stats()
    print(t.round(3).to_string())
    demo = dsr_constant_cash_demo()
    print("\nDSR reward of an all-cash policy (rf 2%/yr):")
    for k, v in demo.items():
        print(" ", k, {kk: f"{vv:.3g}" for kk, vv in v.items()})
    out = REPORTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    t.round(3).reset_index().to_csv(out / "hacking_training.csv", index=False)
    pd.DataFrame(demo).T.to_csv(out / "dsr_cash_demo.csv")
