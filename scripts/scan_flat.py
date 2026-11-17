"""Find evaluations of exactly-flat policies, whose Sharpe ratio was float noise before the FLAT_SD fix.

Reports, per experiment: test evaluations with zero gross exposure, and validation evaluations with zero gross
that were (or tied with) the selected checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt.utils.io import RUNS_DIR, load_json, read_jsonl  # noqa: E402

if __name__ == "__main__":
    tot_test, tot_sel = 0, 0
    for exp in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and not p.name.startswith(("_", "baselines", "robust"))):
        flat_test, flat_sel, flat_any = [], [], 0
        for rd in exp.iterdir():
            ev_p = rd / "eval.json"
            if not ev_p.exists():
                continue
            ev = load_json(ev_p)
            if (ev.get("test_avg_gross") or 0) == 0:
                flat_test.append(rd.name)
            prog = read_jsonl(rd / "progress.jsonl")
            vals = [r for r in prog if r.get("val_sharpe") is not None]
            flat_vals = [r for r in vals if (r.get("val_avg_gross") or 0) == 0]
            flat_any += bool(flat_vals)
            best = ev.get("best_step")
            if any(r["step"] == best for r in flat_vals):
                flat_sel.append(rd.name)
        if flat_test or flat_sel or flat_any:
            print(f"{exp.name:18s} flat test evals: {len(flat_test):3d}  runs with any flat val eval: {flat_any:3d}  "
                  f"flat checkpoint SELECTED: {len(flat_sel):3d}  e.g. {(flat_sel or flat_test)[:3]}")
        tot_test += len(flat_test)
        tot_sel += len(flat_sel)
    print("total flat test evaluations:", tot_test, " total runs whose selected checkpoint was flat:", tot_sel)
