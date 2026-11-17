"""Side-by-side baseline Sharpe under the default fill/costs and the alternative assumptions.

    python scripts/compare_execution_baselines.py   -> reports/tables/execution_baselines.md (+ csv)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from rlt.utils.io import REPORTS_DIR, RUNS_DIR  # noqa: E402

VARIANTS = {"spy": "next open (default)", "spy_next_close": "next close", "spy_same_close": "same close (unrealistic)",
            "spy_nocost": "zero costs", "spy_cost4x": "4x costs"}

if __name__ == "__main__":
    cols = {}
    turn = None
    for d, label in VARIANTS.items():
        p = RUNS_DIR / "baselines" / d / "oos.csv"
        if p.exists():
            df = pd.read_csv(p, index_col=0)
            df = df[~df.index.str.startswith("random_")]
            cols[label] = df["sharpe"].astype(float)
            if d == "spy":
                turn = df["turnover"].astype(float)
    t = pd.DataFrame(cols)
    if turn is not None:
        t.insert(0, "turnover/yr", turn)
    base = "next open (default)"
    if "same close (unrealistic)" in t:
        t["same-close inflation"] = t["same close (unrealistic)"] - t[base]
    if "zero costs" in t:
        t["cost drag (Sharpe)"] = t["zero costs"] - t[base]
    t = t.sort_values("turnover/yr", ascending=False)
    pd.set_option("display.width", 200)
    print(t.round(2).to_string())
    out = REPORTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    t.round(3).to_csv(out / "execution_baselines.csv")
    lines = ["| strategy | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
    for k, r in t.iterrows():
        lines.append(f"| {k} | " + " | ".join(f"{v:.1f}" if c == "turnover/yr" else f"{v:+.2f}" if "inflation" in c or "drag" in c
                                             else f"{v:.2f}" for c, v in r.items()) + " |")
    (out / "execution_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
