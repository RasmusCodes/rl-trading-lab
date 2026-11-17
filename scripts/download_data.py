"""Download and cache every instrument the lab uses (data/raw/*.csv), then print a data-quality summary."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from rlt.data.market import load_market  # noqa: E402
from rlt.data.sources import fetch_yahoo  # noqa: E402
from rlt.data.universe import SPECS, UNIVERSES  # noqa: E402
from rlt.utils.io import DATA_DIR, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    extra = ["^IRX", "^GSPC", "^NDX", "^DJI"]
    syms = sorted({SPECS[s].source for u in UNIVERSES.values() for s in u} | set(extra) | {"SPY"})
    for s in syms:
        df = fetch_yahoo(s, refresh=args.refresh)
        print(f"{s:7s} {len(df):6d} bars  {df.index[0].date()} .. {df.index[-1].date()}")
    summary = {}
    for u in UNIVERSES:
        md = load_market(u)
        i0 = md.first_valid_index()
        lr = np.diff(np.log(md.close[i0:]), axis=0)
        summary[u] = {
            "rows": md.T, "first_common_row": md.dates[i0].date().isoformat(), "end": md.dates[-1].date().isoformat(),
            "symbols": md.meta["symbols"],
            "ann_vol": dict(zip(md.symbols, np.round(np.nanstd(lr, axis=0) * np.sqrt(252), 3).tolist())),
            "max_abs_daily_logret": dict(zip(md.symbols, np.round(np.nanmax(np.abs(lr), axis=0), 3).tolist())),
        }
        print(u, summary[u]["first_common_row"], "->", summary[u]["end"], summary[u]["max_abs_daily_logret"])
    save_json(DATA_DIR / "data_summary.json", summary)


if __name__ == "__main__":
    main()
