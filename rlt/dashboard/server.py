"""Training dashboard: a small local web server over runs/.

    python -m rlt.dashboard.server --port 8795      ->  http://127.0.0.1:8795

Read-only JSON API (the page polls it every few seconds, so runs appear and update while they train):
    /api/experiments                       experiments with run counts by state
    /api/runs?exp=E                        one row per run: status + config + validation/test metrics
    /api/progress?exp=E&run=R              training diagnostics + periodic validation evaluations
    /api/ledger?exp=E&run=R                test-window equity curve of the selected checkpoint
    /api/summary?exp=E                     fold x algorithm test-Sharpe matrix, stitched OOS table
    /api/baseline?universe=U&strategy=S    walk-forward test equity of a baseline (for overlays)
    /api/baselines?universe=U              baseline OOS table
Binds to 127.0.0.1 only. Experiment/run names are matched against the directory listing (no path input).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rlt.utils.io import RUNS_DIR, dumps, read_jsonl, safe_name  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
_cache: dict = {}


def _experiments() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir() and p.name not in ("baselines",)
                  and any(p.glob("*/status.json")))


def _safe_exp(name: str | None) -> Path | None:
    if name and name in _experiments():
        return RUNS_DIR / name
    return None


def _safe_run(exp_dir: Path, run: str | None) -> Path | None:
    if not run:
        return None
    for p in exp_dir.iterdir():
        if p.is_dir() and p.name == run:
            return p
    return None


def _read(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def api_experiments():
    out = []
    for e in _experiments():
        states: dict = {}
        latest = 0.0
        for s in (RUNS_DIR / e).glob("*/status.json"):
            st = _read(s) or {}
            k = st.get("state", "unknown")
            states[k] = states.get(k, 0) + 1
            latest = max(latest, s.stat().st_mtime)
        out.append({"name": e, "states": states, "runs": sum(states.values()), "updated": latest})
    out.sort(key=lambda r: -r["updated"])
    return out


def api_runs(exp: str):
    d = _safe_exp(exp)
    if d is None:
        return []
    rows = []
    for rd in sorted(p for p in d.iterdir() if p.is_dir()):
        st = _read(rd / "status.json")
        if st is None:
            continue
        cfg = _read(rd / "config.json") or {}
        spec = cfg.get("spec", {})
        ev = _read(rd / "eval.json") or {}
        rows.append({
            "run": rd.name, "state": st.get("state"), "step": st.get("step"), "total_steps": st.get("total_steps"),
            "elapsed": st.get("elapsed"), "best_val_sharpe": st.get("best_val_sharpe"), "error": st.get("error"),
            "algo": spec.get("algo"), "fold": spec.get("fold"), "seed": spec.get("seed"), "tag": spec.get("tag"),
            "universe": spec.get("universe"),
            **{k: ev.get(k) for k in ("val_sharpe", "test_sharpe", "test_cagr", "test_max_drawdown", "test_turnover",
                                      "test_avg_gross", "test_last_sharpe", "best_step", "test_ew_sharpe")},
        })
    return rows


def api_progress(exp: str, run: str):
    d = _safe_exp(exp)
    rd = _safe_run(d, run) if d else None
    if rd is None:
        return []
    return read_jsonl(rd / "progress.jsonl")


def _equity_json(df: pd.DataFrame, max_points: int = 1500):
    """Growth of 1 per date (pooled instruments averaged), thinned for the browser."""
    book = df.groupby("date")["ret"].mean().sort_index()
    eq = (1 + book).cumprod()
    if len(eq) > max_points:
        eq = eq.iloc[np.linspace(0, len(eq) - 1, max_points).astype(int)]
    return {"dates": [str(d)[:10] for d in eq.index], "equity": [round(float(v), 5) for v in eq.values]}


def api_ledger(exp: str, run: str):
    d = _safe_exp(exp)
    rd = _safe_run(d, run) if d else None
    if rd is None or not (rd / "ledger_test.csv").exists():
        return {}
    df = pd.read_csv(rd / "ledger_test.csv", parse_dates=["date"])
    return _equity_json(df)


def api_baseline(universe: str, strategy: str):
    universe = safe_name(universe)
    base = RUNS_DIR / "baselines"
    if not base.exists() or universe not in [p.name for p in base.iterdir()]:
        return {}
    f = base / universe / "ledgers" / f"{strategy}.csv"
    if strategy not in [p.stem for p in (base / universe / "ledgers").glob("*.csv")]:
        return {}
    df = pd.read_csv(f, parse_dates=["date"])
    out = {}
    for fold, g in df.groupby("fold"):
        out[fold] = _equity_json(g)
    out["_all"] = _equity_json(df)
    return out


def api_baselines(universe: str):
    universe = safe_name(universe)
    base = RUNS_DIR / "baselines"
    if not base.exists() or universe not in [p.name for p in base.iterdir()]:
        return []
    f = base / universe / "oos.csv"
    if not f.exists():
        return []
    df = pd.read_csv(f, index_col=0)
    keep = ["family", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "cagr", "vol", "max_drawdown", "turnover", "cost_drag"]
    df = df[[c for c in keep if c in df.columns]]
    rnd = df[df.index.str.startswith("random_")]
    rows = [{"strategy": k, **{c: (None if pd.isna(v) else v) for c, v in r.items()}}
            for k, r in df[~df.index.str.startswith("random_")].iterrows()]
    if len(rnd):
        s = rnd["sharpe"].astype(float)
        rows.append({"strategy": f"random (n={len(s)}, mean)", "family": "random", "sharpe": float(s.mean()),
                     "sharpe_ci_lo": float(s.quantile(0.05)), "sharpe_ci_hi": float(s.quantile(0.95))})
    return rows


def api_summary(exp: str):
    d = _safe_exp(exp)
    if d is None:
        return {}
    mt = max((p.stat().st_mtime for p in d.glob("*/eval.json")), default=0)
    key = ("summary", exp)
    if key in _cache and _cache[key][0] >= mt:
        return _cache[key][1]
    from rlt.experiments.analysis import config_table, load_runs

    runs = load_runs(exp)
    if runs.empty or "test_sharpe" not in runs:
        return {}
    runs["label"] = runs.apply(lambda r: r["algo"] if not r.get("tag") else f"{r['algo']}:{r['tag']}", axis=1)
    mat = runs.groupby(["label", "fold"])["test_sharpe"].mean().unstack("fold")
    # same statistic as the report: Sharpe of the seed ensemble, 95% stationary-bootstrap CI (complete folds only)
    tab = config_table(exp, group=("label",), runs=runs)
    fold_dates = {}
    for fold, g in runs.groupby("fold"):
        cfg = _read(Path(g["dir"].iloc[0]) / "config.json") or {}
        te = (cfg.get("fold") or {}).get("dates", {}).get("test")
        if te:
            fold_dates[fold] = te
    res = {
        "fold_dates": fold_dates,
        "folds": list(mat.columns), "labels": list(mat.index),
        "matrix": [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in mat.values],
        "table": json.loads(tab.to_json(orient="records")) if len(tab) else [],
        "universe": str(runs["universe"].dropna().iloc[0]) if "universe" in runs else None,
    }
    _cache[key] = (mt, res)
    return res


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path.startswith("/api/"):
                ep = u.path[5:]
                if ep == "experiments":
                    data = api_experiments()
                elif ep == "runs":
                    data = api_runs(q.get("exp"))
                elif ep == "progress":
                    data = api_progress(q.get("exp"), q.get("run"))
                elif ep == "ledger":
                    data = api_ledger(q.get("exp"), q.get("run"))
                elif ep == "summary":
                    data = api_summary(q.get("exp"))
                elif ep == "baseline":
                    data = api_baseline(q.get("universe", ""), q.get("strategy", ""))
                elif ep == "baselines":
                    data = api_baselines(q.get("universe", ""))
                else:
                    return self._send(404, b"{}", "application/json")
                return self._send(200, dumps(data).encode(), "application/json")
            name = "index.html" if u.path in ("/", "/index.html") else u.path.lstrip("/")
            f = (STATIC / name).resolve()
            if STATIC.resolve() not in f.parents or not f.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            return self._send(200, f.read_bytes(), ctype)
        except Exception as e:  # noqa: BLE001
            return self._send(500, dumps({"error": repr(e)}).encode(), "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8795)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard on http://127.0.0.1:{args.port}  (runs dir: {RUNS_DIR})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        _ = time


if __name__ == "__main__":
    main()
