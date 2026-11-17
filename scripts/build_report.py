"""Build report figures and tables from runs/ (sections are skipped if their experiment has not run).

    python scripts/build_report.py   -> reports/figures/*.png, reports/tables/*.md, reports/results.json

Statistics (post-audit): every strategy, RL or baseline, is summarised by the Sharpe of one daily return series
(RL: the seed ensemble) with a stationary-bootstrap 95% CI; development records stop before 2019-01-01;
incomplete fold sets are skipped; the deflated Sharpe ratio is computed on the same statistic that selected
the best strategy, with every feasible strategy evaluated on the SPY development test windows counted as a trial.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlt.evaluation.metrics import bootstrap_sharpe_ci, deflated_sharpe, perf_stats  # noqa: E402
from rlt.experiments.analysis import (HOLDOUT_START, baseline_book, config_table, daily_book,  # noqa: E402
                                      ensemble_book, load_runs)
from rlt.report.figures import INK2, MUTED, SERIES, _finish, diverging_heatmap, hbar  # noqa: E402
from rlt.utils.io import REPORTS_DIR, RUNS_DIR, read_jsonl, safe_name  # noqa: E402

FIG = REPORTS_DIR / "figures"
TAB = REPORTS_DIR / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
RESULTS: dict = {}
ALGO_ORDER = ["qlearning", "dqn", "ddqn", "a2c", "ppo", "sac", "td3", "rrl", "ppo_lstm"]
ALGO_NAME = {"qlearning": "Q-learning", "dqn": "DQN", "ddqn": "Double DQN", "a2c": "A2C", "ppo": "PPO", "sac": "SAC",
             "td3": "TD3", "rrl": "RRL (Moody-Saffell)", "ppo_lstm": "PPO-LSTM"}
PCT = {"cagr": "{:.1%}", "vol": "{:.1%}", "max_drawdown": "{:.1%}", "turnover": "{:.1f}", "cost_drag": "{:.2%}",
       "p_vs_bench": "{:.3f}", "avg_gross": "{:.2f}"}


def md_table(df: pd.DataFrame, path: Path, floatfmt: dict | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("–" if not np.isfinite(v) else floatfmt.get(c, "{:.2f}").format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def exists(exp: str) -> bool:
    return (RUNS_DIR / exp).exists() and any((RUNS_DIR / exp).glob("*/eval.json"))


# ------------------------------------------------------------------------------------------ baselines
def baseline_table(universe: str, holdout: bool = False) -> pd.DataFrame | None:
    """Stats for every baseline of a universe, recomputed from its (trimmed) ledgers with bootstrap CIs."""
    d = RUNS_DIR / "baselines" / safe_name(universe) / "ledgers"
    if not d.exists():
        return None
    oos = pd.read_csv(d.parent / "oos.csv", index_col=0) if (d.parent / "oos.csv").exists() else None
    rows = []
    for f in sorted(d.glob("*.csv")):
        name = f.stem
        book = baseline_book(universe, name, holdout=holdout)
        st = perf_stats(book["ret"].values, book["rf"].values, book["turnover"].values, book["cost"].values,
                        book["gross"].values)
        lo, hi = bootstrap_sharpe_ci((book["ret"] - book["rf"]).values, n_boot=1000)
        fam = oos.loc[name, "family"] if oos is not None and name in oos.index and "family" in oos else ""
        rows.append({"strategy": name, "family": fam, "sharpe": st["sharpe"], "ci_lo": lo, "ci_hi": hi,
                     "cagr": st["cagr"], "vol": st["vol"], "max_drawdown": st["max_drawdown"],
                     "turnover": st.get("turnover"), "cost_drag": st.get("cost_drag"), "avg_gross": st.get("avg_gross")})
    return pd.DataFrame(rows).set_index("strategy")


def bench_name(universe: str) -> str:
    return "equal_weight" if universe in ("etf9", "fut4") else "buy_hold"


def section_baselines(universe="spy"):
    bt = baseline_table(universe)
    if bt is None:
        return None
    rnd = bt[bt.index.str.startswith("random_")]["sharpe"]
    b = bt[~bt.index.str.startswith("random_")].sort_values("sharpe")
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(b) + 1.3))
    hbar(ax, list(b.index[::-1]), b["sharpe"].values[::-1], b["ci_lo"].values[::-1], b["ci_hi"].values[::-1])
    if len(rnd):
        ax.axvspan(rnd.quantile(0.05), rnd.quantile(0.95), color=MUTED, alpha=0.12, zorder=1)
        ax.text(rnd.quantile(0.95), -0.6, " random strategies 5-95%", color=INK2, fontsize=8, va="bottom")
    ax.set_xlabel("Sharpe (excess of T-bills), stitched walk-forward test blocks; whiskers: 95% bootstrap CI")
    ax.set_title(f"Baselines through the same environment ({universe.upper()}, all costs)")
    _finish(fig, FIG / f"baselines_{safe_name(universe)}.png")
    t = b.sort_values("sharpe", ascending=False).reset_index()
    md_table(t, TAB / f"baselines_{safe_name(universe)}.md", PCT)
    RESULTS[f"baselines_{universe}"] = {"table": t.to_dict("records"),
                                        "random": {"mean": float(rnd.mean()), "p05": float(rnd.quantile(0.05)),
                                                   "p95": float(rnd.quantile(0.95)), "n": int(len(rnd))}}
    return bt


# ------------------------------------------------------------------------------------------ controls
def section_controls():
    if not exists("controls"):
        return
    runs = load_runs("controls")
    markets = ["gbm", "ar1", "regime"]
    titles = {"gbm": "Random walk (no edge)", "ar1": "AR(1) returns (planted edge)",
              "regime": "Regime switching (planted edge)"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharex=True)
    out, tabs = {}, []
    for ax, mkt in zip(axes, markets):
        sub_runs = runs[runs["tag"] == mkt]
        tab = config_table("controls", group=("algo",), runs=sub_runs)
        tab["order"] = tab["algo"].map({a: i for i, a in enumerate(ALGO_ORDER)})
        tab = tab.sort_values("order")
        bt = baseline_table(f"syn:{mkt}")
        refs = []
        if bt is not None:
            refs.append(("B&H", float(bt.loc["buy_hold", "sharpe"])))
        hbar(ax, [ALGO_NAME[a] for a in tab["algo"]], tab["sharpe"].values, tab["ci_lo"].values, tab["ci_hi"].values,
             refs=refs)
        ax.set_title(titles[mkt])
        ax.set_xlabel("test Sharpe, 3-seed ensemble (95% CI)")
        tab["market"] = mkt
        tabs.append(tab)
        out[mkt] = {"agents": tab[["algo", "sharpe", "ci_lo", "ci_hi", "seed_min", "seed_max", "turnover"]].to_dict("records"),
                    "baselines": None if bt is None else bt[~bt.index.str.startswith("random_")]["sharpe"].to_dict()}
    _finish(fig, FIG / "controls.png")
    t = pd.concat(tabs)[["market", "algo", "seeds", "sharpe", "ci_lo", "ci_hi", "seed_min", "seed_max", "turnover",
                         "avg_gross", "cost_drag"]]
    md_table(t, TAB / "controls.md", PCT)
    RESULTS["controls"] = out


def section_gamma():
    if not exists("gamma_controls"):
        return
    runs = load_runs("gamma_controls")
    runs["market"] = runs["tag"].str.split("_g").str[0]
    runs["gamma"] = runs["tag"].str.split("_g").str[1].astype(float)
    g = runs.groupby(["market", "algo", "gamma"]).agg(test=("test_sharpe", "mean"), turnover=("test_turnover", "mean"),
                                                       n=("test_sharpe", "size")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, mkt in zip(axes, ["ar1", "regime"]):
        for k, a in enumerate(["ppo", "sac", "dqn"]):
            s = g[(g["market"] == mkt) & (g["algo"] == a)].sort_values("gamma")
            ax.plot(s["gamma"], s["test"], marker="o", ms=5, color=SERIES[k], label=ALGO_NAME[a])
            if len(s):
                ax.text(s["gamma"].iloc[-1] + 0.02, s["test"].iloc[-1], ALGO_NAME[a], color=INK2, fontsize=8, va="center")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title({"ar1": "AR(1) control", "regime": "Regime control"}[mkt])
        ax.set_xlabel("discount factor gamma")
        ax.set_xticks([0, 0.5, 0.9])
    axes[0].set_ylabel("test Sharpe (2 seeds, mean per run)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "gamma.png", dpi=160)
    plt.close(fig)
    md_table(g, TAB / "gamma.md", {"turnover": "{:.0f}", "gamma": "{:g}"})
    RESULTS["gamma"] = g.to_dict("records")


# ------------------------------------------------------------------------------------- main comparison
def main_section(exp="main_spy", universe="spy", label="main"):
    if not exists(exp):
        return
    runs = load_runs(exp)
    bh = baseline_book(universe, "buy_hold")
    tab = config_table(exp, group=("algo",), runs=runs, bench=bh)
    tab["name"] = tab["algo"].map(ALGO_NAME)
    bt = baseline_table(universe)
    refs = [("buy & hold", float(bt.loc["buy_hold", "sharpe"]))] if bt is not None else []
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(tab) + 1.4))
    t2 = tab.sort_values("sharpe")
    hbar(ax, list(t2["name"])[::-1], t2["sharpe"].values[::-1], t2["ci_lo"].values[::-1], t2["ci_hi"].values[::-1],
         refs=refs)
    ax.set_xlabel("Sharpe of the 3-seed ensemble, test blocks 2007-2018 chained; whiskers: 95% bootstrap CI")
    ax.set_title("RL agents on SPY, walk-forward out-of-sample, after all costs")
    _finish(fig, FIG / f"{label}_bars.png")
    mat = runs.groupby(["algo", "fold"])["test_sharpe"].mean().unstack("fold")
    mat = mat.reindex([a for a in ALGO_ORDER if a in mat.index])
    mat.index = [ALGO_NAME.get(a, a) for a in mat.index]
    lab = {c: f"{c.split('_')[1]}-{str(int(c.split('_')[1]) + 1)[2:]}" for c in mat.columns}
    mat.columns = [lab[c] for c in mat.columns]
    s = pd.read_csv(RUNS_DIR / "baselines" / universe / "summary.csv")
    for strat, name in (("buy_hold", "(buy & hold)"), ("ma_cross", "(MA 50/200)")):
        x = s[(s["strategy"] == strat) & (s["split"] == "test")].groupby("fold")["sharpe"].mean()
        mat.loc[name] = x.rename(index=lab)
    fig, ax = plt.subplots(figsize=(7.4, 0.33 * len(mat) + 1.2))
    diverging_heatmap(ax, mat, title="Test Sharpe by 2-year block (mean of 3 seeds per algorithm)", vmax=2.0)
    _finish(fig, FIG / f"{label}_heatmap.png")
    cols = ["name", "seeds", "sharpe", "ci_lo", "ci_hi", "seed_mean", "seed_min", "seed_max", "diff_vs_bench",
            "p_vs_bench", "cagr", "vol", "max_drawdown", "turnover", "cost_drag", "avg_gross"]
    md_table(tab[[c for c in cols if c in tab]], TAB / f"{label}.md", PCT)
    RESULTS[label] = tab.to_dict("records")
    # equity of the top-3 ensembles vs buy & hold and the best baseline
    books = {ALGO_NAME[a]: ensemble_book(runs[runs["algo"] == a])["ret"] for a in list(tab["algo"])[:3]}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, (name, ser) in enumerate(books.items()):
        eq = (1 + ser).cumprod()
        ax.plot(eq.index, eq.values, color=SERIES[k], lw=1.7, label=name)
    if bt is not None:
        nonr = bt[~bt.index.str.startswith("random_") & (bt.index != "buy_hold")]
        best = nonr["sharpe"].idxmax()
        for name, col in (("buy_hold", MUTED), (best, INK2)):
            ser = baseline_book(universe, name)["ret"]
            eq = (1 + ser).cumprod()
            ax.plot(eq.index, eq.values, color=col, lw=1.3, label=name + (" (best baseline)" if name == best else ""))
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (log); each 2-year block restarts flat")
    ax.legend(loc="upper left", ncols=2)
    ax.set_title("Chained test blocks 2007-2018: top-3 agent ensembles vs buy & hold and the best baseline")
    _finish(fig, FIG / f"{label}_equity.png")


def section_val_vs_test(exp="main_spy"):
    if not exists(exp):
        return
    runs = load_runs(exp)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    for k, a in enumerate(ALGO_ORDER):
        g = runs[runs["algo"] == a]
        ax.scatter(g["val_sharpe"], g["test_sharpe"], s=14, color=SERIES[k % 8] if k < 8 else INK2, alpha=0.8,
                   label=ALGO_NAME[a], edgecolors="none")
    rho = runs[["val_sharpe", "test_sharpe"]].corr(method="spearman").iloc[0, 1]
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("validation Sharpe of the selected checkpoint")
    ax.set_ylabel("test Sharpe (following 2 years)")
    ax.set_title(f"Validation barely predicts test (Spearman {rho:.2f}, n={len(runs)})")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=5, handletextpad=0.2,
              columnspacing=0.8)
    ax = axes[1]
    d = runs.groupby("algo")[["test_sharpe", "test_last_sharpe"]].mean().reindex(ALGO_ORDER).dropna()
    y = np.arange(len(d))[::-1]
    ax.barh(y + 0.18, d["test_sharpe"], height=0.32, color=SERIES[0], label="selected on validation")
    ax.barh(y - 0.18, d["test_last_sharpe"], height=0.32, color=SERIES[1], label="last checkpoint")
    ax.set_yticks(y, [ALGO_NAME[a] for a in d.index])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_xlabel("mean per-block test Sharpe")
    ax.set_title("Checkpoint selection vs training to the end")
    _finish(fig, FIG / "val_vs_test.png")
    RESULTS["val_vs_test"] = {"spearman": float(rho), "n": int(len(runs)),
                              "by_algo": d.reset_index().to_dict("records"),
                              "mean_selected": float(runs["test_sharpe"].mean()),
                              "mean_last": float(runs["test_last_sharpe"].mean())}


def training_curves(exp="main_spy", algos=("ppo", "sac", "dqn"), fold="wf3_2011"):
    if not exists(exp):
        return
    fig, axes = plt.subplots(1, len(algos), figsize=(11, 3.2), sharey=True)
    for ax, a in zip(np.atleast_1d(axes), algos):
        for s in range(3):
            rows = read_jsonl(RUNS_DIR / exp / f"{a}__{fold}__s{s}" / "progress.jsonl")
            pts = [(r["step"], r["val_sharpe"]) for r in rows if r.get("val_sharpe") is not None]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, color=SERIES[s], label=f"seed {s}")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{ALGO_NAME[a]}: validation Sharpe during training")
        ax.set_xlabel("environment steps")
    np.atleast_1d(axes)[0].set_ylabel("validation Sharpe (2009-10)")
    np.atleast_1d(axes)[0].legend(loc="lower left")
    _finish(fig, FIG / "training_curves.png")


# ----------------------------------------------------------------------------------- variant studies
def variant_section(exp, title, fname, universe="spy", order=None, label_map=None):
    if not exists(exp):
        return None
    runs = load_runs(exp)
    tab = config_table(exp, group=("tag",), runs=runs, bench=baseline_book(universe, "buy_hold"))
    if order:
        tab["o"] = tab["tag"].map({t: i for i, t in enumerate(order)}).fillna(99)
        tab = tab.sort_values("o")
    labels = [label_map.get(t, t) if label_map else t for t in tab["tag"]]
    bt = baseline_table(universe)
    refs = [("buy & hold", float(bt.loc["buy_hold", "sharpe"]))] if bt is not None else []
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.34 * len(tab) + 1.6), sharey=True,
                             gridspec_kw={"width_ratios": [2.2, 1, 1]})
    hbar(axes[0], labels, tab["sharpe"].values, tab["ci_lo"].values, tab["ci_hi"].values, refs=refs)
    axes[0].set_xlabel("OOS Sharpe, seed ensemble (95% CI)")
    axes[0].set_title(title)
    y = np.arange(len(tab))[::-1]
    axes[1].barh(y, tab["avg_gross"].values, height=0.5, color=SERIES[0])
    axes[1].set_title("avg gross exposure")
    axes[1].grid(axis="y", visible=False)
    axes[2].barh(y, tab["turnover"].values, height=0.5, color=SERIES[0])
    axes[2].set_title("turnover / year")
    axes[2].grid(axis="y", visible=False)
    _finish(fig, FIG / f"{fname}.png")
    cols = ["tag", "seeds", "sharpe", "ci_lo", "ci_hi", "seed_min", "seed_max", "diff_vs_bench", "p_vs_bench", "cagr",
            "max_drawdown", "turnover", "cost_drag", "avg_gross"]
    md_table(tab[[c for c in cols if c in tab]], TAB / f"{fname}.md", PCT)
    RESULTS[fname] = tab[[c for c in cols if c in tab]].to_dict("records")
    return tab


def section_hacking():
    exp = "hacking_spy"
    if not exists(exp):
        return
    runs = load_runs(exp)
    tab = config_table(exp, group=("tag",), runs=runs)
    diag = []
    for d, tag in zip(runs["dir"], runs["tag"]):
        rows = [r for r in read_jsonl(Path(d) / "progress.jsonl") if r.get("ep_length") is not None]
        if not rows:
            continue
        last = rows[-5:]
        diag.append({"tag": tag, "ep_length": np.mean([r["ep_length"] for r in last]),
                     "ep_reward": np.mean([r.get("ep_reward", np.nan) for r in last]),
                     "ep_terminated": np.mean([r.get("ep_terminated", np.nan) for r in last]),
                     "ep_log_return": np.mean([r["ep_log_return"] for r in last]),
                     "ep_gross": np.mean([r["ep_gross"] for r in last])})
    dg = pd.DataFrame(diag).groupby("tag").mean() if diag else pd.DataFrame()
    t = tab.set_index("tag").join(dg, how="left").reset_index()
    cols = ["tag", "sharpe", "ci_lo", "ci_hi", "avg_gross", "turnover", "max_drawdown", "cagr", "ep_length",
            "ep_terminated", "ep_reward", "ep_log_return"]
    md_table(t[[c for c in cols if c in t]], TAB / "hacking.md",
             {**PCT, "ep_length": "{:.0f}", "ep_terminated": "{:.0%}", "ep_reward": "{:.1f}", "ep_log_return": "{:.3f}"})
    RESULTS["hacking"] = t.to_dict("records")
    t = t.set_index("tag")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    ax = axes[0]
    for k, (pref, lab, lams) in enumerate([("dd_pen_", "drawdown penalty λ", ["0", "1", "5", "20"]),
                                           ("inv_pen_", "inventory penalty λ (bp/day)", ["0.0002", "0.0005", "0.002"])]):
        xs = [l_ for l_ in lams if f"{pref}{l_}" in t.index]
        if xs:
            ys = [t.loc[f"{pref}{l_}", "avg_gross"] for l_ in xs]
            ax.plot(range(len(xs)), ys, marker="o", color=SERIES[k], label=lab)
            for i, (xv, yv) in enumerate(zip(xs, ys)):
                ax.text(i, yv, f"  {float(xv) * (1e4 if pref == 'inv_pen_' else 1):g}", fontsize=7.5, color=INK2)
    ax.set_ylabel("test average gross exposure")
    ax.set_xlabel("penalty strength (increasing)")
    ax.set_title("Penalties push the policy toward flat")
    ax.legend(loc="upper right", fontsize=7.5)
    ax = axes[1]
    tags = [x for x in ("ddlevel_noterm", "ddlevel_term_pen", "ddlevel_term_nopen") if x in t.index]
    vals = [t.loc[x, "ep_length"] for x in tags]
    ax.bar(range(len(tags)), vals, color=SERIES[0], width=0.5)
    ax.set_xticks(range(len(tags)), ["no risk stop", "stop + penalty", "stop, no penalty"][: len(tags)])
    ax.set_ylabel("training episode length (sessions)")
    ax.set_title("Drawdown-level penalty + risk stop")
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    tags = [x for x in ("lev3_log", "lev3_pnl", "dsr_excess_floor", "dsr_raw_nofloor", "riskadj_floor",
                        "riskadj_nofloor") if x in t.index]
    yy = np.arange(len(tags))[::-1]
    ax.barh(yy, [t.loc[x, "sharpe"] for x in tags], height=0.5, color=SERIES[0])
    for yi, x in zip(yy, tags):
        lo, hi = t.loc[x, "ci_lo"], t.loc[x, "ci_hi"]
        ax.plot([lo, hi], [yi, yi], color=INK2, lw=0.9)
    ax.set_yticks(yy, tags)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_title("Objective variants: test Sharpe (95% CI)")
    _finish(fig, FIG / "hacking.png")


def section_leakage():
    exp = "leakage"
    if not exists(exp):
        return
    runs = load_runs(exp)
    agg = runs.groupby(["tag", "algo"]).agg(test_sharpe=("test_sharpe", "mean"), val_sharpe=("val_sharpe", "mean"),
                                            turnover=("test_turnover", "mean"), n=("test_sharpe", "size")).reset_index()
    md_table(agg, TAB / "leakage.md", {"turnover": "{:.0f}"})
    RESULTS["leakage"] = agg.to_dict("records")
    ppo = agg[agg["algo"] == "ppo"].copy()
    ppo["market"] = ppo["tag"].str.split("_").str[0]
    ppo["leak"] = ppo["tag"].str.split("_", n=1).str[1]
    order = ["none", "centered_ma", "smoothed_regime", "global_zscore", "future_return"]
    names = {"none": "no leak", "centered_ma": "centred moving average", "smoothed_regime": "smoothed HMM regime",
             "global_zscore": "full-sample z-score", "future_return": "next-session return"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharey=True)
    for ax, mkt in zip(axes, ["gbm", "spy"]):
        s = ppo[ppo["market"] == mkt].set_index("leak").reindex(order)
        hbar(ax, [names[o] for o in order], s["test_sharpe"].values)
        ax.set_title({"gbm": "PPO on a random walk (1 fold, 2 seeds)", "spy": "PPO on SPY (3 folds, 2 seeds)"}[mkt])
        ax.set_xlabel("mean per-run test Sharpe")
    _finish(fig, FIG / "leakage.png")


# ------------------------------------------------------------------------------------------- portfolio
def section_universe(exp, universe, title, fname, holdout=False):
    if not exists(exp):
        return
    runs = load_runs(exp)
    if "label" not in runs:
        runs["label"] = runs.apply(lambda r: r["algo"] if not r["tag"] else f"{r['algo']}:{r['tag']}", axis=1)
    bench = baseline_book(universe, bench_name(universe), holdout=holdout)
    tab = config_table(exp, group=("label",), runs=runs, bench=bench)
    rows = [{"name": ALGO_NAME.get(r["label"], r["label"]), "kind": "RL", "sharpe": r["sharpe"], "lo": r["ci_lo"],
             "hi": r["ci_hi"], "seed_min": r["seed_min"], "seed_max": r["seed_max"], "cagr": r["cagr"],
             "max_drawdown": r["max_drawdown"], "turnover": r["turnover"], "p_vs_bench": r.get("p_vs_bench")}
            for _, r in tab.iterrows()]
    bt = baseline_table(universe, holdout=holdout)
    if bt is not None:
        for name, r in bt[~bt.index.str.startswith("random_")].iterrows():
            rows.append({"name": name, "kind": "baseline", "sharpe": r["sharpe"], "lo": r["ci_lo"], "hi": r["ci_hi"],
                         "cagr": r["cagr"], "max_drawdown": r["max_drawdown"], "turnover": r["turnover"]})
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(df) + 1.4))
    hbar(ax, list(df["name"]), df["sharpe"].values, df["lo"].values, df["hi"].values,
         highlight=set(df[df["kind"] == "RL"]["name"]))
    ax.set_xlabel("Sharpe with 95% bootstrap CI. Blue: RL seed ensembles. Gray: baselines")
    ax.set_title(title)
    _finish(fig, FIG / f"{fname}.png")
    md_table(df, TAB / f"{fname}.md", PCT)
    RESULTS[fname] = df.to_dict("records")


def section_hpo():
    """v1 = original search (objective corrupted by float-noise Sharpe of flat policies); v2 = repeated after the
    fix. Each is judged on its own clean-fold evaluation (folds 4-6)."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    RESULTS["hpo"] = {}
    for k, (exp, ev, label) in enumerate([("hpo_ppo_spy", "hpo_eval_spy", "original search (flawed metric)"),
                                          ("hpo_ppo_spy_v2", "hpo_eval_spy_v2", "repeated after the fix")]):
        log = RUNS_DIR / exp / "hpo_log.json"
        if not log.exists():
            continue
        trials = json.loads(log.read_text())
        sc = [t["score"] for t in trials if np.isfinite(t["score"])]
        bp = RUNS_DIR / exp / "best.json"
        best = json.loads(bp.read_text()) if bp.exists() else {}
        out = {"n_trials": len(trials), "best": max(sc) if sc else None, "median": float(np.median(sc)) if sc else None,
               "best_params": best.get("params")}
        if exists(ev):
            runs = load_runs(ev)
            tab = config_table(ev, group=("tag",), runs=runs, bench=baseline_book("spy", "buy_hold"))
            md_table(tab, TAB / f"{ev}.md", PCT)
            out["eval"] = tab.to_dict("records")
        RESULTS["hpo"][exp] = out
        ax.plot(range(len(sc)), sc, marker="o", ms=4, color=SERIES[k], lw=1.0, alpha=0.8, label=f"{label}: trial")
        ax.plot(range(len(sc)), np.maximum.accumulate(sc), color=SERIES[k], lw=2.0, label=f"{label}: best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("mean validation Sharpe, folds 1-3")
    ax.set_title("PPO hyperparameter search (TPE, 24 trials each)")
    ax.legend(loc="lower right", fontsize=7.5)
    _finish(fig, FIG / "hpo.png")


def section_robustness():
    p = RUNS_DIR / "robustness" / "sweeps.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if "error" in df:
        df = df[df["error"].isna()]
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    # eight categorical slots + buy & hold in neutral gray (never cycle hues); full numbers are in the tables
    shown = ["ma_cross", "vol_target", "rrl", "ppo_lstm", "td3", "ppo", "dqn", "sac"]
    names = {"ma_cross": "MA 50/200", "vol_target": "vol target", "rrl": "RRL", "ppo_lstm": "PPO-LSTM", "td3": "TD3",
             "ppo": "PPO", "dqn": "DQN", "sac": "SAC", "buy_hold": "buy & hold"}
    for k, name in enumerate(["buy_hold"] + shown):
        s = df[(df["label"] == name) & (df["kind"] == "cost_mult")].groupby("x")["sharpe"].mean()
        a = df[(df["label"] == name) & (df["kind"] == "aum")].groupby("x")["sharpe"].mean()
        if s.empty:
            continue
        col = MUTED if name == "buy_hold" else SERIES[k - 1]
        axes[0].plot(s.index.astype(float), s.values, marker="o", ms=4, color=col, label=names[name])
        axes[1].plot(a.index.astype(float), a.values, marker="o", ms=4, color=col, label=names[name])
    axes[0].set_xlabel("cost multiplier (1 = base case)")
    axes[0].set_ylabel("mean per-block test Sharpe")
    axes[0].set_title("Cost sensitivity")
    axes[1].set_xscale("log")
    aums = [1e6, 1e7, 1e8, 1e9, 1e10]
    axes[1].xaxis.set_major_locator(FixedLocator(aums))
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: {1e6: "$1M", 1e7: "$10M", 1e8: "$100M",
                                                                   1e9: "$1B", 1e10: "$10B"}.get(v, "")))
    axes[1].set_xlabel("assets under management")
    axes[1].set_title("Capacity: Sharpe vs AUM (sqrt impact, 10% ADV cap)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=9, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "robustness.png", dpi=160)
    plt.close(fig)
    ex = df[df["kind"] == "execution"].groupby(["label", "x"])["sharpe"].mean().unstack("x")
    md_table(ex.reset_index(), TAB / "execution_reeval.md")
    RESULTS["robustness_execution"] = ex.reset_index().to_dict("records")
    cap = RUNS_DIR / "robustness" / "capacity.csv"
    if cap.exists():
        c = pd.read_csv(cap)
        md_table(c, TAB / "capacity.md", {"sharpe_1m": "{:.2f}", "aum_half_sharpe": "{:.2e}", "aum_below_bench": "{:.2e}"})
        RESULTS["capacity"] = c.to_dict("records")


def section_holdout():
    if not exists("holdout_spy"):
        return
    section_universe("holdout_spy", "spy_holdout", "Final holdout 2019-01 to 2026-11 (evaluated once)", "holdout",
                     holdout=True)


def section_dsr():
    """Deflated Sharpe of the best feasible SPY strategy, with every strategy tried on the development test
    windows (2007-2018, any subset of folds) counted as a trial. Selection and test use the same statistic: the
    Sharpe of one daily series (baseline book or RL seed ensemble)."""
    books, sr = {}, {}
    bt = baseline_table("spy")
    if bt is not None:
        for n in bt.index:
            if not n.startswith("random_"):
                books[f"baseline:{n}"] = baseline_book("spy", n)
    exps = {"main_spy": "algo", "rewards_spy": "tag", "features_spy": "tag", "randomized_spy": "tag",
            "hacking_spy": "tag", "execution_spy": "tag", "leakage": "tag", "hpo_eval_spy": "tag",
            "hpo_eval_spy_v2": "tag"}
    for exp, key in exps.items():
        if not exists(exp):
            continue
        runs = load_runs(exp)
        if exp == "execution_spy":
            runs = runs[runs["tag"] == "next_open"]  # other fills are infeasible / not deployable
        if exp == "leakage":
            runs = runs[runs["tag"] == "spy_none"]  # leaky variants are infeasible by construction
        for k, g in runs.groupby(key):
            b = ensemble_book(g)
            if len(b):
                books[f"{exp}:{k}"] = b
    full_days = max(len(b) for b in books.values())
    full = {}  # strategies evaluated on the whole 2007-2018 window: the only ones comparable and eligible as "best"
    for name, b in books.items():
        ex = (b["ret"] - b["rf"]).values
        sr[name] = float(ex.mean() / ex.std(ddof=1) * np.sqrt(252)) if ex.std() > 0 else 0.0
        if len(b) >= 0.98 * full_days:
            full[name] = sr[name]
    if len(full) < 3:
        return
    best = max(full, key=full.get)
    b = books[best]
    ex = (b["ret"] - b["rf"]).values
    out = {"n_trials": len(sr), "n_full_window": len(full), "best": best, "best_sharpe": full[best],
           "sharpe_sd_across_trials": float(np.std(list(full.values()), ddof=1)),
           # trials = every strategy tried on SPY dev test windows; SR dispersion from the comparable ones
           "dsr": deflated_sharpe(ex, len(sr), list(full.values())),
           "psr_vs_zero": perf_stats(b["ret"].values, b["rf"].values)["psr"]}
    bh = books.get("baseline:buy_hold")
    if bh is not None:
        exb = (bh["ret"] - bh["rf"]).values
        out["buy_hold_sharpe"] = sr["baseline:buy_hold"]
        out["buy_hold_dsr"] = deflated_sharpe(exb, len(sr), list(full.values()))
    best_rl = max((k for k in full if not k.startswith("baseline:")), key=full.get, default=None)
    if best_rl:
        brl = books[best_rl]
        out["best_rl"] = best_rl
        out["best_rl_sharpe"] = full[best_rl]
        out["best_rl_dsr"] = deflated_sharpe((brl["ret"] - brl["rf"]).values, len(sr), list(full.values()))
    top = sorted(full.items(), key=lambda kv: -kv[1])[:10]
    out["top10"] = top
    # the question that matters: does anything beat buy & hold once the search is accounted for?
    # Deflate the information ratio of active returns (strategy excess - buy & hold excess) across the same trials.
    if bh is not None:
        ir, act = {}, {}
        for name in full:
            if name == "baseline:buy_hold":
                continue
            j = books[name].join(bh, how="inner", rsuffix="_bh")
            a = ((j["ret"] - j["rf"]) - (j["ret_bh"] - j["rf_bh"])).values
            act[name] = a
            ir[name] = float(a.mean() / a.std(ddof=1) * np.sqrt(252)) if a.std() > 0 else 0.0
        best_ir = max(ir, key=ir.get)
        out["active"] = {"best": best_ir, "best_ir": ir[best_ir], "ir_sd": float(np.std(list(ir.values()), ddof=1)),
                         "dsr_active": deflated_sharpe(act[best_ir], len(sr), list(ir.values())),
                         "top5": sorted(ir.items(), key=lambda kv: -kv[1])[:5]}
    RESULTS["dsr"] = out
    pd.DataFrame(top, columns=["strategy", "sharpe"]).to_csv(TAB / "dsr_top10.csv", index=False)


if __name__ == "__main__":
    section_baselines("spy")
    section_controls()
    section_gamma()
    main_section()
    section_val_vs_test()
    training_curves()
    variant_section("rewards_spy", "Reward functions (PPO, SPY)", "rewards")
    variant_section("features_spy", "State representation (PPO, SPY)", "features")
    variant_section("randomized_spy", "Training-environment variants (PPO, evaluated on SPY)", "randomized")
    variant_section("execution_spy", "Fill assumption (PPO, SPY)", "execution",
                    order=["next_open", "next_close", "same_close"])
    section_hacking()
    section_leakage()
    section_universe("portfolio_etf9", "etf9", "Nine-ETF portfolio, test 2011-2018", "portfolio_etf9")
    section_universe("futures_fut4", "fut4", "Four futures, test 2011-2018", "futures_fut4")
    section_universe("equities_eq10", "eq10_2000", "Ten end-2000 large caps, pooled agents, test 2007-2018",
                     "equities_eq10")
    section_hpo()
    section_robustness()
    section_holdout()
    section_dsr()
    (REPORTS_DIR / "results.json").write_text(json.dumps(RESULTS, indent=1, default=float), encoding="utf-8")
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    if "dsr" in RESULTS:
        print("DSR:", {k: v for k, v in RESULTS["dsr"].items() if k != "top10"})
