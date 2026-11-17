"""Matplotlib figures for the research report (light theme, reference palette, thin marks, no dual axes)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIV_POS, DIV_NEG, DIV_MID = "#2a78d6", "#e34948", "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "text.parse_math": False, "figure.dpi": 110,
})


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def hbar(ax, labels, values, lo=None, hi=None, refs=(), color=SERIES[0], fmt="{:.2f}", highlight=None):
    """Horizontal bars (one series -> one colour), optional whiskers and labelled reference lines."""
    y = np.arange(len(labels))[::-1]
    cols = [color if (highlight is None or lab in highlight) else MUTED for lab in labels]
    ax.barh(y, values, height=0.5, color=cols, zorder=3)
    if lo is not None and hi is not None:
        for yi, a, b in zip(y, lo, hi):
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([a, b], [yi, yi], color=INK2, lw=0.9, zorder=4)
                ax.plot([a, a], [yi - 0.12, yi + 0.12], color=INK2, lw=0.9, zorder=4)
                ax.plot([b, b], [yi - 0.12, yi + 0.12], color=INK2, lw=0.9, zorder=4)
    ax.set_yticks(y, labels)
    ax.axvline(0, color=AXIS, lw=0.9, zorder=2)
    ax.grid(axis="y", visible=False)
    allv = np.concatenate([np.asarray(values, float), np.asarray(hi if hi is not None else values, float),
                           np.asarray(lo if lo is not None else values, float), [0.0]])
    span = np.nanmax(allv) - np.nanmin(allv) or 1.0
    for yi, v, h in zip(y, values, hi if hi is not None else values):
        if np.isfinite(v):
            ax.text(max(v, h if np.isfinite(h) else v, 0) + 0.025 * span, yi, fmt.format(v),
                    va="center", ha="left", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xlim(np.nanmin(allv) - 0.04 * span, np.nanmax(allv) + 0.16 * span)
    for name, val in refs:
        ax.axvline(val, color=INK2, lw=0.9, zorder=2)
        ax.text(val, len(labels) - 0.35, f" {name} {val:.2f}", color=INK2, fontsize=8, va="bottom", ha="left")
    ax.set_ylim(-0.7, len(labels) - 0.1)


def diverging_heatmap(ax, mat: pd.DataFrame, title="", vmax=None, fmt="{:.2f}"):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    v = np.nanmax(np.abs(mat.values)) if vmax is None else vmax
    im = ax.imshow(mat.values, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=0)
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=7.5,
                        color=INK if abs(val) < 0.6 * v else "#ffffff")
    ax.set_xticks(np.arange(-0.5, mat.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title)
    return im


def equity_lines(ax, books: dict, ref_names=(), log=True):
    k = 0
    for name, ser in books.items():
        is_ref = name in ref_names
        color = MUTED if is_ref else SERIES[k % len(SERIES)]
        if not is_ref:
            k += 1
        eq = (1 + ser).cumprod()
        ax.plot(eq.index, eq.values, color=color, lw=1.2 if is_ref else 1.6, label=name, zorder=2 if is_ref else 3)
        ax.text(eq.index[-1], eq.values[-1], f" {name}", color=INK2, fontsize=7.5, va="center")
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (log scale)")
