"""Walk-forward splits, purging/embargo and the locked final holdout.

Timing convention (see rlt.env): a decision made at close t is filled at the open of t+1 and its reward is
realised by close t+1; the supervised baselines' label is the open(t+1)->open(t+2) return. So the last
training *decision* row must be at least 2 sessions before the first validation row. We purge the last
`embargo` rows of every training window and validation window (default 5 sessions) so no reward or label
straddles a boundary.

Folds are expanding-window:  [train ........][embargo][val][embargo][test]
The final holdout (default 2019-01-01 onwards) is excluded from every development fold. Code that wants
to evaluate on it must go through `HoldoutLedger.unlock`, which appends a line to runs/holdout_ledger.jsonl;
the report states how many times the holdout was touched.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import pandas as pd

from ..utils.io import RUNS_DIR


@dataclass(frozen=True)
class Fold:
    name: str
    train: tuple  # (i0, i1) inclusive row indices of *decision* rows
    val: tuple
    test: tuple
    dates: dict  # ISO dates of the same boundaries, for humans

    def to_dict(self):
        return asdict(self)


def _row(dates: pd.DatetimeIndex, d) -> int:
    return int(dates.searchsorted(pd.Timestamp(d)))


def make_fold(dates: pd.DatetimeIndex, name: str, train_start, val_start, test_start, test_end, embargo: int = 5,
              min_row: int = 0) -> Fold:
    i_tr0 = max(_row(dates, train_start), min_row)
    i_va0 = _row(dates, val_start)
    i_te0 = _row(dates, test_start)
    i_te1 = min(_row(dates, test_end) - 1, len(dates) - 2)  # last decision needs one more bar for its fill
    tr = (i_tr0, i_va0 - 1 - embargo)
    va = (i_va0, i_te0 - 1 - embargo)
    te = (i_te0, i_te1)
    for a, b in (tr, va, te):
        if b <= a:
            raise ValueError(f"empty segment in fold {name}: {tr} {va} {te}")
    iso = lambda i: dates[i].date().isoformat()  # noqa: E731
    return Fold(name, tr, va, te, {"train": [iso(tr[0]), iso(tr[1])], "val": [iso(va[0]), iso(va[1])],
                                   "test": [iso(te[0]), iso(te[1])]})


def walk_forward(dates: pd.DatetimeIndex, first_test_year: int, last_test_year: int, test_years: int = 2,
                 val_years: int = 2, train_start=None, embargo: int = 5, min_row: int = 0) -> list[Fold]:
    folds = []
    y = first_test_year
    k = 0
    while y + test_years - 1 <= last_test_year:
        k += 1
        ts = f"{y}-01-01"
        te = f"{y + test_years}-01-01"
        vs = f"{y - val_years}-01-01"
        folds.append(make_fold(dates, f"wf{k}_{y}", train_start or dates[min_row], vs, ts, te, embargo, min_row))
        y += test_years
    return folds


def holdout_fold(dates: pd.DatetimeIndex, holdout_start="2019-01-01", val_years: int = 2, train_start=None,
                 embargo: int = 5, min_row: int = 0, end=None) -> Fold:
    hs = pd.Timestamp(holdout_start)
    vs = f"{hs.year - val_years}-01-01"
    end = end or (dates[-1] + pd.Timedelta(days=1))
    return make_fold(dates, "holdout", train_start or dates[min_row], vs, hs, end, embargo, min_row)


class HoldoutLedger:
    path = RUNS_DIR / "holdout_ledger.jsonl"

    @classmethod
    def unlock(cls, purpose: str, details: dict | None = None) -> None:
        cls.path.parent.mkdir(parents=True, exist_ok=True)
        with open(cls.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "purpose": purpose,
                                 **(details or {})}) + "\n")

    @classmethod
    def count(cls) -> int:
        if not cls.path.exists():
            return 0
        return sum(1 for line in cls.path.read_text(encoding="utf-8").splitlines() if line.strip())
