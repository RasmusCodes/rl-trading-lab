"""Look-ahead tests: deleting the future must not change anything computed for the past.

For random cut points t, features (and baseline weights) computed on rows 0..t must equal row t of the
full-sample computation. The test is also run on the deliberately leaky features to prove it catches them.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt.data.features import ALL_FEATURES, compute_features  # noqa: E402
from rlt.data.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def market():
    md = make_synthetic("garch", T=1500, n_assets=2, seed=3)
    return md


def _same(a, b):
    return np.allclose(np.nan_to_num(a, nan=-999.0), np.nan_to_num(b, nan=-999.0), rtol=1e-9, atol=1e-10)


def test_features_are_causal(market):
    fit_end = 900  # HMM fitted on rows <= 900 in both computations
    full = compute_features(market, ALL_FEATURES, fit_end=fit_end)
    for cut in (950, 1100, 1333, 1499):
        part = compute_features(market.slice(0, cut + 1), ALL_FEATURES, fit_end=fit_end)
        bad = [n for k, n in enumerate(full.names) if not _same(full.values[cut, :, k], part.values[cut, :, k])]
        assert not bad, f"features changed at row {cut} when the future was removed: {bad}"


@pytest.mark.parametrize("leak", ["future_return", "global_zscore", "smoothed_regime", "centered_ma"])
def test_leak_detector_catches_planted_leaks(market, leak):
    fit_end = 900
    names = ["ret_1d"] + (["hmm_p_high"] if leak == "smoothed_regime" else [])
    full = compute_features(market, names, fit_end=fit_end, leak=(leak,))
    k = full.names.index(f"leak_{leak}")
    changed = 0
    for cut in (950, 1100, 1333, 1480):
        part = compute_features(market.slice(0, cut + 1), names, fit_end=fit_end, leak=(leak,))
        changed += not _same(full.values[cut, :, k], part.values[cut, :, k])
    assert changed > 0, f"causality test failed to flag the {leak} leak"


def test_baseline_weights_are_causal(market):
    from rlt.baselines.rules import all_rule_weights

    full = all_rule_weights(market)
    for cut in (700, 1200, 1499):
        part = all_rule_weights(market.slice(0, cut + 1))
        for name, W in full.items():
            assert _same(W[cut], part[name][cut]), f"baseline {name} uses future data at row {cut}"
