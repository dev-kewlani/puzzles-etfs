"""Environment stage. The target at t reads the basket from the fill at t+1 on. The persistence baseline at t reads
no return after the open of t. The stress mask at t uses no prediction after t."""
import numpy as np

from etf66 import environment

import pytest

ENV = {"stress": {"input": "dd", "threshold": 0.8}}


@pytest.fixture
def small(monkeypatch):
    """Set short horizons, a short stress lookback and two toy models, so the toy series can stay short."""
    for name, value in {"HORIZONS": [3, 5], "MIN_VOL_WINDOW": 5, "MIN_CORR_ASSETS": 3, "STRESS_HORIZON": 5,
                        "STRESS_LOOKBACK": 20, "STRESS_MIN_N": 10,
                        "MODELS": [{"name": "m1", "kind": "ridge", "alpha": 1.0},
                                   {"name": "m2", "kind": "ridge", "alpha": 1.0}]}.items():
        monkeypatch.setattr(environment, name, value)


def test_path_drawdown_counts_the_peak_from_one():
    # The path starts at 1, so a first loss is a drawdown from 1.
    assert np.isclose(environment.path_drawdown(np.array([-0.1, 0.05])), -0.1)
    assert np.isclose(environment.path_drawdown(np.array([0.1, -0.1])), -0.1)
    assert environment.path_drawdown(np.array([0.01, 0.02])) == 0.0


def test_targets_and_persistence_align(small):
    rng = np.random.default_rng(2)
    etf_r = rng.normal(0, 0.01, (80, 6))
    basket = etf_r.mean(axis=1)
    tg = environment.build_targets(basket, etf_r, 252)
    pe = environment.build_persistence(basket, etf_r, 252)
    t, h, m = 30, 3, 5                                        # m = max(h, MIN_VOL_WINDOW)
    assert np.isclose(tg[("env_ret", h)][t], np.prod(1 + basket[t:t + h]) - 1)
    assert np.isclose(tg[("env_vol", h)][t], basket[t:t + m].std(ddof=1) * np.sqrt(252))
    assert np.isnan(tg[("env_ret", h)][-(m + 1):]).all()
    # The persistence value at t equals the target of row t - h - 1 (a path of h returns) or t - m - 1 (a window of
    # m returns). Both end with the return into the open of t.
    for name, back in (("env_ret", h), ("env_dd", h), ("env_vol", m), ("env_corr", m)):
        assert np.isclose(pe[(name, h)][t], tg[(name, h)][t - back - 1])


def test_global_columns_take_the_global_groups_and_the_global_pca_names():
    names = ["ret_1", "vix_change", "etf_pc1_63", "pc1_loading", "day_of_week"]
    groups = ["price", "macro", "pca_etf", "pca_etf", "calendar"]
    # Global: the macro and calendar groups, and the pca_etf names that start with "etf_". pc1_loading is per-ETF.
    assert environment.global_columns(names, groups) == [1, 2, 4]


def test_stress_marks_the_deepest_predicted_drawdowns_from_past_predictions_only(small):
    rng = np.random.default_rng(3)
    p1, p2 = rng.normal(-0.02, 0.01, 100), rng.normal(-0.02, 0.01, 100)
    preds = {("env_dd", 5, "m1"): p1, ("env_dd", 5, "m2"): p2}
    pct, stress = environment.stress_mask(preds, ENV)
    # STRESS_MIN_N = 10: the first 10 sessions have too few earlier predictions for a percentile.
    assert np.isnan(pct[:10]).all() and 0.0 < stress[10:].mean() < 0.5
    deep = np.flatnonzero(stress)
    assert deep.size and all((p1[i] + p2[i]) / 2 < np.median((p1 + p2) / 2) for i in deep)
    # Change every prediction from session 60 on. The mask before session 60 must not move.
    moved = {k: v.copy() for k, v in preds.items()}
    for v in moved.values():
        v[60:] = -0.5
    _, later = environment.stress_mask(moved, ENV)
    assert np.array_equal(stress[:60], later[:60])
