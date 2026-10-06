"""Lookahead canary for the features. The test changes every price and volume after a cut session. No feature value
on or before the cut may change.

The values are the normalized features, so the check covers the cross-sectional rank and the expanding percentile.
The test covers the 11 groups that read prices only. The macro, calendar and pca_stock groups need external series.
The synthetic data has none.
"""
import numpy as np

from etf66.features.build import build_features
from tests.conftest import context_from_bars

GROUPS = ["price", "volatility", "drawdown", "trend", "candles", "gaps", "compression", "volume", "relative",
          "screeners", "pca_etf"]


def test_features_do_not_see_the_future(bars):
    cut = 450
    rng = np.random.default_rng(1)
    moved = {k: v.copy() for k, v in bars.items()}
    n_after = len(bars["close"]) - cut - 1
    # A random walk with 5% daily steps after the cut, so the future is far from the past.
    shock = np.exp(np.cumsum(rng.normal(0, 0.05, (n_after, bars["close"].shape[1])), axis=0))
    for k in ("open", "high", "low", "close"):
        moved[k].iloc[cut + 1:] = moved[k].iloc[cut + 1:].to_numpy() * shock
    moved["volume"].iloc[cut + 1:] *= 3.0
    # adv reads close and volume, so it changes with them.
    moved["adv"] = (moved["close"] * moved["volume"]).rolling(30, min_periods=24).median()
    a = build_features(context_from_bars(bars), GROUPS)
    b = build_features(context_from_bars(moved), GROUPS)
    bad = [n for i, n in enumerate(a.names)
           if not np.allclose(a.values[: cut + 1, :, i], b.values[: cut + 1, :, i], equal_nan=True, atol=1e-6)]
    assert not bad, f"features that change before the cut: {bad}"
    # The check means nothing if most values are NaN.
    assert np.isfinite(a.values[300:]).mean() > 0.5
