"""Bins. ts_bin at t uses no prediction after t and is NaN until 63 earlier predictions exist. cs_bin ranks the
ETFs of one session."""
import numpy as np

from etf66 import bins


def test_ts_bin_needs_warmup_and_ignores_future():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(200, 3))
    b = bins.time_series_bins(p, 63, 10)
    assert np.isnan(b[:63]).all()
    # Change every prediction from row 150 on. A bin before row 150 must not move.
    p2 = p.copy()
    p2[150:] = 99.0
    assert np.array_equal(b[:150], bins.time_series_bins(p2, 63, 10)[:150], equal_nan=True)


def test_ts_bin_top_when_max():
    # A rising series: the last prediction is above its 63 earlier ones, so it gets the top bin.
    p = np.arange(100, dtype=float)[:, None]
    assert bins.time_series_bins(p, 63, 10)[99, 0] == 10


def test_cs_bins():
    p = np.array([[1.0, 2.0, 3.0, np.nan]])
    pct = bins.cross_section_pct(p)
    assert np.allclose(pct[0, :3], [0.0, 0.5, 1.0]) and np.isnan(pct[0, 3])
    # bin = floor(pct x 10) + 1, clipped to 10: 0 gives 1, 0.5 gives 6, 1 gives 10.
    assert list(bins.to_bins(pct, 10)[0, :3]) == [1, 6, 10]


def test_cs_tie_rules():
    # ordinal breaks the tie in column order. average gives the two tied ETFs the same mean rank.
    p = np.array([[1.0, 1.0, 2.0]])
    assert np.allclose(bins.cross_section_pct(p, "ordinal")[0], [0.0, 0.5, 1.0])
    assert np.allclose(bins.cross_section_pct(p, "average")[0], [0.25, 0.25, 1.0])
