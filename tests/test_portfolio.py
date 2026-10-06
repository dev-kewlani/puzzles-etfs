"""Simulator accounting on toy books with known answers: drift, cash, turnover, the no-trade band, the long-short
spread, the cap on the long book, the rules and the loan."""
import numpy as np

from etf66 import portfolio


def run(target, reb, r, cash=None, band=0.0, use_cash=True, max_book=1.0):
    """Return the simulate() result of a toy book."""
    cash = np.zeros(len(target)) if cash is None else cash
    return portfolio.simulate(np.asarray(target, float), np.asarray(reb), np.asarray(r, float), cash, band, use_cash,
                              max_book)


def test_drift_cash_and_turnover():
    g, tv = run([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]], [True, False, True],
                            [[0.10, 0.0], [0.0, 0.0], [0.0, 0.0]])
    # Row 0: the book starts flat and buys 0.5 of an ETF that gains 10%.
    assert np.isclose(g[0], 0.05)
    assert np.isclose(tv[0], 0.5)
    # The weight drifts to 0.55 / 1.05. The rebalance at row 2 trades it back to 0.5.
    assert np.isclose(tv[2], abs(0.5 - 0.55 / 1.05))


def test_band_keeps_small_changes():
    # A change of 0.01 is below the 0.02 band, so the held position does not trade.
    _, tv = run([[0.50], [0.51]], [True, True], np.zeros((2, 1)), band=0.02)
    assert np.isclose(tv[1], 0.0)


def test_long_short_spread():
    # Long +2% and short -1%: the book earns 3%. The long-short book has no cash.
    g, _ = run([[1.0, -1.0]], [True], [[0.02, -0.01]], use_cash=False)
    assert np.isclose(g[0], 0.03)


def test_book_scales_to_max_long_book():
    g, tv = run([[0.5, 0.5]], [True], [[0.1, 0.0]], max_book=0.5)
    assert np.isclose(tv[0], 1.5)          # 1.0 bought, then 0.5 cut to the 0.5 cap; the cut counts as turnover
    assert np.isclose(g[0], 0.025)         # 0.25 in the first ETF earns 10%


def test_ts_gate_slots():
    ts = np.array([[9, 9] + [1] * 18], dtype=float)
    cs = np.ones_like(ts)
    ok = np.ones_like(ts, dtype=bool)
    # 20 tradable ETFs and a slot fraction of 0.1 give 2 slots.
    assert np.isclose(portfolio.ts_gate_cash(ts, cs, ok, 9, 0, 0.1).sum(), 1.0)   # 2 slots, 2 pass
    ts[0, 1] = 1
    assert np.isclose(portfolio.ts_gate_cash(ts, cs, ok, 9, 0, 0.1).sum(), 0.5)   # 1 of 2 slots used, the rest is cash


def test_score_capped_respects_caps():
    pct = np.linspace(0, 1, 10)[None, :]
    codes = np.array([0] * 5 + [1] * 5)
    w = portfolio.score_capped(pct, codes, 0.15, 0.40, 10)     # 10 ETFs in 2 asset groups
    assert w.max() <= 0.15 + 1e-12
    assert w[0, codes == 1].sum() <= 0.40 + 1e-12


def test_rule_labels_keep_their_names():
    labels = [r["label"] for r in portfolio.expand_rules(portfolio.RULES)]
    assert "top_bins_equal_top_bins1" in labels
    assert "score_capped_max_weight0.15_max_block0.4" in labels
    assert "ts_gate_cash_ts_min7_cs_min0" in labels
    assert "long_short_top_bins1" in labels


def test_levered_book_pays_the_loan():
    r, cash = np.array([[0.01, 0.0]]), np.array([0.0001])
    g, tv = portfolio.simulate_levered(np.array([[1.5, 0.0]]), np.array([True]), r, cash, 0.0002)
    # 1.5 x 1%, minus the loan of 0.5 at the cash return plus the spread.
    assert np.isclose(g[0], 1.5 * 0.01 - 0.5 * (0.0001 + 0.0002))
    assert np.isclose(tv[0], 1.5)


def test_unlevered_book_earns_cash():
    # Half the book is cash and earns the cash return.
    g, _ = portfolio.simulate_levered(np.array([[0.5]]), np.array([True]), np.array([[0.02]]), np.array([0.001]), 0.01)
    assert np.isclose(g[0], 0.5 * 0.02 + 0.5 * 0.001)
