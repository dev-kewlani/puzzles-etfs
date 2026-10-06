"""Groups stage. The group map, the group returns and features over the tradable members, group labels that read
no return past their own window, the purge, and the ETF scores."""
import numpy as np
import pandas as pd
import pytest

from etf66 import config, groups

TABLE = pd.DataFrame({"ticker": ["SPY", "IWM", "XLK", "TLT", "GLD", "SHY"],
                      "block_name": ["equity", "equity", "equity", "rates", "precious_metals", "rates"],
                      "role": ["broad_us_equity", "small_cap", "sector", "long_treasury", "gold", "short_treasury"]})
TICKERS = TABLE["ticker"].tolist()


def test_group_map_splits_equity_by_role_and_drops_the_cash_etf():
    names, gmap = groups.group_map(TABLE, TICKERS)
    # SPY and IWM join eq_broad, XLK joins eq_sector. SHY is cash-like and gets -1 (no group).
    assert names == ["eq_broad", "eq_sector", "precious_metals", "rates"]
    assert gmap.tolist() == [0, 0, 1, 3, 2, -1]


def test_group_map_refuses_an_unknown_equity_role():
    table = TABLE.copy()
    table.loc[2, "role"] = "nonsense"
    with pytest.raises(config.ConfigError, match="EQUITY_SPLIT"):
        groups.group_map(table, TICKERS)


def test_group_returns_and_features_are_means_over_tradable_members():
    names, gmap = groups.group_map(TABLE, TICKERS)
    r = np.array([[0.01, 0.03, 0.02, 0.0, -0.01, 0.0], [0.02, np.nan, 0.01, 0.0, 0.0, 0.0]])
    tradable = np.array([[True, True, True, True, False, True], [True, True, True, True, True, True]])
    g_ret, g_ok = groups.group_returns(r, tradable, gmap, len(names))
    # Row 0: eq_broad is the mean of SPY and IWM. GLD is not tradable, so precious_metals has no return.
    assert np.isclose(g_ret[0, 0], 0.02) and np.isnan(g_ret[0, 2]) and not g_ok[0, 2]
    assert np.isclose(g_ret[1, 0], 0.02) and g_ok[1, 2]              # IWM has no return at row 1
    feats = np.arange(2 * 6 * 2, dtype=np.float32).reshape(2, 6, 2)
    gf = groups.group_features(feats, tradable, gmap, len(names))
    assert np.allclose(gf[0, 0], feats[0, [0, 1]].mean(axis=0)) and np.isnan(gf[0, 2]).all()


def test_group_labels_read_only_their_own_window():
    rng = np.random.default_rng(5)
    g_ret = rng.normal(0, 0.01, (120, 3))
    g_ok = np.ones_like(g_ret, dtype=bool)
    h, m = 5, 5
    lab = groups.group_labels(g_ret, g_ok, h, m, 0.003)
    t = 40
    # The vol-scaled forward return of each group, ranked among the 3 groups (average ties) and scaled to 0..1.
    want = (np.prod(1 + g_ret[t:t + h], axis=0) - 1) / (np.maximum(g_ret[t:t + m].std(axis=0, ddof=1), 0.003) * np.sqrt(h))
    assert np.allclose(lab[t], pd.Series(want).rank().sub(1).div(2).to_numpy())
    assert np.isnan(lab[-(m + 1):]).all() and np.isfinite(lab[: -(m + 1)]).all()
    # Change every return from row t + m on. The labels up to row t must not move.
    moved = g_ret.copy()
    moved[t + m:] = rng.normal(0, 0.05, moved[t + m:].shape)
    assert np.allclose(groups.group_labels(moved, g_ok, h, m, 0.003)[: t + 1], lab[: t + 1])


def test_train_end_row_leaves_the_label_window_before_the_retrain():
    d, h = 500, 21
    end = groups.train_end_row(d, h, 2)
    assert end == d - 21 - 21 - 2                      # d - h - max(h, 5) - purge_extra
    assert end + max(h, 5) + 1 < d                     # the last label row reads returns that end before d


def test_etf_scores_rank_each_etf_by_its_group():
    names, gmap = groups.group_map(TABLE, TICKERS)
    # Each ETF takes the score of its group: XLK 0.9, GLD 0.5, TLT 0.3, SPY and IWM 0.1. SHY is not eligible.
    score = np.array([[0.1, 0.9, 0.5, 0.3]])
    eligible = np.array([[True, True, True, True, True, False]])
    got = groups.etf_scores(score, gmap, eligible)
    assert np.isnan(got[0, 5])
    assert got[0, 2] == 1.0 and got[0, 0] == got[0, 1] and got[0, 0] < got[0, 3] < got[0, 4]
