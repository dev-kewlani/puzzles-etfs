"""Labels. The label of t starts at the open of t+1 and ends at the open of t+1+h. It reads no open after t + reach,
and the last reach rows have no label."""
import numpy as np

from etf66 import labels as lab


def label_cfg(cfg, horizons):
    """Return the labels settings with other horizons."""
    return {**cfg["labels"], "horizons": horizons}


def test_label_alignment(bars, cfg):
    open_ = bars["open"]
    out = lab.build_labels(open_, open_.notna(), label_cfg(cfg, [5]), 1e-12)     # a vol floor that never binds
    t, j, h = 100, 3, 5
    r = open_.iloc[t + 1 + h, j] / open_.iloc[t + 1, j] - 1.0
    # fwd vol: the open-to-open returns of sessions t+2 to t+1+m, with m = max(h, 5).
    fwd = (open_ / open_.shift(1) - 1.0).iloc[t + 2:t + 2 + max(h, 5), j].std(ddof=1)
    assert np.isclose(out[("vol_scaled", h)].iloc[t, j], r / (fwd * np.sqrt(h)), rtol=1e-9)


def test_payoff_variants(bars, cfg):
    open_ = bars["open"]
    lc = {**label_cfg(cfg, [5]), "variants": ["rank", "payoff_l0.0", "payoff_l1.0"]}
    out = lab.build_labels(open_, open_.notna(), lc)
    # lambda 0 gives the rank label. lambda 1 gives other values, still in [0, 1].
    assert np.allclose(out[("payoff_l0.0", 5)].to_numpy(), out[("rank", 5)].to_numpy(), equal_nan=True)
    p1 = out[("payoff_l1.0", 5)].to_numpy()
    assert np.nanmin(p1) >= 0 and np.nanmax(p1) <= 1
    assert not np.allclose(p1, out[("rank", 5)].to_numpy(), equal_nan=True)


def test_last_rows_have_no_label(bars, cfg):
    open_ = bars["open"]
    out = lab.build_labels(open_, open_.notna(), label_cfg(cfg, [5]))
    reach = lab.label_reach(5, 5)                          # 1 + max(5, 5) = 6
    # The last 6 rows have no label. The second check stops the test from passing on an all-NaN table.
    for v in cfg["labels"]["variants"]:
        assert out[(v, 5)].iloc[-reach:].isna().all().all(), v
        assert out[(v, 5)].iloc[: -reach - 1].notna().any().any(), v


def test_label_does_not_read_past_reach(bars, cfg):
    open_ = bars["open"].copy()
    lc = label_cfg(cfg, [5])
    base = lab.build_labels(open_, open_.notna(), lc)
    t = 300
    # Scale every open after t + reach. The labels up to row t must not move.
    open_.iloc[t + lab.label_reach(5, 5) + 1:] *= 1.5
    moved = lab.build_labels(open_, open_.notna(), lc)
    for v in cfg["labels"]["variants"]:
        a, b = base[(v, 5)].iloc[: t + 1], moved[(v, 5)].iloc[: t + 1]
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), v
