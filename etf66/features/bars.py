"""Per-ETF features from the shape of the bars: candles, gaps, compression and breakout, volume and liquidity.

Every value at session t uses data up to the close of t. A "prior" window (for example a breakout above the prior
20-session high) ends at t - 1.
"""
from __future__ import annotations

import numpy as np

from etf66 import ops
from etf66.features.context import Context, log_safe, on_balance_volume

PERCENT = 100.0               # the money flow index uses percent units

CANDLES = {
    "bar_sessions": [1, 5, 21],           # must hold 1: the means below read the 1-session candle
    "mean_windows": [5, 21],
    "range_mean_window": 21,
    "range_ratio_window": 5,
    "clv_window": 21,
}
GAPS = {
    "gap_threshold": 0.005,               # a gap above this size counts as a gap up or down
    "fill_threshold": 0.0005,             # a gap above this size counts for the fill rate
    "mean_window": 21,
    "balance_window": 63,
    "recovery_window": 63,
    "down_recovery_window": 126,
    "follow_window": 63,
    "overnight_windows": [21, 63],
    "share_window": 63,                   # must be in overnight_windows: overnight_share reads those sums
    "fill_window": 66,
}
COMPRESSION = {
    "expansion_window": 21,
    "hl_pair": [21, 126],
    "squeeze_window": 63,
    "donchian_windows": [20, 55],
    "donchian_width_window": 20,
    "vcp": {"window": 21, "band": 0.2},
    "narrow_range": {"range_min": 7, "window": 21},
    "inside_window": 21,
    "breakout_windows": [20, 50],
}
VOLUME = {
    "dollar_volume_z_windows": [21, 63],
    "ratio_pairs": [[5, 63], [21, 252]],
    "migration_window": 21,
    "block_product": {"sum": 6, "shifts": [30, 5, 20]},
    "up_down_windows": [21, 63],
    "up_down_offset": 1.0,                # added to the down-volume sum so that the ratio stays finite
    "climactic_window": 21,
    "pvt_window": 21,
    "obv_slope_windows": [21, 63],
    "cmf_window": 20,
    "mfi_window": 14,
    "amihud": {"window": 63, "scale": 1.0e9},
    "range_per_dollar": {"window": 21, "scale": 1.0e9},
    "liquidity_relief": [21, 126],
    "kyle": {"window": 63, "scale": 1.0e6},
    "attention": {"base": 60, "diff": 5, "mean": 10, "mean_min_n": 8},
}


def multi_session_bar(x: Context, n: int):
    """Return the open, high, low and close of the n-session window that ends at t."""
    high = ops.ts_max(x.high, n) if n > 1 else x.high
    low = ops.ts_min(x.low, n) if n > 1 else x.low
    return x.open.shift(n - 1), high, low, x.close


def candles(x: Context) -> dict:
    """Return candle-shape features for each bar length in CANDLES, and rolling means of the 1-session candle."""
    p = CANDLES
    f = {}
    for n in p["bar_sessions"]:
        open_, high, low, close = multi_session_bar(x, n)
        rng = high - low
        f[f"candle_body_{n}"] = ops.safe_div((close - open_).abs(), rng)
        f[f"candle_upper_wick_{n}"] = ops.safe_div(high - np.maximum(open_, close), rng)
        f[f"candle_lower_wick_{n}"] = ops.safe_div(np.minimum(open_, close) - low, rng)
        f[f"candle_close_location_{n}"] = ops.safe_div(close - low, rng)
        f[f"candle_return_{n}"] = close / open_ - 1.0
    for name in ("candle_body_1", "candle_upper_wick_1", "candle_lower_wick_1", "candle_close_location_1"):
        for w in p["mean_windows"]:
            f[f"{name}_mean_{w}"] = ops.ts_mean(f[name], w)
    f["range_1"] = ops.safe_div(x.high - x.low, x.close)
    w = p["range_mean_window"]
    f[f"range_mean_{w}"] = ops.ts_mean(f["range_1"], w)
    k = p["range_ratio_window"]
    f[f"range_ratio_{k}_vs_prior_{k}"] = ops.safe_div(ops.ts_sum(x.high - x.low, k), ops.ts_sum(x.high - x.low, k).shift(k))
    clv = ops.safe_div((x.close - x.low) - (x.high - x.close), x.high - x.low)
    w = p["clv_window"]
    f[f"clv_imbalance_{w}"] = ops.safe_div(ops.ts_sum(clv * x.volume, w), ops.ts_sum(x.volume, w))
    return f


def gaps(x: Context) -> dict:
    """Return gap, gap-recovery, gap-follow and overnight versus intraday features."""
    p = GAPS
    open_, close, high, low = x.open, x.close, x.high, x.low
    big, small = p["gap_threshold"], p["fill_threshold"]
    gap = open_ / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    present = gap.notna()
    w = p["mean_window"]
    f = {"gap_1": gap, f"gap_mean_{w}": ops.ts_mean(gap, w), f"gap_abs_sum_{w}": ops.ts_sum(gap.abs(), w)}
    w = p["balance_window"]
    f[f"gap_balance_{w}"] = ops.ts_sum(((gap > big).astype(float) - (gap < -big)).where(present), w)
    recovery = ops.safe_div(close - open_, (close.shift(1) - open_).abs()).clip(lower=0)
    down = (gap < -big)
    w = p["recovery_window"]
    f[f"gap_recovery_average_{w}"] = ops.ts_mean((recovery * down).where(present), w)
    w = p["down_recovery_window"]
    n_down = ops.ts_sum(down.astype(float).where(present), w)
    f[f"gap_down_recovery_rate_{w}"] = ops.safe_div(ops.ts_sum((down & (close > open_)).astype(float).where(present), w),
                                                    n_down)
    up = (gap > big)
    w = p["follow_window"]
    f[f"gap_follow_through_{w}"] = ops.safe_div(ops.ts_sum((up & (close > open_)).astype(float).where(present), w),
                                                ops.ts_sum(up.astype(float).where(present), w))
    for w in p["overnight_windows"]:
        f[f"overnight_sum_{w}"] = ops.ts_sum(gap, w)
        f[f"intraday_sum_{w}"] = ops.ts_sum(intraday, w)
    w = p["share_window"]
    f[f"overnight_share_{w}"] = ops.safe_div(f[f"overnight_sum_{w}"],
                                             f[f"overnight_sum_{w}"].abs() + f[f"intraday_sum_{w}"].abs())
    f[f"overnight_minus_intraday_{w}"] = f[f"overnight_sum_{w}"] - f[f"intraday_sum_{w}"]
    filled_up = (gap > small) & (low <= close.shift(1))
    filled_dn = (gap < -small) & (high >= close.shift(1))
    w = p["fill_window"]
    f[f"gap_fill_rate_{w}"] = ops.safe_div(ops.ts_sum((filled_up | filled_dn).astype(float).where(present), w),
                                           ops.ts_sum((gap.abs() > small).astype(float).where(present), w))
    return f


def compression(x: Context) -> dict:
    """Return range compression, squeeze, channel position, inside-bar and breakout features."""
    p = COMPRESSION
    high, low, close = x.high, x.low, x.close
    rng = high - low
    w = p["expansion_window"]
    short, long = p["hl_pair"]
    sq = p["squeeze_window"]
    f = {f"range_expansion_{w}": ops.safe_div(rng, ops.ts_mean(rng, w)),
         f"hl_compression_{short}_{long}": ops.safe_div(ops.ts_sum(rng, short), ops.ts_sum(rng, long)) * (long / short),
         f"bollinger_squeeze_{sq}": ops.safe_div(ops.ts_std(close, sq), ops.ts_mean(close, sq))}
    for w in p["donchian_windows"]:
        lo, hi = ops.ts_min(low, w), ops.ts_max(high, w)
        f[f"donchian_position_{w}"] = ops.safe_div(close - lo, hi - lo)
    w = p["donchian_width_window"]
    f[f"donchian_width_{w}"] = ops.safe_div(ops.ts_max(high, w) - ops.ts_min(low, w), close)
    vcp = p["vcp"]
    w = vcp["window"]
    lo, hi = ops.ts_min(low, w), ops.ts_max(high, w)
    f[f"vcp_touch_count_{w}"] = ops.ts_sum((close <= lo + vcp["band"] * (hi - lo)).astype(float).where(close.notna()), w)
    nr = p["narrow_range"]
    range_min = ops.ts_min(rng, nr["range_min"])
    f[f"nr{nr['range_min']}_count_{nr['window']}"] = ops.ts_sum((rng <= range_min).astype(float).where(rng.notna()),
                                                                nr["window"])
    inside = (high < high.shift(1)) & (low > low.shift(1))
    w = p["inside_window"]
    f[f"inside_bar_count_{w}"] = ops.ts_sum(inside.astype(float).where(high.notna()), w)
    for n in p["breakout_windows"]:
        f[f"breakout_up_{n}"] = close / ops.ts_max(high, n).shift(1) - 1.0
        f[f"breakdown_{n}"] = ops.ts_min(low, n).shift(1) / close - 1.0
    return f


def volume(x: Context) -> dict:
    """Return volume, dollar-volume, money-flow and liquidity features.

    Dollar volume here is the adjusted close times the raw volume.
    """
    p = VOLUME
    high, low, close, vol, ret = x.high, x.low, x.close, x.volume, x.ret
    dollar_volume = close * vol
    log_dv = log_safe(dollar_volume)
    f = {}
    for w in p["dollar_volume_z_windows"]:
        f[f"dollar_volume_z_{w}"] = ops.ts_zscore(log_dv, w)
    for a, b in p["ratio_pairs"]:
        f[f"volume_ratio_{a}_{b}"] = ops.safe_div(ops.ts_mean(vol, a), ops.ts_mean(vol, b))
    w = p["migration_window"]
    f[f"volume_migration_{w}"] = ops.safe_div(ops.ts_sum(vol, w), ops.ts_sum(vol, w).shift(w))
    bp = p["block_product"]
    s1, s2, s3 = bp["shifts"]
    vsum = ops.ts_sum(vol, bp["sum"])
    f["volume_block_product"] = (ops.safe_div(ops.ts_sum(vol, bp["sum"]), vsum.shift(s1))
                                 * ops.safe_div(vsum.shift(s2), vsum.shift(s3)))
    for w in p["up_down_windows"]:
        f[f"up_down_volume_ratio_{w}"] = ops.safe_div(ops.ts_sum(vol.where(ret > 0, 0.0), w),
                                                      ops.ts_sum(vol.where(ret < 0, 0.0), w) + p["up_down_offset"])
    w = p["climactic_window"]
    f[f"climactic_volume_{w}"] = ops.safe_div(vol, ops.ts_mean(vol, w))
    w = p["pvt_window"]
    f[f"price_volume_trend_{w}"] = ops.safe_div(ops.ts_sum(vol * ret, w), ops.ts_sum(vol, w))
    obv = on_balance_volume(ret, vol)
    for w in p["obv_slope_windows"]:
        f[f"obv_slope_{w}"] = ops.safe_div(ops.ts_slope(obv, w), ops.ts_mean(vol, w))
    money_flow_mult = ops.safe_div((close - low) - (high - close), high - low)
    w = p["cmf_window"]
    f[f"chaikin_money_flow_{w}"] = ops.safe_div(ops.ts_sum(money_flow_mult * vol, w), ops.ts_sum(vol, w))
    typical = (high + low + close) / 3.0
    pos = (typical * vol).where(typical > typical.shift(1), 0.0)
    neg = (typical * vol).where(typical < typical.shift(1), 0.0)
    w = p["mfi_window"]
    f[f"money_flow_index_{w}"] = PERCENT * ops.safe_div(ops.ts_sum(pos, w), ops.ts_sum(pos, w) + ops.ts_sum(neg, w))
    am = p["amihud"]
    f[f"amihud_{am['window']}"] = log_safe(ops.ts_mean(ops.safe_div(ret.abs(), dollar_volume), am["window"]) * am["scale"])
    rp = p["range_per_dollar"]
    f[f"range_per_dollar_{rp['window']}"] = log_safe(ops.safe_div(ops.ts_sum(high - low, rp["window"]),
                                                                  ops.ts_sum(dollar_volume, rp["window"])) * rp["scale"])
    short, long = p["liquidity_relief"]
    f["liquidity_relief"] = (ops.safe_div(ops.ts_mean(dollar_volume, short), ops.ts_mean(dollar_volume, long))
                             * (ops.ts_std(ret, short) < ops.ts_std(ret, long)).astype(float))
    signed_vol = np.sign(ret) * vol
    ky = p["kyle"]
    f[f"kyle_proxy_{ky['window']}"] = ops.safe_div(ops.ts_cov(ret, signed_vol, ky["window"]),
                                                   ops.ts_cov(signed_vol, signed_vol, ky["window"])) * ky["scale"]
    # volume_attention_x1: z-score of log volume against the prior `base` sessions; x2: its change over `diff`
    # sessions; x4: the mean over `mean` sessions of the cross-sectional rank of x2, scaled to [-1, 1].
    at = p["attention"]
    log_vol = log_safe(vol)
    prior = log_vol.shift(1)
    x1 = ops.safe_div(log_vol - ops.ts_mean(prior, at["base"]), ops.ts_std(prior, at["base"]))
    f["volume_attention_x1"] = x1
    f["volume_attention_x2"] = x1 - x1.shift(at["diff"])
    ranked = 2.0 * ops.cs_rank(f["volume_attention_x2"], x.tradable) - 1.0
    f["volume_attention_x4"] = ranked.rolling(at["mean"], min_periods=at["mean_min_n"]).mean()
    f["log_adv"] = log_safe(x.adv)
    return f
