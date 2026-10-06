"""Per-ETF features against other assets (relative strength, beta, capture) and the screener lenses.

The screener lenses come from the owner's earlier equity screeners. Each lens is a 0/1 flag, and some lenses also
give a continuous score. The lens codes:
  x3  volatility expansion          o2  Corwin-Schultz high-low spread    o3  Roll serial-covariance spread
  p2  long uptrend confirmation     x5  left tail larger than right tail   r1  compressed range, then expanding
  g5  gap-downs with recoveries     a4  upside volatility at an extreme    m2  on-balance-volume divergence
The original screeners ranked a value against the full history, which is lookahead. Here a time-series percentile
ranks the value against earlier values of the same ETF only (ops.ts_rank_past).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf66 import ops
from etf66.features.context import MARKET_TICKER, Context, log_safe, on_balance_volume, true_range

CORWIN_SCHULTZ_K = 3.0 - 2.0 * np.sqrt(2.0)   # the constant of the Corwin-Schultz (2012) spread estimator

RELATIVE = {
    "beta_windows": [63, 252],
    "decoupling_pair": [63, 252],         # must be in beta_windows
    "down_beta": {"window": 63, "min_n": 20},
    "corr_windows": [21, 63],
    "capture": {"window": 63, "min_n": 20},
    "rel_return_windows": [5, 21, 63],
    "residual_window": 63,                # must be in beta_windows
    "group_rel_windows": [5, 21, 63],     # return against the mean of the asset group
    "group_z_window": 5,
    "drivers": ["TLT", "HYG", "GLD", "UUP", "VXX"],
    "driver_window": 63,
    "beta_drivers": ["TLT", "UUP"],
    "beta_driver_window": 63,
    "lead": {"lag": 1, "window": 63},
}
SCREENERS = {
    "annualize_sessions": 252,
    "x3": {"short_vol": 21, "long_vol": 63, "pct_window": 252, "pct_min_n": 200, "pct_min": 0.99, "ratio_lag": 252,
           "ratio_min": 2.0},
    "o2": {"alpha_clip": 5, "spread_cap": 0.5, "window": 63, "min_n": 32, "z_window": 252, "cs_min": 0.9, "z_min": 3},
    "o3": {"jump_max": 0.5, "window": 63, "cs_min": 0.99},
    "p2": {"ma": 200, "ma_min_n": 150, "share": 126, "share_min_n": 63, "cs_min": 0.95, "run_min": 200},
    "x5": {"window": 252, "q_hi": 0.95, "q_lo": 0.05, "pct_window": 252, "pct_min_n": 200, "pct_max": 0.01},
    "r1": {"atr": 14, "atr_min_n": 10, "pct_window": 252, "pct_min_n": 200, "low_window": 63, "low_min_n": 40,
           "low_max": 0.10, "jump": 1.5, "jump_lags": [5, 21]},
    "g5": {"vol_window": 63, "vol_mult": 1.0, "count_window": 63, "count_min": 6, "recovery_window": 126,
           "recovery_min": 0.6},
    "a4": {"window": 21, "min_n": 8, "pct_window": 252, "pct_min_n": 200, "pct_min": 0.98},
    "m2": {"window": 63, "obv_cs_min": 0.8, "price_cs_max": 0.5},
}


def relative(x: Context) -> dict:
    """Return beta, correlation, capture, relative-return, residual-momentum and asset-group-relative features."""
    p = RELATIVE
    market = MARKET_TICKER
    ret, mask = x.ret, x.tradable
    mkt = x.etf(market)
    tag = market.lower()
    ew = ops.cs_mean(ret, mask)
    f = {}
    for w in p["beta_windows"]:
        f[f"beta_{tag}_{w}"] = ops.ts_beta(ret, mkt, w)
    db = p["down_beta"]
    f[f"down_beta_{tag}_{db['window']}"] = ops.ts_beta(ret.where(mkt < 0), mkt.where(mkt < 0), db["window"], min_n=db["min_n"])
    a, b = p["decoupling_pair"]
    f[f"beta_decoupling_{a}_{b}"] = f[f"beta_{tag}_{a}"] - f[f"beta_{tag}_{b}"]
    for w in p["corr_windows"]:
        f[f"corr_{tag}_{w}"] = ops.ts_corr(ret, mkt, w)
    cap = p["capture"]
    for name, sessions in ((f"up_capture_{cap['window']}", ew > 0), (f"down_capture_{cap['window']}", ew < 0)):
        in_sessions = ops.broadcast(sessions, ret).astype(bool)
        etf_mean = ret.where(in_sessions).rolling(cap["window"], min_periods=cap["min_n"]).mean()
        ew_mean = ew.where(sessions).rolling(cap["window"], min_periods=cap["min_n"]).mean()
        f[name] = ops.safe_div(etf_mean, ops.broadcast(ew_mean, ret))
    close, mkt_close = x.close, x.close[market]
    for k in p["rel_return_windows"]:
        f[f"relative_return_{tag}_{k}"] = (close / close.shift(k) - 1.0).sub(mkt_close / mkt_close.shift(k) - 1.0, axis=0)
    w = p["residual_window"]
    f[f"residual_momentum_{w}"] = ops.ts_mean(ret, w) - f[f"beta_{tag}_{w}"] * ops.broadcast(ops.ts_mean(mkt, w), ret)
    for k in p["group_rel_windows"]:
        rk = close / close.shift(k) - 1.0
        f[f"block_relative_return_{k}"] = rk - ops.group_mean(rk, x.asset_groups, mask)
    k = p["group_z_window"]
    rk = close / close.shift(k) - 1.0
    group_mean, group_sq = ops.group_mean(rk, x.asset_groups, mask), ops.group_mean(rk ** 2, x.asset_groups, mask)
    f[f"block_z_return_{k}"] = ops.safe_div(rk - group_mean, np.sqrt((group_sq - group_mean ** 2).clip(lower=0)))
    for d in p["drivers"]:
        if d in ret.columns:
            f[f"corr_{d.lower()}_{p['driver_window']}"] = ops.ts_corr(ret, x.etf(d), p["driver_window"])
    for d in p["beta_drivers"]:
        f[f"beta_{d.lower()}_{p['beta_driver_window']}"] = ops.ts_beta(ret, x.etf(d), p["beta_driver_window"])
    lead = p["lead"]
    f[f"lead_corr_{tag}_lag{lead['lag']}_{lead['window']}"] = ops.ts_corr(ret, mkt.shift(lead["lag"]), lead["window"])
    return f


def screeners(x: Context) -> dict:
    """Return the screener lenses as 0/1 flags plus their continuous scores (all point in time)."""
    p = SCREENERS
    open_, high, low, close, vol, ret = x.open, x.high, x.low, x.close, x.volume, x.ret
    present = ret.notna()
    annual = np.sqrt(p["annualize_sessions"])
    f = {}
    c = p["x3"]
    rv_short = ops.ts_std(ret, c["short_vol"]) * annual
    rv_long = ops.ts_std(ret, c["long_vol"]) * annual
    rv_short_pct = ops.ts_rank_past(rv_short, c["pct_window"], c["pct_min_n"])
    f["x3_vol_expansion_ratio"] = ops.safe_div(rv_long, rv_long.shift(c["ratio_lag"]))
    f["x3_flag"] = ((rv_short_pct >= c["pct_min"]) & (f["x3_vol_expansion_ratio"] > c["ratio_min"])
                    ).astype(float).where(rv_short_pct.notna())
    c = p["o2"]
    hl_sq = log_safe(high / low) ** 2
    beta = hl_sq + hl_sq.shift(1)
    gamma = log_safe(pd.concat([high, high.shift(1)], keys=[0, 1]).groupby(level=1).max().reindex(high.index)
                     / pd.concat([low, low.shift(1)], keys=[0, 1]).groupby(level=1).min().reindex(low.index)) ** 2
    k = CORWIN_SCHULTZ_K
    alpha = ((np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)).clip(-c["alpha_clip"], c["alpha_clip"])
    spread = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(0, c["spread_cap"])
    name = f"o2_cs_spread_{c['window']}"
    f[name] = spread.rolling(c["window"], min_periods=c["min_n"]).mean()
    o2z = ops.ts_zscore(f[name], c["z_window"])
    f["o2_flag"] = ((ops.cs_rank(f[name], x.tradable) >= c["cs_min"]) & (o2z >= c["z_min"])).astype(float).where(o2z.notna())
    c = p["o3"]
    dp = log_safe(close).diff().where(lambda d: d.abs() <= c["jump_max"])
    cov1 = ops.ts_cov(dp, dp.shift(1), c["window"])
    name = f"o3_roll_spread_{c['window']}"
    f[name] = (2.0 * np.sqrt((-cov1).clip(lower=0))).where(cov1.notna())
    f["o3_flag"] = (ops.cs_rank(f[name], x.tradable) >= c["cs_min"]).astype(float).where(cov1.notna())
    c = p["p2"]
    ma = close.rolling(c["ma"], min_periods=c["ma_min_n"]).mean()
    above = (close >= ma) & ma.notna()
    share_above = above.astype(float).where(ma.notna()).rolling(c["share"], min_periods=c["share_min_n"]).mean()
    run = ops.ts_run_length(above)
    f["p2_flag"] = ((ops.cs_rank(share_above, x.tradable) >= c["cs_min"]) & (run >= c["run_min"])
                    ).astype(float).where(share_above.notna())
    c = p["x5"]
    tail = ops.safe_div(ops.ts_quantile(ret, c["window"], c["q_hi"]).abs(), ops.ts_quantile(ret, c["window"], c["q_lo"]).abs())
    tail_pct = ops.ts_rank_past(tail, c["pct_window"], c["pct_min_n"])
    f["x5_tail_ratio_pct"] = tail_pct
    f["x5_flag"] = (tail_pct <= c["pct_max"]).astype(float).where(tail_pct.notna())
    c = p["r1"]
    atr_n = ops.safe_div(true_range(high, low, close).rolling(c["atr"], min_periods=c["atr_min_n"]).mean(), close)
    atr_pct = ops.ts_rank_past(atr_n, c["pct_window"], c["pct_min_n"])
    compressed = atr_pct.rolling(c["low_window"], min_periods=c["low_min_n"]).min() <= c["low_max"]
    jump = compressed
    for lag in c["jump_lags"]:
        jump = jump & (atr_n / atr_n.shift(lag) > c["jump"])
    f["r1_flag"] = jump.astype(float).where(atr_pct.notna())
    c = p["g5"]
    gap = open_ / close.shift(1) - 1.0
    gap_down = (gap < -(c["vol_mult"] * ops.ts_std(ret, c["vol_window"]))) & present
    count_name = f"g5_gap_down_count_{c['count_window']}"
    f[count_name] = ops.ts_sum(gap_down.astype(float).where(present), c["count_window"])
    rate_name = f"g5_recovery_rate_{c['recovery_window']}"
    f[rate_name] = ops.safe_div(ops.ts_sum((gap_down & (close > open_)).astype(float).where(present), c["recovery_window"]),
                                ops.ts_sum(gap_down.astype(float).where(present), c["recovery_window"]))
    f["g5_flag"] = ((f[count_name] >= c["count_min"]) & (f[rate_name] >= c["recovery_min"])).astype(float).where(present)
    c = p["a4"]
    rv_up = ops.ts_std(ret.where(ret > 0), c["window"], min_n=c["min_n"])
    a4 = ops.ts_rank_past(rv_up, c["pct_window"], c["pct_min_n"])
    f["a4_upside_vol_pct"] = a4
    f["a4_flag"] = (a4 >= c["pct_min"]).astype(float).where(a4.notna())
    c = p["m2"]
    obv_slope = ops.ts_slope(on_balance_volume(ret, vol), c["window"])
    price_slope = ops.ts_slope(log_safe(close), c["window"])
    f["m2_flag"] = ((ops.cs_rank(obv_slope, x.tradable) >= c["obv_cs_min"])
                    & (ops.cs_rank(price_slope, x.tradable) < c["price_cs_max"])).astype(float).where(price_slope.notna())
    flags = [k for k in f if k.endswith("_flag")]
    f["lens_vote_count"] = sum(f[k].fillna(0.0) for k in flags).where(present)
    return f
