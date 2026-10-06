"""Per-ETF features from prices: momentum, volatility, drawdown and trend.

Each function returns {feature name: sessions x tickers table}. Every value at session t uses data up to the close
of t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf66 import ops
from etf66.features.context import Context, log_safe, true_range

YANG_ZHANG_ALPHA = 0.34       # the constant of the Yang-Zhang (2000) estimator
PERCENT = 100.0               # the Ulcer index and the RSI use percent units
CCI_CONSTANT = 0.015          # Lambert's constant: about 70-80% of CCI values fall in [-100, 100]

# Windows count sessions. A window that another key reads (for example pullback.long) must be in ret_windows.
PRICE = {
    "ret_windows": [1, 3, 5, 10, 21, 63, 126, 252],
    "skip_momentum": {"mom_12_1": [21, 252], "mom_6_1": [21, 126], "mom_6_3": [63, 126]},   # name: [skip, lookback]
    "sharpe_windows": [21, 63, 126, 252],
    "trend_skip": {"skip": 21, "windows": [63, 126, 252]},
    "accel_pairs": [[5, 21], [21, 63], [63, 252]],
    "pullback": {"long": 126, "short": 21},
    "drawdown_adjusted": {"ret_window": 126, "high_window": 126},
    "positive_share_windows": [21, 63, 126],
    "crash": {"threshold": -0.15, "window": 63, "bounce": 5},
    "slope_windows": [10, 21, 63],
    "slope_accel": {"base": 21, "window": 21},
    "up_day_share_windows": [21, 63, 150],
    "upside_capture_window": 63,
}
VOLATILITY = {
    "vol_windows": [10, 21, 63, 126],
    "downside_vol_windows": [21, 63],
    "up_down_vol_window": 63,
    "vol_ratio_pairs": [[21, 63], [63, 252]],
    "vol_change": {"window": 21, "lag": 21},
    "vol_of_vol": {"inner": 5, "outer": 63},
    "parkinson_windows": [10, 21, 50],
    "garman_klass_window": 21,
    "rogers_satchell_window": 21,
    "yang_zhang_windows": [21, 63],
    "atr_windows": [14, 63],
    "atr_ratio_pair": [14, 63],
    "vol_pct_self": {"windows": [21, 63], "lookback": 252, "min_n": 200},
    "moment_windows": [63, 126],
    "worst_day_window": 21,
    "expected_shortfall": {"window": 63, "q": 0.05, "min_n": 3},
    "tail_balance": {"window": 252, "q_hi": 0.95, "q_lo": 0.05},
}
DRAWDOWN = {
    "drawdown_windows": [22, 63, 252],
    "low_window": 63,
    "ulcer_windows": [14, 50],
    "max_drawdown_windows": [63, 252],
    "days_since_high_windows": [63, 252],
    "days_since_low_windows": [63],
    "underwater": {"window": 63, "threshold": -0.02},
    "range_position_windows": [20, 63, 252],
}
TREND = {
    "sma_windows": [10, 20, 50, 100, 200],
    "ema_windows": [5, 10, 12, 20, 26, 50, 100, 200],     # must hold every ema_pairs and macd window
    "ema_pairs": [[5, 20], [10, 50], [20, 100], [50, 200]],
    "sma_slope_windows": [20, 50],
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "rsi_windows": [2, 5, 14, 21],
    "stochastic_window": 14,
    "cci": {"window": 20, "min_n": 16},
    "adx_window": 14,
    "aroon_window": 25,
    "bollinger": {"window": 20, "n_sd": 2},
    "efficiency_windows": [21, 63],
    "long_ma": {"window": 200, "share_window": 126, "run_scale": 252},
}


def momentum(x: Context) -> dict:
    """Return the return, skip-momentum, Sharpe, acceleration and path-shape features."""
    p = PRICE
    close, high, ret = x.close, x.high, x.ret
    f = {}
    for k in p["ret_windows"]:
        f[f"ret_{k}"] = close / close.shift(k) - 1.0
    for name, (skip, lookback) in p["skip_momentum"].items():
        f[name] = close.shift(skip) / close.shift(lookback) - 1.0
    for w in p["sharpe_windows"]:
        f[f"sharpe_{w}"] = ops.safe_div(ops.ts_mean(ret, w), ops.ts_std(ret, w))
    skip = p["trend_skip"]["skip"]
    for w in p["trend_skip"]["windows"]:
        f[f"trend_skip{skip}_{w}"] = f[f"ret_{w}"] - f[f"ret_{skip}"]
    for s, l in p["accel_pairs"]:
        f[f"accel_{s}_{l}"] = f[f"ret_{s}"] - f[f"ret_{l}"] * s / l
    pb = p["pullback"]
    f["momentum_pullback"] = f[f"ret_{pb['long']}"] * (1.0 - f[f"ret_{pb['short']}"].abs())
    da = p["drawdown_adjusted"]
    f["drawdown_adjusted_momentum"] = f[f"ret_{da['ret_window']}"] * (1.0 + (close / ops.ts_max(high, da["high_window"]) - 1.0))
    windows = p["positive_share_windows"]
    share = (f[f"ret_{windows[0]}"] > 0).astype(float)
    for w in windows[1:]:
        share = share + (f[f"ret_{w}"] > 0)
    f["multi_horizon_positive_share"] = (share / float(len(windows))).where(f[f"ret_{windows[-1]}"].notna())
    cr = p["crash"]
    crash_ret = f[f"ret_{cr['window']}"]
    f["post_crash_bounce"] = (crash_ret <= cr["threshold"]).astype(float).where(crash_ret.notna()) * f[f"ret_{cr['bounce']}"]
    log_close = log_safe(close)
    for w in p["slope_windows"]:
        f[f"slope_log_price_{w}"] = ops.ts_slope(log_close, w)
    sa = p["slope_accel"]
    f[f"slope_accel_{sa['base']}"] = ops.ts_slope(f[f"slope_log_price_{sa['base']}"], sa["window"])
    for w in p["up_day_share_windows"]:
        f[f"up_day_share_{w}"] = ops.ts_mean((ret > 0).astype(float).where(ret.notna()), w)
    w = p["upside_capture_window"]
    f[f"upside_capture_{w}"] = ops.safe_div(ops.ts_sum(ret.clip(lower=0), w), ops.ts_sum(ret.abs(), w))
    f["up_streak"] = ops.ts_run_length(ret > 0).where(ret.notna())
    f["down_streak"] = ops.ts_run_length(ret < 0).where(ret.notna())
    return f


def volatility(x: Context) -> dict:
    """Return volatility estimators, volatility ratios, tail measures and the volatility percentile against the
    ETF's own past."""
    p = VOLATILITY
    open_, high, low, close, ret = x.open, x.high, x.low, x.close, x.ret
    f = {}
    for w in p["vol_windows"]:
        f[f"vol_{w}"] = ops.ts_std(ret, w)

    def vol(w: int) -> pd.DataFrame:
        """Return the volatility over w sessions (from f when the family already built it)."""
        return f[f"vol_{w}"] if f"vol_{w}" in f else ops.ts_std(ret, w)

    for w in p["downside_vol_windows"]:
        f[f"downside_vol_{w}"] = ops.ts_std(ret.clip(upper=0), w)
    w = p["up_down_vol_window"]
    up = np.sqrt(ops.ts_mean(ret.clip(lower=0) ** 2, w))
    dn = np.sqrt(ops.ts_mean(ret.clip(upper=0) ** 2, w))
    f[f"up_down_vol_ratio_{w}"] = ops.safe_div(up, dn)
    for a, b in p["vol_ratio_pairs"]:
        f[f"vol_ratio_{a}_{b}"] = ops.safe_div(vol(a), vol(b))
    vc = p["vol_change"]
    f[f"vol_change_{vc['lag']}"] = f[f"vol_{vc['window']}"] - f[f"vol_{vc['window']}"].shift(vc["lag"])
    vv = p["vol_of_vol"]
    f[f"vol_of_vol_{vv['outer']}"] = ops.ts_std(ops.ts_std(ret, vv["inner"]), vv["outer"])
    hl = log_safe(high / low)
    co = log_safe(close / open_)
    for w in p["parkinson_windows"]:
        f[f"parkinson_{w}"] = np.sqrt(ops.ts_mean(hl ** 2, w) / (4.0 * np.log(2.0)))
    w = p["garman_klass_window"]
    f[f"garman_klass_{w}"] = np.sqrt(ops.ts_mean(0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2, w).clip(lower=0))
    rs = log_safe(high / close) * log_safe(high / open_) + log_safe(low / close) * log_safe(low / open_)
    w = p["rogers_satchell_window"]
    f[f"rogers_satchell_{w}"] = np.sqrt(ops.ts_mean(rs, w).clip(lower=0))
    oc = log_safe(open_ / close.shift(1))
    for w in p["yang_zhang_windows"]:
        k = YANG_ZHANG_ALPHA / (1.0 + YANG_ZHANG_ALPHA + (w + 1) / (w - 1))
        f[f"yang_zhang_{w}"] = np.sqrt((ops.ts_std(oc, w) ** 2 + k * ops.ts_std(co, w) ** 2
                                        + (1 - k) * ops.ts_mean(rs, w)).clip(lower=0))
    tr = true_range(high, low, close)
    for w in p["atr_windows"]:
        f[f"atr_{w}"] = ops.safe_div(ops.ts_mean(tr, w), close)
    a, b = p["atr_ratio_pair"]
    f[f"atr_ratio_{a}_{b}"] = ops.safe_div(f[f"atr_{a}"], f[f"atr_{b}"])
    vp = p["vol_pct_self"]
    for w in vp["windows"]:
        f[f"vol_pct_self_{w}"] = ops.ts_rank_past(f[f"vol_{w}"], vp["lookback"], vp["min_n"])
    for w in p["moment_windows"]:
        f[f"skew_{w}"] = ops.ts_skew(ret, w)
        f[f"kurt_{w}"] = ops.ts_kurt(ret, w)
    w = p["worst_day_window"]
    f[f"worst_day_{w}"] = ops.ts_min(ret, w)
    es = p["expected_shortfall"]
    q_lo = ops.ts_quantile(ret, es["window"], es["q"])
    f[f"expected_shortfall_{es['window']}"] = ret.where(ret <= q_lo).rolling(es["window"], min_periods=es["min_n"]).mean()
    tb = p["tail_balance"]
    f[f"tail_balance_{tb['window']}"] = ops.safe_div(ops.ts_quantile(ret, tb["window"], tb["q_hi"]).abs(),
                                                     ops.ts_quantile(ret, tb["window"], tb["q_lo"]).abs())
    return f


def drawdown(x: Context) -> dict:
    """Return drawdown, underwater path, Ulcer index, sessions-since-extreme and range-position features."""
    p = DRAWDOWN
    close, low, high = x.close, x.low, x.high
    f = {}
    for w in p["drawdown_windows"]:
        f[f"drawdown_{w}"] = close / ops.ts_max(close, w) - 1.0
    w = p["low_window"]
    f[f"distance_from_low_{w}"] = close / ops.ts_min(low, w) - 1.0
    for w in p["ulcer_windows"]:
        underwater_pct = (close / ops.ts_max(close, w) - 1.0) * PERCENT
        f[f"ulcer_{w}"] = np.sqrt(ops.ts_mean(underwater_pct ** 2, w))
    for w in p["max_drawdown_windows"]:
        f[f"max_drawdown_{w}"] = ops.ts_min(close / ops.ts_max(close, w) - 1.0, w)
    for w in p["days_since_high_windows"]:
        f[f"days_since_high_{w}"] = ops.ts_days_since_max(close, w)
    for w in p["days_since_low_windows"]:
        f[f"days_since_low_{w}"] = ops.ts_days_since_min(close, w)
    uw = p["underwater"]
    w = uw["window"]
    underwater = close / ops.ts_max(close, w) - 1.0
    f[f"underwater_share_{w}"] = ops.ts_mean((underwater < uw["threshold"]).astype(float).where(underwater.notna()), w)
    f[f"underwater_area_{w}"] = ops.ts_mean(underwater, w)
    for w in p["range_position_windows"]:
        lo, hi = ops.ts_min(low, w), ops.ts_max(high, w)
        f[f"range_position_{w}"] = ops.safe_div(close - lo, hi - lo)
    return f


def trend(x: Context) -> dict:
    """Return moving-average gaps and slopes, oscillators, Bollinger, efficiency and long-trend features."""
    p = TREND
    close, high, low, ret = x.close, x.high, x.low, x.ret
    f = {}
    sma = {w: ops.ts_mean(close, w) for w in p["sma_windows"]}
    for w, s in sma.items():
        f[f"ma_gap_{w}"] = close / s - 1.0
    ema = {w: ops.ts_ema(close, w) for w in p["ema_windows"]}
    for a, b in p["ema_pairs"]:
        f[f"ema_ratio_{a}_{b}"] = ema[a] / ema[b] - 1.0
    for w in p["sma_slope_windows"]:
        f[f"sma_slope_{w}"] = ops.safe_div(sma[w] - sma[w].shift(w), close * w)
    fast, slow, signal = p["macd"]["fast"], p["macd"]["slow"], p["macd"]["signal"]
    f[f"ppo_{fast}_{slow}"] = ema[fast] / ema[slow] - 1.0
    macd = ema[fast] - ema[slow]
    f["macd_hist"] = ops.safe_div(macd - ops.ts_ema(macd, signal), close)
    gain, loss = ret.clip(lower=0), (-ret).clip(lower=0)
    for w in p["rsi_windows"]:
        f[f"rsi_{w}"] = PERCENT * ops.safe_div(ops.ts_mean(gain, w), ops.ts_mean(gain, w) + ops.ts_mean(loss, w))
    w = p["stochastic_window"]
    lo, hi = ops.ts_min(low, w), ops.ts_max(high, w)
    f[f"stochastic_{w}"] = ops.safe_div(close - lo, hi - lo)
    cci = p["cci"]
    w = cci["window"]
    typical = (high + low + close) / 3.0
    mad = (typical - ops.ts_mean(typical, w)).abs().rolling(w, min_periods=cci["min_n"]).mean()
    f[f"cci_{w}"] = ops.safe_div(typical - ops.ts_mean(typical, w), CCI_CONSTANT * mad)
    up_move, dn_move = high - high.shift(1), low.shift(1) - low
    plus_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)
    both = up_move.notna() & dn_move.notna()
    plus_dm, minus_dm = plus_dm.where(both), minus_dm.where(both)
    w = p["adx_window"]
    atr = ops.ts_mean(true_range(high, low, close), w)
    di_p, di_m = ops.safe_div(ops.ts_mean(plus_dm, w), atr), ops.safe_div(ops.ts_mean(minus_dm, w), atr)
    f[f"adx_{w}"] = ops.ts_mean(ops.safe_div((di_p - di_m).abs(), di_p + di_m), w)
    w = p["aroon_window"]
    f[f"aroon_osc_{w}"] = ops.ts_days_since_min(low, w) - ops.ts_days_since_max(high, w)
    bb = p["bollinger"]
    w, n_sd = bb["window"], bb["n_sd"]
    mean, sd = ops.ts_mean(close, w), ops.ts_std(close, w)
    f[f"bollinger_pctb_{w}"] = ops.safe_div(close - (mean - n_sd * sd), 2 * n_sd * sd)
    f[f"bollinger_width_{w}"] = ops.safe_div(2 * n_sd * sd, mean)
    for w in p["efficiency_windows"]:
        f[f"efficiency_{w}"] = ops.safe_div((close - close.shift(w)).abs(), ops.ts_sum((close - close.shift(1)).abs(), w))
    lm = p["long_ma"]
    w = lm["window"]
    ma = sma[w] if w in sma else ops.ts_mean(close, w)
    above = (close >= ma) & ma.notna()
    f[f"time_above_ma{w}_{lm['share_window']}"] = ops.ts_mean(above.astype(float).where(ma.notna()), lm["share_window"])
    f[f"run_above_ma{w}"] = ops.ts_run_length(above).where(ma.notna()) / float(lm["run_scale"])
    return f
