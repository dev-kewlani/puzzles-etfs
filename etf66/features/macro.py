"""Global features (one value per session): macro, volatility indices, correlation indices, cross-asset, calendar.

Time rules:
- The context lags the market and CBOE series by one session.
- Treasury yields count from their publication date.
- ETF spreads use the ETF closes of session t. The decision is at the same close, so this is not lookahead.
- Calendar features use the weekday calendar (no exchange holidays) and the session index. Both are known in advance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf66.features.context import MARKET_TICKER, Context

EPS = 1e-12                   # floor of a standard deviation in a division
MONTHS_PER_YEAR = 12.0

MACRO = {
    "trailing_z_window": 252,
    "trailing_z_min_frac": 0.8,           # min_periods = int(frac x window); this rounds down, ops.min_obs rounds up
    "tenors": {"long": "10_year", "short": "3_month", "mid": "2_year"},
    "rate_change": 21,
    "curve_change": 63,
    "vix": {"instrument": "vix", "nasdaq_instrument": "vxn_nasdaq_vol", "change": 5},
    "cboe_change": 5,
    "spread_windows": [5, 21],            # must hold regime.window: the regime reads that credit spread
    "credit": ["HYG", "LQD"],
    "rates": ["IEF", "SHY"],
    "dollar": {"ticker": "UUP", "window": 63},
    "copper_gold": {"copper": "copper", "gold": "gold", "window": 63},
    "oil": {"instrument": "crude_oil_wti", "window": 21},
    "spy": {"ma": 200, "ma_min_n": 150, "ret": 21, "vol": 21, "vol_min_n": 17},
    "regime": {"stock_bond": ["SPY", "TLT"], "window": 21, "min_legs": 3},
}
TURN_DAYS_TO_END = 2          # turn of month: at most this many weekdays to the month end,
TURN_DAY_IN_MONTH = 3         # or at most this session of the month


def col(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return one column, or an all-NaN series when the column does not exist."""
    return frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)


def trailing_z(s: pd.Series, window: int, min_frac: float) -> pd.Series:
    """Return the trailing z-score of a series over `window` sessions (min_periods = int(min_frac x window))."""
    n = int(min_frac * window)
    m, sd = s.rolling(window, min_periods=n).mean(), s.rolling(window, min_periods=n).std()
    return (s - m) / sd.where(sd > EPS)


def etf_return(x: Context, ticker: str, k: int) -> pd.Series:
    """Return the k-session return of one ETF (NaN when the ETF has no data)."""
    if ticker not in x.close.columns:
        return pd.Series(np.nan, index=x.close.index)
    c = x.close[ticker]
    return c / c.shift(k) - 1.0


def tenor_label(tenor: str) -> str:
    """Return the short label of a Treasury tenor, for example 10_year -> 10y and 3_month -> 3m."""
    return tenor.replace("_year", "y").replace("_month", "m")


def macro(x: Context) -> dict:
    """Return rates, volatility, skew, correlation-index, credit, dollar, commodity and regime features."""
    p = MACRO
    treasury, market, cboe = x.treasury, x.market, x.cboe
    zw = p["trailing_z_window"]

    def z(s: pd.Series) -> pd.Series:
        """Return the trailing z-score with the macro settings."""
        return trailing_z(s, zw, p["trailing_z_min_frac"])

    f = {}
    ten = p["tenors"]
    long_, short, mid = tenor_label(ten["long"]), tenor_label(ten["short"]), tenor_label(ten["mid"])
    y_long, y_short, y_mid = col(treasury, ten["long"]), col(treasury, ten["short"]), col(treasury, ten["mid"])
    f[f"rate_{long_}_level"] = y_long
    f[f"curve_{long_}_{short}"] = y_long - y_short
    f[f"curve_{long_}_{mid}"] = y_long - y_mid
    f[f"rate_{long_}_change_{p['rate_change']}"] = y_long - y_long.shift(p["rate_change"])
    f[f"curve_{long_}_{short}_change_{p['curve_change']}"] = f[f"curve_{long_}_{short}"] - f[f"curve_{long_}_{short}"].shift(p["curve_change"])
    f[f"rate_{short}_level"] = y_short
    vx = p["vix"]
    vix = col(market, vx["instrument"])
    f["vix_level"] = vix
    f[f"vix_z_{zw}"] = z(vix)
    f[f"vix_change_{vx['change']}"] = vix - vix.shift(vx["change"])
    f["vxn_over_vix"] = col(market, vx["nasdaq_instrument"]) / vix
    f["vix_over_vix3m"] = col(cboe, "VIX") / col(cboe, "VIX3M")
    f["vix9d_over_vix"] = col(cboe, "VIX9D") / col(cboe, "VIX")
    f[f"vvix_z_{zw}"] = z(col(cboe, "VVIX"))
    f[f"skew_z_{zw}"] = z(col(cboe, "SKEW"))
    cor1, cor3 = col(cboe, "COR1M"), col(cboe, "COR3M")
    f["cor1m_level"], f["cor3m_level"] = cor1, cor3
    f["cor1m_minus_cor3m"] = cor1 - cor3
    f[f"cor3m_change_{p['cboe_change']}"] = cor3 - cor3.shift(p["cboe_change"])
    (hi_yield, ig), (mid_bond, short_bond) = p["credit"], p["rates"]
    for k in p["spread_windows"]:
        f[f"credit_{hi_yield.lower()}_minus_{ig.lower()}_{k}"] = etf_return(x, hi_yield, k) - etf_return(x, ig, k)
        f[f"rates_{mid_bond.lower()}_minus_{short_bond.lower()}_{k}"] = etf_return(x, mid_bond, k) - etf_return(x, short_bond, k)
    dl = p["dollar"]
    dollar_name = f"dollar_{dl['ticker'].lower()}_{dl['window']}"
    f[dollar_name] = etf_return(x, dl["ticker"], dl["window"])
    cg = p["copper_gold"]
    copper, gold = col(market, cg["copper"]), col(market, cg["gold"])
    cg_name = f"copper_over_gold_{cg['window']}"
    log_ratio = np.log(copper / gold)
    f[cg_name] = log_ratio - log_ratio.shift(cg["window"])
    oil = p["oil"]
    f[f"oil_wti_{oil['window']}"] = col(market, oil["instrument"]) / col(market, oil["instrument"]).shift(oil["window"]) - 1.0
    sp = p["spy"]
    tag = MARKET_TICKER.lower()
    mkt = x.close[MARKET_TICKER]
    ma = mkt.rolling(sp["ma"], min_periods=sp["ma_min_n"]).mean()
    f[f"{tag}_above_ma{sp['ma']}"] = (mkt >= ma).astype(float).where(mkt.notna() & ma.notna())
    f[f"{tag}_return_{sp['ret']}"] = mkt / mkt.shift(sp["ret"]) - 1.0
    f[f"{tag}_vol_{sp['vol']}"] = (mkt / mkt.shift(1) - 1.0).rolling(sp["vol"], min_periods=sp["vol_min_n"]).std()
    rg = p["regime"]
    stock, bond = rg["stock_bond"]
    legs = pd.concat([
        z(etf_return(x, stock, rg["window"]) - etf_return(x, bond, rg["window"])),
        z(f[f"credit_{hi_yield.lower()}_minus_{ig.lower()}_{rg['window']}"]),
        z(f[cg_name]),
        -z(f[dollar_name]),
        -z(vix),
    ], axis=1)
    f["regime_composite"] = legs.mean(axis=1).where(legs.notna().sum(axis=1) >= rg["min_legs"])
    return f


def calendar(x: Context) -> dict:
    """Return day-of-week, month, weekdays-to-month-end and turn-of-month features."""
    idx = x.close.index
    weekdays = pd.bdate_range(idx.min(), idx.max() + pd.offsets.MonthEnd(2))
    month_end = idx + pd.offsets.BMonthEnd(0)
    to_end = np.array([((weekdays > d) & (weekdays <= e)).sum() for d, e in zip(idx, month_end)], dtype=float)
    day_in_month = idx.to_series().groupby([idx.year, idx.month]).cumcount().to_numpy() + 1.0
    turn = (to_end <= TURN_DAYS_TO_END) | (day_in_month <= TURN_DAY_IN_MONTH)
    return {
        "day_of_week": pd.Series(idx.dayofweek.astype(float), index=idx),
        "month_sin": pd.Series(np.sin(2 * np.pi * idx.month / MONTHS_PER_YEAR), index=idx),
        "month_cos": pd.Series(np.cos(2 * np.pi * idx.month / MONTHS_PER_YEAR), index=idx),
        "business_days_to_month_end": pd.Series(to_end, index=idx),
        "turn_of_month": pd.Series(turn.astype(float), index=idx),
    }
