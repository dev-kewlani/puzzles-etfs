"""The inputs that every feature function reads, prepared once.

Time rule for external series: a feature at session t reads the market series (VIX, indices, futures) and the CBOE
indices of session t - 1. These series can close after the ETF close, so we lag them (context.market_lag_sessions
and context.cboe_lag_sessions in config.yaml). The code fills CBOE gaps of up to CBOE_FFILL_LIMIT sessions before
the lag. Treasury yields need no lag: the code uses a yield from its publication date.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from etf66.data import Panel

MARKET_TICKER = "SPY"         # the ETF that stands for the market in beta, capture and regime features
CBOE_FFILL_LIMIT = 5          # longest CBOE gap (sessions) that the forward fill closes, before the lag


@dataclass
class Context:
    """Prepared inputs on the session calendar. Tables are sessions x tickers unless the comment says otherwise."""
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame              # raw volume
    adv: pd.DataFrame                 # 30-session median dollar volume, from the snapshot
    ret: pd.DataFrame                 # close-to-close return from the adjusted close
    tradable: pd.DataFrame
    asset_groups: pd.Series           # ticker -> asset group (equity, rates, credit, ...)
    market: pd.DataFrame              # sessions x instrument, lagged
    cboe: pd.DataFrame                # sessions x series, filled, then lagged
    treasury: pd.DataFrame            # sessions x tenor, by publication date
    stock_returns: pd.DataFrame       # sessions x stock (members of the year only)
    stock_members: pd.DataFrame

    def etf(self, ticker: str, field: str = "ret") -> pd.Series:
        """Return one ETF column of a field (for example SPY returns) as a session series."""
        return getattr(self, field)[ticker]


def build_context(panel: Panel, tradable: pd.DataFrame, cfg: dict) -> Context:
    """Return the prepared feature inputs for a panel and the settings of a run."""
    ctx = cfg["context"]
    close = panel.field("close")
    return Context(
        open=panel.field("open"), high=panel.field("high"), low=panel.field("low"), close=close,
        volume=panel.field("volume"), adv=panel.field("adv"),
        ret=close / close.shift(1) - 1.0,
        tradable=tradable, asset_groups=panel.asset_groups,
        market=panel.market.shift(ctx["market_lag_sessions"]),
        cboe=panel.cboe.ffill(limit=CBOE_FFILL_LIMIT).shift(ctx["cboe_lag_sessions"]),
        treasury=panel.treasury,
        stock_returns=panel.stock_returns, stock_members=panel.stock_members,
    )


def log_safe(x):
    """Return log(x) with NaN for values at or below zero."""
    return np.log(x.where(x > 0))


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Return the true range: the largest of high - low, |high - previous close| and |low - previous close|."""
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
                   keys=[0, 1, 2]).groupby(level=1).max()
    return tr.reindex(close.index)


def on_balance_volume(ret: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Return the cumulative signed volume (sign of the return times the volume), NaN where the return is missing."""
    return (np.sign(ret) * volume).fillna(0.0).cumsum().where(ret.notna())
