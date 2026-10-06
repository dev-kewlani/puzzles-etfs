"""The tradable set of each session.

An ETF is tradable at session t when three rules hold. It has a close at t. The source does not flag it at t (the
source column is non_tradeable). It has at least MIN_HISTORY_SESSIONS closes up to t. The rules read data up to
t only.
"""
from __future__ import annotations

import pandas as pd

from etf66.data import Panel
from etf66.settings import MIN_HISTORY_SESSIONS


def tradable_mask(panel: Panel) -> pd.DataFrame:
    """Return a bool table (sessions x tickers): True where the ETF is tradable at the close of that session."""
    close = panel.field("close")
    has_close = close.notna()
    history = has_close.cumsum()
    return has_close & ~panel.non_tradeable & (history >= MIN_HISTORY_SESSIONS)
