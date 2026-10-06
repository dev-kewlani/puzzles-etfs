"""Synthetic data for the tests. No test reads the real snapshot."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf66 import config
from etf66.features.context import Context

NAMED = ["SPY", "TLT", "HYG", "GLD", "UUP", "VXX", "IEF", "SHY"]


@pytest.fixture(scope="session")
def cfg():
    """Return the default settings of config.yaml."""
    return config.load_config()


def synthetic_bars(n_sessions: int = 600, n_other: int = 22, seed: int = 7) -> dict:
    """Return random-walk bars: open, high, low, close, volume and adv tables (sessions x tickers).

    The tickers are NAMED, which the features name, plus n_other filler ETFs.
    """
    rng = np.random.default_rng(seed)
    tickers = NAMED + [f"E{i:02d}" for i in range(n_other)]
    dates = pd.bdate_range("2005-01-03", periods=n_sessions)
    ret = rng.normal(0.0003, 0.012, (n_sessions, len(tickers)))
    close = 100 * np.exp(np.cumsum(ret, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, close.shape)))
    volume = rng.lognormal(13, 0.4, close.shape)
    frame = lambda a: pd.DataFrame(a, index=dates, columns=tickers)  # noqa: E731
    out = {"open": frame(open_), "high": frame(high), "low": frame(low), "close": frame(close), "volume": frame(volume)}
    out["adv"] = (out["close"] * out["volume"]).rolling(30, min_periods=24).median()
    return out


def context_from_bars(b: dict) -> Context:
    """Return a feature Context from synthetic bars. The external series are empty."""
    close = b["close"]
    # The tradable rule without the source flag: a close at t and 252 or more closes up to t.
    tradable = close.notna() & (close.notna().cumsum() >= 252)
    groups = pd.Series(["equity" if i % 3 else "rates" for i in range(close.shape[1])], index=close.columns)
    empty = pd.DataFrame(index=close.index)
    return Context(open=b["open"], high=b["high"], low=b["low"], close=close, volume=b["volume"], adv=b["adv"],
                   ret=close / close.shift(1) - 1.0, tradable=tradable, asset_groups=groups, market=empty, cboe=empty,
                   treasury=empty, stock_returns=empty, stock_members=pd.DataFrame())


@pytest.fixture(scope="session")
def bars():
    """Return the synthetic bars that the tests share."""
    return synthetic_bars()
