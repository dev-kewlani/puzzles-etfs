"""Load the snapshot into one Panel of wide tables (sessions x tickers).

Two data modes:
- "dev"  reads data/dev only. It checks the fence on every table with a date column, on the publication date of the
         Treasury table, and on the membership year of the stock members.
- "full" reads data/dev plus data/holdout. Only the holdout command uses it, after fence.read_unlock().
Two rows with the same (date, key) keep the last row.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from etf66 import fence, settings

PRICE_FIELDS = ("open", "high", "low", "close", "volume", "adv")


@dataclass
class Panel:
    """All inputs on one session calendar. Every wide table has the index `dates`."""
    dates: pd.DatetimeIndex
    tickers: list
    asset_groups: pd.Series               # ticker -> asset group, the block_name column of the universe file
    bars: dict                            # field -> DataFrame (sessions x tickers): open, high, low, close, volume, adv
    non_tradeable: pd.DataFrame           # bool (sessions x tickers)
    market: pd.DataFrame                  # sessions x instrument, same-session value; features/context.py lags it
    treasury: pd.DataFrame                # sessions x tenor, the last value published on or before each session
    cboe: pd.DataFrame                    # sessions x series, same-session value; features/context.py lags it
    stock_returns: pd.DataFrame           # sessions x instrument_uid (NaN when the stock is not a member that year)
    stock_members: pd.DataFrame = field(default_factory=pd.DataFrame)   # effective_year, instrument_uid, size_rank

    def field(self, name: str) -> pd.DataFrame:
        """Return one price field as a sessions x tickers table."""
        return self.bars[name]


def read_table(name: str, mode: str) -> pd.DataFrame:
    """Return one snapshot table for the data mode. Dev mode checks the fence on the date column."""
    dev = pd.read_parquet(settings.DEV_DIR / f"{name}.parquet")
    if mode == "dev":
        if "date" in dev.columns:
            fence.check_dev_frame(dev, name, date_column="date")
        return dev
    if mode != "full":
        raise ValueError(f"unknown mode {mode}")
    fence.read_unlock()
    hold = pd.read_parquet(settings.HOLDOUT_DIR / f"{name}.parquet")
    return pd.concat([dev, hold], ignore_index=True)


def to_wide(long: pd.DataFrame, column: str, key: str, dates: pd.DatetimeIndex, columns=None) -> pd.DataFrame:
    """Pivot a long table (date, key, column) to sessions x keys, on the session calendar."""
    if long.empty:
        return pd.DataFrame(index=dates, columns=columns or [], dtype=float)
    wide = long.pivot_table(index="date", columns=key, values=column, aggfunc="last")
    wide = wide.reindex(index=dates)
    return wide.reindex(columns=columns) if columns is not None else wide


def treasury_by_session(treasury: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return Treasury yields per session: the last value whose publication date is on or before the session."""
    out = {}
    for tenor, grp in treasury.groupby("tenor"):
        s = grp.sort_values(["available_at_date", "date"], kind="mergesort").groupby("available_at_date")["value"].last()
        out[tenor] = s.reindex(dates, method="ffill")
    return pd.DataFrame(out, index=dates)


def load_panel(mode: str = "dev") -> Panel:
    """Load the snapshot as a Panel on the ETF session calendar."""
    tickers_meta = pd.read_csv(settings.DATA_DIR / "tickers.csv", dtype=str)
    tickers = tickers_meta["ticker"].tolist()
    asset_groups = tickers_meta.set_index("ticker")["block_name"]

    bars_long = read_table("etf_bars", mode)
    bars_long["date"] = pd.to_datetime(bars_long["date"])
    dates = pd.DatetimeIndex(sorted(bars_long["date"].unique()), name="date")
    bars = {f: to_wide(bars_long, f, "ticker", dates, tickers).astype(float) for f in PRICE_FIELDS}
    non_tradeable = to_wide(bars_long, "non_tradeable", "ticker", dates, tickers).astype("boolean").fillna(False).astype(bool)

    market = read_table("market", mode)
    market["date"] = pd.to_datetime(market["date"])
    treasury = read_table("treasury", mode)
    if mode == "dev":
        fence.check_dev_frame(treasury, "treasury", date_column="available_at_date")
    cboe = read_table("cboe", mode)
    cboe["date"] = pd.to_datetime(cboe["date"])
    stock_ret = read_table("stock_returns", mode)
    stock_ret["date"] = pd.to_datetime(stock_ret["date"])
    members = pd.read_parquet(settings.DEV_DIR / "stock_members.parquet")
    if (members["effective_year"] >= settings.FENCE_DATE.year).any():
        raise fence.FenceError("stock_members (dev) holds a membership year on or after the fence year")
    if mode == "full":
        members = pd.concat([members, pd.read_parquet(settings.HOLDOUT_DIR / "stock_members.parquet")], ignore_index=True)

    return Panel(
        dates=dates, tickers=tickers, asset_groups=asset_groups, bars=bars, non_tradeable=non_tradeable,
        market=to_wide(market, "value", "instrument", dates),
        treasury=treasury_by_session(treasury, dates),
        cboe=to_wide(cboe, "value", "series", dates),
        stock_returns=to_wide(stock_ret, "ret", "instrument_uid", dates).astype("float32"),
        stock_members=members,
    )
