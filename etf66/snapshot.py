"""Build the data snapshot. This is the only module that reads the owner's market folder.

Run once:  python -m etf66 snapshot

What the snapshot holds (each file split into data/dev = dates before the fence and data/holdout = the rest):
  etf_bars.parquet       date, ticker, open, high, low, close, volume, adv, non_tradeable for the universe ETFs
                         (adjusted OHLC, raw volume, the 30-session median dollar volume of the source, the
                         non-tradeable flag of the source)
  market.parquet         date, instrument, value: the cross-asset series of panels_long/market (VIX, indices...)
  treasury.parquet       date, available_at_date, tenor, value: Treasury yields with their publication date
  cboe.parquet           date, series, value: the CBOE_SERIES indices from the CBOE file cache
  stock_members.parquet  effective_year, instrument_uid, size_rank: the largest stocks of each year (stock PCA state)
  stock_returns.parquet  date, instrument_uid, ret: daily returns of those stocks
  MANIFEST.json          SHA-256, rows and date range of every file
The snapshot sets the holdout files to read-only.
A missing CBOE file or stock universe file gives fewer rows, not an error (see docs/DESIGN.md, open items).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import stat

import duckdb
import pandas as pd

from etf66 import settings

REFRESH_WINDOW = ("17:15", "18:15")     # the owner's daily job rewrites panels_long in this window
CBOE_SERIES = ["SKEW", "VVIX", "COR1M", "COR3M", "VIX9D", "VIX3M", "VIX"]
TREASURY_TENORS = ["1_month", "3_month", "1_year", "2_year", "5_year", "10_year", "30_year"]
STOCK_RANK_FIELD = "mktcap_63d_median"

def q(path) -> str:
    """Return a path as a DuckDB string literal."""
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def refuse_refresh_window(window) -> None:
    """Raise RuntimeError inside REFRESH_WINDOW, when the owner's daily job rewrites panels_long."""
    start, end = (dt.time.fromisoformat(t) for t in window)
    if start <= dt.datetime.now().time() <= end:
        raise RuntimeError(f"panels_long is rewritten between {window[0]} and {window[1]}; run the snapshot later")


def resolve_tickers(con) -> pd.DataFrame:
    """Return the universe ETFs with their asset group, role and instrument_uid.

    The instrument_uid of a ticker is the ETF instrument with the latest valid_to.
    """
    members = pd.read_csv(settings.UNIVERSE_FILE, dtype=str)
    overrides = pd.read_csv(settings.UNIVERSE_OVERRIDES_FILE, dtype=str, keep_default_na=False)
    con.register("_members", members)
    picked = con.execute(
        f"""
        SELECT u.ticker, arg_max(m.instrument_uid, m.valid_to) AS instrument_uid
        FROM _members u
        JOIN read_parquet({q(settings.PANELS_LONG / '_meta' / 'ticker_instrument_map.parquet')}) m
          ON m.ticker = u.ticker AND m.asset = 'etf'
        GROUP BY u.ticker
        """
    ).df()
    out = members.merge(picked, on="ticker", how="left").merge(overrides, on="ticker", how="left")
    out["first_usable_date"] = pd.to_datetime(out["first_usable_date"].replace("", None))
    missing = out.loc[out["instrument_uid"].isna(), "ticker"].tolist()
    if missing:
        raise RuntimeError(f"no ETF instrument found for {missing}")
    return out


def read_etf_bars(con, tickers: pd.DataFrame) -> pd.DataFrame:
    """Return the daily bars of the universe ETFs as one long frame keyed on (date, ticker)."""
    con.register("_tickers", tickers[["ticker", "instrument_uid"]])

    def p(field: str) -> str:
        """Return the DuckDB path literal of one ETF field file."""
        return q(settings.PANELS_LONG / "etf" / f"{field}.parquet")

    bars = con.execute(
        f"""
        SELECT c.date, t.ticker, o.value AS open, h.value AS high, l.value AS low, c.value AS close,
               v.value AS volume, a.value AS adv, coalesce(nt.value, false) AS non_tradeable
        FROM read_parquet({p('c')}) c
        JOIN _tickers t USING (instrument_uid)
        LEFT JOIN read_parquet({p('o')}) o USING (date, instrument_uid)
        LEFT JOIN read_parquet({p('h')}) h USING (date, instrument_uid)
        LEFT JOIN read_parquet({p('l')}) l USING (date, instrument_uid)
        LEFT JOIN read_parquet({p('v')}) v USING (date, instrument_uid)
        LEFT JOIN read_parquet({p('adv')}) a USING (date, instrument_uid)
        LEFT JOIN read_parquet({p('non_tradeable')}) nt USING (date, instrument_uid)
        ORDER BY c.date, t.ticker
        """
    ).df()
    bars["date"] = pd.to_datetime(bars["date"])
    first = tickers.set_index("ticker")["first_usable_date"]
    cut = bars["ticker"].map(first)
    return bars[cut.isna() | (bars["date"] >= cut)].reset_index(drop=True)


def read_market(con) -> pd.DataFrame:
    """Return the cross-asset market series (long frame: date, instrument, value)."""
    df = con.execute(f"SELECT date, instrument, value FROM read_parquet({q(settings.PANELS_LONG / 'market' / 'c.parquet')})").df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def read_treasury(con, tenors: list) -> pd.DataFrame:
    """Return the Treasury yields with their publication date (long frame)."""
    parts = []
    for tenor in tenors:
        f = settings.PANELS_LONG / "treasury" / f"yield_{tenor}.parquet"
        df = con.execute(f"SELECT date, available_at_date, value FROM read_parquet({q(f)})").df()
        df["tenor"] = tenor
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["available_at_date"] = pd.to_datetime(out["available_at_date"])
    return out


def read_cboe(series: list) -> pd.DataFrame:
    """Return the CBOE index closes from the owner's file cache (long frame: date, series, value)."""
    parts = []
    for name in series:
        f = settings.CBOE_CACHE / f"{name}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        col = "CLOSE" if "CLOSE" in df.columns else name
        parts.append(pd.DataFrame({"date": pd.to_datetime(df["DATE"], format="%m/%d/%Y"), "series": name,
                                   "value": pd.to_numeric(df[col], errors="coerce")}))
    return pd.concat(parts, ignore_index=True).dropna(subset=["value"])


def read_stocks(con, size: int, rank_field: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the stock-state members (largest `size` stocks of each effective year) and their daily returns.

    Membership for effective year Y uses the market cap at the last session of Y - 1 (point in time).
    """
    if not settings.STOCK_UNIVERSE.exists():
        return pd.DataFrame(columns=["effective_year", "instrument_uid", "size_rank"]), \
               pd.DataFrame(columns=["date", "instrument_uid", "ret"])
    members = con.execute(
        f"""
        SELECT effective_year, instrument_uid, size_rank FROM (
            SELECT effective_year, instrument_uid,
                   row_number() OVER (PARTITION BY effective_year ORDER BY {rank_field} DESC) AS size_rank
            FROM read_parquet({q(settings.STOCK_UNIVERSE)})
            WHERE value AND {rank_field} IS NOT NULL
        ) WHERE size_rank <= {int(size)}
        """
    ).df()
    con.register("_stock_members", members)
    rets = con.execute(
        f"""
        SELECT r.date, r.instrument_uid, r.value AS ret
        FROM read_parquet({q(settings.PANELS_LONG / 'equity' / 'ret.parquet')}) r
        JOIN _stock_members m ON m.instrument_uid = r.instrument_uid AND m.effective_year = year(r.date)
        """
    ).df()
    rets["date"] = pd.to_datetime(rets["date"])
    return members, rets


def split_and_write(name: str, frame: pd.DataFrame, dev_mask: pd.Series, manifest: dict) -> None:
    """Write the dev rows and the holdout rows of one table, and record both in the manifest."""
    for part, mask, folder in (("dev", dev_mask, settings.DEV_DIR), ("holdout", ~dev_mask, settings.HOLDOUT_DIR)):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.parquet"
        if path.exists():                               # a holdout file of an earlier snapshot is read-only
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        sub = frame[mask.values].reset_index(drop=True)
        sub.to_parquet(path, index=False)
        date_col = "date" if "date" in sub.columns else None
        manifest[f"{part}/{name}.parquet"] = {
            "sha256": settings.file_hash(path), "rows": int(len(sub)),
            "first": str(sub[date_col].min().date()) if date_col and len(sub) else None,
            "last": str(sub[date_col].max().date()) if date_col and len(sub) else None,
        }
        if part == "holdout":
            os.chmod(path, stat.S_IREAD)


def build_snapshot() -> dict:
    """Build every snapshot file and return the manifest."""
    refuse_refresh_window(REFRESH_WINDOW)
    fence = pd.Timestamp(settings.FENCE_DATE)
    con = duckdb.connect()
    manifest: dict = {"built_at": dt.datetime.now().isoformat(timespec="seconds"), "fence_date": str(settings.FENCE_DATE),
                      "source": str(settings.MARKET_ROOT), "stock_state_size": settings.STOCK_STATE_SIZE,
                      "files": {}}
    tickers = resolve_tickers(con)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tickers.to_csv(settings.DATA_DIR / "tickers.csv", index=False)

    bars = read_etf_bars(con, tickers)
    split_and_write("etf_bars", bars, bars["date"] < fence, manifest["files"])
    market = read_market(con)
    split_and_write("market", market, market["date"] < fence, manifest["files"])
    treasury = read_treasury(con, TREASURY_TENORS)
    # A yield dated before the fence but published on or after it is holdout data.
    split_and_write("treasury", treasury, (treasury["date"] < fence) & (treasury["available_at_date"] < fence),
                    manifest["files"])
    cboe = read_cboe(CBOE_SERIES)
    split_and_write("cboe", cboe, cboe["date"] < fence, manifest["files"])
    members, rets = read_stocks(con, settings.STOCK_STATE_SIZE, STOCK_RANK_FIELD)
    split_and_write("stock_members", members, members["effective_year"] < settings.FENCE_DATE.year, manifest["files"])
    split_and_write("stock_returns", rets, rets["date"] < fence, manifest["files"])
    con.close()

    with open(settings.DATA_DIR / "MANIFEST.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
