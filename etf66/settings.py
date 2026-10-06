"""Paths, constants and the hash helpers.

The environment variables ETF66_DATA_DIR, ETF66_RUNS_DIR and ETF66_MARKET_ROOT override the default paths.
The module reads no data when Python imports it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
FENCE_DATE = dt.date(2021, 1, 1)                # rules.py checks config.yaml against it
UNLOCK_FILE = REPO_ROOT / "HOLDOUT_UNLOCK.yaml"

DATA_DIR = Path(os.environ.get("ETF66_DATA_DIR", REPO_ROOT / "data"))
DEV_DIR = DATA_DIR / "dev"                      # data before the fence date
HOLDOUT_DIR = DATA_DIR / "holdout"              # data on or after the fence date (sealed)
RUNS_DIR = Path(os.environ.get("ETF66_RUNS_DIR", REPO_ROOT / "runs"))

# The owner's market folder. Only the snapshot command reads these paths.
MARKET_ROOT = Path(os.environ.get("ETF66_MARKET_ROOT", "D:/Data/market"))
PANELS_LONG = MARKET_ROOT / "panels_long"
CBOE_CACHE = MARKET_ROOT / "mission_data/dev/derived/state_bins/pca566/lab/round5_final/cboe_cache"
STOCK_UNIVERSE = MARKET_ROOT / "DailyMarketViews/cache/universe_compat/annual_core_pass.parquet"

UNIVERSE_FILE = CONFIG_DIR / "universe66.csv"
UNIVERSE_OVERRIDES_FILE = CONFIG_DIR / "universe66_overrides.csv"

SESSIONS_PER_YEAR = 252                         # annualizes returns, volatility, Sharpe and turnover
CASH_TICKER = "SHY"                             # unused weight of a long-only book sits here
CASH_LIKE = ("SGOV", "SHY")                     # out of groups, basket and book: gross above 1 on cash fakes a Sharpe
MIN_HISTORY_SESSIONS = 252                      # an ETF is tradable from its 252nd close on (one year of history)
STOCK_STATE_SIZE = 500                          # the largest stocks per year in the stock PCA state

HASH_CHARS = 12                                 # length of the short hashes in file names and reports
HASH_CHUNK_BYTES = 1 << 20


def text_hash(text: str) -> str:
    """Return the first HASH_CHARS hex characters of the SHA-256 of a text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_CHARS]


def file_hash(path: Path) -> str:
    """Return the SHA-256 of a file (full hex)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash(*names: str) -> str:
    """Return a hash of the named .py files and of every .py file under the named folders (CRLF read as LF)."""
    h = hashlib.sha256()
    for name in names:
        root = REPO_ROOT / name
        for p in sorted(root.rglob("*.py")) if root.is_dir() else [root]:
            h.update(p.relative_to(REPO_ROOT).as_posix().encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:HASH_CHARS]


def code_hash() -> str:
    """Return a hash of the etf66 package, so a run records its code."""
    return source_hash("etf66")
