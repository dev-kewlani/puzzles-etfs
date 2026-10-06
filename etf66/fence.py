"""The holdout fence. Dev code never reads a date on or after FENCE_DATE.

Rules:
- check_dev_frame refuses any frame or index that holds a date on or after the fence date.
- The holdout loader runs only when HOLDOUT_UNLOCK.yaml exists, has its 5 fields, and names a run with trials.
  The key config_ids holds trial_id values of that run. The code does not check a signature.
- A holdout result folder is write-once: a second holdout run of the same run id is refused.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import yaml

from etf66 import settings


class FenceError(RuntimeError):
    """A dev step met a date on or after the fence date, or the holdout is not unlocked."""


def fence_timestamp() -> pd.Timestamp:
    """Return the fence date as a pandas Timestamp."""
    return pd.Timestamp(settings.FENCE_DATE)


def check_dev_index(index: pd.Index, source_label: str) -> None:
    """Raise FenceError if a date index holds a missing date or a date on or after the fence date."""
    if len(index) == 0:
        return
    if pd.isna(index).any():
        raise FenceError(f"{source_label}: holds a missing date")
    if pd.Timestamp(index.max()) >= fence_timestamp():
        raise FenceError(f"{source_label}: holds {pd.Timestamp(index.max()).date()}, on or after the fence {settings.FENCE_DATE}")


def check_dev_frame(frame: pd.DataFrame, source_label: str, date_column: str | None = None) -> None:
    """Raise FenceError if a frame holds a date on or after the fence date (in its index or a date column)."""
    if date_column is not None:
        check_dev_index(pd.Index(pd.to_datetime(frame[date_column])), source_label)
    else:
        check_dev_index(frame.index, source_label)


def read_unlock() -> dict:
    """Return the unlock file as a dict.

    Raise FenceError if the file is missing, lacks a field or names a run without trials.parquet.
    """
    if not settings.UNLOCK_FILE.exists():
        raise FenceError("the holdout is sealed: write HOLDOUT_UNLOCK.yaml first (see docs/DESIGN.md, section 3)")
    with open(settings.UNLOCK_FILE, encoding="utf-8") as fh:
        unlock = yaml.safe_load(fh) or {}
    for key in ("owner", "date", "run_id", "config_ids", "statement"):
        if not unlock.get(key):
            raise FenceError(f"HOLDOUT_UNLOCK.yaml lacks the field '{key}'")
    unlock["date"] = dt.date.fromisoformat(str(unlock["date"]))
    if not (settings.RUNS_DIR / str(unlock["run_id"]) / "trials.parquet").exists():
        raise FenceError(f"HOLDOUT_UNLOCK.yaml names the run {unlock['run_id']}, which has no trials.parquet")
    return unlock
