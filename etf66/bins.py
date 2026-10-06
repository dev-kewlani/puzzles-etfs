"""Bins of the daily predictions.

Two bins per (session, ETF), 1 (lowest) to n_bins (highest), NaN when undefined:
  ts_bin  the prediction at t against the same ETF's predictions of the warmup sessions before t. All of those
          sessions must hold a prediction, or ts_bin is NaN. They can come from different monthly fits.
  cs_bin  the prediction at t ranked among the ETFs with a prediction at t. Ties break by column order (ordinal).
          tie_rule="average" is the other option; the choice is still open.
The backtest builds both bins in memory for each prediction series and does not save them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf66.ops import rank_vs_past

N_BINS = 10
WARMUP_PREDICTIONS = 63       # about one quarter; ts_bin ranks against this many earlier predictions of the ETF


def to_bins(pct: np.ndarray, n_bins: int) -> np.ndarray:
    """Return bins 1..n_bins from percentiles in [0, 1] (NaN stays NaN)."""
    b = np.floor(pct * n_bins) + 1.0
    return np.where(np.isnan(pct), np.nan, np.clip(b, 1, n_bins))


def time_series_bins(pred: np.ndarray, warmup: int, n_bins: int) -> np.ndarray:
    """Return the ts_bin of each prediction. pred: (sessions, tickers)."""
    return to_bins(rank_vs_past(pred.astype(np.float64), int(warmup), int(warmup)), n_bins)


def cross_section_pct(pred: np.ndarray, tie_rule: str = "ordinal") -> np.ndarray:
    """Return the cross-sectional rank of each prediction within its session, in [0, 1].

    pred: (sessions, tickers). NaN where the prediction is NaN or the session has fewer than 2 predictions.
    """
    n = np.isfinite(pred).sum(axis=1, keepdims=True).astype(float)
    if tie_rule == "average":
        order = pd.DataFrame(pred).rank(axis=1, method="average").to_numpy() - 1.0
    else:
        order = np.argsort(np.argsort(np.where(np.isnan(pred), np.inf, pred), axis=1), axis=1).astype(float)
    pct = order / np.maximum(n - 1.0, 1.0)
    return np.where(np.isnan(pred) | (n < 2), np.nan, pct)
