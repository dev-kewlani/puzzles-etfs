"""Labels: what an ETF earns after the decision, measured from the fill.

We decide at the close of session t and fill at the open of t+1. For horizon h:
  r(t, h)   = open[t+1+h] / open[t+1] - 1
  fwd_vol   = std of the open-to-open returns of sessions t+2 .. t+1+m, with m = max(h, min_vol_window)
  scale     = max(fwd_vol, VOL_FLOOR_DAILY) * sqrt(h)
An ETF gets a label at t when it is tradable at t, r exists and all m forward returns exist.

Variants, each a sessions x tickers table:
  vol_scaled         r / scale
  rank               cross-sectional rank of vol_scaled among the labelled ETFs of session t, in [0, 1]
  excess_vol_scaled  (r - mean r of the labelled ETFs) / scale
  bin20              rank cut into 20 equal bins, coded 1/20, 2/20, ..., 1
  drawdown_rank      cross-sectional rank of (max drawdown of the open path t+1 .. t+1+h) / scale; high = small dd
  payoff_l<lambda>   cross-sectional rank of (r - lambda x |max drawdown|) / scale; built only if labels.variants has it

The label at row t reads opens up to t + reach(h), with reach(h) = 1 + max(h, min_vol_window).
The walk forward purges with label_reach().
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

from etf66 import ops

VOL_FLOOR_DAILY = 0.003       # about 4.8% a year; a near-zero fwd vol cannot make r / scale huge
BIN20_BINS = 20
LABEL_VARIANTS = ("vol_scaled", "rank", "excess_vol_scaled", "bin20", "drawdown_rank")
PAYOFF_PREFIX = "payoff_l"


def payoff_lambda(variant: str):
    """Return lambda from a name payoff_l<lambda>, or None when the name is not a valid payoff variant."""
    if not isinstance(variant, str) or not variant.startswith(PAYOFF_PREFIX):
        return None
    try:
        value = float(variant[len(PAYOFF_PREFIX):])
    except ValueError:
        return None
    return value if value >= 0 else None


def is_label_variant(variant) -> bool:
    """Return True for a known label variant: one of LABEL_VARIANTS or payoff_l<lambda>."""
    return variant in LABEL_VARIANTS or payoff_lambda(variant) is not None


def label_reach(h: int, min_vol_window: int) -> int:
    """Return reach(h): the label at row t reads opens up to t + reach."""
    return 1 + max(h, min_vol_window)


@nb.njit(parallel=True, cache=True)
def _forward_drawdown(open_, h):
    """Return the lowest open[s] / (running peak of open[t+1 .. s]) - 1 for s in t+1 .. t+1+h.

    open_: (sessions, tickers). NaN when any open of the path is missing.
    """
    t_n, n = open_.shape
    out = np.full((t_n, n), np.nan)
    for j in nb.prange(n):
        for t in range(t_n - 1 - h):
            base = open_[t + 1, j]
            if np.isnan(base) or np.isnan(open_[t + 1 + h, j]):
                continue
            peak, worst, ok = base, 0.0, True
            for s in range(t + 1, t + 2 + h):
                v = open_[s, j]
                if np.isnan(v):
                    ok = False
                    break
                if v > peak:
                    peak = v
                dd = v / peak - 1.0
                if dd < worst:
                    worst = dd
            if ok:
                out[t, j] = worst
    return out


def build_labels(open_: pd.DataFrame, tradable: pd.DataFrame, labels_cfg: dict, floor: float = VOL_FLOOR_DAILY) -> dict:
    """Return {(variant, h): sessions x tickers table} for every horizon and variant."""
    min_vol_window, n_bins = int(labels_cfg["min_vol_window"]), BIN20_BINS
    r_oo = open_ / open_.shift(1) - 1.0
    out = {}
    for h in labels_cfg["horizons"]:
        m = max(h, min_vol_window)
        r = open_.shift(-(1 + h)) / open_.shift(-1) - 1.0
        fwd_vol = r_oo.rolling(m, min_periods=m).std().shift(-(1 + m))
        scale = fwd_vol.clip(lower=floor) * np.sqrt(h)
        valid = tradable & r.notna() & fwd_vol.notna()
        vs = (r / scale).where(valid)
        rank = ops.cs_rank(vs)
        dd = pd.DataFrame(_forward_drawdown(open_.to_numpy(float), int(h)), index=open_.index, columns=open_.columns)
        out[("vol_scaled", h)] = vs
        out[("rank", h)] = rank
        out[("excess_vol_scaled", h)] = (r.sub(r.where(valid).mean(axis=1), axis=0) / scale).where(valid)
        out[("bin20", h)] = (np.minimum(np.floor(rank * n_bins), n_bins - 1) + 1.0) / n_bins
        out[("drawdown_rank", h)] = ops.cs_rank((dd / scale).where(valid))
        for variant in labels_cfg["variants"]:
            lam = payoff_lambda(variant)
            if lam is not None:
                out[(variant, h)] = ops.cs_rank(((r - lam * dd.abs()) / scale).where(valid & dd.notna()))
    return out
