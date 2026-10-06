"""Benchmarks. A benchmark is a table of target weights. It goes through the same simulator, timing and costs as the
model portfolios.

Kinds in FIXED:
  equal_weight   equal weight over the tradable ETFs
  momentum       the top TOP_FRACTION of the tradable ETFs by close[t - skip] / close[t - lookback] - 1
  fixed_weights  constant weights on named tickers
  inverse_vol    1 / (volatility over `window` sessions), over the tradable ETFs
  trend          the ticker when its close is above its `window`-session mean, else cash
Random controls (random_f<fraction>_n<interval>): equal weight on a random fraction of the tradable ETFs, drawn again
on each rebalance session.
No lookahead: every score reads closes up to t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf66 import ops
from etf66.portfolio import EPS, MAX_LONG_BOOK, equal_weight, rebalance_mask, simulate
from etf66.backtest import COST_BPS_PER_SIDE

TOP_FRACTION = 0.10
RANDOM_FRACTIONS = [0.10, 0.20, 0.30]
RANDOM_REBALANCE = [1, 5, 10, 21, 63]
RANDOM_SEED = 20261005
FIXED = {                     # rebalance: a session count, monthly or once
    "ew_monthly": {"kind": "equal_weight", "rebalance": "monthly"},
    "mom_12_1": {"kind": "momentum", "skip": 21, "lookback": 252, "rebalance": 21},
    # same name as the feature mom_6_3, other formula
    "mom_6_3": {"kind": "momentum", "skip": 0, "lookback": 126, "rebalance": 63},
    "spy": {"kind": "fixed_weights", "weights": {"SPY": 1.0}, "rebalance": "once"},
    "spy_ief_60_40": {"kind": "fixed_weights", "weights": {"SPY": 0.6, "IEF": 0.4}, "rebalance": "monthly"},
    "inverse_vol_63": {"kind": "inverse_vol", "window": 63, "rebalance": "monthly"},
    "trend_200d": {"kind": "trend", "ticker": "SPY", "window": 200, "rebalance": 1},
}


def month_start_mask(dates: pd.DatetimeIndex, first: int) -> np.ndarray:
    """Return a bool mask of the first session of each month, from position `first` on."""
    month = dates.year * 12 + dates.month
    m = np.r_[True, month[1:] != month[:-1]]
    m[:first] = False
    return m


def top_fraction(score: np.ndarray, tradable: np.ndarray, fraction: float) -> np.ndarray:
    """Return equal weights over the top `fraction` of the tradable ETFs by score, at least one ETF.

    score, tradable: (sessions, tickers).
    """
    s = np.where(tradable & np.isfinite(score), score, -np.inf)
    n_ok = (tradable & np.isfinite(score)).sum(axis=1)
    k = np.maximum(1, np.ceil(fraction * n_ok)).astype(int)
    order = np.argsort(-s, axis=1)
    rank = np.empty_like(order)
    rows = np.arange(s.shape[0])[:, None]
    rank[rows, order] = np.arange(s.shape[1])[None, :]
    select = (rank < k[:, None]) & np.isfinite(s) & (n_ok[:, None] > 0)
    return equal_weight(select)


def fixed_weights(tickers: list, weights: dict, n_sessions: int) -> np.ndarray:
    """Return a weight table with the same weights on every session."""
    w = np.zeros((n_sessions, len(tickers)))
    for t, x in weights.items():
        w[:, tickers.index(t)] = x
    return w


def rebalance_sessions(spec_rebalance, dates: pd.DatetimeIndex, first: int) -> np.ndarray:
    """Return the rebalance mask of a benchmark: every N sessions, monthly, or once at position `first`."""
    if spec_rebalance == "monthly":
        return month_start_mask(dates, first)
    if spec_rebalance == "once":
        return rebalance_mask(len(dates), first, len(dates) + 1)
    return rebalance_mask(len(dates), first, int(spec_rebalance))


def benchmark_weights(spec: dict, close: pd.DataFrame, tradable: np.ndarray, top_frac: float) -> np.ndarray:
    """Return the target weights (sessions, tickers) of one fixed benchmark."""
    c = close.to_numpy(float)
    tickers = list(close.columns)
    kind = spec["kind"]
    if kind == "equal_weight":
        return equal_weight(tradable)
    if kind == "momentum":
        lookback = int(spec["lookback"])
        score = np.roll(c, int(spec["skip"]), axis=0) / np.roll(c, lookback, axis=0) - 1.0
        score[:lookback] = np.nan                 # np.roll wraps around; the first rows have no history
        return top_fraction(score, tradable, top_frac)
    if kind == "fixed_weights":
        return fixed_weights(tickers, spec["weights"], len(close))
    if kind == "inverse_vol":
        vol = ops.ts_std(close / close.shift(1) - 1.0, int(spec["window"])).to_numpy(float)
        inv = np.where(tradable & (vol > 0), 1.0 / vol, 0.0)
        return inv / np.maximum(inv.sum(axis=1, keepdims=True), EPS)
    if kind == "trend":
        ticker = spec["ticker"]
        px = close[ticker]
        above = (px > ops.ts_mean(px.to_frame(), int(spec["window"]))[ticker]).to_numpy()
        w = np.zeros((len(close), len(tickers)))
        w[:, tickers.index(ticker)] = np.where(above, 1.0, 0.0)
        return w
    raise ValueError(f"unknown benchmark kind {kind}")


def run_benchmarks(close: pd.DataFrame, tradable: pd.DataFrame, r_next: np.ndarray, cash_next: np.ndarray,
                   cfg: dict, first: int) -> tuple[dict, dict]:
    """Return ({name: net daily returns}, {name: turnover}). A random control gives a (draws, sessions) array."""
    cost_per_side = COST_BPS_PER_SIDE / 1e4
    tr = tradable.to_numpy(bool)
    nets, turns = {}, {}
    for name, spec in FIXED.items():
        target = benchmark_weights(spec, close, tr, TOP_FRACTION)
        reb = rebalance_sessions(spec["rebalance"], close.index, first)
        g, tv = simulate(np.asarray(target, float), reb, r_next, cash_next, 0.0, True, MAX_LONG_BOOK)
        nets[name], turns[name] = g - tv * cost_per_side, tv
    rng = np.random.default_rng(RANDOM_SEED)
    for frac in RANDOM_FRACTIONS:
        for every in RANDOM_REBALANCE:
            reb = rebalance_mask(len(close), first, int(every))
            draws = []
            for _ in range(int(cfg["benchmarks"]["random_draws"])):
                score = rng.random(tr.shape)
                target = np.zeros(tr.shape)
                rows = np.flatnonzero(reb)
                target[rows] = top_fraction(score[rows], tr[rows], float(frac))
                g, tv = simulate(target, reb, r_next, cash_next, 0.0, True, MAX_LONG_BOOK)
                draws.append(g - tv * cost_per_side)
            nets[f"random_f{frac:.2f}_n{every}"] = np.stack(draws)
    return nets, turns
