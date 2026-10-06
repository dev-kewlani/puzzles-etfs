"""Evaluation measures. Every return measure reads net daily returns over the evaluation period only.

Definitions (Y = 252 sessions a year):
  sharpe          mean / std of daily returns x sqrt(Y); no risk-free rate (cash is part of the book)
  cagr            (product of (1 + r)) ^ (Y / sessions) - 1
  max_drawdown    the most negative value of equity / running peak - 1
  turnover_year   sum of |weight change| per year (one unit = the full book bought or sold once)
  hit_rate        share of sessions with a positive return, among sessions with a non-zero return
  period Sharpe   the Sharpe in each period block of evaluation.blocks (columns block1_sharpe ...); the median of
                  the blocks (median_block_sharpe) is the main ranking column
  ic              per session, the Spearman correlation (average ranks for ties) of predictions and labels across
                  ETFs. t-statistic = mean / std x sqrt(sessions / h). The division by h is a rough allowance for
                  the overlap of h-session labels.
  dsr             deflated Sharpe probability: the chance that the true Sharpe is above the best Sharpe expected from
                  N trials of pure noise, given the variance of the trial Sharpes, the skew and the kurtosis
  pbo             probability of backtest overfitting by CSCV (combinatorially symmetric cross-validation) over every
                  split of the evaluation slices into two halves
These are evidence columns. No measure removes a trial.
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from etf66.settings import SESSIONS_PER_YEAR

PBO_SLICES = 16               # CSCV cuts the evaluation sessions into this many equal slices
IC_MIN_ETFS = 5               # a session needs this many ETFs with a prediction and a label for an IC
IC_MIN_SESSIONS = 20          # the IC summary needs this many session ICs
EULER = 0.5772156649015329    # the Euler-Mascheroni constant of the expected maximum of N normal draws
DEN_FLOOR = 1e-12             # floor of the DSR denominator
VAR_FLOOR = 1e-30             # floor of a variance in the CSCV Sharpe


def sharpe(r: np.ndarray) -> float:
    """Return the yearly Sharpe ratio of daily returns (NaN when the standard deviation is 0)."""
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * math.sqrt(SESSIONS_PER_YEAR)) if sd > 0 else math.nan


def cagr(r: np.ndarray) -> float:
    """Return the compound yearly growth rate of daily returns."""
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return math.nan
    total = np.prod(1.0 + r)
    return float(total ** (SESSIONS_PER_YEAR / len(r)) - 1.0) if total > 0 else -1.0


def max_drawdown(r: np.ndarray) -> float:
    """Return the maximum drawdown (a negative number) of daily returns."""
    eq = np.cumprod(1.0 + np.nan_to_num(r))
    return float((eq / np.maximum.accumulate(eq) - 1.0).min()) if len(eq) else math.nan


def hit_rate(r: np.ndarray) -> float:
    """Return the share of positive sessions among sessions with a non-zero return."""
    nz = r[np.isfinite(r) & (r != 0)]
    return float((nz > 0).mean()) if len(nz) else math.nan


def period_sharpes(r: np.ndarray, dates, periods) -> list:
    """Return the Sharpe of each (start, end) period block. r: (sessions,); dates: the sessions of r."""
    out = []
    for start, end in periods:
        m = (dates >= start) & (dates <= end)
        out.append(sharpe(r[m]))
    return out


def summary(r: np.ndarray, turnover: np.ndarray, dates, periods) -> dict:
    """Return the standard measures of one net return series. r, turnover: (sessions,)."""
    ps = period_sharpes(r, dates, periods)
    finite = r[np.isfinite(r)]
    return {
        "n_sessions": int(np.isfinite(r).sum()), "sharpe": sharpe(r), "cagr": cagr(r),
        "vol": float(np.nanstd(r, ddof=1) * math.sqrt(SESSIONS_PER_YEAR)),
        "max_drawdown": max_drawdown(r), "turnover_year": float(np.nansum(turnover) * SESSIONS_PER_YEAR / max(len(r), 1)),
        "hit_rate": hit_rate(r), "median_block_sharpe": float(np.nanmedian(ps)) if np.isfinite(ps).any() else math.nan,
        "min_block_sharpe": float(np.nanmin(ps)) if np.isfinite(ps).any() else math.nan,
        "skew": float(stats.skew(finite)), "kurtosis": float(stats.kurtosis(finite, fisher=False)),
        **{f"block{k + 1}_sharpe": v for k, v in enumerate(ps)},
    }


def daily_ic(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Return the IC of each session. pred, label: (sessions, tickers).

    NaN for a session with fewer than IC_MIN_ETFS ETFs that have both a prediction and a label.
    """
    ok = np.isfinite(pred) & np.isfinite(label)
    keep = ok.sum(axis=1) >= IC_MIN_ETFS
    ok &= keep[:, None]
    if not keep.any():
        return np.full(len(pred), np.nan)
    a = pd.DataFrame(np.where(ok, pred, np.nan)).rank(axis=1).to_numpy()
    b = pd.DataFrame(np.where(ok, label, np.nan)).rank(axis=1).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # sessions with no value give "Mean of empty slice"
        a = a - np.nanmean(a, axis=1, keepdims=True)
        b = b - np.nanmean(b, axis=1, keepdims=True)
    num = np.nansum(a * b, axis=1)
    den = np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
    ic = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return np.where(keep, ic, np.nan)


def ic_summary(ic: np.ndarray, h: int) -> dict:
    """Return the mean IC and its t-statistic with the session count divided by h."""
    x = ic[np.isfinite(ic)]
    if len(x) < IC_MIN_SESSIONS:
        return {"ic_mean": math.nan, "ic_t": math.nan, "ic_sessions": len(x)}
    t = x.mean() / x.std(ddof=1) * math.sqrt(len(x) / max(int(h), 1))
    return {"ic_mean": float(x.mean()), "ic_t": float(t), "ic_sessions": len(x)}


def expected_max_sharpe(n_trials: int, sharpe_var: float) -> float:
    """Return the expected best Sharpe (per session) of n_trials trials of noise with the given Sharpe variance."""
    if n_trials < 2 or not sharpe_var > 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_var) * ((1.0 - EULER) * z1 + EULER * z2)


def deflated_sharpe(sr_session: np.ndarray, n_sessions: np.ndarray, skew: np.ndarray, kurt: np.ndarray,
                    n_trials: int, sharpe_var: float) -> np.ndarray:
    """Return the deflated Sharpe probability of each trial (per-session Sharpe, sessions, skew, raw kurtosis)."""
    sr0 = expected_max_sharpe(n_trials, sharpe_var)
    den = np.sqrt(np.maximum(1.0 - skew * sr_session + (kurt - 1.0) / 4.0 * sr_session ** 2, DEN_FLOOR))
    return stats.norm.cdf((sr_session - sr0) * np.sqrt(np.maximum(n_sessions - 1, 1)) / den)


def slice_moments(returns: np.ndarray, n_slices: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sum, sum of squares, count) per slice of a (sessions, trials) return matrix. NaN counts as 0.

    The slices are n_slices equal parts of the sessions, in time order.
    """
    r = np.nan_to_num(returns.astype(np.float64))
    edges = np.linspace(0, len(r), n_slices + 1).astype(int)
    s1 = np.stack([r[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    s2 = np.stack([(r[a:b] ** 2).sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    n = np.diff(edges).astype(float)
    return s1, s2, n


def pbo_from_moments(s1: np.ndarray, s2: np.ndarray, n: np.ndarray) -> dict:
    """Return the CSCV probability of backtest overfitting from per-slice moments (slices, trials).

    For every split of the slices into two halves, take the trial with the best Sharpe in the first half and record
    its rank in the second half as a logit. PBO is the share of splits with a logit at or below 0.
    """
    n_slices, m = s1.shape
    logits = []
    for ins in itertools.combinations(range(n_slices), n_slices // 2):
        ins = np.array(ins)
        oos = np.setdiff1d(np.arange(n_slices), ins)
        sr = []
        for part in (ins, oos):
            k = n[part].sum()
            mean = s1[part].sum(axis=0) / k
            var = s2[part].sum(axis=0) / k - mean ** 2
            sr.append(np.where(var > 0, mean / np.sqrt(np.maximum(var, VAR_FLOOR)), -np.inf))
        best = int(np.argmax(sr[0]))
        rank = (sr[1] < sr[1][best]).sum() + 0.5 * ((sr[1] == sr[1][best]).sum() - 1) + 1
        w = rank / (m + 1)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "logit_median": float(np.median(logits)), "splits": len(logits)}
