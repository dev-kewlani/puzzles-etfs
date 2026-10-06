"""Operators on whole (sessions x tickers) tables. expanding_pct is the exception: it works on one series.

Time rule: the value at session t reads sessions t and earlier only. No operator reads a session after t.
ts_* operators work along time for each ticker. cs_* operators work across the tickers of one session.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np
import pandas as pd

MIN_FRAC = 0.8          # a rolling value needs this share of its window present, else NaN
MIN_OBS_FLOOR = 2       # and at least this many present values
DIV_EPS = 1e-12         # safe_div gives NaN when |denominator| <= DIV_EPS


def min_obs(window: int) -> int:
    """Return the minimum count of present values for a rolling window (MIN_FRAC of it, rounded up)."""
    return max(MIN_OBS_FLOOR, int(math.ceil(MIN_FRAC * window)))


def safe_div(a, b, eps: float | None = None):
    """Return a / b, with NaN where |b| <= eps (default DIV_EPS)."""
    eps = DIV_EPS if eps is None else eps
    b = b.where(b.abs() > eps) if isinstance(b, (pd.DataFrame, pd.Series)) else (np.nan if abs(b) <= eps else b)
    return a / b


def ts_mean(df, w):
    """Return the rolling mean over w sessions."""
    return df.rolling(w, min_periods=min_obs(w)).mean()


def ts_sum(df, w):
    """Return the rolling sum over w sessions."""
    return df.rolling(w, min_periods=min_obs(w)).sum()


def ts_std(df, w, min_n=None):
    """Return the rolling standard deviation over w sessions (min_n present values, default min_obs(w))."""
    return df.rolling(w, min_periods=min_n or min_obs(w)).std()


def ts_min(df, w):
    """Return the rolling minimum over w sessions."""
    return df.rolling(w, min_periods=min_obs(w)).min()


def ts_max(df, w):
    """Return the rolling maximum over w sessions."""
    return df.rolling(w, min_periods=min_obs(w)).max()


def ts_skew(df, w):
    """Return the rolling skewness over w sessions."""
    return df.rolling(w, min_periods=min_obs(w)).skew()


def ts_kurt(df, w):
    """Return the rolling excess kurtosis over w sessions."""
    return df.rolling(w, min_periods=min_obs(w)).kurt()


def ts_quantile(df, w, q):
    """Return the rolling q-quantile over w sessions."""
    return df.rolling(w, min_periods=min_obs(w)).quantile(q)


def ts_zscore(df, w):
    """Return (value - rolling mean) / rolling std over w sessions."""
    return safe_div(df - ts_mean(df, w), ts_std(df, w))


def ts_ema(df, span):
    """Return the exponential moving average with alpha 2 / (span + 1)."""
    return df.ewm(span=span, min_periods=span, adjust=False).mean()


def as_frame(y, like: pd.DataFrame) -> pd.DataFrame:
    """Return y as a table like `like`; a session series repeats for every column."""
    return pd.DataFrame({c: y for c in like.columns}, index=like.index) if isinstance(y, pd.Series) else y


def ts_cov(x, y, w, min_n=None):
    """Return the rolling covariance of x and y over w sessions (y may be a session series)."""
    y = as_frame(y, x)
    both = x.notna() & y.notna()
    xm, ym = x.where(both), y.where(both)
    k = min_n or min_obs(w)
    n = both.astype(float).rolling(w, min_periods=1).sum()
    mxy = (xm * ym).rolling(w, min_periods=k).mean()
    mx, my = xm.rolling(w, min_periods=k).mean(), ym.rolling(w, min_periods=k).mean()
    return (mxy - mx * my) * n / (n - 1)


def ts_var_like(x, y, w, min_n=None):
    """Return the rolling variance of y over the sessions where x is also present."""
    y = as_frame(y, x)
    return ts_cov(y.where(x.notna()), y.where(x.notna()), w, min_n)


def ts_beta(x, y, w, min_n=None):
    """Return the rolling OLS beta of x on y over w sessions."""
    return safe_div(ts_cov(x, y, w, min_n), ts_var_like(x, y, w, min_n))


def ts_corr(x, y, w):
    """Return the rolling correlation of x and y over w sessions."""
    y = as_frame(y, x)
    both = x.notna() & y.notna()
    return x.where(both).rolling(w, min_periods=min_obs(w)).corr(y.where(both))


def ts_slope(df, w):
    """Return the rolling OLS slope of the values on time (per session) over w sessions."""
    t = pd.DataFrame(np.arange(len(df), dtype=float)[:, None].repeat(df.shape[1], 1), index=df.index, columns=df.columns)
    return ts_beta(df, t.where(df.notna()), w)


@nb.njit(parallel=True, cache=True)
def rank_vs_past(x, w, min_n):
    """Return the percentile of x[t, j] among x[t-w .. t-1, j] (ties count half); NaN with fewer than min_n values.

    x: (sessions, tickers) float. The window holds the w sessions before t, so t itself is not in it.
    """
    t_n, n = x.shape
    out = np.full((t_n, n), np.nan)
    for j in nb.prange(n):
        for t in range(t_n):
            v = x[t, j]
            if np.isnan(v):
                continue
            lo, eq, cnt = 0, 0, 0
            for s in range(max(0, t - w), t):
                u = x[s, j]
                if np.isnan(u):
                    continue
                cnt += 1
                if u < v:
                    lo += 1
                elif u == v:
                    eq += 1
            if cnt >= min_n:
                out[t, j] = (lo + 0.5 * eq) / cnt
    return out


def ts_rank_past(df, w, min_n=None):
    """Return the percentile of the value at t among the previous w values of the same ticker (strictly earlier)."""
    min_n = min_n or min_obs(w)
    return pd.DataFrame(rank_vs_past(df.to_numpy(float), int(w), int(min_n)), index=df.index, columns=df.columns)


@nb.njit(cache=True)
def _expanding_pct(x, min_n):
    """Return the percentile of x[t] among x[0 .. t-1] (ties count half) once min_n earlier values exist.

    x: (sessions,) float.
    """
    out = np.full(x.shape[0], np.nan)
    seen = np.empty(x.shape[0])
    k = 0
    for t in range(x.shape[0]):
        v = x[t]
        if np.isnan(v):
            continue
        if k >= min_n:
            lo = np.searchsorted(seen[:k], v, side="left")
            hi = np.searchsorted(seen[:k], v, side="right")
            out[t] = (lo + 0.5 * (hi - lo)) / k
        pos = np.searchsorted(seen[:k], v)
        seen[pos + 1:k + 1] = seen[pos:k].copy()
        seen[pos] = v
        k += 1
    return out


def expanding_pct(s: pd.Series, min_n: int) -> pd.Series:
    """Return the percentile of each value among all strictly earlier values (point in time)."""
    return pd.Series(_expanding_pct(s.to_numpy(float), int(min_n)), index=s.index, name=s.name)


@nb.njit(parallel=True, cache=True)
def _sessions_since_extreme(x, w, use_max, min_frac):
    """Return (sessions since the w-session max or min) / w. x: (sessions, tickers) float."""
    t_n, n = x.shape
    out = np.full((t_n, n), np.nan)
    for j in nb.prange(n):
        for t in range(w - 1, t_n):
            best, age, cnt = np.nan, 0, 0
            for s in range(t - w + 1, t + 1):
                u = x[s, j]
                if np.isnan(u):
                    continue
                cnt += 1
                if np.isnan(best) or (use_max and u >= best) or ((not use_max) and u <= best):
                    best, age = u, t - s
            if cnt >= min_frac * w:
                out[t, j] = age / w
    return out


def ts_days_since_max(df, w):
    """Return (sessions since the w-session maximum) / w."""
    arr = _sessions_since_extreme(df.to_numpy(float), int(w), True, MIN_FRAC)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def ts_days_since_min(df, w):
    """Return (sessions since the w-session minimum) / w."""
    arr = _sessions_since_extreme(df.to_numpy(float), int(w), False, MIN_FRAC)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


@nb.njit(parallel=True, cache=True)
def _run_length(cond):
    """Return the count of consecutive true values up to each session. cond: (sessions, tickers) bool."""
    t_n, n = cond.shape
    out = np.zeros((t_n, n))
    for j in nb.prange(n):
        run = 0.0
        for t in range(t_n):
            run = run + 1.0 if cond[t, j] else 0.0
            out[t, j] = run
    return out


def ts_run_length(cond: pd.DataFrame) -> pd.DataFrame:
    """Return the count of consecutive sessions, up to session t, on which the condition is true."""
    return pd.DataFrame(_run_length(cond.fillna(False).to_numpy(bool)), index=cond.index, columns=cond.columns)


def cs_rank(df: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the rank of each value among the tickers of the same session, scaled to [0, 1]. Ties share a rank."""
    x = df.where(mask) if mask is not None else df
    r = x.rank(axis=1, method="average")
    n = x.notna().sum(axis=1)
    return (r - 1).div((n - 1).where(n > 1), axis=0)


def cs_zscore(df: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the cross-sectional z-score of each value among the tickers of the same session."""
    x = df.where(mask) if mask is not None else df
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).where(lambda s: s > DIV_EPS), axis=0)


def cs_mean(df: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.Series:
    """Return the cross-sectional mean of each session."""
    return (df.where(mask) if mask is not None else df).mean(axis=1)


def group_mean(df: pd.DataFrame, groups: pd.Series, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return, for each ticker, the mean of its asset group on the same session. groups: ticker -> group name."""
    x = df.where(mask) if mask is not None else df
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for cols in groups.groupby(groups).groups.values():
        cols = [c for c in cols if c in df.columns]
        out[cols] = np.repeat(x[cols].mean(axis=1).to_numpy()[:, None], len(cols), axis=1)
    return out


def broadcast(s: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """Return a sessions x tickers table that repeats a session series for every ticker."""
    return pd.DataFrame(np.repeat(s.to_numpy()[:, None], like.shape[1], axis=1), index=like.index, columns=like.columns)
