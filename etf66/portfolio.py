"""Portfolio rules and the daily simulator.

Weights decided at the close of session t fill at the open of t+1 and earn open[t+2] / open[t+1] - 1.
Between rebalances the weights drift with the returns. On a rebalance session the book moves to the target.
A held position inside the no-trade band stays; a new entry or an exit always trades.
If the kept positions push a long-only book above 1, the book scales down to 1 and the cut counts as turnover.
Cost = ETF turnover x cost per side. Trades into and out of cash are free.
Long-only books keep unused weight in the cash ETF.
The long-short book holds no cash and pays no borrow fee on its shorts.
"""
from __future__ import annotations

import numba as nb
import numpy as np

EPS = 1e-12                   # floor of a weight sum in a division
MIN_BOOK_SCALE = 1e-9         # a book that lost almost all value keeps its weights unscaled
MAX_LONG_BOOK = 1.0

# The rules of the backtest grid. Each list value is one variant; cs_min_bin 0 means no cross-sectional condition.
RULES = [
    {"name": "top_bins_equal", "top_bins": [1, 2, 3]},
    {"name": "score_capped", "max_weight": 0.15, "max_block_weight": 0.40, "iterations": 10},
    {"name": "ts_gate_cash", "ts_min_bin": [7, 8, 9], "cs_min_bin": [0, 6], "slot_fraction": 0.1},
    {"name": "long_short", "top_bins": [1]},
]


@nb.njit(cache=True)
def simulate(target, rebalance, r_next, cash_next, band, use_cash, max_long_book):
    """Return (gross return, ETF turnover) per decision session.

    target: (sessions, tickers) target weights, read on rebalance sessions only (NaN = 0).
    rebalance: (sessions,) bool. r_next: (sessions, tickers) return from open t+1 to open t+2 (NaN = 0).
    cash_next: (sessions,) the same return for the cash ETF.
    """
    n_sessions, n = target.shape
    w = np.zeros(n)
    cash = 1.0 if use_cash else 0.0
    gross = np.zeros(n_sessions)
    turnover = np.zeros(n_sessions)
    started = False
    for t in range(n_sessions):
        if rebalance[t]:
            tot, turn = 0.0, 0.0
            for i in range(n):
                tw = target[t, i]
                if np.isnan(tw):
                    tw = 0.0
                if abs(tw - w[i]) >= band or tw == 0.0 or w[i] == 0.0:
                    turn += abs(tw - w[i])
                    w[i] = tw
                tot += w[i]
            if use_cash and tot > max_long_book:
                for i in range(n):
                    scaled = w[i] / tot * max_long_book
                    turn += abs(w[i] - scaled)
                    w[i] = scaled
                tot = max_long_book
            if use_cash:
                cash = 1.0 - tot
            turnover[t] = turn
            started = True
        if not started:
            continue
        g = 0.0
        for i in range(n):
            r = r_next[t, i]
            if np.isnan(r):
                r = 0.0
            g += w[i] * r
            w[i] = w[i] * (1.0 + r)
        rc = cash_next[t]
        if np.isnan(rc):
            rc = 0.0
        g += cash * rc
        cash = cash * (1.0 + rc)
        gross[t] = g
        scale = 1.0 + g
        if scale > MIN_BOOK_SCALE:
            for i in range(n):
                w[i] = w[i] / scale
            cash = cash / scale
    return gross, turnover


@nb.njit(cache=True)
def simulate_levered(target, rebalance, r_next, cash_next, spread_daily):
    """Return (gross return, turnover) per session for target weights that can sum above 1 (the book stage).

    target: (sessions, tickers), read on rebalance sessions only (NaN = 0). Cash = 1 - sum of weights. Cash above 0
    earns cash_next. Cash below 0 is a loan that pays cash_next + spread_daily. Turnover counts the ETF legs only.
    """
    t_n, n = target.shape
    w = np.zeros(n)
    cash = 1.0
    gross = np.zeros(t_n)
    turnover = np.zeros(t_n)
    started = False
    for t in range(t_n):
        if rebalance[t]:
            tot, turn = 0.0, 0.0
            for i in range(n):
                tw = target[t, i]
                if np.isnan(tw):
                    tw = 0.0
                turn += abs(tw - w[i])
                w[i] = tw
                tot += tw
            cash = 1.0 - tot
            turnover[t] = turn
            started = True
        if not started:
            continue
        g = 0.0
        for i in range(n):
            r = r_next[t, i]
            if np.isnan(r):
                r = 0.0
            g += w[i] * r
            w[i] = w[i] * (1.0 + r)
        rc = cash_next[t]
        if np.isnan(rc):
            rc = 0.0
        rate = rc + spread_daily if cash < 0.0 else rc
        g += cash * rate
        cash = cash * (1.0 + rate)
        gross[t] = g
        scale = 1.0 + g
        if scale > MIN_BOOK_SCALE:
            for i in range(n):
                w[i] = w[i] / scale
            cash = cash / scale
    return gross, turnover

def rebalance_mask(n_sessions: int, first: int, every: int) -> np.ndarray:
    """Return a bool mask that marks every `every`-th session from position `first` on."""
    m = np.zeros(n_sessions, dtype=np.bool_)
    if first < n_sessions:
        m[first::max(1, int(every))] = True
    return m


def equal_weight(select: np.ndarray) -> np.ndarray:
    """Return equal weights over the selected tickers of each session (rows with no selection get zeros).

    select: (sessions, tickers) bool.
    """
    k = select.sum(axis=1, keepdims=True)
    return np.where(select, 1.0 / np.maximum(k, 1), 0.0)


def top_bins_equal(cs_bin: np.ndarray, top_bins: int, n_bins: int) -> np.ndarray:
    """Return equal weights over the ETFs in the top `top_bins` cross-sectional bins. cs_bin: (sessions, tickers)."""
    return equal_weight(cs_bin >= n_bins - top_bins + 1)


def score_capped(cs_pct: np.ndarray, group_codes: np.ndarray, max_weight: float, max_group: float,
                 iterations: int) -> np.ndarray:
    """Return weights proportional to max(rank - 0.5, 0), capped per ETF and per asset group.

    cs_pct: (sessions, tickers) percentile; group_codes: (tickers,) int. Both caps only lower weights. The cut
    weight goes to cash, with no redistribution.
    """
    raw = np.nan_to_num(np.clip(cs_pct - 0.5, 0.0, None))
    w = raw / np.maximum(raw.sum(axis=1, keepdims=True), EPS)
    for _ in range(iterations):
        w = np.minimum(w, max_weight)
        for g in np.unique(group_codes):
            cols = group_codes == g
            s = w[:, cols].sum(axis=1, keepdims=True)
            w[:, cols] = np.where(s > max_group, w[:, cols] * max_group / np.maximum(s, EPS), w[:, cols])
    return w


def ts_gate_cash(ts_bin: np.ndarray, cs_bin: np.ndarray, tradable: np.ndarray, ts_min_bin: int, cs_min_bin: int,
                 slot_fraction: float) -> np.ndarray:
    """Return gate weights: an ETF passes when ts_bin >= ts_min_bin and (cs_min_bin = 0 or cs_bin >= cs_min_bin).

    K = max(1, ceil(slot_fraction x tradable ETFs)). Each passing ETF gets 1 / max(K, number that pass). When fewer
    than K pass, the rest of the book is cash. All arrays: (sessions, tickers).
    """
    sel = (ts_bin >= ts_min_bin) & ((cs_bin >= cs_min_bin) if cs_min_bin > 0 else True)
    sel = sel & tradable
    k_slots = np.maximum(1, np.ceil(slot_fraction * tradable.sum(axis=1, keepdims=True)))
    count = sel.sum(axis=1, keepdims=True)
    return np.where(sel, 1.0 / np.maximum(k_slots, count), 0.0)


def long_short(cs_bin: np.ndarray, top_bins: int, n_bins: int) -> np.ndarray:
    """Return equal weights long the top bins (sum +1) and short the bottom bins (sum -1)."""
    return equal_weight(cs_bin >= n_bins - top_bins + 1) - equal_weight(cs_bin <= top_bins)


def expand_rules(rules: list) -> list:
    """Return every variant of the rule entries, each a dict with the rule, its label and its parameters.

    The label is the name of the variant in the trials table (for example ts_gate_cash_ts_min7_cs_min0).
    """
    out = []
    for r in rules:
        name = r["name"]
        if name in ("top_bins_equal", "long_short"):
            out += [{"rule": name, "label": f"{name}_top_bins{int(k)}", "top_bins": int(k)} for k in r["top_bins"]]
        elif name == "score_capped":
            mw, mb = float(r["max_weight"]), float(r["max_block_weight"])
            out.append({"rule": name, "label": f"{name}_max_weight{mw}_max_block{mb}", "max_weight": mw,
                        "max_block_weight": mb, "iterations": int(r["iterations"])})
        elif name == "ts_gate_cash":
            out += [{"rule": name, "label": f"{name}_ts_min{int(a)}_cs_min{int(b)}", "ts_min_bin": int(a),
                     "cs_min_bin": int(b), "slot_fraction": float(r["slot_fraction"])}
                    for a in r["ts_min_bin"] for b in r["cs_min_bin"]]
        else:
            raise ValueError(f"unknown rule {name}")
    return out


def rule_weights(rule: dict, ts_bin, cs_bin, cs_pct, tradable, group_codes, n_bins: int) -> tuple[np.ndarray, bool]:
    """Return (target weights, uses cash) for one rule variant. Weights are 0 on ETFs that are not tradable."""
    name = rule["rule"]
    if name == "top_bins_equal":
        w, use_cash = top_bins_equal(cs_bin, rule["top_bins"], n_bins), True
    elif name == "score_capped":
        w = score_capped(cs_pct, group_codes, rule["max_weight"], rule["max_block_weight"], rule["iterations"])
        use_cash = True
    elif name == "ts_gate_cash":
        w = ts_gate_cash(ts_bin, cs_bin, tradable, rule["ts_min_bin"], rule["cs_min_bin"], rule["slot_fraction"])
        use_cash = True
    elif name == "long_short":
        w, use_cash = long_short(cs_bin, rule["top_bins"], n_bins), False
    else:
        raise ValueError(rule)
    return np.where(tradable, w, 0.0), use_cash
