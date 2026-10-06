"""The backtest grid: every prediction series x every portfolio variant.

A prediction series is one (model, window, target). A portfolio variant is one (rule variant, rebalance interval,
rebalance offset, no-trade band). The grid is the constants below plus portfolio.RULES.
For each series the code builds the bins once and computes the IC against the series' own label. Then it runs
every variant through the simulator. Evaluation covers evaluation.common_start to evaluation.end (or the last
session). Output per (model, window) file: one row per trial, the net daily returns (float32) and the per-slice
moments for PBO.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from etf66 import bins, metrics, portfolio, settings

REBALANCE_EVERY = [1, 5, 10, 21, "horizon"]   # "horizon" means h of the target
REBALANCE_OFFSETS = [0]       # sessions after the evaluation start at which the rebalance clock starts
NO_TRADE_BANDS = [0.0, 0.02]
COST_BPS_PER_SIDE = 5
STRESS_COST_BPS_PER_SIDE = [10, 20]


def next_open_returns(open_: np.ndarray) -> np.ndarray:
    """Return r_next[t] = open[t+2] / open[t+1] - 1 (the return of a position decided at the close of t).

    open_: (sessions, tickers).
    """
    out = np.full(open_.shape, np.nan)
    out[:-2] = open_[2:] / open_[1:-1] - 1.0
    return out


def rebalance_choices(h: int) -> list:
    """Return the distinct rebalance intervals for a target of horizon h ("horizon" becomes h)."""
    out = []
    for n in REBALANCE_EVERY:
        n = h if n == "horizon" else int(n)
        if n not in out:
            out.append(n)
    return out


def evaluation_rows(cfg: dict, dates: pd.DatetimeIndex, start: str) -> tuple[int, int]:
    """Return (first, stop): the positions of the first evaluation session and one past the last."""
    first = int(np.searchsorted(dates, pd.Timestamp(start)))
    end = cfg["evaluation"]["end"]
    stop = int(np.searchsorted(dates, pd.Timestamp(end), side="right")) if end is not None else len(dates)
    return first, stop


def run_series_file(job: dict) -> dict:
    """Return the summary of one (model, window) prediction file, and write its trial rows, returns and moments."""
    preds = np.load(job["pred_path"], mmap_mode="r")
    labels = np.load(job["labels_path"], mmap_mode="r")
    targets = [tuple(t) for t in json.loads(Path(job["labels_path"]).with_suffix(".json").read_text())]
    ctx = np.load(job["context_path"])
    r_next, cash_next, tradable, group_codes = ctx["r_next"], ctx["cash_next"], ctx["tradable"], ctx["group_codes"]
    ew_net = ctx["ew_net"]
    dates = pd.DatetimeIndex(json.loads(Path(job["context_path"]).with_suffix(".json").read_text())["dates"])
    first, stop = int(job["first"]), int(job["stop"])
    ev_dates = dates[first:stop]
    periods = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in job["periods"]]
    n_bins = bins.N_BINS
    cost_per_side = COST_BPS_PER_SIDE / 1e4
    rules = portfolio.expand_rules(portfolio.RULES)
    ew_eval = ew_net[first:stop]
    rows, nets = [], []
    t0 = time.time()
    for j, (variant, h) in enumerate(targets):
        pred = np.asarray(preds[j], dtype=np.float64)
        if not np.isfinite(pred[first:stop]).any():
            continue                    # a groups score file fills only the slots of its own label variant
        ic = metrics.ic_summary(metrics.daily_ic(pred[first:stop], np.asarray(labels[j], float)[first:stop]), h)
        cs_pct = bins.cross_section_pct(pred)
        cs_bin = bins.to_bins(cs_pct, n_bins)
        ts_bin = bins.time_series_bins(pred, bins.WARMUP_PREDICTIONS, n_bins)
        pred_share = float(np.isfinite(pred[first:stop]).any(axis=1).mean())
        for rule in rules:
            target, use_cash = portfolio.rule_weights(rule, ts_bin, cs_bin, cs_pct, tradable, group_codes, n_bins)
            for every in rebalance_choices(h):
                for offset in REBALANCE_OFFSETS:
                    reb = portfolio.rebalance_mask(len(dates), first + int(offset), every)
                    for band in NO_TRADE_BANDS:
                        g, tv = portfolio.simulate(target, reb, r_next, cash_next, band, use_cash, portfolio.MAX_LONG_BOOK)
                        g, tv = g[first:stop], tv[first:stop]
                        net = g - tv * cost_per_side
                        row = {"model": job["model"], "window": job["window"], "variant": variant, "h": h,
                               "rule": rule["label"], "rule_family": rule["rule"], "rebalance_every": every,
                               "rebalance_offset": int(offset), "no_trade_band": band, "prediction_share": pred_share,
                               **ic, **metrics.summary(net, tv, ev_dates, periods)}
                        for s in STRESS_COST_BPS_PER_SIDE:
                            net_s = g - tv * s / 1e4
                            row[f"sharpe_cost{int(s)}"] = metrics.sharpe(net_s)
                            row[f"cagr_cost{int(s)}"] = metrics.cagr(net_s)
                        row["info_ratio_vs_ew"] = metrics.sharpe(net - ew_eval)
                        row["excess_cagr_vs_ew"] = row["cagr"] - metrics.cagr(ew_eval)
                        # The mean of the target gross weight on rebalance sessions, not the held book after drift.
                        row["avg_gross_exposure"] = float(np.abs(target[first:stop][reb[first:stop]]).sum(axis=1).mean())
                        rows.append(row)
                        nets.append(net.astype(np.float32))
    out = Path(job["out_dir"])
    stem = f"{job['model']}__{job['window']}"
    net_arr = np.stack(nets)
    np.save(out / f"returns_{stem}.npy", net_arr)
    s1, s2, n = metrics.slice_moments(net_arr.T, metrics.PBO_SLICES)
    np.savez(out / f"moments_{stem}.npz", s1=s1, s2=s2, n=n)
    trials = pd.DataFrame(rows)
    trials.to_parquet(out / f"trials_{stem}.parquet", index=False)
    return {"file": stem, "trials": len(trials), "seconds": round(time.time() - t0, 1)}


def run_grid(pred_dir: Path, labels_file: Path, context_path: Path, first: int, stop: int, periods: list,
             out_dir: Path, workers: int) -> pd.DataFrame:
    """Return all trial rows of every prediction file in pred_dir, backtested in parallel processes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in sorted(pred_dir.glob("*__*.npy")):
        model, window = p.stem.split("__")
        jobs.append({"pred_path": str(p), "labels_path": str(labels_file), "context_path": str(context_path),
                     "first": first, "stop": stop, "periods": periods, "out_dir": str(out_dir), "model": model,
                     "window": window})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(run_series_file, j) for j in jobs]):
            print(f"[backtest {time.strftime('%H:%M:%S')}] done {fut.result()}", flush=True)
    frames = []
    for j in jobs:
        stem = f"{j['model']}__{j['window']}"
        f = pd.read_parquet(out_dir / f"trials_{stem}.parquet")
        f["returns_file"], f["returns_row"] = f"returns_{stem}.npy", np.arange(len(f))
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def add_deflated_sharpe(trials: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Return the trials with the deflated Sharpe probability. N and the Sharpe variance come from this run.

    dsr_all_runs uses the same variance, but N counts this run plus every earlier run in runs/ledger.parquet.
    Each repeated search on the same dev data makes a high Sharpe weaker evidence.
    """
    sr = trials["sharpe"].to_numpy(float) / np.sqrt(metrics.SESSIONS_PER_YEAR)
    ok = np.isfinite(sr)
    n_sessions = trials["n_sessions"].to_numpy(float)
    skew, kurt = trials["skew"].to_numpy(float), trials["kurtosis"].to_numpy(float)
    trials = trials.copy()
    var = float(np.var(sr[ok], ddof=1)) if ok.sum() > 1 else 0.0
    trials["dsr"] = metrics.deflated_sharpe(sr, n_sessions, skew, kurt, int(ok.sum()), var)
    trials["dsr_trials"] = int(ok.sum())
    earlier = 0
    ledger = settings.RUNS_DIR / "ledger.parquet"
    if ledger.exists():
        runs = pd.read_parquet(ledger, columns=["run_id"])["run_id"]
        earlier = int((runs != run_id).sum())
    trials["dsr_all_runs"] = metrics.deflated_sharpe(sr, n_sessions, skew, kurt, int(ok.sum()) + earlier, var)
    trials["dsr_all_runs_trials"] = int(ok.sum()) + earlier
    return trials


def grid_pbo(out_dir: Path, trials: pd.DataFrame) -> dict:
    """Return the CSCV PBO of the whole grid from the stored slice moments (files in the order of the trials)."""
    parts = [np.load(out_dir / f.replace("returns_", "moments_").replace(".npy", ".npz"))
             for f in trials["returns_file"].drop_duplicates()]
    s1 = np.concatenate([p["s1"] for p in parts], axis=1)
    s2 = np.concatenate([p["s2"] for p in parts], axis=1)
    return metrics.pbo_from_moments(s1, s2, parts[0]["n"])
