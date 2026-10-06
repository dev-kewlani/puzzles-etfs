"""Environment stage: the basket a few sessions ahead, its predictions, and the stress mask.

Basket: equal weight over the tradable ETFs outside settings.CASH_LIKE, rebalanced each session, open to open.
basket[t] is the mean over those ETFs of r_next[t] = open[t+2] / open[t+1] - 1, the return that a decision at the
close of t earns. A session with no return counts as 0.
Targets at t, for h in HORIZONS and m = max(h, MIN_VOL_WINDOW):
  env_ret   product of (1 + basket) over t .. t+h-1, minus 1
  env_dd    the lowest value of equity / running peak - 1 along that path, with the peak at or above 1
  env_vol   standard deviation (ddof 1) of basket over t .. t+m-1, times sqrt(sessions_per_year)
  env_corr  mean pairwise correlation of the ETF returns over t .. t+m-1, over the ETFs with a full window;
            NaN with fewer than MIN_CORR_ASSETS ETFs
A target at t reads opens up to t+m+1, so reach = 1 + m.
Walk forward: an expanding window and a monthly retrain at d. The training rows are
t <= d - reach - environment.purge_extra_sessions, with a finite target and more than MIN_FEATURE_SHARE of the
features finite.
Stress: per model, the percentile of the predicted stress input at STRESS_HORIZON (minus the predicted drawdown, or
the predicted volatility or correlation) against the model's own previous STRESS_LOOKBACK predictions. A stress
session is one where the mean over the models is at or above environment.stress.threshold.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from etf66 import settings, stage, walkforward
from etf66.ops import rank_vs_past

MIN_MEASURE_ROWS = 100          # fewest evaluation rows with a prediction and an outcome for a Spearman
HORIZONS = [3, 5, 10, 21]
MIN_VOL_WINDOW = 10             # shortest window of env_vol and env_corr
TARGETS = ["env_ret", "env_dd", "env_vol", "env_corr"]
MIN_CORR_ASSETS = 6
FEATURE_GROUPS = ["pca_stock", "macro", "calendar"]   # plus the global pca_etf features
GLOBAL_PCA_PREFIX = "etf_"
FEATURE_TICKER = "SPY"          # the column that carries the global features (equal on every tradable ETF)
MIN_FEATURE_SHARE = 0.5
MODELS = [
    {"name": "ridge_a1", "kind": "ridge", "alpha": 1.0},
    {"name": "lgbm_env", "kind": "lightgbm",
     "params": {"num_leaves": 7, "learning_rate": 0.05, "num_boost_round": 150, "min_data_in_leaf": 100, "max_bin": 31,
                "feature_fraction": 0.3, "bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 5.0,
                "seed": 20261005}},
]
RETRAIN_EVERY_MONTHS = 1
MIN_TRAIN_ROWS = 500
LEAD_SESSIONS = 30              # the fits start this many sessions before common_start
BASELINE_MIN_SESSIONS = 252     # the expanding-mean baseline of the out-of-sample R2 needs this many labels
STRESS_HORIZON = 21
STRESS_LOOKBACK = 252           # the percentile of a prediction against the model's own previous predictions
STRESS_MIN_N = 126


def basket_returns(r_next: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Return the equal-weight basket return per session (sessions,); 0 when no ETF has a return."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmean(np.where(eligible, r_next, np.nan), axis=1))


def path_drawdown(path: np.ndarray) -> float:
    """Return the lowest value of equity / running peak - 1 along a return path, with the peak at or above 1."""
    eq = np.cumprod(1 + path)
    return float((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min())


def mean_pair_correlation(x: np.ndarray, min_assets: int) -> float:
    """Return the mean pairwise correlation of the columns of x with a full window; NaN with too few columns."""
    x = x[:, np.isfinite(x).all(axis=0)]
    if x.shape[1] < min_assets:
        return math.nan
    c = np.corrcoef(x, rowvar=False)
    return float((c.sum() - len(c)) / (len(c) * (len(c) - 1)))


def build_targets(basket: np.ndarray, etf_r: np.ndarray, year: int) -> dict:
    """Return {(target, h): (sessions,)} for every target and horizon."""
    n_sessions = len(basket)
    out = {}
    for h in HORIZONS:
        m = max(h, MIN_VOL_WINDOW)
        ret, dd, vol, corr = (np.full(n_sessions, np.nan) for _ in range(4))
        for t in range(n_sessions - m - 1):
            path = basket[t:t + h]
            ret[t] = np.prod(1 + path) - 1
            dd[t] = path_drawdown(path)
            vol[t] = basket[t:t + m].std(ddof=1) * math.sqrt(year)
            corr[t] = mean_pair_correlation(etf_r[t:t + m], MIN_CORR_ASSETS)
        for name, arr in (("env_ret", ret), ("env_dd", dd), ("env_vol", vol), ("env_corr", corr)):
            out[(name, h)] = arr
    return out


def build_persistence(basket: np.ndarray, etf_r: np.ndarray, year: int) -> dict:
    """Return the persistence baseline {(target, h): (sessions,)} from the returns known at the close of t."""
    n_sessions, n_tickers = etf_r.shape
    past_basket = np.r_[np.full(2, np.nan), basket[:-2]]  # row t: open[t] / open[t-1] - 1, known at t
    past_etf = np.vstack([np.full((2, n_tickers), np.nan), etf_r[:-2]])
    out = {}
    for h in HORIZONS:
        m = max(h, MIN_VOL_WINDOW)
        for name in TARGETS:
            p = np.full(n_sessions, np.nan)
            for t in range(m + 2, n_sessions):
                pb = past_basket[t - m + 1:t + 1]
                if name == "env_ret":
                    p[t] = np.prod(1 + pb[-h:]) - 1
                elif name == "env_dd":
                    p[t] = path_drawdown(pb[-h:])
                elif name == "env_vol":
                    p[t] = pb.std(ddof=1) * math.sqrt(year)
                else:
                    p[t] = mean_pair_correlation(past_etf[t - m + 1:t + 1], MIN_CORR_ASSETS)
            out[(name, h)] = p
    return out


def global_columns(names: list, groups: list) -> list:
    """Return the positions of the global features that the environment models read."""
    return [i for i, (n, g) in enumerate(zip(names, groups))
            if g in FEATURE_GROUPS or (g == "pca_etf" and n.startswith(GLOBAL_PCA_PREFIX))]


def run_job(job: dict) -> dict:
    """Return the predictions of one (target, h, model) walk-forward and its summary."""
    cfg, spec, h = job["cfg"], job["model"], int(job["h"])
    x, y = job["x"], job["y"]
    dates = pd.DatetimeIndex(job["dates"])
    n_sessions = len(y)
    reach = 1 + max(h, MIN_VOL_WINDOW)
    pred = np.full(n_sessions, np.nan)
    retrains = walkforward.retrain_indices(dates, RETRAIN_EVERY_MONTHS)
    first_fit = int(job["first"]) - LEAD_SESSIONS
    share = np.isfinite(x).mean(axis=1)
    fits, t0 = 0, time.time()
    for k, d in enumerate(retrains):
        stop = retrains[k + 1] if k + 1 < len(retrains) else n_sessions
        end = d - reach - int(cfg["environment"]["purge_extra_sessions"])
        if d < first_fit or end < 0:
            continue
        rows = np.arange(0, end + 1)
        rows = rows[np.isfinite(y[rows]) & (share[rows] > MIN_FEATURE_SHARE)]
        if len(rows) < MIN_TRAIN_ROWS:
            continue
        pred[d:stop] = stage.fit_predict(spec, cfg, x[rows], y[rows], x[d:stop], job["threads"])
        fits += 1
    return {"key": job["key"], "pred": pred, "fits": fits, "seconds": round(time.time() - t0, 1)}


def spearman(p: np.ndarray, y: np.ndarray, h: int, first: int, stop: int) -> tuple[float, float]:
    """Return (Spearman correlation, t-statistic with n / h) of a prediction and its outcome over [first, stop)."""
    ok = np.isfinite(p) & np.isfinite(y)
    ok[:first] = False
    ok[stop:] = False
    if ok.sum() < MIN_MEASURE_ROWS:
        return math.nan, math.nan
    c = pd.Series(p[ok]).rank().corr(pd.Series(y[ok]).rank())
    return float(c), float(c * math.sqrt(ok.sum() / h))


def oos_r2(p: np.ndarray, y: np.ndarray, reach: int, min_sessions: int, first: int, stop: int) -> float:
    """Return 1 - SSE(prediction) / SSE(expanding mean of the labels) over [first, stop).

    The shift of reach + 1 keeps every baseline label known at t.
    """
    base = pd.Series(y).shift(reach + 1).expanding(min_sessions).mean().to_numpy()
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(base)
    ok[:first] = False
    ok[stop:] = False
    if ok.sum() < MIN_MEASURE_ROWS:
        return math.nan
    return float(1 - np.sum((y[ok] - p[ok]) ** 2) / np.sum((y[ok] - base[ok]) ** 2))


def stress_mask(preds: dict, env_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (stress percentile, stress mask) from the predictions of the stress input and horizon."""
    st = env_cfg["stress"]
    sign = -1.0 if st["input"] == "dd" else 1.0
    parts = []
    for m in MODELS:
        p = preds[(f"env_{st['input']}", STRESS_HORIZON, m["name"])]
        parts.append(rank_vs_past((sign * p).reshape(-1, 1), STRESS_LOOKBACK, STRESS_MIN_N)[:, 0])
    pct = stage.nanmean_stack(parts)
    return pct, np.nan_to_num(pct) >= float(st["threshold"])


def run_environment(cfg: dict, run_cfg: dict, prep: dict, run_dir: Path, workers: int, say) -> Path:
    """Build the targets, run every walk-forward, write the predictability table and the stress mask."""
    year = settings.SESSIONS_PER_YEAR
    folder = run_dir / "environment"
    folder.mkdir(parents=True, exist_ok=True)
    panel, tradable = prep["panel"], prep["tradable"].to_numpy(bool)
    tickers = list(panel.tickers)
    from etf66.backtest import evaluation_rows, next_open_returns
    first, stop = evaluation_rows(cfg, panel.dates, cfg["evaluation"]["common_start"])
    r_next = next_open_returns(panel.field("open").to_numpy(float))
    eligible = stage.eligible_mask(tradable, tickers, settings.CASH_LIKE)
    basket = basket_returns(r_next, eligible)
    etf_r = np.where(eligible, r_next, np.nan)
    t0 = time.time()
    targets = build_targets(basket, etf_r, year)
    persist = build_persistence(basket, etf_r, year)
    np.save(folder / "basket.npy", basket)
    np.savez(folder / "targets.npz", **{f"{k}__h{h}": v for (k, h), v in targets.items()})
    np.savez(folder / "persistence.npz", **{f"{k}__h{h}": v for (k, h), v in persist.items()})
    say(f"environment targets built ({time.time() - t0:.0f}s)")
    feats = np.load(prep["features"], mmap_mode="r")
    meta = json.loads(Path(prep["features"]).with_suffix(".json").read_text(encoding="utf-8"))
    cols = global_columns(meta["names"], meta["groups"])
    x = np.asarray(feats[:, tickers.index(FEATURE_TICKER), :][:, cols], dtype=np.float64)
    models = MODELS
    threads = stage.threads_per_worker(workers, cfg)
    jobs = [{"cfg": cfg, "model": m, "h": h, "x": x, "y": y, "dates": [str(d.date()) for d in panel.dates],
             "first": first, "threads": threads, "key": (name, h, m["name"])}
            for (name, h), y in targets.items() for m in sorted(models, key=lambda s: s["kind"] != "lightgbm")]
    say(f"{len(cols)} global features; {len(jobs)} walk-forwards start")
    preds = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(run_job, j) for j in jobs]):
            res = fut.result()
            preds[res["key"]] = res["pred"]
    np.savez(folder / "predictions.npz", **{f"{k}__h{h}__{m}": v for (k, h, m), v in preds.items()})
    rows = []
    for (name, h), y in targets.items():
        reach = 1 + max(h, MIN_VOL_WINDOW)
        series = {m["name"]: preds[(name, h, m["name"])] for m in models}
        series["persistence"] = persist[(name, h)]
        for model, p in series.items():
            c, t = spearman(p, y, h, first, stop)
            rows.append({"target": name, "h": h, "model": model, "spearman": c, "t": t,
                         "oos_r2": oos_r2(p, y, reach, BASELINE_MIN_SESSIONS, first, stop)})
    table = pd.DataFrame(rows)
    table.to_csv(folder / "predictability.csv", index=False)
    pct, stress = stress_mask(preds, cfg["environment"])
    np.save(folder / "stress_pct.npy", pct)
    np.save(folder / "stress.npy", stress)
    say(f"stress share in the evaluation period: {stress[first:stop].mean():.3f}")
    stage.write_stage_settings(folder, cfg, "environment", run_cfg)
    return folder
