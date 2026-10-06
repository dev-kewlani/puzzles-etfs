"""Walk forward: monthly refits and daily predictions.

The model retrains at the close of the first session of every k-th month (k = retrain_every_months), counted from
the first month of the data. For a retrain at session d and horizon h, the training rows are the sessions t with
    t <= end = d - reach(h) - purge_extra_sessions,   reach(h) = 1 + max(h, min_vol_window)
so every training label is fully known before d. That is the purge.
first_row is the first session with a tradable ETF. The windows:
    expanding   first_row to end; starts once end - first_row + 1 >= EXPANDING_MIN_SESSIONS
    rolling_W   the last W sessions up to end; starts once end - W + 1 >= first_row
A training row is a (session, ETF) that is tradable at t and has every label variant of h finite.
With fewer than MIN_ROWS rows there is no fit, and the predictions up to the next retrain stay NaN.
A fit predicts every tradable (session, ETF) from d to the session before the next retrain, from that session's
features. The variants of one horizon share the rows and one fit.
Output per (model, window): a (targets, sessions, tickers) prediction array and a fits table.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from etf66 import config as config_mod
from etf66 import labels as lab
from etf66 import settings, stage
from etf66.models import make_model

EXPANDING_MIN_SESSIONS = 252  # one year of sessions before the first expanding fit
MIN_ROWS = 1000               # (session, ETF) rows


def target_list(cfg: dict) -> list:
    """Return the ordered list of (variant, h) targets."""
    return [(v, int(h)) for h in cfg["labels"]["horizons"] for v in cfg["labels"]["variants"]]


def labels_path(tag: str) -> Path:
    """Return the path of the label array of a cache tag (data/derived/labels_<tag>.npy)."""
    return settings.DATA_DIR / "derived" / f"labels_{tag}.npy"


def save_labels(label_tables: dict, targets: list, tag: str) -> Path:
    """Write the labels as one float32 array (targets, sessions, tickers) and return its path."""
    arr = np.stack([label_tables[t].to_numpy(np.float32) for t in targets])
    path = labels_path(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    path.with_suffix(".json").write_text(json.dumps([list(t) for t in targets]), encoding="utf-8")
    return path


def retrain_indices(dates: pd.DatetimeIndex, every_months: int) -> list:
    """Return the session positions of the first session of every k-th month, from the first month of the data."""
    month = dates.year * 12 + dates.month
    first = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])
    return [int(i) for i in first[::every_months]]


def window_start(window: str, end_row: int, first_row: int, cfg: dict) -> int | None:
    """Return the first training row of a window, or None when the window cannot start yet."""
    wf = cfg["walkforward"]
    if window == "expanding":
        return first_row if end_row - first_row + 1 >= EXPANDING_MIN_SESSIONS else None
    start_row = end_row - int(wf["window_sessions"][window]) + 1
    return start_row if start_row >= first_row else None


def train_end_row(d: int, h: int, cfg: dict) -> int:
    """Return the last training row of a retrain at session d for horizon h (the purge)."""
    reach = lab.label_reach(h, int(cfg["labels"]["min_vol_window"]))
    return d - reach - int(cfg["walkforward"]["purge_extra_sessions"])


def gather_rows(x: np.ndarray, y: np.ndarray, ok: np.ndarray, start_row: int, end_row: int):
    """Return x_rows (rows, features) float32 and y_rows (rows, targets) float64 for t in [start_row, end_row] where ok.

    x: (sessions, tickers, features); y: (targets, sessions, tickers); ok: (sessions, tickers).
    """
    t_idx, i_idx = np.nonzero(ok[start_row:end_row + 1])
    t_idx = t_idx + start_row
    return np.asarray(x[t_idx, i_idx, :], dtype=np.float32), np.asarray(y[:, t_idx, i_idx].T, dtype=np.float64)


def run_job(job: dict) -> dict:
    """Run one (model, window) job: save its predictions and fits table, and return a summary."""
    cfg, spec, window = job["cfg"], job["model"], job["window"]
    feats = np.load(job["features_path"], mmap_mode="r")
    labels = np.load(job["labels_path"], mmap_mode="r")
    tradable = np.load(job["tradable_path"])
    targets = [tuple(t) for t in json.loads(Path(job["labels_path"]).with_suffix(".json").read_text())]
    dates = pd.DatetimeIndex(job["dates"])
    n_sessions, n_tickers, _ = feats.shape
    first_row = int(np.flatnonzero(tradable.any(axis=1))[0])
    preds = np.full((len(targets), n_sessions, n_tickers), np.nan, dtype=np.float32)
    horizons = sorted({h for _, h in targets})
    wf = cfg["walkforward"]
    retrains = retrain_indices(dates, int(wf["retrain_every_months"]))
    log, t0 = [], time.time()
    for k, d in enumerate(retrains):
        stop = retrains[k + 1] if k + 1 < len(retrains) else n_sessions
        p_t, p_i = np.nonzero(tradable[d:stop])
        if len(p_t) == 0:
            continue
        p_t = p_t + d
        xp = np.asarray(feats[p_t, p_i, :], dtype=np.float32)
        for h in horizons:
            idx = [j for j, (_, hh) in enumerate(targets) if hh == h]
            end_row = train_end_row(d, h, cfg)
            start_row = window_start(window, end_row, first_row, cfg) if end_row >= first_row else None
            if start_row is None:
                continue
            y_h = labels[idx]
            ok = tradable & np.isfinite(y_h).all(axis=0)
            xt, yt = gather_rows(feats, y_h, ok, start_row, end_row)
            if len(yt) < MIN_ROWS:
                continue
            model = make_model(spec, cfg, job["threads"]).fit(xt, yt)
            out = model.predict(xp).astype(np.float32)
            for col, j in enumerate(idx):
                preds[j, p_t, p_i] = out[:, col]
            log.append({"retrain": str(dates[d].date()), "h": h, "train_start": str(dates[start_row].date()),
                        "train_end": str(dates[end_row].date()), "rows": int(len(yt))})
    out_path = Path(job["out_dir"]) / f"{spec['name']}__{window}.npy"
    pd.DataFrame(log).to_csv(out_path.with_suffix(".fits.csv"), index=False)
    tmp = out_path.with_suffix(".tmp.npy")
    np.save(tmp, preds)
    tmp.replace(out_path)                     # resume skips any job whose .npy exists, so it appears only when complete
    return {"model": spec["name"], "window": window, "fits": len(log), "seconds": round(time.time() - t0, 1)}


def run_walkforward(cfg: dict, features_path: Path, labels_file: Path, tradable: np.ndarray,
                    dates: pd.DatetimeIndex, out_dir: Path, workers: int) -> list:
    """Run every (model, window) job in parallel processes and return the summaries.

    A job whose prediction file exists is skipped, so a stopped run continues where it stopped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tradable_path = out_dir / "tradable.npy"
    np.save(tradable_path, tradable)
    threads = stage.threads_per_worker(workers, cfg)
    # LightGBM jobs are the slowest, so they start first.
    models = sorted(config_mod.enabled_models(cfg), key=lambda s: s["kind"] != "lightgbm")
    jobs = [{"cfg": cfg, "model": m, "window": w, "features_path": str(features_path), "labels_path": str(labels_file),
             "tradable_path": str(tradable_path), "dates": [str(x.date()) for x in dates], "out_dir": str(out_dir),
             "threads": threads}
            for m in models for w in cfg["walkforward"]["windows"]]
    done = [j for j in jobs if (out_dir / f"{j['model']['name']}__{j['window']}.npy").exists()]
    for j in done:
        print(f"[walkforward] resume: skip {j['model']['name']}__{j['window']} (predictions exist)", flush=True)
    jobs = [j for j in jobs if j not in done]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j): (j["model"]["name"], j["window"]) for j in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            print(f"[walkforward {time.strftime('%H:%M:%S')}] done {res}", flush=True)
            results.append(res)
    return results


def first_full_bin_index(cfg: dict, dates: pd.DatetimeIndex, first_row: int) -> dict:
    """Return {(window, h): session position of the first full ts_bin}, or None when the window accepts no retrain.

    The first fit of (window, h) is at the first retrain d whose training end row the window accepts.
    The first full ts_bin comes WARMUP_PREDICTIONS sessions after d.
    """
    from etf66.bins import WARMUP_PREDICTIONS
    retrains = retrain_indices(dates, int(cfg["walkforward"]["retrain_every_months"]))
    out = {}
    for window in cfg["walkforward"]["windows"]:
        for h in cfg["labels"]["horizons"]:
            first = None
            for d in retrains:
                end = train_end_row(d, h, cfg)
                if end >= first_row and window_start(window, end, first_row, cfg) is not None:
                    first = d + WARMUP_PREDICTIONS
                    break
            out[(window, h)] = first
    return out


def late_bin_warnings(cfg: dict, dates: pd.DatetimeIndex, first_row: int) -> list:
    """Return a warning for each (window, h) whose first full ts_bin comes after evaluation.common_start."""
    start = int(dates.searchsorted(pd.Timestamp(cfg["evaluation"]["common_start"])))
    late = []
    for (window, h), pos in first_full_bin_index(cfg, dates, first_row).items():
        if pos is None or pos > start:
            when = "never" if pos is None or pos >= len(dates) else str(dates[pos].date())
            late.append(f"Window {window}, horizon {h}: the first full ts_bin is on {when}, "
                        f"after evaluation.common_start {cfg['evaluation']['common_start']}.")
    return late
