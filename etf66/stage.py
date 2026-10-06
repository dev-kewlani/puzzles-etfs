"""Helpers shared by the groups, environment and book stages.

A stage reads a finished run (runs/<run>/) and writes its own folder inside it. It uses the current settings for its
own section and the frozen settings of the run for the features, the labels and the prediction layout.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from etf66 import config, settings
from etf66.models import make_model


def eligible_mask(tradable: np.ndarray, tickers: list, exclude: list) -> np.ndarray:
    """Return the bool table (sessions, tickers) of the tradable ETFs that are not in `exclude`."""
    return tradable & ~np.isin(tickers, list(exclude))[None, :]


def avg_pct(x: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Return the cross-sectional percentile of x among the eligible ETFs of each session, from 0 to 1.

    x, eligible: (sessions, tickers). Ties get the average rank. A row with one value gives 0.
    """
    x = np.where(eligible, x, np.nan)
    r = pd.DataFrame(x).rank(axis=1)
    n = np.isfinite(x).sum(axis=1)
    return r.sub(1).div(np.maximum(n - 1, 1), axis=0).to_numpy()


def rank_across(x: np.ndarray) -> np.ndarray:
    """Return the rank of each finite value among the finite values of its row, from 0 to 1 (average ties)."""
    return pd.DataFrame(x).rank(axis=1).sub(1.0).div(np.maximum(np.isfinite(x).sum(axis=1) - 1, 1), axis=0).to_numpy()


def nanmean_stack(arrays: list) -> np.ndarray:
    """Return the mean over a list of equal-shaped arrays, cell by cell, over the finite values only."""
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(arrays), axis=0)


def fit_predict(spec: dict, cfg: dict, x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray,
                threads: int) -> np.ndarray:
    """Return the predictions (rows,) of a model entry fit on one target. The arrays are float64."""
    model = make_model(spec, cfg, threads).fit(x_train, y_train[:, None])
    return np.asarray(model.predict(x_pred), dtype=np.float64)[:, 0]


def threads_per_worker(workers: int, cfg: dict) -> int:
    """Return the thread count of one worker process: compute.lightgbm_threads, or the CPU count over the workers."""
    fixed = cfg["compute"]["lightgbm_threads"]
    if fixed is not None:
        return int(fixed)
    return max(1, (os.cpu_count() or workers) // workers)


def run_dir_of(run_id: str | None, command: str) -> Path:
    """Return the folder of the run named by --run-id. Raise ConfigError when it is missing."""
    if run_id is None:
        raise config.ConfigError(f"The {command} command needs --run-id (the run it reads).")
    run_dir = settings.RUNS_DIR / run_id
    if not run_dir.exists():
        raise config.ConfigError(f"{run_dir} does not exist.")
    return run_dir


def write_stage_settings(folder: Path, cfg: dict, section: str, run_cfg: dict) -> None:
    """Save the section of the current settings that a stage used, with the hash of the run settings."""
    text = yaml.safe_dump({section: cfg[section], "evaluation": cfg["evaluation"], "model_defaults": cfg["model_defaults"],
                           "run_config_hash": config.config_hash(run_cfg)}, sort_keys=False)
    (folder / "settings.yaml").write_text(text, encoding="utf-8")


def read_dates(run_dir: Path) -> pd.DatetimeIndex:
    """Return the session calendar of a run from its context.json."""
    return pd.DatetimeIndex(json.loads((run_dir / "context.json").read_text(encoding="utf-8"))["dates"])


def read_tickers() -> pd.DataFrame:
    """Return the ticker table of the snapshot (ticker, block_name, role, ...)."""
    return pd.read_csv(settings.DATA_DIR / "tickers.csv", dtype=str)


def link_predictions(src_dir: Path, dst_dir: Path) -> int:
    """Hard-link (or copy, when a link fails) the prediction files of src_dir into dst_dir. Return the count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.iterdir()):
        if p.suffix not in (".npy", ".csv") or p.name.endswith(".tmp.npy"):
            continue
        q = dst_dir / p.name
        if q.exists():
            continue
        try:
            os.link(p, q)
        except OSError:
            shutil.copy2(p, q)
        n += 1
    return n
