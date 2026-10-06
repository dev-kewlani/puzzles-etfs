"""Build every feature, normalize it point in time, and stack the result into one array.

Normalization is the same for every model:
- Per-ETF feature: the cross-sectional rank among the tradable ETFs of the session, minus 0.5 (range -0.5 .. 0.5).
- Global feature: the percentile among all strictly earlier values, minus 0.5. It needs GLOBAL_MIN_HISTORY earlier
  values. Every tradable ETF of the session gets the same value; other ETFs get NaN.
Missing values stay NaN. Ridge standardizes the features and then sets NaN to 0, which is the training mean.
LightGBM uses NaN directly.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from etf66 import ops, settings
from etf66.features import bars, cross, macro, price, state
from etf66.features.context import Context

PER_ETF_GROUPS = {
    "price": price.momentum, "volatility": price.volatility, "drawdown": price.drawdown, "trend": price.trend,
    "candles": bars.candles, "gaps": bars.gaps, "compression": bars.compression, "volume": bars.volume,
    "relative": cross.relative, "screeners": cross.screeners,
}
GLOBAL_GROUPS = {"pca_stock": state.stock_state, "macro": macro.macro, "calendar": macro.calendar}
FEATURE_GROUPS = (*PER_ETF_GROUPS, "pca_etf", *GLOBAL_GROUPS)
GLOBAL_MIN_HISTORY = 252      # one year of earlier values before a global percentile exists


@dataclass
class FeatureSet:
    """The stacked feature array and its metadata."""
    values: np.ndarray                    # float32 (sessions, tickers, features)
    names: list
    groups: list                          # group of each feature, in the same order as names
    dates: pd.DatetimeIndex
    tickers: list


def log_step(msg: str) -> None:
    """Print a progress line of the feature build."""
    print(f"[features {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_per_etf(frame: pd.DataFrame, tradable: pd.DataFrame) -> np.ndarray:
    """Return a per-ETF feature as float32: the cross-sectional rank minus 0.5."""
    return (ops.cs_rank(frame.astype(float), tradable) - 0.5).to_numpy(np.float32)


def normalize_global(series: pd.Series, tradable: pd.DataFrame) -> np.ndarray:
    """Return the point-in-time percentile of a global series, minus 0.5, on every tradable ETF (NaN elsewhere)."""
    pct = ops.expanding_pct(series.astype(float).replace([np.inf, -np.inf], np.nan), GLOBAL_MIN_HISTORY) - 0.5
    arr = np.repeat(pct.to_numpy(np.float32)[:, None], tradable.shape[1], axis=1)
    arr[~tradable.to_numpy()] = np.nan
    return arr


def build_features(x: Context, groups=FEATURE_GROUPS) -> FeatureSet:
    """Build the features of `groups` and return the stacked, normalized FeatureSet."""
    columns, names, kinds = [], [], []

    def add(name: str, values: np.ndarray, group: str) -> None:
        columns.append(values)
        names.append(name)
        kinds.append(group)

    for g, fn in PER_ETF_GROUPS.items():
        if g not in groups:
            continue
        t0 = time.time()
        for name, frame in fn(x).items():
            add(name, normalize_per_etf(frame.replace([np.inf, -np.inf], np.nan), x.tradable), g)
        log_step(f"{g}: done in {time.time() - t0:.1f}s")
    if "pca_etf" in groups:
        t0 = time.time()
        per_etf, glob = state.etf_factor_features(x)
        for name, frame in per_etf.items():
            add(name, normalize_per_etf(frame, x.tradable), "pca_etf")
        glob.update(state.etf_state(x))
        for name, s in glob.items():
            add(name, normalize_global(s, x.tradable), "pca_etf")
        log_step(f"pca_etf: done in {time.time() - t0:.1f}s")
    for g, fn in GLOBAL_GROUPS.items():
        if g not in groups:
            continue
        t0 = time.time()
        for name, s in fn(x).items():
            add(name, normalize_global(s, x.tradable), g)
        log_step(f"{g}: done in {time.time() - t0:.1f}s")
    values = np.stack(columns, axis=2).astype(np.float32)
    return FeatureSet(values=values, names=names, groups=kinds, dates=x.close.index, tickers=list(x.close.columns))


def features_path(tag: str):
    """Return the path of the feature array of a cache tag (data/derived/features_<tag>.npy)."""
    return settings.DATA_DIR / "derived" / f"features_{tag}.npy"


def save_features(fs: FeatureSet, tag: str) -> None:
    """Write the array to data/derived/features_<tag>.npy and the names, groups, dates and tickers to a .json file."""
    path = features_path(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, fs.values)
    meta = {"names": fs.names, "groups": fs.groups, "dates": [str(d.date()) for d in fs.dates], "tickers": fs.tickers}
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def load_features(tag: str) -> FeatureSet:
    """Read a FeatureSet written by save_features (the array is memory-mapped)."""
    path = features_path(tag)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return FeatureSet(values=np.load(path, mmap_mode="r"), names=meta["names"], groups=meta["groups"],
                      dates=pd.DatetimeIndex(meta["dates"]), tickers=meta["tickers"])
