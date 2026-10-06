"""Groups stage: which asset groups lead over the next h sessions. A row is one (session, group).

A group is an asset group of the universe file (block_name). An equity ETF joins eq_<name> by its role (EQUITY_SPLIT).
The cash-like ETFs (settings.CASH_LIKE) belong to no group.
Group return at t: the mean over the tradable members of r_next[t] = open[t+2] / open[t+1] - 1, the return that a
decision at the close of t earns. A group exists at t when it has a tradable member.
Label of horizon h, with m = max(h, MIN_VOL_WINDOW): the rank among the groups (average ties, 0 to 1) of
    (product of (1 + group return) over t .. t+h-1, minus 1) / (max(sd, VOL_FLOOR_DAILY) x sqrt(h)),
with sd the standard deviation (ddof 1) of the group returns over t .. t+m-1. A missing return counts as 0 in the
product.
Features: the mean of each ETF feature over the tradable members. A global feature is equal on every tradable ETF, so
its group mean is the global value.
Walk forward: a retrain on the first session d of every k-th month. The training rows end at
    d - h - max(h, MIN_VOL_WINDOW) - groups.purge_extra_sessions,
so no training label reads past d. The predictions run from d to the session before the next retrain.
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

from etf66 import config, settings, stage, walkforward

ENSEMBLE_STEM = "grp_ens__all"
ENSEMBLE_ALL_H_STEM = "grp_ens_all_h__all"

EQUITY_SPLIT = {              # group eq_<name>: the universe-file roles it takes
    "broad": ["broad_us_equity", "growth_mega_cap", "small_cap", "mid_cap", "value", "growth", "equal_weight"],
    "sector": ["sector"],
    "industry": ["industry"],
    "factor": ["factor"],
}
HORIZONS = [5, 10, 21]
MIN_VOL_WINDOW = 5            # shortest window of the forward volatility in the label scale
VOL_FLOOR_DAILY = 0.003
MODELS = [
    {"name": "ridge_a1", "kind": "ridge", "alpha": 1.0},
    {"name": "lgbm_groups", "kind": "lightgbm",
     "params": {"num_leaves": 7, "learning_rate": 0.1, "num_boost_round": 80, "min_data_in_leaf": 300, "max_bin": 31,
                "feature_fraction": 0.2, "bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 1.0,
                "seed": 20261005}},
]
WINDOWS = {"expanding": None, "rolling_5y": 1260}   # window name: length in sessions (None = expanding)
RETRAIN_EVERY_MONTHS = 1
MIN_END_ROW = 300             # a fit needs a training end row at or after this session position
MIN_TRAIN_ROWS = 1000         # a fit with fewer (session, group) rows is skipped
SCORE_VARIANT = "rank"        # the ETF-level score files fill the slots of this label variant of the run


def group_map(table: pd.DataFrame, tickers: list) -> tuple[list, np.ndarray]:
    """Return (group names, group index per ticker; -1 for no group) from a ticker table with block_name and role."""
    block = dict(zip(table["ticker"], table["block_name"]))
    role = dict(zip(table["ticker"], table["role"]))
    split = {r: name for name, roles in EQUITY_SPLIT.items() for r in roles}
    exclude = set(settings.CASH_LIKE)
    names_of = {}
    for t in tickers:
        if t in exclude:
            continue
        if block[t] == "equity":
            if role[t] not in split:
                raise config.ConfigError(f"EQUITY_SPLIT does not name the role {role[t]} of {t}.")
            names_of[t] = f"eq_{split[role[t]]}"
        else:
            names_of[t] = block[t]
    names = sorted(set(names_of.values()))
    gmap = np.array([names.index(names_of[t]) if t in names_of else -1 for t in tickers], dtype=np.int64)
    return names, gmap


def group_returns(r_next: np.ndarray, tradable: np.ndarray, gmap: np.ndarray, n_groups: int):
    """Return (group return, group exists): (sessions, groups). The return is the mean over the tradable members."""
    g_ret = np.full((r_next.shape[0], n_groups), np.nan)
    g_ok = np.zeros((r_next.shape[0], n_groups), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a group with no member gives "Mean of empty slice"
        for g in range(n_groups):
            cols = np.flatnonzero(gmap == g)
            mask = tradable[:, cols]
            g_ret[:, g] = np.nanmean(np.where(mask, r_next[:, cols], np.nan), axis=1)
            g_ok[:, g] = mask.any(axis=1)
    return g_ret, g_ok


def group_features(feats: np.ndarray, tradable: np.ndarray, gmap: np.ndarray, n_groups: int) -> np.ndarray:
    """Return the group feature array (sessions, groups, features): the mean over the tradable members."""
    out = np.full((feats.shape[0], n_groups, feats.shape[2]), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for g in range(n_groups):
            cols = np.flatnonzero(gmap == g)
            x = np.where(tradable[:, cols][:, :, None], np.asarray(feats[:, cols, :], dtype=np.float32), np.nan)
            out[:, g, :] = np.nanmean(x, axis=1)
    return out


def group_labels(g_ret: np.ndarray, g_ok: np.ndarray, h: int, min_vol_window: int, vol_floor: float) -> np.ndarray:
    """Return the rank label (sessions, groups) of horizon h. The last m + 1 rows have no label."""
    n_sessions, n_groups = g_ret.shape
    m = max(h, min_vol_window)
    fwd = np.full((n_sessions, n_groups), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a window of missing returns gives a NaN deviation
        for t in range(n_sessions - m - 1):
            path = np.nan_to_num(g_ret[t:t + h])
            sd = np.nanstd(g_ret[t:t + m], axis=0, ddof=1)
            fwd[t] = (np.prod(1 + path, axis=0) - 1) / (np.maximum(sd, vol_floor) * math.sqrt(h))
    return stage.rank_across(np.where(g_ok, fwd, np.nan))


def train_end_row(d: int, h: int, purge_extra: int) -> int:
    """Return the last training row of a retrain at session d for horizon h."""
    return d - h - max(h, MIN_VOL_WINDOW) - purge_extra


def run_job(job: dict) -> dict:
    """Return the summary of one (model, window, h) job, and save its predictions (sessions, groups)."""
    cfg, spec, window, h = job["cfg"], job["model"], job["window"], int(job["h"])
    gfeat = np.load(job["features_path"], mmap_mode="r")
    labels = np.load(job["labels_path"])
    g_ok = np.load(job["ok_path"])
    dates = pd.DatetimeIndex(job["dates"])
    n_sessions, n_groups, n_feat = gfeat.shape
    pred = np.full((n_sessions, n_groups), np.nan)
    retrains = walkforward.retrain_indices(dates, RETRAIN_EVERY_MONTHS)
    fits, t0 = 0, time.time()
    for k, d in enumerate(retrains):
        stop = retrains[k + 1] if k + 1 < len(retrains) else n_sessions
        end = train_end_row(d, h, int(cfg["groups"]["purge_extra_sessions"]))
        start = 0 if WINDOWS[window] is None else end - WINDOWS[window] + 1
        if end < MIN_END_ROW or start < 0:
            continue
        xs = np.asarray(gfeat[start:end + 1]).reshape(-1, n_feat)
        ys = labels[start:end + 1].reshape(-1)
        ok = np.isfinite(ys) & g_ok[start:end + 1].reshape(-1)
        if ok.sum() < MIN_TRAIN_ROWS:
            continue
        xp = np.asarray(gfeat[d:stop]).reshape(-1, n_feat).astype(np.float64)
        p = stage.fit_predict(spec, cfg, xs[ok].astype(np.float64), ys[ok], xp, job["threads"])
        pred[d:stop] = np.where(g_ok[d:stop], p.reshape(stop - d, n_groups), np.nan)
        fits += 1
    out = Path(job["out_dir"]) / f"{spec['name']}__{window}__h{h}.npy"
    np.save(out, pred)
    return {"model": spec["name"], "window": window, "h": h, "fits": fits, "seconds": round(time.time() - t0, 1)}


def etf_scores(score: np.ndarray, gmap: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Return the ETF-level percentile (sessions, tickers) of a group score (sessions, groups)."""
    mapped = np.where(gmap[None, :] >= 0, score[:, np.clip(gmap, 0, None)], np.nan)
    return stage.avg_pct(mapped, eligible)


def write_etf_files(folder: Path, pred_dir: Path, scores: dict, score_all: np.ndarray, gmap: np.ndarray,
                    eligible: np.ndarray, targets: list, variant: str) -> list:
    """Write the ETF-level score files and return their paths.

    Both files have the layout of a model file (targets, sessions, tickers).
    grp_ens__all.npy fills the slots of `variant` with the score of their horizon. grp_ens_all_h__all.npy puts the
    all-horizon score in every slot of `variant`. Each ETF gets its group's score, ranked among the eligible ETFs.
    """
    slots = [j for j, (v, _) in enumerate(targets) if v == variant]
    if not slots:
        raise config.ConfigError(f"The score variant {variant} is not a label variant of the run.")
    n_sessions, n_tickers = eligible.shape
    per_h = np.full((len(targets), n_sessions, n_tickers), np.nan, dtype=np.float32)
    for j, (v, h) in enumerate(targets):
        if v == variant and h in scores:
            per_h[j] = etf_scores(scores[h], gmap, eligible)
    all_h = np.full_like(per_h, np.nan)
    all_h[slots] = etf_scores(score_all, gmap, eligible)
    pred_dir.mkdir(parents=True, exist_ok=True)
    paths = [pred_dir / f"{ENSEMBLE_STEM}.npy", pred_dir / f"{ENSEMBLE_ALL_H_STEM}.npy"]
    np.save(paths[0], per_h)
    np.save(paths[1], all_h)
    return paths


def run_groups(cfg: dict, run_cfg: dict, prep: dict, run_dir: Path, workers: int, say) -> Path:
    """Build the groups, their labels and features, run every job, write the scores, and return the stage folder."""
    folder = run_dir / "groups"
    (folder / "preds").mkdir(parents=True, exist_ok=True)
    panel, tradable = prep["panel"], prep["tradable"].to_numpy(bool)
    tickers = list(panel.tickers)
    names, gmap = group_map(stage.read_tickers(), tickers)
    (folder / "group_map.json").write_text(json.dumps({"names": names, "gmap": gmap.tolist(), "tickers": tickers}),
                                           encoding="utf-8")
    say(f"{len(names)} groups: {names}")
    from etf66.backtest import next_open_returns
    r_next = next_open_returns(panel.field("open").to_numpy(float))
    g_ret, g_ok = group_returns(r_next, tradable, gmap, len(names))
    np.save(folder / "group_ok.npy", g_ok)
    t0 = time.time()
    feats = np.load(prep["features"], mmap_mode="r")
    np.save(folder / "group_features.npy", group_features(feats, tradable, gmap, len(names)))
    say(f"group features built ({time.time() - t0:.0f}s)")
    horizons = HORIZONS
    for h in horizons:
        np.save(folder / f"labels_h{h}.npy", group_labels(g_ret, g_ok, h, MIN_VOL_WINDOW, VOL_FLOOR_DAILY))
    models = MODELS
    threads = stage.threads_per_worker(workers, cfg)
    jobs = [{"cfg": cfg, "model": m, "window": w, "h": h, "features_path": str(folder / "group_features.npy"),
             "labels_path": str(folder / f"labels_h{h}.npy"), "ok_path": str(folder / "group_ok.npy"),
             "dates": [str(x.date()) for x in panel.dates], "out_dir": str(folder / "preds"), "threads": threads}
            for m in sorted(models, key=lambda s: s["kind"] != "lightgbm") for w in WINDOWS for h in horizons]
    done = [j for j in jobs if (folder / "preds" / f"{j['model']['name']}__{j['window']}__h{j['h']}.npy").exists()]
    for j in done:
        say(f"resume: skip {j['model']['name']}__{j['window']}__h{j['h']} (predictions exist)")
    jobs = [j for j in jobs if j not in done]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(run_job, j) for j in jobs]):
            say(f"done {fut.result()}")
    # Keep the stack order (horizon, model, window). The float rounding of the mean decides an exact tie between two
    # groups, so another order can change the result.
    index, pct = [], []
    for h in horizons:
        for m in models:
            for w in WINDOWS:
                arr = np.load(folder / "preds" / f"{m['name']}__{w}__h{h}.npy")
                index.append({"model": m["name"], "window": w, "h": h})
                pct.append(stage.rank_across(arr))
    pct = np.stack(pct)
    np.save(folder / "group_pct.npy", pct)
    (folder / "group_pct_index.json").write_text(json.dumps(index), encoding="utf-8")
    scores = {}
    for h in horizons:
        rows = [i for i, e in enumerate(index) if e["h"] == h]
        scores[h] = stage.nanmean_stack([pct[i] for i in rows])
        np.save(folder / f"score_h{h}.npy", scores[h])
    score_all = stage.nanmean_stack(list(pct))
    np.save(folder / "score_all.npy", score_all)
    eligible = stage.eligible_mask(tradable, tickers, settings.CASH_LIKE)
    paths = write_etf_files(folder, run_dir / "preds", scores, score_all, gmap, eligible,
                            walkforward.target_list(run_cfg), SCORE_VARIANT)
    say(f"ETF-level score files written: {[p.name for p in paths]}")
    stage.write_stage_settings(folder, cfg, "groups", run_cfg)
    return folder
