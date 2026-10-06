"""Command line of etf66. Commands, in the order of a full run:

  snapshot   copy the source data into data/dev and data/holdout (owner machine only)
  features   build the feature array and the label array from the dev data (the same as labels)
  labels     build the feature array and the label array from the dev data (the same as features)
  train      run the walk-forward models and save the daily predictions of a run
  backtest   run the bins, portfolios, benchmarks and measures of a run, then write the report and the logs
  all        features and labels (when their cache files are missing), train and backtest
  groups     group targets and group predictions of the run --run-id (output: runs/<run_id>/groups/ and two
             ETF-level score files in runs/<run_id>/preds/); groups.source_run copies a run into --run-id first
  environment  basket targets, their predictions and the stress mask of the run --run-id (runs/<run_id>/environment/)
  book       the three-layer book of the run --run-id (runs/<run_id>/book/); --controls adds the control books
  holdout    run the trials named in HOLDOUT_UNLOCK.yaml on the holdout (owner only, one time)

Settings: config.yaml, then each --overlay file, then each --set value (see etf66/config.py).
--smoke adds the overlay configs/overlays/smoke.yaml for a fast end-to-end check.
train, backtest and all on an existing run folder with no --overlay, --set or --smoke use the frozen settings of
that run (runs/<run_id>/config.merged.yaml).
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from etf66 import backtest, benchmarks, config, fence, settings, stage, walkforward
from etf66 import labels as lab
from etf66 import report
from etf66.data import load_panel
from etf66.features.build import build_features, features_path, load_features, save_features
from etf66.features.context import build_context
from etf66.universe import tradable_mask

SMOKE_OVERLAY = settings.CONFIG_DIR / "overlays" / "smoke.yaml"


def say(msg: str) -> None:
    """Print a progress line with the time."""
    print(f"[etf66 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_stock_members(panel) -> None:
    """Raise ConfigError when the snapshot holds a different stock member count than settings.STOCK_STATE_SIZE."""
    if panel.stock_members.empty:
        return
    size = int(panel.stock_members["size_rank"].max())
    if size != settings.STOCK_STATE_SIZE:
        raise config.ConfigError(f"STOCK_STATE_SIZE is {settings.STOCK_STATE_SIZE}, but the snapshot holds {size} stocks "
                                 "per year. Run the snapshot again.")


def prepare(mode: str, cfg: dict, rebuild: bool) -> dict:
    """Return the panel, the tradable mask, the feature and label paths, and the warnings of a data mode.

    The cache key in each file name covers the settings, the source files and the data that change the file.
    The function builds a file when it is missing or when rebuild is True.
    """
    panel = load_panel(mode)
    check_stock_members(panel)
    tradable = tradable_mask(panel)
    feature_tag = f"{mode}_{config.features_cache_key(cfg)}"
    label_tag = f"{mode}_{config.labels_cache_key(cfg)}"
    if rebuild or not features_path(feature_tag).exists():
        say(f"features ({feature_tag}) start")
        save_features(build_features(build_context(panel, tradable, cfg)), feature_tag)
    fs = load_features(feature_tag)
    if list(fs.dates) != list(panel.dates) or fs.tickers != list(panel.tickers):
        raise RuntimeError(f"features_{feature_tag} do not match the panel; run with --rebuild")
    l_path = walkforward.labels_path(label_tag)
    if rebuild or not l_path.exists():
        say(f"labels ({label_tag}) start")
        tables = lab.build_labels(panel.field("open"), tradable, cfg["labels"])
        walkforward.save_labels(tables, walkforward.target_list(cfg), label_tag)
    first_row = int(np.flatnonzero(tradable.to_numpy().any(axis=1))[0])
    warnings = walkforward.late_bin_warnings(cfg, panel.dates, first_row) if mode == "dev" else []
    for w in warnings:
        say(f"warning: {w}")
    return {"panel": panel, "tradable": tradable, "features": features_path(feature_tag), "labels": l_path,
            "warnings": warnings}


def create_run_dir(cfg: dict, run_id: str | None, overlays: list, sets: list, frozen: bool = False) -> tuple[str, Path]:
    """Return (run_id, folder) of runs/<run_id>/, with the settings and the code of this start saved in it.

    With frozen True the folder keeps its saved settings files (the run uses them as they are).
    """
    run_id = run_id or f"{dt.datetime.now():%Y%m%d_%H%M}_{cfg['run_name']}"
    run_dir = settings.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not frozen:
        config.write_run_config(run_dir, cfg, overlays, sets)
    copy_code_to_run_dir(run_dir)
    return run_id, run_dir


def copy_code_to_run_dir(run_dir) -> None:
    """Record the code hash of this start in code_hash.txt and copy the code to code_<hash>/ (a resume is a start)."""
    h = settings.code_hash()
    with open(run_dir / "code_hash.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {h}\n")
    if not (run_dir / f"code_{h}").exists():
        shutil.copytree(settings.REPO_ROOT / "etf66", run_dir / f"code_{h}",
                        ignore=shutil.ignore_patterns("__pycache__", "*.nbi", "*.nbc"))


def read_code_hashes(run_dir) -> str:
    """Return the distinct code hashes of every start of a run, joined with '+'."""
    lines = (run_dir / "code_hash.txt").read_text(encoding="utf-8").splitlines()
    return "+".join(dict.fromkeys(line.split()[-1] for line in lines if line))


def save_eval_context(prep: dict, cfg: dict, run_dir, start: str, use_end: bool = True) -> tuple[dict, int, int]:
    """Return (benchmark output, first, stop) after the benchmarks run, and save the arrays the backtest workers need.

    With use_end False the evaluation runs to the last session (the holdout ignores evaluation.end).
    """
    panel, tradable = prep["panel"], prep["tradable"]
    dates = panel.dates
    first, stop = backtest.evaluation_rows(cfg, dates, start)
    if not use_end:
        stop = len(dates)
    r_next = backtest.next_open_returns(panel.field("open").to_numpy(float))
    cash_next = r_next[:, panel.tickers.index(settings.CASH_TICKER)].copy()
    group_codes = pd.factorize(panel.asset_groups.reindex(panel.tickers).astype(str))[0]
    say("benchmarks start")
    nets, turns = benchmarks.run_benchmarks(panel.field("close"), tradable, r_next, cash_next, cfg, first)
    np.savez(run_dir / "context.npz", r_next=r_next, cash_next=cash_next, tradable=tradable.to_numpy(bool),
             group_codes=group_codes, ew_net=nets["ew_monthly"])
    (run_dir / "context.json").write_text(json.dumps({"dates": [str(d.date()) for d in dates]}), encoding="utf-8")
    np.savez(run_dir / "benchmarks.npz", **nets)
    return {"nets": nets, "turns": turns}, first, stop


def cmd_train(cfg: dict, prep: dict, run_dir, workers: int) -> None:
    """Train the models of a run in the walk forward."""
    say("train start")
    res = walkforward.run_walkforward(cfg, prep["features"], prep["labels"], prep["tradable"].to_numpy(bool),
                                      prep["panel"].dates, run_dir / "preds", workers)
    pd.DataFrame(res).to_csv(run_dir / "train_summary.csv", index=False)


def derive_run(cfg: dict, source: str, run_id: str, sections: tuple) -> tuple[Path, dict]:
    """Return (folder, settings) of the new run --run-id made from the frozen settings of a source run.

    The new run takes the named sections from the current settings. It links the source predictions and copies the
    context and benchmark files. The source run stays as it is.
    """
    if source == run_id:
        raise config.ConfigError("The source run and --run-id must differ. The source run stays as it is.")
    src = settings.RUNS_DIR / source
    run_cfg = copy.deepcopy(config.read_run_config(src))
    for name in sections:
        run_cfg[name] = copy.deepcopy(cfg[name])
    run_cfg["run"]["notes"] = f"derived from the run {source}"
    config.check_rules(run_cfg)
    _, run_dir = create_run_dir(run_cfg, run_id, [], [])
    for name in ("context.npz", "context.json", "benchmarks.npz"):
        if (src / name).exists() and not (run_dir / name).exists():
            shutil.copy2(src / name, run_dir / name)
    n = stage.link_predictions(src / "preds", run_dir / "preds")
    say(f"linked {n} prediction files of {source} into {run_dir / 'preds'}")
    return run_dir, run_cfg


def stage_run(cfg: dict, run_id: str | None, command: str) -> tuple[Path, dict]:
    """Return (folder, frozen settings) of the run that a stage command reads."""
    run_dir = stage.run_dir_of(run_id, command)
    return run_dir, config.read_run_config(run_dir)


def prepare_run_data(run_cfg: dict) -> dict:
    """Return the panel, the tradable mask and the feature and label files of a run (built when missing)."""
    return prepare("dev", run_cfg, rebuild=False)


def cmd_groups(cfg: dict, run_id: str | None, workers: int) -> None:
    """Run the groups stage on the run --run-id (or on a new run made from groups.source_run)."""
    from etf66 import groups
    source = cfg["groups"]["source_run"]
    if source is not None:
        if run_id is None:
            raise config.ConfigError("The groups command needs --run-id (the new run).")
        run_dir, run_cfg = derive_run(cfg, source, run_id, ("groups", "environment", "book"))
    else:
        run_dir, run_cfg = stage_run(cfg, run_id, "groups")
    prep = prepare_run_data(run_cfg)
    say("groups start")
    folder = groups.run_groups(cfg, run_cfg, prep, run_dir, workers, say)
    say(f"groups written: {folder}")


def cmd_environment(cfg: dict, run_id: str | None, workers: int) -> None:
    """Run the environment stage on the run --run-id."""
    from etf66 import environment
    run_dir, run_cfg = stage_run(cfg, run_id, "environment")
    prep = prepare_run_data(run_cfg)
    say("environment start")
    folder = environment.run_environment(cfg, run_cfg, prep, run_dir, workers, say)
    say(f"environment written: {folder}")


def cmd_book(cfg: dict, run_id: str | None, with_controls: bool) -> None:
    """Run the book stage on the run --run-id."""
    from etf66 import book
    run_dir, run_cfg = stage_run(cfg, run_id, "book")
    say("book start")
    folder = book.run_book_stage(cfg, run_cfg, run_dir, with_controls, say)
    say(f"book report written: {folder / 'REPORT.md'}")


def cmd_backtest(cfg: dict, prep: dict, run_id: str, run_dir, workers: int) -> None:
    """Run the backtest grid of a run, then write the report, RUNLOG.yaml and the ledger."""
    periods = cfg["evaluation"]["blocks"]
    bench_out, first, stop = save_eval_context(prep, cfg, run_dir, cfg["evaluation"]["common_start"])
    say("backtest grid start")
    trials = backtest.run_grid(run_dir / "preds", prep["labels"], run_dir / "context.npz", first, stop, periods,
                               run_dir / "backtest", workers)
    trials = backtest.add_deflated_sharpe(trials, run_id)
    pbo = backtest.grid_pbo(run_dir / "backtest", trials)
    trials.insert(0, "trial_id", [f"{run_id}:{i}" for i in range(len(trials))])
    trials.to_parquet(run_dir / "trials.parquet", index=False)
    dates = prep["panel"].dates
    period_ts = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in periods]
    bench, rand = report.benchmark_table(bench_out["nets"], bench_out["turns"], dates, period_ts, first, stop)
    bench.to_csv(run_dir / "benchmarks.csv")
    rand.to_csv(run_dir / "random_controls.csv")
    if prep["warnings"]:
        (run_dir / "config_warnings.txt").write_text("\n".join(prep["warnings"]) + "\n", encoding="utf-8")
    meta = {"run_id": run_id, "time": dt.datetime.now().isoformat(timespec="seconds"),
            "config_hash": config.config_hash(cfg), "code_hash": read_code_hashes(run_dir),
            "manifest_hash": settings.file_hash(settings.DATA_DIR / "MANIFEST.json")[:settings.HASH_CHARS],
            "eval_start": str(dates[first].date()), "eval_end": str(dates[stop - 1].date()), "eval_sessions": stop - first}
    report.write_report(run_dir, meta, trials, bench, rand, pbo)
    report.write_runlog_entry({**meta, "trials": int(len(trials)), "pbo": round(pbo["pbo"], 4),
                               "best_median_block_sharpe": round(float(trials["median_block_sharpe"].max()), 4),
                               "ew_sharpe": round(float(bench.loc["ew_monthly", "sharpe"]), 4),
                               "notes": cfg["run"]["notes"]})
    report.write_ledger_rows(trials, run_id)
    say(f"report written: {run_dir / 'REPORT.md'}")


def cmd_holdout(workers: int) -> None:
    """Run the trials named in HOLDOUT_UNLOCK.yaml on the holdout, one time only, with the frozen dev settings."""
    unlock = fence.read_unlock()
    src = settings.RUNS_DIR / str(unlock["run_id"])
    out = settings.RUNS_DIR / f"holdout_{unlock['run_id']}"
    if out.exists():
        raise fence.FenceError(f"{out} exists. The holdout result is write-once.")
    cfg = config.read_run_config(src)
    chosen = pd.read_parquet(src / "trials.parquet")
    chosen = chosen[chosen["trial_id"].isin([str(x) for x in unlock["config_ids"]])]
    if chosen.empty:
        raise fence.FenceError("HOLDOUT_UNLOCK.yaml names no trial of the run")
    cfg["models"] = [m for m in cfg["models"] if m["name"] in set(chosen["model"])]
    cfg["walkforward"]["windows"] = sorted(set(chosen["window"]))
    out.mkdir(parents=True)
    shutil.copy(settings.UNLOCK_FILE, out / "HOLDOUT_UNLOCK.yaml")
    (out / "config.merged.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    prep = prepare("full", cfg, rebuild=True)
    cmd_train(cfg, prep, out, workers)
    periods = cfg["evaluation"]["holdout_blocks"] or cfg["evaluation"]["blocks"]
    _, first, stop = save_eval_context(prep, cfg, out, str(settings.FENCE_DATE), use_end=False)
    trials = backtest.run_grid(out / "preds", prep["labels"], out / "context.npz", first, stop, periods,
                               out / "backtest", workers)
    keys = ["model", "window", "variant", "h", "rule", "rebalance_every", "no_trade_band"]
    trials = trials.merge(chosen[keys + ["trial_id"]], on=keys, how="inner")
    trials.to_parquet(out / "holdout_trials.parquet", index=False)
    say(f"holdout result written: {out / 'holdout_trials.parquet'}")


def main(argv=None) -> None:
    """Parse the command line and run one command."""
    ap = argparse.ArgumentParser(prog="etf66", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["snapshot", "features", "labels", "train", "backtest", "all", "groups",
                                        "environment", "book", "holdout"])
    ap.add_argument("--overlay", action="append", default=[], help="YAML file merged over config.yaml (repeatable)")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="key.path=value, applied last (repeatable)")
    ap.add_argument("--smoke", action="store_true", help="add the overlay configs/overlays/smoke.yaml (not for holdout)")
    ap.add_argument("--run-id", default=None, help="run folder name under runs/ (train and backtest share it)")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default compute.workers)")
    ap.add_argument("--rebuild", action="store_true", help="build the features and labels again")
    ap.add_argument("--controls", action="store_true", help="book: also run the control books of the reference book")
    a = ap.parse_args(argv)
    overlays = ([str(SMOKE_OVERLAY)] if a.smoke else []) + a.overlay
    if a.command == "holdout":
        if overlays or a.sets:
            raise config.ConfigError("The holdout runs the frozen settings of the dev run only. Remove --overlay, --set "
                                     "and --smoke.")
        cmd_holdout(a.workers or int(config.load_config()["compute"]["workers"]))
        return
    cfg = config.load_config(overlays, a.sets)
    workers = a.workers or int(cfg["compute"]["workers"])
    if a.command == "snapshot":
        from etf66.snapshot import build_snapshot
        build_snapshot()
        return
    if a.command == "groups":
        cmd_groups(cfg, a.run_id, workers)
        return
    if a.command == "environment":
        cmd_environment(cfg, a.run_id, workers)
        return
    if a.command == "book":
        cmd_book(cfg, a.run_id, a.controls)
        return
    frozen = (a.command in ("train", "backtest", "all") and a.run_id is not None and not overlays and not a.sets
              and (settings.RUNS_DIR / a.run_id / "config.merged.yaml").exists())
    if frozen:
        cfg = config.read_run_config(settings.RUNS_DIR / a.run_id)
        say(f"using the frozen settings of the run {a.run_id}")
    prep = prepare("dev", cfg, a.rebuild or a.command in ("features", "labels"))
    if a.command in ("features", "labels"):
        return
    run_id, run_dir = create_run_dir(cfg, a.run_id, overlays, a.sets, frozen)
    if a.command in ("train", "all"):
        cmd_train(cfg, prep, run_dir, workers)
    if a.command in ("backtest", "all"):
        cmd_backtest(cfg, prep, run_id, run_dir, workers)
