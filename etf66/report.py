"""The run log and the run report.

runs/RUNLOG.yaml     one entry per run: id, time, hashes, trial count, PBO and notes. Write the notes by hand. A
                     second backtest of the same run replaces its entry and keeps the old notes.
runs/ledger.parquet  every trial of every run, with the run id. A second backtest of the same run replaces the rows
                     of that run. dsr_all_runs counts the trials of every run in the ledger.
runs/<id>/REPORT.md  the report of one run.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import yaml

from etf66 import metrics, settings

RANK_COLUMN = "median_block_sharpe"
TOP_TRIALS = 25
TOP_IC_SERIES = 20
STRESS_COLUMN = 20            # bps per side; must be one of backtest.STRESS_COST_BPS_PER_SIDE

MAIN_COLUMNS = ["sharpe", "median_block_sharpe", "min_block_sharpe", "cagr", "vol", "max_drawdown", "turnover_year"]
MARGINAL_COLUMNS = ["sharpe", "median_block_sharpe", "cagr", "max_drawdown", "turnover_year", "dsr"]


def format_value(x) -> str:
    """Return a short text form of a value for a Markdown table."""
    if isinstance(x, (float, np.floating)):
        return "" if not math.isfinite(x) else f"{x:.3f}"
    return str(x)


def md_table(frame: pd.DataFrame, index: bool = True) -> str:
    """Return a Markdown table of a DataFrame."""
    f = frame.reset_index() if index else frame
    head = "| " + " | ".join(str(c) for c in f.columns) + " |"
    rule = "|" + "|".join("---" for _ in f.columns) + "|"
    body = ["| " + " | ".join(format_value(v) for v in row) + " |" for row in f.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def benchmark_table(nets: dict, turns: dict, dates, periods, first: int, stop: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fixed benchmark table, random control table) over the evaluation sessions [first, stop)."""
    fixed, rand = {}, {}
    ev_dates = dates[first:stop]
    for name, net in nets.items():
        if net.ndim == 1:
            fixed[name] = metrics.summary(net[first:stop], turns[name][first:stop], ev_dates, periods)
        else:
            sr = np.array([metrics.sharpe(x[first:stop]) for x in net])
            mb = np.array([np.nanmedian(metrics.period_sharpes(x[first:stop], ev_dates, periods)) for x in net])
            rand[name] = {"sharpe_p05": np.nanpercentile(sr, 5), "sharpe_p50": np.nanpercentile(sr, 50),
                          "sharpe_p95": np.nanpercentile(sr, 95), "median_block_p50": np.nanpercentile(mb, 50),
                          "median_block_p95": np.nanpercentile(mb, 95), "draws": len(sr)}
    return pd.DataFrame(fixed).T[MAIN_COLUMNS], pd.DataFrame(rand).T


def medians_by(trials: pd.DataFrame, by: str) -> pd.DataFrame:
    """Return the median of the main measures of the trials, grouped by one column."""
    g = trials.groupby(by)
    out = g[MARGINAL_COLUMNS].median()
    out["trials"] = g.size()
    return out


def write_report(run_dir, meta: dict, trials: pd.DataFrame, bench: pd.DataFrame, rand: pd.DataFrame,
                 pbo: dict) -> None:
    """Write runs/<id>/REPORT.md."""
    stress = f"sharpe_cost{STRESS_COLUMN}"
    ew = bench.loc["ew_monthly"]
    p95 = rand["sharpe_p95"].max() if len(rand) else math.nan
    series = trials.drop_duplicates(["model", "window", "variant", "h"])
    top = trials.sort_values(RANK_COLUMN, ascending=False).head(TOP_TRIALS)
    cols = ["model", "window", "variant", "h", "rule", "rebalance_every", "no_trade_band", "sharpe",
            "median_block_sharpe", "min_block_sharpe", "cagr", "max_drawdown", "turnover_year", stress,
            "info_ratio_vs_ew", "ic_mean", "ic_t", "dsr", "dsr_all_runs"]
    lines = [
        f"# Run {meta['run_id']}",
        "",
        "## 1. Run facts",
        "",
        f"- Time: {meta['time']}. Config hash: {meta['config_hash']}. Code hash: {meta['code_hash']}.",
        f"- Data manifest hash: {meta['manifest_hash']}. Data mode: dev (all sessions before {settings.FENCE_DATE}).",
        f"- Evaluation period: {meta['eval_start']} to {meta['eval_end']} ({meta['eval_sessions']} sessions).",
        f"- Trials: {len(trials)}. Prediction series: {len(series)}.",
        f"- This run does not read the holdout ({settings.FENCE_DATE} and later).",
        "- The merged settings of the run are in config.merged.yaml in the run folder.",
        "",
        "## 2. Benchmarks (net of the base cost)",
        "",
        md_table(bench),
        "",
        "Random controls: equal weight over random picks of a fraction of the tradable ETFs.",
        "",
        md_table(rand),
        "",
        "## 3. The grid against the benchmarks",
        "",
        f"- Median trial Sharpe: {format_value(trials['sharpe'].median())}. "
        f"Equal-weight Sharpe: {format_value(ew['sharpe'])}.",
        f"- Share of trials with a Sharpe above equal weight: {format_value((trials['sharpe'] > ew['sharpe']).mean())}.",
        f"- Share of trials with a Sharpe above the highest sharpe_p95 of the random controls ({format_value(p95)}): "
        f"{format_value((trials['sharpe'] > p95).mean())}.",
        f"- PBO of the full grid ({pbo['splits']} splits of {metrics.PBO_SLICES} slices): "
        f"{format_value(pbo['pbo'])}. A value near 0.5 or above means the best in-sample trial is not better than the "
        "median out of sample.",
        f"- DSR uses {int(trials['dsr_trials'].iloc[0])} trials and the Sharpe variance of the grid.",
        "",
        "## 4. Medians by one choice at a time",
        "",
    ]
    for by in ["model", "window", "variant", "h", "rule_family", "rule", "rebalance_every", "no_trade_band"]:
        lines += [f"### By {by}", "", md_table(medians_by(trials, by)), ""]
    lines += [
        "## 5. Prediction quality (IC: rank correlation of prediction and label, per session)",
        "",
        "### Mean IC by model and window",
        "",
        md_table(series.pivot_table(index="model", columns="window", values="ic_mean")),
        "",
        "### Mean IC by label variant and horizon (average over models and windows)",
        "",
        md_table(series.pivot_table(index="variant", columns="h", values="ic_mean")),
        "",
        f"### The {TOP_IC_SERIES} series with the highest IC t-statistic "
        "(a rough correction for the label overlap)",
        "",
        md_table(series.sort_values("ic_t", ascending=False).head(TOP_IC_SERIES)[
            ["model", "window", "variant", "h", "ic_mean", "ic_t", "ic_sessions"]], index=False),
        "",
        f"## 6. The {TOP_TRIALS} trials with the highest {RANK_COLUMN}",
        "",
        "This list is a selection from many trials. Read the DSR column and the PBO before you trust a row.",
        f"Key: {stress} = Sharpe at {STRESS_COLUMN} bps per side. info_ratio_vs_ew = Sharpe of the daily "
        "return of the portfolio minus the daily return of equal weight. dsr_all_runs = DSR with the trials of all "
        "runs in the ledger.",
        "",
        md_table(top[cols], index=False),
        "",
        "## 7. Known limits",
        "",
        "See README.md section 7 for the limits of the data and the universe. Limits of this report:",
        "",
        "- A missing next open counts as a 0 return. A position in an ETF that stops trading keeps its weight.",
        "- Sharpe uses no risk-free rate; cash is part of the book.",
        "- The time-series bin compares predictions of one ETF across retrains. A scale change at a retrain moves the "
        "time-series bin.",
        "- info_ratio_vs_ew is the Sharpe of a difference of returns, not the difference of two Sharpes.",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runlog_entry(entry: dict) -> None:
    """Write the entry of a run to runs/RUNLOG.yaml."""
    path = settings.RUNS_DIR / "RUNLOG.yaml"
    log = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    log = log or {"runs": []}
    old = [r for r in log["runs"] if r.get("run_id") == entry["run_id"]]
    if old and old[-1].get("notes"):
        entry = {**entry, "notes": old[-1]["notes"]}
    log["runs"] = [r for r in log["runs"] if r.get("run_id") != entry["run_id"]] + [entry]
    path.write_text(yaml.safe_dump(log, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_ledger_rows(trials: pd.DataFrame, run_id: str) -> None:
    """Write the trials of a run to runs/ledger.parquet."""
    path = settings.RUNS_DIR / "ledger.parquet"
    t = trials.assign(run_id=run_id, logged_at=dt.datetime.now().isoformat(timespec="seconds"))
    if path.exists():
        old = pd.read_parquet(path)
        t = pd.concat([old[old["run_id"] != run_id], t], ignore_index=True)
    t.to_parquet(path, index=False)
