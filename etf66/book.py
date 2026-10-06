"""The book of a run, in three layers.

Layer 1, the switch. In a stress session (environment/stress.npy), book.switch.defensive_share of the book moves to
the defensive basket: equal weight over the tradable ETFs in DEFENSIVE_TICKERS.
Layer 2, the selection. A source gives each ETF a score, a percentile (average ties) among the eligible ETFs. A rule
turns the score into weights: top_fraction (equal weight on the top fraction) or score_capped (weight in proportion
to max(score - 0.5, 0), capped per ETF and per asset group; the rest stays in cash).
Layer 3, the gross. book.gross (at most limits.max_gross) multiplies the weights. A gross above 1 is a loan that pays
the cash return plus BORROW_SPREAD_YEAR.
Timing: the book rebalances every N sessions from the first evaluation session, fills at the open of t+1 and earns
open[t+2] / open[t+1] - 1. Cost: COST_BPS_PER_SIDE on the ETF turnover.
Main measure: sharpe_excess, the Sharpe of the net return minus the cash return. info_ratio_vs_ew is the Sharpe of
the net return minus the ew_monthly benchmark of the run.
Controls (book --controls) run book.controls.reference with one change each: the source shuffled (across groups for
kind groups, else across ETFs), the source STALE_LAG sessions old, a random stress mask with the same stress share,
the switch off, or the stress cost.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from etf66 import config, metrics, portfolio, settings, stage, walkforward
from etf66.bins import cross_section_pct
from etf66.groups import ENSEMBLE_ALL_H_STEM, ENSEMBLE_STEM
from etf66.report import md_table

DEFENSIVE_TICKERS = ["TLT", "IEF", "GLD", "USMV", "UUP"]
COST_BPS_PER_SIDE = 5
STRESS_COST_BPS_PER_SIDE = 20
BORROW_SPREAD_YEAR = 0.005
EQUAL_WEIGHT_REBALANCE = 21   # sessions between rebalances of the equal-weight baselines
STALE_LAG = 126               # control: the source score from this many sessions earlier
RANDOM_SWITCH_BLOCK = 21      # control: the random stress mask changes every this many sessions
CONTROL_SEED = 20261005
OLD_ENSEMBLE_PREFIX = "ens_"  # ensemble files (ens_*) that some run folders hold; a source skips them

MEASURE_COLUMNS = ["sharpe_excess", "sharpe", "cagr", "vol_year", "max_drawdown", "info_ratio_vs_ew",
                   "min_block_sharpe_excess", "turnover_year"]


class BookInputs:
    """The arrays of a run that every book reads."""

    def __init__(self, cfg: dict, run_cfg: dict, run_dir: Path):
        self.cfg, self.run_cfg, self.run_dir = cfg, run_cfg, run_dir
        b = cfg["book"]
        for name in ("context.npz", "context.json", "benchmarks.npz"):
            if not (run_dir / name).exists():
                raise config.ConfigError(f"{run_dir / name} is missing. Run the backtest of the run first.")
        ctx = np.load(run_dir / "context.npz")
        self.r_next, self.cash_next, self.tradable = ctx["r_next"], ctx["cash_next"], ctx["tradable"]
        self.dates = stage.read_dates(run_dir)
        table = stage.read_tickers()
        self.tickers = table["ticker"].tolist()
        self.group_codes = pd.factorize(table.set_index("ticker")["block_name"].reindex(self.tickers).astype(str))[0]
        self.eligible = stage.eligible_mask(self.tradable, self.tickers, settings.CASH_LIKE)
        from etf66.backtest import evaluation_rows
        self.first, self.stop = evaluation_rows(cfg, self.dates, cfg["evaluation"]["common_start"])
        self.periods = [(pd.Timestamp(a), pd.Timestamp(c)) for a, c in cfg["evaluation"]["blocks"]]
        self.cash = np.nan_to_num(self.cash_next)
        bench = np.load(run_dir / "benchmarks.npz")
        self.benchmarks = {k: bench[k] for k in bench.files if bench[k].ndim == 1}
        self.ew_net = self.benchmarks["ew_monthly"]
        defensive = np.isin(self.tickers, DEFENSIVE_TICKERS)
        w = np.where(defensive[None, :], self.tradable, False).astype(float)
        self.w_defensive = w / np.maximum(w.sum(axis=1, keepdims=True), 1)
        self.w_equal = portfolio.equal_weight(self.eligible)
        self.stress = None
        if any(b["switch"]["states"]):
            path = run_dir / "environment" / "stress.npy"
            if not path.exists():
                raise config.ConfigError(f"{path} is missing. Run the environment command of the run first.")
            self.stress = np.load(path)
        self.year = settings.SESSIONS_PER_YEAR


def prediction_files(pred_dir: Path, wanted) -> list:
    """Return the model prediction files of a folder (no ensembles), or the named stems."""
    files = sorted(p for p in pred_dir.glob("*__*.npy") if not p.name.endswith(".tmp.npy")
                   and not p.stem.startswith(OLD_ENSEMBLE_PREFIX) and p.stem not in (ENSEMBLE_STEM, ENSEMBLE_ALL_H_STEM))
    if wanted == "all":
        return files
    stems = {p.stem: p for p in files}
    missing = [w for w in wanted if w not in stems]
    if missing:
        raise config.ConfigError(f"book source files {missing} are not prediction files of {pred_dir}.")
    return [stems[w] for w in wanted]


def predictions_source(spec: dict, inputs: BookInputs) -> np.ndarray:
    """Return the ETF score of a kind predictions source: the mean member percentile, ranked among the eligible."""
    pred_dir = inputs.run_dir / "preds"
    targets = walkforward.target_list(inputs.run_cfg)
    horizons = sorted({h for _, h in targets}) if spec["horizons"] == "all" else [int(h) for h in spec["horizons"]]
    slots = [j for j, (v, h) in enumerate(targets) if v in spec["variants"] and h in horizons]
    if not slots:
        raise config.ConfigError(f"book source: no target of {pred_dir} matches variants {spec['variants']} and "
                                 f"horizons {horizons}.")
    parts = []
    for p in prediction_files(pred_dir, spec["files"]):
        arr = np.load(p, mmap_mode="r")
        for j in slots:
            parts.append(cross_section_pct(np.where(inputs.eligible, arr[j], np.nan).astype(np.float64)))
    return stage.avg_pct(stage.nanmean_stack(parts), inputs.eligible)


def group_score(spec: dict, inputs: BookInputs) -> tuple[np.ndarray, np.ndarray]:
    """Return (group score (sessions, groups), group index per ticker) of a kind groups source."""
    folder = inputs.run_dir / "groups"
    if not (folder / "group_pct.npy").exists():
        raise config.ConfigError(f"{folder} holds no group predictions. Run the groups command of the run first.")
    index = json.loads((folder / "group_pct_index.json").read_text(encoding="utf-8"))
    gmap = np.array(json.loads((folder / "group_map.json").read_text(encoding="utf-8"))["gmap"])
    pct = np.load(folder / "group_pct.npy")
    rows = list(range(len(index))) if spec["horizons"] == "all" else [i for i, e in enumerate(index) if e["h"] in spec["horizons"]]
    if not rows:
        raise config.ConfigError(f"book source: the groups of the run hold no horizon in {spec['horizons']}.")
    return stage.nanmean_stack([pct[i] for i in rows]), gmap


def group_to_etf(score: np.ndarray, gmap: np.ndarray, inputs: BookInputs) -> np.ndarray:
    """Return the ETF score of a group score: each ETF takes its group's score, ranked among the eligible ETFs."""
    mapped = np.where(gmap[None, :] >= 0, score[:, np.clip(gmap, 0, None)], np.nan)
    return stage.avg_pct(mapped, inputs.eligible)


def build_sources(inputs: BookInputs) -> tuple[dict, dict]:
    """Return ({name: ETF score}, {name: (group score, gmap)} for the kind groups sources).

    Kind mean sources come last because they read the other sources.
    """
    specs = inputs.cfg["book"]["sources"]
    scores, groups = {}, {}
    for name, spec in specs.items():
        if spec["kind"] == "predictions":
            scores[name] = predictions_source(spec, inputs)
        elif spec["kind"] == "groups":
            groups[name] = group_score(spec, inputs)
            scores[name] = group_to_etf(*groups[name], inputs)
    for name, spec in specs.items():
        if spec["kind"] == "mean":
            scores[name] = stage.avg_pct(stage.nanmean_stack([scores[o] for o in spec["of"]]), inputs.eligible)
    return scores, groups


def powered_score(score: np.ndarray, power: float) -> np.ndarray:
    """Return 0.5 + sign(s - 0.5) x (2 |s - 0.5|) ^ power / 2: the same percentiles with a sharper or flatter spread."""
    if power == 1.0:
        return score
    d = score - 0.5
    return 0.5 + np.sign(d) * (2.0 * np.abs(d)) ** power / 2.0


def rule_label(rule: dict, power: float) -> str:
    """Return the name of a rule in the book rows: the rule name, plus _p<power> for a score power other than 1."""
    return rule["name"] if power == 1.0 else f"{rule['name']}_p{power:g}"


def parse_rule_label(label: str, rules: list) -> tuple[dict, float]:
    """Return (rule entry, power) of a rule label of the book rows."""
    name, power = label, 1.0
    if "_p" in label and label.rsplit("_p", 1)[0] in {r["name"] for r in rules}:
        name, p = label.rsplit("_p", 1)
        power = float(p)
    return [r for r in rules if r["name"] == name][0], power


def select(score: np.ndarray, rule: dict, inputs: BookInputs, power: float = 1.0) -> np.ndarray:
    """Return the selection weights (sessions, tickers) of one rule. power applies to score_capped only."""
    if rule["name"] == "top_fraction":
        return portfolio.equal_weight(np.nan_to_num(score, nan=-1.0) >= 1.0 - float(rule["fraction"]))
    return portfolio.score_capped(powered_score(score, power), inputs.group_codes, float(rule["max_weight"]),
                                  float(rule["max_group_weight"]), int(rule["iterations"]))


def run_book(w_sel: np.ndarray, use_switch: bool, gross: float, every: int, inputs: BookInputs,
             stress=None, bps=None) -> dict:
    """Return the measures of one book. stress replaces the run's stress mask; bps replaces the base cost."""
    b = inputs.cfg["book"]
    bps = float(COST_BPS_PER_SIDE) if bps is None else float(bps)
    stress = inputs.stress if stress is None else stress
    share = float(b["switch"]["defensive_share"]) if use_switch else 0.0
    d = (np.where(stress, share, 0.0) if use_switch else np.zeros(len(w_sel)))[:, None]
    w = gross * ((1 - d) * w_sel + d * inputs.w_defensive)
    w = np.where(inputs.tradable, w, 0.0)
    reb = portfolio.rebalance_mask(len(inputs.dates), inputs.first, every)
    g, tv = portfolio.simulate_levered(w, reb, inputs.r_next, inputs.cash_next, BORROW_SPREAD_YEAR / inputs.year)
    first, stop = inputs.first, inputs.stop
    net = (g - tv * bps / 1e4)[first:stop]
    ex = net - inputs.cash[first:stop]
    ps = metrics.period_sharpes(ex, inputs.dates[first:stop], inputs.periods)
    return {"sharpe_excess": metrics.sharpe(ex), "sharpe": metrics.sharpe(net), "cagr": metrics.cagr(net),
            "vol_year": float(np.std(net, ddof=1) * math.sqrt(inputs.year)), "max_drawdown": metrics.max_drawdown(net),
            "info_ratio_vs_ew": metrics.sharpe(net - inputs.ew_net[first:stop]),
            "median_block_sharpe_excess": float(np.nanmedian(ps)), "min_block_sharpe_excess": float(np.nanmin(ps)),
            **{f"block{i + 1}_sharpe_excess": v for i, v in enumerate(ps)},
            "turnover_year": float(tv[first:stop].sum() * inputs.year / len(net)),
            "skew": float(pd.Series(ex).skew()), "kurtosis": float(pd.Series(ex).kurt() + 3.0), "n_sessions": len(ex)}


def book_grid(scores: dict, inputs: BookInputs) -> pd.DataFrame:
    """Return one row per book of the grid: sources x rules x switch states x gross x rebalance intervals."""
    b = inputs.cfg["book"]
    rows = []
    for sname, score in scores.items():
        for rule in config.enabled_entries(b["selection"]):
            powers = [float(p) for p in rule.get("score_powers", [1])] if rule["name"] == "score_capped" else [1.0]
            for power in powers:
                w_sel = select(score, rule, inputs, power)
                for sw in b["switch"]["states"]:
                    for gross in b["gross"]:
                        for every in b["rebalance_every_sessions"]:
                            rows.append({"source": sname, "rule": rule_label(rule, power), "switch": bool(sw),
                                         "gross": float(gross), "rebalance_every_sessions": int(every),
                                         **run_book(w_sel, bool(sw), float(gross), int(every), inputs)})
    out = pd.DataFrame(rows)
    sr = out["sharpe_excess"].to_numpy(float) / math.sqrt(inputs.year)
    ok = np.isfinite(sr)
    var = float(np.var(sr[ok], ddof=1)) if ok.sum() > 1 else 0.0
    out["dsr"] = metrics.deflated_sharpe(sr, out["n_sessions"].to_numpy(float), out["skew"].to_numpy(float),
                                         out["kurtosis"].to_numpy(float), int(ok.sum()), var)
    out["dsr_trials"] = int(ok.sum())
    return out


def baselines(inputs: BookInputs) -> pd.DataFrame:
    """Return the equal-weight books (switch states x gross) and the fixed benchmarks of the run."""
    b = inputs.cfg["book"]
    every = EQUAL_WEIGHT_REBALANCE
    rows = [{"book": "equal_weight", "switch": bool(sw), "gross": float(g), "rebalance_every_sessions": every,
             **run_book(inputs.w_equal, bool(sw), float(g), every, inputs)}
            for sw in b["switch"]["states"] for g in b["gross"]]
    first, stop = inputs.first, inputs.stop
    for name, net in inputs.benchmarks.items():
        r = net[first:stop]
        ex = r - inputs.cash[first:stop]
        rows.append({"book": f"benchmark {name}", "sharpe_excess": metrics.sharpe(ex), "sharpe": metrics.sharpe(r),
                     "cagr": metrics.cagr(r), "max_drawdown": metrics.max_drawdown(r)})
    return pd.DataFrame(rows)


def shuffle_rows(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return x with the finite values of each row permuted."""
    out = x.copy()
    for t in range(len(x)):
        ok = np.flatnonzero(np.isfinite(x[t]))
        out[t, ok] = x[t, ok][rng.permutation(len(ok))]
    return out


def random_stress(stress: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random stress mask with the same share of stress sessions, constant over blocks of `block` sessions."""
    n = len(stress)
    blocks = rng.random((n + block - 1) // block) < stress.mean()
    return np.repeat(blocks, block)[:n]


def controls(scores: dict, groups: dict, inputs: BookInputs) -> pd.DataFrame:
    """Return the control rows of the reference book."""
    b = inputs.cfg["book"]
    c, ref = b["controls"], b["controls"]["reference"]
    rule, power = parse_rule_label(ref["rule"], config.enabled_entries(b["selection"]))
    sw, gross, every = bool(ref["switch"]), float(ref["gross"]), int(ref["rebalance_every_sessions"])
    rng = np.random.default_rng(CONTROL_SEED)
    name = ref["source"]

    def book_of(score, **kw):
        return run_book(select(score, rule, inputs, power), sw, gross, every, inputs, **kw)

    def with_score(fn):
        """Apply fn to the group score (kind groups) or the ETF score, and return the ETF score."""
        if name in groups:
            g, gmap = groups[name]
            return group_to_etf(fn(g), gmap, inputs)
        return stage.avg_pct(fn(scores[name]), inputs.eligible)

    rows = [{"case": "real", **book_of(scores[name])},
            {"case": "real, stress cost", **book_of(scores[name], bps=STRESS_COST_BPS_PER_SIDE)},
            {"case": "real, switch off", **run_book(select(scores[name], rule, inputs, power), False, gross, every, inputs)}]
    lag = STALE_LAG
    rows.append({"case": f"stale source ({lag})",
                 **book_of(with_score(lambda x: np.vstack([np.full((lag, x.shape[1]), np.nan), x[:-lag]])))})
    for d in range(int(c["shuffle_draws"])):
        rows.append({"case": f"shuffled source {d}", **book_of(with_score(lambda x: shuffle_rows(x, rng)))})
    if sw and inputs.stress is not None:
        for d in range(int(c["random_switch_draws"])):
            mask = random_stress(inputs.stress, RANDOM_SWITCH_BLOCK, rng)
            rows.append({"case": f"random switch {d}", **book_of(scores[name], stress=mask)})
    out = pd.DataFrame(rows)
    out["family"] = out["case"].str.replace(r" \d+$", "", regex=True)
    return out


def medians_by(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Return the median of the main measures grouped by one column, with the book count."""
    g = frame.groupby(by)
    out = g[["sharpe_excess", "cagr", "max_drawdown", "info_ratio_vs_ew", "min_block_sharpe_excess", "dsr"]].median()
    out["books"] = g.size()
    return out


def write_report(folder: Path, inputs: BookInputs, books: pd.DataFrame, base: pd.DataFrame,
                 ctrl: pd.DataFrame | None) -> None:
    """Write runs/<run>/book/REPORT.md."""
    cfg = inputs.cfg
    keys = ["source", "rule", "switch", "gross", "rebalance_every_sessions"]
    blocks = [c for c in books.columns if c.startswith("block") and c.endswith("_sharpe_excess")]
    lines = [
        f"# Book of the run {inputs.run_dir.name}",
        "",
        "## 1. Facts",
        "",
        f"- Evaluation: {inputs.dates[inputs.first].date()} to {inputs.dates[inputs.stop - 1].date()} "
        f"({inputs.stop - inputs.first} sessions). Dev data only.",
        f"- Books: {len(books)}. Sources: {list(cfg['book']['sources'])}. Cost: {COST_BPS_PER_SIDE} bps "
        f"per side. A loan pays the cash return plus {BORROW_SPREAD_YEAR:.2%} a year.",
        "- sharpe_excess = Sharpe of the net return minus the cash return. info_ratio_vs_ew = Sharpe of the net return "
        "minus the equal-weight benchmark. dsr uses the books of this grid and the variance of their Sharpes.",
        "- The settings of this stage are in settings.yaml in this folder.",
        "",
        "## 2. Baselines",
        "",
        md_table(base[[c for c in ["book", "switch", "gross", "rebalance_every_sessions", *MEASURE_COLUMNS]
                       if c in base.columns]], index=False),
        "",
        "## 3. Medians by one choice at a time",
        "",
    ]
    for by in keys:
        lines += [f"### By {by}", "", md_table(medians_by(books, by)), ""]
    lines += ["## 4. All books, sorted by sharpe_excess", "",
              md_table(books.sort_values("sharpe_excess", ascending=False)[keys + MEASURE_COLUMNS + blocks + ["dsr"]],
                       index=False), ""]
    if ctrl is not None:
        cols = ["sharpe_excess", "cagr", "max_drawdown", "info_ratio_vs_ew", "min_block_sharpe_excess"]
        agg = ctrl.groupby("family", sort=False)[cols].agg(["median", "max"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        lines += ["## 5. Controls on the reference book", "",
                  f"Reference: {cfg['book']['controls']['reference']}. Each control changes one layer.", "",
                  md_table(agg), ""]
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_book_stage(cfg: dict, run_cfg: dict, run_dir: Path, with_controls: bool, say) -> Path:
    """Run the book grid of a run (and the controls when asked), write the tables and the report."""
    folder = run_dir / "book"
    folder.mkdir(parents=True, exist_ok=True)
    inputs = BookInputs(cfg, run_cfg, run_dir)
    scores, groups = build_sources(inputs)
    say(f"sources ready: {list(scores)}")
    books = book_grid(scores, inputs)
    books.to_parquet(folder / "books.parquet", index=False)
    base = baselines(inputs)
    base.to_csv(folder / "baselines.csv", index=False)
    say(f"{len(books)} books done")
    ctrl = None
    if with_controls:
        ctrl = controls(scores, groups, inputs)
        ctrl.to_csv(folder / "controls.csv", index=False)
        say(f"{len(ctrl)} control books done")
    write_report(folder, inputs, books, base, ctrl)
    stage.write_stage_settings(folder, cfg, "book", run_cfg)
    return folder
