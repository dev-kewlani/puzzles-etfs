"""The rules that the settings of a run must pass before any work starts. They need no data.

check_rules() collects every broken rule and raises one ConfigError that lists them all.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from etf66 import settings
from etf66.labels import is_label_variant

MODEL_KEYS = {"name", "kind", "alpha", "enabled", "params"}
SOURCE_KEYS = {"predictions": {"kind", "files", "variants", "horizons"}, "groups": {"kind", "horizons"},
               "mean": {"kind", "of"}}
SELECTION_KEYS = {"top_fraction": {"name", "enabled", "fraction"},
                  "score_capped": {"name", "enabled", "max_weight", "max_group_weight", "iterations", "score_powers"}}
STRESS_INPUTS = ("dd", "vol", "corr")


class ConfigError(ValueError):
    """A setting is unknown, read-only, out of range, or in conflict with another setting."""


def enabled_entries(entries: list) -> list:
    """Return the entries of a named list with enabled: true, in file order."""
    return [e for e in entries if e.get("enabled", True)]


def is_int_at_least(x, low: int) -> bool:
    """Return True when x is an integer (not a bool) of at least `low`."""
    return isinstance(x, int) and not isinstance(x, bool) and x >= low


def is_number(x) -> bool:
    """Return True when x is an int or a float (not a bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_folder_name(x) -> bool:
    """Return True when x is a non-empty text with no path separator."""
    return isinstance(x, str) and x != "" and "/" not in x and "\\" not in x


def is_series_name(x) -> bool:
    """Return True when x has the form <model>__<window> of a prediction file stem."""
    return isinstance(x, str) and len(x.split("__")) == 2 and all(x.split("__"))


def universe_size() -> int:
    """Return the number of ETFs in the universe file."""
    return len(pd.read_csv(settings.UNIVERSE_FILE, dtype=str))


def check_rules(cfg: dict) -> None:
    """Raise ConfigError with every broken rule."""
    errors = []

    def need(ok: bool, msg: str) -> None:
        if not ok:
            errors.append(msg)

    fence = dt.date.fromisoformat(str(cfg["data"]["fence_date"]))
    need(fence == settings.FENCE_DATE, f"data.fence_date is {fence}. The code fixes it at {settings.FENCE_DATE}.")
    need(cfg["paths"]["unlock_file"] == settings.UNLOCK_FILE.name,
         f"paths.unlock_file must be {settings.UNLOCK_FILE.name}.")
    ctx = cfg["context"]
    need(is_int_at_least(ctx["market_lag_sessions"], 1) and is_int_at_least(ctx["cboe_lag_sessions"], 1),
         "A lag below 1 lets a feature read a close after the ETF close.")
    check_label_rules(cfg["labels"], need)
    check_model_rules(cfg, need)
    check_walkforward_rules(cfg, need)
    need(is_int_at_least(cfg["benchmarks"]["random_draws"], 1), "benchmarks.random_draws must be 1 or more.")
    check_evaluation_rules(cfg["evaluation"], need)
    for stage in ("groups", "environment"):
        need(is_int_at_least(cfg[stage]["purge_extra_sessions"], 0),
             f"{stage}.purge_extra_sessions below 0 lets training labels read data after the retrain session.")
    source = cfg["groups"]["source_run"]
    need(source is None or is_folder_name(source), "groups.source_run must be null or a run folder name.")
    st = cfg["environment"]["stress"]
    need(st["input"] in STRESS_INPUTS, f"environment.stress.input must be one of {list(STRESS_INPUTS)}.")
    need(is_number(st["threshold"]) and 0 < st["threshold"] <= 1, "environment.stress.threshold must be in (0, 1].")
    need(is_number(cfg["limits"]["max_gross"]) and cfg["limits"]["max_gross"] >= 1, "limits.max_gross must be 1 or more.")
    check_book_rules(cfg["book"], cfg["limits"]["max_gross"], need)
    lt = cfg["compute"]["lightgbm_threads"]
    need(lt is None or is_int_at_least(lt, 1), "compute.lightgbm_threads must be null or 1 or more.")
    need(is_int_at_least(cfg["compute"]["workers"], 1), "compute.workers must be 1 or more.")
    if errors:
        raise ConfigError("The settings break these rules:\n- " + "\n- ".join(errors))


def check_label_rules(lab: dict, need) -> None:
    """Add the errors of the label settings."""
    horizons = lab["horizons"]
    need(len(horizons) > 0 and all(is_int_at_least(h, 1) for h in horizons) and len(set(horizons)) == len(horizons),
         "labels.horizons must be unique integers of 1 or more.")
    need(is_int_at_least(lab["min_vol_window"], 2), "labels.min_vol_window must be 2 or more (a std needs 2 returns).")
    variants = lab["variants"]
    need(len(variants) > 0 and all(is_label_variant(v) for v in variants) and len(set(variants)) == len(variants),
         "labels.variants holds an unknown or repeated variant (see labels.py).")


def check_model_rules(cfg: dict, need) -> None:
    """Add the errors of the model settings."""
    from etf66.walkforward import MIN_ROWS
    names = [m.get("name") for m in cfg["models"]]
    need(len(set(names)) == len(names), "Model names must be unique.")
    need(len(enabled_entries(cfg["models"])) > 0, "No model is enabled.")
    for m in cfg["models"]:
        name = m.get("name")
        need(set(m) <= MODEL_KEYS, f"Model {name}: unknown keys {sorted(set(m) - MODEL_KEYS)}.")
        need(m.get("kind") in ("ridge", "lightgbm"), f"Model {name}: kind must be ridge or lightgbm.")
        if m.get("kind") == "ridge":
            need(is_number(m.get("alpha")) and m["alpha"] > 0, f"Model {name}: alpha must be above 0.")
        if m.get("kind") == "lightgbm":
            params = m.get("params") or {}
            need(not {"objective", "num_threads"} & set(params),
                 f"Model {name}: params must not hold objective or num_threads. The code sets them.")
            need(int(params.get("min_data_in_leaf", 20)) <= MIN_ROWS,
                 f"Model {name}: min_data_in_leaf is above walkforward.MIN_ROWS ({MIN_ROWS}).")
    ridge = cfg["model_defaults"]["ridge"]
    need(ridge["device"] in ("auto", "cpu", "gpu"), "model_defaults.ridge.device must be auto, cpu or gpu.")
    need(ridge["cpu_precision"] in ("float32", "float64"), "model_defaults.ridge.cpu_precision must be float32 or float64.")


def check_walkforward_rules(cfg: dict, need) -> None:
    """Add the errors of the walkforward section."""
    from etf66.walkforward import MIN_ROWS
    wf = cfg["walkforward"]
    need(is_int_at_least(wf["purge_extra_sessions"], 0),
         "walkforward.purge_extra_sessions below 0 lets training labels read data after the retrain session.")
    need(is_int_at_least(wf["retrain_every_months"], 1), "walkforward.retrain_every_months must be 1 or more.")
    need(len(wf["windows"]) > 0, "walkforward.windows is empty.")
    for w in wf["windows"]:
        need(w == "expanding" or w in wf["window_sessions"], f"walkforward window {w} has no window_sessions entry.")
    n_etfs = universe_size()
    for w, size in wf["window_sessions"].items():
        need(is_int_at_least(size, 1) and size * n_etfs >= MIN_ROWS, f"Window {w} can never reach {MIN_ROWS} rows.")


def check_evaluation_rules(ev: dict, need) -> None:
    """Add the errors of the evaluation dates: every date of the dev evaluation is before the fence."""
    fence = pd.Timestamp(settings.FENCE_DATE)
    start = pd.Timestamp(ev["common_start"])
    need(start < fence, "evaluation.common_start must be before the fence date.")
    end = pd.Timestamp(ev["end"]) if ev["end"] is not None else None
    need(end is None or start < end < fence, "evaluation.end is outside the dev period.")
    blocks = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in ev["blocks"]]
    need(all(a <= b for a, b in blocks), "evaluation.blocks: a block ends before it starts.")
    need(all(blocks[i][1] < blocks[i + 1][0] for i in range(len(blocks) - 1)), "evaluation.blocks overlap or are out of order.")
    need(all(a >= start and b < fence for a, b in blocks), "evaluation.blocks must be inside [common_start, fence_date).")


def check_book_rules(b: dict, max_gross: float, need) -> None:
    """Add the errors of the book settings."""
    sources = b["sources"]
    need(len(sources) > 0 and all(is_folder_name(n) for n in sources), "book.sources needs at least one named source.")
    for name, s in sources.items():
        kind = s.get("kind") if isinstance(s, dict) else None
        need(kind in SOURCE_KEYS and set(s) == SOURCE_KEYS.get(kind, set()),
             f"book.sources.{name}: kind {kind} needs exactly the keys {sorted(SOURCE_KEYS.get(kind, []))}.")
        if kind not in SOURCE_KEYS or set(s) != SOURCE_KEYS[kind]:
            continue
        if kind == "predictions":
            need(s["files"] == "all" or (isinstance(s["files"], list) and len(s["files"]) > 0
                                         and all(is_series_name(f) for f in s["files"])),
                 f"book.sources.{name}.files must be all or a list of <model>__<window>.")
            need(isinstance(s["variants"], list) and len(s["variants"]) > 0 and all(is_label_variant(v) for v in s["variants"]),
                 f"book.sources.{name}.variants must be a list of label variants.")
        if kind == "mean":
            need(isinstance(s["of"], list) and len(s["of"]) > 0 and all(o in sources and o != name for o in s["of"])
                 and all(sources[o].get("kind") != "mean" for o in s["of"] if o in sources),
                 f"book.sources.{name}.of must name other sources of kind predictions or groups.")
        if kind in ("predictions", "groups"):
            hs = s["horizons"]
            need(hs == "all" or (isinstance(hs, list) and len(hs) > 0 and all(is_int_at_least(h, 1) for h in hs)),
                 f"book.sources.{name}.horizons must be all or a list of integers of 1 or more.")
    sel = b["selection"]
    need(len(enabled_entries([r for r in sel if isinstance(r, dict)])) > 0, "book.selection has no enabled rule.")
    for r in sel:
        name = r.get("name") if isinstance(r, dict) else None
        keys = set(r) | ({"score_powers"} if name == "score_capped" else set())   # older run settings lack it
        need(name in SELECTION_KEYS and keys == SELECTION_KEYS.get(name, set()),
             f"book.selection.{name} needs exactly the keys {sorted(SELECTION_KEYS.get(name, []))}.")
        if name == "top_fraction" and set(r) == SELECTION_KEYS[name]:
            need(0 < r["fraction"] <= 1, "book.selection.top_fraction.fraction must be in (0, 1].")
        if name == "score_capped" and keys == SELECTION_KEYS[name]:
            need(0 < r["max_weight"] <= r["max_group_weight"] <= 1 and is_int_at_least(r["iterations"], 1),
                 "book.selection.score_capped caps are out of range.")
            powers = r.get("score_powers", [1])
            need(isinstance(powers, list) and len(powers) > 0 and len(set(powers)) == len(powers)
                 and all(is_number(x) and x > 0 for x in powers),
                 "book.selection.score_capped.score_powers must be a list of distinct numbers above 0.")
    sw = b["switch"]
    need(len(sw["states"]) > 0 and all(isinstance(x, bool) for x in sw["states"]) and len(set(sw["states"])) == len(sw["states"]),
         "book.switch.states must be a list of distinct true or false values.")
    need(is_number(sw["defensive_share"]) and 0 <= sw["defensive_share"] <= 1, "book.switch.defensive_share must be in [0, 1].")
    need(len(b["gross"]) > 0 and all(is_number(g) and 0 < g <= max_gross for g in b["gross"]),
         f"book.gross values must be above 0 and at most limits.max_gross ({max_gross}).")
    need(len(b["rebalance_every_sessions"]) > 0 and all(is_int_at_least(n, 1) for n in b["rebalance_every_sessions"]),
         "book.rebalance_every_sessions must hold integers of 1 or more.")
    c = b["controls"]
    ref = c["reference"]
    names = [r.get("name") for r in enabled_entries([r for r in sel if isinstance(r, dict)])]
    # A score power p other than 1 adds the rule name score_capped_p<p>.
    for r in enabled_entries([r for r in sel if isinstance(r, dict)]):
        if r.get("name") == "score_capped" and isinstance(r.get("score_powers"), list):
            names += [f"score_capped_p{float(p):g}" for p in r["score_powers"] if is_number(p) and p != 1]
    need(ref["source"] in sources and ref["rule"] in names and ref["switch"] in sw["states"]
         and ref["gross"] in b["gross"] and ref["rebalance_every_sessions"] in b["rebalance_every_sessions"],
         "book.controls.reference must be a book inside the grid of the book section.")
    need(is_int_at_least(c["shuffle_draws"], 0) and is_int_at_least(c["random_switch_draws"], 0),
         "book.controls draw counts must be 0 or more.")
