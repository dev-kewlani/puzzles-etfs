"""Load, merge and save the settings of a run. The rule checks are in etf66/rules.py.

config.yaml at the repo root holds the settings. A run changes them in two ways, applied in this order:
  1. --overlay FILE   a YAML file with the keys to change. Dicts merge key by key. Lists replace the base list,
                      except the named lists (NAMED_LISTS), which merge entry by entry on the key `name`.
  2. --set PATH=VALUE one key, for example --set labels.horizons=[5,21] or --set models.lgbm_light.enabled=false.
                      VALUE is read as YAML. A model or a rule is addressed by its name.
The loader refuses a key that config.yaml does not have, except inside the open maps. It refuses a change to a
read-only key. Then it runs rules.check_rules(). A run saves the merged result in its folder.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import yaml

from etf66 import settings
from etf66.rules import ConfigError, check_rules, enabled_entries  # noqa: F401 - callers use config.*

BASE_FILE = settings.REPO_ROOT / "config.yaml"
NAMED_LISTS = {("models",), ("book", "selection")}
OPEN_MAPS = {("walkforward", "window_sessions"), ("book", "sources")}   # the user names the keys of these maps
READ_ONLY = {("data", "fence_date"), ("paths", "unlock_file")}    # no overlay or --set can move the holdout
NOT_HASHED = (("compute",), ("run", "notes"))


def read_yaml(path: Path) -> dict:
    """Return the content of a YAML file as a dict (an empty file gives an empty dict)."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def is_open(path: tuple) -> bool:
    """Return True when the dict at `path` accepts keys that the base file does not have."""
    return path in OPEN_MAPS or (len(path) > 0 and path[-1] == "params")


def merge(base: dict, over: dict, path: tuple = ()) -> dict:
    """Return a copy of `base` with `over` merged in. Refuse unknown keys and changes to read-only keys."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        here = path + (key,)
        if key not in out and not is_open(path):
            raise ConfigError(f"unknown key {'.'.join(here)}")
        if here in READ_ONLY and value != out.get(key):
            raise ConfigError(f"{'.'.join(here)} is read-only. A run cannot change it.")
        if here in NAMED_LISTS:
            out[key] = merge_named_list(out[key], value, here)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value, here)
        else:
            out[key] = copy.deepcopy(value)
    return out


def deep_update(base: dict, over: dict) -> dict:
    """Return a copy of `base` with `over` merged in key by key (no key check; the rules check the result)."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def merge_named_list(base: list, over: list, path: tuple) -> list:
    """Return the base list of named entries with the overlay entries merged in by `name`. New names append."""
    out = copy.deepcopy(base)
    names = [e["name"] for e in out]
    for entry in over:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ConfigError(f"every entry of {'.'.join(path)} needs a name")
        if entry["name"] in names:
            i = names.index(entry["name"])
            out[i] = deep_update(out[i], entry)
        else:
            out.append(copy.deepcopy(entry))
            names.append(entry["name"])
    return out


def parse_set(text: str) -> tuple[list, object]:
    """Return (key path, value) from one --set argument of the form a.b.c=VALUE."""
    if "=" not in text:
        raise ConfigError(f"--set needs the form key.path=value, not {text!r}")
    key, raw = text.split("=", 1)
    return key.strip().split("."), yaml.safe_load(raw)


def apply_set(cfg: dict, text: str) -> dict:
    """Return a copy of cfg with one --set value applied. A model or a rule is addressed by its name."""
    keys, value = parse_set(text)
    out = copy.deepcopy(cfg)
    node, path = out, ()
    for i, key in enumerate(keys):
        last = i == len(keys) - 1
        if isinstance(node, list):
            match = [e for e in node if isinstance(e, dict) and e.get("name") == key]
            if not match:
                raise ConfigError(f"{'.'.join(path)} has no entry named {key}")
            node, path = match[0], path + (key,)
            if last:
                raise ConfigError(f"--set {text}: name a field of {key}, not the whole entry")
            continue
        if key not in node and not is_open(path):
            raise ConfigError(f"unknown key {'.'.join(path + (key,))}")
        here = path + (key,)
        if last:
            if here in READ_ONLY and value != node.get(key):
                raise ConfigError(f"{'.'.join(here)} is read-only. A run cannot change it.")
            node[key] = value
        else:
            node = node.setdefault(key, {})
            path = here
    return out


def load_config(overlays=(), sets=(), base_file: Path | None = None) -> dict:
    """Return the checked settings: the base file, then each overlay file, then each --set value."""
    cfg = read_yaml(base_file or BASE_FILE)
    for p in overlays:
        cfg = merge(cfg, read_yaml(Path(p)))
    for text in sets:
        cfg = apply_set(cfg, text)
    check_rules(cfg)
    return cfg


def enabled_models(cfg: dict) -> list:
    """Return the model specs with enabled: true, in file order."""
    return enabled_entries(cfg["models"])


def hashable(cfg: dict) -> dict:
    """Return a copy of cfg without the keys that do not change a result (compute, run.notes)."""
    out = copy.deepcopy(cfg)
    for path in NOT_HASHED:
        node = out
        for key in path[:-1]:
            node = node.get(key, {})
        node.pop(path[-1], None)
    return out


def config_hash(cfg: dict) -> str:
    """Return the hash of the settings that change results."""
    return settings.text_hash(yaml.safe_dump(hashable(cfg), sort_keys=True))


def data_fingerprint() -> str:
    """Return a hash of the snapshot manifest and the ticker list ("none" when the snapshot does not exist)."""
    parts = []
    for name in ("MANIFEST.json", "tickers.csv"):
        p = settings.DATA_DIR / name
        parts.append(settings.file_hash(p) if p.exists() else "none")
    return settings.text_hash("|".join(parts))


# The source files whose constants change the cached arrays. An edit in one of them gives a new cache file.
FEATURE_SOURCES = ("etf66/features", "etf66/ops.py", "etf66/universe.py", "etf66/data.py", "etf66/settings.py")
LABEL_SOURCES = ("etf66/labels.py", "etf66/ops.py", "etf66/universe.py", "etf66/data.py", "etf66/settings.py")


def features_cache_key(cfg: dict) -> str:
    """Return the cache key of the feature array: the context settings, the feature source files and the data."""
    keys = {"context": cfg["context"], "code": settings.source_hash(*FEATURE_SOURCES), "data": data_fingerprint()}
    return settings.text_hash(json.dumps(keys, sort_keys=True, default=str))


def labels_cache_key(cfg: dict) -> str:
    """Return the cache key of the label array: the label settings, the label source files and the data."""
    keys = {"labels": cfg["labels"], "code": settings.source_hash(*LABEL_SOURCES), "data": data_fingerprint()}
    return settings.text_hash(json.dumps(keys, sort_keys=True, default=str))


def drop_old_keys(cfg: dict, base: dict, path: tuple = ()) -> dict:
    """Return cfg without the keys that config.yaml does not have.

    A run folder can hold keys from an earlier layout of the settings. Their values equal the constants in the code.
    A model entry also loses a scope of global, any scope when the entry is disabled, and empty feature filters. A book
    source loses empty study paths.
    """
    out = {}
    for key, value in cfg.items():
        here = path + (key,)
        if key not in base and not is_open(path):
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict) and not is_open(here):
            out[key] = drop_old_keys(value, base[key], here)
        else:
            out[key] = copy.deepcopy(value)
    for m in out.get("models", []):
        if m.get("scope", "global") == "global" or not m.get("enabled", True):
            m.pop("scope", None)
        for key in ("feature_groups", "features_include", "features_exclude"):
            if key in m and m[key] is None:
                del m[key]
    for spec in out.get("book", {}).get("sources", {}).values():
        for key in ("pred_dir", "targets_file"):
            if key in spec and spec[key] is None:
                del spec[key]
    return out


def write_run_config(run_dir: Path, cfg: dict, overlays=(), sets=()) -> None:
    """Save the merged settings, a copy of the base file and each overlay, and the --set lines in a run folder.

    Refuse a run folder that holds different merged settings (a re-used --run-id must hold the same settings).
    """
    merged = run_dir / "config.merged.yaml"
    if merged.exists() and hashable(read_run_config(run_dir)) != hashable(cfg):
        raise ConfigError(f"{run_dir} holds different merged settings. Use a new --run-id.")
    merged.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    shutil.copy(BASE_FILE, run_dir / "config.base.yaml")
    for i, p in enumerate(overlays):
        shutil.copy(p, run_dir / f"overlay_{i + 1}_{Path(p).name}")
    (run_dir / "overrides.txt").write_text("".join(f"{s}\n" for s in sets), encoding="utf-8")


def read_run_config(run_dir: Path) -> dict:
    """Return the frozen settings of a run (config.merged.yaml, or experiments.yaml for the first runs).

    Keys that config.yaml added after the run take their config.yaml values. Keys that config.yaml dropped since
    the run are dropped here.
    """
    base = read_yaml(BASE_FILE)
    for name in ("config.merged.yaml", "experiments.yaml"):
        path = run_dir / name
        if path.exists():
            cfg = deep_update(base, drop_old_keys(read_yaml(path), base))
            check_rules(cfg)
            return cfg
    raise ConfigError(f"{run_dir} holds no config.merged.yaml and no experiments.yaml.")
