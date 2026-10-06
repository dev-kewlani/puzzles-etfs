"""The settings loader. Merge order, named lists, unknown and read-only keys, the rules on the dials, the hashes and
cache keys, and old run folders. Also the module constants: windows that one family reads from another must agree."""
import pandas as pd
import pytest
import yaml

from etf66 import benchmarks, config, settings
from etf66.features import bars, cross, macro, price, state


def write(tmp_path, name, data):
    """Write a YAML overlay file and return its path."""
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_base_file_passes_the_rules(cfg):
    # The fence date and the 4 models that config.yaml enables.
    assert cfg["data"]["fence_date"] == "2021-01-01"
    assert [m["name"] for m in config.enabled_models(cfg)] == ["ridge_a001", "ridge_a01", "ridge_a1", "lgbm_light"]


def test_unknown_key_is_refused(tmp_path):
    with pytest.raises(config.ConfigError, match="unknown key"):
        config.load_config(overlays=[write(tmp_path, "o.yaml", {"labels": {"horizonz": [5]}})])
    with pytest.raises(config.ConfigError, match="unknown key"):
        config.load_config(sets=["bins.n_bins=5"])


def test_overlay_then_set_order(tmp_path):
    o1 = write(tmp_path, "o1.yaml", {"walkforward": {"retrain_every_months": 3, "purge_extra_sessions": 2}})
    o2 = write(tmp_path, "o2.yaml", {"walkforward": {"retrain_every_months": 6}})
    # A later overlay wins over an earlier one, and --set wins over every overlay. A key that only o1 sets stays.
    c = config.load_config(overlays=[o1, o2], sets=["walkforward.retrain_every_months=12"])
    assert c["walkforward"]["retrain_every_months"] == 12
    assert c["walkforward"]["purge_extra_sessions"] == 2


def test_lists_replace_and_models_merge_by_name(tmp_path):
    o = write(tmp_path, "o.yaml", {"labels": {"horizons": [5, 21]},
                                   "models": [{"name": "lgbm_light", "params": {"num_boost_round": 30}},
                                              {"name": "ridge_a001", "enabled": False}]})
    c = config.load_config(overlays=[o])
    # A list replaces the base list. A model entry merges into the base model of the same name, so max_bin stays.
    assert c["labels"]["horizons"] == [5, 21]
    lgbm = [m for m in c["models"] if m["name"] == "lgbm_light"][0]
    assert lgbm["params"]["num_boost_round"] == 30 and lgbm["params"]["max_bin"] == 31
    assert "ridge_a001" not in [m["name"] for m in config.enabled_models(c)]


def test_set_addresses_models_by_name():
    c = config.load_config(sets=["models.ridge_a1.alpha=2.5", "models.lgbm_light.enabled=false"])
    assert [m for m in c["models"] if m["name"] == "ridge_a1"][0]["alpha"] == 2.5
    assert "lgbm_light" not in [m["name"] for m in config.enabled_models(c)]


def test_read_only_keys(tmp_path):
    with pytest.raises(config.ConfigError, match="read-only"):
        config.load_config(sets=["paths.unlock_file=MY_UNLOCK.yaml"])


@pytest.mark.parametrize("setting, message", [
    ("labels.min_vol_window=1", "min_vol_window"),
    ("labels.horizons=[5,5]", "horizons"),
    ("labels.variants=[vol_scaled,wrong]", "variants"),
    ("labels.variants=[rank,payoff_lx]", "variants"),
    ("walkforward.purge_extra_sessions=-1", "purge_extra_sessions"),
    ("walkforward.windows=[rolling_9y]", "rolling_9y"),
    ("walkforward.retrain_every_months=0", "retrain_every_months"),
    ("walkforward.window_sessions.rolling_1y=10", "can never reach"),
    ("context.market_lag_sessions=0", "lag below 1"),
    ("evaluation.common_start=2021-06-01", "common_start"),
    ("evaluation.end=2009-01-01", "evaluation.end"),
    ("evaluation.blocks=[[2019-01-01,2021-03-31]]", "blocks"),
    ("models.lgbm_light.params.objective=huber", "objective"),
    ("models.lgbm_light.params.min_data_in_leaf=5000", "min_data_in_leaf"),
    ("model_defaults.ridge.device=tpu", "device"),
    ("groups.purge_extra_sessions=-1", "groups.purge_extra_sessions"),
    ("environment.stress.input=ret", "stress.input"),
    ("environment.stress.threshold=1.5", "threshold"),
    ("book.gross=[3.0]", "max_gross"),
    ("limits.max_gross=0.5", "limits.max_gross"),
    ("book.switch.defensive_share=1.5", "defensive_share"),
    ("book.selection.top_fraction.fraction=0", "fraction"),
    ("book.sources.mix.of=[ret_ens,nope]", "book.sources.mix"),
    ("book.controls.reference.source=nope", "reference"),
    ("book.sources.ret_ens.variants=[wrong]", "variants"),
])
def test_rules(setting, message):
    with pytest.raises(config.ConfigError, match=message):
        config.load_config(sets=[setting])


def test_config_hash_ignores_compute_and_notes(cfg):
    # The worker count and the notes do not change a result. The retrain schedule does.
    other = config.load_config(sets=["compute.workers=9", "run.notes=hello"])
    assert config.config_hash(other) == config.config_hash(cfg)
    assert config.config_hash(config.load_config(sets=["walkforward.retrain_every_months=2"])) != config.config_hash(cfg)


def test_cache_keys_follow_their_settings(cfg):
    # A cboe lag changes the features only. min_vol_window changes the labels only. The retrain schedule changes
    # neither file.
    f0, l0 = config.features_cache_key(cfg), config.labels_cache_key(cfg)
    c = config.load_config(sets=["context.cboe_lag_sessions=2"])
    assert config.features_cache_key(c) != f0 and config.labels_cache_key(c) == l0
    c = config.load_config(sets=["labels.min_vol_window=10"])
    assert config.labels_cache_key(c) != l0 and config.features_cache_key(c) == f0
    c = config.load_config(sets=["walkforward.retrain_every_months=2"])
    assert config.features_cache_key(c) == f0 and config.labels_cache_key(c) == l0


def test_run_folder_refuses_other_settings(tmp_path, cfg):
    # A run folder keeps one set of merged settings. A change of compute only is allowed.
    config.write_run_config(tmp_path, cfg)
    config.write_run_config(tmp_path, config.load_config(sets=["compute.workers=2"]))
    with pytest.raises(config.ConfigError, match="different merged settings"):
        config.write_run_config(tmp_path, config.load_config(sets=["walkforward.retrain_every_months=2"]))


def test_old_run_folders_drop_the_folded_keys(tmp_path, cfg):
    # An experiments.yaml of the first runs. The keys that are now constants (vol_floor_daily, portfolio, scope,
    # pred_dir, ...) must drop out, and the keys added since then take their config.yaml values.
    old = {"run_name": "grid_v1", "data": {"fence_date": "2021-01-01", "min_history_sessions": 252},
           "labels": {"horizons": [1, 5], "vol_floor_daily": 0.003},
           "models": [{"name": "ridge_a1", "kind": "ridge", "alpha": 1.0, "scope": "global"},
                      {"name": "per_etf", "kind": "ridge", "alpha": 1.0, "scope": "per_etf", "enabled": False}],
           "portfolio": {"rules": [{"name": "top_bins_equal", "top_bins": [1]}]},
           "book": {"sources": {"ret_ens": {"kind": "predictions", "files": "all", "variants": ["rank"],
                                            "horizons": "all", "pred_dir": None, "targets_file": None}}}}
    (tmp_path / "experiments.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
    c = config.read_run_config(tmp_path)
    assert c["labels"]["horizons"] == [1, 5] and "vol_floor_daily" not in c["labels"]
    assert "portfolio" not in c and c["limits"] == cfg["limits"]
    assert [set(m) for m in c["models"]] == [{"name", "kind", "alpha"}, {"name", "kind", "alpha", "enabled"}]


def test_constants_name_universe_tickers():
    # Every ticker that a constant names must be in the universe file.
    tickers = set(pd.read_csv(settings.UNIVERSE_FILE, dtype=str)["ticker"])
    named = {settings.CASH_TICKER, *settings.CASH_LIKE, *cross.RELATIVE["drivers"], *cross.RELATIVE["beta_drivers"],
             *macro.MACRO["credit"], *macro.MACRO["rates"], macro.MACRO["dollar"]["ticker"],
             *macro.MACRO["regime"]["stock_bond"]}
    for spec in benchmarks.FIXED.values():
        named |= set(spec.get("weights", {})) | ({spec["ticker"]} if "ticker" in spec else set())
    assert named <= tickers, sorted(named - tickers)


def test_feature_windows_agree():
    # A window that a derived feature uses must be in the list of base windows of its module.
    p, v, tr = price.PRICE, price.VOLATILITY, price.TREND
    used = {p["trend_skip"]["skip"], *p["trend_skip"]["windows"], *[w for pair in p["accel_pairs"] for w in pair],
            p["pullback"]["long"], p["pullback"]["short"], p["drawdown_adjusted"]["ret_window"],
            *p["positive_share_windows"], p["crash"]["window"], p["crash"]["bounce"]}
    assert used <= set(p["ret_windows"])
    assert p["slope_accel"]["base"] in p["slope_windows"]
    assert v["vol_change"]["window"] in v["vol_windows"] and set(v["vol_pct_self"]["windows"]) <= set(v["vol_windows"])
    assert set(v["atr_ratio_pair"]) <= set(v["atr_windows"])
    assert {w for pair in tr["ema_pairs"] for w in pair} | {tr["macd"]["fast"], tr["macd"]["slow"]} <= set(tr["ema_windows"])
    assert set(tr["sma_slope_windows"]) <= set(tr["sma_windows"])
    assert 1 in bars.CANDLES["bar_sessions"] and bars.GAPS["share_window"] in bars.GAPS["overnight_windows"]
    rel = cross.RELATIVE
    assert rel["residual_window"] in rel["beta_windows"] and set(rel["decoupling_pair"]) <= set(rel["beta_windows"])
    assert macro.MACRO["regime"]["window"] in macro.MACRO["spread_windows"]
    pca = state.PCA
    derived = set(pca["derived"]["ratio_measures"]) | set(pca["derived"]["cross_measures"]) | {"absorption", "eff_n"}
    assert derived <= set(pca["measures"])
    for k in ("etf_state", "stock"):
        assert pca[k]["short"] in pca[k]["windows"] and pca[k]["long"] in pca[k]["windows"]
    assert pca["stock"]["min_assets"] <= settings.STOCK_STATE_SIZE
