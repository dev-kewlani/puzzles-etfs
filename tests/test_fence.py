"""The fence. A dev date index with a date on or after 2021-01-01, or a missing date, raises FenceError. The holdout
needs an unlock file that names a run with trials. The fence date and the unlock path are read-only keys."""
import pandas as pd
import pytest

from etf66 import config, fence, settings


def test_dev_index_before_fence_passes():
    fence.check_dev_index(pd.DatetimeIndex(["2020-12-31"]), "x")


def test_dev_index_on_fence_fails():
    # 2021-01-01 is the fence date, so it is the first holdout date.
    with pytest.raises(fence.FenceError):
        fence.check_dev_index(pd.DatetimeIndex(["2020-12-31", "2021-01-01"]), "x")


def test_dev_index_with_missing_date_fails():
    with pytest.raises(fence.FenceError):
        fence.check_dev_index(pd.DatetimeIndex(["2020-12-31", None]), "x")


def test_unlock_missing_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UNLOCK_FILE", tmp_path / "HOLDOUT_UNLOCK.yaml")
    with pytest.raises(fence.FenceError):
        fence.read_unlock()


def test_unlock_unknown_run_fails(tmp_path, monkeypatch):
    p = tmp_path / "HOLDOUT_UNLOCK.yaml"
    # All 5 fields are present. Only the run has no trials.parquet.
    p.write_text("owner: x\ndate: 2027-01-01\nrun_id: no_such_run\nconfig_ids: [a]\nstatement: s\n", encoding="utf-8")
    monkeypatch.setattr(settings, "UNLOCK_FILE", p)
    monkeypatch.setattr(settings, "RUNS_DIR", tmp_path)
    with pytest.raises(fence.FenceError):
        fence.read_unlock()


def test_fence_date_is_read_only():
    with pytest.raises(config.ConfigError, match="read-only"):
        config.load_config(sets=["data.fence_date=2022-01-01"])


def test_unlock_file_path_is_read_only(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text("paths: {unlock_file: other.yaml}\n", encoding="utf-8")
    with pytest.raises(config.ConfigError, match="read-only"):
        config.load_config(overlays=[p])
