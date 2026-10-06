"""Book stage on a toy run with known returns: the selection rules, the switch, the gross and its loan, the score
sources and the controls. Costs are off where a test needs an exact CAGR."""
import json

import numpy as np
import pandas as pd
import pytest

from etf66 import book, config, settings

TICKERS = ["SPY", "IWM", "XLK", "TLT", "GLD", "SHY"]
GROUPS = ["equity", "equity", "equity", "rates", "precious_metals", "rates"]
N_SESSIONS = 60


def make_run(tmp_path, monkeypatch, cfg, stress_all=True):
    """Return (book settings, run settings, run folder, r_next) of a toy run with 6 tickers and 60 sessions.

    stress_all sets the stress mask of every session. cfg is not used.
    """
    rng = np.random.default_rng(1)
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"ticker": TICKERS, "block_name": GROUPS, "role": ["x"] * 6}).to_csv(data / "tickers.csv", index=False)
    monkeypatch.setattr(settings, "DATA_DIR", data)
    monkeypatch.setattr(settings, "RUNS_DIR", tmp_path / "runs")
    # The default stale lag (126 sessions) is longer than the toy run.
    monkeypatch.setattr(book, "STALE_LAG", 10)
    run = tmp_path / "runs" / "toy"
    (run / "environment").mkdir(parents=True)
    (run / "preds").mkdir()
    r_next = rng.normal(0.0005, 0.01, (N_SESSIONS, 6))
    cash = np.full(N_SESSIONS, 0.0001)
    np.savez(run / "context.npz", r_next=r_next, cash_next=cash, tradable=np.ones((N_SESSIONS, 6), dtype=bool))
    dates = pd.bdate_range("2010-01-04", periods=N_SESSIONS)
    (run / "context.json").write_text(json.dumps({"dates": [str(d.date()) for d in dates]}))
    np.savez(run / "benchmarks.npz", ew_monthly=r_next.mean(axis=1), spy_ief_60_40=r_next[:, 0])
    np.save(run / "environment" / "stress.npy", np.full(N_SESSIONS, stress_all))
    # One slot per target of run_cfg: the variants rank and vol_scaled at h = 5.
    preds = rng.normal(size=(2, N_SESSIONS, 6)).astype(np.float32)
    np.save(run / "preds" / "m1__expanding.npy", preds)
    c = config.load_config(sets=["evaluation.common_start=2010-01-04",
                                 'evaluation.blocks=[["2010-01-04","2010-02-15"],["2010-02-16","2010-12-31"]]',
                                 "book.sources={ret_ens: {kind: predictions, files: all, variants: [rank], horizons: all}}",
                                 "book.controls.reference.source=ret_ens",
                                 "book.switch.defensive_share=0.5"])
    run_cfg = config.load_config(sets=["labels.horizons=[5]", "labels.variants=[rank,vol_scaled]"])
    return c, run_cfg, run, r_next


def test_top_fraction_and_switch_and_gross(tmp_path, monkeypatch):
    c, run_cfg, run, r_next = make_run(tmp_path, monkeypatch, None)
    inputs = book.BookInputs(c, run_cfg, run)
    score = np.tile(np.array([[0.0, 0.2, 0.4, 0.6, 0.8, np.nan]]), (N_SESSIONS, 1))
    # top_fraction 0.2 keeps the scores at or above 0.8: GLD only.
    w = book.select(score, {"name": "top_fraction", "fraction": 0.2}, inputs)
    assert np.allclose(w[0], [0, 0, 0, 0, 1.0, 0])
    first = inputs.first

    def cagr_of(daily):
        return np.prod(1 + daily) ** (252 / len(daily)) - 1

    plain = book.run_book(w, False, 1.0, 1, inputs, bps=0.0)
    assert np.isclose(plain["cagr"], cagr_of(r_next[first:, 4]), rtol=1e-9)
    # Gross 1.5: each session earns 1.5 x GLD. The loan of 0.5 pays the cash return plus 50 bps a year.
    gross15 = book.run_book(w, False, 1.5, 1, inputs, bps=0.0)
    loan = 0.5 * (0.0001 + 0.005 / 252)
    assert np.isclose(gross15["cagr"], cagr_of(1.5 * r_next[first:, 4] - loan), rtol=1e-9)
    # Switch with a defensive share of 0.5 and stress on every session: half the book is the defensive basket.
    # In the toy universe the defensive basket is TLT and GLD.
    switched = book.run_book(w, True, 1.0, 1, inputs, bps=0.0)
    defensive = (r_next[first:, 3] + r_next[first:, 4]) / 2
    assert np.isclose(switched["cagr"], cagr_of(0.5 * r_next[first:, 4] + 0.5 * defensive), rtol=1e-9)


def test_predictions_source_is_the_mean_member_percentile(tmp_path, monkeypatch):
    c, run_cfg, run, _ = make_run(tmp_path, monkeypatch, None)
    inputs = book.BookInputs(c, run_cfg, run)
    scores, groups = book.build_sources(inputs)
    s = scores["ret_ens"]
    # SHY is cash-like and not eligible, so it has no score. The 5 eligible ETFs span the percentiles 0 to 1.
    assert groups == {} and s.shape == (N_SESSIONS, 6) and np.isnan(s[:, 5]).all()
    assert np.allclose(np.sort(s[0, :5]), [0, 0.25, 0.5, 0.75, 1.0])


def test_random_stress_keeps_the_share_and_the_blocks():
    # The input has 30% stress sessions. The draws use blocks of 10 sessions.
    stress = np.r_[np.ones(30, dtype=bool), np.zeros(70, dtype=bool)]
    rng = np.random.default_rng(0)
    draws = np.stack([book.random_stress(stress, 10, rng) for _ in range(200)])
    assert abs(draws.mean() - 0.3) < 0.05
    assert all((d[:10] == d[0]).all() for d in draws)


def test_shuffle_keeps_the_values_of_each_row():
    x = np.array([[1.0, 2.0, np.nan, 3.0]])
    y = book.shuffle_rows(x, np.random.default_rng(0))
    assert sorted(y[0, [0, 1, 3]]) == [1.0, 2.0, 3.0] and np.isnan(y[0, 2])


def test_book_grid_and_controls_run_on_the_toy(tmp_path, monkeypatch):
    c, run_cfg, run, _ = make_run(tmp_path, monkeypatch, None)
    c["book"]["controls"]["shuffle_draws"] = 2
    c["book"]["controls"]["random_switch_draws"] = 2
    folder = book.run_book_stage(c, run_cfg, run, True, lambda m: None)
    books = pd.read_parquet(folder / "books.parquet")
    # 1 source x 3 selections (top_fraction, score_capped at power 1 and 2) x 2 switch states x 2 gross x 2 rebalance
    # intervals.
    assert len(books) == 1 * 3 * 2 * 2 * 2 and books["dsr"].between(0, 1).all()
    ctrl = pd.read_csv(folder / "controls.csv")
    assert ctrl["family"].tolist()[:3] == ["real", "real, stress cost", "real, switch off"]
    assert (ctrl["family"] == "random switch").sum() == 2 and (folder / "REPORT.md").exists()


def test_book_needs_the_stress_file_for_a_switch(tmp_path, monkeypatch):
    c, run_cfg, run, _ = make_run(tmp_path, monkeypatch, None)
    (run / "environment" / "stress.npy").unlink()
    with pytest.raises(config.ConfigError, match="environment"):
        book.BookInputs(c, run_cfg, run)


def test_score_power_two_squares_the_distance_from_the_middle(tmp_path, monkeypatch):
    c, run_cfg, run, _ = make_run(tmp_path, monkeypatch, None)
    inputs = book.BookInputs(c, run_cfg, run)
    rule = {"name": "score_capped", "max_weight": 1.0, "max_group_weight": 1.0, "iterations": 10}   # no cap binds
    score = np.tile(np.array([[0.5, 0.6, 0.7, 0.8, 0.9, np.nan]]), (N_SESSIONS, 1))
    w1 = book.select(score, rule, inputs, power=1.0)
    w2 = book.select(score, rule, inputs, power=2.0)
    # The distances of the scores above the middle (0.5).
    raw1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    assert np.allclose(w1[0, :5], raw1 / raw1.sum())
    assert np.allclose(w2[0, :5], raw1 ** 2 / (raw1 ** 2).sum())
