# etf66

We predict the forward returns of 66 ETFs and test books built from the predictions. Every result here uses data
before 2021-01-01. The data from 2021-01-01 on is the holdout. No run in this repo read it.

The repo holds code only. `DEFENSE.md` has the design in short and a card for each file.

## 1. The pipeline

1. `snapshot` copies the source data into `data/dev` (before 2021-01-01) and `data/holdout` (2021-01-01 on).
2. `features` builds 339 features for each (session, ETF). Every value at t uses data up to the close of t.
3. `labels` builds 45 targets: 5 label variants x 9 horizons (1, 3, 5, 7, 10, 15, 21, 32, 44 sessions). A label
   starts at the open of t+1, the fill.
4. `train` fits the models walk forward: a refit on the first session of each month, purged training rows, expanding
   and rolling windows. The models predict every session until the next refit.
5. `backtest` turns the predictions into bins, the bins into books, and the books into daily net returns. It runs the
   benchmarks and random books through the same simulator and costs.
6. `groups`, `environment` and `book` are stages on a finished run: group predictions, the market environment and the
   stress switch, and the three-layer book (switch, selection, gross).

Each run writes `runs/<run_id>/` with its merged settings, the code of each start, the predictions, the trials and
`REPORT.md`. The ledger `runs/ledger.parquet` stores every trial of every run. The `dsr` column counts one run;
`dsr_all_runs` counts the ledger, which does not hold the stage and study trials.

Read `docs/DESIGN.md` for the full design, `docs/GLOSSARY.md` for the words and `docs/FEATURES.md` for the feature
list.

## 2. Find your way around

Every command starts in `etf66/cli.py`, function `main`. Read in this order (about 30 minutes):

1. [`config.yaml`](config.yaml): the dials of a run, one comment each.
2. [`etf66/labels.py`](etf66/labels.py) `build_labels`: what we predict and when it starts (the open of t+1).
3. [`etf66/walkforward.py`](etf66/walkforward.py) `train_end_row` and `run_job`: the purge and the monthly refit.
4. [`etf66/portfolio.py`](etf66/portfolio.py) `simulate`: how weights become daily returns and turnover.
5. [`etf66/book.py`](etf66/book.py) `run_book` and `controls`: the best book and the tests against it.
6. [`DEFENSE.md`](DEFENSE.md): the design in short and a card for each file.

| Question | Where to look |
|---|---|
| Can a feature see the future? | [`etf66/ops.py`](etf66/ops.py) (past-only operators), [`etf66/features/context.py`](etf66/features/context.py) `build_context` (the lags), [`tests/test_lookahead.py`](tests/test_lookahead.py) |
| How is the holdout kept out? | [`etf66/fence.py`](etf66/fence.py), [`etf66/data.py`](etf66/data.py) `read_table`, [`docs/DESIGN.md`](docs/DESIGN.md) section 3 |
| What are the 339 features? | [`docs/FEATURES.md`](docs/FEATURES.md); windows at the top of each file in [`etf66/features/`](etf66/features) |
| What is the label, and the purge? | [`etf66/labels.py`](etf66/labels.py) `label_reach`, [`etf66/walkforward.py`](etf66/walkforward.py) `train_end_row` |
| Which models, with which settings? | [`etf66/models.py`](etf66/models.py), the `models` list in [`config.yaml`](config.yaml) |
| How do predictions become a book? | [`etf66/bins.py`](etf66/bins.py), [`etf66/portfolio.py`](etf66/portfolio.py) `RULES` and `rule_weights` |
| What does a trade cost, and when does it fill? | [`etf66/backtest.py`](etf66/backtest.py) `COST_BPS_PER_SIDE` and `next_open_returns` |
| How are Sharpe, DSR and PBO computed? | [`etf66/metrics.py`](etf66/metrics.py) |
| What are the benchmarks? | [`etf66/benchmarks.py`](etf66/benchmarks.py) `FIXED` |
| How does the stress switch decide? | [`etf66/environment.py`](etf66/environment.py) `stress_mask` |
| Where do the group scores come from? | [`etf66/groups.py`](etf66/groups.py) `run_groups` |
| Which settings does a run refuse? | [`etf66/rules.py`](etf66/rules.py) `check_rules` |
| What does a word mean? | [`docs/GLOSSARY.md`](docs/GLOSSARY.md) |

## 3. Dials

`config.yaml` holds the 83 dials we changed between runs, plus the keys that guard time order and the holdout.
Every other choice is a named constant next to the code that uses it. Change a dial for one run:

```
python -m etf66 all --run-id test1 --set walkforward.retrain_every_months=3
python -m etf66 all --run-id test2 --overlay my_changes.yaml
```

An overlay is a YAML file with only the keys to change. The loader refuses an unknown key, refuses a change to the
fence date, and checks `etf66/rules.py` before any work starts. Each run saves its merged settings in
`runs/<run_id>/config.merged.yaml`.

## 4. Install

Use Python 3.12.

```
pip install -r requirements.txt pytest
pip install cupy-cuda12x        # optional: the ridge fit uses the GPU when CuPy works
```

## 5. Run

```
python -m pytest -q tests                          # synthetic data, no snapshot needed
python -m etf66 snapshot                           # owner machine only (reads D:/Data/market)
python -m etf66 all --smoke --run-id smoke_01      # a small grid, about 3 minutes
python -m etf66 all --run-id grid_v1 --workers 3   # the full grid of config.yaml
python -m etf66 groups --run-id grid_v1_stages --set groups.source_run=grid_v1 --workers 2
python -m etf66 environment --run-id grid_v1_stages --workers 3
python -m etf66 book --run-id grid_v1_stages --controls
```

The stages read a finished run and write `groups/`, `environment/` and `book/` inside it. `groups.source_run` links
the predictions of a run into a new run folder, so the source run stays as it is.

`train`, `backtest` and `all` on an existing run folder, with no `--overlay`, `--set` or `--smoke`, use the frozen
settings of that run. A stopped run continues with the same command: train skips every job whose predictions exist.
`ETF66_DATA_DIR` and `ETF66_RUNS_DIR` move the data and runs folders.

On the owner's machine (24 cores, 32 GB RAM, one RTX 5090) the feature build takes about 1 minute, the smoke grid
about 3 minutes with the build, and the full grid of 72,160 trials about 1 hour.

## 6. Without the data

- The tests need no data: `python -m pytest -q tests`.
- To run the pipeline you need `data/dev/`, `data/MANIFEST.json` and `data/tickers.csv` (about 35 MB) from the
  owner. Dev runs never need `data/holdout/`.

## 7. Flags

- The source bars lack 4 sessions (2020-12-03 to 2020-12-08). The next return spans the gap.
- SPY is a price-only series before 2007-03 in the source.
- Dollar volume is the adjusted close times the raw volume.
- The 66 ETFs were chosen with knowledge of 2010-2020, so the universe has hindsight.
