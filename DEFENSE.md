# Defense notes

The design in short, and what each file does. Every run uses dev data only (2003-09 to 2020-12; evaluation
2010-01-04 to 2020-12-31), net of 5 bps per side.

## 1. Design

1. We predict, for each of 66 ETFs, the forward return over h sessions (h from 1 to 44), scaled by forward
   volatility and ranked across ETFs (`etf66/labels.py`).
2. The inputs are 339 features known at the close of t: price, volatility, trend, bar shape, volume, relative
   strength, screener flags, PCA market state of the ETFs and of the 500 largest stocks, macro and calendar.
3. We decide at the close of t and fill at the open of t+1. The label and the trade start at that same open; the
   label is the volatility-scaled rank of that return.
4. Ridge and LightGBM models fit walk forward: a refit on the first session of each month, on expanding and rolling
   windows, and every training label must end before the refit session (the purge).
5. Predictions become bins and weights through fixed rules; every rule runs through one simulator with drift,
   turnover and costs.
6. The default grid has 72,160 trials; a run records every trial with its DSR and the grid PBO.
7. The book stage stacks three layers on one run: a stress switch to a defensive basket, a selection by group
   predictions, and a gross of 1.0 or 1.5.
8. The holdout (2021-01-01 on) sits behind a code fence; only an owner-written unlock file opens it, one time.

## 2. Files

| File | What it does. Why it exists. What can go wrong. |
|---|---|
| `config.yaml` | The 83 dials and the time-order and holdout keys. One place to change a run. A dial set in an overlay but read from the frozen run settings by a stage. |
| `etf66/config.py` | Loads, merges and saves settings. Runs must be repeatable. A run folder key that `config.yaml` lacks is dropped on read, so its value goes unseen. |
| `etf66/rules.py` | Checks the settings before any work. A typo must stop the run. A rule that is too loose lets a bad dial through. |
| `etf66/settings.py` | Paths and fixed constants (fence date, cash ticker, 252). One home for shared constants. A wrong data folder through the environment variables. |
| `etf66/fence.py` | Refuses holdout dates in dev mode; reads the unlock file. Keeps the holdout sealed. A table without a date column is not checked. |
| `etf66/snapshot.py` | Copies source data into dev and holdout. The only reader of the owner's market folder. The stock universe file comes from outside this repo and cannot be checked here. |
| `etf66/data.py`, `universe.py` | Load the panel; mark tradable ETFs. One calendar for every table. A missing open counts as a 0 return. |
| `etf66/ops.py` | Rolling and cross-sectional operators, past only. Every feature uses them. A min-observation rule that differs between two operators. |
| `etf66/features/*` | 339 features; windows are constants at the top of each module. The model inputs. A shift in the wrong direction; the canary test guards it. |
| `etf66/labels.py` | Forward returns from the open of t+1, five variants. What the models learn. The reach (1 + max(h, 5)) and the purge must agree. |
| `etf66/walkforward.py` | The refit schedule, the purge and the windows. No training label may see the refit session. The purge is one session wider than the minimum; a change below 0 is refused. |
| `etf66/models.py` | Ridge (GPU or CPU) and LightGBM. Two model families, cheap enough for the grid. LightGBM results change with the thread count. |
| `etf66/bins.py`, `portfolio.py` | Bins, rules, and the two simulators. One accounting for every book. Ties in cs_bin break by column order. |
| `etf66/backtest.py`, `benchmarks.py`, `metrics.py`, `report.py` | The trial grid, benchmarks, Sharpe, DSR, PBO, the report. No filter drops a trial. DSR counts the trials of one run only. |
| `etf66/groups.py` | Group targets and predictions (18 groups). The selection source of the best book. Exact ties between group scores depend on float rounding. |
| `etf66/environment.py` | Basket targets, their predictions, the stress mask. Layer 1 of the book. The stress input, a 21-session dd forecast, is weak; each run writes its predictability.csv. |
| `etf66/book.py` | The three-layer book and its controls. The book layers we test. Its grid DSR counts 72 books, not the whole search. |
| `tests/` | 83 tests on synthetic data. Proof without the data. They test rules, not the real data. |
