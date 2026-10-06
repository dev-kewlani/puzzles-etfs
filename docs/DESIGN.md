# Design

This file records each design decision, the dial or constant that holds it, and the code that does it. The values below are the defaults.

## 1. Settings

- `config.yaml` holds the dials (keys we change between runs) and the keys that guard time order and the holdout.
  Every other choice is a named constant next to the code that uses it.
- A run changes dials in two ways, in this order: `--overlay FILE`, then `--set key.path=value`. `--smoke` adds the
  overlay `configs/overlays/smoke.yaml`.
- Dicts merge key by key. Lists replace the base list. `models` and `book.selection` merge entry by entry on `name`.
- The loader refuses an unknown key and a change to a read-only key (`data.fence_date`, `paths.unlock_file`). It then
  checks `etf66/rules.py`.
- Each run saves `config.merged.yaml`, `config.base.yaml`, each overlay and `overrides.txt` in its folder. A second
  start with the same `--run-id` must hold the same merged settings. A run folder key that `config.yaml` does not have is dropped
  when the folder is read (`config.drop_old_keys`).
- The config hash covers every setting except `compute` and `run.notes`.

## 2. Time rules

- A "session" is one trading day. All windows count sessions.
- The model decides at the close of session t. The fill is at the open of session t+1.
- A feature at session t reads data up to the close of t.
- A feature at session t reads the market series (VIX, indices, futures) and the CBOE indices of session t-1. They can
  close after the ETF close. The code fills CBOE gaps of up to 5 sessions before the lag (`context`).
- A Treasury yield is used from its publication date (`available_at_date`). It needs no lag.
- The stock PCA state uses the 500 largest stocks of each year. The snapshot reads the size rank
  (`snapshot.STOCK_RANK_FIELD`, the median market cap over 63 sessions) and the flag `value` from the owner's file
  `annual_core_pass.parquet`. That file fixes the rank at the end of the year before. This repo cannot check that file.
- The prices are adjusted open, high, low and close from the owner's source. Volume is raw volume.

## 3. Data and the fence

- The fence date is 2021-01-01. The code fixes it in `etf66/settings.py`; `rules.py` refuses another date.
- `data/dev` holds rows before the fence. `data/holdout` holds rows on or after the fence. The holdout files are
  read-only at file level.
- Dev mode checks every table with a `date` column, the publication date of the Treasury table, and the membership
  year of the stock members. A table without these columns is not checked.
- An ETF is tradable at session t when it has a close at t, the source does not flag it at t, and it has 252 or more
  closes up to t (`settings.MIN_HISTORY_SESSIONS`).

## 4. Labels

For horizon h and decision row t:

- r = open[t+1+h] / open[t+1] - 1.
- fwd_vol = the standard deviation of the open-to-open returns of sessions t+2 to t+1+m, with m = max(h, 5).
- scale = max(fwd_vol, 0.003) x sqrt(h). The floor stops T-bill ETFs from getting very large labels.
- An ETF is labelled at t when it is tradable at t, r exists and all m forward returns exist.

| Variant | Formula |
|---|---|
| vol_scaled | r / scale |
| rank | cross-sectional rank of vol_scaled among the labelled ETFs, from 0 to 1 |
| excess_vol_scaled | (r - mean r of the labelled ETFs) / scale |
| bin20 | rank cut into 20 equal bins, coded 0.05 to 1.00 |
| drawdown_rank | cross-sectional rank of (forward drawdown / scale); high = small drawdown |

The forward drawdown is the lowest value of open[s] / (running peak of open[t+1..s]) - 1, for s from t+1 to t+1+h.
It is missing when any open of the path is missing.

## 5. Features

- 339 features in 14 groups. `docs/FEATURES.md` lists each feature. The windows and thresholds are the
  constants at the top of each feature module, next to the formulas.
- A per-ETF feature becomes its cross-sectional rank among the tradable ETFs, minus 0.5.
- A global feature (one value per session) becomes its expanding past-only percentile, minus 0.5. It needs 252
  earlier values. Each tradable ETF gets the same value. A global feature is NaN for the first 252 values and for
  ETFs that are not tradable.
- A missing value stays missing. The ridge model standardizes each feature with the training mean and standard
  deviation, then sets a missing value to 0, which is the training mean. LightGBM uses its own missing-value path.
- The feature file and the label file are named by a hash of the dials and the source files that change them, plus
  the hash of the snapshot manifest and ticker list. A change of a setting builds a new file; an old file is never used by mistake.

## 6. Models and the walk-forward schedule

- Models (default): ridge with alpha 0.01, 0.1 and 1 on standardized features, and `lgbm_light` (7 leaves, 50 rounds,
  learning rate 0.15, a fixed seed). Each model is one pooled model for all ETFs and uses every feature.
- Windows: expanding, rolling 5 years (1260 sessions), rolling 3 years (756) and rolling 1 year (252).
- first_row is the first session on which an ETF is tradable. Expanding starts when end - first_row + 1 >= 252.
  Rolling W starts when end - W + 1 >= first_row.
- Retrain: the first session of each month (`walkforward.retrain_every_months`).
- Purge: for a retrain at session d and horizon h, the training rows end at end = d - (1 + max(h, 5)) - 1. So every
  training label is known before the close of d.
- A training row is a (session, ETF) that is tradable and has all 5 label variants of h. The variants share one fit.
- A fit needs 1000 training rows (`walkforward.MIN_ROWS`). With fewer rows the code skips the fit and the predictions
  stay NaN.
- The model predicts every tradable (session, ETF) from d to the session before the next retrain.
- LightGBM builds its binned dataset with the full training params, so its feature pre-filter uses the model's
  `min_data_in_leaf`. The run `grid_v1_light` built its dataset with `max_bin` only, so its
  pre-filter used the LightGBM default of 20; a retrain of `lgbm_light` does not give its files bit for bit.
- LightGBM results depend on the thread count. `compute.lightgbm_threads` fixes it (null = the CPU count divided by
  the workers).

## 7. Bins

- `ts_bin`: the prediction at session t against the predictions of the same ETF in the 63 sessions before t. All 63
  sessions must hold a prediction. These predictions can come from different model fits.
- `cs_bin`: the prediction at session t ranked among the ETFs with a prediction at t. Ties break in column order
  (`bins.cross_section_pct`, tie rule ordinal).
- The backtest computes both bins in memory for each prediction series. It does not save them.
- With the default grid, the window rolling_5y for horizons 15, 21, 32 and 44 has its first full ts_bin in February
  or March 2010, after `evaluation.common_start`. The run writes this in `config_warnings.txt`. A ts_gate_cash book
  of those series holds cash until then.

## 8. Portfolios

- A position decided at the close of t earns open[t+2] / open[t+1] - 1.
- Cost: 5 basis points per side on the traded ETF weight. The stress costs are 10 and 20 basis points.
  A trade into or out of cash is free. The long-short book pays no borrow fee on its shorts.
- Unused weight is in SHY (cash) in long-only books. The long-short book has no cash.
- The book starts flat. The first rebalance is the first evaluation session; then every N-th session, N in
  {1, 5, 10, 21, h}. `backtest.REBALANCE_OFFSETS` moves the start. Between rebalances the weights drift.
- No-trade band 0 or 0.02: a held position changes only when the change is 0.02 or more. An entry or an exit always
  trades.
- If kept positions make a long-only book larger than 100% (`portfolio.MAX_LONG_BOOK`), the simulator scales the
  book down to 100%. The cut counts as turnover.
- A missing next open counts as a 0 return.

| Rule | Weights |
|---|---|
| top_bins_equal (k = 1, 2, 3) | equal weight on the top k cross-sectional bins |
| score_capped | weight in proportion to max(cs percentile - 0.5, 0); 15% cap per ETF, 40% cap per asset group; the cut weight goes to cash |
| ts_gate_cash | pick an ETF when ts_bin >= T and (C = 0 or cs_bin >= C), with T in {7, 8, 9} and C in {0, 6}. K = max(1, ceil(0.10 x tradable ETFs)). Each picked ETF gets 1 / max(K, count picked); the rest is cash |
| long_short | equal weight long the top bin, equal weight short the bottom bin |

The asset groups come from the column `block_name` of `configs/universe66.csv`.

## 9. Benchmarks

`benchmarks.FIXED` lists each fixed benchmark. Every benchmark runs through the same simulator and costs.

- ew_monthly: equal weight over the tradable ETFs, rebalanced on the first session of each month.
- mom_12_1: the top 10% by close[t-21] / close[t-252] - 1, rebalanced every 21 sessions.
- mom_6_3: the top 10% by close[t] / close[t-126] - 1, rebalanced every 63 sessions. The feature `mom_6_3` is a
  different formula (close[t-63] / close[t-126] - 1).
- spy: 100% SPY. spy_ief_60_40: 60% SPY and 40% IEF, rebalanced monthly.
- inverse_vol_63: weight 1 / (63-session volatility), rebalanced monthly.
- trend_200d: SPY when the SPY close is above its 200-session mean, else cash; checked every session.
- Random controls: equal weight over a random 10%, 20% or 30% of the tradable ETFs, drawn again every 1, 5, 10, 21
  or 63 sessions. There are `benchmarks.random_draws` draws of each (500), with a fixed seed.

## 10. Evaluation

- The evaluation period starts on 2010-01-01 and ends on the last dev session (`evaluation.end` can end it earlier).
- Four period blocks: 2010-2012, 2013-2015, 2016-2018 and 2019-2020. The median block Sharpe is the main ranking
  column (`report.RANK_COLUMN`).
- Each trial gets: CAGR, yearly volatility, maximum drawdown, turnover per year and hit rate. It also gets the Sharpe
  and CAGR at the stress costs and the IC of its prediction series.
- `info_ratio_vs_ew` is the yearly Sharpe of (net book return minus net equal-weight return).
  `ic_t` is the mean IC / std IC x sqrt(sessions / h).
- `dsr` uses N = the count of trials with a finite Sharpe in this run, and the variance of the trial Sharpes.
  `dsr_all_runs` adds the trials of every earlier run in `runs/ledger.parquet`.
- The PBO cuts the evaluation sessions into 16 equal slices and uses all C(16, 8) = 12,870 splits and all trials of
  the run.
- Other columns: `min_block_sharpe`, `excess_cagr_vs_ew`, `avg_gross_exposure` (the mean target gross weight on
  rebalance sessions), `prediction_share` (the share of evaluation sessions with a prediction), `rebalance_offset`.
- These measures are evidence columns. No measure removes a trial.

## 11. Grid size

4 models x 4 windows x 45 targets = 720 prediction series. Each series runs 11 rule variants x 2 bands x 4 intervals
(h in 1, 5, 10, 21) or 5 intervals (other h): 88 or 110 portfolio variants. The full grid has 72,160 trials.

## 12. Groups stage

The dials are `groups.source_run` and `groups.purge_extra_sessions`; the rest are constants in `etf66/groups.py`. The command
`python -m etf66 groups --run-id <run>` writes `runs/<run>/groups/` and two ETF-level score files in
`runs/<run>/preds/`. With `groups.source_run` set, the command first makes the run `--run-id` from the frozen
settings of the source run and links its predictions; the source run stays as it is.

- A group is an asset group of the universe file. An equity ETF joins the group `eq_<name>` by its role, as
  `EQUITY_SPLIT` says. The cash-like ETFs (SGOV, SHY) belong to no group. The default universe gives 18 groups.
- The group return at t is the mean over the tradable members of the return that a decision at the close of t earns
  (open[t+2] / open[t+1] - 1). A group exists at t when it has a tradable member.
- The label of horizon h is the rank among the groups (average ties, 0 to 1) of the forward h-session return of the
  group, divided by max(forward volatility over max(h, min_vol_window) sessions, vol_floor_daily) x sqrt(h).
- A row is one (session, group). The features are the mean of each ETF feature of the run over the tradable members.
- The walk-forward follows the main one: a retrain on the first session of every k-th month, an expanding window
  from the first session or a rolling window of `window_sessions`, and training rows that end at
  d - h - max(h, min_vol_window) - purge_extra_sessions. A fit needs `min_train_rows` rows and an end row at or
  after `min_end_row`.
- The group score of a horizon is the mean over the models and windows of the group percentile of their predictions.
  `score_all` is the mean over the horizons too. The ETF-level files `grp_ens__all.npy` (one slot per horizon) and
  `grp_ens_all_h__all.npy` (the all-horizon score in every slot of `score_variant`) give each ETF the score of its
  group, ranked among the eligible ETFs. The backtest reads them as prediction files; the book reads the group percentiles directly.
- The stage uses the frozen settings of the run for the features and the label layout, and the current settings for
  its own section and `model_defaults`. It saves the section it used in `groups/settings.yaml`.

## 13. Environment stage

The dials are `environment.purge_extra_sessions` and `environment.stress`; the rest are constants in `etf66/environment.py`. The command
`python -m etf66 environment --run-id <run>` writes `runs/<run>/environment/`.

- The basket is the equal weight of the tradable ETFs outside SGOV and SHY, open to open, rebalanced each session.
- The targets at t, for each horizon h and m = max(h, min_vol_window): `env_ret` (the basket return over sessions
  t to t+h-1), `env_dd` (the lowest drawdown along that path, from a peak at or above 1), `env_vol` (the yearly
  volatility of the basket over t to t+m-1) and `env_corr` (the mean pairwise correlation of the ETF returns over
  the same sessions, over the ETFs with a full window; at least `min_corr_assets` ETFs).
- The persistence baseline is the same quantity over the sessions that end at t.
- The features are the global features of the run: the groups in `feature_groups` plus the pca_etf features whose
  name starts with `global_pca_prefix`, read from the column of `feature_ticker`.
- The walk-forward uses an expanding window, a retrain on the first session of every k-th month, and training rows
  t <= d - (1 + m) - purge_extra_sessions with a finite target and at least `min_feature_share` finite features.
  The fits start at the first retrain at or after `evaluation.common_start` minus `lead_sessions_before_start`.
- `predictability.csv` holds, per target, horizon and model (and the persistence baseline): the Spearman
  correlation with the outcome over the evaluation sessions, its t-statistic with the session count divided by h,
  and the out-of-sample R2 against the expanding mean of the labels known at t (`baseline_min_sessions` labels).
- Stress: for each model, the percentile of the predicted stress input (minus the predicted
  drawdown for `dd`; the predicted volatility or correlation for `vol` and `corr`) against the model's own previous
  252 predictions (at least 126). A session is stressed when the mean over the models is at or
  above `stress.threshold`. `stress.npy` holds the mask; the book reads it.

## 14. Book stage

The dials are in config.yaml, section `book`. The code is in `etf66/book.py`. The command
`python -m etf66 book --run-id <run>` writes `runs/<run>/book/`. It needs the backtest files of the run
(`context.npz`, `benchmarks.npz`), the environment stage when a switch state is true, and the groups stage when a
source has kind `groups`.

- Layer 1, the switch. In a stress session, `switch.defensive_share` of the book moves to the defensive basket
  (equal weight over the tradable ETFs in `book.DEFENSIVE_TICKERS`: TLT, IEF, GLD, USMV, UUP). `switch.states` lists the states to run.
  The default share is 1.0 and the default powers are [1, 2]: the pocket study found that a
  full defensive move and a score power of 2 hold in 3 of the 4 period blocks and in both halves of the data.
- Layer 2, the selection. A source gives each ETF a score, a cross-sectional percentile (average ties) among the
  eligible ETFs (tradable, not SGOV or SHY). Kind `predictions`: the mean percentile of the predictions of
  the named files (`all` = every model file of the run), label variants and horizons. Kind `groups`: the mean group percentile
  of the groups stage (the horizons named), mapped to each ETF. Kind `mean`: the mean of other sources, ranked
  again. The rules: `top_fraction` (equal weight on the top fraction) and `score_capped` (weight in proportion to
  max(score - 0.5, 0), capped per ETF and per asset group; the rest stays in cash). `score_powers` runs the capped
  rule once per power p: the score becomes 0.5 + sign(s - 0.5) x (2 |s - 0.5|) ^ p / 2 before the weights, so a
  power above 1 moves weight toward the highest scores. The row name is `score_capped` for p = 1 and
  `score_capped_p<p>` for other powers. A frozen run settings file without `score_powers` means `[1]`.
- Layer 3, the gross. The weights are scaled by each value of `gross`. A gross above 1 is a loan that pays the cash
  return plus 0.5% a year. The loader keeps `gross` at or below `limits.max_gross` (2.0), the owner's leverage limit.
- The book rebalances every N sessions from the first evaluation session, fills at the open of t+1 and earns
  open[t+2] / open[t+1] - 1. The cost is 5 bps per side on the ETF turnover.
- Measures over the evaluation sessions: `sharpe_excess` (the Sharpe of the net return minus the cash return),
  `sharpe`, `cagr`, `vol_year`, `max_drawdown`, `info_ratio_vs_ew`, the period Sharpes of the excess return,
  `turnover_year`, and `dsr` with the book count of the grid and the variance of their Sharpes.
- Baselines: the equal-weight book at every switch state and gross, and the fixed benchmarks of the run.
- Controls (`book --controls`) on `controls.reference`: the source shuffled across groups (kind `groups`) or across
  ETFs (`shuffle_draws` draws), the source 126 sessions old, a random stress mask with the same share of
  stress sessions in blocks of 21 sessions (`random_switch_draws` draws), the switch off, and the 20 bps cost.

