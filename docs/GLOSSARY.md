# Glossary

One word, one meaning. The code and the docs use these words.

| Word | Meaning |
|---|---|
| session | one trading day; every window counts sessions |
| t | the decision session; we decide at the close of t |
| h | the label horizon in sessions |
| fill | the trade at the open of t+1 |
| lookahead | any use of data from after the close of t in a decision at t; the code must have none |
| dev | data before 2021-01-01; every result in this repo uses dev data only |
| holdout | data on or after 2021-01-01; only the `holdout` command reads it, one time, after the owner unlocks it |
| fence | the code that keeps the holdout out of dev runs (`etf66/fence.py`) |
| ticker | the symbol of one ETF; a table column |
| names | the ETFs that a book holds on a session |
| tradable | an ETF with a close at t, no source flag at t, and 252 or more closes up to t |
| asset group | the group of an ETF (equity, rates, credit, ...), the column `block_name` of the universe file |
| group | an asset group, with the equity ETFs split by role into broad, sector, industry and factor (18 groups) |
| dial | a key of `config.yaml` that we change between runs; every other choice is a constant in the code |
| settings | the merged result of `config.yaml`, the overlays and the `--set` values of a run |
| overlay | a YAML file that changes some dials for one run |
| cache key | the hash in the name of a feature or label file; it covers the dials and the source files that build it |
| feature | a number known at the close of t for one (session, ETF) |
| per-ETF feature | a feature with one value for each ETF |
| global feature | a feature with one value for each session, the same for every ETF |
| PCA state | measures of the eigenvalues of the return correlation matrix over a window (for example eff_n) |
| screener lens | a 0/1 flag from the owner's earlier equity screeners (x3, o2, o3, p2, x5, r1, g5, a4, m2) |
| label | the value a model learns; it reads data after t, so training must purge it |
| label variant | one label formula: vol_scaled, rank, excess_vol_scaled, bin20, drawdown_rank, or payoff_l<lambda> |
| target | one (label variant, h) pair; the default grid has 45 |
| reach | how many sessions after t a label reads: 1 + max(h, 5) |
| purge | the gap that keeps every training label known before the retrain session |
| walk forward | training on past rows only, refit on the first session of each month, then predicting until the next refit |
| window | the training rows of a fit: expanding, or rolling over the last N sessions |
| prediction series | the daily predictions of one (model, window, target) |
| bin | a number from 1 (lowest) to 10 (highest) |
| ts_bin | the bin of a prediction against the same ETF's predictions of the previous 63 sessions |
| cs_bin | the bin of a prediction among the ETFs of the same session |
| rule | the method that turns predictions, bins or scores into target weights |
| book | the weights of one portfolio over time, and its daily returns |
| rebalance | a session on which the book moves to its target weights |
| no-trade band | the smallest weight change that a held position trades |
| turnover | the sum of the absolute weight changes; 1 = the full book bought or sold once |
| benchmark | a fixed or random book that runs through the same simulator and costs |
| trial | one prediction series with one portfolio variant (rule, rebalance interval, offset, band) |
| period block | one of the four evaluation periods in `evaluation.blocks` (2010-2012, 2013-2015, 2016-2018, 2019-2020) |
| IC | the rank correlation of the predictions and the labels across ETFs on one session |
| DSR | deflated Sharpe probability: the chance that the Sharpe is real after the number of trials |
| PBO | probability of backtest overfitting: the chance that the best in-sample trial is below the median out of sample |
| CSCV | combinatorially symmetric cross-validation: every split of 16 slices of the sessions into two halves |
| run | one execution of a command; it has a run id and a folder in `runs/` |
| ledger | the table of every trial of every run (`runs/ledger.parquet`) |
| stage | a command that reads a finished run and writes its own folder in it: groups, environment or book |
| group score | the mean percentile of the group predictions over the models, windows and horizons of the groups stage |
| environment | the basket a few sessions ahead: its return, drawdown, volatility and correlation, and their predictions |
| basket | the equal weight of the tradable ETFs outside SGOV and SHY, open to open |
| stress | a session whose predicted 21-session basket drawdown is in the top 20% of the model's own last 252 predictions |
| defensive basket | equal weight over the tradable ones of TLT, IEF, GLD, USMV and UUP |
| switch | book layer 1: in stress, a share of the book moves to the defensive basket |
| selection | book layer 2: a source and a rule pick the ETFs and their weights |
| gross | book layer 3: the weights times a number; above 1 is a loan |
| source | what gives each ETF a score in the book: predictions, groups, or the mean of other sources |
| ensemble | the mean percentile of several prediction files (the book source `ret_ens`) |
| control | a book with one layer broken on purpose (shuffled, stale or random) to test whether that layer adds value |
| sharpe_excess | the Sharpe of the book's net return minus the cash (SHY) return; the main book measure |
