"""Walk forward. The retrain falls on the first session of each month. The purge ends every training label before
the retrain session. A window starts only when it has enough history."""
import pandas as pd

from etf66 import labels as lab
from etf66 import walkforward


def test_retrain_on_month_starts():
    d = pd.bdate_range("2010-01-01", "2010-04-30")
    idx = walkforward.retrain_indices(d, 1)
    assert [str(d[i].date()) for i in idx] == ["2010-01-01", "2010-02-01", "2010-03-01", "2010-04-01"]


def test_purge_ends_before_the_retrain_session(cfg):
    d = 2000
    for h in cfg["labels"]["horizons"]:
        end_row = walkforward.train_end_row(d, h, cfg)
        reach = lab.label_reach(h, cfg["labels"]["min_vol_window"])    # the last open that a label reads is t + reach
        assert end_row == d - reach - cfg["walkforward"]["purge_extra_sessions"]
        assert end_row + reach < d                     # the last open of the last training label is before d


def test_window_start(cfg):
    # rolling_1y is 252 sessions. rolling_5y needs 1260 sessions, so row 1000 has no start. expanding needs 252 rows
    # from first_row (100): row 300 has 201 rows and row 400 has 301.
    assert walkforward.window_start("rolling_1y", 1000, 0, cfg) == 1000 - 252 + 1
    assert walkforward.window_start("rolling_5y", 1000, 0, cfg) is None
    assert walkforward.window_start("expanding", 300, 100, cfg) is None
    assert walkforward.window_start("expanding", 400, 100, cfg) == 100

