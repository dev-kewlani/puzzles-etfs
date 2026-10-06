"""Models. BoostedModels gives the same predictions, bit for bit, as a plain lgb.train call with the same params
and thread count."""
import lightgbm as lgb
import numpy as np

from etf66.models import BoostedModels

PARAMS = {"num_leaves": 7, "learning_rate": 0.1, "num_boost_round": 20, "min_data_in_leaf": 300, "max_bin": 31,
          "feature_fraction": 0.5, "bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 1.0, "seed": 3}


def test_boosted_models_match_a_plain_train_call():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 12))
    x[:, 3] = rng.integers(0, 3, 1500)           # a feature with 3 values, about 500 rows each
    y = x[:, 0] + 0.5 * x[:, 3] + rng.normal(size=1500)
    x_new = rng.normal(size=(50, 12))
    mine = BoostedModels(PARAMS, 2).fit(x, y[:, None]).predict(x_new)[:, 0]
    params = {**{k: v for k, v in PARAMS.items() if k != "num_boost_round"}, "objective": "regression", "verbose": -1,
              "num_threads": 2}
    plain = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=20).predict(x_new)
    assert np.array_equal(mine, plain)
