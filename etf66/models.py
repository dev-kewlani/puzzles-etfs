"""Models. Each model fits on pooled (session, ETF) training rows and predicts one value per row and target.

Ridge: standardize the features with the training mean and std, and set a missing value to 0 (the training mean).
Then solve (X'X / n + alpha I) b = X'(Y - mean Y) / n for all target columns at once.
So alpha is a per-row penalty on standardized features.
With CuPy (device auto or gpu) the fit runs on the GPU in float64. On the CPU the standardized inputs are rounded
to model_defaults.ridge.cpu_precision and the products run in float64. The default is float32, so CPU and GPU
fits are not identical; float64 on the CPU is an open item.

LightGBM: one boosted tree model per target, with the params of the model entry (objective regression; 150 rounds
and max_bin 63 when the params omit them). Missing values stay missing. One binned dataset serves every target of
the same rows.
"""
from __future__ import annotations

import os
import warnings

import lightgbm as lgb
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import cupy as cp
        cp.zeros(1).sum()
        GPU_WORKS = True
    except Exception:  # noqa: BLE001 - no CuPy or a broken CUDA setup; the ridge fit falls back to the CPU
        cp, GPU_WORKS = None, False

STD_FLOOR = 1e-9              # a feature with a smaller training std is left unscaled
LGBM_ROUNDS = 150
LGBM_MAX_BIN = 63


class RidgeModel:
    """Multi-target ridge regression on standardized features."""

    def __init__(self, alpha: float, device: str, cpu_precision: str):
        self.alpha = float(alpha)
        self.cpu_dtype = np.dtype(cpu_precision)
        if device == "gpu" and not GPU_WORKS:
            raise RuntimeError("model_defaults.ridge.device is gpu, but CuPy does not work on this machine")
        self.use_gpu = GPU_WORKS and device in ("auto", "gpu")
        self.mean = self.std = self.coef = self.intercept = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeModel":
        """Fit every column of y on x. x: (rows, features); y: (rows, targets)."""
        return self._fit_gpu(x, y) if self.use_gpu else self._fit_cpu(x, y)

    def _fit_cpu(self, x: np.ndarray, y: np.ndarray) -> "RidgeModel":
        self.mean = np.nan_to_num(np.nanmean(x, axis=0))
        self.std = np.nanstd(x, axis=0)
        self.std[~(self.std > STD_FLOOR)] = 1.0
        z = np.nan_to_num((x - self.mean) / self.std, nan=0.0).astype(self.cpu_dtype)
        z64 = z.astype(np.float64)
        y_mean = y.mean(axis=0)
        n = len(z)
        a = z64.T @ z64 / n + self.alpha * np.eye(z.shape[1])
        b = (z64.T @ (y - y_mean)) / n
        self.coef = np.linalg.solve(a, b)
        self.intercept = y_mean
        return self

    def _fit_gpu(self, x: np.ndarray, y: np.ndarray) -> "RidgeModel":
        """Fit on the GPU. The float64 copy of the training rows lives in GPU memory, not in RAM."""
        xg = cp.asarray(x, dtype=cp.float64)
        mean = cp.nan_to_num(cp.nanmean(xg, axis=0))
        std = cp.nanstd(xg, axis=0)
        std = cp.where(std > STD_FLOOR, std, 1.0)
        xg -= mean
        xg /= std
        cp.nan_to_num(xg, copy=False, nan=0.0)
        n = xg.shape[0]
        yg = cp.asarray(y, dtype=cp.float64)
        y_mean = yg.mean(axis=0)
        a = xg.T @ xg / n + self.alpha * cp.eye(xg.shape[1])
        b = xg.T @ (yg - y_mean) / n
        self.coef = cp.asnumpy(cp.linalg.solve(a, b))
        self.mean, self.std, self.intercept = cp.asnumpy(mean), cp.asnumpy(std), cp.asnumpy(y_mean)
        del xg, yg
        cp.get_default_memory_pool().free_all_blocks()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predictions (rows, targets)."""
        z = np.nan_to_num((x - self.mean) / self.std, nan=0.0)
        return z @ self.coef + self.intercept


class BoostedModels:
    """One LightGBM model per target, trained on one shared binned dataset."""

    def __init__(self, params: dict, threads: int | None = None):
        self.params = dict(params)
        self.rounds = int(self.params.pop("num_boost_round", LGBM_ROUNDS))
        self.params.setdefault("max_bin", LGBM_MAX_BIN)
        self.params.update({"objective": "regression", "verbose": -1, "num_threads": threads or os.cpu_count()})
        self.boosters = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BoostedModels":
        """Fit one model for each column of y. x: (rows, features); y: (rows, targets)."""
        # free_raw_data: the dataset keeps no raw copy, so the job holds the training rows once.
        # The dataset gets the full params because LightGBM pre-filters features with min_data_in_leaf at
        # construction. With max_bin only, the pre-filter uses the default of 20 and the trees differ. With the full
        # params the result equals a plain lgb.train call with the same params and thread count.
        data = lgb.Dataset(x, label=y[:, 0], free_raw_data=True, params=dict(self.params))
        data.construct()
        self.boosters = []
        for j in range(y.shape[1]):
            data.set_label(y[:, j])
            self.boosters.append(lgb.train(self.params, data, num_boost_round=self.rounds))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predictions (rows, targets)."""
        return np.column_stack([b.predict(x, num_threads=self.params["num_threads"]) for b in self.boosters])


def make_model(spec: dict, cfg: dict, threads: int | None = None):
    """Return a new model object for a model entry of the settings."""
    if spec["kind"] == "ridge":
        ridge = cfg["model_defaults"]["ridge"]
        return RidgeModel(spec["alpha"], ridge["device"], ridge["cpu_precision"])
    if spec["kind"] == "lightgbm":
        return BoostedModels(spec.get("params") or {}, threads)
    raise ValueError(f"unknown model kind {spec['kind']}")
