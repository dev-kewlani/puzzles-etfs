"""PCA market-state features (global, one value per session) and per-ETF factor features.

For each session t and window w, the code takes the w daily returns that end on t. It keeps the assets with at least
PCA["coverage"] of the window present. It standardizes each asset inside the window (missing = 0, clipped at
PCA["clip_sd"] standard deviations) and computes the eigenvalues of the correlation matrix. When there are more
assets N than sessions w, it uses the sessions x sessions Gram matrix, which has the same non-zero eigenvalues.
The measures (formulas from the owner's research code, which is not in this repo):
  eff_n        effective number of factors, exp(-sum p ln p), with p = eigenvalue shares
  k50/k70/k90  fractional number of factors for 50, 70 and 90% of the variance
  pc1          share of the first factor
  absorption   share of the top ceil(min(N, w) / absorption_divisor) factors. min(N, w) is the matrix rank; with
               N > w, ceil(N / 5) would cover every non-zero eigenvalue.
  avg_corr     mean off-diagonal correlation
  mp_share     share of variance above the Marchenko-Pastur edge (1 + sqrt(N / w))^2
  dispersion   mean over the window of the cross-sectional standard deviation of returns
  n_kept       the count of assets in the window; it shows the yearly cycle of the stock members
All values at t use returns up to t. The code uses two universes: the tradable ETFs, and the largest stocks of each
year. The stock members of a year are known at the end of the year before.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from etf66.features.context import Context

EPS = 1e-12                   # floor of a variance or a share in a division
K_LEVELS = {"k50": 0.5, "k70": 0.7, "k90": 0.9}

PCA = {
    "coverage": 0.9,                      # an asset stays in a window when this share of the window is present
    "clip_sd": 6.0,                       # standardized returns are clipped at this many standard deviations
    "absorption_divisor": 5,
    "measures": ["eff_n", "k50", "k70", "k90", "pc1", "absorption", "avg_corr", "mp_share", "dispersion", "n_kept"],
    "derived": {"ratio_measures": ["eff_n", "k70", "pc1", "avg_corr"], "ratio_lags": [5, 20],
                "cross_measures": ["eff_n", "avg_corr"], "absorption_shift": {"short": 15, "long": 252, "min_n": 200}},
    "etf_factor": {"window": 252, "min_assets": 10, "top_factors": 3, "rotation_lag": 20, "min_common": 10,
                   "loading_change_lag": 20},
    "etf_state": {"windows": [63, 126, 252], "min_assets": 20, "short": 63, "long": 252},
    "stock": {"windows": [90, 250], "min_assets": 200, "short": 90, "long": 250,
              "tercile": {"window": 756, "min_n": 250, "bins": 3}},
}


def fractional_k(cum: np.ndarray, level: float) -> float:
    """Return the fractional number of factors at which the cumulative share reaches `level`."""
    k = int(np.searchsorted(cum, level))
    prev = cum[k - 1] if k > 0 else 0.0
    nxt = cum[min(k, len(cum) - 1)]
    return k + (level - prev) / max(nxt - prev, EPS)


def window_spectrum(window: np.ndarray, min_assets: int, coverage: float, clip_sd: float):
    """Return (eigenvalues descending, eigenvectors or None, kept column positions, standardized matrix) of a window.

    window: (sessions, assets) returns. Return None when fewer than min_assets assets stay.
    """
    present = ~np.isnan(window)
    keep = np.where(present.mean(axis=0) >= coverage)[0]
    if len(keep) < min_assets:
        return None
    z = window[:, keep]
    mu = np.nanmean(z, axis=0)
    sd = np.nanstd(z, axis=0)
    good = sd > EPS
    keep, z, mu, sd = keep[good], z[:, good], mu[good], sd[good]
    if len(keep) < min_assets:
        return None
    z = np.clip(np.nan_to_num((z - mu) / sd), -clip_sd, clip_sd)
    z = z / np.sqrt(np.maximum((z ** 2).mean(axis=0), EPS))     # rescale after the clip: exact unit diagonal
    w, n = z.shape
    if n <= w:
        corr = z.T @ z / w
        lam, vec = np.linalg.eigh(corr)
        lam, vec = lam[::-1], vec[:, ::-1]
    else:
        lam = np.linalg.eigvalsh(z @ z.T / w)[::-1]
        vec = None
    lam = np.clip(lam, 0.0, None)
    return lam, vec, keep, z


def measures_from_spectrum(lam: np.ndarray, z: np.ndarray, raw: np.ndarray, absorption_divisor: int) -> dict:
    """Return the state measures of one window from its eigenvalues, standardized matrix and raw returns."""
    w, n = z.shape
    p = lam / lam.sum()
    cum = np.cumsum(p)
    nz = p[p > 0]
    rowsum = z.sum(axis=1)
    out = {"eff_n": float(np.exp(-(nz * np.log(nz)).sum()))}
    out.update({name: fractional_k(cum, level) for name, level in K_LEVELS.items()})
    out.update({
        "pc1": float(p[0]),
        "absorption": float(p[: int(math.ceil(min(n, w) / absorption_divisor))].sum()),
        "avg_corr": float(((rowsum ** 2).sum() / w - n) / (n * (n - 1))),
        "mp_share": float(p[lam > (1.0 + math.sqrt(n / w)) ** 2].sum()),
        "dispersion": float(np.nanmean(np.nanstd(raw, axis=1))),
        "n_kept": float(n),
    })
    return out


def state_series(returns: pd.DataFrame, windows, min_assets: int, prefix: str, pca: dict) -> pd.DataFrame:
    """Return the state measures for every session and window, with columns named <prefix>_<measure>_<w>."""
    x = returns.to_numpy(np.float64)
    measures = pca["measures"]
    rows = {}
    for w in windows:
        out = np.full((len(x), len(measures)), np.nan)
        for t in range(w - 1, len(x)):
            raw = x[t - w + 1: t + 1]
            spec = window_spectrum(raw, min_assets, pca["coverage"], pca["clip_sd"])
            if spec is None:
                continue
            lam, _vec, keep, z = spec
            m = measures_from_spectrum(lam, z, raw[:, keep], pca["absorption_divisor"])
            out[t] = [m[k] for k in measures]
        for i, k in enumerate(measures):
            rows[f"{prefix}_{k}_{w}"] = out[:, i]
    return pd.DataFrame(rows, index=returns.index)


def derived_state(base: pd.DataFrame, prefix: str, short: int, long: int, derived: dict) -> dict:
    """Return the ratio, cross-window and absorption-shift transforms of a state table."""
    f = {}
    for m in derived["ratio_measures"]:
        s = base[f"{prefix}_{m}_{short}"]
        for lag in derived["ratio_lags"]:
            f[f"{prefix}_{m}_{short}_ratio_l{lag}"] = s / s.shift(lag)
    for m in derived["cross_measures"]:
        f[f"{prefix}_{m}_{short}_over_{long}"] = base[f"{prefix}_{m}_{short}"] / base[f"{prefix}_{m}_{long}"]
    ab = derived["absorption_shift"]
    ar = base[f"{prefix}_absorption_{long}"]
    f[f"{prefix}_absorption_shift"] = (ar.rolling(ab["short"]).mean() - ar.rolling(ab["long"], min_periods=ab["min_n"]).mean()) \
        / ar.rolling(ab["long"], min_periods=ab["min_n"]).std()
    return f


def etf_factor_features(x: Context) -> tuple[dict, dict]:
    """Return per-ETF factor features (loadings, communality, idiosyncratic share) and two global ETF-state extras.

    Eigenvector signs are arbitrary. We flip PC1 so that its loadings sum to zero or more. We flip PC2 so that it
    agrees with the PC2 of the previous session.
    """
    pca = PCA
    p = pca["etf_factor"]
    w, top, lag = p["window"], p["top_factors"], p["rotation_lag"]
    ret = x.ret.where(x.tradable)
    arr = ret.to_numpy(np.float64)
    t_n, n = arr.shape
    load1, load2, comm = (np.full((t_n, n), np.nan) for _ in range(3))
    rotation = np.full(t_n, np.nan)
    prev_v2, prev_basis = None, {}
    for t in range(w - 1, t_n):
        spec = window_spectrum(arr[t - w + 1: t + 1], p["min_assets"], pca["coverage"], pca["clip_sd"])
        if spec is None:
            continue
        lam, vec, keep, _z = spec
        v1 = vec[:, 0] * (1.0 if vec[:, 0].sum() >= 0 else -1.0)
        v2 = vec[:, 1].copy()
        if prev_v2 is not None:
            common = np.intersect1d(keep, prev_v2[0])
            if len(common):
                a = v2[np.searchsorted(keep, common)]
                b = prev_v2[1][np.searchsorted(prev_v2[0], common)]
                if (a * b).sum() < 0:
                    v2 = -v2
        prev_v2 = (keep, v2)
        load1[t, keep] = v1 * math.sqrt(lam[0])
        load2[t, keep] = v2 * math.sqrt(lam[1])
        comm[t, keep] = (vec[:, :top] ** 2 * lam[:top]).sum(axis=1)
        prev_basis[t] = (keep, vec[:, :top])
        old = prev_basis.pop(t - lag, None)
        if old is not None:
            common = np.intersect1d(keep, old[0])
            if len(common) >= p["min_common"]:
                a = vec[np.searchsorted(keep, common), :top]
                b = old[1][np.searchsorted(old[0], common), :top]
                rotation[t] = float((np.linalg.svd(a.T @ b, compute_uv=False) ** 2).mean())
    idx, cols = x.ret.index, x.ret.columns
    comm_df = pd.DataFrame(comm, index=idx, columns=cols)
    per_etf = {
        "pc1_loading": pd.DataFrame(load1, index=idx, columns=cols),
        "pc2_loading": pd.DataFrame(load2, index=idx, columns=cols),
        f"factor_share_top{top}": comm_df,
        "idiosyncratic_share": 1.0 - comm_df,
    }
    k = p["loading_change_lag"]
    per_etf[f"pc1_loading_change_{k}"] = per_etf["pc1_loading"] - per_etf["pc1_loading"].shift(k)
    glob = {f"etf_rotation_overlap_{lag}": pd.Series(rotation, index=idx),
            "etf_idiosyncratic_share_median": (1.0 - comm_df).median(axis=1)}
    return per_etf, glob


def etf_state(x: Context) -> dict:
    """Return the global ETF PCA-state features and their transforms."""
    pca = PCA
    p = pca["etf_state"]
    base = state_series(x.ret.where(x.tradable), p["windows"], p["min_assets"], "etf", pca)
    out = {c: base[c] for c in base.columns}
    out.update(derived_state(base, "etf", p["short"], p["long"], pca["derived"]))
    return out


def stock_state(x: Context) -> dict:
    """Return the global stock PCA-state features, their transforms and the tercile of stock_eff_n_<short>."""
    if x.stock_returns.empty:
        return {}
    pca = PCA
    p = pca["stock"]
    base = state_series(x.stock_returns, p["windows"], p["min_assets"], "stock", pca)
    out = {c: base[c] for c in base.columns}
    out.update(derived_state(base, "stock", p["short"], p["long"], pca["derived"]))
    tc = p["tercile"]
    eff_n = base[f"stock_eff_n_{p['short']}"]
    out[f"stock_eff_n_{p['short']}_tercile"] = np.ceil(eff_n.rolling(tc["window"], min_periods=tc["min_n"]).rank(pct=True)
                                                       * float(tc["bins"]))
    return out
