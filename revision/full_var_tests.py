"""Dense VAR(p) coefficient tests for arbitrary region-variable source/target pairs.

All 66 x V observed regional variables enter every equation at lags 1..p
(no penalty, exact full-rank QR). Unlike run_full_var.py, every node can be an
outcome, so reverse-direction candidates (e.g. cloud -> humidity, humidity ->
wind) are tested under the same conditioning set as the forward candidates.
Inference uses Bartlett HAC on the contiguous calendar of the fitted period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg, stats

from revision.inference import score_autocovariances, adjust_bh


def build_design(data: np.ndarray, history_lags: int = 3):
    """Rows are response times t = history_lags..T-1 (index into the calendar)."""
    t, r, v = data.shape
    x = np.column_stack([np.ones(t - history_lags)] + [data[history_lags - l:t - l].reshape(t - history_lags, -1)
                                                      for l in range(1, history_lags + 1)])
    y = data[history_lags:].reshape(t - history_lags, -1)
    return x, y


def column_of(source_region, source_var_index, lag, n_regions, n_vars):
    return 1 + (lag - 1) * n_regions * n_vars + source_region * n_vars + source_var_index


def outcome_of(target_region, target_var_index, n_vars):
    return target_region * n_vars + target_var_index


def _qr(x):
    q, r = linalg.qr(x, mode="economic", check_finite=False)
    sv = linalg.svdvals(r, check_finite=False)
    if int(np.count_nonzero(sv > max(x.shape) * np.finfo(float).eps * sv[0])) != x.shape[1]:
        raise ValueError("Rank-deficient dense VAR design")
    inv_r = linalg.solve_triangular(r, np.eye(x.shape[1]), check_finite=False)
    return q, r, inv_r, float(sv[0] / sv[-1])


def _hac_se(scores, n_params, bandwidth):
    n = len(scores)
    acov = score_autocovariances(scores, bandwidth)
    h = np.arange(1, len(acov))
    meat = acov[0] + 2 * np.sum((1 - h[:, None] / (bandwidth + 1)) * acov[h], axis=0)
    return np.sqrt(np.maximum(meat * n / (n - n_params), 0))


def fit_tests(x, y, tests: pd.DataFrame, bandwidth: int = 64):
    """tests needs integer columns 'column' and 'outcome'. Rows of x,y = one contiguous period."""
    n, p = x.shape
    q, r, inv_r, cond = _qr(x)
    coef = linalg.solve_triangular(r, q.T @ y, check_finite=False)
    resid = y - x @ coef
    inv_gram_diag = np.sum(inv_r * inv_r, axis=1)
    rss = np.sum(resid ** 2, axis=0)
    tss = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    cols = np.sort(tests.column.unique())
    lookup = {c: k for k, c in enumerate(cols)}
    directions = q @ inv_r.T[:, cols]
    out = tests.copy()
    beta = np.empty(len(out)); se_h = np.empty(len(out)); se_o = np.empty(len(out))
    for o, idx in out.groupby("outcome").groups.items():
        idx = np.asarray(list(idx))
        cc = out.loc[idx, "column"].to_numpy(int)
        b = coef[cc, o]
        sc = directions[:, [lookup[c] for c in cc]] * resid[:, o, None]
        beta[out.index.get_indexer(idx)] = b
        se_h[out.index.get_indexer(idx)] = _hac_se(sc, p, bandwidth)
        se_o[out.index.get_indexer(idx)] = np.sqrt(rss[o] / (n - p) * inv_gram_diag[cc])
    out["effect"] = beta
    out[f"se_hac{bandwidth}"] = se_h
    out[f"p_hac{bandwidth}"] = 2 * stats.norm.sf(np.abs(beta) / np.maximum(se_h, 1e-300))
    out["se_ols"] = se_o
    out["p_ols"] = 2 * stats.t.sf(np.abs(beta) / np.maximum(se_o, 1e-300), n - p)
    drop = beta ** 2 / inv_gram_diag[out.column.to_numpy(int)]
    out["partial_r2"] = drop / (rss[out.outcome.to_numpy(int)] + drop)
    out["delta_r2"] = drop / tss[out.outcome.to_numpy(int)]
    out["n_samples"] = n
    out["n_parameters"] = p
    fit = {"coef": coef, "inv_gram": inv_r @ inv_r.T, "condition_number": cond}
    return out, fit


def frozen_gains(fit, x_test, y_test, tests: pd.DataFrame, bandwidth: int = 64):
    """Exact drop-one-coefficient restricted refit on training, both frozen; evaluate on test rows."""
    coef, inv_gram = fit["coef"], fit["inv_gram"]
    err_full = y_test - x_test @ coef
    cols = np.sort(tests.column.unique())
    lookup = {c: k for k, c in enumerate(cols)}
    directions = x_test @ inv_gram[:, cols]
    out = tests.copy()
    gain = np.empty(len(out)); se = np.empty(len(out)); mse = np.empty(len(out))
    for o, idx in out.groupby("outcome").groups.items():
        idx = np.asarray(list(idx))
        pos = out.index.get_indexer(idx)
        cc = out.loc[idx, "column"].to_numpy(int)
        delta = directions[:, [lookup[c] for c in cc]] * (coef[cc, o] / np.diag(inv_gram)[cc])
        e = err_full[:, o, None]
        g = (e + delta) ** 2 - e ** 2
        m = g.mean(axis=0)
        gain[pos] = m
        se[pos] = _hac_se(g - m, 1, bandwidth) / len(g)
        mse[pos] = float(np.mean(e ** 2))
    out["mse_gain"] = gain
    out["mse_gain_se_hac64"] = se
    out["p_gain"] = 2 * stats.norm.sf(np.abs(gain) / np.maximum(se, 1e-300))
    out["p_gain_one_sided"] = stats.norm.sf(gain / np.maximum(se, 1e-300))
    out["mse_full"] = mse
    out["relative_gain"] = gain / mse
    return out


def attach_specs(tests: pd.DataFrame, names, n_regions):
    vi = {n: i for i, n in enumerate(names)}
    v = len(names)
    tests = tests.copy()
    tests["column"] = [column_of(int(a), vi[b], int(l), n_regions, v) for a, b, l in zip(tests.source_region, tests.source_var, tests.lag)]
    tests["outcome"] = [outcome_of(int(a), vi[b], v) for a, b in zip(tests.target_region, tests.target_var)]
    return tests.reset_index(drop=True)


def expand_lags(candidates: pd.DataFrame, lags=(1, 2, 3)):
    rows = []
    for lag in lags:
        c = candidates.copy()
        c["lag"] = lag
        rows.append(c)
    return pd.concat(rows, ignore_index=True)


__all__ = ["build_design", "fit_tests", "frozen_gains", "attach_specs", "expand_lags", "adjust_bh"]
