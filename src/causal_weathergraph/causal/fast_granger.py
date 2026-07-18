"""Fast residualized Granger-style tests for repeated target controls."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution

from causal_weathergraph.candidate_edges import CandidateEdge

from .fdr import apply_fdr_by_group


def run_fast_granger_discovery(
    data: np.ndarray,
    variable_names: list[str],
    candidates: list[CandidateEdge],
    regimes: dict[str, np.ndarray],
    max_lag: int,
    alpha: float = 0.05,
    controls: dict[str, Any] | None = None,
    fdr_method: str = "benjamini_hochberg",
    region_lat: np.ndarray | None = None,
    region_lon: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run nested OLS F-tests while reusing each target-control QR fit.

    This is algebraically equivalent to testing a single additional source lag
    after target-region controls, but avoids refitting the restricted model for
    every candidate-lag pair.
    """
    if fdr_method != "benjamini_hochberg":
        fdr_method = "benjamini_hochberg"
    controls = controls or {}
    max_lag = int(max_lag)
    var_index = {v: i for i, v in enumerate(variable_names)}
    grouped: dict[tuple[int, str], list[CandidateEdge]] = defaultdict(list)
    for edge in candidates:
        if edge.source_var in var_index and edge.target_var in var_index:
            grouped[(int(edge.target_region), str(edge.target_var))].append(edge)

    records: list[dict[str, Any]] = []
    for regime_name, mask in regimes.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != data.shape[0] or int(mask.sum()) <= max_lag + 8:
            continue
        for (target_region, target_var), edges in grouped.items():
            target_idx = var_index[target_var]
            shared_fit = prepare_target_fit(
                data,
                variable_names,
                target_region,
                target_idx,
                mask,
                max_lag,
                controls,
            )
            if shared_fit is None:
                continue
            edge_specific_fits: dict[tuple[int, int], dict[str, Any] | None] = {}
            for edge in edges:
                source_idx = var_index[edge.source_var]
                for lag in range(1, max_lag + 1):
                    fit = shared_fit
                    if (
                        bool(controls.get("include_target_region_all_vars", False))
                        and int(edge.source_region) == int(target_region)
                    ):
                        cache_key = (source_idx, lag)
                        if cache_key not in edge_specific_fits:
                            edge_specific_fits[cache_key] = prepare_target_fit(
                                data,
                                variable_names,
                                target_region,
                                target_idx,
                                mask,
                                max_lag,
                                controls,
                                exclude_control=cache_key,
                            )
                        fit = edge_specific_fits[cache_key]
                    if fit is None:
                        continue
                    result = test_residualized_source(
                        data=data,
                        fit=fit,
                        edge=edge,
                        source_idx=source_idx,
                        lag=lag,
                        max_lag=max_lag,
                    )
                    if result is None:
                        continue
                    result["regime"] = regime_name
                    result["method"] = "granger_ols_fast"
                    if region_lat is not None and region_lon is not None:
                        result["source_lat"] = float(region_lat[edge.source_region])
                        result["source_lon"] = float(region_lon[edge.source_region])
                        result["target_lat"] = float(region_lat[edge.target_region])
                        result["target_lon"] = float(region_lon[edge.target_region])
                    records.append(result)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return _empty_edge_table()
    df = apply_fdr_by_group(df, ["regime", "target_var"], alpha=alpha)
    df["effect_score"] = (
        np.sign(df["effect_coefficient"].to_numpy(dtype=float))
        * (-np.log10(df["q_value"].to_numpy(dtype=float) + 1e-12))
        * np.abs(df["effect_coefficient"].to_numpy(dtype=float))
    )
    return df.sort_values(["regime", "q_value", "p_value", "effect_abs"]).reset_index(drop=True)


def prepare_target_fit(
    data: np.ndarray,
    variable_names: list[str],
    target_region: int,
    target_idx: int,
    regime_mask: np.ndarray,
    max_lag: int,
    controls: dict[str, Any],
    exclude_control: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    n_time, _, n_var = data.shape
    if n_time <= max_lag + 5:
        return None

    y = data[max_lag:, target_region, target_idx]
    row_mask = regime_mask[max_lag:].copy()
    include_own = bool(controls.get("include_target_own_lags", True))
    include_all = bool(controls.get("include_target_region_all_vars", False))
    if include_all:
        control_vars = list(range(n_var))
    elif include_own:
        control_vars = [target_idx]
    else:
        control_vars = []

    cols = []
    for control_var in control_vars:
        for control_lag in range(1, max_lag + 1):
            if exclude_control == (control_var, control_lag):
                continue
            cols.append(data[max_lag - control_lag : n_time - control_lag, target_region, control_var])
    x = np.column_stack(cols) if cols else np.empty((len(y), 0))
    finite = np.isfinite(y) & row_mask
    if x.size:
        finite &= np.isfinite(x).all(axis=1)
    if int(finite.sum()) <= x.shape[1] + 7:
        return None

    y_fit = y[finite]
    x_fit = x[finite]
    base = np.column_stack([np.ones(len(y_fit), dtype=float), x_fit])
    q, _ = np.linalg.qr(base, mode="reduced")
    y_resid = y_fit - q @ (q.T @ y_fit)
    rss_restricted = float(y_resid @ y_resid)
    y_centered = y_fit - float(np.mean(y_fit))
    tss = float(y_centered @ y_centered)
    if not np.isfinite(rss_restricted) or np.nanstd(y_fit) < 1e-10:
        return None
    return {
        "finite": finite,
        "q": q,
        "y_resid": y_resid,
        "rss_restricted": rss_restricted,
        "tss": tss,
        "n_controls": int(x.shape[1]),
    }


def test_residualized_source(
    data: np.ndarray,
    fit: dict[str, Any],
    edge: CandidateEdge,
    source_idx: int,
    lag: int,
    max_lag: int,
) -> dict[str, Any] | None:
    n_time = data.shape[0]
    source_col_full = data[max_lag - lag : n_time - lag, edge.source_region, source_idx]
    finite = fit["finite"]
    source_col = source_col_full[finite]
    if not np.isfinite(source_col).all() or np.nanstd(source_col) < 1e-10:
        return None

    q = fit["q"]
    z_resid = source_col - q @ (q.T @ source_col)
    z_ss = float(z_resid @ z_resid)
    if z_ss <= 1e-12 or not np.isfinite(z_ss):
        return None

    y_resid = fit["y_resid"]
    zy = float(z_resid @ y_resid)
    coef = zy / z_ss
    rss_restricted = float(fit["rss_restricted"])
    rss_drop = max(0.0, zy * zy / z_ss)
    rss_full = max(rss_restricted - rss_drop, 1e-18)
    n_samples = int(len(source_col))
    n_controls = int(fit["n_controls"])
    df_full = n_samples - n_controls - 2
    if df_full <= 0:
        return None
    f_stat = (rss_drop) / (rss_full / df_full)
    p_value = float(f_distribution.sf(f_stat, 1, df_full))
    if not np.isfinite(p_value) or not np.isfinite(f_stat):
        return None
    residual_variance = rss_full / df_full
    coefficient_se = float(np.sqrt(max(residual_variance / z_ss, 0.0)))
    ci_half_width = 1.959963984540054 * coefficient_se
    tss = max(float(fit.get("tss", 0.0)), 1e-18)
    r2_restricted = float(np.clip(1.0 - rss_restricted / tss, -np.inf, 1.0))
    r2_full = float(np.clip(1.0 - rss_full / tss, -np.inf, 1.0))
    delta_r2 = float(max(0.0, rss_drop / tss))
    partial_r2 = float(max(0.0, rss_drop / max(rss_restricted, 1e-18)))
    cohen_f2 = float(max(0.0, rss_drop / rss_full))

    return {
        "source_region": int(edge.source_region),
        "target_region": int(edge.target_region),
        "source_var": edge.source_var,
        "target_var": edge.target_var,
        "edge_type": edge.edge_type,
        "lag": int(lag),
        "p_value": p_value,
        "F_statistic": float(f_stat),
        "effect_coefficient": float(coef),
        "effect_abs": float(abs(coef)),
        "effect_se_model": coefficient_se,
        "effect_ci95_low_model": float(coef - ci_half_width),
        "effect_ci95_high_model": float(coef + ci_half_width),
        "r2_restricted": r2_restricted,
        "r2_full": r2_full,
        "delta_r2": delta_r2,
        "partial_r2": partial_r2,
        "cohen_f2": cohen_f2,
        "distance_km": float(edge.distance_km),
        "n_samples": n_samples,
        "n_controls": n_controls,
    }


def _empty_edge_table() -> pd.DataFrame:
    columns = [
        "source_region",
        "target_region",
        "source_var",
        "target_var",
        "edge_type",
        "lag",
        "p_value",
        "F_statistic",
        "effect_coefficient",
        "effect_abs",
        "effect_se_model",
        "effect_ci95_low_model",
        "effect_ci95_high_model",
        "r2_restricted",
        "r2_full",
        "delta_r2",
        "partial_r2",
        "cohen_f2",
        "distance_km",
        "regime",
        "n_samples",
        "method",
        "q_value",
        "significant",
        "effect_score",
    ]
    return pd.DataFrame(columns=columns)
