"""Granger-style lagged directed dependency tests."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from causal_weathergraph.candidate_edges import CandidateEdge
from causal_weathergraph.utils import get_logger

from .fdr import apply_fdr_by_group

logger = get_logger("causal.granger")


def run_granger_discovery(
    data: np.ndarray,
    variable_names: list[str],
    candidates: list[CandidateEdge],
    regimes: dict[str, np.ndarray],
    max_lag: int = 12,
    alpha: float = 0.05,
    controls: dict[str, Any] | None = None,
    fdr_method: str = "benjamini_hochberg",
    region_lat: np.ndarray | None = None,
    region_lon: np.ndarray | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run lagged OLS nested-model tests for every candidate edge and regime.

    The returned directed edges are best read as observational lagged
    dependencies, not as interventional proof.
    """
    if fdr_method != "benjamini_hochberg":
        logger.warning("Only benjamini_hochberg is implemented; using it.")
    controls = controls or {}
    max_lag = int(max_lag)
    var_index = {v: i for i, v in enumerate(variable_names)}
    records: list[dict[str, Any]] = []

    for regime_name, mask in regimes.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != data.shape[0]:
            logger.warning("Skipping regime %s because mask length does not match data.", regime_name)
            continue
        if int(mask.sum()) <= max_lag + 8:
            logger.warning("Skipping regime %s because it has too few samples.", regime_name)
            continue
        iterator = tqdm(
            candidates,
            desc=f"Granger tests [{regime_name}]",
            disable=not show_progress,
            leave=False,
        )
        for edge in iterator:
            if edge.source_var not in var_index or edge.target_var not in var_index:
                continue
            for lag in range(1, max_lag + 1):
                result = test_candidate_lag(
                    data=data,
                    var_index=var_index,
                    edge=edge,
                    regime_mask=mask,
                    lag=lag,
                    max_lag=max_lag,
                    controls=controls,
                )
                if result is None:
                    continue
                result["regime"] = regime_name
                result["method"] = "granger_ols"
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


def test_candidate_lag(
    data: np.ndarray,
    var_index: dict[str, int],
    edge: CandidateEdge,
    regime_mask: np.ndarray,
    lag: int,
    max_lag: int,
    controls: dict[str, Any],
) -> dict[str, Any] | None:
    """Test one candidate source at one lag using a nested OLS F-test."""
    source_idx = var_index[edge.source_var]
    target_idx = var_index[edge.target_var]
    n_time, _, n_var = data.shape
    if n_time <= max_lag + 5:
        return None

    y = data[max_lag:, edge.target_region, target_idx]
    row_mask = regime_mask[max_lag:].copy()

    restricted_cols: list[np.ndarray] = []
    restricted_names: list[str] = []
    include_own = bool(controls.get("include_target_own_lags", True))
    include_all = bool(controls.get("include_target_region_all_vars", True))

    if include_all:
        control_vars = range(n_var)
    elif include_own:
        control_vars = [target_idx]
    else:
        control_vars = []

    for control_var in control_vars:
        for control_lag in range(1, max_lag + 1):
            same_as_candidate = (
                edge.source_region == edge.target_region
                and control_var == source_idx
                and control_lag == lag
            )
            if same_as_candidate:
                continue
            restricted_cols.append(data[max_lag - control_lag : n_time - control_lag, edge.target_region, control_var])
            restricted_names.append(f"target_region_var{control_var}_lag{control_lag}")

    source_col = data[max_lag - lag : n_time - lag, edge.source_region, source_idx]
    x_restricted = np.column_stack(restricted_cols) if restricted_cols else np.empty((len(y), 0))
    x_full = np.column_stack([x_restricted, source_col])

    finite = np.isfinite(y) & np.isfinite(source_col) & row_mask
    if x_restricted.size:
        finite &= np.isfinite(x_restricted).all(axis=1)
    y = y[finite]
    x_restricted = x_restricted[finite]
    x_full = x_full[finite]

    if len(y) <= x_full.shape[1] + 6:
        return None
    if np.nanstd(source_col[finite]) < 1e-10 or np.nanstd(y) < 1e-10:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            restricted_model = sm.OLS(y, sm.add_constant(x_restricted, has_constant="add")).fit()
            full_model = sm.OLS(y, sm.add_constant(x_full, has_constant="add")).fit()
            f_stat, p_value, _ = full_model.compare_f_test(restricted_model)
        coef = float(full_model.params[-1])
        coefficient_se = float(full_model.bse[-1])
    except Exception:
        return None

    if not np.isfinite(p_value) or not np.isfinite(f_stat):
        return None
    rss_restricted = float(restricted_model.ssr)
    rss_full = float(full_model.ssr)
    rss_drop = max(0.0, rss_restricted - rss_full)
    tss = max(float(full_model.centered_tss), 1e-18)
    ci_half_width = 1.959963984540054 * coefficient_se

    return {
        "source_region": int(edge.source_region),
        "target_region": int(edge.target_region),
        "source_var": edge.source_var,
        "target_var": edge.target_var,
        "edge_type": edge.edge_type,
        "lag": int(lag),
        "p_value": float(p_value),
        "F_statistic": float(f_stat),
        "effect_coefficient": coef,
        "effect_abs": abs(coef),
        "effect_se_model": coefficient_se,
        "effect_ci95_low_model": float(coef - ci_half_width),
        "effect_ci95_high_model": float(coef + ci_half_width),
        "r2_restricted": float(restricted_model.rsquared),
        "r2_full": float(full_model.rsquared),
        "delta_r2": float(max(0.0, rss_drop / tss)),
        "partial_r2": float(max(0.0, rss_drop / max(rss_restricted, 1e-18))),
        "cohen_f2": float(max(0.0, rss_drop / max(rss_full, 1e-18))),
        "distance_km": float(edge.distance_km),
        "n_samples": int(len(y)),
        "n_controls": int(x_restricted.shape[1]),
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
