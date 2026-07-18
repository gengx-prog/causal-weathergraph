"""Humidity mediation checks for Wind -> Cloud lagged dependencies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_humidity_mediation(
    data: np.ndarray,
    variable_names: list[str],
    wind_cloud_edges: pd.DataFrame,
    regimes: dict[str, np.ndarray],
    max_lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit direct and humidity-controlled Wind -> Cloud models."""
    if wind_cloud_edges.empty:
        empty_details = pd.DataFrame(
            columns=[
                "regime",
                "source_region",
                "target_region",
                "lag",
                "beta_direct",
                "beta_controlled",
                "reduction",
                "n_samples",
            ]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "regime",
                "edge_count",
                "mean_beta_direct",
                "mean_beta_controlled",
                "mean_reduction",
                "fraction_reduction_gt_30",
                "fraction_reduction_gt_50",
            ]
        )
        return empty_details, empty_summary

    var_index = {v: i for i, v in enumerate(variable_names)}
    required = {"wind", "humidity", "cloud_cover"}
    missing = required.difference(var_index)
    if missing:
        raise ValueError(f"Missing required variables for mediation test: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    for row in wind_cloud_edges.itertuples(index=False):
        regime_name = str(row.regime)
        if regime_name not in regimes:
            continue
        result = fit_wind_cloud_edge(
            data=data,
            var_index=var_index,
            regime_mask=regimes[regime_name],
            source_region=int(row.source_region),
            target_region=int(row.target_region),
            lag=int(row.lag),
            max_lag=max_lag,
        )
        if result is None:
            continue
        records.append(
            {
                "regime": regime_name,
                "source_region": int(row.source_region),
                "target_region": int(row.target_region),
                "lag": int(row.lag),
                **result,
            }
        )

    details = pd.DataFrame.from_records(records)
    if details.empty:
        return details, pd.DataFrame()

    summary_records = []
    for regime, df in details.groupby("regime", sort=True):
        reductions = df["reduction"].replace([np.inf, -np.inf], np.nan).dropna()
        summary_records.append(
            {
                "regime": regime,
                "edge_count": int(len(df)),
                "mean_beta_direct": float(df["beta_direct"].mean()),
                "mean_beta_controlled": float(df["beta_controlled"].mean()),
                "mean_reduction": float(reductions.mean()) if len(reductions) else np.nan,
                "fraction_reduction_gt_30": float((reductions > 0.30).mean()) if len(reductions) else np.nan,
                "fraction_reduction_gt_50": float((reductions > 0.50).mean()) if len(reductions) else np.nan,
            }
        )
    return details, pd.DataFrame.from_records(summary_records)


def fit_wind_cloud_edge(
    data: np.ndarray,
    var_index: dict[str, int],
    regime_mask: np.ndarray,
    source_region: int,
    target_region: int,
    lag: int,
    max_lag: int,
) -> dict[str, Any] | None:
    """Fit one direct model and one humidity-controlled model."""
    n_time = data.shape[0]
    if n_time <= max_lag + 8:
        return None

    cloud_idx = var_index["cloud_cover"]
    humidity_idx = var_index["humidity"]
    wind_idx = var_index["wind"]

    y = data[max_lag:, target_region, cloud_idx]
    row_mask = np.asarray(regime_mask, dtype=bool)[max_lag:].copy()

    cloud_lags = [
        data[max_lag - control_lag : n_time - control_lag, target_region, cloud_idx]
        for control_lag in range(1, max_lag + 1)
    ]
    humidity_lags = [
        data[max_lag - control_lag : n_time - control_lag, target_region, humidity_idx]
        for control_lag in range(1, max_lag + 1)
    ]
    wind_col = data[max_lag - lag : n_time - lag, source_region, wind_idx]

    x_direct = np.column_stack([*cloud_lags, wind_col])
    x_controlled = np.column_stack([*cloud_lags, *humidity_lags, wind_col])
    finite = np.isfinite(y) & np.isfinite(x_direct).all(axis=1) & np.isfinite(x_controlled).all(axis=1) & row_mask
    y_fit = y[finite]
    x_direct_fit = x_direct[finite]
    x_controlled_fit = x_controlled[finite]
    if len(y_fit) <= x_controlled_fit.shape[1] + 6:
        return None
    if np.nanstd(y_fit) < 1e-10 or np.nanstd(wind_col[finite]) < 1e-10:
        return None

    beta_direct = _last_beta(y_fit, x_direct_fit)
    beta_controlled = _last_beta(y_fit, x_controlled_fit)
    if not np.isfinite(beta_direct) or not np.isfinite(beta_controlled):
        return None
    reduction = np.nan
    if abs(beta_direct) > 1e-12:
        reduction = 1.0 - abs(beta_controlled) / abs(beta_direct)
    return {
        "beta_direct": float(beta_direct),
        "beta_controlled": float(beta_controlled),
        "reduction": float(reduction),
        "n_samples": int(len(y_fit)),
    }


def _last_beta(y: np.ndarray, x: np.ndarray) -> float:
    x_with_intercept = np.column_stack([np.ones(len(y), dtype=float), x])
    beta, *_ = np.linalg.lstsq(x_with_intercept, y, rcond=None)
    return float(beta[-1])
