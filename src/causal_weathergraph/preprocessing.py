"""Preprocessing for atmospheric time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data_io import AtmosphericDataset
from .utils import get_logger

logger = get_logger("preprocessing")


def preprocess_dataset(
    dataset: AtmosphericDataset,
    config: dict[str, Any] | None = None,
) -> tuple[AtmosphericDataset, dict[str, np.ndarray]]:
    """Clean, anomaly-transform, optionally difference, and standardize data.

    The operations are intentionally conservative. They improve comparability
    of lagged directed-dependency tests but do not make observational results
    equivalent to interventional causal effects.
    """
    config = config or {}
    data = np.asarray(dataset.data, dtype=float).copy()
    max_nan_ratio = float(config.get("max_nan_ratio_per_node", 0.2))

    node_nan_ratio = np.mean(~np.isfinite(data), axis=(0, 2))
    keep_nodes = node_nan_ratio <= max_nan_ratio
    if not np.any(keep_nodes):
        raise ValueError("All nodes exceeded the configured NaN ratio threshold.")
    if np.sum(~keep_nodes):
        logger.info("Dropping %d nodes with too many missing values.", int(np.sum(~keep_nodes)))
    data = data[:, keep_nodes, :]
    lat = dataset.lat[keep_nodes]
    lon = dataset.lon[keep_nodes]
    node_ids = dataset.node_ids[keep_nodes] if dataset.node_ids is not None else None

    data = fill_missing_values(data, interpolation_limit=int(config.get("interpolation_limit", 6)))

    if bool(config.get("remove_monthly_climatology", True)):
        data = remove_seasonal_component(
            data,
            dataset.timestamps,
            rolling_window=int(config.get("rolling_anomaly_window", 31)),
        )

    timestamps = dataset.timestamps
    if bool(config.get("use_first_difference", False)):
        data = np.diff(data, axis=0)
        if timestamps is not None:
            timestamps = timestamps[1:]

    stats: dict[str, np.ndarray] = {
        "keep_nodes": keep_nodes,
        "node_nan_ratio": node_nan_ratio,
    }
    if bool(config.get("standardize", True)):
        data, mean, std = standardize_per_node(data)
        stats["mean"] = mean
        stats["std"] = std

    processed = AtmosphericDataset(
        data=data,
        variable_names=list(dataset.variable_names),
        lat=lat,
        lon=lon,
        timestamps=timestamps,
        node_ids=node_ids,
        metadata={**dataset.metadata, "preprocessed": True},
    )
    return processed, stats


def fill_missing_values(data: np.ndarray, interpolation_limit: int = 6) -> np.ndarray:
    """Interpolate short gaps, then forward/backward fill remaining gaps."""
    filled = np.asarray(data, dtype=float).copy()
    n_time, n_node, n_var = filled.shape
    iterator = tqdm(
        range(n_node * n_var),
        desc="Filling missing values",
        leave=False,
        disable=(n_node * n_var < 256),
    )
    for flat_idx in iterator:
        node = flat_idx // n_var
        var = flat_idx % n_var
        series = pd.Series(filled[:, node, var])
        if series.isna().all():
            filled[:, node, var] = 0.0
            continue
        series = series.interpolate(limit=interpolation_limit, limit_direction="both")
        series = series.ffill().bfill()
        filled[:, node, var] = series.to_numpy(dtype=float)
    if not np.isfinite(filled).all():
        logger.warning("Non-finite values remained after filling; replacing with zero.")
        filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
    return filled


def remove_seasonal_component(
    data: np.ndarray,
    timestamps: pd.DatetimeIndex | None,
    rolling_window: int = 31,
) -> np.ndarray:
    """Remove monthly climatology when possible, otherwise remove a rolling mean."""
    anomalies = np.asarray(data, dtype=float).copy()
    if timestamps is not None and len(timestamps) == data.shape[0]:
        months = pd.DatetimeIndex(timestamps).month.to_numpy()
        for month in range(1, 13):
            mask = months == month
            if not np.any(mask):
                continue
            climatology = np.nanmean(anomalies[mask], axis=0, keepdims=True)
            anomalies[mask] = anomalies[mask] - climatology
        logger.info("Removed monthly climatology.")
        return anomalies

    window = max(3, int(rolling_window))
    logger.info("No usable timestamps found; removing rolling mean with window=%d.", window)
    n_time, n_node, n_var = anomalies.shape
    for node in range(n_node):
        for var in range(n_var):
            series = pd.Series(anomalies[:, node, var])
            rolling = series.rolling(window=window, center=True, min_periods=1).mean()
            anomalies[:, node, var] = (series - rolling).to_numpy(dtype=float)
    return anomalies


def standardize_per_node(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each node-variable series across time."""
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    standardized = (data - mean[None, :, :]) / std[None, :, :]
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    return standardized, mean, std
