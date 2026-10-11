"""Regime definitions for atmospheric causal discovery."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger("regimes")


def define_regimes(
    data: np.ndarray,
    variable_names: list[str],
    timestamps: pd.DatetimeIndex | None,
    config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Create boolean time masks for configured regimes."""
    config = config or {}
    n_time = data.shape[0]
    regimes: dict[str, np.ndarray] = {"all": np.ones(n_time, dtype=bool)}

    if bool(config.get("use_season", True)):
        seasonal = season_regimes(timestamps, n_time)
        requested = config.get("seasons", ["DJF", "JJA"])
        regimes.update({name: mask for name, mask in seasonal.items() if name in requested})

    high_q = float(config.get("high_quantile", 0.8))
    low_q = float(config.get("low_quantile", 0.2))

    if bool(config.get("use_humidity_state", True)):
        humidity_idx = _var_index(variable_names, "humidity")
        if humidity_idx is not None:
            humidity_regimes = quantile_state_regimes(data[:, :, humidity_idx], "humidity", low_q, high_q)
            if not bool(config.get("include_low_humidity", False)):
                humidity_regimes.pop("low_humidity", None)
            regimes.update(humidity_regimes)
        else:
            logger.warning("Humidity variable unavailable; humidity-state regimes skipped.")

    if bool(config.get("use_cloud_extreme", True)):
        cloud_idx = _var_index(variable_names, "cloud_cover")
        if cloud_idx is not None:
            cloud = np.nanmean(data[:, :, cloud_idx], axis=1)
            threshold = np.nanquantile(cloud, high_q)
            regimes["high_cloud"] = cloud >= threshold
            regimes["normal_cloud"] = cloud < threshold
        else:
            logger.warning("Cloud cover variable unavailable; cloud-extreme regimes skipped.")

    return {name: mask.astype(bool) for name, mask in regimes.items() if mask.shape[0] == n_time}


def season_regimes(
    timestamps: pd.DatetimeIndex | None,
    n_time: int,
) -> dict[str, np.ndarray]:
    """Return DJF/MAM/JJA/SON masks when timestamps are available."""
    if timestamps is None or len(timestamps) != n_time:
        logger.warning("Timestamps unavailable; seasonal regimes skipped.")
        return {}
    months = pd.DatetimeIndex(timestamps).month.to_numpy()
    return {
        "DJF": np.isin(months, [12, 1, 2]),
        "MAM": np.isin(months, [3, 4, 5]),
        "JJA": np.isin(months, [6, 7, 8]),
        "SON": np.isin(months, [9, 10, 11]),
    }


def quantile_state_regimes(
    values: np.ndarray,
    name: str,
    low_quantile: float,
    high_quantile: float,
) -> dict[str, np.ndarray]:
    """Create low, normal, and high regimes from global mean quantiles."""
    series = np.nanmean(values, axis=1)
    low = np.nanquantile(series, low_quantile)
    high = np.nanquantile(series, high_quantile)
    return {
        f"low_{name}": series <= low,
        f"normal_{name}": (series > low) & (series < high),
        f"high_{name}": series >= high,
    }


def latitude_band_masks(lat: np.ndarray) -> dict[str, np.ndarray]:
    """Return region masks for tropical, midlatitude, and polar target regions."""
    abs_lat = np.abs(np.asarray(lat, dtype=float))
    return {
        "tropical": abs_lat < 23.5,
        "midlatitude": (abs_lat >= 23.5) & (abs_lat < 60.0),
        "polar": abs_lat >= 60.0,
    }


def _var_index(variable_names: list[str], target: str) -> int | None:
    try:
        return variable_names.index(target)
    except ValueError:
        return None
