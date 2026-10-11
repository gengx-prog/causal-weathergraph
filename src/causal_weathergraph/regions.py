"""Spatial aggregation from nodes/grid cells to analysis regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .data_io import AtmosphericDataset, save_dataset_npz
from .utils import ensure_dir, get_logger

logger = get_logger("regions")


@dataclass
class RegionAggregation:
    """Result of aggregating node-level data into regions."""

    dataset: AtmosphericDataset
    node_to_region: np.ndarray
    metadata: pd.DataFrame


def aggregate_regions(
    dataset: AtmosphericDataset,
    config: dict[str, Any] | None = None,
) -> RegionAggregation:
    """Aggregate node-level data to region-level mean time series."""
    config = config or {}
    method = str(config.get("method", "latlon_bins"))
    n_regions = min(int(config.get("n_regions", 64)), dataset.data.shape[1])

    if n_regions <= 0:
        raise ValueError("n_regions must be positive.")

    if method == "kmeans_sphere":
        mapping = assign_kmeans_sphere(
            dataset.lat,
            dataset.lon,
            n_regions=n_regions,
            random_seed=int(config.get("random_seed", 42)),
        )
    elif method == "latlon_bins":
        mapping = assign_latlon_bins(dataset.lat, dataset.lon, n_regions=n_regions)
    else:
        raise ValueError(f"Unknown region aggregation method: {method}")

    mapping = _remap_consecutive(mapping)
    n_actual = int(mapping.max()) + 1
    region_data = np.zeros((dataset.data.shape[0], n_actual, dataset.data.shape[2]), dtype=float)
    centroid_lat = np.zeros(n_actual, dtype=float)
    centroid_lon = np.zeros(n_actual, dtype=float)
    counts = np.zeros(n_actual, dtype=int)

    for region in range(n_actual):
        mask = mapping == region
        counts[region] = int(np.sum(mask))
        region_data[:, region, :] = np.nanmean(dataset.data[:, mask, :], axis=1)
        centroid_lat[region] = np.nanmean(dataset.lat[mask]) if np.isfinite(dataset.lat[mask]).any() else np.nan
        centroid_lon[region] = _mean_longitude(dataset.lon[mask])

    if not np.isfinite(centroid_lat).any():
        centroid_lat = np.linspace(-60.0, 60.0, n_actual)
    if not np.isfinite(centroid_lon).any():
        centroid_lon = np.linspace(-180.0, 180.0, n_actual, endpoint=False)

    metadata = pd.DataFrame(
        {
            "region": np.arange(n_actual),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "n_nodes": counts,
            "latitude_band": [latitude_band(lat) for lat in centroid_lat],
        }
    )
    region_dataset = AtmosphericDataset(
        data=region_data,
        variable_names=list(dataset.variable_names),
        lat=centroid_lat,
        lon=centroid_lon,
        timestamps=dataset.timestamps,
        node_ids=np.arange(n_actual),
        metadata={**dataset.metadata, "region_method": method, "n_regions": n_actual},
    )
    logger.info("Aggregated %d nodes into %d regions using %s.", dataset.data.shape[1], n_actual, method)
    return RegionAggregation(region_dataset, mapping, metadata)


def assign_latlon_bins(lat: np.ndarray, lon: np.ndarray, n_regions: int = 64) -> np.ndarray:
    """Assign nodes to approximate latitude-longitude bins."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if not np.isfinite(lat).any() or not np.isfinite(lon).any():
        logger.warning("Latitude/longitude unavailable; using equal node chunks for regions.")
        return assign_equal_chunks(lat.size, n_regions)

    n_lat = max(1, int(round(np.sqrt(n_regions / 2))))
    n_lon = max(1, int(np.ceil(n_regions / n_lat)))
    lat_bins = np.linspace(-90.0, 90.0, n_lat + 1)
    lon_bins = np.linspace(-180.0, 180.0, n_lon + 1)
    lat_clipped = np.clip(lat, -89.999, 89.999)
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    lat_idx = np.clip(np.digitize(lat_clipped, lat_bins) - 1, 0, n_lat - 1)
    lon_idx = np.clip(np.digitize(lon_wrapped, lon_bins) - 1, 0, n_lon - 1)
    return lat_idx * n_lon + lon_idx


def assign_kmeans_sphere(
    lat: np.ndarray,
    lon: np.ndarray,
    n_regions: int = 64,
    random_seed: int = 42,
) -> np.ndarray:
    """Cluster nodes using 3-D spherical coordinates."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if not np.isfinite(lat).any() or not np.isfinite(lon).any():
        logger.warning("Latitude/longitude unavailable; using equal node chunks for regions.")
        return assign_equal_chunks(lat.size, n_regions)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    features = np.column_stack(
        [
            np.sin(lat_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.cos(lat_rad) * np.cos(lon_rad),
        ]
    )
    model = KMeans(n_clusters=n_regions, random_state=random_seed, n_init=10)
    return model.fit_predict(features)


def assign_equal_chunks(n_nodes: int, n_regions: int) -> np.ndarray:
    """Fallback region assignment when coordinates are unavailable."""
    n_regions = min(max(1, n_regions), n_nodes)
    return np.floor(np.arange(n_nodes) * n_regions / n_nodes).astype(int)


def save_region_outputs(
    aggregation: RegionAggregation,
    output_root: str | Path,
) -> None:
    """Save region metadata, node mapping, and region time series."""
    root = Path(output_root)
    ensure_dir(root / "regions")
    ensure_dir(root / "processed")
    aggregation.metadata.to_csv(root / "regions" / "region_metadata.csv", index=False)
    np.savez_compressed(root / "regions" / "node_to_region.npz", node_to_region=aggregation.node_to_region)
    save_dataset_npz(aggregation.dataset, root / "processed" / "region_series.npz")


def latitude_band(lat: float) -> str:
    """Return tropical, midlatitude, polar, or unknown for a latitude."""
    if not np.isfinite(lat):
        return "unknown"
    abs_lat = abs(float(lat))
    if abs_lat < 23.5:
        return "tropical"
    if abs_lat < 60.0:
        return "midlatitude"
    return "polar"


def _remap_consecutive(mapping: np.ndarray) -> np.ndarray:
    unique = {old: new for new, old in enumerate(np.unique(mapping))}
    return np.array([unique[x] for x in mapping], dtype=int)


def _mean_longitude(lon: np.ndarray) -> float:
    lon = np.asarray(lon, dtype=float)
    if not np.isfinite(lon).any():
        return np.nan
    lon_rad = np.deg2rad(lon[np.isfinite(lon)])
    mean = np.arctan2(np.mean(np.sin(lon_rad)), np.mean(np.cos(lon_rad)))
    return float(np.rad2deg(mean))
