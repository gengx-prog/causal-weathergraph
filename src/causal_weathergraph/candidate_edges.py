"""Candidate directed edge generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import get_logger

logger = get_logger("candidate_edges")


@dataclass(frozen=True)
class CandidateEdge:
    """A lagged directed dependency to test."""

    source_region: int
    target_region: int
    source_var: str
    target_var: str
    edge_type: str
    distance_km: float

    def to_dict(self) -> dict[str, object]:
        """Return a CSV-friendly representation."""
        return {
            "source_region": self.source_region,
            "target_region": self.target_region,
            "source_var": self.source_var,
            "target_var": self.target_var,
            "edge_type": self.edge_type,
            "distance_km": self.distance_km,
        }


INTRA_REGION_PAIRS = (
    ("wind", "humidity"),
    ("humidity", "cloud_cover"),
    ("temperature", "humidity"),
    ("temperature", "cloud_cover"),
    ("wind", "cloud_cover"),
)

SPATIAL_TRANSPORT_PAIRS = (
    ("wind", "humidity"),
    ("humidity", "humidity"),
    ("humidity", "cloud_cover"),
    ("wind", "cloud_cover"),
)


def build_candidate_edges(
    variable_names: list[str],
    region_lat: np.ndarray,
    region_lon: np.ndarray,
    config: dict[str, Any] | None = None,
    data: np.ndarray | None = None,
) -> list[CandidateEdge]:
    """Generate a targeted candidate edge set to avoid all-to-all explosion."""
    config = config or {}
    n_regions = len(region_lat)
    variables = set(variable_names)
    candidates: dict[tuple[int, int, str, str], CandidateEdge] = {}

    if bool(config.get("include_intra_region_edges", True)):
        for region in range(n_regions):
            for source_var, target_var in INTRA_REGION_PAIRS:
                if source_var in variables and target_var in variables:
                    edge = CandidateEdge(
                        region,
                        region,
                        source_var,
                        target_var,
                        edge_type_name(source_var, target_var),
                        0.0,
                    )
                    candidates[(region, region, source_var, target_var)] = edge

    if bool(config.get("include_spatial_transport_edges", True)):
        k = int(config.get("candidate_k_nearest", 6))
        for target in range(n_regions):
            for source, distance in nearest_regions(
                target,
                region_lat,
                region_lon,
                k=min(k, max(0, n_regions - 1)),
            ):
                for source_var, target_var in SPATIAL_TRANSPORT_PAIRS:
                    if source_var in variables and target_var in variables:
                        edge = CandidateEdge(
                            source,
                            target,
                            source_var,
                            target_var,
                            edge_type_name(source_var, target_var),
                            float(distance),
                        )
                        candidates[(source, target, source_var, target_var)] = edge

    if bool(config.get("wind_aligned_candidates", False)):
        logger.warning(
            "Wind-aligned candidate selection was requested, but directional u/v fields are not "
            "kept in the default standardized region tensor. Skipping this optional mode."
        )

    prescreen_k = config.get("correlation_prescreen_top_k")
    if prescreen_k and data is not None:
        _add_correlation_prescreened_edges(
            candidates,
            data,
            variable_names,
            region_lat,
            region_lon,
            top_k=int(prescreen_k),
        )

    logger.info("Generated %d candidate edges.", len(candidates))
    return list(candidates.values())


def nearest_regions(
    target: int,
    lat: np.ndarray,
    lon: np.ndarray,
    k: int = 6,
) -> list[tuple[int, float]]:
    """Return the k nearest source regions to a target region."""
    if k <= 0:
        return []
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if np.isfinite(lat).all() and np.isfinite(lon).all():
        distances = haversine_km(lat, lon, lat[target], lon[target])
    else:
        distances = np.abs(np.arange(len(lat)) - target).astype(float)
    distances[target] = np.inf
    order = np.argsort(distances)[:k]
    return [(int(i), float(distances[i])) for i in order if np.isfinite(distances[i])]


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Great-circle distance in kilometers."""
    radius = 6371.0
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)
    dlat = lat1_rad - lat2_rad
    dlon = lon1_rad - lon2_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return radius * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def edge_type_name(source_var: str, target_var: str) -> str:
    """Return a readable edge type label."""
    return f"{source_var}_to_{target_var}"


def _add_correlation_prescreened_edges(
    candidates: dict[tuple[int, int, str, str], CandidateEdge],
    data: np.ndarray,
    variable_names: list[str],
    region_lat: np.ndarray,
    region_lon: np.ndarray,
    top_k: int,
) -> None:
    """Add a small number of longer-range lag-correlation candidates."""
    var_index = {v: i for i, v in enumerate(variable_names)}
    n_regions = data.shape[1]
    for source_var, target_var in SPATIAL_TRANSPORT_PAIRS:
        if source_var not in var_index or target_var not in var_index:
            continue
        source_idx = var_index[source_var]
        target_idx = var_index[target_var]
        for target in range(n_regions):
            target_series = data[1:, target, target_idx]
            scores = []
            for source in range(n_regions):
                if source == target:
                    continue
                source_series = data[:-1, source, source_idx]
                if np.std(source_series) < 1e-8 or np.std(target_series) < 1e-8:
                    continue
                corr = np.corrcoef(source_series, target_series)[0, 1]
                scores.append((abs(float(corr)), source))
            for _, source in sorted(scores, reverse=True)[:top_k]:
                key = (source, target, source_var, target_var)
                if key in candidates:
                    continue
                distance = float(haversine_km(region_lat[source], region_lon[source], region_lat[target], region_lon[target]))
                candidates[key] = CandidateEdge(
                    source,
                    target,
                    source_var,
                    target_var,
                    edge_type_name(source_var, target_var),
                    distance,
                )
