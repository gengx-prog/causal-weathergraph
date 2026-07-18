"""Null edge generators for Causal WeatherGraph validation experiments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .candidate_edges import CandidateEdge, edge_type_name, haversine_km


def fully_random_edges(
    variable_names: list[str],
    region_lat: np.ndarray,
    region_lon: np.ndarray,
    n_edges: int,
    rng: np.random.Generator,
) -> list[CandidateEdge]:
    """Generate random source/target region and variable pairs."""
    n_regions = len(region_lat)
    variables = list(variable_names)
    edges: list[CandidateEdge] = []
    seen: set[tuple[int, int, str, str]] = set()
    max_unique = n_regions * n_regions * len(variables) * len(variables)
    target = min(int(n_edges), max_unique)

    while len(edges) < target:
        source_region = int(rng.integers(0, n_regions))
        target_region = int(rng.integers(0, n_regions))
        source_var = str(rng.choice(variables))
        target_var = str(rng.choice(variables))
        key = (source_region, target_region, source_var, target_var)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            _make_edge(source_region, target_region, source_var, target_var, region_lat, region_lon)
        )
    return edges


def distance_matched_random_edges(
    physical_edges: Iterable[CandidateEdge],
    region_lat: np.ndarray,
    region_lon: np.ndarray,
    rng: np.random.Generator,
    tolerance: float = 0.1,
    fallback_tolerance: float = 0.2,
) -> list[CandidateEdge]:
    """Randomize regions while approximately matching each physical edge distance."""
    physical = list(physical_edges)
    n_regions = len(region_lat)
    source_grid, target_grid = np.meshgrid(np.arange(n_regions), np.arange(n_regions), indexing="ij")
    source_flat = source_grid.reshape(-1)
    target_flat = target_grid.reshape(-1)
    distances = _distance_matrix(region_lat, region_lon).reshape(-1)

    out: list[CandidateEdge] = []
    seen: set[tuple[int, int, str, str]] = set()
    for edge in physical:
        true_distance = float(edge.distance_km)
        if true_distance <= 1e-9:
            candidates = np.flatnonzero(distances <= 1e-9)
        else:
            relative = np.abs(distances - true_distance) / true_distance
            candidates = np.flatnonzero(relative <= tolerance)
            if len(candidates) == 0:
                candidates = np.flatnonzero(relative <= fallback_tolerance)
            if len(candidates) == 0:
                candidates = np.array([int(np.nanargmin(relative))])
        rng.shuffle(candidates)
        chosen = _first_unused_pair(
            candidates,
            source_flat,
            target_flat,
            edge.source_var,
            edge.target_var,
            seen,
            avoid=(edge.source_region, edge.target_region),
        )
        if chosen is None:
            chosen = _fallback_unused_pair(
                source_flat,
                target_flat,
                edge.source_var,
                edge.target_var,
                seen,
                rng,
                avoid=(edge.source_region, edge.target_region),
            )
        if chosen is None:
            continue
        source_region, target_region = chosen
        seen.add((source_region, target_region, edge.source_var, edge.target_var))
        out.append(
            _make_edge(source_region, target_region, edge.source_var, edge.target_var, region_lat, region_lon)
        )
    return out


def variable_preserved_random_edges(
    physical_edges: Iterable[CandidateEdge],
    region_lat: np.ndarray,
    region_lon: np.ndarray,
    rng: np.random.Generator,
) -> list[CandidateEdge]:
    """Preserve source/target variable type but randomize source and target regions."""
    physical = list(physical_edges)
    n_regions = len(region_lat)
    out: list[CandidateEdge] = []
    seen: set[tuple[int, int, str, str]] = set()
    for edge in physical:
        chosen: tuple[int, int] | None = None
        for _ in range(200):
            source_region = int(rng.integers(0, n_regions))
            target_region = int(rng.integers(0, n_regions))
            key = (source_region, target_region, edge.source_var, edge.target_var)
            if key in seen:
                continue
            if source_region == edge.source_region and target_region == edge.target_region:
                continue
            chosen = (source_region, target_region)
            break
        if chosen is None:
            source_grid, target_grid = np.meshgrid(np.arange(n_regions), np.arange(n_regions), indexing="ij")
            chosen = _fallback_unused_pair(
                source_grid.reshape(-1),
                target_grid.reshape(-1),
                edge.source_var,
                edge.target_var,
                seen,
                rng,
                avoid=(edge.source_region, edge.target_region),
            )
        if chosen is None:
            continue
        source_region, target_region = chosen
        seen.add((source_region, target_region, edge.source_var, edge.target_var))
        out.append(
            _make_edge(source_region, target_region, edge.source_var, edge.target_var, region_lat, region_lon)
        )
    return out


def _make_edge(
    source_region: int,
    target_region: int,
    source_var: str,
    target_var: str,
    region_lat: np.ndarray,
    region_lon: np.ndarray,
) -> CandidateEdge:
    distance = float(
        haversine_km(
            float(region_lat[source_region]),
            float(region_lon[source_region]),
            float(region_lat[target_region]),
            float(region_lon[target_region]),
        )
    )
    return CandidateEdge(
        source_region=int(source_region),
        target_region=int(target_region),
        source_var=str(source_var),
        target_var=str(target_var),
        edge_type=edge_type_name(str(source_var), str(target_var)),
        distance_km=distance,
    )


def _distance_matrix(region_lat: np.ndarray, region_lon: np.ndarray) -> np.ndarray:
    n_regions = len(region_lat)
    distances = np.zeros((n_regions, n_regions), dtype=float)
    for target in range(n_regions):
        distances[:, target] = haversine_km(region_lat, region_lon, region_lat[target], region_lon[target])
    return distances


def _first_unused_pair(
    candidates: np.ndarray,
    source_flat: np.ndarray,
    target_flat: np.ndarray,
    source_var: str,
    target_var: str,
    seen: set[tuple[int, int, str, str]],
    avoid: tuple[int, int],
) -> tuple[int, int] | None:
    for idx in candidates:
        source_region = int(source_flat[int(idx)])
        target_region = int(target_flat[int(idx)])
        if (source_region, target_region) == avoid:
            continue
        if (source_region, target_region, source_var, target_var) in seen:
            continue
        return source_region, target_region
    return None


def _fallback_unused_pair(
    source_flat: np.ndarray,
    target_flat: np.ndarray,
    source_var: str,
    target_var: str,
    seen: set[tuple[int, int, str, str]],
    rng: np.random.Generator,
    avoid: tuple[int, int],
) -> tuple[int, int] | None:
    order = np.arange(len(source_flat))
    rng.shuffle(order)
    return _first_unused_pair(order, source_flat, target_flat, source_var, target_var, seen, avoid)
