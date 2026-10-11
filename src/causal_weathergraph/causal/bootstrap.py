"""Temporal block-bootstrap stability for discovered edges."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from causal_weathergraph.candidate_edges import CandidateEdge
from causal_weathergraph.utils import get_logger

from .fast_granger import run_fast_granger_discovery

logger = get_logger("causal.bootstrap")


def add_bootstrap_stability(
    edges: pd.DataFrame,
    data: np.ndarray,
    variable_names: list[str],
    candidates: list[CandidateEdge],
    regimes: dict[str, np.ndarray],
    config: dict[str, Any],
    region_lat: np.ndarray | None = None,
    region_lon: np.ndarray | None = None,
) -> pd.DataFrame:
    """Estimate edge stability from temporal block bootstrap runs."""
    if edges.empty:
        edges = edges.copy()
        edges["stability"] = []
        return edges

    n_bootstrap = int(config.get("n_bootstrap", 20))
    block_length = int(config.get("block_length", 128))
    max_lag = int(config.get("max_lag", 12))
    alpha = float(config.get("alpha", 0.05))
    controls = config.get("controls", {})
    rng = np.random.default_rng(int(config.get("random_seed", 42)))
    counts: Counter[tuple[object, ...]] = Counter()
    possible: Counter[str] = Counter()

    for regime_name, mask in regimes.items():
        indices = np.flatnonzero(mask)
        if len(indices) <= max_lag + 8:
            continue
        for _ in tqdm(range(n_bootstrap), desc=f"Bootstrap [{regime_name}]", leave=False):
            boot_idx = sample_block_indices(indices, block_length, rng)
            if len(boot_idx) <= max_lag + 8:
                continue
            possible[regime_name] += 1
            boot_data = data[boot_idx]
            boot_regimes = {regime_name: np.ones(len(boot_idx), dtype=bool)}
            boot_edges = run_fast_granger_discovery(
                boot_data,
                variable_names,
                candidates,
                boot_regimes,
                max_lag=max_lag,
                alpha=alpha,
                controls=controls,
                fdr_method=str(config.get("fdr_method", "benjamini_hochberg")),
                region_lat=region_lat,
                region_lon=region_lon,
            )
            significant = boot_edges[boot_edges.get("significant", False).astype(bool)]
            for row in significant.itertuples(index=False):
                counts[_edge_key(row)] += 1

    stable = edges.copy()
    stable["stability"] = [
        counts[_edge_key(row)] / max(1, possible[str(row.regime)])
        for row in stable.itertuples(index=False)
    ]
    return stable


def sample_block_indices(
    valid_indices: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a bootstrap sequence from ordered valid time indices."""
    valid_indices = np.asarray(valid_indices, dtype=int)
    n = len(valid_indices)
    if n == 0:
        return valid_indices
    block_length = max(1, min(int(block_length), n))
    sampled: list[np.ndarray] = []
    while sum(len(x) for x in sampled) < n:
        start = int(rng.integers(0, n - block_length + 1))
        sampled.append(valid_indices[start : start + block_length])
    return np.concatenate(sampled)[:n]


def _edge_key(row: object) -> tuple[object, ...]:
    return (
        str(getattr(row, "regime")),
        int(getattr(row, "source_region")),
        int(getattr(row, "target_region")),
        str(getattr(row, "source_var")),
        str(getattr(row, "target_var")),
        int(getattr(row, "lag")),
    )
