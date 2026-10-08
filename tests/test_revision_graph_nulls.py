"""Tests for scientific invariants of the E2 randomizations."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from revision.graph_nulls import (
    CORE_PAIRS, EDGE_COLUMNS, candidate_null_draws, candidate_strata,
    degree_signature, distance_bin, distance_matrix, graph_metrics,
    key, make_edge, recalibrate_subset, rewiring_ensemble, symmetric_candidates,
)


def test_symmetric_variable_and_geographical_directions():
    lat = np.array([-50., -15., 10., 40., 65.])
    lon = np.array([-170., -80., 0., 60., 150.])
    edges = symmetric_candidates(["wind", "humidity", "cloud_cover"], lat, lon, k=1)
    keys = {key(e) for e in edges}
    assert len(keys) == len(edges)
    assert all((j, i, b, a) in keys for i, j, a, b in keys)
    spatial = [{(i, j) for i, j, x, y in keys if (x, y) == pair}
               for pair in CORE_PAIRS + tuple((b, a) for a, b in CORE_PAIRS)]
    assert all(s == spatial[0] for s in spatial)


def test_candidate_null_exact_strata_unique_and_reproducible():
    lat, lon = np.zeros(12), np.arange(12) * 30.
    distances = distance_matrix(lat, lon)
    reference = symmetric_candidates(["wind", "humidity", "cloud_cover"], lat, lon)
    draws, meta = candidate_null_draws(reference, distances, 8, seed=11)
    repeated, _ = candidate_null_draws(reference, distances, 8, seed=11)
    for draw, other in zip(draws, repeated):
        assert candidate_strata(draw) == candidate_strata(reference)
        assert len({key(e) for e in draw}) == len(reference)
        assert all(not (e.source_region == e.target_region and e.source_var == e.target_var) for e in draw)
        assert [key(e) for e in draw] == [key(e) for e in other]
    assert "source degree" in meta["not_preserved"]


def test_rewiring_preserves_degrees_distances_and_reconstructible_snapshots():
    n = 8
    distances = np.full((n, n), 1500.)
    np.fill_diagonal(distances, 0.)
    edges = {(i, (i + shift) % n, a, b) for a, b in CORE_PAIRS
             for i in range(n) for shift in (1, 3)}
    frame, diag, snapshots = rewiring_ensemble(edges, distances, n_replicates=20,
                                              seed=31, burnin_attempts_per_edge=10,
                                              gap_attempts_per_edge=3)
    bins = distance_bin(distances)
    expected_hist = Counter((a, b, int(bins[i, j])) for i, j, a, b in edges)
    source_rows = sorted(edges)
    for snapshot in snapshots:
        reconstructed = {(i, int(target), a, b) for (i, _, a, b), target in zip(source_rows, snapshot)}
        assert len(reconstructed) == len(edges)
        assert degree_signature(reconstructed) == degree_signature(edges)
        assert Counter((a, b, int(bins[i, j])) for i, j, a, b in reconstructed) == expected_hist
        assert graph_metrics(reconstructed, n)["whc_unique_edge_chains"] == graph_metrics(edges, n)["whc_unique_edge_chains"]
    assert diag["accepted_swaps"] > 0
    assert diag["distinct_graphs"] > 1
    assert frame.whc_unique_edge_chains.nunique() == 1


def test_saturated_support_reports_no_mixing_not_significance():
    n = 4
    distances = np.full((n, n), 1500.)
    np.fill_diagonal(distances, 0.)
    edges = {(i, j, "wind", "humidity") for i in range(n) for j in range(n)}
    frame, diag, _ = rewiring_ensemble(edges, distances, support=edges, reference=edges,
                                      n_replicates=5, burnin_attempts_per_edge=2,
                                      gap_attempts_per_edge=1)
    assert diag["accepted_swaps"] == 0
    assert diag["distinct_graphs"] == 1
    assert (frame.jaccard_with_reference == 1).all()


def test_bh_recomputed_on_each_candidate_subset():
    distances = np.array([[0., 1500.], [1500., 0.]])
    candidates = [make_edge((0, 1, "wind", "humidity"), distances),
                  make_edge((1, 0, "wind", "humidity"), distances)]
    rows = []
    for e, p in zip(candidates, [.03, .8]):
        rows.append({**e.to_dict(), "lag": 1, "regime": "all", "p_ols": p,
                     "p_hac64": p, "q_ols_global": .99})
    pool = pd.DataFrame(rows)
    one = recalibrate_subset(pool, candidates[:1])
    both = recalibrate_subset(pool, candidates)
    assert one.q_ols_global.iloc[0] == .03
    assert both.q_ols_global.iloc[0] == .06
    assert one.significant_ols_global.iloc[0]
    assert not both.significant_ols_global.iloc[0]
