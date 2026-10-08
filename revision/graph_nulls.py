"""Frozen E2 candidate and density-conditioned topology randomizations.

Candidate randomization refits observed-data hypotheses; it is NOT a no-effect
data generator. Selected-graph rewiring conditions on discoveries and cannot
calibrate their p-values. These two reference distributions stay separate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from causal_weathergraph.candidate_edges import (
    CandidateEdge, build_candidate_edges, edge_type_name, haversine_km,
    nearest_regions,
)

EDGE_COLUMNS = ["source_region", "target_region", "source_var", "target_var"]
CORE_PAIRS = (("wind", "humidity"), ("humidity", "cloud_cover"),
              ("wind", "cloud_cover"))
# Predeclared physical distance strata; local cross-variable links are separate.
DISTANCE_CUTS_KM = (1e-8, 1000., 2000., 3000., 4000., 6000., 10000.)
EdgeKey = tuple[int, int, str, str]


def key(edge: CandidateEdge) -> EdgeKey:
    return (edge.source_region, edge.target_region, edge.source_var, edge.target_var)


def distance_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.asarray(haversine_km(np.asarray(lat)[:, None], np.asarray(lon)[:, None],
                                  np.asarray(lat)[None, :], np.asarray(lon)[None, :]))


def distance_bin(distance: float | np.ndarray) -> int | np.ndarray:
    values = np.searchsorted(np.asarray(DISTANCE_CUTS_KM), distance, side="right")
    return int(values) if np.ndim(values) == 0 else values


def make_edge(edge_key: EdgeKey, distances: np.ndarray) -> CandidateEdge:
    i, j, a, b = edge_key
    return CandidateEdge(int(i), int(j), str(a), str(b), edge_type_name(a, b),
                         float(distances[i, j]))


def symmetric_candidates(variable_names: list[str], lat: np.ndarray,
                         lon: np.ndarray, k: int = 2) -> list[CandidateEdge]:
    """The UNION of kNN undirected links, then both region and variable directions.

    Every W/H/C variable pair has the same spatial support and its exact reverse.
    Local cross-variable links are legitimate links, not graph-node self loops.
    Temperature and H->H legacy candidates are deliberately not part of this set.
    """
    required = {x for pair in CORE_PAIRS for x in pair}
    if not required.issubset(variable_names):
        raise ValueError(f"Missing core variables: {required - set(variable_names)}")
    n = len(lat)
    skeleton = {(j, j) for j in range(n)}
    for j in range(n):
        for i, _ in nearest_regions(j, lat, lon, min(k, n - 1)):
            skeleton.update(((i, j), (j, i)))
    pairs = sorted(set(CORE_PAIRS) | {(b, a) for a, b in CORE_PAIRS})
    distances = distance_matrix(lat, lon)
    return [make_edge((i, j, a, b), distances)
            for a, b in pairs for i, j in sorted(skeleton)]


def legacy_candidates(variable_names: list[str], lat: np.ndarray,
                      lon: np.ndarray) -> list[CandidateEdge]:
    return build_candidate_edges(variable_names, lat, lon,
                                 {"candidate_k_nearest": 2,
                                  "include_intra_region_edges": True,
                                  "include_spatial_transport_edges": True})


def candidate_strata(edges: Iterable[CandidateEdge]) -> Counter:
    return Counter((e.source_var, e.target_var, distance_bin(e.distance_km)) for e in edges)


def candidate_null_draws(reference: list[CandidateEdge], distances: np.ndarray,
                         n_replicates: int = 100, seed: int = 2026092902
                         ) -> tuple[list[list[CandidateEdge]], dict]:
    """Independent samples preserving exact type-by-distance-bin edge counts.

    Degrees are NOT preserved. No fallback to a different distance bin is used.
    The original candidates are allowed, as required for an exchangeable reference
    support; forbidding them can be impossible for saturated/local strata.
    """
    rng = np.random.default_rng(seed)
    bins = distance_bin(distances)
    counts = candidate_strata(reference)
    pools: dict[tuple, list[EdgeKey]] = {}
    metadata = []
    for (a, b, d), count in sorted(counts.items()):
        region_pairs = np.argwhere(bins == d)
        pool = [(int(i), int(j), a, b) for i, j in region_pairs
                if not (i == j and a == b)]
        if len(pool) < count:
            raise ValueError(f"Insufficient exact stratum support {(a,b,d)}")
        pools[a, b, d] = pool
        metadata.append({"source_var": a, "target_var": b, "distance_bin": d,
                         "sample_count": count, "available_count": len(pool),
                         "sampling_fraction": count / len(pool),
                         "fixed_stratum": count == len(pool)})
    draws = []
    for _ in range(n_replicates):
        selected = []
        for stratum, count in sorted(counts.items()):
            pool = pools[stratum]
            selected.extend(make_edge(pool[int(ix)], distances)
                            for ix in rng.choice(len(pool), count, replace=False))
        draws.append(selected)
    return draws, {"seed": seed, "replicates": n_replicates, "strata": metadata,
                   "preserved": ["edge count", "variable-pair count", "type-by-distance-bin count"],
                   "not_preserved": ["source degree", "target degree", "exact distance within bin"],
                   "null_meaning": "candidate-space comparator on observed data; not no-effect truth"}


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("All hypothesis p-values must be finite and in [0,1]")
    if not len(p):
        return p.copy()
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p), dtype=float)
    out[order] = np.minimum(adjusted, 1.)
    return out


def recalibrate_subset(pool_results: pd.DataFrame, candidates: list[CandidateEdge]) -> pd.DataFrame:
    """Recompute BH for EACH candidate family, never reuse pooled null-draw q-values."""
    selected = pd.DataFrame([e.to_dict() for e in candidates])[EDGE_COLUMNS]
    out = pool_results.merge(selected, on=EDGE_COLUMNS, how="inner", validate="many_to_one")
    if out.empty:
        raise ValueError("No pool hypotheses matched requested candidates")
    out = out.reset_index(drop=True)
    for pcol in [c for c in out if c.startswith("p_") and not c.startswith("partial")]:
        if pcol not in {"p_ols", "p_hac32", "p_hac64", "p_hac128"}:
            continue
        stem = pcol[2:]
        out[f"q_{stem}_global"] = bh(out[pcol].to_numpy())
        out[f"q_{stem}_within"] = np.nan
        for ix in out.groupby(["regime", "target_var"], sort=False).groups.values():
            out.loc[ix, f"q_{stem}_within"] = bh(out.loc[ix, pcol].to_numpy())
        for scope in ("within", "global"):
            out[f"significant_{stem}_{scope}"] = out[f"q_{stem}_{scope}"] < .05
    return out


def selected_keys(results: pd.DataFrame, q_column: str, alpha: float = .05) -> set[EdgeKey]:
    return set(results.loc[results[q_column] < alpha, EDGE_COLUMNS]
               .drop_duplicates().itertuples(index=False, name=None))


def degree_signature(edges: Iterable[EdgeKey]) -> tuple[Counter, Counter]:
    """Type-specific node degrees, stronger than untyped total node degrees."""
    edges = list(edges)
    return (Counter((a, b, i) for i, _, a, b in edges),
            Counter((a, b, j) for _, j, a, b in edges))


def graph_metrics(edges: Iterable[EdgeKey], n_regions: int,
                  reference: set[EdgeKey] | None = None) -> dict[str, float | int]:
    edges = set(edges)
    wh = np.zeros((n_regions, n_regions), dtype=np.int64)
    hc = np.zeros_like(wh)
    wc = np.zeros_like(wh)
    for i, j, a, b in edges:
        if (a, b) == ("wind", "humidity"):
            wh[i, j] = 1
        elif (a, b) == ("humidity", "cloud_cover"):
            hc[i, j] = 1
        elif (a, b) == ("wind", "cloud_cover"):
            wc[i, j] = 1
    paths = wh @ hc
    out: dict[str, float | int] = {
        "selected_unique_edges": len(edges), "whc_unique_edge_chains": int(paths.sum()),
        "whc_unique_endpoint_pairs": int((paths > 0).sum()),
        "whc_cross_region_endpoint_chains": int(paths.sum() - np.trace(paths)),
        "whc_closed_with_wc_chains": int((paths * wc).sum()),
        "whc_multiple_route_endpoint_pairs": int((paths > 1).sum()),
    }
    if reference is not None:
        union = len(edges | reference)
        out["jaccard_with_reference"] = len(edges & reference) / union if union else 1.
    return out


def lag_resolved_chain_count(results: pd.DataFrame, q_column: str) -> int:
    sig = results[results[q_column] < .05]
    left = sig[(sig.source_var == "wind") & (sig.target_var == "humidity")]
    right = sig[(sig.source_var == "humidity") & (sig.target_var == "cloud_cover")]
    incoming = left.groupby("target_region").size()
    outgoing = right.groupby("source_region").size()
    return int(incoming.mul(outgoing, fill_value=0).sum())


def discovery_summary(results: pd.DataFrame, n_regions: int, family: str) -> pd.DataFrame:
    records = []
    for qcol in [c for c in results if c.startswith("q_") and c.endswith(("within", "global"))]:
        for edge_type, group in [("ALL", results), *list(results.groupby("edge_type", sort=True))]:
            sig = group[group[qcol] < .05]
            row = {"family": family, "correction": qcol, "edge_type": edge_type,
                   "tests": len(group), "significant_tests": len(sig),
                   "significant_fraction": len(sig) / len(group),
                   "unique_candidates": len(group[EDGE_COLUMNS].drop_duplicates()),
                   "unique_significant_edges": len(sig[EDGE_COLUMNS].drop_duplicates())}
            for col in ["effect_coefficient", "beta", "partial_r2"]:
                if col in sig:
                    row["median_abs_beta" if col != "partial_r2" else "median_partial_r2"] = float(sig[col].abs().median())
            if edge_type == "ALL":
                row.update(graph_metrics(selected_keys(group, qcol), n_regions))
                row["whc_lag_resolved_chains"] = lag_resolved_chain_count(group, qcol)
            records.append(row)
    return pd.DataFrame(records)


def rewiring_ensemble(edges: set[EdgeKey], distances: np.ndarray,
                      support: set[EdgeKey] | None = None,
                      reference: set[EdgeKey] | None = None,
                      n_replicates: int = 200, seed: int = 2026092911,
                      burnin_attempts_per_edge: int = 20,
                      gap_attempts_per_edge: int = 5) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Symmetric two-edge swap MCMC with exact typed in/out degree preservation.

    Each swap additionally preserves the distance-bin MULTISET within its edge
    type. Reject duplicates, identical-node loops, unavailable support, and moves
    changing this multiset. Local W/H/C edges are not identical-node loops.
    Fixed proposal counts (including rejected moves) avoid acceptance-time bias.
    The kernel is symmetric; mixing/connectivity are not guaranteed and are
    explicitly diagnosed. Samples are Markov-chain samples, not independent.
    """
    ordered = sorted(edges)
    if not ordered:
        raise ValueError("Cannot rewire an empty selected graph")
    if support is not None and not edges.issubset(support):
        raise ValueError("Selected graph outside declared candidate support")
    rng = np.random.default_rng(seed)
    current = set(ordered)
    bins = distance_bin(distances)
    group_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for ix, (_, _, a, b) in enumerate(ordered):
        group_indices[a, b].append(ix)
    groups = [np.asarray(ix, dtype=np.int64) for ix in group_indices.values() if len(ix) >= 2]
    probabilities = np.asarray([len(g) * (len(g) - 1) for g in groups], dtype=float)
    probabilities = probabilities / probabilities.sum() if len(groups) else probabilities
    rejection = Counter()
    accepted = 0
    attempted = 0

    def advance(n_attempts: int) -> None:
        nonlocal accepted, attempted
        if not groups:
            rejection["no_two_edges_same_type"] += n_attempts
            attempted += n_attempts
            return
        # Vectorized proposal RNG is considerably faster than per-step choice.
        group_choices = rng.choice(len(groups), size=n_attempts, p=probabilities)
        random_first = rng.random(n_attempts)
        random_second = rng.random(n_attempts)
        for group_ix, uf, us in zip(group_choices, random_first, random_second):
            attempted += 1
            group = groups[int(group_ix)]
            first = int(uf * len(group))
            second = int(us * (len(group) - 1))
            second += second >= first
            ix, jx = int(group[first]), int(group[second])
            i, j, a, b = ordered[ix]
            k, l, _, _ = ordered[jx]
            if i == k or j == l:
                rejection["identity"] += 1
                continue
            e1, e2 = (i, l, a, b), (k, j, a, b)
            if (a == b) and (i == l or k == j):
                rejection["node_self_loop"] += 1
                continue
            if e1 in current or e2 in current:
                rejection["duplicate"] += 1
                continue
            if support is not None and (e1 not in support or e2 not in support):
                rejection["outside_support"] += 1
                continue
            old1, old2 = int(bins[i, j]), int(bins[k, l])
            new1, new2 = int(bins[i, l]), int(bins[k, j])
            if not ((old1 == new1 and old2 == new2) or (old1 == new2 and old2 == new1)):
                rejection["distance_bins"] += 1
                continue
            current.remove(ordered[ix]); current.remove(ordered[jx])
            ordered[ix], ordered[jx] = e1, e2
            current.add(e1); current.add(e2)
            accepted += 1

    advance(burnin_attempts_per_edge * len(edges))
    burnin_accepted = accepted
    rows = []
    snapshots = []
    observed_metrics = graph_metrics(edges, len(distances), reference)
    original_signature = degree_signature(edges)
    original_distance = Counter((a, b, int(bins[i, j])) for i, j, a, b in edges)
    for replicate in range(n_replicates):
        before = accepted
        advance(gap_attempts_per_edge * len(edges))
        assert degree_signature(current) == original_signature
        assert Counter((a, b, int(bins[i, j])) for i, j, a, b in current) == original_distance
        row = {"replicate": replicate, "accepted_since_previous": accepted - before,
               "changed_fraction_from_observed": 1 - len(current & edges) / len(edges),
               "graph_sha256": hashlib.sha256(repr(sorted(current)).encode()).hexdigest()}
        row.update(graph_metrics(current, len(distances), reference))
        rows.append(row)
        snapshots.append([j for _, j, _, _ in ordered])
    frame = pd.DataFrame(rows)
    metric_summaries = {}
    for name, observed in observed_metrics.items():
        values = frame[name].to_numpy(dtype=float)
        constant = bool(np.ptp(values) == 0)
        if len(values) > 2 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0:
            corr = float(np.corrcoef(values[:-1], values[1:])[0, 1])
        else:
            corr = None
        metric_summaries[name] = {
            "observed": observed, "null_mean": float(values.mean()),
            "null_q025": float(np.quantile(values, .025)),
            "null_q975": float(np.quantile(values, .975)), "null_sd": float(values.std(ddof=1)),
            "upper_tail_fraction": float(np.mean(values >= observed)),
            "lower_tail_fraction": float(np.mean(values <= observed)),
            "lag1_autocorrelation": corr, "constant_reference_distribution": constant,
            "invariant_by_typed_degrees": name in {"selected_unique_edges", "whc_unique_edge_chains"},
        }
    diagnostics = {
        "seed": seed, "replicates": n_replicates, "attempted_swaps": attempted,
        "accepted_swaps": accepted, "acceptance_fraction": accepted / attempted if attempted else 0.,
        "burnin_accepted_swaps": burnin_accepted,
        "burnin_attempts_per_edge": burnin_attempts_per_edge,
        "gap_attempts_per_edge": gap_attempts_per_edge,
        "rejections": dict(rejection), "distinct_graphs": int(frame.graph_sha256.nunique()),
        "mean_changed_fraction": float(frame.changed_fraction_from_observed.mean()),
        "support": "original candidate set" if support is not None else "all region pairs within preserved type-distance margins",
        "sampling": "symmetric constrained-swap Markov chain; no uniform-mixing guarantee",
        "tail_interpretation": "descriptive MCMC tail fractions, not independent-replicate exact p-values",
        "preserved": ["selected edge count", "variable-pair count", "type-specific source and target degrees", "type-specific distance-bin histogram"],
        "not_preserved": ["exact within-bin distances", "lag identity", "coefficient weights"],
        "chain_count_warning": "WHC count equals sum_h in_WH(h)*out_HC(h), hence is exactly fixed by preserved degrees; it cannot test pathway enrichment in this null.",
        "metrics": metric_summaries,
        "snapshot_edge_order": [list(e) for e in sorted(edges)],
    }
    # Each row retains its original source and variable types throughout swaps.
    return frame, diagnostics, np.asarray(snapshots, dtype=np.int16)
