"""Helpers for turning edge tables into graph objects and edge sets."""

from __future__ import annotations

import pandas as pd
import networkx as nx


def edge_identity(row: object, include_lag: bool = False) -> tuple[object, ...]:
    """Return a hashable identity for a directed variable-region edge."""
    base = (
        int(getattr(row, "source_region")),
        int(getattr(row, "target_region")),
        str(getattr(row, "source_var")),
        str(getattr(row, "target_var")),
    )
    if include_lag:
        return (*base, int(getattr(row, "lag")))
    return base


def edge_set(df: pd.DataFrame, include_lag: bool = False) -> set[tuple[object, ...]]:
    """Return a set of edge identities from a DataFrame."""
    if df.empty:
        return set()
    return {edge_identity(row, include_lag=include_lag) for row in df.itertuples(index=False)}


def to_region_digraph(edges: pd.DataFrame, top_k: int | None = None) -> nx.DiGraph:
    """Build a region-level directed graph aggregated over variable edge types."""
    graph = nx.DiGraph()
    if edges.empty:
        return graph
    df = edges.copy()
    if top_k is not None and "effect_score" in df:
        df = df.reindex(df["effect_score"].abs().sort_values(ascending=False).index).head(top_k)
    for row in df.itertuples(index=False):
        source = int(row.source_region)
        target = int(row.target_region)
        weight = abs(float(getattr(row, "effect_score", 1.0)))
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
            graph[source][target]["count"] += 1
        else:
            graph.add_edge(source, target, weight=weight, count=1)
    return graph
