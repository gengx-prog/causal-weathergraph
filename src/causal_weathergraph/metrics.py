"""Summary metrics for regime-aware causal edge tables."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .causal.graph_utils import edge_set
from .utils import ensure_dir


def compute_and_save_metrics(
    all_edges: pd.DataFrame,
    significant_edges: pd.DataFrame,
    output_root: str | Path,
) -> dict[str, pd.DataFrame]:
    """Compute summary, regime-comparison, and causal-chain metrics."""
    root = Path(output_root)
    ensure_dir(root / "reports")
    summary = summary_metrics(all_edges, significant_edges)
    comparison = regime_comparison(significant_edges)
    chains = causal_chain_summary(significant_edges)
    summary.to_csv(root / "reports" / "summary_metrics.csv", index=False)
    comparison.to_csv(root / "reports" / "regime_comparison.csv", index=False)
    chains.to_csv(root / "reports" / "causal_chain_summary.csv", index=False)
    return {"summary": summary, "comparison": comparison, "chains": chains}


def summary_metrics(all_edges: pd.DataFrame, significant_edges: pd.DataFrame) -> pd.DataFrame:
    """Compute per-regime counts and fractions for key chain types."""
    regimes = sorted(set(all_edges.get("regime", pd.Series(dtype=str))).union(significant_edges.get("regime", pd.Series(dtype=str))))
    records = []
    for regime in regimes:
        sig = significant_edges[significant_edges["regime"] == regime] if not significant_edges.empty else significant_edges
        tested = all_edges[all_edges["regime"] == regime] if not all_edges.empty else all_edges
        n_sig = int(len(sig))
        wind_humidity = _is_edge(sig, "wind", "humidity")
        humidity_cloud = _is_edge(sig, "humidity", "cloud_cover")
        cross_region = sig["source_region"].to_numpy() != sig["target_region"].to_numpy() if n_sig else np.array([])
        records.append(
            {
                "regime": regime,
                "n_tested_edges": int(len(tested)),
                "n_significant_edges": n_sig,
                "fraction_wind_to_humidity": float(wind_humidity.mean()) if n_sig else 0.0,
                "fraction_humidity_to_cloud": float(humidity_cloud.mean()) if n_sig else 0.0,
                "fraction_cross_region": float(cross_region.mean()) if n_sig else 0.0,
                "mean_lag_wind_to_humidity": _mean_lag(sig[wind_humidity]) if n_sig else np.nan,
                "mean_lag_humidity_to_cloud": _mean_lag(sig[humidity_cloud]) if n_sig else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def regime_comparison(significant_edges: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Jaccard distance between regime edge sets."""
    if significant_edges.empty:
        return pd.DataFrame(columns=["regime_a", "regime_b", "jaccard_distance", "n_union", "n_intersection"])
    regimes = sorted(significant_edges["regime"].dropna().unique())
    records = []
    for a, b in combinations(regimes, 2):
        set_a = edge_set(significant_edges[significant_edges["regime"] == a])
        set_b = edge_set(significant_edges[significant_edges["regime"] == b])
        union = set_a | set_b
        intersection = set_a & set_b
        distance = 0.0 if not union else 1.0 - len(intersection) / len(union)
        records.append(
            {
                "regime_a": a,
                "regime_b": b,
                "jaccard_distance": float(distance),
                "n_union": int(len(union)),
                "n_intersection": int(len(intersection)),
            }
        )
    return pd.DataFrame.from_records(records)


def causal_chain_summary(significant_edges: pd.DataFrame) -> pd.DataFrame:
    """Count Wind -> Humidity -> Cloud paths within each regime."""
    columns = [
        "regime",
        "n_chains",
        "n_unique_humidity_regions",
        "mean_total_lag",
        "mean_chain_score",
    ]
    if significant_edges.empty:
        return pd.DataFrame(columns=columns)
    records = []
    for regime, df in significant_edges.groupby("regime"):
        wind_humidity = df[_is_edge(df, "wind", "humidity")]
        humidity_cloud = df[_is_edge(df, "humidity", "cloud_cover")]
        chain_lags = []
        chain_scores = []
        humidity_regions = set()
        for e1 in wind_humidity.itertuples(index=False):
            downstream = humidity_cloud[humidity_cloud["source_region"] == int(e1.target_region)]
            for e2 in downstream.itertuples(index=False):
                chain_lags.append(int(e1.lag) + int(e2.lag))
                score_1 = abs(float(getattr(e1, "effect_score", 0.0)))
                score_2 = abs(float(getattr(e2, "effect_score", 0.0)))
                chain_scores.append(score_1 + score_2)
                humidity_regions.add(int(e1.target_region))
        records.append(
            {
                "regime": regime,
                "n_chains": int(len(chain_lags)),
                "n_unique_humidity_regions": int(len(humidity_regions)),
                "mean_total_lag": float(np.mean(chain_lags)) if chain_lags else np.nan,
                "mean_chain_score": float(np.mean(chain_scores)) if chain_scores else np.nan,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def edge_type_counts(significant_edges: pd.DataFrame) -> pd.DataFrame:
    """Return per-regime counts for each edge type."""
    if significant_edges.empty:
        return pd.DataFrame(columns=["regime", "edge_type", "count"])
    out = significant_edges.groupby(["regime", "edge_type"]).size().reset_index(name="count")
    return out.sort_values(["regime", "edge_type"])


def _is_edge(df: pd.DataFrame, source_var: str, target_var: str) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool, index=df.index)
    return (df["source_var"] == source_var) & (df["target_var"] == target_var)


def _mean_lag(df: pd.DataFrame) -> float:
    return float(df["lag"].mean()) if len(df) else np.nan
