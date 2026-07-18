"""Matplotlib visualizations for causal WeatherGraph outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .causal.graph_utils import to_region_digraph
from .metrics import edge_type_counts
from .regions import latitude_band
from .utils import ensure_dir


def generate_figures(
    significant_edges: pd.DataFrame,
    region_metadata: pd.DataFrame,
    reports: dict[str, pd.DataFrame],
    output_root: str | Path,
    top_k_edges: int = 100,
) -> None:
    """Generate the standard figure set."""
    fig_dir = ensure_dir(Path(output_root) / "figures")
    plot_causal_edge_map(
        significant_edges,
        region_metadata,
        "wind",
        "humidity",
        fig_dir / "causal_edge_map_wind_to_humidity.png",
        top_k_edges,
    )
    plot_causal_edge_map(
        significant_edges,
        region_metadata,
        "humidity",
        "cloud_cover",
        fig_dir / "causal_edge_map_humidity_to_cloud.png",
        top_k_edges,
    )
    plot_regime_heatmap(significant_edges, fig_dir / "regime_comparison_heatmap.png")
    plot_lag_distribution(significant_edges, fig_dir / "lag_distribution.png")
    plot_network(significant_edges, region_metadata, fig_dir / "region_network.png", top_k_edges)
    plot_chain_diagram(reports.get("chains", pd.DataFrame()), fig_dir / "causal_chain_summary.png")


def plot_causal_edge_map(
    edges: pd.DataFrame,
    region_metadata: pd.DataFrame,
    source_var: str,
    target_var: str,
    path: str | Path,
    top_k: int = 100,
) -> None:
    """Plot directed edges on a simple lon-lat plane."""
    fig, ax = plt.subplots(figsize=(11, 5.8))
    _plot_region_points(ax, region_metadata)
    subset = _edge_subset(edges, source_var, target_var, top_k)
    for row in subset.itertuples(index=False):
        source = _region_lookup(region_metadata, int(row.source_region))
        target = _region_lookup(region_metadata, int(row.target_region))
        if source is None or target is None:
            continue
        width = 0.5 + 2.5 * min(1.0, abs(float(getattr(row, "effect_score", 0.0))) / 5.0)
        ax.annotate(
            "",
            xy=(target["centroid_lon"], target["centroid_lat"]),
            xytext=(source["centroid_lon"], source["centroid_lat"]),
            arrowprops={
                "arrowstyle": "->",
                "color": "#2a6f97" if source_var == "wind" else "#8f3d56",
                "alpha": 0.35,
                "lw": width,
                "shrinkA": 4,
                "shrinkB": 4,
            },
        )
    title = f"Top {source_var.replace('_', ' ')} -> {target_var.replace('_', ' ')} directed dependencies"
    ax.set_title(title)
    _format_lon_lat_axes(ax)
    _empty_note(ax, subset, "No significant edges for this edge type")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_regime_heatmap(edges: pd.DataFrame, path: str | Path) -> None:
    """Plot counts of significant edge types by regime."""
    counts = edge_type_counts(edges)
    fig, ax = plt.subplots(figsize=(10, 5))
    if counts.empty:
        _placeholder(ax, "No significant edges")
    else:
        pivot = counts.pivot(index="regime", columns="edge_type", values="count").fillna(0)
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, label="Significant edge count")
        ax.set_title("Regime comparison by edge type")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lag_distribution(edges: pd.DataFrame, path: str | Path) -> None:
    """Plot lag histograms by edge type."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if edges.empty:
        _placeholder(ax, "No significant edges")
    else:
        for edge_type, df in edges.groupby("edge_type"):
            ax.hist(df["lag"].to_numpy(dtype=float), bins=20, alpha=0.45, label=edge_type)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Significant edge count")
        ax.set_title("Lag distribution by edge type")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_network(
    edges: pd.DataFrame,
    region_metadata: pd.DataFrame,
    path: str | Path,
    top_k: int = 100,
) -> None:
    """Plot a region-level directed dependency network."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if edges.empty:
        _placeholder(ax, "No significant edges")
    else:
        graph = to_region_digraph(edges, top_k=top_k)
        pos = {
            int(row.region): (float(row.centroid_lon), float(row.centroid_lat))
            for row in region_metadata.itertuples(index=False)
        }
        colors = [_band_color(latitude_band(pos.get(node, (0, np.nan))[1])) for node in graph.nodes]
        widths = [0.5 + min(4.0, graph[u][v].get("weight", 1.0)) for u, v in graph.edges]
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.35, arrows=True, width=widths, edge_color="#555555")
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=60, node_color=colors, linewidths=0.4, edgecolors="black")
        ax.set_title("Region-level directed dependency network")
        _format_lon_lat_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_chain_diagram(chains: pd.DataFrame, path: str | Path) -> None:
    """Plot Wind -> Humidity -> Cloud chain counts by regime."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if chains.empty or "n_chains" not in chains or chains["n_chains"].sum() == 0:
        _placeholder(ax, "No Wind -> Humidity -> Cloud chains detected")
    else:
        chains = chains.sort_values("n_chains", ascending=False)
        ax.bar(chains["regime"], chains["n_chains"], color="#2a9d8f")
        ax.set_ylabel("Chain count")
        ax.set_title("Wind -> Humidity -> Cloud chain summary")
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _edge_subset(edges: pd.DataFrame, source_var: str, target_var: str, top_k: int) -> pd.DataFrame:
    if edges.empty:
        return edges
    subset = edges[(edges["source_var"] == source_var) & (edges["target_var"] == target_var)].copy()
    if subset.empty:
        return subset
    return subset.reindex(subset["effect_score"].abs().sort_values(ascending=False).index).head(top_k)


def _plot_region_points(ax: plt.Axes, region_metadata: pd.DataFrame) -> None:
    if region_metadata.empty:
        return
    ax.scatter(
        region_metadata["centroid_lon"],
        region_metadata["centroid_lat"],
        s=16,
        c="#343a40",
        alpha=0.55,
        linewidths=0,
    )


def _region_lookup(region_metadata: pd.DataFrame, region: int) -> dict[str, float] | None:
    match = region_metadata[region_metadata["region"] == region]
    if match.empty:
        return None
    row = match.iloc[0]
    return {"centroid_lat": float(row["centroid_lat"]), "centroid_lon": float(row["centroid_lon"])}


def _format_lon_lat_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-190, 190)
    ax.set_ylim(-95, 95)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)


def _placeholder(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _empty_note(ax: plt.Axes, subset: pd.DataFrame, text: str) -> None:
    if subset.empty:
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)


def _band_color(band: str) -> str:
    return {
        "tropical": "#e9c46a",
        "midlatitude": "#457b9d",
        "polar": "#b56576",
    }.get(band, "#adb5bd")
