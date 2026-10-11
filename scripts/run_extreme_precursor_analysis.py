#!/usr/bin/env python
"""Summarize precursor signals before regional high-humidity and high-cloud events."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.data_io import load_dataset_npz
from causal_weathergraph.utils import ensure_dir


WINDOWS = ((6, 1), (12, 2), (24, 4), (48, 8))
EDGE_MAP = {
    "WH": ("wind", "humidity", "wind_to_humidity"),
    "HC": ("humidity", "cloud_cover", "humidity_to_cloud_cover"),
    "WC": ("wind", "cloud_cover", "wind_to_cloud_cover"),
    "HH": ("humidity", "humidity", "humidity_to_humidity"),
}
EVENTS = {
    "high_cloud": "cloud_cover",
    "high_humidity": "humidity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--edge_table", default=None)
    parser.add_argument("--event_quantile", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")
    edge_path = Path(args.edge_table) if args.edge_table else output_root / "causal_edges" / "targeted_maxlag12_significant_edges.csv"
    if not edge_path.exists():
        edge_path = output_root / "causal_edges" / "significant_edges.csv"
    edges = pd.read_csv(edge_path)
    if "regime" in edges.columns and "all" in set(edges["regime"].astype(str)):
        edges = edges[edges["regime"] == "all"].copy()

    event_masks = build_event_masks(dataset.data, dataset.variable_names, args.event_quantile)
    summary = summarize_precursors(dataset.data, dataset.variable_names, edges, event_masks)

    report_dir = ensure_dir(output_root / "reports")
    figure_dir = ensure_dir(output_root / "figures")
    summary.to_csv(report_dir / "extreme_precursor_summary.csv", index=False)
    plot_heatmap(summary, figure_dir / "extreme_precursor_heatmap.png")
    print(f"Saved extreme precursor reports to {report_dir}", flush=True)


def build_event_masks(
    data: np.ndarray,
    variable_names: list[str],
    event_quantile: float,
) -> dict[str, np.ndarray]:
    masks = {}
    for event_type, variable in EVENTS.items():
        var_idx = variable_names.index(variable)
        values = data[:, :, var_idx]
        threshold = np.nanquantile(values, event_quantile, axis=0)
        masks[event_type] = values >= threshold.reshape(1, -1)
    return masks


def summarize_precursors(
    data: np.ndarray,
    variable_names: list[str],
    edges: pd.DataFrame,
    event_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    records = []
    for event_type, event_var in EVENTS.items():
        for hours, lag_steps in WINDOWS:
            record = {
                "event_type": event_type,
                "precursor_window_hours": int(hours),
                "lag_steps": int(lag_steps),
                "n_regional_events": int(event_masks[event_type][lag_steps:].sum()),
            }
            for label, (source_var, target_var, edge_type) in EDGE_MAP.items():
                if target_var != event_var:
                    record[f"{label}_strength"] = np.nan
                    record[f"{label}_edge_count"] = 0
                    continue
                strength, count = edge_triggered_strength(
                    data,
                    variable_names,
                    edges,
                    event_masks[event_type],
                    source_var,
                    target_var,
                    edge_type,
                    lag_steps,
                )
                record[f"{label}_strength"] = strength
                record[f"{label}_edge_count"] = count
            if event_type == "high_cloud":
                chain_count, chain_strength = cloud_chain_precursor_strength(
                    data,
                    variable_names,
                    edges,
                    event_masks[event_type],
                    lag_steps,
                )
            else:
                chain_count, chain_strength = humidity_mediator_chain_count(edges, lag_steps), np.nan
            record["WHC_chain_count"] = int(chain_count)
            record["WHC_chain_strength"] = chain_strength
            records.append(record)
    return pd.DataFrame.from_records(records)


def edge_triggered_strength(
    data: np.ndarray,
    variable_names: list[str],
    edges: pd.DataFrame,
    event_mask: np.ndarray,
    source_var: str,
    target_var: str,
    edge_type: str,
    lag_steps: int,
) -> tuple[float, int]:
    source_idx = variable_names.index(source_var)
    target_idx = variable_names.index(target_var)
    subset = edges[(edges["edge_type"] == edge_type) & (edges["lag"].astype(int) == int(lag_steps))]
    scores = []
    for row in subset.itertuples(index=False):
        target_region = int(row.target_region)
        source_region = int(row.source_region)
        events = np.flatnonzero(event_mask[:, target_region])
        events = events[events >= lag_steps]
        if len(events) == 0:
            continue
        source_values = data[events - lag_steps, source_region, source_idx]
        target_values = data[events, target_region, target_idx]
        finite = np.isfinite(source_values) & np.isfinite(target_values)
        if not finite.any():
            continue
        source_anomaly = float(np.nanmean(source_values[finite]))
        weight = abs(float(getattr(row, "effect_score", 1.0)))
        scores.append(abs(source_anomaly) * weight)
    return (float(np.mean(scores)) if scores else np.nan, int(len(scores)))


def cloud_chain_precursor_strength(
    data: np.ndarray,
    variable_names: list[str],
    edges: pd.DataFrame,
    event_mask: np.ndarray,
    total_lag: int,
) -> tuple[int, float]:
    wind_idx = variable_names.index("wind")
    wh = edges[edges["edge_type"] == "wind_to_humidity"]
    hc = edges[edges["edge_type"] == "humidity_to_cloud_cover"]
    hc_by_source = {int(region): group for region, group in hc.groupby("source_region")}
    strengths = []
    chain_count = 0
    for left in wh.itertuples(index=False):
        humidity_region = int(left.target_region)
        downstream = hc_by_source.get(humidity_region)
        if downstream is None:
            continue
        for right in downstream.itertuples(index=False):
            if int(left.lag) + int(right.lag) != int(total_lag):
                continue
            chain_count += 1
            target_region = int(right.target_region)
            events = np.flatnonzero(event_mask[:, target_region])
            events = events[events >= total_lag]
            if len(events) == 0:
                continue
            wind_values = data[events - total_lag, int(left.source_region), wind_idx]
            finite = np.isfinite(wind_values)
            if not finite.any():
                continue
            chain_score = abs(float(left.effect_score)) + abs(float(right.effect_score))
            strengths.append(abs(float(np.nanmean(wind_values[finite]))) * chain_score)
    return int(chain_count), float(np.mean(strengths)) if strengths else np.nan


def humidity_mediator_chain_count(edges: pd.DataFrame, wh_lag: int) -> int:
    wh = edges[(edges["edge_type"] == "wind_to_humidity") & (edges["lag"].astype(int) == int(wh_lag))]
    hc = edges[edges["edge_type"] == "humidity_to_cloud_cover"]
    hc_sources = set(hc["source_region"].astype(int))
    return int(sum(1 for row in wh.itertuples(index=False) if int(row.target_region) in hc_sources))


def plot_heatmap(summary: pd.DataFrame, path: Path) -> None:
    plot_df = summary[summary["event_type"] == "high_cloud"].copy()
    metrics = ["HC_strength", "WC_strength", "WHC_chain_strength"]
    values = plot_df[metrics].to_numpy(dtype=float).T
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels([f"{int(v)}h" for v in plot_df["precursor_window_hours"]])
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(["HC", "WC", "WHC chain"])
    ax.set_xlabel("Precursor window")
    ax.set_title("High-cloud event precursor strength")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
