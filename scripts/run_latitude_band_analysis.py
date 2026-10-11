#!/usr/bin/env python
"""Summarize core edges and chains by latitude band."""

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

from causal_weathergraph.regimes import latitude_band_masks
from causal_weathergraph.utils import ensure_dir


EDGE_TYPES = {
    "WH": "wind_to_humidity",
    "HC": "humidity_to_cloud_cover",
    "WC": "wind_to_cloud_cover",
    "HH": "humidity_to_humidity",
}
BAND_ORDER = ["tropical", "midlatitude", "polar"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    sig_path = output_root / "causal_edges" / "significant_edges.csv"
    stable_path = output_root / "causal_edges" / "stable_edges.csv"
    region_path = output_root / "regions" / "region_metadata.csv"
    significant = pd.read_csv(sig_path)
    stable = pd.read_csv(stable_path) if stable_path.exists() else pd.DataFrame(columns=significant.columns)
    regions = pd.read_csv(region_path)
    band_lookup = make_band_lookup(regions)

    significant = add_bands(significant, band_lookup)
    stable = add_bands(stable, band_lookup) if not stable.empty else stable

    report_dir = ensure_dir(output_root / "reports")
    figure_dir = ensure_dir(output_root / "figures")

    edge_summary = summarize_edges(significant)
    edge_summary.to_csv(report_dir / "latitude_band_edge_summary.csv", index=False)

    chain_summary = summarize_chains(significant, stable, band_lookup)
    chain_summary.to_csv(report_dir / "latitude_band_chain_summary.csv", index=False)

    transfer_summary = summarize_transfers(significant)
    transfer_summary.to_csv(report_dir / "latitude_band_transfer_summary.csv", index=False)

    plot_edge_composition(edge_summary, figure_dir / "latitude_band_edge_composition.png")
    plot_chain_summary(chain_summary, figure_dir / "latitude_band_chain_summary.png")
    print(f"Saved latitude-band reports to {report_dir}")


def make_band_lookup(regions: pd.DataFrame) -> dict[int, str]:
    if "latitude_band" in regions.columns:
        return {int(row.region): str(row.latitude_band).lower() for row in regions.itertuples(index=False)}
    masks = latitude_band_masks(regions["centroid_lat"].to_numpy(dtype=float))
    lookup: dict[int, str] = {}
    region_ids = regions["region"].to_numpy(dtype=int)
    for band, mask in masks.items():
        for region in region_ids[mask]:
            lookup[int(region)] = band
    return lookup


def add_bands(edges: pd.DataFrame, band_lookup: dict[int, str]) -> pd.DataFrame:
    out = edges.copy()
    out["source_band"] = out["source_region"].map(lambda value: band_lookup.get(int(value), "unknown"))
    out["target_band"] = out["target_region"].map(lambda value: band_lookup.get(int(value), "unknown"))
    return out


def summarize_edges(edges: pd.DataFrame) -> pd.DataFrame:
    records = []
    for band in BAND_ORDER:
        group = edges[edges["target_band"] == band]
        record = {"latitude_band": band}
        for label, edge_type in EDGE_TYPES.items():
            sub = group[group["edge_type"] == edge_type]
            record[f"{label}_edges"] = int(len(sub))
        wh = group[group["edge_type"] == EDGE_TYPES["WH"]]
        hc = group[group["edge_type"] == EDGE_TYPES["HC"]]
        record["mean_WH_lag"] = float(wh["lag"].mean()) if len(wh) else np.nan
        record["mean_HC_lag"] = float(hc["lag"].mean()) if len(hc) else np.nan
        record["mean_effect_score"] = float(group["effect_score"].abs().mean()) if len(group) else np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_chains(
    significant: pd.DataFrame,
    stable: pd.DataFrame,
    band_lookup: dict[int, str],
) -> pd.DataFrame:
    sig_records = chain_records(significant, band_lookup)
    stable_records = chain_records(stable, band_lookup) if not stable.empty else []
    sig_df = pd.DataFrame(sig_records)
    stable_df = pd.DataFrame(stable_records)
    records = []
    for band in BAND_ORDER:
        sig_band = sig_df[sig_df["latitude_band"] == band] if not sig_df.empty else sig_df
        stable_band = stable_df[stable_df["latitude_band"] == band] if not stable_df.empty else stable_df
        records.append(
            {
                "latitude_band": band,
                "WHC_chains": int(len(sig_band)),
                "stable_WHC_chains": int(len(stable_band)),
                "mean_total_lag": float(sig_band["total_lag"].mean()) if len(sig_band) else np.nan,
                "mean_chain_score": float(sig_band["chain_score"].mean()) if len(sig_band) else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def chain_records(edges: pd.DataFrame, band_lookup: dict[int, str]) -> list[dict[str, float | int | str]]:
    if edges.empty:
        return []
    records = []
    for regime, regime_edges in edges.groupby("regime"):
        wh = regime_edges[regime_edges["edge_type"] == EDGE_TYPES["WH"]]
        hc = regime_edges[regime_edges["edge_type"] == EDGE_TYPES["HC"]]
        hc_by_source = {int(region): group for region, group in hc.groupby("source_region")}
        for left in wh.itertuples(index=False):
            humidity_region = int(left.target_region)
            downstream = hc_by_source.get(humidity_region)
            if downstream is None:
                continue
            band = band_lookup.get(humidity_region, "unknown")
            for right in downstream.itertuples(index=False):
                records.append(
                    {
                        "regime": regime,
                        "latitude_band": band,
                        "humidity_region": humidity_region,
                        "cloud_region": int(right.target_region),
                        "total_lag": int(left.lag) + int(right.lag),
                        "chain_score": abs(float(left.effect_score)) + abs(float(right.effect_score)),
                    }
                )
    return records


def summarize_transfers(edges: pd.DataFrame) -> pd.DataFrame:
    core = edges[edges["edge_type"].isin(EDGE_TYPES.values())].copy()
    if core.empty:
        return pd.DataFrame(columns=["source_band", "target_band", "edge_type", "edge_count", "mean_lag", "mean_effect_score"])
    grouped = core.groupby(["source_band", "target_band", "edge_type"], sort=True)
    out = grouped.agg(
        edge_count=("edge_type", "size"),
        mean_lag=("lag", "mean"),
        mean_effect_score=("effect_score", lambda values: float(np.abs(values).mean())),
    ).reset_index()
    return out.sort_values(["edge_type", "source_band", "target_band"])


def plot_edge_composition(summary: pd.DataFrame, path: Path) -> None:
    labels = [band.title() for band in summary["latitude_band"]]
    x = np.arange(len(labels))
    width = 0.18
    colors = ["#2A6FBB", "#3B8C6E", "#C56A43", "#6D5DA8"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, label in enumerate(["WH", "HC", "WC", "HH"]):
        ax.bar(x + (i - 1.5) * width, summary[f"{label}_edges"], width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Significant edge-lag-regime rows")
    ax.legend(frameon=False, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_chain_summary(summary: pd.DataFrame, path: Path) -> None:
    labels = [band.title() for band in summary["latitude_band"]]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, summary["WHC_chains"], width, label="All significant", color="#2A6FBB")
    ax.bar(x + width / 2, summary["stable_WHC_chains"], width, label="Stable", color="#3B8C6E")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WHC chains")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
