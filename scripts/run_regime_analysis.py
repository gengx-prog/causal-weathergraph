#!/usr/bin/env python
"""Compute reports and figures from causal discovery outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.metrics import compute_and_save_metrics
from causal_weathergraph.visualization import generate_figures
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def run_analysis(config: dict, output_root: str | Path) -> None:
    output_root = Path(output_root)
    edge_dir = output_root / "causal_edges"
    all_path = edge_dir / "all_edges.csv"
    sig_path = edge_dir / "significant_edges.csv"
    if not all_path.exists() or not sig_path.exists():
        raise FileNotFoundError("Causal edge CSVs not found. Run run_causal_discovery.py first.")
    all_edges = pd.read_csv(all_path)
    significant_edges = pd.read_csv(sig_path)
    region_path = output_root / "regions" / "region_metadata.csv"
    if region_path.exists():
        region_metadata = pd.read_csv(region_path)
    else:
        region_metadata = _region_metadata_from_edges(all_edges)

    reports = compute_and_save_metrics(all_edges, significant_edges, output_root)
    generate_figures(
        significant_edges,
        region_metadata,
        reports,
        output_root,
        top_k_edges=int(config.get("visualization", {}).get("top_k_edges", 100)),
    )
    ensure_dir(output_root / "reports")
    print(f"Saved reports to {output_root / 'reports'}")
    print(f"Saved figures to {output_root / 'figures'}")


def _region_metadata_from_edges(edges: pd.DataFrame) -> pd.DataFrame:
    records = {}
    for row in edges.itertuples(index=False):
        if hasattr(row, "source_lat"):
            records[int(row.source_region)] = {
                "region": int(row.source_region),
                "centroid_lat": float(row.source_lat),
                "centroid_lon": float(row.source_lon),
            }
            records[int(row.target_region)] = {
                "region": int(row.target_region),
                "centroid_lat": float(row.target_lat),
                "centroid_lon": float(row.target_lon),
            }
    return pd.DataFrame(records.values(), columns=["region", "centroid_lat", "centroid_lon"])


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    run_analysis(config, args.output_root)


if __name__ == "__main__":
    main()
