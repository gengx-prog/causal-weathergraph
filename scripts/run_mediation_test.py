#!/usr/bin/env python
"""Test whether humidity attenuates Wind -> Cloud effects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.data_io import load_dataset_npz
from causal_weathergraph.mediation import run_humidity_mediation
from causal_weathergraph.regimes import define_regimes
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--max_lag", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    output_root = Path(args.output_root)
    dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")
    significant_path = output_root / "causal_edges" / "significant_edges.csv"
    if not significant_path.exists():
        raise FileNotFoundError(f"Missing {significant_path}. Run causal discovery first.")
    significant = pd.read_csv(significant_path)
    wind_cloud = significant[
        (significant["source_var"] == "wind") & (significant["target_var"] == "cloud_cover")
    ].copy()
    max_lag = int(args.max_lag or config.get("causal", {}).get("max_lag", 3))
    regimes = define_regimes(dataset.data, dataset.variable_names, dataset.timestamps, config.get("regimes", {}))
    details, summary = run_humidity_mediation(
        data=dataset.data,
        variable_names=dataset.variable_names,
        wind_cloud_edges=wind_cloud,
        regimes=regimes,
        max_lag=max_lag,
    )
    edge_dir = ensure_dir(output_root / "causal_edges")
    report_dir = ensure_dir(output_root / "reports")
    details.to_csv(edge_dir / "wind_cloud_humidity_mediation_edges.csv", index=False)
    summary.to_csv(report_dir / "wind_cloud_humidity_mediation_summary.csv", index=False)
    print(f"Saved mediation summary to {report_dir / 'wind_cloud_humidity_mediation_summary.csv'}")


if __name__ == "__main__":
    main()
