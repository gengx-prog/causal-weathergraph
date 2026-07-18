#!/usr/bin/env python
"""Load local data, preprocess it, and aggregate to regions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.data_io import load_atmospheric_data
from causal_weathergraph.preprocessing import preprocess_dataset
from causal_weathergraph.regions import aggregate_regions, save_region_outputs
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, save_json, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def run_preprocess(config: dict, output_root: str | Path) -> None:
    output_root = ensure_dir(output_root)
    raw = load_atmospheric_data(config.get("data", {}).get("root"), config.get("data", {}))
    processed, stats = preprocess_dataset(raw, config.get("preprocess", {}))
    aggregation = aggregate_regions(processed, config.get("regions", {}))
    save_region_outputs(aggregation, output_root)
    stats_path = Path(output_root) / "processed" / "preprocess_stats.npz"
    np.savez_compressed(stats_path, **stats)
    save_json(
        {
            "raw_shape": raw.shape,
            "processed_shape": processed.shape,
            "region_shape": aggregation.dataset.shape,
            "variables": aggregation.dataset.variable_names,
            "n_regions": int(aggregation.dataset.shape[1]),
        },
        Path(output_root) / "processed" / "preprocess_summary.json",
    )
    print(f"Saved region series to {Path(output_root) / 'processed' / 'region_series.npz'}")


def main() -> None:
    args = parse_args()
    setup_logging()
    config_path = resolve_path(args.config, REPO_ROOT)
    config = load_config(config_path)
    run_preprocess(config, args.output_root)


if __name__ == "__main__":
    main()
