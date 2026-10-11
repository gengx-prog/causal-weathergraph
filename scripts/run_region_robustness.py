#!/usr/bin/env python
"""Run region aggregation robustness experiments."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.data_io import load_atmospheric_data, load_dataset_npz
from causal_weathergraph.preprocessing import preprocess_dataset
from causal_weathergraph.region_robustness import (
    SETTINGS,
    plot_region_summary,
    region_dataset_for_setting,
    region_macro_overlap,
    run_region_granger,
    summarize_region_robustness,
)
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "local_full_period_fast.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--settings", nargs="*", default=["R32", "R64", "R128", "K64", "K128"])
    parser.add_argument("--recompute_baseline", action="store_true", help="Recompute R64 instead of reusing main outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    output_root = Path(args.output_root)
    setting_order = list(dict.fromkeys(args.settings))
    unknown = [name for name in setting_order if name not in SETTINGS]
    if unknown:
        raise ValueError(f"Unknown region settings: {unknown}")

    if all(name == "R64" for name in setting_order) and not args.recompute_baseline:
        processed = load_dataset_npz(output_root / "processed" / "region_series.npz")
    else:
        raw = load_atmospheric_data(config.get("data", {}).get("root"), config.get("data", {}))
        processed, _ = preprocess_dataset(raw, config.get("preprocess", {}))
        del raw
        gc.collect()

    all_edges_by_setting: dict[str, pd.DataFrame] = {}
    significant_by_setting: dict[str, pd.DataFrame] = {}
    metadata_by_setting: dict[str, pd.DataFrame] = {}

    for name in setting_order:
        print(f"\n=== Region robustness: {name} ===", flush=True)
        dataset, metadata = region_dataset_for_setting(
            processed,
            SETTINGS[name],
            output_root,
            random_seed=int(config.get("regions", {}).get("random_seed", 42)),
            reuse_r64=not args.recompute_baseline,
        )
        if name == "R64" and not args.recompute_baseline:
            edges = pd.read_csv(output_root / "causal_edges" / "all_edges.csv")
            significant = pd.read_csv(output_root / "causal_edges" / "significant_edges.csv")
            edges["region_setting"] = name
            significant["region_setting"] = name
        else:
            edges, significant = run_region_granger(dataset, config, name)
        all_edges_by_setting[name] = edges
        significant_by_setting[name] = significant
        metadata_by_setting[name] = metadata
        print(f"{name}: regions={len(metadata)} tested={len(edges)} significant={len(significant)}", flush=True)
        del dataset
        gc.collect()

    report_dir = ensure_dir(output_root / "reports")
    edge_dir = ensure_dir(output_root / "causal_edges")
    figure_dir = ensure_dir(output_root / "figures")

    all_edges = pd.concat(all_edges_by_setting.values(), ignore_index=True)
    significant_edges = pd.concat(significant_by_setting.values(), ignore_index=True)
    all_edges.to_csv(edge_dir / "region_robustness_edges.csv", index=False)
    significant_edges.to_csv(edge_dir / "region_robustness_significant_edges.csv", index=False)

    summary = summarize_region_robustness(all_edges_by_setting, significant_by_setting, metadata_by_setting, setting_order)
    overlap = region_macro_overlap(summary)
    summary.to_csv(report_dir / "region_robustness_summary.csv", index=False)
    overlap.to_csv(report_dir / "region_robustness_overlap.csv", index=False)
    plot_region_summary(summary, figure_dir / "region_robustness_summary.png")
    print(f"Saved region robustness outputs to {report_dir}", flush=True)


if __name__ == "__main__":
    main()
