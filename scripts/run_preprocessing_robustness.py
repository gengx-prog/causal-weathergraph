#!/usr/bin/env python
"""Run preprocessing robustness experiments."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.data_io import load_atmospheric_data
from causal_weathergraph.preprocessing_robustness import (
    SETTINGS,
    plot_preprocessing_heatmap,
    preprocessing_overlap,
    run_granger_for_region_dataset,
    run_preprocessing_setting,
    summarize_preprocessing,
)
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "local_full_period_fast.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--settings", nargs="*", default=["P1_monthly", "P2_monthly_diff", "P3_diff_only", "P4_rolling"])
    parser.add_argument("--include_p5", action="store_true", help="Also run P5 monthly anomaly + region-level standardization.")
    parser.add_argument("--recompute_baseline", action="store_true", help="Recompute P1 instead of reusing main outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    output_root = Path(args.output_root)
    setting_order = list(dict.fromkeys(args.settings + (["P5_monthly_regionstd"] if args.include_p5 else [])))
    unknown = [name for name in setting_order if name not in SETTINGS]
    if unknown:
        raise ValueError(f"Unknown preprocessing settings: {unknown}")

    raw = None
    needs_raw = args.recompute_baseline or any(name != "P1_monthly" for name in setting_order)
    if needs_raw:
        raw = load_atmospheric_data(config.get("data", {}).get("root"), config.get("data", {}))

    all_edges_by_setting: dict[str, pd.DataFrame] = {}
    significant_by_setting: dict[str, pd.DataFrame] = {}

    for name in setting_order:
        print(f"\n=== Preprocessing robustness: {name} ===", flush=True)
        dataset, _ = run_preprocessing_setting(
            SETTINGS[name],
            config,
            output_root,
            raw=raw,
            reuse_p1=not args.recompute_baseline,
        )
        if name == "P1_monthly" and not args.recompute_baseline:
            edges = pd.read_csv(output_root / "causal_edges" / "all_edges.csv")
            significant = pd.read_csv(output_root / "causal_edges" / "significant_edges.csv")
            edges["preprocess_setting"] = name
            significant["preprocess_setting"] = name
        else:
            edges, significant = run_granger_for_region_dataset(dataset, config, name)
        all_edges_by_setting[name] = edges
        significant_by_setting[name] = significant
        print(f"{name}: tested={len(edges)} significant={len(significant)}", flush=True)
        del dataset
        gc.collect()

    report_dir = ensure_dir(output_root / "reports")
    edge_dir = ensure_dir(output_root / "causal_edges")
    figure_dir = ensure_dir(output_root / "figures")

    all_edges = pd.concat(all_edges_by_setting.values(), ignore_index=True)
    significant_edges = pd.concat(significant_by_setting.values(), ignore_index=True)
    all_edges.to_csv(edge_dir / "preprocessing_robustness_edges.csv", index=False)
    significant_edges.to_csv(edge_dir / "preprocessing_robustness_significant_edges.csv", index=False)

    summary = summarize_preprocessing(all_edges_by_setting, significant_by_setting, setting_order)
    overlap = preprocessing_overlap(significant_by_setting, setting_order)
    summary.to_csv(report_dir / "preprocessing_robustness_summary.csv", index=False)
    overlap.to_csv(report_dir / "preprocessing_robustness_overlap.csv", index=False)
    plot_preprocessing_heatmap(overlap, figure_dir / "preprocessing_robustness_heatmap.png")
    print(f"Saved preprocessing robustness outputs to {report_dir}", flush=True)


if __name__ == "__main__":
    main()
