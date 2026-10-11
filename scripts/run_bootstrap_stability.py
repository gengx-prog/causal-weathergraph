#!/usr/bin/env python
"""Estimate temporal block-bootstrap stability for causal edges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.candidate_edges import build_candidate_edges
from causal_weathergraph.causal.bootstrap import add_bootstrap_stability
from causal_weathergraph.causal.granger import run_granger_discovery
from causal_weathergraph.data_io import load_dataset_npz
from causal_weathergraph.regimes import define_regimes
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--n_bootstrap", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--block_length_years", type=float, default=None)
    parser.add_argument("--stability_threshold", type=float, default=0.6)
    parser.add_argument("--coef_quantile", type=float, default=0.5)
    parser.add_argument("--random_seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    output_root = Path(args.output_root)
    dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")
    causal_cfg = dict(config.get("causal", {}))
    bootstrap_cfg = {**causal_cfg, **causal_cfg.get("bootstrap", {})}
    if args.n_bootstrap is not None:
        bootstrap_cfg["n_bootstrap"] = args.n_bootstrap
    if args.block_length_years is not None:
        bootstrap_cfg["block_length"] = int(round(args.block_length_years * 365 * 4))
    if args.block_length is not None:
        bootstrap_cfg["block_length"] = args.block_length
    bootstrap_cfg["random_seed"] = args.random_seed

    regimes = define_regimes(dataset.data, dataset.variable_names, dataset.timestamps, config.get("regimes", {}))
    candidates = build_candidate_edges(dataset.variable_names, dataset.lat, dataset.lon, causal_cfg, data=dataset.data)
    baseline = load_or_run_baseline(output_root, dataset, candidates, regimes, causal_cfg)
    with_stability = add_bootstrap_stability(
        baseline,
        dataset.data,
        dataset.variable_names,
        candidates,
        regimes,
        bootstrap_cfg,
        region_lat=dataset.lat,
        region_lon=dataset.lon,
    )

    edge_dir = ensure_dir(output_root / "causal_edges")
    report_dir = ensure_dir(output_root / "reports")
    with_stability.to_csv(edge_dir / "bootstrap_edges.csv", index=False)
    significant = with_stability[with_stability["significant"].astype(bool)].copy()
    delta = float(significant["effect_abs"].quantile(args.coef_quantile)) if len(significant) else 0.0
    stable = significant[
        (significant["q_value"] < float(causal_cfg.get("alpha", 0.05)))
        & (significant["stability"] >= args.stability_threshold)
        & (significant["effect_abs"] >= delta)
    ].copy()
    stable["coef"] = stable["effect_coefficient"]
    stable.to_csv(edge_dir / "stable_edges.csv", index=False)
    summary = summarize_stability(significant, stable)
    summary.to_csv(report_dir / "bootstrap_stability_summary.csv", index=False)
    print(f"Saved {len(stable)} stable edges to {edge_dir / 'stable_edges.csv'}")
    print(f"Bootstrap coefficient threshold delta={delta:.6g}")


def load_or_run_baseline(
    output_root: Path,
    dataset,
    candidates,
    regimes,
    causal_cfg: dict,
) -> pd.DataFrame:
    path = output_root / "causal_edges" / "all_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return run_granger_discovery(
        data=dataset.data,
        variable_names=dataset.variable_names,
        candidates=candidates,
        regimes=regimes,
        max_lag=int(causal_cfg.get("max_lag", 3)),
        alpha=float(causal_cfg.get("alpha", 0.05)),
        controls=causal_cfg.get("controls", {}),
        fdr_method=str(causal_cfg.get("fdr_method", "benjamini_hochberg")),
        region_lat=dataset.lat,
        region_lon=dataset.lon,
        show_progress=False,
    )


def summarize_stability(significant: pd.DataFrame, stable: pd.DataFrame) -> pd.DataFrame:
    edge_types = [
        "wind_to_humidity",
        "humidity_to_cloud_cover",
        "wind_to_cloud_cover",
        "humidity_to_humidity",
        "temperature_to_humidity",
        "temperature_to_cloud_cover",
    ]
    records = []
    for edge_type in edge_types:
        sig_df = significant[significant["edge_type"] == edge_type]
        stable_df = stable[stable["edge_type"] == edge_type]
        records.append(
            {
                "edge_type": edge_type,
                "significant_edges": int(len(sig_df)),
                "stable_edges": int(len(stable_df)),
                "stable_ratio": float(len(stable_df) / len(sig_df)) if len(sig_df) else 0.0,
                "mean_stability": float(sig_df["stability"].mean()) if len(sig_df) else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    main()
