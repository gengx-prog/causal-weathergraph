#!/usr/bin/env python
"""Run random-edge null model comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.candidate_edges import build_candidate_edges
from causal_weathergraph.causal.granger import run_granger_discovery
from causal_weathergraph.data_io import load_dataset_npz
from causal_weathergraph.null_models import (
    distance_matched_random_edges,
    fully_random_edges,
    variable_preserved_random_edges,
)
from causal_weathergraph.regimes import define_regimes
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--random_seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    output_root = Path(args.output_root)
    dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")
    causal_cfg = config.get("causal", {})
    regimes = define_regimes(dataset.data, dataset.variable_names, dataset.timestamps, config.get("regimes", {}))
    physical_candidates = build_candidate_edges(
        dataset.variable_names,
        dataset.lat,
        dataset.lon,
        causal_cfg,
        data=dataset.data,
    )

    physical_edges = load_or_run_physical(output_root, dataset, physical_candidates, regimes, causal_cfg)
    rng = np.random.default_rng(args.random_seed)
    null_sets = {
        "random_edges": fully_random_edges(
            dataset.variable_names, dataset.lat, dataset.lon, len(physical_candidates), rng
        ),
        "distance_matched_random": distance_matched_random_edges(
            physical_candidates, dataset.lat, dataset.lon, rng
        ),
        "variable_preserved_random": variable_preserved_random_edges(
            physical_candidates, dataset.lat, dataset.lon, rng
        ),
    }

    edge_dir = ensure_dir(output_root / "causal_edges" / "null_models")
    report_dir = ensure_dir(output_root / "reports")
    null_results: dict[str, pd.DataFrame] = {}
    for name, candidates in null_sets.items():
        pd.DataFrame([edge.to_dict() for edge in candidates]).to_csv(edge_dir / f"{name}_candidates.csv", index=False)
        edges = run_granger_discovery(
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
        null_results[name] = edges
        edges.to_csv(edge_dir / f"{name}_all_edges.csv", index=False)
        edges[edges["significant"].astype(bool)].to_csv(edge_dir / f"{name}_significant_edges.csv", index=False)

    summary = pd.DataFrame(
        [
            summarize("physical_candidates", physical_edges),
            summarize("random_edges", null_results["random_edges"]),
            summarize("distance_matched_random", null_results["distance_matched_random"]),
            summarize("variable_preserved_random", null_results["variable_preserved_random"]),
        ]
    )
    summary.to_csv(report_dir / "null_model_summary.csv", index=False)

    distance_summary = pd.DataFrame(
        [
            summarize_distance("physical_candidates", physical_edges),
            summarize_distance("distance_matched_random", null_results["distance_matched_random"]),
        ]
    )
    distance_summary.to_csv(report_dir / "null_model_distance_summary.csv", index=False)

    variable_summary = variable_preserved_summary(physical_edges, null_results["variable_preserved_random"])
    variable_summary.to_csv(report_dir / "null_model_variable_preserved_summary.csv", index=False)
    print(f"Saved null model reports to {report_dir}")


def load_or_run_physical(
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


def summarize(edge_set: str, edges: pd.DataFrame) -> dict[str, float | int | str]:
    sig = edges[edges["significant"].astype(bool)] if "significant" in edges else edges.iloc[0:0]
    return {
        "edge_set": edge_set,
        "tested_edges": int(len(edges)),
        "significant_edges": int(len(sig)),
        "significant_ratio": float(len(sig) / len(edges)) if len(edges) else 0.0,
        "mean_abs_coef": float(sig["effect_coefficient"].abs().mean()) if len(sig) else 0.0,
        "mean_effect_score": float(sig["effect_score"].abs().mean()) if len(sig) else 0.0,
    }


def summarize_distance(edge_set: str, edges: pd.DataFrame) -> dict[str, float | str]:
    sig = edges[edges["significant"].astype(bool)] if "significant" in edges else edges.iloc[0:0]
    return {
        "edge_set": edge_set,
        "mean_distance": float(edges["distance_km"].mean()) if len(edges) else 0.0,
        "significant_ratio": float(len(sig) / len(edges)) if len(edges) else 0.0,
        "mean_effect_score": float(sig["effect_score"].abs().mean()) if len(sig) else 0.0,
    }


def variable_preserved_summary(physical: pd.DataFrame, random_region: pd.DataFrame) -> pd.DataFrame:
    edge_types = [
        "wind_to_humidity",
        "humidity_to_cloud_cover",
        "wind_to_cloud_cover",
    ]
    records = []
    for edge_type in edge_types:
        true_ratio = significant_ratio(physical[physical["edge_type"] == edge_type])
        random_ratio = significant_ratio(random_region[random_region["edge_type"] == edge_type])
        records.append(
            {
                "edge_type": edge_type.replace("_to_", "->").replace("_cover", ""),
                "true_sig_ratio": true_ratio,
                "random_region_sig_ratio": random_ratio,
                "enrichment": float(true_ratio / random_ratio) if random_ratio > 0 else np.inf,
            }
        )
    return pd.DataFrame.from_records(records)


def significant_ratio(edges: pd.DataFrame) -> float:
    if len(edges) == 0 or "significant" not in edges:
        return 0.0
    return float(edges["significant"].astype(bool).mean())


if __name__ == "__main__":
    main()
