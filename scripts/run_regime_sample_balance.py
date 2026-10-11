#!/usr/bin/env python
"""Matched-sample controls for high-versus-normal regime graph differences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.candidate_edges import build_candidate_edges
from causal_weathergraph.causal.fast_granger import run_fast_granger_discovery
from causal_weathergraph.causal.graph_utils import edge_set
from causal_weathergraph.data_io import load_dataset_npz
from causal_weathergraph.regimes import define_regimes
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


COMPARISONS = [
    ("high_humidity", "normal_humidity"),
    ("high_cloud", "normal_cloud"),
]
EDGE_TYPES = [
    "wind_to_humidity",
    "humidity_to_cloud_cover",
    "wind_to_cloud_cover",
    "humidity_to_humidity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--n_repeats", type=int, default=20)
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
    candidates = build_candidate_edges(
        dataset.variable_names,
        dataset.lat,
        dataset.lon,
        causal_cfg,
        data=dataset.data,
    )
    significant = pd.read_csv(output_root / "causal_edges" / "significant_edges.csv")

    rng = np.random.default_rng(args.random_seed)
    summary_records = []
    jaccard_records = []
    edge_count_records = []
    report_dir = ensure_dir(output_root / "reports")

    for high_regime, normal_regime in COMPARISONS:
        high_sig = significant[significant["regime"] == high_regime]
        normal_sig = significant[significant["regime"] == normal_regime]
        high_set = edge_set(high_sig, include_lag=False)
        normal_set = edge_set(normal_sig, include_lag=False)
        original_jaccard = jaccard_distance(high_set, normal_set)
        original_edge_diff = len(high_set) - len(normal_set)
        high_indices = np.flatnonzero(regimes[high_regime])
        normal_indices = np.flatnonzero(regimes[normal_regime])

        matched_jaccards = []
        matched_diffs = []
        matched_edge_type_counts: dict[str, list[int]] = {edge_type: [] for edge_type in EDGE_TYPES}
        for repeat in range(args.n_repeats):
            replace = len(normal_indices) < len(high_indices)
            sampled = rng.choice(normal_indices, size=len(high_indices), replace=replace)
            mask = np.zeros(dataset.data.shape[0], dtype=bool)
            mask[sampled] = True
            matched_name = f"{normal_regime}_matched"
            matched_edges = run_fast_granger_discovery(
                data=dataset.data,
                variable_names=dataset.variable_names,
                candidates=candidates,
                regimes={matched_name: mask},
                max_lag=int(causal_cfg.get("max_lag", 3)),
                alpha=float(causal_cfg.get("alpha", 0.05)),
                controls=causal_cfg.get("controls", {}),
                fdr_method=str(causal_cfg.get("fdr_method", "benjamini_hochberg")),
                region_lat=dataset.lat,
                region_lon=dataset.lon,
            )
            matched_sig = matched_edges[matched_edges["significant"].astype(bool)].copy()
            matched_set = edge_set(matched_sig, include_lag=False)
            matched_jaccard = jaccard_distance(high_set, matched_set)
            matched_diff = len(high_set) - len(matched_set)
            matched_jaccards.append(matched_jaccard)
            matched_diffs.append(matched_diff)
            for edge_type in EDGE_TYPES:
                matched_edge_type_counts[edge_type].append(int((matched_sig["edge_type"] == edge_type).sum()))
            jaccard_records.append(
                {
                    "comparison": f"{high_regime} vs {normal_regime}",
                    "repeat": repeat + 1,
                    "matched_jaccard": matched_jaccard,
                    "high_edges": int(len(high_set)),
                    "matched_normal_edges": int(len(matched_set)),
                    "edge_diff": int(matched_diff),
                    "n_high_samples": int(len(high_indices)),
                    "n_normal_samples": int(len(normal_indices)),
                }
            )
            print(f"{high_regime} vs {normal_regime}: finished repeat {repeat + 1}/{args.n_repeats}")

        summary_records.append(
            {
                "comparison": f"{high_regime} vs {normal_regime}",
                "original_jaccard": original_jaccard,
                "matched_jaccard_mean": float(np.mean(matched_jaccards)),
                "matched_jaccard_std": float(np.std(matched_jaccards, ddof=1)) if len(matched_jaccards) > 1 else 0.0,
                "original_edge_diff": int(original_edge_diff),
                "matched_edge_diff_mean": float(np.mean(matched_diffs)),
                "matched_edge_diff_std": float(np.std(matched_diffs, ddof=1)) if len(matched_diffs) > 1 else 0.0,
                "n_high_samples": int(len(high_indices)),
                "n_normal_samples": int(len(normal_indices)),
                "n_repeats": int(args.n_repeats),
            }
        )

        for edge_type in EDGE_TYPES:
            high_count = int((high_sig["edge_type"] == edge_type).sum())
            matched_counts = np.asarray(matched_edge_type_counts[edge_type], dtype=float)
            mean_matched = float(matched_counts.mean()) if len(matched_counts) else 0.0
            edge_count_records.append(
                {
                    "regime_pair": f"{high_regime} vs {normal_regime}",
                    "edge_type": edge_type,
                    "high_edges": high_count,
                    "matched_normal_edges_mean": mean_matched,
                    "diff": float(high_count - mean_matched),
                    "p_value": empirical_two_sided_p_value(high_count, matched_counts),
                }
            )

    pd.DataFrame.from_records(summary_records).to_csv(report_dir / "regime_sample_balance_summary.csv", index=False)
    pd.DataFrame.from_records(jaccard_records).to_csv(report_dir / "regime_sample_balance_jaccard.csv", index=False)
    pd.DataFrame.from_records(edge_count_records).to_csv(
        report_dir / "regime_sample_balance_edge_type_summary.csv",
        index=False,
    )
    print(f"Saved regime sample-balance reports to {report_dir}")


def jaccard_distance(a: set[tuple[object, ...]], b: set[tuple[object, ...]]) -> float:
    union = a | b
    if not union:
        return 0.0
    return float(1.0 - len(a & b) / len(union))


def empirical_two_sided_p_value(observed: int, samples: np.ndarray) -> float:
    if len(samples) == 0:
        return float("nan")
    center = float(samples.mean())
    observed_delta = abs(float(observed) - center)
    sample_delta = np.abs(samples - center)
    return float(min(1.0, (1.0 + np.sum(sample_delta >= observed_delta)) / (len(samples) + 1.0)))


if __name__ == "__main__":
    main()
