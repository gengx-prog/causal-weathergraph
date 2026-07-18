#!/usr/bin/env python
"""Run directionality and pseudo-causality controls for core pathways."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.candidate_edges import CandidateEdge, build_candidate_edges, edge_type_name
from causal_weathergraph.causal.fast_granger import run_fast_granger_discovery
from causal_weathergraph.data_io import load_dataset_npz
from causal_weathergraph.regimes import define_regimes
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


CORE_FORWARD_TYPES = {
    "wind_to_humidity",
    "humidity_to_cloud_cover",
    "wind_to_cloud_cover",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--random_seed", type=int, default=2026)
    parser.add_argument("--circular_shift_steps", type=int, default=365 * 4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    output_root = Path(args.output_root)
    dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")
    causal_cfg = config.get("causal", {})
    max_lag = int(causal_cfg.get("max_lag", 3))
    alpha = float(causal_cfg.get("alpha", 0.05))
    controls = causal_cfg.get("controls", {})
    regimes = define_regimes(dataset.data, dataset.variable_names, dataset.timestamps, config.get("regimes", {}))

    candidates = build_candidate_edges(
        dataset.variable_names,
        dataset.lat,
        dataset.lon,
        causal_cfg,
        data=dataset.data,
    )
    forward_candidates = [edge for edge in candidates if edge.edge_type in CORE_FORWARD_TYPES]
    reverse_candidates = reverse_edges(forward_candidates)

    rng = np.random.default_rng(args.random_seed)
    pieces: list[pd.DataFrame] = []

    pieces.append(
        run_control(
            "forward",
            dataset.data,
            regimes,
            dataset.variable_names,
            forward_candidates,
            max_lag,
            alpha,
            controls,
            dataset.lat,
            dataset.lon,
        )
    )
    pieces.append(
        run_control(
            "reverse",
            dataset.data,
            regimes,
            dataset.variable_names,
            reverse_candidates,
            max_lag,
            alpha,
            controls,
            dataset.lat,
            dataset.lon,
        )
    )

    permutation = rng.permutation(dataset.data.shape[0])
    shuffled_data = dataset.data[permutation]
    shuffled_regimes = {name: mask[permutation] for name, mask in regimes.items()}
    pieces.append(
        run_control(
            "time_shuffled",
            shuffled_data,
            shuffled_regimes,
            dataset.variable_names,
            forward_candidates,
            max_lag,
            alpha,
            controls,
            dataset.lat,
            dataset.lon,
        )
    )

    pieces.append(
        run_control(
            "future_to_past",
            dataset.data[::-1],
            {name: mask[::-1] for name, mask in regimes.items()},
            dataset.variable_names,
            forward_candidates,
            max_lag,
            alpha,
            controls,
            dataset.lat,
            dataset.lon,
        )
    )

    pieces.append(
        run_circular_shift_controls(
            dataset.data,
            regimes,
            dataset.variable_names,
            forward_candidates,
            max_lag,
            alpha,
            controls,
            dataset.lat,
            dataset.lon,
            int(args.circular_shift_steps),
        )
    )

    edge_dir = ensure_dir(output_root / "causal_edges")
    report_dir = ensure_dir(output_root / "reports")
    edges = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
    edges["display_edge_type"] = [
        display_edge_type(source, target)
        for source, target in zip(edges["source_var"], edges["target_var"], strict=False)
    ]
    edges.to_csv(edge_dir / "directionality_control_edges.csv", index=False)
    summary = summarize(edges)
    summary.to_csv(report_dir / "directionality_control_summary.csv", index=False)
    print(f"Saved directionality controls to {report_dir / 'directionality_control_summary.csv'}")


def run_control(
    test_type: str,
    data: np.ndarray,
    regimes: dict[str, np.ndarray],
    variable_names: list[str],
    candidates: list[CandidateEdge],
    max_lag: int,
    alpha: float,
    controls: dict,
    region_lat: np.ndarray,
    region_lon: np.ndarray,
) -> pd.DataFrame:
    edges = run_fast_granger_discovery(
        data=data,
        variable_names=variable_names,
        candidates=candidates,
        regimes=regimes,
        max_lag=max_lag,
        alpha=alpha,
        controls=controls,
        region_lat=region_lat,
        region_lon=region_lon,
    )
    edges = edges.copy()
    edges["test_type"] = test_type
    return edges


def run_circular_shift_controls(
    data: np.ndarray,
    regimes: dict[str, np.ndarray],
    variable_names: list[str],
    candidates: list[CandidateEdge],
    max_lag: int,
    alpha: float,
    controls: dict,
    region_lat: np.ndarray,
    region_lon: np.ndarray,
    shift_steps: int,
) -> pd.DataFrame:
    pieces = []
    for edge_type, source_var in [
        ("wind_to_humidity", "wind"),
        ("humidity_to_cloud_cover", "humidity"),
        ("wind_to_cloud_cover", "wind"),
    ]:
        shifted = data.copy()
        var_index = variable_names.index(source_var)
        shifted[:, :, var_index] = np.roll(shifted[:, :, var_index], shift_steps, axis=0)
        subset = [edge for edge in candidates if edge.edge_type == edge_type]
        if not subset:
            continue
        pieces.append(
            run_control(
                "circular_shift",
                shifted,
                regimes,
                variable_names,
                subset,
                max_lag,
                alpha,
                controls,
                region_lat,
                region_lon,
            )
        )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def reverse_edges(edges: list[CandidateEdge]) -> list[CandidateEdge]:
    reversed_by_key: dict[tuple[int, int, str, str], CandidateEdge] = {}
    for edge in edges:
        key = (edge.target_region, edge.source_region, edge.target_var, edge.source_var)
        reversed_by_key[key] = CandidateEdge(
            source_region=edge.target_region,
            target_region=edge.source_region,
            source_var=edge.target_var,
            target_var=edge.source_var,
            edge_type=edge_type_name(edge.target_var, edge.source_var),
            distance_km=edge.distance_km,
        )
    return list(reversed_by_key.values())


def summarize(edges: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (test_type, edge_type), group in edges.groupby(["test_type", "display_edge_type"], sort=True):
        sig = group[group["significant"].astype(bool)]
        records.append(
            {
                "test_type": test_type,
                "edge_type": edge_type,
                "tested_edges": int(len(group)),
                "significant_edges": int(len(sig)),
                "significant_ratio": float(len(sig) / len(group)) if len(group) else 0.0,
                "mean_abs_coef": float(sig["effect_coefficient"].abs().mean()) if len(sig) else 0.0,
                "mean_effect_score": float(sig["effect_score"].abs().mean()) if len(sig) else 0.0,
                "mean_stability": float(sig["stability"].mean()) if "stability" in sig.columns and len(sig) else np.nan,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["edge_type", "test_type"])


def display_edge_type(source_var: str, target_var: str) -> str:
    labels = {
        "wind": "Wind",
        "humidity": "Humidity",
        "cloud_cover": "Cloud",
        "temperature": "Temperature",
    }
    return f"{labels.get(source_var, source_var)} -> {labels.get(target_var, target_var)}"


if __name__ == "__main__":
    main()
