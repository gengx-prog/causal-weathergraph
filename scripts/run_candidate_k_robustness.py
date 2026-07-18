#!/usr/bin/env python
"""Run candidate-neighborhood robustness experiments."""

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
from causal_weathergraph.metrics import causal_chain_summary
from causal_weathergraph.regimes import define_regimes
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


DEFAULT_K_VALUES = (2, 4, 6, 8)
CORE_EDGE_TYPES = (
    "wind_to_humidity",
    "humidity_to_cloud_cover",
    "wind_to_cloud_cover",
    "humidity_to_humidity",
    "temperature_to_humidity",
    "temperature_to_cloud_cover",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "local_full_period_fast.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--k_values", nargs="*", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--max_lag", type=int, default=None)
    parser.add_argument("--edge_types", nargs="*", default=list(CORE_EDGE_TYPES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    output_root = Path(args.output_root)
    dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")

    base_causal_cfg = dict(config.get("causal", {}))
    max_lag = int(args.max_lag if args.max_lag is not None else base_causal_cfg.get("max_lag", 3))
    alpha = float(base_causal_cfg.get("alpha", 0.05))
    controls = base_causal_cfg.get("controls", {})
    regimes = define_regimes(dataset.data, dataset.variable_names, dataset.timestamps, config.get("regimes", {}))
    selected_types = set(args.edge_types)

    all_edges_by_k: dict[int, pd.DataFrame] = {}
    significant_by_k: dict[int, pd.DataFrame] = {}
    candidate_counts: dict[int, int] = {}

    for k in args.k_values:
        causal_cfg = dict(base_causal_cfg)
        causal_cfg["candidate_k_nearest"] = int(k)
        causal_cfg["max_lag"] = max_lag
        candidates = build_candidate_edges(
            dataset.variable_names,
            dataset.lat,
            dataset.lon,
            causal_cfg,
            data=dataset.data,
        )
        if selected_types:
            candidates = [edge for edge in candidates if edge.edge_type in selected_types]
        candidate_counts[int(k)] = len(candidates)
        edges = run_fast_granger_discovery(
            data=dataset.data,
            variable_names=dataset.variable_names,
            candidates=candidates,
            regimes=regimes,
            max_lag=max_lag,
            alpha=alpha,
            controls=controls,
            fdr_method=str(base_causal_cfg.get("fdr_method", "benjamini_hochberg")),
            region_lat=dataset.lat,
            region_lon=dataset.lon,
        )
        edges = edges.copy()
        edges["candidate_k_nearest"] = int(k)
        significant = edges[edges["significant"].astype(bool)].copy()
        all_edges_by_k[int(k)] = edges
        significant_by_k[int(k)] = significant
        print(f"k={k}: tested={len(edges)} significant={len(significant)} candidates={len(candidates)}", flush=True)

    edge_dir = ensure_dir(output_root / "causal_edges")
    report_dir = ensure_dir(output_root / "reports")
    all_edges = pd.concat(all_edges_by_k.values(), ignore_index=True) if all_edges_by_k else pd.DataFrame()
    significant_edges = (
        pd.concat(significant_by_k.values(), ignore_index=True) if significant_by_k else pd.DataFrame()
    )
    all_edges.to_csv(edge_dir / "candidate_k_robustness_edges.csv", index=False)
    significant_edges.to_csv(edge_dir / "candidate_k_robustness_significant_edges.csv", index=False)

    summary = summarize_k_robustness(all_edges_by_k, significant_by_k, candidate_counts, args.k_values)
    summary.to_csv(report_dir / "candidate_k_robustness_summary.csv", index=False)
    stable_edges = stable_edge_table(significant_by_k)
    stable_edges.to_csv(edge_dir / "candidate_k_robustness_stable_edges.csv", index=False)
    print(f"Saved candidate-k robustness reports to {report_dir}", flush=True)


def summarize_k_robustness(
    all_edges_by_k: dict[int, pd.DataFrame],
    significant_by_k: dict[int, pd.DataFrame],
    candidate_counts: dict[int, int],
    k_values: list[int],
) -> pd.DataFrame:
    sig_sets = {k: edge_set(df) for k, df in significant_by_k.items()}
    ordered_k = [int(k) for k in k_values if int(k) in significant_by_k]
    baseline_k = ordered_k[0] if ordered_k else None
    baseline_set = sig_sets.get(baseline_k, set()) if baseline_k is not None else set()
    persistent = set.intersection(*(sig_sets[k] for k in ordered_k)) if ordered_k else set()
    persistent_wh = {edge for edge in persistent if edge[2] == "wind" and edge[3] == "humidity"}
    persistent_hc = {edge for edge in persistent if edge[2] == "humidity" and edge[3] == "cloud_cover"}

    records = []
    for k in ordered_k:
        all_edges = all_edges_by_k[k]
        sig = significant_by_k[k]
        sig_ids = sig_sets[k]
        union = sig_ids | baseline_set
        chains = causal_chain_summary(sig)
        wh = sig[sig["edge_type"] == "wind_to_humidity"]
        hc = sig[sig["edge_type"] == "humidity_to_cloud_cover"]
        cross_region = sig["source_region"].to_numpy() != sig["target_region"].to_numpy() if len(sig) else np.array([])
        records.append(
            {
                "k": int(k),
                "candidate_edges": int(candidate_counts[k]),
                "tested_edges": int(len(all_edges)),
                "significant_edges": int(len(sig)),
                "significant_ratio": float(len(sig) / len(all_edges)) if len(all_edges) else 0.0,
                "unique_significant_edges": int(len(sig_ids)),
                "WH_edges": int(len(wh)),
                "HC_edges": int(len(hc)),
                "WHC_chains": int(chains["n_chains"].sum()) if not chains.empty else 0,
                "stable_WH": int(len(persistent_wh)),
                "stable_HC": int(len(persistent_hc)),
                "stable_WHC_chains": int(count_stable_whc_chains(sig, persistent)),
                "cross_region_frac": float(cross_region.mean()) if len(cross_region) else 0.0,
                "mean_WH_lag": float(wh["lag"].mean()) if len(wh) else np.nan,
                "mean_HC_lag": float(hc["lag"].mean()) if len(hc) else np.nan,
                "mean_effect_score": float(sig["effect_score"].abs().mean()) if len(sig) else 0.0,
                "overlap_with_k2": float(len(sig_ids & baseline_set) / len(union)) if union else 0.0,
                "recovered_k2_edges": int(len(sig_ids & baseline_set)),
                "new_edges_vs_k2": int(len(sig_ids - baseline_set)),
            }
        )
    return pd.DataFrame.from_records(records)


def count_stable_whc_chains(significant: pd.DataFrame, stable_ids: set[tuple[object, ...]]) -> int:
    if significant.empty or not stable_ids:
        return 0
    stable_sig = significant[
        [
            (
                int(row.source_region),
                int(row.target_region),
                str(row.source_var),
                str(row.target_var),
            )
            in stable_ids
            for row in significant.itertuples(index=False)
        ]
    ]
    if stable_sig.empty:
        return 0
    chains = causal_chain_summary(stable_sig)
    return int(chains["n_chains"].sum()) if not chains.empty else 0


def stable_edge_table(significant_by_k: dict[int, pd.DataFrame]) -> pd.DataFrame:
    if not significant_by_k:
        return pd.DataFrame(columns=["source_region", "target_region", "source_var", "target_var", "n_k_values"])
    records: dict[tuple[object, ...], set[int]] = {}
    for k, df in significant_by_k.items():
        for edge_id in edge_set(df):
            records.setdefault(edge_id, set()).add(int(k))
    max_count = len(significant_by_k)
    rows = []
    for edge_id, present_k in sorted(records.items()):
        if len(present_k) != max_count:
            continue
        source_region, target_region, source_var, target_var = edge_id
        rows.append(
            {
                "source_region": int(source_region),
                "target_region": int(target_region),
                "source_var": str(source_var),
                "target_var": str(target_var),
                "edge_type": f"{source_var}_to_{target_var}",
                "n_k_values": int(len(present_k)),
                "k_values": ",".join(str(k) for k in sorted(present_k)),
            }
        )
    return pd.DataFrame.from_records(rows)


if __name__ == "__main__":
    main()
