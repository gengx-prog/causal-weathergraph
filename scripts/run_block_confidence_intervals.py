#!/usr/bin/env python
"""Estimate year-block bootstrap confidence intervals for stable core edges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.candidate_edges import CandidateEdge
from causal_weathergraph.causal.fast_granger import prepare_target_fit, test_residualized_source
from causal_weathergraph.data_io import load_dataset_npz
from causal_weathergraph.utils import ensure_dir, setup_logging


CORE_EDGE_TYPES = (
    "wind_to_humidity",
    "humidity_to_cloud_cover",
    "wind_to_cloud_cover",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--random_seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    output_root = Path(args.output_root)
    dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")
    controlled_path = output_root / "causal_edges" / "multivariable_control_significant_edges.csv"
    stable_path = output_root / "causal_edges" / "stable_edges.csv"
    if not controlled_path.exists():
        raise FileNotFoundError(f"Missing {controlled_path}. Run the multivariable-control experiment first.")
    controlled = pd.read_csv(controlled_path)
    stable = pd.read_csv(stable_path)
    selected = select_edges(controlled, stable, args.top_n)
    if selected.empty:
        raise RuntimeError("No all-regime stable edges survived the multivariable control.")

    years = pd.DatetimeIndex(dataset.timestamps).year.to_numpy()
    unique_years = np.unique(years)
    var_index = {name: idx for idx, name in enumerate(dataset.variable_names)}
    detail_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    rng = np.random.default_rng(args.random_seed)
    controls = {
        "include_target_own_lags": True,
        "include_target_region_all_vars": True,
    }

    for row in selected.itertuples(index=False):
        edge = CandidateEdge(
            source_region=int(row.source_region),
            target_region=int(row.target_region),
            source_var=str(row.source_var),
            target_var=str(row.target_var),
            edge_type=str(row.edge_type),
            distance_km=float(row.distance_km),
        )
        block_results = []
        for year in unique_years:
            mask = years == year
            fit = prepare_target_fit(
                dataset.data,
                dataset.variable_names,
                edge.target_region,
                var_index[edge.target_var],
                mask,
                max_lag=3,
                controls=controls,
                exclude_control=(
                    (var_index[edge.source_var], int(row.lag))
                    if edge.source_region == edge.target_region
                    else None
                ),
            )
            if fit is None:
                continue
            result = test_residualized_source(
                dataset.data,
                fit,
                edge,
                var_index[edge.source_var],
                int(row.lag),
                max_lag=3,
            )
            if result is None:
                continue
            result["year"] = int(year)
            block_results.append(result)
            detail_records.append(
                {
                    "edge_type": edge.edge_type,
                    "source_region": edge.source_region,
                    "target_region": edge.target_region,
                    "lag": int(row.lag),
                    "year": int(year),
                    "beta": float(result["effect_coefficient"]),
                    "delta_r2": float(result["delta_r2"]),
                    "partial_r2": float(result["partial_r2"]),
                }
            )
        if not block_results:
            continue
        betas = np.asarray([x["effect_coefficient"] for x in block_results], dtype=float)
        partial_r2 = np.asarray([x["partial_r2"] for x in block_results], dtype=float)
        bootstrap_means = rng.choice(
            betas,
            size=(args.n_bootstrap, len(betas)),
            replace=True,
        ).mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
        full_beta = float(row.effect_coefficient)
        summary_records.append(
            {
                "edge_type": edge.edge_type,
                "source_region": edge.source_region,
                "target_region": edge.target_region,
                "lag": int(row.lag),
                "n_year_blocks": int(len(betas)),
                "full_period_beta": full_beta,
                "mean_year_beta": float(betas.mean()),
                "year_block_bootstrap_ci95_low": float(ci_low),
                "year_block_bootstrap_ci95_high": float(ci_high),
                "bootstrap_ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                "sign_consistency": float((np.sign(betas) == np.sign(full_beta)).mean()),
                "median_year_partial_r2": float(np.median(partial_r2)),
            }
        )

    details = pd.DataFrame.from_records(detail_records)
    edge_summary = pd.DataFrame.from_records(summary_records)
    type_summary = summarize_by_type(edge_summary, selected)
    edge_dir = ensure_dir(output_root / "causal_edges")
    report_dir = ensure_dir(output_root / "reports")
    details.to_csv(edge_dir / "year_block_effect_estimates.csv", index=False)
    edge_summary.to_csv(report_dir / "year_block_confidence_intervals.csv", index=False)
    type_summary.to_csv(report_dir / "year_block_confidence_summary.csv", index=False)
    print(f"Saved year-block confidence intervals to {report_dir}.", flush=True)


def select_edges(controlled: pd.DataFrame, stable: pd.DataFrame, top_n: int) -> pd.DataFrame:
    stable_all = stable[stable["regime"] == "all"].copy()
    stable_keys = {
        (
            int(row.source_region),
            int(row.target_region),
            str(row.source_var),
            str(row.target_var),
            int(row.lag),
        )
        for row in stable_all.itertuples(index=False)
    }
    eligible = controlled[controlled["regime"] == "all"].copy()
    eligible = eligible[
        [
            (
                int(row.source_region),
                int(row.target_region),
                str(row.source_var),
                str(row.target_var),
                int(row.lag),
            )
            in stable_keys
            for row in eligible.itertuples(index=False)
        ]
    ].copy()
    eligible["abs_effect_score"] = eligible["effect_score"].abs()
    pieces = []
    for edge_type in CORE_EDGE_TYPES:
        pieces.append(
            eligible[eligible["edge_type"] == edge_type]
            .sort_values(["partial_r2", "abs_effect_score"], ascending=[False, False])
            .head(top_n)
        )
    return pd.concat(pieces, ignore_index=True)


def summarize_by_type(edge_summary: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    records = []
    for edge_type in CORE_EDGE_TYPES:
        subset = edge_summary[edge_summary["edge_type"] == edge_type]
        selected_subset = selected[selected["edge_type"] == edge_type]
        records.append(
            {
                "edge_type": edge_type,
                "n_edges": int(len(subset)),
                "median_full_period_abs_beta": float(selected_subset["effect_abs"].median()) if len(selected_subset) else np.nan,
                "median_full_period_delta_r2": float(selected_subset["delta_r2"].median()) if len(selected_subset) else np.nan,
                "median_full_period_partial_r2": float(selected_subset["partial_r2"].median()) if len(selected_subset) else np.nan,
                "fraction_block_ci_excludes_zero": float(subset["bootstrap_ci_excludes_zero"].mean()) if len(subset) else np.nan,
                "median_sign_consistency": float(subset["sign_consistency"].median()) if len(subset) else np.nan,
                "median_year_partial_r2": float(subset["median_year_partial_r2"].median()) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    main()
