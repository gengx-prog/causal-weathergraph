#!/usr/bin/env python
"""Re-estimate the graph with effect sizes and target-region multivariable controls."""

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
from causal_weathergraph.data_io import load_dataset_npz
from causal_weathergraph.metrics import causal_chain_summary, summary_metrics
from causal_weathergraph.regimes import define_regimes
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


CORE_EDGE_TYPES = (
    "wind_to_humidity",
    "humidity_to_cloud_cover",
    "wind_to_cloud_cover",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "local_full_period_fast.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    output_root = Path(args.output_root)
    dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")
    causal_cfg = dict(config.get("causal", {}))
    regimes = define_regimes(
        dataset.data,
        dataset.variable_names,
        dataset.timestamps,
        config.get("regimes", {}),
    )
    candidates = build_candidate_edges(
        dataset.variable_names,
        dataset.lat,
        dataset.lon,
        causal_cfg,
        data=dataset.data,
    )
    common = {
        "data": dataset.data,
        "variable_names": dataset.variable_names,
        "candidates": candidates,
        "regimes": regimes,
        "max_lag": int(causal_cfg.get("max_lag", 3)),
        "alpha": float(causal_cfg.get("alpha", 0.05)),
        "fdr_method": str(causal_cfg.get("fdr_method", "benjamini_hochberg")),
        "region_lat": dataset.lat,
        "region_lon": dataset.lon,
    }

    print("Running own-history baseline with expanded effect-size outputs.", flush=True)
    baseline = run_fast_granger_discovery(
        **common,
        controls={
            "include_target_own_lags": True,
            "include_target_region_all_vars": False,
        },
    )
    print("Running target-region multivariable control model.", flush=True)
    multivariable = run_fast_granger_discovery(
        **common,
        controls={
            "include_target_own_lags": True,
            "include_target_region_all_vars": True,
        },
    )

    baseline_sig = baseline[baseline["significant"].astype(bool)].copy()
    multivariable_sig = multivariable[multivariable["significant"].astype(bool)].copy()
    edge_dir = ensure_dir(output_root / "causal_edges")
    report_dir = ensure_dir(output_root / "reports")

    baseline.to_csv(edge_dir / "effect_size_baseline_edges.csv", index=False)
    baseline_sig.to_csv(edge_dir / "effect_size_baseline_significant_edges.csv", index=False)
    multivariable.to_csv(edge_dir / "multivariable_control_edges.csv", index=False)
    multivariable_sig.to_csv(edge_dir / "multivariable_control_significant_edges.csv", index=False)

    effect_summary = pd.concat(
        [
            summarize_effects(baseline_sig, "own_history"),
            summarize_effects(multivariable_sig, "target_region_all_variables"),
        ],
        ignore_index=True,
    )
    effect_summary.to_csv(report_dir / "effect_size_summary.csv", index=False)
    compare_graphs(baseline_sig, multivariable_sig).to_csv(
        report_dir / "multivariable_control_overlap.csv",
        index=False,
    )
    summary_metrics(multivariable, multivariable_sig).to_csv(
        report_dir / "multivariable_control_summary.csv",
        index=False,
    )
    causal_chain_summary(multivariable_sig).to_csv(
        report_dir / "multivariable_control_chain_summary.csv",
        index=False,
    )
    print(
        f"Saved effect-size and multivariable-control outputs under {report_dir} and {edge_dir}.",
        flush=True,
    )


def summarize_effects(significant: pd.DataFrame, model: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for regime in sorted(significant["regime"].dropna().unique()):
        regime_df = significant[significant["regime"] == regime]
        for edge_type in (*CORE_EDGE_TYPES, "all_candidate_types"):
            subset = regime_df if edge_type == "all_candidate_types" else regime_df[regime_df["edge_type"] == edge_type]
            if subset.empty:
                continue
            records.append(
                {
                    "model": model,
                    "regime": regime,
                    "edge_type": edge_type,
                    "n_significant": int(len(subset)),
                    "median_beta": float(subset["effect_coefficient"].median()),
                    "median_abs_beta": float(subset["effect_abs"].median()),
                    "q25_abs_beta": float(subset["effect_abs"].quantile(0.25)),
                    "q75_abs_beta": float(subset["effect_abs"].quantile(0.75)),
                    "median_delta_r2": float(subset["delta_r2"].median()),
                    "median_partial_r2": float(subset["partial_r2"].median()),
                    "median_cohen_f2": float(subset["cohen_f2"].median()),
                    "positive_fraction": float((subset["effect_coefficient"] > 0).mean()),
                    "model_ci_excludes_zero_fraction": float(
                        (
                            (subset["effect_ci95_low_model"] > 0)
                            | (subset["effect_ci95_high_model"] < 0)
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def compare_graphs(baseline: pd.DataFrame, controlled: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    regimes = sorted(set(baseline["regime"]).union(controlled["regime"]))
    for regime in regimes:
        base_regime = baseline[baseline["regime"] == regime]
        ctrl_regime = controlled[controlled["regime"] == regime]
        for edge_type in (*CORE_EDGE_TYPES, "all_candidate_types"):
            base_subset = base_regime if edge_type == "all_candidate_types" else base_regime[base_regime["edge_type"] == edge_type]
            ctrl_subset = ctrl_regime if edge_type == "all_candidate_types" else ctrl_regime[ctrl_regime["edge_type"] == edge_type]
            base_set = edge_keys(base_subset)
            ctrl_set = edge_keys(ctrl_subset)
            union = base_set | ctrl_set
            intersection = base_set & ctrl_set
            records.append(
                {
                    "regime": regime,
                    "edge_type": edge_type,
                    "baseline_significant": int(len(base_set)),
                    "controlled_significant": int(len(ctrl_set)),
                    "intersection": int(len(intersection)),
                    "baseline_recovery_fraction": float(len(intersection) / len(base_set)) if base_set else np.nan,
                    "controlled_precision_vs_baseline": float(len(intersection) / len(ctrl_set)) if ctrl_set else np.nan,
                    "jaccard_similarity": float(len(intersection) / len(union)) if union else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def edge_keys(df: pd.DataFrame) -> set[tuple[object, ...]]:
    return {
        (
            str(row.regime),
            int(row.source_region),
            int(row.target_region),
            str(row.source_var),
            str(row.target_var),
            int(row.lag),
        )
        for row in df.itertuples(index=False)
    }


if __name__ == "__main__":
    main()
