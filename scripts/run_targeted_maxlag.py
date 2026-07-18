#!/usr/bin/env python
"""Run a targeted long-lag Granger experiment."""

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
from causal_weathergraph.metrics import causal_chain_summary
from causal_weathergraph.regimes import define_regimes
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging

DEFAULT_EDGE_TYPES = (
    "wind_to_humidity",
    "humidity_to_cloud_cover",
    "wind_to_cloud_cover",
    "humidity_to_humidity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--max_lag", type=int, default=12)
    parser.add_argument("--edge_types", nargs="*", default=list(DEFAULT_EDGE_TYPES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    output_root = Path(args.output_root)
    dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")
    causal_cfg = dict(config.get("causal", {}))
    regimes = define_regimes(dataset.data, dataset.variable_names, dataset.timestamps, config.get("regimes", {}))
    candidates = build_candidate_edges(dataset.variable_names, dataset.lat, dataset.lon, causal_cfg, data=dataset.data)
    selected_types = set(args.edge_types)
    targeted = [edge for edge in candidates if edge.edge_type in selected_types]
    edge_dir = ensure_dir(output_root / "causal_edges")
    report_dir = ensure_dir(output_root / "reports")
    pd.DataFrame([edge.to_dict() for edge in targeted]).to_csv(
        edge_dir / f"targeted_maxlag{args.max_lag}_candidate_edges.csv", index=False
    )

    edges = run_fast_granger_discovery(
        data=dataset.data,
        variable_names=dataset.variable_names,
        candidates=targeted,
        regimes=regimes,
        max_lag=args.max_lag,
        alpha=float(causal_cfg.get("alpha", 0.05)),
        controls=causal_cfg.get("controls", {}),
        fdr_method=str(causal_cfg.get("fdr_method", "benjamini_hochberg")),
        region_lat=dataset.lat,
        region_lon=dataset.lon,
    )
    significant = edges[edges["significant"].astype(bool)].copy()
    edges.to_csv(edge_dir / f"targeted_maxlag{args.max_lag}_all_edges.csv", index=False)
    significant.to_csv(edge_dir / f"targeted_maxlag{args.max_lag}_significant_edges.csv", index=False)
    lag_summary(significant, list(DEFAULT_EDGE_TYPES), args.max_lag).to_csv(
        report_dir / f"targeted_maxlag{args.max_lag}_lag_summary.csv", index=False
    )
    causal_chain_summary(significant).to_csv(
        report_dir / f"targeted_maxlag{args.max_lag}_causal_chain_summary.csv", index=False
    )
    print(f"Saved targeted max_lag={args.max_lag} outputs to {edge_dir} and {report_dir}")


def lag_summary(significant: pd.DataFrame, edge_types: list[str], max_lag: int) -> pd.DataFrame:
    records = []
    for edge_type in edge_types:
        df = significant[significant["edge_type"] == edge_type]
        lags = df["lag"].to_numpy(dtype=int) if len(df) else np.array([], dtype=int)
        if len(lags):
            counts = pd.Series(lags).value_counts().sort_index()
            peak_lag = int(counts.idxmax())
            mean_lag = float(np.mean(lags))
            median_lag = float(np.median(lags))
        else:
            peak_lag = np.nan
            mean_lag = np.nan
            median_lag = np.nan
        records.append(
            {
                "edge_type": edge_type,
                "peak_lag": peak_lag,
                "mean_lag": mean_lag,
                "median_lag": median_lag,
                "lag_1_frac": frac_between(lags, 1, 1),
                "lag_2_3_frac": frac_between(lags, 2, 3),
                "lag_4_8_frac": frac_between(lags, 4, 8),
                "lag_9_12_frac": frac_between(lags, 9, min(12, max_lag)),
            }
        )
    return pd.DataFrame.from_records(records)


def frac_between(lags: np.ndarray, low: int, high: int) -> float:
    if len(lags) == 0:
        return 0.0
    return float(((lags >= low) & (lags <= high)).mean())


if __name__ == "__main__":
    main()
