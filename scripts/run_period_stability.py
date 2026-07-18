#!/usr/bin/env python
"""Evaluate core graph stability across historical periods."""

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
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


PERIODS = (
    ("P1", "1979-01-01", "1989-12-31 23:59:59"),
    ("P2", "1990-01-01", "1999-12-31 23:59:59"),
    ("P3", "2000-01-01", "2009-12-31 23:59:59"),
    ("P4", "2010-01-01", "2018-12-31 23:59:59"),
    ("P5", "2019-01-01", "2025-12-31 23:59:59"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "local_full_period_fast.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--max_lag", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    output_root = Path(args.output_root)
    dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")
    if dataset.timestamps is None:
        raise ValueError("Period stability requires timestamps in region_series.npz.")

    causal_cfg = dict(config.get("causal", {}))
    max_lag = int(args.max_lag if args.max_lag is not None else causal_cfg.get("max_lag", 3))
    causal_cfg["max_lag"] = max_lag
    candidates = build_candidate_edges(
        dataset.variable_names,
        dataset.lat,
        dataset.lon,
        causal_cfg,
        data=dataset.data,
    )
    regimes = period_masks(dataset.timestamps)
    edges = run_fast_granger_discovery(
        data=dataset.data,
        variable_names=dataset.variable_names,
        candidates=candidates,
        regimes=regimes,
        max_lag=max_lag,
        alpha=float(causal_cfg.get("alpha", 0.05)),
        controls=causal_cfg.get("controls", {}),
        fdr_method=str(causal_cfg.get("fdr_method", "benjamini_hochberg")),
        region_lat=dataset.lat,
        region_lon=dataset.lon,
    )
    significant = edges[edges["significant"].astype(bool)].copy()

    edge_dir = ensure_dir(output_root / "causal_edges")
    report_dir = ensure_dir(output_root / "reports")
    edges.to_csv(edge_dir / "period_stability_edges.csv", index=False)
    significant.to_csv(edge_dir / "period_stability_significant_edges.csv", index=False)
    full_significant = load_full_significant(output_root)
    summary = summarize_periods(edges, significant, full_significant, regimes)
    summary.to_csv(report_dir / "period_stability_summary.csv", index=False)
    persistent = persistent_period_edges(significant)
    persistent.to_csv(edge_dir / "period_stability_persistent_edges.csv", index=False)
    print(f"Saved period stability reports to {report_dir}", flush=True)


def period_masks(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    ts = pd.DatetimeIndex(timestamps)
    masks = {}
    for name, start, end in PERIODS:
        masks[name] = np.asarray((ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end)), dtype=bool)
    return masks


def load_full_significant(output_root: Path) -> pd.DataFrame:
    path = output_root / "causal_edges" / "significant_edges.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "regime" in df.columns:
        all_df = df[df["regime"] == "all"].copy()
        return all_df if not all_df.empty else df
    return df


def summarize_periods(
    all_edges: pd.DataFrame,
    significant: pd.DataFrame,
    full_significant: pd.DataFrame,
    regimes: dict[str, np.ndarray],
) -> pd.DataFrame:
    full_ids = edge_set(full_significant) if not full_significant.empty else set()
    full_wh = edge_set(full_significant[full_significant["edge_type"] == "wind_to_humidity"]) if not full_significant.empty else set()
    full_hc = (
        edge_set(full_significant[full_significant["edge_type"] == "humidity_to_cloud_cover"])
        if not full_significant.empty
        else set()
    )
    records = []
    for name, start, end in PERIODS:
        tested = all_edges[all_edges["regime"] == name]
        sig = significant[significant["regime"] == name]
        sig_ids = edge_set(sig)
        union = sig_ids | full_ids
        wh = sig[sig["edge_type"] == "wind_to_humidity"]
        hc = sig[sig["edge_type"] == "humidity_to_cloud_cover"]
        chains = causal_chain_summary(sig)
        records.append(
            {
                "period": name,
                "time_range": f"{start[:4]}-{end[:4]}",
                "n_samples": int(regimes[name].sum()),
                "tested_edges": int(len(tested)),
                "significant_edges": int(len(sig)),
                "significant_ratio": float(len(sig) / len(tested)) if len(tested) else 0.0,
                "WH_edges": int(len(wh)),
                "HC_edges": int(len(hc)),
                "WC_edges": int((sig["edge_type"] == "wind_to_cloud_cover").sum()) if len(sig) else 0,
                "WHC_chains": int(chains["n_chains"].sum()) if not chains.empty else 0,
                "stable_WH": int(len(edge_set(wh) & full_wh)),
                "stable_HC": int(len(edge_set(hc) & full_hc)),
                "mean_WH_lag": float(wh["lag"].mean()) if len(wh) else np.nan,
                "mean_HC_lag": float(hc["lag"].mean()) if len(hc) else np.nan,
                "mean_effect_score": float(sig["effect_score"].abs().mean()) if len(sig) else 0.0,
                "overlap_with_full": float(len(sig_ids & full_ids) / len(union)) if union else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def persistent_period_edges(significant: pd.DataFrame) -> pd.DataFrame:
    if significant.empty:
        return pd.DataFrame(columns=["source_region", "target_region", "source_var", "target_var", "edge_type", "periods"])
    sets = {period: edge_set(group) for period, group in significant.groupby("regime")}
    if not sets:
        return pd.DataFrame()
    persistent = set.intersection(*sets.values())
    rows = []
    for source_region, target_region, source_var, target_var in sorted(persistent):
        rows.append(
            {
                "source_region": int(source_region),
                "target_region": int(target_region),
                "source_var": str(source_var),
                "target_var": str(target_var),
                "edge_type": f"{source_var}_to_{target_var}",
                "periods": ",".join(sorted(sets)),
            }
        )
    return pd.DataFrame.from_records(rows)


if __name__ == "__main__":
    main()
