"""Utilities for preprocessing robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidate_edges import build_candidate_edges
from .causal.fast_granger import run_fast_granger_discovery
from .causal.graph_utils import edge_set
from .data_io import AtmosphericDataset, load_atmospheric_data, load_dataset_npz
from .metrics import causal_chain_summary
from .preprocessing import fill_missing_values, preprocess_dataset, remove_seasonal_component, standardize_per_node
from .regions import aggregate_regions, save_region_outputs
from .regimes import define_regimes
from .utils import ensure_dir, get_logger

logger = get_logger("preprocessing_robustness")


@dataclass(frozen=True)
class PreprocessSetting:
    """One preprocessing setting for robustness comparison."""

    name: str
    description: str
    remove_monthly: bool
    first_difference: bool
    standardize: bool
    rolling_anomaly: bool = False
    standardize_after_region: bool = False


SETTINGS: dict[str, PreprocessSetting] = {
    "P1_monthly": PreprocessSetting(
        "P1_monthly",
        "monthly climatology removal + standardization",
        remove_monthly=True,
        first_difference=False,
        standardize=True,
    ),
    "P2_monthly_diff": PreprocessSetting(
        "P2_monthly_diff",
        "monthly climatology removal + first difference + standardization",
        remove_monthly=True,
        first_difference=True,
        standardize=True,
    ),
    "P3_diff_only": PreprocessSetting(
        "P3_diff_only",
        "first difference only + standardization",
        remove_monthly=False,
        first_difference=True,
        standardize=True,
    ),
    "P4_rolling": PreprocessSetting(
        "P4_rolling",
        "rolling anomaly + standardization",
        remove_monthly=False,
        first_difference=False,
        standardize=True,
        rolling_anomaly=True,
    ),
    "P5_monthly_regionstd": PreprocessSetting(
        "P5_monthly_regionstd",
        "monthly climatology removal + region-level standardization",
        remove_monthly=True,
        first_difference=False,
        standardize=False,
        standardize_after_region=True,
    ),
}


def run_preprocessing_setting(
    setting: PreprocessSetting,
    config: dict[str, Any],
    output_root: Path,
    raw: AtmosphericDataset | None = None,
    reuse_p1: bool = True,
) -> tuple[AtmosphericDataset, pd.DataFrame]:
    """Create or load a region-level dataset for one preprocessing setting."""
    setting_root = output_root / "robustness" / "preprocessing" / setting.name
    region_path = setting_root / "processed" / "region_series.npz"
    region_meta_path = setting_root / "regions" / "region_metadata.csv"
    if region_path.exists() and region_meta_path.exists():
        return load_dataset_npz(region_path), pd.read_csv(region_meta_path)

    if setting.name == "P1_monthly" and reuse_p1:
        baseline = load_dataset_npz(output_root / "processed" / "region_series.npz")
        regions = pd.read_csv(output_root / "regions" / "region_metadata.csv")
        ensure_dir(setting_root / "processed")
        ensure_dir(setting_root / "regions")
        return baseline, regions

    if raw is None:
        raw = load_atmospheric_data(config.get("data", {}).get("root"), config.get("data", {}))

    processed = preprocess_for_setting(raw, setting, config.get("preprocess", {}))
    aggregation = aggregate_regions(processed, config.get("regions", {}))
    if setting.standardize_after_region:
        data, mean, std = standardize_per_node(aggregation.dataset.data)
        aggregation.dataset.data = data
        aggregation.dataset.metadata = {
            **aggregation.dataset.metadata,
            "region_standardized": True,
            "region_standardization_mean_shape": tuple(mean.shape),
            "region_standardization_std_shape": tuple(std.shape),
        }
    save_region_outputs(aggregation, setting_root)
    return aggregation.dataset, aggregation.metadata


def preprocess_for_setting(
    dataset: AtmosphericDataset,
    setting: PreprocessSetting,
    base_preprocess_config: dict[str, Any],
) -> AtmosphericDataset:
    """Apply a configured preprocessing variant at node level."""
    if not setting.rolling_anomaly:
        cfg = dict(base_preprocess_config)
        cfg["remove_monthly_climatology"] = bool(setting.remove_monthly)
        cfg["use_first_difference"] = bool(setting.first_difference)
        cfg["standardize"] = bool(setting.standardize)
        return preprocess_dataset(dataset, cfg)[0]

    config = dict(base_preprocess_config)
    data = np.asarray(dataset.data, dtype=float).copy()
    max_nan_ratio = float(config.get("max_nan_ratio_per_node", 0.2))
    node_nan_ratio = np.mean(~np.isfinite(data), axis=(0, 2))
    keep_nodes = node_nan_ratio <= max_nan_ratio
    data = data[:, keep_nodes, :]
    lat = dataset.lat[keep_nodes]
    lon = dataset.lon[keep_nodes]
    node_ids = dataset.node_ids[keep_nodes] if dataset.node_ids is not None else None
    data = fill_missing_values(data, interpolation_limit=int(config.get("interpolation_limit", 6)))
    data = rolling_anomaly(data, window=int(config.get("rolling_anomaly_window", 31)))
    timestamps = dataset.timestamps
    if setting.first_difference:
        data = np.diff(data, axis=0)
        timestamps = timestamps[1:] if timestamps is not None else None
    if setting.standardize:
        data = standardize_per_node(data)[0]
    return AtmosphericDataset(
        data=data,
        variable_names=list(dataset.variable_names),
        lat=lat,
        lon=lon,
        timestamps=timestamps,
        node_ids=node_ids,
        metadata={**dataset.metadata, "preprocess_setting": setting.name},
    )


def rolling_anomaly(data: np.ndarray, window: int = 31) -> np.ndarray:
    """Remove a centered rolling mean from each node-variable series."""
    window = max(3, int(window))
    out = np.asarray(data, dtype=float).copy()
    n_time, n_node, n_var = out.shape
    for node in range(n_node):
        for var in range(n_var):
            series = pd.Series(out[:, node, var])
            rolling = series.rolling(window=window, center=True, min_periods=1).mean()
            out[:, node, var] = (series - rolling).to_numpy(dtype=float)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def run_granger_for_region_dataset(
    dataset: AtmosphericDataset,
    config: dict[str, Any],
    setting_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the fast Granger backend for a robustness dataset."""
    causal_cfg = dict(config.get("causal", {}))
    regimes = define_regimes(dataset.data, dataset.variable_names, dataset.timestamps, config.get("regimes", {}))
    candidates = build_candidate_edges(dataset.variable_names, dataset.lat, dataset.lon, causal_cfg, data=dataset.data)
    edges = run_fast_granger_discovery(
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
    )
    edges = edges.copy()
    edges["preprocess_setting"] = setting_name
    significant = edges[edges["significant"].astype(bool)].copy()
    return edges, significant


def summarize_preprocessing(
    all_edges_by_setting: dict[str, pd.DataFrame],
    significant_by_setting: dict[str, pd.DataFrame],
    setting_order: list[str],
) -> pd.DataFrame:
    """Summarize core pathway metrics for preprocessing variants."""
    persistent = set.intersection(*(edge_set(significant_by_setting[name]) for name in setting_order))
    persistent_wh = {edge for edge in persistent if edge[2] == "wind" and edge[3] == "humidity"}
    persistent_hc = {edge for edge in persistent if edge[2] == "humidity" and edge[3] == "cloud_cover"}
    rows = []
    for name in setting_order:
        all_edges = all_edges_by_setting[name]
        sig = significant_by_setting[name]
        wh = sig[sig["edge_type"] == "wind_to_humidity"]
        hc = sig[sig["edge_type"] == "humidity_to_cloud_cover"]
        wc = sig[sig["edge_type"] == "wind_to_cloud_cover"]
        chains = causal_chain_summary(sig)
        rows.append(
            {
                "preprocess_setting": name,
                "tested_edges": int(len(all_edges)),
                "significant_edges": int(len(sig)),
                "significant_ratio": float(len(sig) / len(all_edges)) if len(all_edges) else 0.0,
                "WH_edges": int(len(wh)),
                "HC_edges": int(len(hc)),
                "WC_edges": int(len(wc)),
                "WHC_chains": int(chains["n_chains"].sum()) if not chains.empty else 0,
                "stable_WH": int(len(persistent_wh)),
                "stable_HC": int(len(persistent_hc)),
                "mean_WH_lag": float(wh["lag"].mean()) if len(wh) else np.nan,
                "mean_HC_lag": float(hc["lag"].mean()) if len(hc) else np.nan,
                "mean_effect_score": float(sig["effect_score"].abs().mean()) if len(sig) else 0.0,
            }
        )
    return pd.DataFrame.from_records(rows)


def preprocessing_overlap(
    significant_by_setting: dict[str, pd.DataFrame],
    setting_order: list[str],
    baseline: str = "P1_monthly",
) -> pd.DataFrame:
    """Compute overlap against the baseline preprocessing setting."""
    base = significant_by_setting[baseline]
    base_all = edge_set(base)
    base_wh = edge_set(base[base["edge_type"] == "wind_to_humidity"])
    base_hc = edge_set(base[base["edge_type"] == "humidity_to_cloud_cover"])
    base_chains = chain_set(base)
    rows = []
    for name in setting_order:
        sig = significant_by_setting[name]
        rows.append(
            {
                "preprocess_setting": name,
                "edge_overlap_with_P1": jaccard(edge_set(sig), base_all),
                "WH_overlap_with_P1": jaccard(edge_set(sig[sig["edge_type"] == "wind_to_humidity"]), base_wh),
                "HC_overlap_with_P1": jaccard(edge_set(sig[sig["edge_type"] == "humidity_to_cloud_cover"]), base_hc),
                "WHC_chain_overlap_with_P1": jaccard(chain_set(sig), base_chains),
            }
        )
    return pd.DataFrame.from_records(rows)


def chain_set(edges: pd.DataFrame) -> set[tuple[int, int, int, str]]:
    """Return WHC chain identities without lag."""
    if edges.empty:
        return set()
    chains: set[tuple[int, int, int, str]] = set()
    for regime, group in edges.groupby("regime"):
        wh = group[group["edge_type"] == "wind_to_humidity"]
        hc = group[group["edge_type"] == "humidity_to_cloud_cover"]
        hc_by_source = {int(source): df for source, df in hc.groupby("source_region")}
        for left in wh.itertuples(index=False):
            downstream = hc_by_source.get(int(left.target_region))
            if downstream is None:
                continue
            for right in downstream.itertuples(index=False):
                chains.add((int(left.source_region), int(left.target_region), int(right.target_region), str(regime)))
    return chains


def jaccard(a: set[object], b: set[object]) -> float:
    """Jaccard similarity for two sets."""
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def plot_preprocessing_heatmap(overlap: pd.DataFrame, path: Path) -> None:
    """Save a compact overlap heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["edge_overlap_with_P1", "WH_overlap_with_P1", "HC_overlap_with_P1", "WHC_chain_overlap_with_P1"]
    values = overlap.set_index("preprocess_setting")[metrics].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["All", "WH", "HC", "WHC"], rotation=0)
    ax.set_yticks(np.arange(len(overlap)))
    ax.set_yticklabels(overlap["preprocess_setting"].tolist())
    ax.set_title("Preprocessing robustness overlap with P1")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
