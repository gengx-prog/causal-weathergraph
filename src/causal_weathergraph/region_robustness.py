"""Utilities for region aggregation robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidate_edges import build_candidate_edges
from .causal.fast_granger import run_fast_granger_discovery
from .causal.graph_utils import edge_set
from .data_io import AtmosphericDataset, load_dataset_npz
from .metrics import causal_chain_summary, regime_comparison
from .regions import aggregate_regions, save_region_outputs
from .regimes import define_regimes
from .utils import ensure_dir, get_logger

logger = get_logger("region_robustness")


@dataclass(frozen=True)
class RegionSetting:
    """One region aggregation robustness setting."""

    name: str
    method: str
    target_regions: int


SETTINGS = {
    "R32": RegionSetting("R32", "latlon_bins", 32),
    "R64": RegionSetting("R64", "latlon_bins", 64),
    "R128": RegionSetting("R128", "latlon_bins", 128),
    "K64": RegionSetting("K64", "kmeans_sphere", 64),
    "K128": RegionSetting("K128", "kmeans_sphere", 128),
}


def region_dataset_for_setting(
    processed: AtmosphericDataset,
    setting: RegionSetting,
    output_root: Path,
    random_seed: int = 42,
    reuse_r64: bool = True,
) -> tuple[AtmosphericDataset, pd.DataFrame]:
    """Create or load region-level data for a region robustness setting."""
    setting_root = output_root / "robustness" / "region" / setting.name
    region_path = setting_root / "processed" / "region_series.npz"
    region_meta_path = setting_root / "regions" / "region_metadata.csv"
    if region_path.exists() and region_meta_path.exists():
        return load_dataset_npz(region_path), pd.read_csv(region_meta_path)
    if setting.name == "R64" and reuse_r64:
        return load_dataset_npz(output_root / "processed" / "region_series.npz"), pd.read_csv(
            output_root / "regions" / "region_metadata.csv"
        )
    aggregation = aggregate_regions(
        processed,
        {
            "method": setting.method,
            "n_regions": setting.target_regions,
            "random_seed": int(random_seed),
        },
    )
    save_region_outputs(aggregation, setting_root)
    return aggregation.dataset, aggregation.metadata


def run_region_granger(
    dataset: AtmosphericDataset,
    config: dict[str, Any],
    setting_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Granger discovery for one region robustness dataset."""
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
    edges["region_setting"] = setting_name
    significant = edges[edges["significant"].astype(bool)].copy()
    return edges, significant


def summarize_region_robustness(
    all_edges_by_setting: dict[str, pd.DataFrame],
    significant_by_setting: dict[str, pd.DataFrame],
    metadata_by_setting: dict[str, pd.DataFrame],
    setting_order: list[str],
) -> pd.DataFrame:
    """Summarize core metrics by region setting."""
    rows = []
    for name in setting_order:
        setting = SETTINGS[name]
        all_edges = all_edges_by_setting[name]
        sig = significant_by_setting[name]
        metadata = metadata_by_setting[name]
        wh = sig[sig["edge_type"] == "wind_to_humidity"]
        hc = sig[sig["edge_type"] == "humidity_to_cloud_cover"]
        wc = sig[sig["edge_type"] == "wind_to_cloud_cover"]
        chains = causal_chain_summary(sig)
        rows.append(
            {
                "setting": name,
                "method": setting.method,
                "target_regions": int(setting.target_regions),
                "actual_regions": int(len(metadata)),
                "tested_edges": int(len(all_edges)),
                "significant_edges": int(len(sig)),
                "significant_ratio": float(len(sig) / len(all_edges)) if len(all_edges) else 0.0,
                "WH_edges": int(len(wh)),
                "HC_edges": int(len(hc)),
                "WC_edges": int(len(wc)),
                "WHC_chains": int(chains["n_chains"].sum()) if not chains.empty else 0,
                "mean_WH_lag": float(wh["lag"].mean()) if len(wh) else np.nan,
                "mean_HC_lag": float(hc["lag"].mean()) if len(hc) else np.nan,
                "DJF_WH_stronger_than_JJA": bool(djf_wh_stronger(sig)),
                "midlatitude_strongest": bool(midlatitude_strongest(sig, metadata)),
                "humidity_regime_jaccard": pair_jaccard(sig, "high_humidity", "normal_humidity"),
                "cloud_regime_jaccard": pair_jaccard(sig, "high_cloud", "normal_cloud"),
            }
        )
    return pd.DataFrame.from_records(rows)


def region_macro_overlap(
    summary: pd.DataFrame,
    baseline_setting: str = "R64",
) -> pd.DataFrame:
    """Return macro-level robustness flags relative to the baseline setting."""
    rows = []
    for row in summary.itertuples(index=False):
        wh_presence = bool(row.WH_edges > 0)
        hc_presence = bool(row.HC_edges > 0)
        whc_presence = bool(row.WHC_chains > 0)
        robust = wh_presence and hc_presence and whc_presence
        rows.append(
            {
                "setting": row.setting,
                "WH_presence": "yes" if wh_presence else "no",
                "HC_presence": "yes" if hc_presence else "no",
                "WHC_presence": "yes" if whc_presence else "no",
                "DJF_WH_stronger_than_JJA": "yes" if bool(row.DJF_WH_stronger_than_JJA) else "no",
                "midlatitude_strongest": "yes" if bool(row.midlatitude_strongest) else "no",
                "edge_jaccard_with_R64": np.nan if row.setting != baseline_setting else 1.0,
                "conclusion": "core_pathway_recovered" if robust else "core_pathway_weakened",
            }
        )
    return pd.DataFrame.from_records(rows)


def djf_wh_stronger(sig: pd.DataFrame) -> bool:
    """Return whether DJF has higher WH fraction among significant rows than JJA."""
    fractions = {}
    for regime in ["DJF", "JJA"]:
        group = sig[sig["regime"] == regime]
        fractions[regime] = float((group["edge_type"] == "wind_to_humidity").mean()) if len(group) else 0.0
    return fractions["DJF"] > fractions["JJA"]


def midlatitude_strongest(sig: pd.DataFrame, metadata: pd.DataFrame) -> bool:
    """Return whether midlatitude has the largest WHC chain count."""
    if sig.empty or "latitude_band" not in metadata.columns:
        return False
    band_lookup = {int(row.region): str(row.latitude_band).lower() for row in metadata.itertuples(index=False)}
    counts = {"tropical": 0, "midlatitude": 0, "polar": 0}
    for _, group in sig.groupby("regime"):
        wh = group[group["edge_type"] == "wind_to_humidity"]
        hc = group[group["edge_type"] == "humidity_to_cloud_cover"]
        hc_by_source = {int(source): df for source, df in hc.groupby("source_region")}
        for left in wh.itertuples(index=False):
            downstream = hc_by_source.get(int(left.target_region))
            if downstream is None:
                continue
            band = band_lookup.get(int(left.target_region), "unknown")
            if band in counts:
                counts[band] += len(downstream)
    return counts["midlatitude"] >= max(counts.values()) if any(counts.values()) else False


def pair_jaccard(sig: pd.DataFrame, a: str, b: str) -> float:
    """Jaccard distance between two regime edge sets."""
    if sig.empty:
        return np.nan
    set_a = edge_set(sig[sig["regime"] == a])
    set_b = edge_set(sig[sig["regime"] == b])
    union = set_a | set_b
    return float(1.0 - len(set_a & set_b) / len(union)) if union else np.nan


def plot_region_summary(summary: pd.DataFrame, path: Path) -> None:
    """Save a small bar chart for region robustness."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = summary["setting"].tolist()
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width, summary["WH_edges"], width, label="WH", color="#2A6FBB")
    ax.bar(x, summary["HC_edges"], width, label="HC", color="#3B8C6E")
    ax.bar(x + width, summary["WC_edges"], width, label="WC", color="#C56A43")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Significant edge-lag-regime rows")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
