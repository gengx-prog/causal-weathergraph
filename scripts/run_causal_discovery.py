#!/usr/bin/env python
"""Run regime-aware Granger-style causal discovery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.candidate_edges import build_candidate_edges
from causal_weathergraph.causal.bootstrap import add_bootstrap_stability
from causal_weathergraph.causal.granger import run_granger_discovery
from causal_weathergraph.causal.pcmci_optional import run_pcmci_optional
from causal_weathergraph.data_io import load_dataset_npz
from causal_weathergraph.regimes import define_regimes
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def run_causal(config: dict, output_root: str | Path) -> None:
    output_root = Path(output_root)
    region_path = output_root / "processed" / "region_series.npz"
    if not region_path.exists():
        raise FileNotFoundError(
            f"Missing {region_path}. Run python scripts/run_preprocess.py --config configs/default.yaml first."
        )
    dataset = load_dataset_npz(region_path)
    causal_cfg = config.get("causal", {})
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
    out_dir = ensure_dir(output_root / "causal_edges")
    candidate_columns = [
        "source_region",
        "target_region",
        "source_var",
        "target_var",
        "edge_type",
        "distance_km",
    ]
    pd.DataFrame([edge.to_dict() for edge in candidates], columns=candidate_columns).to_csv(
        out_dir / "candidate_edges.csv", index=False
    )

    method = str(causal_cfg.get("method", "granger")).lower()
    if method.startswith("pcmci"):
        pcmci_result = run_pcmci_optional(
            dataset.data,
            dataset.variable_names,
            max_lag=int(causal_cfg.get("max_lag", 12)),
            config=causal_cfg,
        )
        if pcmci_result is not None:
            print("PCMCI finished. The default CSV edge table is still produced with the Granger backend.")

    edges = run_granger_discovery(
        data=dataset.data,
        variable_names=dataset.variable_names,
        candidates=candidates,
        regimes=regimes,
        max_lag=int(causal_cfg.get("max_lag", 12)),
        alpha=float(causal_cfg.get("alpha", 0.05)),
        controls=causal_cfg.get("controls", {}),
        fdr_method=str(causal_cfg.get("fdr_method", "benjamini_hochberg")),
        region_lat=dataset.lat,
        region_lon=dataset.lon,
    )

    bootstrap_cfg = {**causal_cfg, **causal_cfg.get("bootstrap", {})}
    if bool(causal_cfg.get("bootstrap", {}).get("enabled", False)):
        edges = add_bootstrap_stability(
            edges,
            dataset.data,
            dataset.variable_names,
            candidates,
            regimes,
            bootstrap_cfg,
            region_lat=dataset.lat,
            region_lon=dataset.lon,
        )

    significant = edges[edges["significant"].astype(bool)].copy() if "significant" in edges else edges.iloc[0:0].copy()
    edges.to_csv(out_dir / "all_edges.csv", index=False)
    significant.to_csv(out_dir / "significant_edges.csv", index=False)
    print(f"Saved {len(edges)} tested lagged edges to {out_dir / 'all_edges.csv'}")
    print(f"Saved {len(significant)} significant edges to {out_dir / 'significant_edges.csv'}")


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    run_causal(config, args.output_root)


if __name__ == "__main__":
    main()
