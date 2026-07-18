#!/usr/bin/env python
"""Run optional small-scale PCMCI validation for top stable edges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.data_io import load_dataset_npz
from causal_weathergraph.pcmci_validation import run_pcmci_validation, select_top_stable_edges, unavailable_outputs
from causal_weathergraph.utils import ensure_dir, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--tau_max", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--regime", default="all")
    parser.add_argument("--start_year", type=int, default=2019)
    parser.add_argument("--max_conds_dim", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    output_root = Path(args.output_root)
    stable_path = output_root / "causal_edges" / "stable_edges.csv"
    if not stable_path.exists():
        raise FileNotFoundError(f"Missing {stable_path}. Run bootstrap stability first.")
    stable = pd.read_csv(stable_path)
    selected = select_top_stable_edges(stable, top_n=args.top_n, regime=args.regime)
    if selected.empty:
        summary, edge_df = unavailable_outputs(selected, "no_selected_edges")
    else:
        dataset = load_dataset_npz(output_root / "processed" / "region_series.npz")
        summary, edge_df = run_pcmci_validation(
            dataset,
            selected,
            tau_max=args.tau_max,
            alpha=args.alpha,
            start_year=args.start_year,
            max_conds_dim=args.max_conds_dim,
        )

    report_dir = ensure_dir(output_root / "reports")
    edge_dir = ensure_dir(output_root / "causal_edges")
    summary.to_csv(report_dir / "pcmci_validation_summary.csv", index=False)
    edge_df.to_csv(edge_dir / "pcmci_validated_edges.csv", index=False)
    print(f"Saved PCMCI validation outputs to {report_dir / 'pcmci_validation_summary.csv'}", flush=True)
    if "status" in summary.columns and (summary["status"] == "tigramite_unavailable").any():
        print("Tigramite is not installed; wrote graceful skip outputs.", flush=True)


if __name__ == "__main__":
    main()
