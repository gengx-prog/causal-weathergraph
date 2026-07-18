#!/usr/bin/env python
"""Inspect local atmospheric data files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.data_io import inspect_data_root
from causal_weathergraph.utils import ensure_dir, save_json, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/home/vipuser/Data", help="Local data root to inspect.")
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    output_root = ensure_dir(args.output_root)
    summary = inspect_data_root(args.data_root)
    save_json(summary, output_root / "data_summary.json")
    print(f"Detected {summary['n_supported_files']} supported files under {args.data_root}")
    for record in summary["files"][:50]:
        dims = record.get("dims") or {}
        variables = record.get("variables") or record.get("arrays") or {}
        time_range = record.get("time_range", "")
        print(f"- {record['path']}")
        if dims:
            print(f"  dims: {dims}")
        if variables:
            print(f"  variables/arrays: {list(variables)[:20]}")
        if time_range:
            print(f"  time_range: {time_range}")
        if "error" in record:
            print(f"  error: {record['error']}")
    if summary["n_supported_files"] > 50:
        print("  ... output truncated; full summary saved to outputs/data_summary.json")
    print(f"Saved summary to {output_root / 'data_summary.json'}")


if __name__ == "__main__":
    main()
