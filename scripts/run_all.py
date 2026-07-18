#!/usr/bin/env python
"""Run the complete Causal WeatherGraph pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.utils import load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config_path = resolve_path(args.config, REPO_ROOT).resolve()
    config = load_config(config_path)
    data_root = config.get("data", {}).get("root", "/home/vipuser/Data")
    output_root = str(Path(args.output_root).resolve())
    steps = [
        ["inspect_data.py", "--data_root", data_root, "--output_root", output_root],
        ["run_preprocess.py", "--config", str(config_path), "--output_root", output_root],
        ["run_causal_discovery.py", "--config", str(config_path), "--output_root", output_root],
        ["run_regime_analysis.py", "--config", str(config_path), "--output_root", output_root],
    ]
    for script, *step_args in steps:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / script), *step_args]
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.check_call(cmd)
    print(f"\nPipeline complete. Outputs are in {output_root}", flush=True)


if __name__ == "__main__":
    main()
