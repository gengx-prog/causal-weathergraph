"""Joint source-lag window deletion gains, evaluated separately in each held-out segment.

Uses the 36 frozen region-variable pairs and the frozen-training predictive gain
of run_path_diagnostics.py, but scores the held-out loss differences in
2019-01-01..2023-01-10, 2023-01-11..2025-12-31 and their union. Coefficients are
fitted on 1979-2018 only, so each segment is an out-of-sample evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.run_path_diagnostics import joint_lag_design, prediction_gain  # noqa: E402

SEGMENTS = {"2019_2023": ("2019-01-01", "2023-01-11"), "2023_2025": ("2023-01-11", "2026-01-01"), "2019_2025": ("2019-01-01", "2026-01-01")}
WINDOWS = [(1, 2, 3), (4, 5, 6, 7, 8), (9, 10, 11, 12)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True, help="frozen_joint_lag_pairs.csv of the path-diagnostics run")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.input, allow_pickle=True) as z:
        data, names, timestamps = z["data"], [str(v) for v in z["variable_names"]], pd.to_datetime(z["timestamps"])
    train_time = np.asarray(timestamps.year <= 2018)
    pairs = pd.read_csv(args.pairs)
    rows = []
    for pair in pairs.itertuples():
        t, y, own, sources = joint_lag_design(data, names, int(pair.source_region), int(pair.target_region), pair.source_var, pair.target_var)
        full = np.column_stack((own, sources))
        for segment, (lo, hi) in SEGMENTS.items():
            evaluate = np.asarray((timestamps >= lo) & (timestamps < hi))[t]
            for window in WINDOWS:
                restricted = np.column_stack((own, np.delete(sources, [lag - 1 for lag in window], axis=1)))
                gain = prediction_gain(y, restricted, full, train_time[t], evaluate)
                rows.append({"pair_id": pair.pair_id, "edge_type": pair.edge_type, "segment": segment,
                             "window": f"{min(window)}-{max(window)}", "mse_gain": gain["mse_gain"],
                             "heldout_delta_r2": gain["heldout_delta_r2"], "evaluation_n": gain["evaluation_n"]})
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "joint_window_by_segment.csv", index=False)
    summary = frame.groupby(["segment", "window"], as_index=False).agg(
        pairs=("pair_id", "size"), evaluation_n=("evaluation_n", "first"),
        positive_predictive_gain=("mse_gain", lambda v: int((v > 0).sum())), median_heldout_delta_r2=("heldout_delta_r2", "median"))
    summary.to_csv(args.output / "joint_window_by_segment_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
