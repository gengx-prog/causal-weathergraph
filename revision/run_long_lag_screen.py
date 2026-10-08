"""Long-lag version of the full-record screen (max lag 12, all-time regime).

Re-creates the long-lag experiment of the original submission (its Table 14) with
the revision's screening code: the 858 candidate edges are tested at lags 1-12 with
own-history nested OLS in the 'all' regime, BH within target-variable families,
and the lag distribution of significant edge-lags is summarized by edge type.
Running it on the legacy full-record series reproduces the original table; running
it on the re-acquired record gives the revised values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from causal_weathergraph.candidate_edges import build_candidate_edges  # noqa: E402
from causal_weathergraph.regimes import define_regimes  # noqa: E402
from revision.inference import run_graph_discovery  # noqa: E402
from revision.prepare_inputs import sha256  # noqa: E402

TYPES = [("wind", "humidity"), ("humidity", "cloud_cover"), ("wind", "cloud_cover"), ("humidity", "humidity")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-lag", type=int, default=12)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    config = yaml.safe_load((ROOT / "configs" / "local_full_period_fast.yaml").read_text())
    with np.load(args.input, allow_pickle=True) as z:
        data, names = z["data"], z["variable_names"].tolist()
        lat, lon, timestamps = z["lat"], z["lon"], pd.DatetimeIndex(z["timestamps"])
    candidates = build_candidate_edges(names, lat, lon, config["causal"], data)
    regimes = define_regimes(data, names, timestamps, config["regimes"])
    with threadpool_limits(limits=2):
        results = run_graph_discovery(data, names, candidates, {"all": regimes["all"]}, max_lag=args.max_lag,
                                      controls=config["causal"]["controls"], bandwidths=(64,))
    results.to_csv(args.output / "long_lag_tests.csv.gz", index=False)
    rows = []
    for source, target in TYPES:
        part = results[(results.source_var == source) & (results.target_var == target) & results.significant_ols_within]
        counts = part.lag.value_counts().reindex(range(1, args.max_lag + 1), fill_value=0)
        total = int(counts.sum())
        rows.append({"edge_type": f"{source}_to_{target}", "significant_edge_lags": total, "peak_lag": int(counts.idxmax()),
                     "mean_lag": float(part.lag.mean()), "median_lag": float(part.lag.median()),
                     "lag1_fraction": counts.loc[1] / total, "lag2_3_fraction": counts.loc[2:3].sum() / total,
                     "lag4_8_fraction": counts.loc[4:8].sum() / total, "lag9_12_fraction": counts.loc[9:12].sum() / total})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "long_lag_summary.csv", index=False)
    manifest = {"input": str(args.input), "input_sha256": sha256(args.input), "max_lag": args.max_lag, "regime": "all",
                "n_tests": int(len(results)), "significance": "own-history OLS, BH within target-variable family",
                "elapsed_seconds": round(time.perf_counter() - start, 1), "code_sha256": sha256(Path(__file__))}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
