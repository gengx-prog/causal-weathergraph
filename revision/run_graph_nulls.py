"""Run E2 without modifying legacy inputs. Execute from the repository root."""
from __future__ import annotations

import os
# Set before importing numpy/scipy; two threads is the maximum for this run.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from revision.graph_nulls import (
    DISTANCE_CUTS_KM, EDGE_COLUMNS, candidate_null_draws, discovery_summary,
    distance_matrix, graph_metrics, key, legacy_candidates, recalibrate_subset,
    rewiring_ensemble, selected_keys, symmetric_candidates,
)
from revision.inference import run_graph_discovery


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str,
                               allow_nan=False), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.input.resolve()
    manifest_path = out / "manifest.json"
    current_identity = {"input_sha256": sha256(source), "candidate_replicates": args.candidate_replicates,
                        "topology_replicates": args.topology_replicates,
                        "max_time": args.max_time, "seed": args.seed}
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity") != current_identity:
            raise ValueError("Output directory belongs to a different frozen input/settings run")
    with np.load(source, allow_pickle=True) as npz:
        data = np.asarray(npz["data"], dtype=np.float64)
        names = [str(x) for x in npz["variable_names"]]
        lat, lon = np.asarray(npz["lat"]), np.asarray(npz["lon"])
        timestamps = pd.to_datetime(npz["timestamps"])
    if args.max_time:
        data, timestamps = data[:args.max_time], timestamps[:args.max_time]
    timings = dict(previous.get("timings_seconds", {}))
    manifest = {
        "identity": current_identity, "experiment": "E2 symmetric candidates and graph reference distributions",
        "started_utc": previous.get("started_utc", datetime.now(timezone.utc).isoformat()),
        "source_path": str(source), "source_read_only": True,
        "source_size_bytes": source.stat().st_size, "data_shape": list(data.shape),
        "time_first": str(timestamps[0]), "time_last": str(timestamps[-1]),
        "variable_names": names, "max_lag": 3, "hac_bandwidth": 64,
        "alpha": .05, "primary_multiplicity": "BH globally across candidate x lag tests within each declared family",
        "sensitivity_multiplicity": "BH separately by regime and target variable",
        "distance_cuts_km": list(DISTANCE_CUTS_KM), "threads_max": 2,
        "legacy_preprocessing_warning": "The legacy region array used full-period climatology and scaling. These E2 analyses are retrospective full-sample structural sensitivities, not independent temporal validation.",
        "period_comparison": "1979-2001 versus 2002-2025 descriptive stability; not an untouched holdout",
        "status": "running", "timings_seconds": timings,
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                        [ROOT / "revision/graph_nulls.py", ROOT / "revision/run_graph_nulls.py",
                         ROOT / "revision/inference.py"]},
    }
    save_json(manifest_path, manifest)
    distances = distance_matrix(lat, lon)
    families = {"legacy858": legacy_candidates(names, lat, lon),
                "symmetric_core": symmetric_candidates(names, lat, lon, k=2)}
    draws_by_family = {}
    pool = {}
    candidate_meta = {}
    for family_no, (family, edges) in enumerate(families.items()):
        pd.DataFrame([e.to_dict() for e in edges]).to_csv(out / f"candidates_{family}.csv", index=False)
        draws, meta = candidate_null_draws(edges, distances, args.candidate_replicates,
                                           seed=args.seed + family_no)
        draws_by_family[family] = draws
        candidate_meta[family] = meta
        for subset in [edges, *draws]:
            for edge in subset:
                pool[key(edge)] = edge
        index_rows = [{"replicate": rep, **e.to_dict()} for rep, draw in enumerate(draws) for e in draw]
        pd.DataFrame(index_rows).to_csv(out / f"candidate_null_draws_{family}.csv.gz", index=False)
    save_json(out / "candidate_null_design.json", candidate_meta)
    pooled_candidates = [pool[k] for k in sorted(pool)]
    manifest["candidate_counts"] = {k: len(v) for k, v in families.items()}
    manifest["deduplicated_refit_pool_size"] = len(pool)
    save_json(manifest_path, manifest)
    print(f"E2 data {data.shape}; candidate families {manifest['candidate_counts']}; dedup pool {len(pool)}", flush=True)

    def discover(label: str, tensor: np.ndarray, candidates: list,
                 multivar: bool = False) -> pd.DataFrame:
        path = out / f"discovery_{label}.csv.gz"
        if path.exists():
            print(f"Reuse completed {label}", flush=True)
            return pd.read_csv(path)
        phase = time.perf_counter()
        print(f"Starting {label}: n={len(tensor)}, candidates={len(candidates)}, multivar={multivar}", flush=True)
        result = run_graph_discovery(
            tensor, names, candidates, {"all": np.ones(len(tensor), dtype=bool)},
            max_lag=3,
            controls={"include_target_own_lags": True, "include_target_region_all_vars": multivar},
            bandwidths=(64,), progress=lambda *message: print(f"{label}: {message}", flush=True))
        result.to_csv(path, index=False)
        timings[label] = time.perf_counter() - phase
        save_json(manifest_path, manifest)
        print(f"Completed {label} in {timings[label]:.1f}s", flush=True)
        return result

    pooled = discover("deduplicated_pool_own", data, pooled_candidates)
    summaries = []
    family_results = {}
    candidate_null_summaries = []
    for family, edges in families.items():
        actual = recalibrate_subset(pooled, edges)
        actual.to_csv(out / f"discovery_{family}_own.csv.gz", index=False)
        family_results[family] = actual
        summaries.append(discovery_summary(actual, len(lat), family + "_own"))
        for rep, draw in enumerate(draws_by_family[family]):
            fitted = recalibrate_subset(pooled, draw)
            summary = discovery_summary(fitted, len(lat), family)
            summary.insert(0, "replicate", rep)
            candidate_null_summaries.append(summary)
    pd.concat(candidate_null_summaries, ignore_index=True).to_csv(out / "candidate_null_summary.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(out / "discovery_summary.csv", index=False)

    # Aggregation uses 100 independent candidate draws. No no-effect/FDR claim.
    null_summary = pd.concat(candidate_null_summaries, ignore_index=True)
    enrichment = []
    for family, actual in family_results.items():
        actual_summary = discovery_summary(actual, len(lat), family)
        for row in actual_summary.to_dict("records"):
            ref = null_summary[(null_summary.family == family) &
                               (null_summary.correction == row["correction"]) &
                               (null_summary.edge_type == row["edge_type"])]
            for metric in ["significant_fraction", "whc_lag_resolved_chains", "whc_unique_edge_chains",
                           "whc_unique_endpoint_pairs", "whc_closed_with_wc_chains"]:
                if metric not in row or not np.isfinite(row[metric]):
                    continue
                values = ref[metric].to_numpy(dtype=float)
                observed = float(row[metric])
                enrichment.append({"family": family, "correction": row["correction"],
                                   "edge_type": row["edge_type"], "metric": metric,
                                   "observed": observed, "null_mean": values.mean(),
                                   "null_q025": np.quantile(values, .025), "null_q975": np.quantile(values, .975),
                                   "observed_over_null_mean": observed / values.mean() if values.mean() else np.nan,
                                   "monte_carlo_upper_tail": (1 + np.sum(values >= observed)) / (len(values) + 1),
                                   "replicates": len(values)})
    pd.DataFrame(enrichment).to_csv(out / "candidate_null_enrichment.csv", index=False)

    if not args.skip_multivar:
        multivar = discover("symmetric_core_multivar", data, families["symmetric_core"], True)
        summaries.append(discovery_summary(multivar, len(lat), "symmetric_core_multivar"))
        pd.concat(summaries, ignore_index=True).to_csv(out / "discovery_summary.csv", index=False)

    topology_dir = out / "topology_nulls"
    topology_dir.mkdir(exist_ok=True)
    topology_index = []

    def topology(label: str, edges: set, support: set | None, reference: set,
                 seed: int, interpretation: str) -> None:
        diagnostic_path = topology_dir / f"{label}.json"
        if diagnostic_path.exists():
            print(f"Reuse completed topology {label}", flush=True)
            diag = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        else:
            phase = time.perf_counter()
            frame, diag, targets = rewiring_ensemble(
                edges, distances, support=support, reference=reference,
                n_replicates=args.topology_replicates, seed=seed)
            diag["comparison_interpretation"] = interpretation
            frame.to_csv(topology_dir / f"{label}.csv", index=False)
            np.savez_compressed(topology_dir / f"{label}_snapshots.npz", target_regions=targets)
            save_json(diagnostic_path, diag)
            timings[label] = time.perf_counter() - phase
            print(f"Topology {label}: accepted {diag['accepted_swaps']}/{diag['attempted_swaps']}, distinct {diag['distinct_graphs']}", flush=True)
        topology_index.append({"label": label, "accepted_swaps": diag["accepted_swaps"],
                               "acceptance_fraction": diag["acceptance_fraction"],
                               "distinct_graphs": diag["distinct_graphs"],
                               "mean_changed_fraction": diag["mean_changed_fraction"],
                               "support": diag["support"]})

    seed_offset = 20
    for family, actual in family_results.items():
        for correction in ["q_ols_global", "q_hac64_global"]:
            selected = selected_keys(actual, correction)
            for support_label, support in [("original_support", {key(e) for e in families[family]}),
                                            ("distance_support", None)]:
                label = f"{family}_{correction}_{support_label}"
                topology(label, selected, support, selected, args.seed + seed_offset,
                         "Topology motif comparison; Jaccard with starting graph is a MIXING diagnostic, not evidence of stability.")
                seed_offset += 1

    if not args.skip_periods and timestamps[-1].year >= 2002:
        split = np.asarray(timestamps.year <= 2001)
        if split.sum() >= 100 and (~split).sum() >= 100:
            early = discover("symmetric_1979_2001_own", data[split], families["symmetric_core"])
            late = discover("symmetric_2002_2025_own", data[~split], families["symmetric_core"])
            for label, frame in [("symmetric_1979_2001_own", early), ("symmetric_2002_2025_own", late)]:
                summaries.append(discovery_summary(frame, len(lat), label))
            pd.concat(summaries, ignore_index=True).to_csv(out / "discovery_summary.csv", index=False)
            for correction in ["q_ols_global", "q_hac64_global"]:
                first, second = selected_keys(early, correction), selected_keys(late, correction)
                for support_label, support in [("original_support", {key(e) for e in families["symmetric_core"]}),
                                                ("distance_support", None)]:
                    label = f"period_overlap_{correction}_{support_label}"
                    topology(label, second, support, first, args.seed + seed_offset,
                             "Observed early-late graph overlap vs late graph rewires preserving its degrees, density and distance-bin margins. Shared full-period preprocessing; not independent validation.")
                    seed_offset += 1
    pd.DataFrame(topology_index).to_csv(out / "topology_null_diagnostics.csv", index=False)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["total_wall_seconds_this_invocation"] = time.perf_counter() - start
    save_json(manifest_path, manifest)
    print(f"E2 COMPLETE {manifest['total_wall_seconds_this_invocation']:.1f}s; outputs {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-replicates", type=int, default=100)
    parser.add_argument("--topology-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026092902)
    parser.add_argument("--max-time", type=int)
    parser.add_argument("--skip-multivar", action="store_true")
    parser.add_argument("--skip-periods", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
