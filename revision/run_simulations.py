"""CLI: python -m revision.run_simulations --output PATH --stage all.

Configurations are frozen before the first repetition. Interrupted runs resume
only when that exact frozen configuration and simulation code hashes match.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

from revision.simulations import (
    SimulationConfig, NULL_SCENARIOS, GRAPH_SCENARIOS, bh, bounded_mean_interval,
    candidate_edges, measured_call, package_versions, pcmci_pvalues,
    regression_pvalues, rng_for, score_discoveries, simulate_graph,
    simulate_independent, wilson_interval,
)


def source_hashes():
    files = [Path(__file__), Path(__file__).with_name("simulations.py"),
             Path(__file__).with_name("inference.py")]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def frozen_record(config):
    return {
        "config": asdict(config), "null_scenarios": list(NULL_SCENARIOS),
        "graph_scenarios": list(GRAPH_SCENARIOS), "source_sha256": source_hashes(),
        "definitions": {
            "candidate_universe": "All ordered cross-node pairs among six scored nodes, lags 1..3; 90 edge-lag hypotheses per regime.",
            "graph_truth": "Exact direct source-target-lag structural edges. Own-history links and common-driver links are conditioned/searched but excluded from scoring.",
            "shd": "Directed lag-specific adjacency Hamming distance FP+FN; reversing an edge contributes two errors.",
            "fdr": "Mean over independent repetitions of FP/max(discoveries,1), after BH over all 90 scored hypotheses per single regime.",
            "precision_empty": "Precision=1 when no edges selected; recall=0 when positive truth exists and no edges selected.",
            "null_calibration": "All and independently specified external-calendar regimes have zero cross-series population regression coefficients.",
            "selection_stress": "Outcome-selected mask uses current spatial mean >= its 80th percentile; conditioning may create population associations. This is structural selection stress, NOT calibration under a guaranteed regression null.",
            "omitted_driver": "Scored relative to full-DGP direct structural graph; latent confounding intentionally violates causal sufficiency. Not scored against marginal predictive graph.",
            "known_regime_switching": "Two externally scheduled, known regimes, 512-step blocks; all methods receive the same masks. This is an oracle-state benchmark, not learned-state discovery.",
            "comparators": "Own-history OLS/HAC64, all-observed-history OLS/HAC64, official Tigramite PCMCI with analytic ParCorr and mask_type=y. No Regime-PCMCI or CaStLe claims.",
            "exploratory_cluster": "Only when null_variance=cluster: full-calendar block CR1 covariance, t_(G-1), block128 main exploratory specification and 64/256 sensitivities; original OLS retained. This is a post-diagnostic extension, not the primary suite.",
            "same_calendar": "Regime masks select response rows without concatenating disjoint times; lagged predictors retain full-calendar alignment.",
            "graph_sample_alignment": "All graph estimators exclude the first 2*max_lag response rows to match PCMCI's default cutoff while retaining original lagged predictors.",
            "paired_driver_scenarios": "Observed and omitted common-driver settings use identical underlying innovations, coefficients and seeds for each repetition.",
            "pcmci_hypotheses": "PCMCI searches the complete observed set including self-links and any observed driver; BH is then applied to the same 90 scored cross-node hypotheses used for regression.",
            "runtime": "Regression pair time includes OLS and HAC64 jointly; it must not be interpreted as OLS-only runtime. PCMCI time includes the full search.",
            "memory": "10ms-sampled process resident memory, with baseline and absolute/incremental sampled peaks. Not isolated per-method maximum memory; allocator caching and sub-10ms spikes can affect values.",
            "ci": "All-null FDR equals P(any discovery): Wilson 95% interval. Graph-FDR: repetition-level mean FDP and t-based Monte Carlo interval clipped [0,1]; zero-variance intervals do not prove population FDR is zero.",
            "limits": "Synthetic six-node benchmark only; no ERA5 accuracy estimate, no nonlinear DGP, no intervention validation; known switching masks not learned.",
        },
    }


def freeze(output, config):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "frozen_config.json"
    current = frozen_record(config)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["frozen"] != current:
            raise RuntimeError("Configuration or source changed. Use a NEW output directory; never merge mismatched runs.")
    else:
        record = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "frozen": current, "environment": {"python": sys.version,
                  "platform": platform.platform(), "packages": package_versions(),
                  "blas_threads": 2}}
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        snapshot = output / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for filename in current["source_sha256"]:
            shutil.copy2(Path(__file__).with_name(filename), snapshot / filename)


def read_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_record(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_null(output, config):
    path = output / "null_repetitions.jsonl"
    completed = {(r["scenario"], r["repetition"]) for r in read_records(path)}
    edges = candidate_edges(config.n_nodes, config.max_lag)
    for s, scenario in enumerate(NULL_SCENARIOS):
        for repetition in range(config.null_repetitions):
            if (scenario, repetition) in completed:
                continue
            data, masks = simulate_independent(config, scenario, rng_for(config.seed, 0, s, repetition))
            rows = []
            for regime, mask in masks.items():
                if config.null_external_only and regime == "outcome_selected":
                    continue
                (pvalues, diagnostics), timing = measured_call(regression_pvalues, data, mask, edges, config,
                                                               variance_method=config.null_variance)
                for method, p in pvalues.items():
                    discoveries, _ = bh(p, config.alpha)
                    row = {"regime": regime, "method": f"own_history_{method}",
                           "estimand": "selection_stress" if regime == "outcome_selected" else "regression_null_calibration",
                           "n_tests": len(edges), "raw_rejection_fraction": float(np.mean(p <= config.alpha)),
                           **score_discoveries(discoveries, set(), edges), **diagnostics,
                           "shared_inference_timing": timing}
                    rows.append(row)
            append_record(path, {"scenario": scenario, "repetition": repetition, "rows": rows})
            if repetition % 10 == 0 or repetition + 1 == config.null_repetitions:
                print(f"NULL {scenario}: {repetition + 1}/{config.null_repetitions}", flush=True)


def run_graph(output, config):
    path = output / "graph_repetitions.jsonl"
    completed = {(r["scenario"], r["repetition"]) for r in read_records(path)}
    edges = candidate_edges(config.n_nodes, config.max_lag)
    for s, scenario in enumerate(GRAPH_SCENARIOS):
        for repetition in range(config.graph_repetitions):
            if (scenario, repetition) in completed:
                continue
            seed_scenario = GRAPH_SCENARIOS.index("common_driver_observed") if scenario == "common_driver_omitted" else s
            data, masks, truths, radii = simulate_graph(config, scenario, rng_for(config.seed, 1, seed_scenario, repetition))
            rows = []
            for state, (regime, mask) in enumerate(masks.items()):
                truth = truths[state]
                for conditioning in ("own_history", "all_observed_history"):
                    (pvalues, diagnostics), timing = measured_call(regression_pvalues, data, mask, edges, config, conditioning)
                    for method, p in pvalues.items():
                        discovered, _ = bh(p, config.alpha)
                        rows.append({"regime": regime, "method": f"{conditioning}_{method}",
                                     "n_tests": len(edges), "n_true_edges": len(truth),
                                     **score_discoveries(discovered, truth, edges), **diagnostics,
                                     "shared_regression_pair_timing": timing})
                (pvalues, diagnostics), timing = measured_call(pcmci_pvalues, data, mask, edges, config)
                discovered, _ = bh(pvalues, config.alpha)
                rows.append({"regime": regime, "method": "pcmci_parcorr", "n_tests": len(edges),
                             "n_true_edges": len(truth), **score_discoveries(discovered, truth, edges),
                             **diagnostics, "timing": timing})
            append_record(path, {"scenario": scenario, "repetition": repetition, "rows": rows,
                                 "companion_radii": radii, "truth": [sorted(t) for t in truths]})
            if repetition % 5 == 0 or repetition + 1 == config.graph_repetitions:
                print(f"GRAPH {scenario}: {repetition + 1}/{config.graph_repetitions}", flush=True)


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    status = {}
    for family in ("null", "graph"):
        records = read_records(output / f"{family}_repetitions.jsonl")
        groups = {}
        flat = []
        for record in records:
            for row in record["rows"]:
                flat_row = {"scenario": record["scenario"], "repetition": record["repetition"],
                            **{k: v for k, v in row.items() if not isinstance(v, dict)}}
                flat_row.update(row.get("timing", row.get("shared_inference_timing", row.get("shared_regression_pair_timing", {}))))
                flat.append(flat_row)
                groups.setdefault((record["scenario"], row["regime"], row["method"]), []).append(flat_row)
        summaries = []
        for (scenario, regime, method), rows in groups.items():
            mean, lo, hi, se = bounded_mean_interval([r["fdp"] for r in rows])
            if family == "null":
                lo, hi = wilson_interval(sum(r["any_false_discovery"] for r in rows), len(rows))
            summary = {"scenario": scenario, "regime": regime, "method": method,
                       "repetitions": len(rows), "empirical_fdr": mean,
                       "fdr_mc_ci95_low": lo, "fdr_mc_ci95_high": hi, "fdr_mc_se": se,
                       "total_invalid_tests": sum(r.get("invalid_tests", 0) for r in rows)}
            if family == "null":
                summary["estimand"] = rows[0]["estimand"]
            for metric in ("precision", "recall", "f1", "shd", "discoveries", "raw_rejection_fraction",
                           "wall_seconds", "sampled_peak_rss_mib", "sampled_incremental_rss_mib"):
                values = [r[metric] for r in rows if metric in r and np.isfinite(r[metric])]
                if values:
                    summary[f"mean_{metric}"] = float(np.mean(values))
                    summary[f"sd_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            summaries.append(summary)
        write_csv(output / f"{family}_results.csv", flat)
        write_csv(output / f"{family}_summary.csv", summaries)
        status[family] = {"completed_repetitions": len(records), "summary_rows": len(summaries)}
    (output / "run_status.json").write_text(json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(), **status}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "null", "graph", "summarize"), default="all")
    parser.add_argument("--null-repetitions", type=int, default=200)
    parser.add_argument("--graph-repetitions", type=int, default=50)
    parser.add_argument("--n-null", type=int, default=2048)
    parser.add_argument("--n-graph", type=int, default=4096)
    parser.add_argument("--external-null-only", action="store_true")
    parser.add_argument("--null-variance", choices=("hac", "cluster"), default="hac")
    parser.add_argument("--post-diagnostic", action="store_true")
    args = parser.parse_args()
    config = SimulationConfig(null_repetitions=args.null_repetitions, graph_repetitions=args.graph_repetitions,
                              n_null=args.n_null, n_graph=args.n_graph,
                              null_external_only=args.external_null_only, null_variance=args.null_variance,
                              protocol_role="post_diagnostic_exploratory" if args.post_diagnostic else "primary")
    if args.null_variance == "cluster" and not args.post_diagnostic:
        parser.error("The cluster addition must be explicitly marked --post-diagnostic")
    freeze(args.output, config)
    try:
        if args.stage in ("all", "null"):
            run_null(args.output, config)
        if args.stage in ("all", "graph"):
            run_graph(args.output, config)
    finally:
        summarize(args.output)


if __name__ == "__main__":
    main()
