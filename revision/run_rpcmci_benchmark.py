"""Direct official Regime-PCMCI comparison on two exclusive synthetic states.

Example: python -m revision.run_rpcmci_benchmark --output PATH --replicates 20
The seven overlapping ERA5 masks are NOT used as latent-state labels.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "2"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd

from revision.simulations import (
    SimulationConfig, bh, bounded_mean_interval, candidate_edges, measured_call,
    package_versions, pcmci_pvalues, regression_pvalues, rng_for,
    score_discoveries, simulate_graph, wilson_interval,
)


def align_two_states(gamma, true_states, valid):
    """Label alignment uses state agreement ONLY, solely after estimation."""
    predicted = np.argmax(gamma, axis=0)
    direct = float(np.mean(predicted[valid] == true_states[valid]))
    flipped = float(np.mean((1 - predicted[valid]) == true_states[valid]))
    swap = flipped > direct
    mapping = {0: 1, 1: 0} if swap else {0: 0, 1: 1}
    return mapping, max(direct, flipped), (1 - predicted if swap else predicted)


def reached_official_tolerance(history, n_samples, num_regimes=2):
    # Official q is zero-based: q>=5 means at least SIX completed iterations.
    tolerance = 2 * num_regimes * n_samples // 100
    return bool(history[-1] == 0 or (len(history) >= 6 and history[-1] <= tolerance))


def run_hidden(data, config, seed, iterations, anneals):
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.rpcmci import RPCMCI

    model = RPCMCI(DataFrame(data.copy()), cond_ind_test=ParCorr(significance="analytic"),
                   seed=seed, verbosity=-1)
    transitions = (len(data) - 1) // 512
    return model.run_rpcmci(num_regimes=2, max_transitions=transitions,
                            switch_thres=0.05, num_iterations=iterations,
                            max_anneal=anneals, tau_min=1, tau_max=config.max_lag,
                            pc_alpha=config.pcmci_pc_alpha, alpha_level=config.alpha,
                            n_jobs=1)


def run_hidden_recording_failure(*args):
    try:
        return run_hidden(*args), None
    except Exception:
        return None, traceback.format_exc()


def metric_rows(method, p_by_state, truths, edges, config, timing, diagnostics=None):
    rows = []
    for state, p in enumerate(p_by_state):
        discoveries, _ = bh(p, config.alpha)
        rows.append({"method": method, "state": state,
                     **score_discoveries(discoveries, truths[state], edges),
                     **(diagnostics or {}), "timing": timing})
    return rows


def combine_timings(timings):
    return {"wall_seconds": sum(t["wall_seconds"] for t in timings),
            "sampled_peak_rss_mib": max(t["sampled_peak_rss_mib"] for t in timings)}


def one_repetition(config, repetition, iterations, anneals, artifact_dir):
    # Family 2 is independent of the earlier primary graph simulations.
    data, masks, truths, radii = simulate_graph(config, "known_external_regime_switching",
                                              rng_for(config.seed, 2, 0, repetition))
    edges = candidate_edges(config.n_nodes, config.max_lag)
    valid = np.any(np.stack(list(masks.values())), axis=0)
    true_states = np.argmax(np.stack(list(masks.values())), axis=0)
    rows, arrays = [], {"true_states": true_states, "valid_rows": valid,
                        "truth_edge_lag": np.asarray([[edge in truth for edge in edges] for truth in truths]),
                        "edge_order": np.asarray(edges)}
    pooled, timing = measured_call(pcmci_pvalues, data, valid, edges, config)
    rows.extend(metric_rows("pooled_pcmci", [pooled[0], pooled[0]], truths, edges, config, timing))
    arrays["pooled_pcmci_p"] = pooled[0]

    for conditioning in ("own_history", "all_observed_history"):
        times, p_state = [], []
        for mask in masks.values():
            (p, _), timing = measured_call(regression_pvalues, data, mask, edges, config, conditioning)
            p_state.append(p)
            times.append(timing)
        for variance in ("ols", "hac64"):
            p = [state_p[variance] for state_p in p_state]
            name = f"oracle_{conditioning}_{variance}"
            rows.extend(metric_rows(name, p, truths, edges, config, combine_timings(times)))
            arrays[f"{name}_p"] = np.stack(p)

    p_state, times = [], []
    for mask in masks.values():
        (p, _), timing = measured_call(pcmci_pvalues, data, mask, edges, config)
        p_state.append(p)
        times.append(timing)
    rows.extend(metric_rows("oracle_pcmci", p_state, truths, edges, config, combine_timings(times)))
    arrays["oracle_pcmci_p"] = np.stack(p_state)

    (result, error), timing = measured_call(run_hidden_recording_failure, data, config,
                                            config.seed + 10000 + repetition * 10,
                                            iterations, anneals)
    if result is None:
        diagnostics = {"run_failed": True, "error_free_annealings": 0,
                       "met_official_change_tolerance": False, "state_accuracy": None,
                       "best_annealing_iterations": 0, "exception": error}
        p_state = [np.ones(len(edges)), np.ones(len(edges))]
        rows.extend(metric_rows("regime_pcmci_hidden", p_state, truths, edges, config, timing, diagnostics))
    else:
        gamma = result["regimes"]
        mapping, accuracy, labels = align_two_states(gamma, true_states, valid)
        best_history = result["diff_g_f"][1]
        tolerance = 2 * 2 * len(data) // 100
        converged = reached_official_tolerance(best_history, len(data))
        diagnostics = {"run_failed": False,
                       "error_free_annealings": int(result["error_free_annealings"]),
                       "met_official_change_tolerance": converged,
                       "state_accuracy": accuracy, "best_annealing_iterations": len(best_history),
                       "final_gamma_change": float(best_history[-1]), "official_change_tolerance": tolerance,
                       "estimated_to_true_state_mapping": mapping,
                       "soft_gamma_fraction": float(np.mean((gamma > 1e-8) & (gamma < 1 - 1e-8))),
                       "all_annealing_histories": result["diff_g_f"][0]}
        p_state = [None, None]
        for estimated, aligned in mapping.items():
            matrix = result["causal_results"][estimated]["p_matrix"]
            p_state[aligned] = np.asarray([matrix[s, t, lag] for s, t, lag in edges])
        rows.extend(metric_rows("regime_pcmci_hidden", p_state, truths, edges, config, timing, diagnostics))
        arrays["estimated_gamma"] = gamma
        arrays["aligned_state_labels"] = labels
    arrays["regime_pcmci_p_aligned"] = np.stack(p_state)
    np.savez_compressed(artifact_dir / f"repetition_{repetition:03d}.npz", **arrays)
    return {"repetition": repetition, "rows": rows, "spectral_radii": radii,
            "truth": [sorted(truth) for truth in truths]}


def summarize(output, repetitions, anneals):
    source = output / "repetitions.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()] if source.exists() else []
    flat, summaries = [], []
    for record in records:
        for row in record["rows"]:
            flat.append({"repetition": record["repetition"],
                         **{k: v for k, v in row.items() if not isinstance(v, (dict, list))}, **row["timing"]})
    frame = pd.DataFrame(flat)
    if len(frame):
        frame.to_csv(output / "state_metrics.csv", index=False, encoding="utf-8-sig")
        for method, rows in frame.groupby("method", sort=False):
            by_rep = rows.groupby("repetition").mean(numeric_only=True)
            summary = {"method": method, "completed_repetitions": len(by_rep), "planned_repetitions": repetitions}
            for metric in ("precision", "recall", "f1", "fdp", "shd", "wall_seconds", "sampled_peak_rss_mib"):
                values = by_rep[metric].to_numpy()
                summary[f"mean_{metric}"] = float(np.nanmean(values))
                summary[f"sd_{metric}"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else None
                if metric in ("precision", "recall", "f1", "fdp"):
                    _, lo, hi, se = bounded_mean_interval(values)
                    summary[f"{metric}_mc_ci95_low"] = lo
                    summary[f"{metric}_mc_ci95_high"] = hi
            if method == "regime_pcmci_hidden":
                first = rows.groupby("repetition").first()
                failures = int(first["run_failed"].sum())
                summary["run_failure_rate"] = failures / len(first)
                summary["run_failure_ci_low"], summary["run_failure_ci_high"] = wilson_interval(failures, len(first))
                summary["official_tolerance_rate"] = float(first["met_official_change_tolerance"].mean())
                summary["error_free_annealing_fraction"] = float(first["error_free_annealings"].sum() / (len(first) * anneals))
                summary["mean_state_accuracy_successes"] = float(first["state_accuracy"].mean())
                accuracies = first["state_accuracy"].dropna().to_numpy(dtype=float)
                if len(accuracies):
                    _, lo, hi, _ = bounded_mean_interval(accuracies)
                    summary["state_accuracy_mc_ci95_low"] = lo
                    summary["state_accuracy_mc_ci95_high"] = hi
                    summary["state_accuracy_below_075_fraction_successes"] = float(np.mean(accuracies < 0.75))
            summaries.append(summary)
        pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
        from scipy.stats import t as student_t
        by_rep = frame.groupby(["method", "repetition"]).mean(numeric_only=True)
        comparisons = []
        for reference in ("pooled_pcmci", "oracle_pcmci", "oracle_all_observed_history_ols"):
            for metric in ("f1", "shd"):
                comparison = pd.concat([by_rep.loc["regime_pcmci_hidden", metric],
                                        by_rep.loc[reference, metric]], axis=1, keys=["hidden", "reference"]).dropna()
                differences = comparison["hidden"] - comparison["reference"]
                mean = float(differences.mean())
                half = float(student_t.ppf(.975, len(differences)-1) * differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else float("nan")
                comparisons.append({"method": "regime_pcmci_hidden", "reference": reference,
                                    "metric": metric, "paired_repetitions": len(differences),
                                    "mean_method_minus_reference": mean,
                                    "mc_ci95_low": mean-half, "mc_ci95_high": mean+half})
        pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    (output / "run_status.json").write_text(json.dumps({"completed": len(records), "planned": repetitions,
                    "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--anneals", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.n_samples % 1024:
        parser.error("Use a multiple of 1024 so both true states have equal exposure")
    config = replace(SimulationConfig(), n_graph=args.n_samples)
    from tigramite import rpcmci
    files = [Path(__file__), Path(__file__).with_name("simulations.py"),
             Path(__file__).with_name("inference.py"), Path(rpcmci.__file__)]
    frozen = {"simulation_config": asdict(config), "replicates": args.replicates,
              "num_regimes": 2, "max_transitions": (args.n_samples - 1) // 512,
              "num_iterations": args.iterations, "max_anneal": args.anneals,
              "parallel_repetition_workers": args.workers,
              "n_jobs": 1, "blas_threads": 2, "switch_thres": 0.05,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "scope": "Two exclusive, externally generated true states with hidden assignment; known number of states and transition cap supplied from predeclared DGP rule. Parameters computationally budgeted, not fully tuned.",
              "evaluation": "90 identical scored cross-node edge-lag hypotheses per state, BH .05 separately per state. Macro average over true states, then average over independent repetitions. SHD=FP+FN at exact direction/lag.",
              "alignment": "Choose label permutation maximizing state accuracy on valid rows solely for scoring, not for fitting or hyperparameter choice.",
              "failure_policy": "Retain every attempt. Exceptions/all annealing failures get empty graphs (F1=0, SHD=6 per state), separate failure and state-accuracy reporting. Reaching iteration budget does NOT equal convergence.",
              "oracle_warning": "Oracle methods receive true masks and are upper-bound/comparator conditions, not a like-for-like unsupervised competition. Pooled PCMCI has no state adaptation.",
              "rpcmci_output": "Score the official causal_results returned by the selected annealing, with no custom refit; latest gamma may differ from the gamma used in its last causal step. Record final gamma change.",
              "runtime": "Oracle regression OLS/HAC rows share combined computation time; PCMCI and Regime-PCMCI include full searches. Memory is sampled process RSS, not isolated per-method peaks. Timings for workers>1 are observed under concurrent independent-repetition load, not isolated speed benchmarks.",
              "sources": ["https://doi.org/10.1063/5.0020538", "https://github.com/jakobrunge/tigramite/blob/master/tigramite/rpcmci.py"]}
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = args.output / "arrays"
    artifacts.mkdir(exist_ok=True)
    protocol = args.output / "frozen_protocol.json"
    frozen = json.loads(json.dumps(frozen))
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8"))["frozen"] != frozen:
            raise RuntimeError("Frozen protocol changed: use a new output directory")
    else:
        versions = package_versions()
        versions["ortools"] = importlib.metadata.version("ortools")
        protocol.write_text(json.dumps({"registered_utc": datetime.now(timezone.utc).isoformat(),
                                       "frozen": frozen, "packages": versions}, indent=2), encoding="utf-8")
        snapshot = args.output / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for file in files:
            shutil.copy2(file, snapshot / file.name)
    path = args.output / "repetitions.jsonl"
    done = {r["repetition"] for r in [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]} if path.exists() else set()
    pending = [rep for rep in range(args.replicates) if rep not in done]

    def save(record):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            hidden = next(row for row in record["rows"] if row["method"] == "regime_pcmci_hidden")
            print(f"RPCMCI repetition={record['repetition']}, completed={len(done)+1}/{args.replicates}: seconds={hidden['timing']['wall_seconds']:.2f}, failed={hidden['run_failed']}, accuracy={hidden['state_accuracy']}", flush=True)
            done.add(record["repetition"])
            summarize(args.output, args.replicates, args.anneals)

    try:
        if args.workers == 1:
            for rep in pending:
                save(one_repetition(config, rep, args.iterations, args.anneals, artifacts))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(one_repetition, config, rep, args.iterations, args.anneals, artifacts) for rep in pending]
                for future in as_completed(futures):
                    save(future.result())
    finally:
        summarize(args.output, args.replicates, args.anneals)


if __name__ == "__main__":
    main()
