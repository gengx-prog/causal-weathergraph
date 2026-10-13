"""Run a small offline CPU demonstration of the actual graph-inference code.

This is one synthetic realization, not a reproduction of the paper's data or
an estimate of a method's false-discovery rate. No model weights are required.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

# Set limits before importing numerical packages; no device inspection is used.
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _variable in THREAD_VARIABLES:
    os.environ[_variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_demo(output, seed=20260929):
    """Create a new output folder, fit saved synthetic data, and verify results."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One synthetic realization using the production calendar-HAC regression code; not paper-result reproduction, neural training, or FDR calibration.",
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "threads": {name: "1" for name in THREAD_VARIABLES},
        "source_sha256": {name: sha256(ROOT / name) for name in
                          ("scripts/run_demo.py", "revision/simulations.py", "revision/inference.py")},
        "checks": {}, "metrics": {},
    }

    def save_report():
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def check(name, condition, **details):
        report["checks"][name] = {"passed": bool(condition), **details}
        if not condition:
            raise AssertionError(f"Demo verification failed: {name}")

    save_report()
    try:
        import numpy as np
        import pandas as pd
        from revision.inference import fit_edge
        from revision.simulations import (
            SimulationConfig, bh, candidate_edges, lagged_design,
            regression_pvalues, rng_for, score_discoveries, simulate_graph,
        )

        report["packages"] = {name: importlib.metadata.version(name)
                              for name in ("numpy", "scipy", "pandas")}
        config = replace(SimulationConfig(), seed=seed, n_graph=2048,
                         burn_in=512, n_nodes=3, max_lag=3, hac_bandwidth=64,
                         graph_repetitions=1, null_repetitions=0,
                         protocol_role="offline_reviewer_demonstration")
        report["configuration"] = asdict(config)
        report["scenario"] = "weak_ar"
        report["random_stream"] = {"family": 1, "scenario": 0, "repetition": 0}
        data, masks, truths, radii = simulate_graph(
            config, "weak_ar", rng_for(seed, 1, 0, 0))
        repeated, _, repeated_truth, _ = simulate_graph(
            config, "weak_ar", rng_for(seed, 1, 0, 0))
        truth = truths[0]
        mask = masks["state_0"]
        edges = candidate_edges(config.n_nodes, config.max_lag)
        check("seed_reproduces_input", np.array_equal(data, repeated) and truths == repeated_truth)
        check("finite_data_and_stable_generator", np.isfinite(data).all() and max(radii) < 1,
              shape=list(data.shape), companion_radii=radii)
        check("candidate_family_and_truth", len(edges) == 18 and len(truth) == 3
              and truth.issubset(set(edges)), candidate_hypotheses=len(edges), true_hypotheses=len(truth))
        check("calendar_response_cutoff", not mask[:6].any() and mask[6:].all(),
              response_samples=int(mask.sum()))

        frame = pd.DataFrame(data, columns=["node_0", "node_1", "node_2"])
        frame.insert(0, "time_index", np.arange(len(data)))
        frame.to_csv(output / "synthetic_data.csv", index=False, float_format="%.17g")
        pd.DataFrame(sorted(truth), columns=["source", "target", "lag"]).to_csv(
            output / "truth.csv", index=False)
        # The fitting stage actually reads the saved input artifact.
        saved = pd.read_csv(output / "synthetic_data.csv", float_precision="round_trip")
        data = saved[["node_0", "node_1", "node_2"]].to_numpy(float)
        check("saved_input_round_trip", np.array_equal(data, repeated)
              and np.array_equal(saved.time_index.to_numpy(), np.arange(config.n_graph)))

        rows = []
        for conditioning in ("own_history", "all_observed_history"):
            probabilities, diagnostics = regression_pvalues(
                data, mask, edges, config, conditioning=conditioning)
            pvalues = probabilities["hac64"]
            selected, qvalues = bh(pvalues, config.alpha)
            check(conditioning + "_valid_fits", diagnostics["invalid_tests"] == 0
                  and diagnostics["min_samples"] == 2042 and diagnostics["max_samples"] == 2042
                  and pvalues.shape == (18,) and np.isfinite(pvalues).all()
                  and ((0 <= pvalues) & (pvalues <= 1)).all(), **diagnostics)
            metrics = score_discoveries(selected, truth, edges)
            check(conditioning + "_score_accounting",
                  metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"] == len(edges)
                  and metrics["tp"] + metrics["fn"] == len(truth))
            report["metrics"][conditioning] = metrics
            for i, (source, target, lag) in enumerate(edges):
                rows.append({"conditioning": conditioning, "source": source, "target": target,
                             "lag": lag, "is_true_direct_edge": (source, target, lag) in truth,
                             "p_ols": float(probabilities["ols"][i]),
                             "p_hac64": float(pvalues[i]), "q_hac64": float(qvalues[i]),
                             "selected": bool(selected[i])})
        pd.DataFrame(rows).to_csv(output / "fitted_tests.csv", index=False, float_format="%.17g")
        report["metric_interpretation"] = (
            "Scores compare selected source-target-lag hypotheses with direct generating edges. "
            "Indirect dependencies can be nonnull for own-history prediction. The observed FDP "
            "in this single realization is not an estimated FDR or a calibration guarantee. "
            "Passing checks does not require perfect recovery or a particular score.")

        # Independent least-squares and direct Bartlett-sum checks of one fit.
        source, target, lag = sorted(truth)[0]
        design = lagged_design(data, config.max_lag)
        source_col = (lag - 1) * config.n_nodes + source
        control_cols = [i for i in range(design.shape[1]) if i != source_col]
        fitted = fit_edge(data[:, target], design[:, control_cols], design[:, source_col],
                          mask, bandwidths=(64,))
        controls = np.column_stack([np.ones(mask.sum()), design[mask][:, control_cols]])
        x = np.column_stack([controls, design[mask, source_col]])
        y = data[mask, target]
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ beta
        z = design[mask, source_col]
        residual_source = z - controls @ np.linalg.lstsq(controls, z, rcond=None)[0]
        scores = np.zeros(len(data))
        scores[mask] = residual_source * residual
        meat = scores @ scores + 2 * sum((1 - h / 65) * (scores[h:] @ scores[:-h])
                                         for h in range(1, 65))
        independent_se = float(np.sqrt(meat * mask.sum() / (mask.sum() - x.shape[1]))
                               / (residual_source @ residual_source))
        check("independent_ols_coefficient", np.isclose(fitted["effect"], beta[-1], rtol=1e-9, atol=1e-11),
              absolute_difference=float(abs(fitted["effect"] - beta[-1])))
        check("independent_calendar_hac", np.isclose(fitted["se_hac64"], independent_se, rtol=1e-9, atol=1e-11),
              absolute_difference=float(abs(fitted["se_hac64"] - independent_se)))
        selected, qvalues = bh([0.001, 0.02, 0.04, np.nan], config.alpha)
        check("bh_hand_calculation_and_invalid_family_member",
              np.array_equal(selected, [True, True, False, False])
              and np.allclose(qvalues, [0.004, 0.04, 0.04 * 4 / 3, 1], rtol=1e-12, atol=1e-14))
        persisted = pd.read_csv(output / "fitted_tests.csv")
        check("saved_result_completeness", len(persisted) == 36
              and not persisted.duplicated(["conditioning", "source", "target", "lag"]).any())
        report["artifact_sha256"] = {name: sha256(output / name) for name in
                                     ("synthetic_data.csv", "truth.csv", "fitted_tests.csv")}
        report["status"] = "passed"
        save_report()
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_report()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing files are never overwritten")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    try:
        report = run_demo(args.output, args.seed)
    except FileExistsError:
        parser.error("Output already exists; choose a new directory to preserve evidence")
    except Exception as exc:
        print(f"Demo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Synthetic CPU demo: {report['status']}; {len(report['checks'])} checks passed.")
    for method, metrics in report["metrics"].items():
        print(f"  {method}: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, F1={metrics['f1']:.3f}")
    print("Results: synthetic_data.csv, truth.csv, fitted_tests.csv, summary.json")
    print("This single synthetic run is not paper-result reproduction or FDR calibration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
