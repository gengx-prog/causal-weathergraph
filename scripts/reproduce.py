"""Run published analyses in a new directory and compare every reference CSV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from verify_artifacts import ROOT, check_records, sha256


def compare_csv(actual, reference):
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal, assert_series_equal
    a, b = pd.read_csv(actual), pd.read_csv(reference)
    # Row/column ordering, categorical labels and Boolean decisions must agree.
    # Floating-point estimates allow small platform/BLAS differences.
    assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-7, atol=1e-10)
    for column in b.select_dtypes(include=["integer", "bool"]).columns:
        assert_series_equal(a[column], b[column], check_dtype=False, check_exact=True)
    maximum = 0.0
    for column in a.select_dtypes(include="number").columns:
        if column in b.select_dtypes(include="number").columns:
            diff = np.abs(a[column].to_numpy(float) - b[column].to_numpy(float))
            finite = diff[np.isfinite(diff)]
            if len(finite):
                maximum = max(maximum, float(finite.max()))
    return {"file": actual.name, "rows": len(a), "columns": len(a.columns),
            "max_absolute_numeric_difference": maximum, "status": "passed"}


def compare_whec_metadata(actual, reference):
    a = json.loads(Path(actual).read_text(encoding="utf-8"))
    b = json.loads(Path(reference).read_text(encoding="utf-8"))

    def compare(left, right, label):
        if isinstance(right, dict):
            if not isinstance(left, dict) or left.keys() != right.keys():
                raise AssertionError(f"WHEC metadata keys differ: {label}")
            for key in right:
                compare(left[key], right[key], f"{label}.{key}")
        elif isinstance(right, float):
            if isinstance(left, bool) or not isinstance(left, (int, float)) or not math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-10):
                raise AssertionError(f"WHEC numeric decision differs: {label}")
        elif type(left) is not type(right) or left != right:
            raise AssertionError(f"WHEC decision differs: {label}: {left!r} != {right!r}")

    for field in ("design_sha256", "periods", "decision"):
        compare(a[field], b[field], field)
    return {"file": "manifest.json", "fields": ["design_sha256", "periods", "decision"],
            "verdicts": {key: value["verdict"] for key, value in a["decision"].items()}, "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "artifacts/data")
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite recorded evidence")
    parser.add_argument("--experiment", choices=("core", "ceres", "whec", "all"), default="core")
    args = parser.parse_args()
    data = args.data_root.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new directory to preserve evidence")
    manifest = json.loads((ROOT / "artifacts/release-manifest.json").read_text(encoding="utf-8"))
    required = {"core-inputs.zip"}
    if args.experiment != "core":
        required |= {"extended-inputs.zip", "experiment-evidence.zip"}
    integrity = []
    for asset in manifest["assets"]:
        if asset["name"] in required:
            count, errors = check_records(data, asset["files"])
            if errors:
                parser.error("Input integrity check failed: " + "; ".join(errors[:5]))
            integrity.append({"asset": asset["name"], "files_verified": count})
    output.mkdir(parents=True)
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "2"
    env["PYTHONUTF8"] = "1"
    jobs = []
    if args.experiment in ("core", "all"):
        jobs.append(("three_stage", "revision.run_three_stage", [
            "--input", str(data / "inputs/region_trainfit.npz"),
            "--symmetric-candidates", str(data / "graph_nulls/candidates_symmetric_core.csv"),
            "--screen-reference", str(data / "holdout_aligned/discovery_edges.csv")], "*.csv"))
    if args.experiment in ("ceres", "all"):
        for climatology in ("month", "month_hour"):
            jobs.append(("ceres_cloud_substitution", "revision.run_ceres_cloud_substitution", [
                "--climatology", climatology, "--inputs", str(data / "inputs"),
                "--ceres-npz", str(data / "ceres_cloud_substitution/ceres_cloud_6h_64x32.npz"),
                "--era5-cloud-npz", str(data / "ceres_cloud_substitution/era5_cloud_6h_64x32_2017_2025.npz"),
                "--symmetric-candidates", str(data / "graph_nulls/candidates_symmetric_core.csv")], f"*_{climatology}.csv"))
    if args.experiment in ("whec", "all"):
        (output / "whec_test").mkdir()
        shutil.copyfile(data / "whec_test/design.json", output / "whec_test/design.json")
        jobs.append(("whec_test", "revision.run_whec_test", [
            "--artifact-root", str(data), "--era5-cloud-npz",
            str(data / "ceres_cloud_substitution/era5_cloud_6h_64x32_2017_2025.npz")], "*.csv"))
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "experiment": args.experiment, "input_integrity": integrity,
              "comparison_tolerance": {"rtol": 1e-7, "atol": 1e-10}, "jobs": [],
              "planned_jobs": len(jobs), "completed_jobs": 0, "status": "running",
              "scope": "Rerun from archived processed inputs; raw acquisition and preprocessing are not rerun."}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    status = 0
    for i, (folder, module, options, pattern) in enumerate(jobs):
        command = [sys.executable, "-m", module, *options, "--output", str(output / folder)]
        started = time.perf_counter()
        log = output / f"{i + 1}_{folder}.log"
        print(f"Running {module}; log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = {"module": module, "options": options, "exit_code": process.returncode,
                  "elapsed_seconds": time.perf_counter() - started,
                  "runner_sha256": sha256(ROOT / (module.replace(".", "/") + ".py")), "comparisons": []}
        if process.returncode:
            record["error"] = f"Experiment failed; inspect {log.name}"
            status = 1
        else:
            reference_files = sorted((data / folder).glob(pattern))
            if not reference_files:
                record["error"] = "No reference tables found"
                status = 1
            for reference in reference_files:
                try:
                    record["comparisons"].append(compare_csv(output / folder / reference.name, reference))
                except (AssertionError, OSError, ValueError) as exc:
                    record["comparisons"].append({"file": reference.name, "status": "failed", "error": str(exc)})
                    status = 1
            if folder == "whec_test":
                try:
                    record["metadata_comparison"] = compare_whec_metadata(output / folder / "manifest.json", data / folder / "manifest.json")
                except (AssertionError, OSError, ValueError, KeyError) as exc:
                    record["metadata_comparison"] = {"status": "failed", "error": str(exc)}
                    status = 1
        report["jobs"].append(record)
        report["completed_jobs"] = len(report["jobs"])
        report["status"] = "failed" if status else ("passed" if len(report["jobs"]) == len(jobs) else "running")
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{module}: exit={process.returncode}; compared {len(record['comparisons'])} tables", flush=True)
        if process.returncode:
            break
    print(f"Verification {report['status']}: {output / 'verification.json'}")
    return status


if __name__ == "__main__":
    sys.exit(main())
