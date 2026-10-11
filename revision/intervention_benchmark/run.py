"""Run the frozen independent-episode intervention benchmark on two threads."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from revision.intervention_benchmark.simulation import make_dataset


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def summarize(out):
    frames = [pd.read_csv(p) for p in sorted(out.glob("models/*/rep_*/metrics.csv"))]
    assert frames
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(out / "all_metrics.csv", index=False)
    # Estimator column names are fixed by the estimator implementation.
    groups = [k for k in ("scenario", "method", "horizon_hours", "outcome") if k in data]
    if len(groups) != 4:
        raise ValueError(f"Unexpected metric keys: {list(data.columns)}")
    rows = []
    for key, g in data.groupby(groups, sort=True):
        rec = dict(zip(groups, key)); n = len(g)
        rec["n_replicates"] = n
        for name in ("effect_estimate", "test_bank_truth_mean", "signed_error_vs_test_bank", "absolute_error_vs_test_bank", "factual_test_mse", "episode_twin_effect_rmse", "truth_mc_standard_error"):
            if name not in g:
                continue
            x = pd.to_numeric(g[name], errors="coerce").dropna().to_numpy()
            if not len(x):
                continue
            rec[f"mean_{name}"] = float(x.mean())
            if len(x) > 1:
                half = float(t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))
                rec[f"mc95_low_{name}"] = float(x.mean()-half)
                rec[f"mc95_high_{name}"] = float(x.mean()+half)
        if "signed_error_vs_test_bank" in g:
            rec["ate_rmse"] = float(np.sqrt(np.mean(g.signed_error_vs_test_bank.to_numpy() ** 2)))
        if "reject_zero_on_exact_null" in g:
            x = g.reject_zero_on_exact_null.dropna().astype(str).map({"True":1,"False":0,"1":1,"0":0,"1.0":1,"0.0":0}).dropna().to_numpy()
            if len(x):
                k = int(x.sum()); total = len(x); p = k/total; z = 1.959963984540054
                den = 1+z*z/total
                centre = (p+z*z/(2*total))/den
                half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
                rec.update(null_rejections=k, null_tests=total, null_rejection_rate=p, null_wilson_low=centre-half, null_wilson_high=centre+half)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    return len(data), len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT.parent / "revision_outputs/intervention_benchmark_v1")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(); out = args.output
    if args.summarize_only:
        print(summarize(out)); return
    from revision.intervention_benchmark.estimators import analyze
    design = json.loads((out / "design.json").read_text(encoding="utf8"))
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="running",
                  design_sha256=sha(out / "design.json"), completed=[],
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    status["source_sha256"]["revision/inference.py"] = sha(ROOT / "revision/inference.py")
    if (out / "run_manifest.json").exists():
        raise FileExistsError("Recorded run exists; use a new output directory")
    save(out / "run_manifest.json", status)
    started = time.monotonic()
    with threadpool_limits(limits=2):
        for replicate in range(design["replicates"]):
            for scenario in design["scenarios"]:
                data, checks, trace = make_dataset(design, scenario, replicate)
                dest = out / "data" / scenario["name"] / f"rep_{replicate:03d}"
                dest.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(dest / "episodes.npz", **data)
                np.savez_compressed(dest / "trajectories.npz", **trace)
                save(dest / "numerics.json", checks)
                analyze(data, out / "models", scenario["name"], replicate)
                status["completed"].append(dict(scenario=scenario["name"], replicate=replicate, episodes_sha256=sha(dest / "episodes.npz"), numerics_sha256=sha(dest / "numerics.json")))
                status["elapsed_seconds"] = time.monotonic() - started
                save(out / "run_manifest.json", status)
                print(f"{len(status['completed'])}/80 {scenario['name']} replicate={replicate} elapsed={status['elapsed_seconds']:.1f}s", flush=True)
    status["metrics_rows"], status["summary_rows"] = summarize(out)
    status.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
    save(out / "run_manifest.json", status)


if __name__ == "__main__":
    main()
