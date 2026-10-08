"""Matched, CPU-only multinomial logistic baseline for the weather-token pilot.

Importing this module does not prepare data or fit models.  The command-line
entry point uses a separate baseline_prep tree, never the Transformer case's
preprocessing directory.  All model settings are fixed, with no validation or
test-set tuning.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import warnings

# Set before importing numerical libraries; threadpool_limits below also caps
# libraries whose thread settings have already been initialized by a caller.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "revision_outputs" / "weather_token_pilot"
CASES = ("pooled", "nh_to_sh", "sh_to_nh")
HORIZONS = (6, 12, 24)
LAGS = tuple(range(-7, 1))
N_FEATURES, N_META = 18, 6


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    """Hash the shape, dtype, and C-order bytes, not an ndarray's repr."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": value.shape, "dtype": value.dtype.str},
                             sort_keys=True).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf8")


def design_matrix(d, index):
    """Oldest-to-newest 8 x 18 history followed by origin metadata: 150 columns."""
    index = np.asarray(index)
    feature, metadata = np.asarray(d["F"]), np.asarray(d["meta"])
    if index.ndim != 2 or index.shape[1] != 2 or not len(index):
        raise ValueError("Expected a nonempty N x 2 index")
    if not np.issubdtype(index.dtype, np.integer):
        raise ValueError("Time/region indices must be integer-valued arrays")
    if feature.ndim != 3 or feature.shape[-1] != N_FEATURES:
        raise ValueError("Expected F[time, region, 18]")
    if metadata.shape != feature.shape[:2] + (N_META,):
        raise ValueError("Expected meta[time, region, 6]")
    t, r = index[:, 0], index[:, 1]
    if t.min() < 7 or t.max() >= feature.shape[0]:
        raise ValueError("An origin has an out-of-bounds history")
    if r.min() < 0 or r.max() >= feature.shape[1]:
        raise ValueError("A region index is out of bounds")
    matrix = np.empty((len(index), len(LAGS) * N_FEATURES + N_META), dtype=np.float64)
    for position, lag in enumerate(LAGS):
        matrix[:, position * N_FEATURES:(position + 1) * N_FEATURES] = feature[t + lag, r]
    matrix[:, -N_META:] = metadata[t, r]
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite design values: baseline does not impute or change rows")
    return matrix


def labels_at(d, index):
    value = np.asarray(d["labels"])[index[:, 0], index[:, 1]]
    if value.shape != (len(index), len(HORIZONS)) or not np.isin(value, [0, 1, 2]).all():
        raise ValueError("Expected valid decrease/stable/increase labels at all horizons")
    return value.astype(np.int64, copy=False)


def predict_models(models, matrix):
    probability = np.stack([model.predict_proba(matrix) for model in models], axis=1)
    if probability.shape != (len(matrix), 3, 3):
        raise ValueError("Unexpected probability shape")
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(-1), 1., atol=1e-12):
        raise ValueError("Invalid class probabilities")
    return probability


def run_case(d, case, output, prep_dir, threads=4):
    """Fit once with fixed settings; write only this case's linear_logistic outputs."""
    if case not in CASES or not 1 <= threads <= 4:
        raise ValueError("Invalid case or CPU thread limit")
    # Reuse exactly the pilot metric definitions. train.py has a guarded main,
    # so this imports functions only, without training or device allocation.
    from train import metrics

    destination = Path(output) / case / "linear_logistic"
    destination.mkdir(parents=True, exist_ok=True)
    prep_dir = Path(prep_dir)
    indices = {name: np.asarray(d[f"{name}_idx"]) for name in ("train", "val", "test")}
    for name, index in indices.items():
        if index.ndim != 2 or index.shape[1] != 2 or not len(index):
            raise ValueError(f"Invalid {name} index")
        if len(np.unique(index, axis=0)) != len(index):
            raise ValueError(f"Duplicate {name} forecast origins")
    if not (indices["train"][:, 0].max() < indices["val"][:, 0].min()
            and indices["val"][:, 0].max() < indices["test"][:, 0].min()):
        raise ValueError("Training, validation and test origins are not time-separated")
    source_regions = np.unique(indices["train"][:, 1])
    target_regions = np.unique(indices["test"][:, 1])
    if not np.array_equal(source_regions, np.unique(indices["val"][:, 1])):
        raise ValueError("Validation must use the same source regions as training")
    if case != "pooled" and np.intersect1d(source_regions, target_regions).size:
        raise ValueError("Transfer source and target regions must be disjoint")

    logistic_parameters = dict(C=1.0, penalty="l2", solver="lbfgs", max_iter=500,
                               tol=1e-4, class_weight=None, fit_intercept=True)
    settings = {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "parameters": logistic_parameters,
        "horizons_hours": list(HORIZONS),
        "class_order": ["decrease", "stable", "increase"],
        "input": "F[t-7:t+1, region, :18] flattened oldest-first; meta[t, region, :6]",
        "input_columns": 150,
        "scaling": "StandardScaler with_mean=True, with_std=True, fitted on train_idx only",
        "selection": "none; all parameters fixed, no validation/test selection",
        "cpu_thread_limit": threads,
    }
    settings_sha = hashlib.sha256(json.dumps(settings, sort_keys=True,
                                             separators=(",", ":")).encode("utf8")).hexdigest()
    data_code = Path(__file__).with_name("data.py")
    protocol_path = Path(output) / "protocol.json"
    provenance = {
        "code_sha256": file_sha(__file__),
        "data_code_sha256": file_sha(data_code),
        "metrics_code_sha256": file_sha(Path(__file__).with_name("train.py")),
        "protocol_sha256": file_sha(protocol_path) if protocol_path.exists() else None,
        "parameters_sha256": settings_sha,
        "index_sha256": {name: array_sha(index) for name, index in indices.items()},
        "prep_json_sha256": {str(path.relative_to(prep_dir)): file_sha(path)
                             for path in sorted(prep_dir.rglob("*.json"))},
    }
    complete = destination / "run.json"
    if complete.exists():
        old = json.loads(complete.read_text(encoding="utf8"))
        if old.get("status") == "completed":
            if old.get("provenance") != provenance:
                raise ValueError("Completed baseline provenance differs; use a new output directory")
            for name, expected in old["output_sha256"].items():
                if not (destination / name).exists() or file_sha(destination / name) != expected:
                    raise ValueError(f"Completed baseline artifact missing or altered: {name}")
            print(f"REUSE {case} linear_logistic", flush=True)
            return old

    start = time.perf_counter()
    record = {"status": "running", "case": case, "variant": "linear_logistic",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "settings": settings, "provenance": provenance,
              "python": platform.python_version(), "numpy": np.__version__,
              "sklearn": sklearn.__version__,
              "sample_counts": {name: len(index) for name, index in indices.items()},
              "source_region_array_indices": source_regions.tolist(),
              "target_region_array_indices": target_regions.tolist(),
              "models": []}
    dump(complete, record)
    try:
        with threadpool_limits(limits=threads):
            record["active_threadpools"] = threadpool_info()
            train_x = design_matrix(d, indices["train"])
            train_y = labels_at(d, indices["train"])
            record["training_design_sha256_before_scaling"] = array_sha(train_x)
            record["training_labels_sha256"] = array_sha(train_y)
            scaler = StandardScaler(copy=False)
            train_x = scaler.fit_transform(train_x)
            record["scaler"] = {
                "fit_rows": int(scaler.n_samples_seen_),
                "mean_sha256": array_sha(scaler.mean_),
                "scale_sha256": array_sha(scaler.scale_),
                "zero_variance_column_indices": np.flatnonzero(scaler.var_ == 0).tolist(),
            }
            models = []
            for h, hours in enumerate(HORIZONS):
                if not np.array_equal(np.unique(train_y[:, h]), [0, 1, 2]):
                    raise ValueError(f"Training lacks one of the three classes at +{hours}h")
                model = LogisticRegression(**logistic_parameters)
                fit_start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(train_x, train_y[:, h])
                if not np.array_equal(model.classes_, [0, 1, 2]):
                    raise ValueError("Estimator changed the required class order")
                model_record = {
                    "horizon_hours": hours,
                    "fit_seconds": time.perf_counter() - fit_start,
                    "n_iter": model.n_iter_.tolist(),
                    "iteration_limit_reached": bool(np.any(model.n_iter_ >= 500)),
                    "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
                    "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                    "classes": model.classes_.tolist(),
                    "training_class_counts": np.bincount(train_y[:, h], minlength=3).tolist(),
                    "coefficient_sha256": array_sha(model.coef_),
                    "intercept_sha256": array_sha(model.intercept_),
                }
                record["models"].append(model_record)
                models.append(model)
                print(json.dumps({"case": case, "variant": "linear_logistic", **model_record}), flush=True)
                dump(complete, record)
            del train_x

            for split in ("val", "test"):
                matrix = scaler.transform(design_matrix(d, indices[split]))
                probability = predict_models(models, matrix)
                label = labels_at(d, indices[split])
                prefix = "validation_" if split == "val" else ""
                np.savez_compressed(destination / f"{prefix}predictions.npz",
                                    index=indices[split], probability=probability, label=label)
                rows = metrics(label, probability)
                pd.DataFrame(rows).to_csv(destination / f"{prefix}metrics.csv", index=False)
                record[f"{split}_metrics"] = rows
                del matrix, probability
            joblib.dump({"models": models, "scaler": scaler, "settings": settings,
                         "provenance": provenance}, destination / "models.joblib", compress=3)

        artifacts = ("predictions.npz", "validation_predictions.npz", "metrics.csv",
                     "validation_metrics.csv", "models.joblib")
        record["output_sha256"] = {name: file_sha(destination / name) for name in artifacts}
        record["status"] = "completed"
        record["total_seconds"] = time.perf_counter() - start
        record["completed_utc"] = datetime.now(timezone.utc).isoformat()
        dump(complete, record)
        print(f"DONE {case} linear_logistic, {record['total_seconds']:.1f}s", flush=True)
        return record
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        record["total_seconds"] = time.perf_counter() - start
        dump(complete, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    from data import load_raw, prepare_case

    raw = load_raw()
    for case in args.cases:
        prep = args.output / "baseline_prep" / case
        d = prepare_case(raw, case, prep)
        run_case(d, case, args.output, prep, threads=args.threads)


if __name__ == "__main__":
    main()
