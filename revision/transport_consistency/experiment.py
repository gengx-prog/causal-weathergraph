"""Frozen, bounded transport-feature forecast comparisons (not causal effects)."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

TASKS = ("humidity_change", "cloud_level")
ALPHAS = np.array([1., 10., 100., 1000.])
HISTORY = np.array([0, -1, -2])
FLUX_NAMES = ["mean_qu", "mean_qv", "cov_qu", "cov_qv", "mfc"]
LOCAL_NAMES = ["u", "v", "q", "temperature", "cloud"]
SEED = 20261001


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf8")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def named_order(names, expected):
    """Require explicit metadata; accept only documented variable synonyms."""
    aliases = {"humidity": "q", "specific_humidity": "q",
               "cloud_cover": "cloud", "total_cloud_cover": "cloud"}
    canonical = [aliases.get(str(n), str(n)) for n in names]
    if len(canonical) != len(set(canonical)) or set(canonical) != set(expected):
        raise ValueError(f"Unexpected names {canonical}; expected {expected}")
    return [canonical.index(n) for n in expected]


def load_input(path):
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    required = ["timestamps", "region_ids", "region_lat", "region_lon", "local",
                "flux", "grid", "controls", "source_segment", "local_names",
                "flux_names", "grid_names", "control_names"]
    for name in required:
        if name not in data:
            raise ValueError(f"Missing input {name}")
    data["timestamps"] = data["timestamps"].astype("datetime64[ns]")
    t = data["timestamps"]
    if not np.all(np.diff(t) == np.timedelta64(6, "h")):
        raise ValueError("Input calendar is not complete six-hourly")
    if data["region_ids"].tolist() != [15, 48]:
        raise ValueError("This experiment is fixed to regions 15 and 48")
    data["local"] = data["local"][..., named_order(data["local_names"], LOCAL_NAMES)]
    data["flux"] = data["flux"][..., named_order(data["flux_names"], FLUX_NAMES)]
    data["grid"] = data["grid"][..., named_order(data["grid_names"], LOCAL_NAMES[:4])]
    for key, shape in [("local", (len(t), 2, 5)), ("flux", (len(t), 2, 5)),
                       ("controls", (len(t), 2, 6))]:
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} input")
    if data["grid"].ndim != 4 or data["grid"].shape[:2] != (len(t), 2):
        raise ValueError("Invalid grid shape")
    mask = data.get("grid_mask", np.ones(data["grid"].shape[1:3], dtype=bool)).astype(bool)
    if mask.shape != data["grid"].shape[1:3]:
        raise ValueError("Invalid grid_mask shape")
    for r in range(2):
        if not mask[r].any() or not np.isfinite(data["grid"][:, r, mask[r], :]).all():
            raise ValueError("Invalid or nonfinite retained grid cells")
    data["grid_mask"] = mask
    if data["source_segment"].shape != (len(t),):
        raise ValueError("Invalid acquisition source labels")
    if np.any((data["local"][..., 4] < 0) | (data["local"][..., 4] > 1)):
        raise ValueError("Cloud input must be fraction [0,1]")
    return data


def eligible_rows(data):
    t = data["timestamps"]
    split = np.where(t < np.datetime64("2015-01-01"), "train",
                     np.where(t < np.datetime64("2019-01-01"), "validation", "test"))
    dates = pd.DatetimeIndex(t)
    origins = np.flatnonzero(dates.hour == 0)
    rows, ledger = [], []
    for i in origins:
        bounds = i >= 2 and i + 1 < len(t)
        same_split = bounds and np.all(split[i - 2:i + 2] == split[i])
        same_source = bounds and np.all(data["source_segment"][i - 2:i + 2] == data["source_segment"][i])
        retained = bool(bounds and same_split and same_source)
        ledger.append({"origin_index": int(i), "origin": str(t[i]), "split": split[i],
                       "source_segment": str(data["source_segment"][i]),
                       "complete_window": bool(bounds), "same_split": bool(same_split),
                       "same_source": bool(same_source), "retained": retained,
                       "context_start": str(t[i - 2]) if bounds else "",
                       "target": str(t[i + 1]) if bounds else ""})
        if retained:
            rows.append(i)
    idx = np.array(rows, dtype=np.int64)
    return idx, split[idx], pd.DataFrame(ledger)


def features(data, idx, region):
    local = data["local"][:, region]
    controls = data["controls"][:, region]
    common = np.concatenate([np.concatenate([local[idx + lag], controls[idx + lag]], axis=1)
                             for lag in HISTORY], axis=1)
    day = pd.DatetimeIndex(data["timestamps"][idx]).dayofyear.to_numpy()
    common = np.column_stack([common, np.sin(2 * np.pi * day / 365.2425),
                              np.cos(2 * np.pi * day / 365.2425)])
    flux = np.concatenate([data["flux"][idx + lag, region] for lag in HISTORY], axis=1)
    other = np.concatenate([data["flux"][idx + lag, 1 - region] for lag in HISTORY], axis=1)
    grid = np.concatenate([data["grid"][idx + lag, region][:, data["grid_mask"][region], :].reshape(len(idx), -1)
                           for lag in HISTORY], axis=1)
    names = []
    for lag in HISTORY:
        names.extend([f"lag{-lag}:{v}" for v in LOCAL_NAMES])
        names.extend([f"lag{-lag}:{v}" for v in data["control_names"].tolist()])
    names.extend(["dayofyear_sin", "dayofyear_cos"])
    flux_names = [f"lag{-lag}:{name}" for lag in HISTORY for name in FLUX_NAMES]
    grid_names = [f"lag{-lag}:cell{g}:{name}" for lag in HISTORY
                  for g in np.flatnonzero(data["grid_mask"][region]) for name in LOCAL_NAMES[:4]]
    mean_cols = [j for j in range(15) if j % 5 < 2]
    cov_cols = [j for j in range(15) if j % 5 < 4]
    return {
        "ridge_local": (common, names),
        "ridge_meanflux": (np.column_stack([common, flux[:, mean_cols]]), names + [flux_names[j] for j in mean_cols]),
        "ridge_meancov": (np.column_stack([common, flux[:, cov_cols]]), names + [flux_names[j] for j in cov_cols]),
        "ridge_flux_mfc": (np.column_stack([common, flux]), names + flux_names),
        "ridge_samegrid": (np.column_stack([common, grid]), names + grid_names),
        "ridge_samegrid_flux": (np.column_stack([common, grid, flux]), names + grid_names + flux_names),
        "ridge_spatial_mismatch": (np.column_stack([common, other]), names + [f"other_region:{n}" for n in flux_names]),
        "ridge_futureflux": (np.column_stack([common, data["flux"][idx + 1, region]]),
                              names + [f"FUTURE6h:{name}" for name in FLUX_NAMES]),
    }


def clip_target(prediction, task):
    return np.clip(prediction, 0., 100.) if task == "cloud_level" else prediction


class RidgeFamily:
    """One float64 Gram eigendecomposition serves all targets and penalties."""
    def __init__(self, x, training):
        self.mean = x[training].mean(axis=0, dtype=np.float64)
        self.scale = x[training].std(axis=0, dtype=np.float64)
        self.scale = np.where(self.scale < 1e-12, 1., self.scale)
        self.x = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        xt = self.x[training]
        gram = xt.T @ xt
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(gram)
        tolerance = max(float(np.max(np.abs(self.eigenvalues))), 1.) * 1e-10
        if self.eigenvalues.min() < -tolerance:
            raise ValueError("Ridge Gram matrix has material negative eigenvalues")
        self.eigenvalues = np.maximum(self.eigenvalues, 0.)
        self.training = training

    def fit(self, y, validation, task):
        intercept = float(y[self.training].mean())
        rhs = self.x[self.training].T @ (y[self.training] - intercept)
        transformed = self.eigenvectors.T @ rhs
        coefficients = self.eigenvectors @ (transformed[:, None] / (self.eigenvalues[:, None] + ALPHAS[None, :]))
        pred_val = self.x[validation] @ coefficients + intercept
        loss = np.mean((clip_target(pred_val, task) - y[validation, None]) ** 2, axis=0)
        selected = int(np.argmin(loss))
        coefficient = coefficients[:, selected]
        prediction = self.x @ coefficient + intercept
        normal_equation = self.x[self.training].T @ (self.x[self.training] @ coefficient - (y[self.training] - intercept)) + ALPHAS[selected] * coefficient
        relative = float(np.linalg.norm(normal_equation) / max(np.linalg.norm(rhs), 1.))
        if relative > 1e-7:
            raise ValueError(f"Ridge normal-equation check failed: {relative}")
        return prediction, coefficient, intercept, {
            "alpha": float(ALPHAS[selected]), "validation_mse_by_alpha": dict(zip(map(str, ALPHAS), map(float, loss))),
            "normal_equation_relative_residual": relative,
            "gram_min_eigenvalue": float(self.eigenvalues.min()),
            "gram_max_eigenvalue": float(self.eigenvalues.max()),
            "n_features": self.x.shape[1], "n_train": int(self.training.sum()),
            "n_validation": int(validation.sum()), "standardization": "training_only",
            "fit_period": "1979-2014", "selection_period": "2015-2018", "refit_after_selection": False,
        }


def prediction_record(data, idx, split, region, task, model, actual, raw, category):
    return pd.DataFrame({"origin_index": idx, "origin": data["timestamps"][idx],
                         "origin_time": data["timestamps"][idx],
                         "target_time": data["timestamps"][idx + 1], "split": split,
                         "source_segment": data["source_segment"][idx].astype(str),
                         "region_id": int(data["region_ids"][region]),
                         "region_lat": float(data["region_lat"][region]),
                         "task": task, "horizon_hours": 6, "model": model,
                         "category": category, "actual": actual, "y_true": actual, "raw_prediction": raw,
                         "prediction": clip_target(raw, task)})


def local_season(months, latitude):
    # NH DJF/MAM/JJA/SON; SH shifted by two seasons.
    season = ((months % 12) // 3 + (0 if latitude >= 0 else 2)) % 4
    return np.array(["winter", "spring", "summer", "autumn"])[season]


def slice_masks(frame):
    yield "all", np.ones(len(frame), dtype=bool)
    for source in sorted(frame.source_segment.unique()):
        yield f"source:{source}", frame.source_segment.to_numpy() == source
    seasons = local_season(pd.DatetimeIndex(frame.origin).month.to_numpy(), float(frame.region_lat.iloc[0]))
    for season in ["winter", "spring", "summer", "autumn"]:
        yield f"local_{season}", seasons == season


def summarize_metrics(predictions):
    rows = []
    for keys, frame in predictions.groupby(["region_id", "task", "model", "category", "split"], sort=True):
        for label, mask in slice_masks(frame):
            if not mask.any():
                continue
            error = frame.prediction.to_numpy()[mask] - frame.actual.to_numpy()[mask]
            raw_error = frame.raw_prediction.to_numpy()[mask] - frame.actual.to_numpy()[mask]
            rows.append(dict(zip(["region_id", "task", "model", "category", "split"], keys),
                             slice=label, n=int(mask.sum()), mse=float(np.mean(error ** 2)),
                             rmse=float(np.sqrt(np.mean(error ** 2))),
                             mae=float(np.mean(np.abs(error))), bias=float(np.mean(error)),
                             raw_mse=float(np.mean(raw_error ** 2)), raw_mae=float(np.mean(np.abs(raw_error))),
                             clipped_count=int(np.sum(frame.prediction.to_numpy()[mask] != frame.raw_prediction.to_numpy()[mask])),
                             clipped_fraction=float(np.mean(frame.prediction.to_numpy()[mask] != frame.raw_prediction.to_numpy()[mask]))))
    frame = pd.DataFrame(rows)
    pooled = []
    for keys, group in frame.groupby(["task", "model", "category", "split", "slice"], sort=True):
        if len(group) != 2:
            raise ValueError("Missing region in pooled metric")
        row = dict(zip(["task", "model", "category", "split", "slice"], keys))
        row.update({"region_id": "pooled_equal_regions", "n": int(group.n.sum()),
                    "clipped_count": int(group.clipped_count.sum())})
        for field in ["mse", "mae", "bias", "raw_mse", "raw_mae", "clipped_fraction"]:
            row[field] = float(group[field].mean())
        row["rmse"] = float(np.sqrt(row["mse"]))
        pooled.append(row)
    return pd.concat([frame, pd.DataFrame(pooled)], ignore_index=True)


def bootstrap_indices(origins, sources, repetitions=500, block_days=30, seed=SEED):
    """Preserve shared dates/regions and source population sizes; no seam-crossing blocks."""
    rng = np.random.default_rng(seed)
    out = np.empty((repetitions, len(origins)), dtype=np.int32)
    day = pd.DatetimeIndex(origins).to_numpy(dtype="datetime64[D]")
    cursor = 0
    strata = []
    for source in sorted(np.unique(sources)):
        positions = np.flatnonzero(sources == source)
        starts = np.array([j for j in range(len(positions) - block_days + 1)
                           if np.all(np.diff(day[positions[j:j + block_days]]) == np.timedelta64(1, "D"))], dtype=int)
        if not len(starts):
            raise ValueError("Insufficient consecutive dates for calendar block bootstrap")
        count = int(np.ceil(len(positions) / block_days))
        draws = rng.choice(starts, size=(repetitions, count), replace=True)
        sampled = positions[(draws[..., None] + np.arange(block_days)).reshape(repetitions, -1)[:, :len(positions)]]
        out[:, cursor:cursor + len(positions)] = sampled
        cursor += len(positions)
        strata.append({"source_segment": str(source), "n_dates": len(positions), "eligible_starts": len(starts)})
    return out, strata


COMPARISONS = [
    ("ridge_local", "persistence"), ("ridge_local", "monthly_climatology"),
    ("ridge_meanflux", "ridge_local"), ("ridge_meancov", "ridge_meanflux"),
    ("ridge_flux_mfc", "ridge_meancov"), ("ridge_flux_mfc", "ridge_local"),
    ("ridge_samegrid", "ridge_local"), ("ridge_samegrid_flux", "ridge_samegrid"),
    ("hgb_samegrid", "ridge_samegrid"), ("hgb_samegrid_flux", "hgb_samegrid"),
    ("ridge_spatial_mismatch", "ridge_local"), ("ridge_flux_mfc", "ridge_spatial_mismatch"),
    ("horizontal_only_budget", "persistence"), ("ridge_flux_mfc", "horizontal_only_budget"),
    ("ridge_futureflux", "ridge_local"), ("ridge_futureflux", "ridge_flux_mfc"),
]


def summarize_comparisons(predictions, output):
    evaluation = predictions[predictions.split == "test"].copy()
    unique = evaluation[["origin_index", "origin", "source_segment"]].drop_duplicates().sort_values("origin_index")
    if unique.origin_index.duplicated().any():
        raise ValueError("Inconsistent evaluation calendars")
    bootstrap, strata = bootstrap_indices(unique.origin, unique.source_segment.to_numpy())
    np.savez_compressed(output / "bootstrap_indices.npz", indices=bootstrap,
                        origin_index=unique.origin_index.to_numpy(), origins=pd.DatetimeIndex(unique.origin).to_numpy())
    save_json(output / "bootstrap_design.json", {
        "seed": SEED, "repetitions": 500, "block_calendar_days": 30,
        "strata": strata, "same_draws_for_both_regions_all_tasks_models": True,
        "scope": "Exploratory paired loss uncertainty; no nominal coverage/FDR guarantee.",
        "blocks": "Overlapping consecutive calendar dates within each acquisition segment; concatenated and truncated to its original date count.",
    })
    rows = []
    pooled = {}
    for (region, task), group in evaluation.groupby(["region_id", "task"]):
        by_model = {m: f.sort_values("origin_index").reset_index(drop=True) for m, f in group.groupby("model")}
        for model, baseline in COMPARISONS:
            if model not in by_model or baseline not in by_model:
                continue
            f, b = by_model[model], by_model[baseline]
            if not np.array_equal(f.origin_index, unique.origin_index) or not np.array_equal(f.actual, b.actual):
                raise ValueError("Comparison rows are not matched")
            squared = (b.prediction.to_numpy() - b.actual.to_numpy()) ** 2 - (f.prediction.to_numpy() - f.actual.to_numpy()) ** 2
            absolute = np.abs(b.prediction.to_numpy() - b.actual.to_numpy()) - np.abs(f.prediction.to_numpy() - f.actual.to_numpy())
            for label, mask in slice_masks(f):
                selected = mask[bootstrap]
                denom = selected.sum(axis=1)
                if np.any(denom == 0):
                    raise ValueError("Empty bootstrap slice")
                d_mse = (squared[bootstrap] * selected).sum(axis=1) / denom
                d_mae = (absolute[bootstrap] * selected).sum(axis=1) / denom
                q_mse = np.quantile(d_mse, [.025, .975])
                q_mae = np.quantile(d_mae, [.025, .975])
                rows.append({"region_id": int(region), "task": task, "model": model,
                             "baseline": baseline, "category": str(f.category.iloc[0]), "slice": label,
                             "n": int(mask.sum()), "delta_mse_baseline_minus_model": float(squared[mask].mean()),
                             "delta_mae_baseline_minus_model": float(absolute[mask].mean()),
                             "exploratory_delta_mse_p025": float(q_mse[0]), "exploratory_delta_mse_p975": float(q_mse[1]),
                             "exploratory_delta_mae_p025": float(q_mae[0]), "exploratory_delta_mae_p975": float(q_mae[1]),
                             "positive_favors_model": True})
                key = (task, model, baseline, str(f.category.iloc[0]), label)
                pooled.setdefault(key, []).append({"n": int(mask.sum()), "mse": float(squared[mask].mean()),
                                                   "mae": float(absolute[mask].mean()), "boot_mse": d_mse, "boot_mae": d_mae})
    for (task, model, baseline, category, label), regions in pooled.items():
        if len(regions) != 2:
            raise ValueError("Missing region in pooled comparison")
        d_mse = np.mean([r["boot_mse"] for r in regions], axis=0)
        d_mae = np.mean([r["boot_mae"] for r in regions], axis=0)
        q_mse, q_mae = np.quantile(d_mse, [.025, .975]), np.quantile(d_mae, [.025, .975])
        rows.append({"region_id": "pooled_equal_regions", "task": task, "model": model,
                     "baseline": baseline, "category": category, "slice": label,
                     "n": sum(r["n"] for r in regions),
                     "delta_mse_baseline_minus_model": float(np.mean([r["mse"] for r in regions])),
                     "delta_mae_baseline_minus_model": float(np.mean([r["mae"] for r in regions])),
                     "exploratory_delta_mse_p025": float(q_mse[0]), "exploratory_delta_mse_p975": float(q_mse[1]),
                     "exploratory_delta_mae_p025": float(q_mae[0]), "exploratory_delta_mae_p975": float(q_mae[1]),
                     "positive_favors_model": True})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Refusing to overwrite existing model outputs")
    design = json.loads(args.design.read_text(encoding="utf8"))
    if design.get("task_horizons") != {"humidity_change": 6, "cloud_level": 6}:
        raise ValueError("Frozen design must specify the two agreed six-hour tasks")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = {"status": "running", "started_utc": utc_now(), "design_sha256": sha256(args.design),
                "input_sha256": sha256(args.input), "code_sha256": sha256(__file__),
                "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                "threads": 2, "tasks": list(TASKS), "task_units": {"humidity_change": "g kg-1", "cloud_level": "percentage points"},
                "prediction_clipping": "cloud_level only, all models, [0,100]; raw predictions retained",
                "models": [], "interpretation": "Retrospective conditional prediction and physical diagnostics; not causal identification."}
    save_json(args.output / "run_manifest.json", manifest)
    data = load_input(args.input)
    idx, split, ledger = eligible_rows(data)
    ledger.to_csv(args.output / "sample_ledger.csv", index=False)
    training, validation = split == "train", split == "validation"
    month = pd.DatetimeIndex(data["timestamps"][idx]).month.to_numpy()
    predictions = []
    with threadpool_limits(limits=2):
        for region in range(2):
            region_id = int(data["region_ids"][region])
            targets = {"humidity_change": 1000. * (data["local"][idx + 1, region, 2].astype(np.float64) - data["local"][idx, region, 2]),
                       "cloud_level": 100. * data["local"][idx + 1, region, 4].astype(np.float64)}
            for task, actual in targets.items():
                persistence = np.zeros(len(idx)) if task == "humidity_change" else 100. * data["local"][idx, region, 4]
                climate = np.array([actual[training & (month == m)].mean() for m in range(1, 13)])
                for model, raw in [("persistence", persistence), ("monthly_climatology", climate[month - 1])]:
                    predictions.append(prediction_record(data, idx, split, region, task, model, actual, raw, "forecast_baseline"))
                save_json(args.output / f"region{region_id}_{task}_climatology.json", {"origin_month_mean_training_target": climate.tolist()})
            predictions.append(prediction_record(data, idx, split, region, "humidity_change", "horizontal_only_budget",
                                                targets["humidity_change"], 21600. * 1000. * data["flux"][idx, region, 4],
                                                "instantaneous_horizontal_tendency_only_not_closed_budget"))
            designs = features(data, idx, region)
            for model, (x, names) in designs.items():
                model_started = time.perf_counter()
                family = RidgeFamily(x, training)
                category = ("deliberate_future_information_diagnostic" if model == "ridge_futureflux" else
                            "spatial_mismatch_sensitivity" if model == "ridge_spatial_mismatch" else "forecast_model")
                for task, actual in targets.items():
                    raw, coefficient, intercept, details = family.fit(actual, validation, task)
                    stem = f"region{region_id}_{task}_{model}"
                    np.savez_compressed(args.output / f"{stem}.npz", coefficient=coefficient, intercept=intercept,
                                        feature_mean=family.mean, feature_scale=family.scale, feature_names=np.array(names))
                    details.update({"region_id": region_id, "task": task, "model": model, "category": category,
                                    "output_parameters": f"{stem}.npz"})
                    save_json(args.output / f"{stem}.json", details)
                    predictions.append(prediction_record(data, idx, split, region, task, model, actual, raw, category))
                    manifest["models"].append(details)
                print(f"region={region_id} {model} features={x.shape[1]} seconds={time.perf_counter()-model_started:.2f}", flush=True)
                save_json(args.output / "run_manifest.json", manifest)
                del family
            for model, source in [("hgb_samegrid", "ridge_samegrid"), ("hgb_samegrid_flux", "ridge_samegrid_flux")]:
                x, names = designs[source]
                for task, actual in targets.items():
                    model_started = time.perf_counter()
                    estimator = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, min_samples_leaf=50,
                                                              learning_rate=.05, l2_regularization=1., max_bins=64,
                                                              early_stopping=False, random_state=SEED)
                    estimator.fit(x[training], actual[training])
                    raw = estimator.predict(x)
                    stem = f"region{region_id}_{task}_{model}"
                    joblib.dump(estimator, args.output / f"{stem}.joblib", compress=3)
                    details = {"region_id": region_id, "task": task, "model": model, "category": "forecast_model",
                               "parameters": estimator.get_params(), "n_features": x.shape[1],
                               "feature_names": names, "n_train": int(training.sum()),
                               "validation_used_for_model_selection": False, "n_iterations": int(estimator.n_iter_),
                               "elapsed_seconds": time.perf_counter() - model_started}
                    save_json(args.output / f"{stem}.json", details)
                    manifest["models"].append(details)
                    predictions.append(prediction_record(data, idx, split, region, task, model, actual, raw, "forecast_model"))
                    print(f"region={region_id} {task} {model} seconds={details['elapsed_seconds']:.2f}", flush=True)
                    save_json(args.output / "run_manifest.json", manifest)
                    del estimator
            del designs
    all_predictions = pd.concat(predictions, ignore_index=True)
    diagnostic = all_predictions.category == "deliberate_future_information_diagnostic"
    all_predictions[~diagnostic].to_csv(args.output / "predictions.csv.gz", index=False, compression="gzip")
    all_predictions[diagnostic].to_csv(args.output / "diagnostic_predictions.csv.gz", index=False, compression="gzip")
    metrics = summarize_metrics(all_predictions)
    metrics[metrics.category != "deliberate_future_information_diagnostic"].to_csv(args.output / "metrics.csv", index=False)
    metrics[metrics.category == "deliberate_future_information_diagnostic"].to_csv(args.output / "diagnostic_metrics.csv", index=False)
    comparisons = summarize_comparisons(all_predictions, args.output)
    comparisons[comparisons.category != "deliberate_future_information_diagnostic"].to_csv(args.output / "paired_comparisons.csv", index=False)
    comparisons[comparisons.category == "deliberate_future_information_diagnostic"].to_csv(args.output / "diagnostic_comparisons.csv", index=False)
    manifest.update({"status": "completed", "completed_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started,
                     "n_prediction_rows": len(all_predictions), "retained_origin_counts": pd.Series(split).value_counts().to_dict(),
                     "diagnostic_predictions_are_not_forecasts": True,
                     "output_sha256": {str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob("*") if p.is_file() and p.name != "run_manifest.json"}})
    save_json(args.output / "run_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ["status", "elapsed_seconds", "retained_origin_counts", "n_prediction_rows"]}), flush=True)


if __name__ == "__main__":
    main()
