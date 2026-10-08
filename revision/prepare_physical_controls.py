"""Prepare six physical controls without learning any statistic from evaluation.

Read-only inputs, a frozen provenance manifest, one temporary float32 grid field,
and bounded float64 chunks. The regional convention matches prepare_inputs.py:
monthly per-grid climatology, anomaly centering/scaling, then arithmetic region
means. Raw physical means are also saved but are not interchangeable with data.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import psutil
import xarray as xr

from prepare_inputs import region_map


FIELDS = {
    "omega_500": ("Pa s**-1", 500),
    "omega_700": ("Pa s**-1", 700),
    "geopotential_500": ("m**2 s**-2", 500),
    "temperature_700": ("K", 700),
    "surface_pressure": ("Pa", None),
    "mean_sea_level_pressure": ("Pa", None),
}
SEGMENTS = ["weatherbench2_training_1979_2018",
            "weatherbench2_evaluation_2019_2023-01-10",
            "cds_conservative_extension_2023-01-11_2025"]


def now():
    return pd.Timestamp.now(tz="UTC").isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    os.replace(temporary, path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def chunks(n, size):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def fit_statistics(raw, months, training, chunk_size=256):
    """All indexing into raw is restricted to training; ddof=0 matches legacy."""
    selected = np.flatnonzero(training)
    ng = raw.shape[1]
    sums = np.zeros((12, ng), dtype=np.float64)
    counts = np.zeros(12, dtype=np.int64)
    for sl in chunks(len(selected), chunk_size):
        ids = selected[sl]
        x = np.asarray(raw[ids], dtype=np.float64)
        check(np.isfinite(x).all(), "Nonfinite training input")
        for m in np.unique(months[ids]):
            keep = months[ids] == m
            sums[m - 1] += x[keep].sum(axis=0)
            counts[m - 1] += int(keep.sum())
    check((counts > 0).all(), "Training month missing")
    climatology = sums / counts[:, None]
    total = np.zeros(ng, dtype=np.float64)
    for sl in chunks(len(selected), chunk_size):
        ids = selected[sl]
        x = np.asarray(raw[ids], dtype=np.float64) - climatology[months[ids] - 1]
        total += x.sum(axis=0)
    mean = total / len(selected)
    squares = np.zeros(ng, dtype=np.float64)
    for sl in chunks(len(selected), chunk_size):
        ids = selected[sl]
        x = np.asarray(raw[ids], dtype=np.float64) - climatology[months[ids] - 1] - mean
        squares += np.square(x).sum(axis=0)
    original_std = np.sqrt(squares / len(selected))
    std = np.where(original_std < 1e-8, 1.0, original_std)
    return {"climatology": climatology, "mean": mean, "std": std,
            "std_before_floor": original_std, "month_counts": counts}


def inspect_sources(source_root, reference, grid_lat, grid_lon):
    """Verify every small derived file before freezing the processing plan."""
    public_path = source_root / "manifest_public.json"
    public = json.loads(public_path.read_text(encoding="utf8"))
    check(signature(public["request"]) == public["request_sha256"], "Public request signature mismatch")
    extension = source_root / "cds_extension/regridded"
    conversion_path = extension / "conversion_plan_native_chunks_v3.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf8"))
    conversion_hash = signature(conversion["definition"])
    check(conversion_hash == conversion["definition_sha256"], "Conversion definition signature mismatch")
    index_path = extension / "index.json"
    index = json.loads(index_path.read_text(encoding="utf8"))
    check(index["status"] == "completed" and index["definition_sha256"] == conversion_hash,
          "Conversion incomplete or signature mismatch")
    check(len(index["records"]) == 144, "Expected 144 conversion reports")
    indexed = {r["job"]: r for r in index["records"]}
    planned = {r["id"]: r for r in conversion["definition"]["source_inventory"]}
    check(set(planned) == set(indexed), "Conversion plan job coverage mismatch")
    public_files = sorted((source_root / "shards").glob("*.nc"))
    cds_files = sorted((extension / "monthly").glob("*.nc"))
    check(len(public_files) == 644 and len(cds_files) == 144, "Unexpected source-file counts")
    coverage = {name: np.zeros(len(reference), dtype=np.uint8) for name in FIELDS}
    rows = []
    for number, path in enumerate(public_files + cds_files):
        is_public = path.parent.name == "shards"
        receipt_path = path.with_suffix(".json") if is_public else extension / "reports" / (path.stem + ".json")
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        expected_hash = receipt["file_sha256"] if is_public else receipt["output_sha256"]
        digest = sha(path)
        check(digest == expected_hash, f"Source SHA256 mismatch: {path}")
        stat = path.stat()
        check(stat.st_size == receipt["file_bytes" if is_public else "output_bytes"], f"Source size mismatch: {path}")
        if is_public:
            check(receipt["request_sha256"] == public["request_sha256"], "Public receipt signature mismatch")
        else:
            check(receipt == indexed[path.stem], f"Report/index mismatch: {path}")
            check(receipt["status"] == "passed" and receipt["definition_sha256"] == conversion_hash,
                  f"Unvalidated conversion: {path}")
            check(receipt["source_sha256"] == planned[path.stem]["source_sha256"], "Conversion source not in frozen plan")
        with xr.open_dataset(path) as ds:
            check(np.array_equal(ds.latitude.values, grid_lat) and np.array_equal(ds.longitude.values, grid_lon),
                  f"Grid mismatch: {path}")
            t = np.asarray(ds.time.values, dtype="datetime64[ns]")
            ids = np.searchsorted(reference, t)
            check(len(t) > 0 and ids[-1] < len(reference) and np.array_equal(reference[ids], t), f"Time mismatch: {path}")
            check(np.all(np.diff(ids) == 1), f"Discontinuous source times: {path}")
            count_key = "n_timestamps" if is_public else "time_count"
            first_key, last_key = ("first", "last") if is_public else ("first_time", "last_time")
            check(len(t) == receipt[count_key] and t[0] == np.datetime64(receipt[first_key])
                  and t[-1] == np.datetime64(receipt[last_key]), f"Receipt time mismatch: {path}")
            wanted = set(FIELDS) if is_public else set(receipt["fields"])
            check(set(ds.data_vars) == wanted, f"Variable coverage mismatch: {path}")
            attr_key = "request_sha256" if is_public else "conversion_plan_sha256"
            check(ds.attrs.get(attr_key) == (public["request_sha256"] if is_public else conversion_hash),
                  f"Dataset signature mismatch: {path}")
            metadata = {}
            for name in wanted:
                da = ds[name]
                units, level = FIELDS[name]
                check(da.dims == ("time", "latitude", "longitude"), f"Dimensions mismatch: {path}/{name}")
                check(da.attrs.get("units") == units, f"Units mismatch: {path}/{name}")
                check(da.attrs.get("pressure_level_hpa") == level, f"Pressure level mismatch: {path}/{name}")
                check(not coverage[name][ids].any(), f"Duplicated source times: {name}")
                coverage[name][ids] += 1
                metadata[name] = {"units": units, "pressure_level_hpa": level, "shape": list(da.shape)}
            check((t[-1] <= np.datetime64("2023-01-10T18")) if is_public
                  else (t[0] >= np.datetime64("2023-01-11T00")), "Unexpected source boundary")
        rows.append({"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                     "sha256": digest, "receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path),
                     "source": "weatherbench2" if is_public else "cds_conservative_extension",
                     "first": str(t[0]), "last": str(t[-1]), "n_timestamps": len(t),
                     "start_index": int(ids[0]), "stop_index_exclusive": int(ids[-1]) + 1, "fields": metadata})
        if (number + 1) % 100 == 0:
            print(json.dumps({"stage": "input_provenance", "verified": number + 1, "total": 788}), flush=True)
    check(all(np.all(c == 1) for c in coverage.values()), "Incomplete per-field coverage")
    return {"verified_utc": now(), "sources": rows, "input_bytes": sum(r["bytes"] for r in rows),
            "all_derived_file_hashes_recomputed": True, "all_coordinate_metadata_checked": True,
            "all_times_exactly_once_per_field": True,
            "public_request_sha256": public["request_sha256"], "conversion_definition_sha256": conversion_hash,
            "provenance_files": {str(p): sha(p) for p in [public_path, conversion_path, index_path]}}


def validate_field(raw, months, training, mapping, stats, regional, physical, seed):
    """Independent direct reductions and destructive evaluation perturbation."""
    rng = np.random.default_rng(seed)
    selected_cells = np.sort(rng.choice(raw.shape[1], 8, replace=False))
    sample = np.asarray(raw[:, selected_cells], dtype=np.float64)
    direct_clim = np.stack([sample[training & (months == m)].mean(axis=0) for m in range(1, 13)])
    anomalies = sample - direct_clim[months - 1]
    direct_mean = anomalies[training].mean(axis=0)
    direct_std = anomalies[training].std(axis=0)
    direct_std = np.where(direct_std < 1e-8, 1., direct_std)
    discrepancies = {}
    for key, direct in [("climatology", direct_clim), ("mean", direct_mean), ("std", direct_std)]:
        stored = stats[key][..., selected_cells]
        check(np.allclose(stored, direct, rtol=2e-11, atol=2e-10), f"Direct grid fit disagrees: {key}")
        discrepancies[key] = float(np.max(np.abs(stored - direct)))
    # NaN and enormous changed evaluation values are never touched by the fitter.
    baseline = fit_statistics(sample, months, training)
    sample[~training] = np.nan
    perturbed = fit_statistics(sample, months, training)
    isolation = all(np.array_equal(baseline[k], perturbed[k]) for k in baseline)
    check(isolation, "Evaluation perturbation changed training fit")
    sample[~training] = 1e12
    check(all(np.array_equal(baseline[k], v) for k, v in fit_statistics(sample, months, training).items()),
          "Large evaluation values changed training fit")
    sample[training, 0] += .375
    changed = fit_statistics(sample, months, training)
    check(np.max(np.abs(changed["climatology"][:, 0] - baseline["climatology"][:, 0])) > .37,
          "Isolation check did not respond to training perturbation")
    region_ids = np.sort(rng.choice(66, 3, replace=False))
    # Direct complete-region fit from raw source grids does not reuse saved stats.
    time_ids = np.unique(np.concatenate(([0, 58439, 58440, 64323, 64324, len(raw)-1],
                                        rng.choice(len(raw), 18, replace=False))))
    max_z, max_physical = 0., 0.
    for r in region_ids:
        cells = np.flatnonzero(mapping == r)
        x = np.asarray(raw[:, cells], dtype=np.float64)
        clim = np.stack([x[training & (months == m)].mean(axis=0) for m in range(1, 13)])
        a = x - clim[months - 1]
        mu, sd = a[training].mean(axis=0), a[training].std(axis=0)
        sd = np.where(sd < 1e-8, 1., sd)
        direct = ((a[time_ids] - mu) / sd).mean(axis=1)
        direct_p = x[time_ids].mean(axis=1)
        z_error = float(np.max(np.abs(direct - regional[time_ids, r])))
        p_error = float(np.max(np.abs(direct_p - physical[time_ids, r])))
        check(z_error < 1e-9 and p_error < 1e-8, "Independent complete-region calculation disagrees")
        max_z, max_physical = max(max_z, z_error), max(max_physical, p_error)
    train_region_mean = np.max(np.abs(regional[training].mean(axis=0)))
    check(train_region_mean < 1e-10, "Training regional means are not centered")
    return {"status": "passed", "seed": seed, "random_grid_cells": selected_cells.tolist(),
            "direct_grid_fit_max_abs_errors": discrepancies,
            "evaluation_nan_and_1e12_perturbation_parameters_bitwise_unchanged": isolation,
            "training_perturbation_positive_control": "passed",
            "random_regions": region_ids.tolist(), "direct_region_time_indices": time_ids.tolist(),
            "direct_region_standardized_max_abs_error": max_z,
            "direct_region_physical_max_abs_error": max_physical,
            "training_region_mean_max_abs": float(train_region_mean)}


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--source-root", type=Path, default=root / "supplementary_data/era5_controls")
    parser.add_argument("--reference-root", type=Path, default=root / "revision_outputs/inputs")
    parser.add_argument("--output", type=Path, default=root / "revision_outputs/physical_controls_inputs")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    check(not (output / "data_manifest.json").exists(), "Completed output exists; choose a new output directory")
    check(not (output / "preprocessing_plan.json").exists(), "Frozen plan exists; choose a new output directory")
    start = time.perf_counter()
    with np.load(args.reference_root / "region_trainfit.npz", allow_pickle=False) as ref:
        timestamps = ref["timestamps"].astype("datetime64[ns]")
        rlat, rlon, nodes = ref["lat"], ref["lon"], ref["node_ids"]
    with np.load(args.reference_root / "trainfit_parameters.npz", allow_pickle=False) as ref:
        grid_lat, grid_lon, mapping = ref["grid_lat"], ref["grid_lon"], ref["node_to_region"]
    rebuilt_mapping, rebuilt_lat, rebuilt_lon = region_map(grid_lat, grid_lon)
    check(np.array_equal(mapping, rebuilt_mapping) and np.array_equal(rlat, rebuilt_lat)
          and np.array_equal(rlon, rebuilt_lon), "Region convention not identical to original")
    expected = np.arange(np.datetime64("1979-01-01T00"), np.datetime64("2026-01-01T00"), np.timedelta64(6, "h"))
    check(np.array_equal(timestamps, expected) and len(timestamps) == 68668, "Reference time axis unexpected")
    training = timestamps < np.datetime64("2019-01-01")
    months = pd.DatetimeIndex(timestamps).month.to_numpy()
    check(training.sum() == 58440 and len(rlat) == 66, "Unexpected training or region counts")
    source_segment = np.where(training, 0, np.where(timestamps < np.datetime64("2023-01-11"), 1, 2)).astype("uint8")
    refs = {str(args.reference_root / name): sha(args.reference_root / name)
            for name in ["region_trainfit.npz", "trainfit_parameters.npz", "data_manifest.json"]}
    inventory = inspect_sources(args.source_root, timestamps, grid_lat, grid_lon)
    save(output / "source_manifest.json", inventory)
    definition = {"schema_version": 1, "script_sha256": sha(__file__), "reference_files": refs,
                  "source_manifest_sha256": sha(output / "source_manifest.json"),
                  "variables": {k: {"units": v[0], "pressure_level_hpa": v[1]} for k, v in FIELDS.items()},
                  "fit_period": ["1979-01-01T00", "2018-12-31T18"],
                  "evaluation_period": ["2019-01-01T00", "2025-12-31T18"],
                  "method": "Per-grid training monthly mean removal; training anomaly mean and population std; arithmetic mean in original 66 regions",
                  "std_floor": "if training std < 1e-8, use 1.0", "chunk_size": 256,
                  "missing_policy": "fail on any nonfinite; no imputation or interpolation",
                  "same_source_evaluation": ["2019-01-01T00", "2023-01-10T18"],
                  "extension_sensitivity": ["2023-01-11T00", "2025-12-31T18"],
                  "segment_homogeneity": "Not established; source segment labels must be retained in downstream evaluation",
                  "terrain_diagnostic": "For surface-pressure coarse grid only, report region and source-segment mean/maximum cell fraction with sp < 50000, 70000, 85000 Pa; no masking or interpolation",
                  "raw_grid_scratch_bytes_max": len(timestamps) * len(mapping) * 4,
                  "random_validation_seed": 20260930, "expected_shape": [68668, 66, 6]}
    plan = {"frozen_utc": now(), "definition_sha256": signature(definition), "definition": definition}
    plan_path = output / "preprocessing_plan.json"
    check(not plan_path.exists(), "Frozen plan exists; use a fresh output directory")
    save(plan_path, plan)
    print(json.dumps({"stage": "plan_frozen", "definition_sha256": plan["definition_sha256"],
                      "input_bytes": inventory["input_bytes"]}), flush=True)
    data = np.empty((len(timestamps), 66, len(FIELDS)), dtype=np.float64)
    physical = np.empty_like(data)
    parameters, fields, validation = {}, {}, {}
    scratch_path = output / "scratch_one_grid_field.npy"
    check(not scratch_path.exists(), "Pre-existing scratch file; refuse to overwrite")
    for index, name in enumerate(FIELDS):
        raw = np.lib.format.open_memmap(scratch_path, mode="w+", dtype=np.float32,
                                       shape=(len(timestamps), len(mapping)))
        minimum, maximum, seen = np.inf, -np.inf, 0
        for row in inventory["sources"]:
            if name not in row["fields"]:
                continue
            path = Path(row["path"])
            check(path.stat().st_size == row["bytes"] and path.stat().st_mtime_ns == row["mtime_ns"],
                  f"Source changed after plan freeze: {path}")
            with xr.open_dataset(path) as ds:
                a = np.asarray(ds[name].values, dtype=np.float32).reshape(row["n_timestamps"], -1)
                check(np.isfinite(a).all(), f"Nonfinite source values: {path}/{name}")
                raw[row["start_index"]:row["stop_index_exclusive"]] = a
                minimum, maximum = min(minimum, float(a.min())), max(maximum, float(a.max()))
                seen += a.size
        raw.flush()
        stats = fit_statistics(raw, months, training)
        pressure_fractions = np.empty((len(timestamps), 66, 3), dtype=np.float32) if name == "surface_pressure" else None
        for sl in chunks(len(timestamps), 256):
            x = np.asarray(raw[sl], dtype=np.float64)
            z = (x - stats["climatology"][months[sl] - 1] - stats["mean"]) / stats["std"]
            for r in range(66):
                keep = mapping == r
                data[sl, r, index] = z[:, keep].mean(axis=1)
                physical[sl, r, index] = x[:, keep].mean(axis=1)
                if pressure_fractions is not None:
                    for j, threshold in enumerate([50000., 70000., 85000.]):
                        pressure_fractions[sl, r, j] = (x[:, keep] < threshold).mean(axis=1)
        if pressure_fractions is not None:
            terrain_rows = []
            for segment, label in enumerate(SEGMENTS):
                for r in range(66):
                    for j, level in enumerate([500, 700, 850]):
                        values = pressure_fractions[source_segment == segment, r, j]
                        terrain_rows.append({"source_segment": segment, "source_segment_label": label,
                                             "region": r, "latitude": float(rlat[r]), "longitude": float(rlon[r]),
                                             "pressure_level_hpa": level, "n_times": len(values),
                                             "mean_coarse_grid_cell_fraction_sp_below_level": float(values.mean(dtype=np.float64)),
                                             "max_coarse_grid_cell_fraction_sp_below_level": float(values.max()),
                                             "fraction_times_with_any_coarse_grid_sp_below_level": float((values > 0).mean())})
            pd.DataFrame(terrain_rows).to_csv(output / "coarse_surface_pressure_terrain_diagnostic.csv", index=False)
            save(output / "coarse_surface_pressure_terrain_diagnostic.json", {
                "scope": "Diagnostic only, no value masking or change to preprocessing definition",
                "limitation": "sp is already conservatively coarsened to 64x32; this is not a native 0.25-degree terrain mask, does not establish above-ground pressure-level validity, and does not resolve support differences with original 850-hPa main variables",
                "csv": "coarse_surface_pressure_terrain_diagnostic.csv", "rows": len(terrain_rows)})
            del pressure_fractions
        validation[name] = validate_field(raw, months, training, mapping, stats,
                                          data[:, :, index], physical[:, :, index], 20260930 + index)
        parameters.update({name + "__" + k: v for k, v in stats.items()})
        fields[name] = {"minimum": minimum, "maximum": maximum, "values_checked": seen,
                        "nonfinite": 0, "units": FIELDS[name][0], "pressure_level_hpa": FIELDS[name][1],
                        "grid_cells_below_std_floor": int((stats["std_before_floor"] < 1e-8).sum())}
        del raw
        gc.collect()
        scratch_path.unlink()  # Only our single, known scratch file within output.
        state = {"status": "running", "updated_utc": now(), "fields_completed": index + 1,
                 "total_fields": 6, "last_field": name, "elapsed_seconds": time.perf_counter() - start}
        save(output / "status.json", state)
        print(json.dumps(state), flush=True)
    check(np.isfinite(data).all() and np.isfinite(physical).all(), "Nonfinite final outputs")
    common = {"lat": rlat, "lon": rlon, "node_ids": nodes, "timestamps": timestamps,
              "variable_names": np.array(list(FIELDS)), "variable_units": np.array([v[0] for v in FIELDS.values()]),
              "source_segment": source_segment, "source_segment_names": np.array(SEGMENTS), "training_mask": training}
    np.savez_compressed(output / "region_controls_trainfit.npz", data=data, physical=physical, **common)
    np.savez_compressed(output / "trainfit_parameters.npz", **parameters, node_to_region=mapping,
                        grid_lat=grid_lat, grid_lon=grid_lon)
    with np.load(output / "region_controls_trainfit.npz", allow_pickle=False) as saved:
        check(np.array_equal(saved["data"], data) and np.array_equal(saved["physical"], physical), "NPZ data roundtrip mismatch")
        check(all(np.array_equal(saved[k], v) for k, v in common.items()), "NPZ metadata roundtrip mismatch")
    validation_report = {"status": "passed", "completed_utc": now(), "fields": validation,
                         "output_roundtrip_all_values_exact": True,
                         "all_source_values_rechecked_finite": True,
                         "cross_source_homogeneity_tested": False,
                         "validation_scope": "Input bytes/metadata/coverage, all-value finite checks, train-only statistics and direct region calculations; no fitted causal experiment"}
    save(output / "validation.json", validation_report)
    manifest = {"status": "completed", "completed_utc": now(), "definition_sha256": plan["definition_sha256"],
                "shape": list(data.shape), "fields": fields,
                "train_n": int(training.sum()), "test_n": int((~training).sum()),
                "segment_counts": dict(zip(SEGMENTS, [int((source_segment == i).sum()) for i in range(3)])),
                "elapsed_seconds": time.perf_counter() - start,
                "peak_working_set_bytes": int(getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)),
                "scratch_remaining_bytes": 0, "input_bytes": inventory["input_bytes"],
                "python": platform.python_version(), "numpy": np.__version__, "xarray": xr.__version__,
                "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                            for p in output.glob("*.npz")},
                "source_manifest_sha256": sha(output / "source_manifest.json"),
                "validation_sha256": sha(output / "validation.json"),
                "terrain_diagnostic_sha256": sha(output / "coarse_surface_pressure_terrain_diagnostic.csv"),
                "limitations": ["WB2 and CDS extension are distinct acquisition/regridding segments; homogeneity has not been demonstrated.",
                                "Pressure-level values below terrain are retained as supplied and are not an above-ground-only mask.",
                                "These are standardized control inputs, not completed causal identification or effect validation."]}
    save(output / "data_manifest.json", manifest)
    save(output / "status.json", {"status": "completed", "updated_utc": now(), "fields_completed": 6,
                                  "total_fields": 6, "manifest": "data_manifest.json"})
    print(json.dumps({k: manifest[k] for k in ["status", "shape", "elapsed_seconds", "peak_working_set_bytes", "segment_counts"]}), flush=True)


if __name__ == "__main__":
    main()
