"""Bounded, resumable conservative remapping of the existing CDS control files.

No retrieval, model fitting, standardization, source overwrite, or source deletion.
The separable spherical overlap method follows WeatherBench2's published code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
from scipy.sparse import csr_matrix
import xarray as xr


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent / "supplementary_data/era5_controls"
SOURCE = ROOT / "cds_extension"
OUTPUT = SOURCE / "regridded"
FIELDS = {
    "omega_500_700": [("w", "omega_500", 500, "Pa s**-1"), ("w", "omega_700", 700, "Pa s**-1")],
    "geopotential_500": [("z", "geopotential_500", 500, "m**2 s**-2")],
    "temperature_700": [("t", "temperature_700", 700, "K")],
    "surface_pressure_msl": [("sp", "surface_pressure", None, "Pa"), ("msl", "mean_sea_level_pressure", None, "Pa")],
}
URLS = [
    "https://weatherbench2.readthedocs.io/en/latest/data-guide.html",
    "https://raw.githubusercontent.com/google-research/weatherbench2/main/weatherbench2/regridding.py",
    "https://raw.githubusercontent.com/google-research/weatherbench2/main/scripts/regrid.py",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    partial.replace(path)


def latitude_bounds(c):
    return np.r_[-90., (c[:-1] + c[1:]) / 2, 90.]


def longitude_bounds(c):
    previous = np.r_[c[-1] - 360., c[:-1]]
    following = np.r_[c[1:], c[0] + 360.]
    return (previous + c) / 2, (following + c) / 2


def geometry(slat, slon, tlat, tlon):
    sb, tb = latitude_bounds(slat), latitude_bounds(tlat)
    lower = np.maximum(tb[:-1, None], sb[None, :-1])
    upper = np.minimum(tb[1:, None], sb[None, 1:])
    lat_overlap = np.where(upper > lower, np.sin(np.deg2rad(upper)) - np.sin(np.deg2rad(lower)), 0.)
    slo, shi = longitude_bounds(slon)
    tlo, thi = longitude_bounds(tlon)
    lon_overlap = np.zeros((len(tlon), len(slon)), dtype=np.float64)
    for shift in [-360., 0., 360.]:
        lon_overlap += np.maximum(0., np.minimum(thi[:, None], shi[None, :] + shift) - np.maximum(tlo[:, None], slo[None, :] + shift))
    sa = np.diff(np.sin(np.deg2rad(sb)))[:, None] * np.deg2rad(shi - slo)[None, :]
    ta = np.diff(np.sin(np.deg2rad(tb)))[:, None] * np.deg2rad(thi - tlo)[None, :]
    wl = csr_matrix(lat_overlap / lat_overlap.sum(axis=1, keepdims=True))
    wo = csr_matrix(lon_overlap / lon_overlap.sum(axis=1, keepdims=True))
    return wl, wo, sa, ta


def remap(values, wl, wo):
    return wl.dot(wo.dot(values.T).T)


def geometry_tests(slat, slon, tlat, tlon, wl, wo, sa, ta):
    np.testing.assert_allclose(sa.sum(), 4 * np.pi, rtol=0, atol=1e-13)
    np.testing.assert_allclose(ta.sum(), 4 * np.pi, rtol=0, atol=1e-13)
    np.testing.assert_allclose(wl.sum(axis=1), 1, rtol=0, atol=1e-13)
    np.testing.assert_allclose(wo.sum(axis=1), 1, rtol=0, atol=1e-13)
    constant = remap(np.ones(sa.shape), wl, wo)
    np.testing.assert_allclose(constant, 1, rtol=0, atol=1e-13)
    impulse = np.zeros(sa.shape)
    impulse[0, -1] = 1  # South polar cap, just west of Greenwich.
    mapped = remap(impulse, wl, wo)
    assert mapped[0, 0] > 0 and np.count_nonzero(mapped) == 1
    np.testing.assert_allclose((mapped * ta).sum(), (impulse * sa).sum(), rtol=1e-12, atol=1e-16)
    # Longitude phase and north/south ordering affect this asymmetric field.
    field = slat[:, None] + np.cos(np.deg2rad(slon))[None, :] * 3.
    descending_source = field[::-1]
    restored = descending_source[np.argsort(slat[::-1])]
    np.testing.assert_array_equal(restored, field)
    output = remap(restored, wl, wo)
    assert output[-1].mean() > output[0].mean()
    assert output[:, 0].mean() > output[:, len(tlon) // 2].mean()
    np.testing.assert_allclose((output * ta).sum(), (field * sa).sum(), rtol=0, atol=1e-12)
    return {"status": "passed", "tests": ["row sums", "constant preservation", "sphere area 4*pi", "south polar cell and longitude wrap impulse", "latitude direction", "asymmetric longitude phase", "global integral"], "latitude_nonzero_weights": int(wl.nnz), "longitude_nonzero_weights": int(wo.nnz)}


def prepared():
    reference = sorted((ROOT / "shards").glob("*.nc"))[0]
    with xr.open_dataset(reference) as ds:
        tlat, tlon = ds.latitude.values.copy(), ds.longitude.values.copy()
        attributes = {name: dict(ds[name].attrs) for name in ds.data_vars}
    slat, slon = np.arange(-90., 90.001, .25), np.arange(0., 360., .25)
    wl, wo, sa, ta = geometry(slat, slon, tlat, tlon)
    tests = geometry_tests(slat, slon, tlat, tlon, wl, wo, sa, ta)
    request_path = SOURCE / "request_plan.json"
    jobs = json.loads(request_path.read_text(encoding="utf8"))["jobs"]
    inventory = []
    for job in jobs:
        receipt = SOURCE / "receipts" / (job["id"] + ".json")
        record = json.loads(receipt.read_text(encoding="utf8"))
        inventory.append({"id": job["id"], "source_sha256": record["sha256"], "source_bytes": record["bytes"], "receipt_sha256": sha(receipt)})
    old_script = REPO.parent.parent / "vipuser/Data/download_era5_cds_5p625_2023_2025.py"
    definition = {
        "schema_version": 1, "script_sha256": sha(__file__), "request_plan_sha256": sha(request_path),
        "source_inventory": inventory, "source_grid": {"latitude": slat.tolist(), "longitude": slon.tolist()},
        "target_grid": {"latitude": tlat.tolist(), "longitude": tlon.tolist()},
        "target_reference": str(reference), "target_reference_sha256": sha(reference),
        "method": "First-order separable spherical cell-overlap conservative mean; midpoint latitude boundaries capped at -90,+90; periodic longitude midpoint boundaries; float64 accumulation, float32 output.",
        "method_primary_sources": URLS,
        "official_source_checked_date": "2026-09-30",
        "official_code_provenance_limit": "Current public main inspected; the historical WeatherBench2 archive build commit is not identified. Method agreement is not a bitwise archive reproduction claim.",
        "weights_sha256": {"latitude": hashlib.sha256(wl.toarray().tobytes()).hexdigest(), "longitude": hashlib.sha256(wo.toarray().tobytes()).hexdigest()},
        "versions": {x: importlib.metadata.version(x) for x in ["numpy", "scipy", "xarray", "netCDF4", "pandas", "psutil"]},
        "chunk_timestamps": "Native NetCDF time chunk, capped at 64; align reads to avoid repeated decompression", "raw_float32_output_bytes": 213516288, "scratch_limit_bytes": 20000000000,
        "missing_policy": "Fail on any nonfinite source value or finite metadata missing sentinel after CF decoding; no imputation or silent partial-area renormalization. Original raw files retained.",
        "validation": tests,
        "interpretation_limits": [
            "No temporal overlap across the WeatherBench2/CDS control source boundary; no measured cross-provider seam error or bias correction.",
            "Existing 2023-2025 primary t850/q850/u850/v850/tcc files were directly requested at CDS 5.625-degree grid, without documented WeatherBench2 conservative remapping; approximate coordinate agreement does not establish value-processing homogeneity.",
            "This conversion supplies control fields and numerical checks only; no merge into a model, standardization, causal validation, or experiment is performed.",
        ],
        "existing_primary_late_download_script": {"path": str(old_script), "sha256": sha(old_script)},
    }
    plan_path = OUTPUT / "conversion_plan_native_chunks_v3.json"
    signature = object_sha(definition)
    if plan_path.exists():
        saved = json.loads(plan_path.read_text(encoding="utf8"))
        if saved["definition_sha256"] != signature:
            raise ValueError("Frozen conversion plan differs; do not reuse outputs under a changed method/source.")
    else:
        save(plan_path, {"frozen_utc": now(), "definition_sha256": signature, "definition": definition})
    return jobs, signature, (slat, slon, tlat, tlon, wl, wo, sa, ta), attributes


def process(job, signature, geo, public_attributes):
    started = time.perf_counter()
    slat, slon, tlat, tlon, wl, wo, sa, ta = geo
    source = SOURCE / "monthly" / (job["id"] + ".nc")
    receipt_path = SOURCE / "receipts" / (job["id"] + ".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    target = OUTPUT / "monthly" / source.name
    report_path = OUTPUT / "reports" / (job["id"] + ".json")
    if target.exists() and report_path.exists():
        record = json.loads(report_path.read_text(encoding="utf8"))
        if record.get("status") == "passed" and record.get("definition_sha256") == signature and record.get("output_sha256") == sha(target):
            if source.stat().st_size == record["source_bytes"] and source.stat().st_mtime_ns == record["source_mtime_ns"]:
                return record
    before = source.stat()
    actual_sha = sha(source)
    if actual_sha != receipt["sha256"] or source.stat().st_size != receipt["bytes"]:
        raise ValueError("Source SHA256 or byte count differs from acquisition receipt: " + job["id"])
    source_hash_seconds = time.perf_counter() - started
    group = next(key for key in FIELDS if job["id"].startswith(key + "_"))
    arrays, stats, attrs = {}, {}, {}
    with xr.open_dataset(source, engine="netcdf4") as ds:
        expected_times = pd.date_range(job["first_time"], job["last_time"], freq="6h")
        if not pd.DatetimeIndex(ds.valid_time.values).equals(expected_times):
            raise ValueError("Time mismatch: " + job["id"])
        times = ds.valid_time.values.copy()
        lat_order = np.argsort(ds.latitude.values)
        lon_order = np.argsort(ds.longitude.values % 360)
        np.testing.assert_array_equal(ds.latitude.values[lat_order], slat)
        np.testing.assert_array_equal((ds.longitude.values % 360)[lon_order], slon)
        if job["request"].get("pressure_level"):
            np.testing.assert_array_equal(np.sort(ds.pressure_level.values), np.sort(np.asarray(job["request"]["pressure_level"], dtype=float)))
        for raw, name, level, units in FIELDS[group]:
            da = ds[raw]
            if da.attrs.get("units") != units:
                raise ValueError("Units mismatch: " + name)
            if level is not None:
                da = da.sel(pressure_level=level)
            da = da.transpose("valid_time", "latitude", "longitude")
            sentinels = []
            for metadata in [ds[raw].attrs, ds[raw].encoding]:
                for key in ["GRIB_missingValue", "missing_value", "_FillValue"]:
                    if key in metadata:
                        for value in np.atleast_1d(metadata[key]):
                            if np.isfinite(value):
                                sentinels.append(float(value))
            sentinels = sorted(set(sentinels))
            out = np.empty((len(times), len(tlat), len(tlon)), dtype=np.float32)
            minimum, maximum, total, count = float("inf"), float("-inf"), 0., 0
            max64, max32, max_abs32 = 0., 0., 0.
            native_chunks = ds[raw].encoding.get("chunksizes")
            step = min(64, int(native_chunks[ds[raw].dims.index("valid_time")])) if native_chunks else 8
            for lo in range(0, len(times), step):
                block = da.isel(valid_time=slice(lo, lo + step)).values
                # A reversed view avoids another full native-grid chunk copy.
                if np.array_equal(lat_order, np.arange(len(slat))[::-1]):
                    block = block[:, ::-1, :]
                elif not np.array_equal(lat_order, np.arange(len(slat))):
                    block = block[:, lat_order, :]
                if not np.array_equal(lon_order, np.arange(len(slon))):
                    block = block[:, :, lon_order]
                nonfinite = int(np.count_nonzero(~np.isfinite(block)))
                if nonfinite:
                    raise ValueError(f"Nonfinite decoded values: {job['id']} {name} block {lo}: {nonfinite}; no output accepted")
                sentinel_count = sum(int(np.count_nonzero(block == np.asarray(value, dtype=block.dtype))) for value in sentinels)
                if sentinel_count:
                    raise ValueError(f"Finite metadata missing sentinel: {job['id']} {name} block {lo}: {sentinel_count}; no output accepted")
                minimum, maximum = min(minimum, float(block.min())), max(maximum, float(block.max()))
                total += float(block.sum(dtype=np.float64)); count += block.size
                for offset, v32 in enumerate(block):
                    v = v32.astype(np.float64)
                    mapped = remap(v, wl, wo)
                    rounded = mapped.astype(np.float32)
                    source_integral = float((v * sa).sum())
                    scale = max(float((abs(v) * sa).sum()), 1e-20)
                    error64 = abs(float((mapped * ta).sum()) - source_integral) / scale
                    error_abs32 = abs(float((rounded.astype(np.float64) * ta).sum()) - source_integral)
                    error32 = error_abs32 / scale
                    if error64 > 2e-12 or error32 > 2e-7:
                        raise ValueError("Area integral preservation failed: " + name)
                    if not np.isfinite(rounded).all() or float(mapped.min()) < float(v.min()) - 1e-8 or float(mapped.max()) > float(v.max()) + 1e-8:
                        raise ValueError("Finite/convex-range check failed: " + name)
                    max64, max32, max_abs32 = max(max64, error64), max(max32, error32), max(max_abs32, error_abs32)
                    out[lo + offset] = rounded
            arrays[name] = out
            stats[name] = {"source_count": int(count), "source_nonfinite": 0, "source_finite_missing_sentinel_count": 0, "finite_metadata_missing_sentinels_checked": sentinels, "source_minimum": minimum, "source_maximum": maximum, "source_mean": total/count,
                           "output_shape": list(out.shape), "output_minimum": float(out.min()), "output_maximum": float(out.max()), "output_nonfinite": 0,
                           "max_float64_integral_error_over_source_L1_integral": max64, "max_float32_integral_error_over_source_L1_integral": max32,
                           "max_float32_integral_absolute_error_unit_sphere": max_abs32, "units": units, "pressure_level_hpa": level, "read_time_chunk": step}
            attrs[name] = {k: v for k, v in public_attributes[name].items() if k != "source_chunk_key"}
            attrs[name].update(source_variable=raw, source_file=source.name, cell_methods="latitude: longitude: mean (spherical conservative remapping)")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during hashing or decoding: " + job["id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.nc")
    dataset = xr.Dataset({name: (("time", "latitude", "longitude"), values, attrs[name]) for name, values in arrays.items()},
                         coords={"time": times, "latitude": tlat, "longitude": tlon},
                         attrs={"conversion_plan_sha256": signature, "source_sha256": actual_sha, "source_job_id": job["id"], "method": "separable spherical first-order conservative remapping", "created_utc": now()})
    dataset.to_netcdf(partial, engine="netcdf4", encoding={name: {"dtype": "float32", "zlib": True, "complevel": 1, "shuffle": True} for name in arrays})
    dataset.close()
    with xr.open_dataset(partial) as verify:
        np.testing.assert_array_equal(verify.time.values, times)
        np.testing.assert_array_equal(verify.latitude.values, tlat)
        np.testing.assert_array_equal(verify.longitude.values, tlon)
        for name, values in arrays.items():
            np.testing.assert_array_equal(verify[name].values, values)
    partial.replace(target)
    record = {"status": "passed", "job": job["id"], "completed_utc": now(), "definition_sha256": signature,
              "source_file": str(source), "source_sha256": actual_sha, "source_hash_matches_acquisition_receipt": True,
              "source_bytes": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns, "source_hash_seconds": source_hash_seconds,
              "output_file": str(target), "output_bytes": target.stat().st_size, "output_sha256": sha(target),
              "time_count": len(times), "first_time": str(times[0]), "last_time": str(times[-1]), "fields": stats,
              "elapsed_seconds": time.perf_counter() - started, "process_peak_working_set_bytes": psutil.Process().memory_info().peak_wset,
              "roundtrip": "all output values and all coordinates identical", "validation_scope": "All scientific source values decoded and finite; all source bytes SHA256-verified; all time planes conservative-integral checked; all output values roundtrip checked. No model or cross-source seam validation."}
    save(report_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit-jobs", type=int)
    args = parser.parse_args()
    jobs, signature, geo, attributes = prepared()
    if not args.run:
        print(json.dumps({"status": "plan_frozen_geometry_tests_passed", "definition_sha256": signature, "job_count": len(jobs)}), flush=True)
        return
    started, records = time.perf_counter(), []
    selected = jobs[:args.limit_jobs] if args.limit_jobs else jobs
    try:
        for job in selected:
            record = process(job, signature, geo, attributes)
            records.append(record)
            state = {"status": "running", "updated_utc": now(), "definition_sha256": signature, "completed_jobs": len(records), "total_jobs": len(jobs),
                     "active_or_last_job": job["id"], "output_bytes": sum(r["output_bytes"] for r in records), "elapsed_seconds_this_run": time.perf_counter()-started}
            save(OUTPUT / "status.json", state)
            print(json.dumps({**state, "last_job_seconds": record["elapsed_seconds"], "peak_working_set_bytes": record["process_peak_working_set_bytes"]}), flush=True)
        state["status"] = "completed" if len(records) == len(jobs) else "limited_batch_completed"
        state["validation_scope"] = "Numerical conversion and file validation; no fitted experiment or cross-source seam calibration."
        save(OUTPUT / "status.json", state)
        save(OUTPUT / "index.json", {"definition_sha256": signature, "status": state["status"], "records": records})
    except Exception as exc:
        save(OUTPUT / "status.json", {"status": "failed_review_required", "updated_utc": now(), "completed_jobs": len(records), "active_job": job["id"], "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
