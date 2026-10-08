"""Build train-fitted regional series directly from the three ERA5 source segments.

No temporal interpolation is performed. Missing/nonfinite inputs stop the run.
Monthly means and anomaly scaling are learned only on 1979--2018.
"""
from __future__ import annotations

import argparse
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


SEGMENTS = [
    "weatherbench2_era5_6h_64x32_850hPa_1979_2018",
    "weatherbench2_era5_6h_64x32_850hPa_2019_2023-01-10",
    "era5_cds_6h_64x32_850hPa_2023-01-11_2025",
]
FILES = {
    "temperature": "temperature_850.nc",
    "humidity": "specific_humidity_850.nc",
    "u": "u_component_of_wind_850.nc",
    "v": "v_component_of_wind_850.nc",
    "cloud_cover": "total_cloud_cover.nc",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def region_map(lat, lon):
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    la, lo = lat2.ravel(), ((lon2.ravel() + 180) % 360) - 180
    a = np.clip(np.digitize(np.clip(la, -89.999, 89.999), np.linspace(-90, 90, 7)) - 1, 0, 5)
    b = np.clip(np.digitize(lo, np.linspace(-180, 180, 12)) - 1, 0, 10)
    _, mapping = np.unique(a * 11 + b, return_inverse=True)
    rlat = np.array([la[mapping == r].mean() for r in np.unique(mapping)])
    rlon = np.array([np.degrees(np.arctan2(np.sin(np.radians(lo[mapping == r])).mean(), np.cos(np.radians(lo[mapping == r])).mean())) for r in np.unique(mapping)])
    return mapping, rlat, rlon


def fit_transform_aggregate(raw, months, training, mapping):
    """Fit statistics before transforming; changes to test data cannot alter fit."""
    x = np.array(raw, dtype=np.float64, copy=True)
    means = np.stack([x[training & (months == m)].mean(axis=0) for m in range(1, 13)])
    x -= means[months - 1]
    mu = x[training].mean(axis=0)
    sd = x[training].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    x -= mu
    x /= sd
    regional = np.column_stack([x[:, mapping == r].mean(axis=1) for r in np.unique(mapping)])
    return regional, {"climatology": means, "mean": mu, "std": sd}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--extension-dir", type=Path, help="Directory replacing the 2023-01-11--2025 CDS segment, e.g. the native-conservative route.")
    p.add_argument("--full-record-output", type=Path, help="Also save the full-record-climatology series in the legacy region_series.npz layout.")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = {"started_utc": pd.Timestamp.now(tz="UTC").isoformat(), "python": platform.python_version(), "numpy": np.__version__, "xarray": xr.__version__, "fit_period": ["1979-01-01", "2018-12-31"], "evaluation_period": ["2019-01-01", "2025-12-31"], "missing_policy": "fail_on_nonfinite_no_imputation", "spatial_mean": "arithmetic mean of grid-standardized anomalies in original 6x11 bins", "coordinate_tolerance_degrees": 0.001, "sources": [], "fields": {}, "code_sha256": sha256(__file__)}
    reference_lat = reference_lon = reference_times = None
    def read_field(name):
        nonlocal reference_lat, reference_lon, reference_times
        parts, times = [], []
        for segment in SEGMENTS:
            path = args.data_root / segment / FILES[name]
            if args.extension_dir is not None and segment == SEGMENTS[-1]:
                path = args.extension_dir / FILES[name]
            with xr.open_dataset(path) as ds:
                key = next(k for k in ds.data_vars if {"time", "latitude", "longitude"}.issubset(ds[k].dims))
                da = ds[key].transpose("time", "latitude", "longitude")
                lat, lon = ds.latitude.values, ds.longitude.values
                if reference_lat is None:
                    reference_lat, reference_lon = lat.copy(), lon.copy()
                if not (np.allclose(lat, reference_lat, atol=.001, rtol=0) and np.allclose(lon, reference_lon, atol=.001, rtol=0)):
                    raise ValueError(f"Grid mismatch: {path}")
                t = np.asarray(ds.time.values, dtype="datetime64[ns]")
                values = np.asarray(da.values, dtype=np.float32).reshape(len(t), -1)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite field {name} in {segment}")
                parts.append(values)
                times.append(t)
                manifest["sources"].append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "variable": key, "units": str(da.attrs.get("units", "unspecified")), "n_timestamps": len(t), "first": str(t[0]), "last": str(t[-1]), "latitude_max_abs_difference": float(np.max(np.abs(lat-reference_lat)))})
        timestamps = np.concatenate(times)
        if reference_times is None:
            reference_times = timestamps
        if not np.array_equal(timestamps, reference_times):
            raise ValueError(f"Time mismatch for {name}")
        if np.any(np.diff(timestamps) != np.timedelta64(6, "h")):
            raise ValueError("Missing, duplicated, reversed, or non-six-hour timestamp")
        raw = np.concatenate(parts)
        manifest["fields"][name] = {"shape": list(raw.shape), "nonfinite": 0, "minimum": float(raw.min()), "maximum": float(raw.max())}
        print(f"Loaded {name}, shape={raw.shape}", flush=True)
        return raw

    raw_q = read_field("humidity")
    months = pd.DatetimeIndex(reference_times).month.to_numpy()
    training = reference_times < np.datetime64("2019-01-01")
    mapping, rlat, rlon = region_map(reference_lat, reference_lon)
    assert len(rlat) == 66 and training.sum() == 58440 and len(reference_times) == 68668
    names, train_arrays, full_arrays, physical_arrays, transforms = [], [], [], [], {}
    def add(name, raw, compare_full=False):
        print(f"Transforming {name}", flush=True)
        z, stats = fit_transform_aggregate(raw, months, training, mapping)
        names.append(name)
        train_arrays.append(z)
        transforms.update({name+"__"+k: v for k,v in stats.items()})
        physical_arrays.append(np.column_stack([raw[:,mapping == r].mean(axis=1, dtype=np.float64) for r in range(len(rlat))]))
        if compare_full:
            f, _ = fit_transform_aggregate(raw, months, np.ones(len(training),dtype=bool), mapping)
            full_arrays.append((name,f))
    add("temperature", read_field("temperature"), True)
    add("humidity", raw_q, True)
    raw_u, raw_v = read_field("u"), read_field("v")
    add("wind", np.sqrt(raw_u.astype(np.float64)**2 + raw_v.astype(np.float64)**2), True)
    add("cloud_cover", read_field("cloud_cover"), True)
    add("u", raw_u)
    add("v", raw_v)
    add("qu", raw_q.astype(np.float64) * raw_u)
    add("qv", raw_q.astype(np.float64) * raw_v)
    data = np.stack(train_arrays, axis=2)
    common = {"lat": rlat, "lon": rlon, "timestamps": reference_times, "node_ids": np.arange(len(rlat))}
    np.savez_compressed(args.output/"region_trainfit.npz", data=data[:,:,:4], variable_names=np.array(names[:4]), **common)
    np.savez_compressed(args.output/"region_trainfit_vectors.npz", data=data, variable_names=np.array(names), physical=np.stack(physical_arrays,axis=2), **common)
    np.savez_compressed(args.output/"trainfit_parameters.npz", **transforms, node_to_region=mapping, grid_lat=reference_lat, grid_lon=reference_lon)
    reconstructed = np.stack([dict(full_arrays)[n] for n in names[:4]], axis=2)
    if args.full_record_output is not None:
        # Same keys, dtypes and string timestamps as the legacy descriptive full-record input.
        stamps = np.array([str(t) for t in pd.DatetimeIndex(reference_times)], dtype=object)
        np.savez_compressed(args.full_record_output, data=reconstructed, variable_names=np.array(names[:4]), lat=rlat, lon=rlon,
                            timestamps=stamps, node_ids=np.arange(len(rlat)))
        manifest["full_record_output"] = {"path": str(args.full_record_output), "sha256": sha256(args.full_record_output),
                                          "climatology": "per cell and calendar month, fitted on 1979-2025 (descriptive full-record screening only)"}
    with np.load(args.legacy, allow_pickle=True) as old:
        if reconstructed.shape != old["data"].shape:
            raise ValueError("Legacy array shape mismatch")
        deltas = np.abs(reconstructed - old["data"])
        manifest["legacy_reconstruction"] = {"legacy_sha256": sha256(args.legacy), "max_abs_difference": float(deltas.max()), "rmse": float(np.sqrt(np.mean(deltas**2))), "region_lat_max_difference": float(np.max(np.abs(rlat-old['lat']))), "region_lon_max_difference": float(np.max(np.abs(rlon-old['lon'])))}
    manifest.update({"train_n": int(training.sum()), "test_n": int((~training).sum()), "regions": len(rlat), "elapsed_seconds": time.perf_counter()-start, "peak_working_set_bytes": int(getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss))})
    manifest["outputs"] = {p.name: sha256(p) for p in args.output.glob("*.npz")}
    (args.output/"data_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ['train_n','test_n','regions','elapsed_seconds','legacy_reconstruction']}),flush=True)


if __name__ == "__main__":
    main()
