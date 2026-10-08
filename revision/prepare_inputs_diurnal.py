"""Diurnally adjusted variant of prepare_inputs: month x UTC-hour climatology.

Identical to prepare_inputs.py except that the per-cell climatology is fitted
for each (calendar month, UTC hour) pair instead of each calendar month, so the
6-hourly diurnal cycle is removed before standardization. All statistics are
fitted on 1979-2018 only and applied unchanged to 2019-2025.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import xarray as xr

from revision.prepare_inputs import SEGMENTS, FILES, region_map, sha256


def fit_transform_aggregate(raw, keys, training, mapping):
    x = np.array(raw, dtype=np.float64, copy=True)
    clim = {}
    for k in np.unique(keys):
        m = keys == k
        clim[int(k)] = x[training & m].mean(axis=0)
        x[m] -= clim[int(k)]
    mu = x[training].mean(axis=0)
    sd = x[training].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    x = (x - mu) / sd
    return np.column_stack([x[:, mapping == r].mean(axis=1) for r in np.unique(mapping)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(r"D:\Paper2\vipuser\Data"))
    p.add_argument("--output", type=Path, default=Path(r"D:\Paper2\Major Revision\revision_outputs\inputs_diurnal"))
    p.add_argument("--reference", type=Path, default=Path(r"D:\Paper2\Major Revision\revision_outputs\inputs\region_trainfit.npz"))
    p.add_argument("--extension-dir", type=Path, help="Directory replacing the 2023-01-11--2025 CDS segment, e.g. the native-conservative route.")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    lat0 = lon0 = t0 = None

    def read(name):
        nonlocal lat0, lon0, t0
        parts, times = [], []
        for seg in SEGMENTS:
            folder = args.extension_dir if (args.extension_dir is not None and seg == SEGMENTS[-1]) else args.data_root / seg
            with xr.open_dataset(folder / FILES[name]) as ds:
                key = next(k for k in ds.data_vars if {"time", "latitude", "longitude"}.issubset(ds[k].dims))
                da = ds[key].transpose("time", "latitude", "longitude")
                if lat0 is None:
                    lat0, lon0 = ds.latitude.values.copy(), ds.longitude.values.copy()
                if not (np.allclose(ds.latitude.values, lat0, atol=1e-3) and np.allclose(ds.longitude.values, lon0, atol=1e-3)):
                    raise ValueError("grid mismatch")
                parts.append(np.asarray(da.values, dtype=np.float32).reshape(da.shape[0], -1))
                times.append(np.asarray(ds.time.values, dtype="datetime64[ns]"))
        t = np.concatenate(times)
        if t0 is None:
            t0 = t
        if not np.array_equal(t, t0) or np.any(np.diff(t) != np.timedelta64(6, "h")):
            raise ValueError("time axis")
        return np.concatenate(parts)

    q = read("humidity")
    idx = pd.DatetimeIndex(t0)
    keys = idx.month.to_numpy() * 100 + idx.hour.to_numpy()
    training = t0 < np.datetime64("2019-01-01")
    mapping, rlat, rlon = region_map(lat0, lon0)
    temperature = read("temperature")
    u, v = read("u"), read("v")
    cloud = read("cloud_cover")
    arrays = [fit_transform_aggregate(temperature, keys, training, mapping),
              fit_transform_aggregate(q, keys, training, mapping),
              fit_transform_aggregate(np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2), keys, training, mapping),
              fit_transform_aggregate(cloud, keys, training, mapping)]
    data = np.stack(arrays, axis=2)
    with np.load(args.reference) as ref:
        if not (np.allclose(ref["lat"], rlat) and np.allclose(ref["lon"], rlon) and np.array_equal(ref["timestamps"], t0)):
            raise ValueError("Region/time layout differs from the monthly-climatology input")
        names = ref["variable_names"]
        corr = [float(np.corrcoef(ref["data"][:, :, k].ravel(), data[:, :, k].ravel())[0, 1]) for k in range(4)]
    np.savez_compressed(args.output / "region_trainfit.npz", data=data, variable_names=names, lat=rlat, lon=rlon, timestamps=t0, node_ids=np.arange(len(rlat)))
    manifest = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "climatology": "per cell, per (calendar month, UTC hour), fitted 1979-2018",
                "correlation_with_monthly_climatology_input": dict(zip([str(n) for n in names], corr)),
                "extension_dir": str(args.extension_dir) if args.extension_dir else None,
                "code_sha256": sha256(Path(__file__)), "elapsed_seconds": time.perf_counter() - start}
    (args.output / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
