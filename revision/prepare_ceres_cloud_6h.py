"""Regrid CERES SYN1deg Ed4.2 hourly total cloud area to the 64x32 analysis grid.

Each instantaneous ERA5 analysis time T (00/06/12/18 UTC) is matched with the
mean of the two CERES hour boxes centred on T, i.e. [T-1h, T) and [T, T+1h),
whose provider timestamps are nominally T-0:30 and T+0:30. Regridding is an
exact spherical-area overlap average from the 1-degree source cells to the
WB2 equiangular 5.625-degree cells (cell bounds at multiples of 5.625 degrees
in latitude, +-2.8125 degrees around longitude centres). Fill values are
masked and excluded from the overlap weights; no temporal interpolation.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(r"D:\Paper2\Major Revision\supplementary_data\cloud_validation\ceres_syn1deg_ed42_global_2001_2025")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def overlap_weights():
    src_lat_edges = np.radians(np.arange(-90.0, 90.0001, 1.0))
    tgt_lat_edges = np.radians(-90.0 + 5.625 * np.arange(33))
    a_lat = np.zeros((32, 180))
    for i in range(32):
        lo, hi = tgt_lat_edges[i], tgt_lat_edges[i + 1]
        bot = np.maximum(lo, src_lat_edges[:-1])
        top = np.minimum(hi, src_lat_edges[1:])
        a_lat[i] = np.where(top > bot, np.sin(top) - np.sin(bot), 0.0)
    src_lon = np.arange(360.0)
    centers = 5.625 * np.arange(64)
    a_lon = np.zeros((64, 360))
    for j, c in enumerate(centers):
        t1, t2 = c - 2.8125, c + 2.8125
        for k in (-360.0, 0.0, 360.0):
            lo = np.maximum(t1, src_lon + k)
            hi = np.minimum(t2, src_lon + 1.0 + k)
            a_lon[j] += np.clip(hi - lo, 0.0, None)
    # Every source cell is fully assigned, and target areas are exact.
    assert np.allclose(a_lat.sum(axis=0), np.sin(src_lat_edges[1:]) - np.sin(src_lat_edges[:-1]))
    assert np.allclose(a_lon.sum(axis=0), 1.0)
    assert np.allclose(a_lon.sum(axis=1), 5.625)
    return a_lat, a_lon


def regrid(block, a_lat, a_lon):
    """block: (n, 180, 360) with NaN for missing; returns (n, 32, 64)."""
    valid = np.isfinite(block)
    filled = np.where(valid, block, 0.0)
    num = np.einsum("ia,nab,jb->nij", a_lat, filled, a_lon, optimize=True)
    den = np.einsum("ia,nab,jb->nij", a_lat, valid.astype(np.float64), a_lon, optimize=True)
    full = a_lat.sum(axis=1)[:, None] * a_lon.sum(axis=1)[None, :]
    out = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    return out, den / full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path(r"D:\Paper2\Major Revision\revision_outputs\ceres_cloud_substitution"))
    ap.add_argument("--chunk", type=int, default=240)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    a_lat, a_lon = overlap_weights()
    files = sorted(glob.glob(str(ROOT / "hourly_*" / "*.nc")))
    if len(files) != 18:
        raise ValueError(f"Expected 18 half-year hourly files, found {len(files)}")
    boxes, coverage, manifest_files = {}, {}, []
    fill_count = total_count = 0
    for f in files:
        path = Path(f)
        with xr.open_dataset(path) as ds:
            raw_t = pd.DatetimeIndex(ds.time.values)
            box_start = (raw_t + pd.Timedelta(minutes=15)).floor("30min") - pd.Timedelta(minutes=30)
            if not np.all(box_start.minute == 0):
                raise ValueError(f"Unexpected time grid in {path.name}")
            hours = box_start.hour.to_numpy()
            need = np.flatnonzero(np.isin(hours, [23, 0, 5, 6, 11, 12, 17, 18]))
            if not np.allclose(ds.lat.values, np.arange(-89.5, 90, 1.0)) or not np.allclose(ds.lon.values, np.arange(0.5, 360, 1.0)):
                raise ValueError(f"Unexpected CERES grid in {path.name}")
            for s in range(0, len(need), args.chunk):
                idx = need[s:s + args.chunk]
                block = np.asarray(ds["cldarea_total_1h"].isel(time=idx).values, dtype=np.float64)
                bad = (block < 0) | (block > 100) | ~np.isfinite(block)
                fill_count += int(bad.sum()); total_count += block.size
                block[bad] = np.nan
                rg, cov = regrid(block, a_lat, a_lon)
                for k, i in enumerate(idx):
                    boxes[box_start[i]] = rg[k].astype(np.float32)
                    coverage[box_start[i]] = float(cov[k].min())
        manifest_files.append({"file": str(path), "bytes": path.stat().st_size, "first_box": str(box_start[0]), "last_box": str(box_start[-1]), "n_hours": int(len(box_start))})
        print(f"done {path.name}: {len(need)} boxes, elapsed {time.perf_counter()-start:.0f}s", flush=True)
    era5_times = pd.date_range("2017-01-01 00:00", "2025-12-31 18:00", freq="6h")
    data = np.full((len(era5_times), 32, 64), np.nan, dtype=np.float32)
    n_pairs = 0
    for k, t in enumerate(era5_times):
        a, b = boxes.get(t - pd.Timedelta(hours=1)), boxes.get(t)
        if a is not None and b is not None:
            data[k] = 0.5 * (a + b); n_pairs += 1
    np.savez_compressed(args.output / "ceres_cloud_6h_64x32.npz", cloud_percent=data, timestamps=era5_times.values.astype("datetime64[ns]"),
                        grid_lat=-87.1875 + 5.625 * np.arange(32), grid_lon=5.625 * np.arange(64))
    min_cov = min(coverage.values())
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": "exact spherical-area overlap regridding 1deg->64x32; centred two-hour mean around each ERA5 time",
        "n_era5_times": int(len(era5_times)), "n_times_with_both_boxes": int(n_pairs),
        "times_missing": [str(t) for t, ok in zip(era5_times, np.isfinite(data).all(axis=(1, 2))) if not ok][:20],
        "fill_or_out_of_range_fraction": fill_count / max(total_count, 1),
        "minimum_target_valid_area_fraction": min_cov,
        "files": manifest_files, "code_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output / "ceres_preparation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["n_era5_times", "n_times_with_both_boxes", "fill_or_out_of_range_fraction", "minimum_target_valid_area_fraction", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
