"""Conservative remapping of the native 0.25-degree primary fields to the WB2 64x32 grid.

Applies the separable spherical cell-overlap method of regrid_era5_cds_controls.py
(which follows WeatherBench 2's regridding code) to the files retrieved by
download_era5_primary_native.py. Every native file is SHA256-checked against its
acquisition receipt and decoded in full; nonfinite values or finite missing
sentinels stop the run, and every time plane is checked for area-integral
preservation and range. Outputs, under supplementary_data/era5_primary_native/:

* regridded/<job>.nc - one 64x32 file per native file, variables named as in WB2;
* era5_cds_native_conservative_6h_64x32_850hPa_2023-01-11_2025/ - the extension
  concatenated into one file per variable, laid out like the existing CDS segment
  so that prepare_inputs.py can read it in place of the 5.625-degree route;
* overlap_64x32/ - the 152 pre-splice timestamps, for the same-date check against WB2.

Run while the download is in progress to remap the files completed so far; rerun
to resume. --combine assembles the segment once all extension files are remapped.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
import xarray as xr

from regrid_era5_cds_controls import geometry, geometry_tests

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO.parent / "supplementary_data/era5_primary_native"
OUTPUT = SOURCE / "regridded"
SEGMENT = SOURCE / "era5_cds_native_conservative_6h_64x32_850hPa_2023-01-11_2025"
OVERLAP = SOURCE / "overlap_64x32"
WB2_REFERENCE = Path(r"D:\Paper2\vipuser\Data\weatherbench2_era5_6h_64x32_850hPa_2019_2023-01-10\temperature_850.nc")
# raw CDS name -> (WB2 variable / file stem, units)
FIELDS = {"t": ("temperature_850", "K"), "q": ("specific_humidity_850", "kg kg**-1"),
          "u": ("u_component_of_wind_850", "m s**-1"), "v": ("v_component_of_wind_850", "m s**-1"),
          "tcc": ("total_cloud_cover", "(0 - 1)")}
BLOCK = 32
METHOD = ("First-order separable spherical cell-overlap conservative mean (WeatherBench 2 regridding); midpoint latitude "
          "boundaries capped at +-90, periodic longitude boundaries; float64 accumulation, float32 output.")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    tmp.replace(path)


def target_grid():
    with xr.open_dataset(WB2_REFERENCE) as ds:
        return ds.latitude.values.copy(), ds.longitude.values.copy()


def remap_block(block, wl, wo):
    """Remap (nt, 721, 1440) float64 planes to (nt, 32, 64) with the sparse separable weights."""
    nt, ny, nx = block.shape
    lon_done = wo.dot(block.reshape(nt * ny, nx).T)               # (64, nt*ny)
    lon_done = lon_done.reshape(wo.shape[0], nt, ny).transpose(2, 1, 0).reshape(ny, nt * wo.shape[0])
    lat_done = wl.dot(lon_done)                                    # (32, nt*64)
    return lat_done.reshape(wl.shape[0], nt, wo.shape[0]).transpose(1, 0, 2)


def process(job, state, geo):
    slat, slon, tlat, tlon, wl, wo, sa, ta = geo
    source = SOURCE / state["file"]
    target = OUTPUT / (job["id"] + ".nc")
    report_path = OUTPUT / "reports" / (job["id"] + ".json")
    if target.exists() and report_path.exists():
        record = json.loads(report_path.read_text(encoding="utf8"))
        if record.get("status") == "passed" and record.get("source_sha256") == state["sha256"] and record.get("output_sha256") == sha(target):
            return record, False
    started = time.perf_counter()
    if sha(source) != state["sha256"] or source.stat().st_size != state["bytes_downloaded"]:
        raise ValueError("Source differs from its acquisition receipt: " + job["id"])
    arrays, stats = {}, {}
    with xr.open_dataset(source, engine="netcdf4") as ds:
        times = pd.DatetimeIndex(ds.valid_time.values)
        if not times.equals(pd.date_range(job["first_time"], job["last_time"], freq="6h")):
            raise ValueError("Time mismatch: " + job["id"])
        lat_order, lon_order = np.argsort(ds.latitude.values), np.argsort(ds.longitude.values % 360)
        np.testing.assert_array_equal(ds.latitude.values[lat_order], slat)
        np.testing.assert_array_equal((ds.longitude.values % 360)[lon_order], slon)
        expver = sorted({str(v) for v in np.atleast_1d(ds["expver"].values)}) if "expver" in ds.variables else None
        for raw in job["expected_variables"]:
            name, units = FIELDS[raw]
            da = ds[raw]
            if da.attrs.get("units") != units:
                raise ValueError(f"Units mismatch for {raw}: {da.attrs.get('units')}")
            if "pressure_level" in da.dims:
                da = da.sel(pressure_level=850.0)
            da = da.transpose("valid_time", "latitude", "longitude")
            sentinels = sorted({float(v) for meta in (ds[raw].attrs, ds[raw].encoding)
                                for key in ("GRIB_missingValue", "missing_value", "_FillValue") if key in meta
                                for v in np.atleast_1d(meta[key]) if np.isfinite(v)})
            out = np.empty((len(times), len(tlat), len(tlon)), dtype=np.float32)
            lo_all, hi_all, worst64, worst32 = np.inf, -np.inf, 0.0, 0.0
            for lo in range(0, len(times), BLOCK):
                block = da.isel(valid_time=slice(lo, lo + BLOCK)).values
                block = block[:, lat_order, :][:, :, lon_order]
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite decoded values: {job['id']} {raw} block {lo}")
                if any(np.any(block == np.float32(s)) for s in sentinels):
                    raise ValueError(f"Finite missing sentinel present: {job['id']} {raw} block {lo}")
                b64 = block.astype(np.float64)
                mapped = remap_block(b64, wl, wo)
                rounded = mapped.astype(np.float32)
                src_int = np.einsum("tij,ij->t", b64, sa)
                scale = np.maximum(np.einsum("tij,ij->t", np.abs(b64), sa), 1e-20)
                err64 = np.abs(np.einsum("tij,ij->t", mapped, ta) - src_int) / scale
                err32 = np.abs(np.einsum("tij,ij->t", rounded.astype(np.float64), ta) - src_int) / scale
                if err64.max() > 2e-12 or err32.max() > 2e-7:
                    raise ValueError(f"Area-integral preservation failed: {job['id']} {raw}")
                pmin, pmax = b64.min(axis=(1, 2)), b64.max(axis=(1, 2))
                tol = 1e-8 + 1e-12 * np.maximum(np.abs(pmin), np.abs(pmax))
                if np.any(mapped.min(axis=(1, 2)) < pmin - tol) or np.any(mapped.max(axis=(1, 2)) > pmax + tol):
                    raise ValueError(f"Convex-range check failed: {job['id']} {raw}")
                out[lo:lo + len(block)] = rounded
                lo_all, hi_all = min(lo_all, float(pmin.min())), max(hi_all, float(pmax.max()))
                worst64, worst32 = max(worst64, float(err64.max())), max(worst32, float(err32.max()))
            arrays[name] = (out, units, raw)
            stats[name] = {"source_minimum": lo_all, "source_maximum": hi_all, "output_minimum": float(out.min()),
                           "output_maximum": float(out.max()), "max_float64_integral_error_over_L1": worst64,
                           "max_float32_integral_error_over_L1": worst32, "missing_sentinels_checked": sentinels}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.nc")
    dataset = xr.Dataset({name: (("time", "latitude", "longitude"), values,
                                 {"units": units, "source_variable": raw, "cell_methods": "area: mean (spherical conservative remapping)"})
                          for name, (values, units, raw) in arrays.items()},
                         coords={"time": times.values, "latitude": tlat, "longitude": tlon},
                         attrs={"source_job_id": job["id"], "source_sha256": state["sha256"], "method": METHOD, "created_utc": now()})
    dataset.to_netcdf(partial, engine="netcdf4",
                      encoding={n: {"dtype": "float32", "zlib": True, "complevel": 1, "shuffle": True} for n in arrays})
    with xr.open_dataset(partial) as check:
        for name, (values, _, _) in arrays.items():
            np.testing.assert_array_equal(check[name].values, values)
    partial.replace(target)
    record = {"status": "passed", "job": job["id"], "kind": job["kind"], "completed_utc": now(),
              "source_file": str(source), "source_sha256": state["sha256"], "expver": expver,
              "output_file": str(target), "output_sha256": sha(target), "output_bytes": target.stat().st_size,
              "time_count": len(times), "first_time": str(times[0]), "last_time": str(times[-1]), "fields": stats,
              "elapsed_seconds": round(time.perf_counter() - started, 2),
              "peak_working_set_bytes": int(getattr(psutil.Process().memory_info(), "peak_wset", 0))}
    save(report_path, record)
    return record, True


def combine(jobs, kind, directory, expected_times):
    files = [OUTPUT / (j["id"] + ".nc") for j in jobs if j["kind"] == kind]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} {kind} files not yet remapped, e.g. {missing[:3]}")
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for raw, (name, units) in FIELDS.items():
        parts = []
        for f in files:
            with xr.open_dataset(f) as ds:
                if name in ds:
                    parts.append(ds[name].load())
        da = xr.concat(parts, dim="time").sortby("time")
        if not pd.DatetimeIndex(da.time.values).equals(expected_times):
            raise ValueError(f"Combined {name} does not cover the expected timestamps")
        path = directory / f"{name}.nc"
        tmp = path.with_suffix(".tmp.nc")
        da.attrs.update(units=units, method=METHOD)
        da.to_dataset(name=name).assign_attrs(source="ERA5 via CDS, native 0.25-degree grid, conservatively remapped to the WB2 64x32 grid",
                                               created_utc=now()).to_netcdf(
            tmp, engine="netcdf4", encoding={name: {"dtype": "float32", "zlib": True, "complevel": 1, "shuffle": True,
                                                    "chunksizes": (min(100, da.sizes["time"]), 32, 64)}})
        tmp.replace(path)
        written[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size, "time_count": int(da.sizes["time"])}
    save(directory / "manifest.json", {"created_utc": now(), "kind": kind, "method": METHOD, "files": written,
                                       "first": str(expected_times[0]), "last": str(expected_times[-1])})
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", choices=["overlap", "extension", "all"], default="all")
    ap.add_argument("--combine", action="store_true", help="Assemble the overlap/extension files after remapping.")
    args = ap.parse_args()
    plan = json.loads((SOURCE / "request_plan.json").read_text(encoding="utf8"))
    jobs = [j for j in plan["jobs"] if args.kind == "all" or j["kind"] == args.kind]
    slat, slon = np.arange(-90., 90.001, .25), np.arange(0., 360., .25)
    tlat, tlon = target_grid()
    wl, wo, sa, ta = geometry(slat, slon, tlat, tlon)
    tests = geometry_tests(slat, slon, tlat, tlon, wl, wo, sa, ta)
    geo = (slat, slon, tlat, tlon, wl, wo, sa, ta)
    done, pending, started = 0, [], time.perf_counter()
    for job in jobs:
        state_path = SOURCE / "jobs" / (job["id"] + ".json")
        state = json.loads(state_path.read_text(encoding="utf8")) if state_path.exists() else {}
        if state.get("status") != "completed":
            pending.append(job["id"])
            continue
        record, new = process(job, state, geo)
        done += 1
        if new:
            print(f"remapped {job['id']}: {record['time_count']} times in {record['elapsed_seconds']:.0f} s", flush=True)
    status = {"updated_utc": now(), "geometry_tests": tests["status"], "remapped": done, "awaiting_download": pending,
              "elapsed_seconds": round(time.perf_counter() - started, 1)}
    save(OUTPUT / f"status_{args.kind}.json", status)
    print(json.dumps({k: (v if k != "awaiting_download" else len(v)) for k, v in status.items()}), flush=True)
    if args.combine:
        if args.kind in ("overlap", "all"):
            times = pd.DatetimeIndex(sorted({t for j in plan["jobs"] if j["kind"] == "overlap"
                                             for t in pd.date_range(j["first_time"], j["last_time"], freq="6h")}))
            print(json.dumps(combine(plan["jobs"], "overlap", OVERLAP, times), indent=1), flush=True)
        if args.kind in ("extension", "all"):
            times = pd.date_range("2023-01-11", "2025-12-31 18:00", freq="6h")
            print(json.dumps(combine(plan["jobs"], "extension", SEGMENT, times), indent=1), flush=True)


if __name__ == "__main__":
    main()
