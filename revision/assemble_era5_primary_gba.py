"""Assemble the CDS grid-box-average primary fields into WB2-layout segment files.

Files retrieved with `download_era5_primary_native.py --route gba` are already on
the WB2 64x32 grid: the CDS (MARS) computes area-weighted grid-box averages
server-side. Each file is checked against its acquisition receipt (SHA256), its
times, its coordinates (against WB2), finite values and the cloud-fraction range,
and the variables receive their WB2 names. Outputs, under
supplementary_data/era5_primary_gba/:

* era5_cds_gba_6h_64x32_850hPa_2023-01-11_2025/ - one file per variable, laid out
  like the existing CDS segment so that prepare_inputs.py can read it;
* overlap_64x32/ - the 152 pre-splice timestamps, for the same-date check against WB2.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO.parent / "supplementary_data/era5_primary_gba"
SEGMENT = SOURCE / "era5_cds_gba_6h_64x32_850hPa_2023-01-11_2025"
OVERLAP = SOURCE / "overlap_64x32"
WB2_REFERENCE = Path(r"D:\Paper2\vipuser\Data\weatherbench2_era5_6h_64x32_850hPa_2019_2023-01-10\temperature_850.nc")
FIELDS = {"t": ("temperature_850", "K"), "q": ("specific_humidity_850", "kg kg**-1"),
          "u": ("u_component_of_wind_850", "m s**-1"), "v": ("v_component_of_wind_850", "m s**-1"),
          "tcc": ("total_cloud_cover", "(0 - 1)")}
METHOD = "CDS/MARS interpolation=grid-box-average onto the WB2 64x32 grid (area-weighted mean of native grid boxes)."


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load(job, raw, tlat, tlon):
    state = json.loads((SOURCE / "jobs" / (job["id"] + ".json")).read_text(encoding="utf8"))
    if state.get("status") != "completed":
        raise SystemExit(f"{job['id']} is not downloaded yet ({state.get('status')})")
    path = SOURCE / state["file"]
    if sha(path) != state["sha256"]:
        raise ValueError("Source differs from its acquisition receipt: " + job["id"])
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds.valid_time.values)
        if not times.equals(pd.date_range(job["first_time"], job["last_time"], freq="6h")):
            raise ValueError("Time mismatch: " + job["id"])
        da = ds[raw]
        if da.attrs.get("units") != FIELDS[raw][1]:
            raise ValueError(f"Units mismatch for {raw} in {job['id']}: {da.attrs.get('units')}")
        if "pressure_level" in da.dims:
            da = da.sel(pressure_level=850.0)
        da = da.sortby("latitude").sortby("longitude").transpose("valid_time", "latitude", "longitude")
        # Coordinates are stored in millidegrees for GRIB1 fields; the exact WB2 values are assigned below.
        if np.max(np.abs(da.latitude.values - tlat)) > 1e-3 or np.max(np.abs(da.longitude.values % 360 - tlon)) > 1e-3:
            raise ValueError("Grid differs from WB2: " + job["id"])
        values = np.asarray(da.values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite values: {job['id']} {raw}")
    # GRIB packing leaves cloud fractions up to ~1e-5 outside [0, 1]; values are kept unclipped.
    if raw == "tcc" and (values.min() < -1e-4 or values.max() > 1 + 1e-4):
        raise ValueError(f"Cloud fraction outside 0-1: {job['id']}")
    return times, values, {"job": job["id"], "sha256": state["sha256"]}


def assemble(jobs, kind, directory, expected, tlat, tlon):
    directory.mkdir(parents=True, exist_ok=True)
    written, sources = {}, []
    for raw, (name, units) in FIELDS.items():
        group = "total_cloud_cover" if raw == "tcc" else "primary_850"
        parts = [load(j, raw, tlat, tlon) for j in jobs if j["kind"] == kind and j["id"].startswith(group + "_")]
        times = pd.DatetimeIndex(np.concatenate([p[0].values for p in parts]))
        order = np.argsort(times.values)
        times, values = times[order], np.concatenate([p[1] for p in parts])[order]
        if not times.equals(expected):
            raise ValueError(f"{kind} {name}: timestamps do not match the expected axis")
        sources += [p[2] for p in parts]
        da = xr.DataArray(values, dims=("time", "latitude", "longitude"),
                          coords={"time": times.values, "latitude": tlat, "longitude": tlon},
                          attrs={"units": units, "method": METHOD})
        path = directory / f"{name}.nc"
        tmp = path.with_suffix(".tmp.nc")
        da.to_dataset(name=name).assign_attrs(source="ERA5 via CDS, " + METHOD,
                                               created_utc=datetime.now(timezone.utc).isoformat()).to_netcdf(
            tmp, engine="netcdf4", encoding={name: {"dtype": "float32", "zlib": True, "complevel": 1, "shuffle": True,
                                                    "chunksizes": (min(100, len(times)), len(tlat), len(tlon))}})
        tmp.replace(path)
        written[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size, "time_count": len(times),
                              "minimum": float(values.min()), "maximum": float(values.max())}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": kind, "method": METHOD,
                "first": str(expected[0]), "last": str(expected[-1]), "files": written,
                "sources": sorted({s["job"]: s["sha256"] for s in sources}.items()), "code_sha256": sha(__file__)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", choices=["overlap", "extension", "all"], default="all")
    args = ap.parse_args()
    jobs = json.loads((SOURCE / "request_plan.json").read_text(encoding="utf8"))["jobs"]
    with xr.open_dataset(WB2_REFERENCE) as ref:
        tlat, tlon = ref.latitude.values.copy(), ref.longitude.values.copy()
    if args.kind in ("overlap", "all"):
        expected = pd.DatetimeIndex(sorted({t for j in jobs if j["kind"] == "overlap"
                                            for t in pd.date_range(j["first_time"], j["last_time"], freq="6h")}))
        print(json.dumps(assemble(jobs, "overlap", OVERLAP, expected, tlat, tlon), indent=1))
    if args.kind in ("extension", "all"):
        expected = pd.date_range("2023-01-11", "2025-12-31 18:00", freq="6h")
        print(json.dumps(assemble(jobs, "extension", SEGMENT, expected, tlat, tlon), indent=1))


if __name__ == "__main__":
    main()
