"""Build physical horizontal moisture-transport inputs, without fitting models.

Only --execute reads meteorological values; --metadata-only checks provenance,
coordinates, timestamps, interface geometry, and analytic operator fixtures.
The 850-hPa flux divergence is a resolved horizontal diagnostic, not a complete
moisture budget, native terrain mask, turbulence closure, or causal estimand.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import psutil
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_inputs import FILES, SEGMENTS, region_map

RADIUS = 6371000.0
REGION_IDS = np.array([15, 48], dtype=np.int64)
LOCAL_NAMES = ["u", "v", "q", "temperature", "cloud"]
GRID_NAMES = LOCAL_NAMES[:4]
SOURCE_FIELDS = ["u", "v", "humidity", "temperature", "cloud_cover"]
FLUX_NAMES = ["mean_qu", "mean_qv", "cov_qu", "cov_qv", "mfc"]
CONTROL_NAMES = ["omega_500", "omega_700", "geopotential_500", "temperature_700",
                 "surface_pressure", "mean_sea_level_pressure"]
UNITS = {"u": "m s**-1", "v": "m s**-1", "humidity": "kg kg**-1",
         "temperature": "K", "cloud_cover": "(0 - 1)"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return pd.Timestamp.now(tz="UTC").isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    os.replace(temporary, path)


def geometry(latitude, longitude):
    """Ascending regular global longitude, midpoint latitude bounds at poles."""
    lat, lon = np.deg2rad(latitude), np.deg2rad(longitude)
    check(np.all(np.diff(lat) > 0), "Latitude must increase south to north")
    dlon = 2 * np.pi / len(lon)
    check(np.allclose(np.diff(lon), dlon, atol=1e-12, rtol=0), "Longitude not uniform")
    check(np.isclose(lon[-1] - lon[0] + dlon, 2*np.pi), "Longitude not periodic global")
    bounds = np.r_[-np.pi / 2, (lat[1:] + lat[:-1]) / 2, np.pi / 2]
    dlat = np.diff(bounds)
    area = np.repeat((RADIUS**2 * dlon * np.diff(np.sin(bounds)))[:, None], len(lon), axis=1)
    cos_bounds = np.cos(bounds)
    cos_bounds[[0, -1]] = 0.0
    return {"lat": lat, "lon": lon, "lat_bounds": bounds, "dlon": dlon,
            "dlat": dlat, "area": area, "cos_bounds": cos_bounds}


def divergence(qu, qv, geo):
    """Fluxes in m/s; output divergence in s^-1. Final axes are lat/lon."""
    check(qu.shape == qv.shape, "Flux component shapes differ")
    east = 0.5 * (qu + np.roll(qu, -1, axis=-1))
    zonal_net = (east - np.roll(east, 1, axis=-1)) * RADIUS * geo["dlat"][:, None]
    north_faces = np.zeros(qv.shape[:-2] + (qv.shape[-2] + 1, qv.shape[-1]), dtype=np.float64)
    north_faces[..., 1:-1, :] = 0.5 * (qv[..., 1:, :] + qv[..., :-1, :])
    north_faces *= RADIUS * geo["dlon"] * geo["cos_bounds"][:, None]
    return (zonal_net + np.diff(north_faces, axis=-2)) / geo["area"]


def direct_region_mfc(qu, qv, indices, geo):
    """Independent external-boundary reduction, avoiding cell divergence sums."""
    ny, nx = qu.shape[-2:]
    rows, cols = np.divmod(indices, nx)
    rr, cc = np.unique(rows), np.unique(cols)
    check(len(indices) == len(rr)*len(cc) and np.all(np.diff(rr) == 1)
          and np.all(np.diff(cc) == 1), "Diagnostic expects contiguous rectangular region")
    south, north, west, east = rr[0], rr[-1], cc[0], cc[-1]
    fe = .5 * (qu[..., rr, east] + qu[..., rr, (east+1) % nx])
    fw = .5 * (qu[..., rr, west] + qu[..., rr, (west-1) % nx])
    zonal_net = ((fe-fw) * RADIUS * geo["dlat"][rr]).sum(axis=-1)
    vn = .5 * (qv[..., north, cc] + qv[..., north+1, cc])
    vs = .5 * (qv[..., south, cc] + qv[..., south-1, cc])
    meridional_net = RADIUS * geo["dlon"] * (vn.sum(axis=-1) * geo["cos_bounds"][north+1]
                                                           - vs.sum(axis=-1) * geo["cos_bounds"][south])
    return -(zonal_net + meridional_net) / geo["area"].ravel()[indices].sum()


def analytic_checks():
    results = {}
    relative_errors = []
    for nlat in (32, 64):
        lat = (np.arange(nlat)+.5)*180/nlat - 90
        lon = np.arange(2*nlat)*360/(2*nlat)
        geo = geometry(lat, lon)
        zeros = np.zeros((nlat, 2*nlat))
        constant = divergence(zeros+0.1, zeros, geo)
        check(np.max(np.abs(constant)) == 0, "Constant zonal flux should be divergence free")
        # For F_phi=cos(phi), integrated analytic divergence is -(sin N+sin S)/R.
        qv = np.repeat(np.cos(geo["lat"])[:, None], 2*nlat, axis=1)
        computed = divergence(zeros, qv, geo)
        exact = -np.repeat((np.sin(geo["lat_bounds"][1:]) + np.sin(geo["lat_bounds"][:-1]))[:, None], 2*nlat, axis=1) / RADIUS
        relative = float(np.sqrt(np.sum((computed-exact)**2 * geo["area"])
                                        / np.sum(exact**2 * geo["area"])))
        relative_errors.append(relative)
        expected = exact * np.cos(np.pi/(2*nlat))
        check(np.allclose(computed, expected, rtol=1e-12, atol=1e-20), "Analytic meridional flux mismatch")
        # Longitude-periodic sine input checks the last/first cell connection.
        qu = np.repeat(np.sin(geo["lon"])[None, :], nlat, axis=0)
        actual = divergence(qu, zeros, geo)
        predicted = (RADIUS * geo["dlat"][:, None] * np.cos(geo["lon"])[None, :]
                     * np.sin(geo["dlon"]) / geo["area"])
        check(np.allclose(actual, predicted, rtol=1e-11, atol=1e-20), "Periodic sine flux mismatch")
        results[str(nlat)] = {"constant_zonal_max_abs": float(np.abs(constant).max()),
                              "smooth_meridional_relative_l2_error": relative,
                              "analytic_meridional_max_abs": float(np.abs(computed-expected).max()),
                              "periodic_zonal_max_abs": float(np.abs(actual-predicted).max()),
                              "global_area_relative_error": float(abs(geo["area"].sum()/(4*np.pi*RADIUS**2)-1))}
    ratio = relative_errors[0] / relative_errors[1]
    check(3.9 < ratio < 4.1, "Smooth flux refinement did not show second-order convergence")
    results["coarse_to_fine_error_ratio"] = ratio
    results["passed"] = True
    return results


def inspect(args):
    old_manifest_path = args.reference / "data_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf8"))
    known = {str(Path(r["path"]).resolve()).lower(): r for r in old_manifest["sources"]}
    parameter_path = args.reference / "trainfit_parameters.npz"
    check(sha256(parameter_path) == old_manifest["outputs"][parameter_path.name], "Reference grid hash mismatch")
    with np.load(parameter_path, allow_pickle=False) as d:
        lat, lon, saved_mapping = d["grid_lat"], d["grid_lon"], d["node_to_region"]
    mapping, rlat, rlon = region_map(lat, lon)
    check(np.array_equal(mapping, saved_mapping), "Region map differs from existing analysis")
    check((len(lat), len(lon)) == (32, 64), "Expected native analysis grid 32x64")
    core = np.stack([np.flatnonzero(mapping == rid) for rid in REGION_IDS])
    check(core.shape == (2, 36), "Expected two regions of 36 cells")
    halo = []
    for indices in core:
        ids = set(indices.tolist())
        for i in indices:
            y, x = divmod(int(i), len(lon))
            check(0 < y < len(lat)-1, "Chosen region touches a pole")
            ids.update([y*len(lon)+(x-1)%len(lon), y*len(lon)+(x+1)%len(lon),
                        (y-1)*len(lon)+x, (y+1)*len(lon)+x])
        halo.append(sorted(ids))
    halo = np.array(halo, dtype=np.int64)
    check(halo.shape == (2, 60), "Expected equal 60-cell core-plus-halo shapes")
    check(np.array_equal(halo[0]-halo[0, 0], halo[1]-halo[1, 0]), "Halo ordering differs by region")
    geo = geometry(lat, lon)
    weights = geo["area"].ravel()[core]
    weights /= weights.sum(axis=1, keepdims=True)
    sources, reference_parts = [], []
    for si, segment in enumerate(SEGMENTS):
        segment_times = None
        for field in SOURCE_FIELDS:
            path = args.data_root / segment / FILES[field]
            old = known[str(path.resolve()).lower()]
            digest = sha256(path)
            check(digest == old["sha256"] and path.stat().st_size == old["bytes"], f"Source provenance mismatch: {path}")
            with xr.open_dataset(path) as ds:
                key = next(k for k in ds.data_vars if {"time", "latitude", "longitude"}.issubset(ds[k].dims))
                da = ds[key]
                check(set(da.dims) == {"time", "latitude", "longitude"}, "Unexpected variable dimensions")
                times = np.asarray(ds.time.values, dtype="datetime64[ns]")
                if segment_times is None:
                    segment_times = times
                check(np.array_equal(times, segment_times), "Times differ between fields")
                dlat = float(np.max(np.abs(ds.latitude.values-lat)))
                dlon = float(np.max(np.abs(ds.longitude.values-lon)))
                check(dlat <= .001 and dlon <= .001, "Grid coordinate discrepancy exceeds declared tolerance")
                units = str(da.attrs.get("units", ""))
                check(units == old["units"], "Source units differ from previous provenance")
                if field != "cloud_cover":
                    check(units == UNITS[field], f"Unexpected physical units {field}: {units}")
                else:
                    check(units in ("(0 - 1)", "1", "(0-1)"), f"Cloud units must be fraction, got {units}")
                row = {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size,
                       "segment": si, "field": field, "variable": key, "units": units,
                       "shape": [len(times), len(lat), len(lon)], "first": str(times[0]), "last": str(times[-1]),
                       "latitude_max_abs_difference_degrees": dlat, "longitude_max_abs_difference_degrees": dlon}
                sources.append(row)
        reference_parts.append(segment_times)
    timestamps = np.concatenate(reference_parts)
    check(len(timestamps) == 68668 and timestamps[0] == np.datetime64("1979-01-01T00")
          and timestamps[-1] == np.datetime64("2025-12-31T18"), "Unexpected analysis coverage")
    check(np.all(np.diff(timestamps) == np.timedelta64(6, "h")), "Noncontinuous six-hour timestamps")
    controls_path = args.controls / "region_controls_trainfit.npz"
    control_manifest_path = args.controls / "data_manifest.json"
    cm = json.loads(control_manifest_path.read_text(encoding="utf8"))
    control_sha = sha256(controls_path)
    check(control_sha == cm["outputs"][controls_path.name]["sha256"], "Control archive hash mismatch")
    with np.load(controls_path, allow_pickle=False) as d:
        check(np.array_equal(d["timestamps"], timestamps), "Control/source timestamps mismatch")
        check(d["variable_names"].tolist() == CONTROL_NAMES, "Unexpected control field order")
        check(np.array_equal(d["node_ids"][REGION_IDS], REGION_IDS), "Control region IDs mismatch")
        check(np.array_equal(d["lat"][REGION_IDS], rlat[REGION_IDS]) and
              np.array_equal(d["lon"][REGION_IDS], rlon[REGION_IDS]), "Control region coordinates mismatch")
        source_segment = d["source_segment"]
        check(np.array_equal(source_segment, np.repeat(np.arange(3), [len(t) for t in reference_parts])),
              "Control/source segment labels mismatch")
        control_units = d["variable_units"].tolist()
    report = {"status": "metadata_and_analytic_checks_passed", "created_utc": now(),
              "code_sha256": sha256(__file__), "sources": sources,
              "reference_manifest": {"path": str(old_manifest_path), "sha256": sha256(old_manifest_path)},
              "grid_reference": {"path": str(parameter_path), "sha256": sha256(parameter_path)},
              "controls": {"path": str(controls_path), "sha256": control_sha, "units": control_units,
                           "manifest_path": str(control_manifest_path), "manifest_sha256": sha256(control_manifest_path)},
              "shape": {"local": [68668, 2, 5], "flux": [68668, 2, 5], "grid": [68668, 2, 60, 4],
                        "controls": [68668, 2, 6]}, "region_ids": REGION_IDS.tolist(),
              "region_lat": rlat[REGION_IDS].tolist(), "region_lon": rlon[REGION_IDS].tolist(),
              "core_indices": core.tolist(), "halo_indices": halo.tolist(),
              "area_weights": weights.tolist(), "analytic_operator_checks": analytic_checks(),
              "coordinate_policy": "Use original WB2 reference coordinate geometry with matched index order; CDS rounded latitude differs by up to 0.0005 degree. This does not establish identical interpolation kernels.",
              "method": {"earth_radius_m": RADIUS, "mfc": "-div_h(q850*(u850,v850)); units s^-1",
                         "flux_interpolation": "arithmetic adjacent-cell product q*u or q*v at faces; periodic longitude; pole-face area flux zero",
                         "region_aggregation": "exact spherical cell-area weights from midpoint latitude bounds",
                         "covariance": "weighted mean(q*u)-weighted mean(q)*weighted mean(u); resolved grid covariance, not turbulence",
                         "control_aggregation": "reuse existing physical arithmetic region means; differs from new area means; same for every model",
                         "halo_order": "ascending original flattened latitude-major indices; identical local offsets in both regions"},
              "limitations": ["No complete vertical moisture budget or condensation/evaporation budget.",
                              "850-hPa pressure-level source values may include below-ground extrapolations; regional surface pressure is only a coarse warning diagnostic.",
                              "Source/regridding transition at 2023-01-11 remains explicit.",
                              "Global horizontal flux closure is a discrete operator check, not empirical conservation of total atmospheric water.",
                              "Two Atlantic regions are limited-scope diagnostics; no global causal generalization."]}
    return report, timestamps, geo, core, halo, weights, rlat[REGION_IDS], rlon[REGION_IDS], source_segment


def run(args, inspected):
    report, timestamps, geo, core, halo, weights, rlat, rlon, source_segment = inspected
    check(args.plan is not None and args.plan.is_file(), "--execute requires an existing frozen --plan")
    frozen_plan_hash = sha256(args.plan)
    output = args.output / "input_data.npz"
    check(not output.exists(), "Refusing to overwrite existing completed input archive")
    start = time.perf_counter()
    n = len(timestamps)
    local, flux = np.empty((n, 2, 5), dtype=np.float64), np.empty((n, 2, 5), dtype=np.float64)
    grid = np.empty((n, 2, 60, 4), dtype=np.float32)
    fields_qc = {f: {"minimum": float("inf"), "maximum": float("-inf"), "values_checked": 0, "nonfinite": 0} for f in SOURCE_FIELDS}
    qc = {"global_area_integral_relative_cancellation_max": 0.0, "global_area_integral_abs_max_m2_per_s": 0.0,
          "region_external_boundary_mfc_max_abs_difference": 0.0, "region_covariance_centered_max_abs_difference": 0.0}
    by_segment = []
    offset = 0
    for si, segment in enumerate(SEGMENTS):
        rows = {r["field"]: r for r in report["sources"] if r["segment"] == si}
        with ExitStack() as stack:
            arrays = {f: stack.enter_context(xr.open_dataset(rows[f]["path"]))[rows[f]["variable"]]
                      .transpose("time", "latitude", "longitude") for f in SOURCE_FIELDS}
            count = arrays["u"].shape[0]
            for begin in range(0, count, args.chunk_size):
                end = min(begin + args.chunk_size, count)
                destination = slice(offset + begin, offset + end)
                raw = {f: np.asarray(da.isel(time=slice(begin, end)).values, dtype=np.float64)
                       for f, da in arrays.items()}
                for f, x in raw.items():
                    nonfinite = int((~np.isfinite(x)).sum())
                    fields_qc[f]["nonfinite"] += nonfinite
                    check(nonfinite == 0, f"Nonfinite {f} in {segment} at {begin}")
                    fields_qc[f]["values_checked"] += int(x.size)
                    fields_qc[f]["minimum"] = min(fields_qc[f]["minimum"], float(x.min()))
                    fields_qc[f]["maximum"] = max(fields_qc[f]["maximum"], float(x.max()))
                check(raw["humidity"].min() >= 0 and raw["humidity"].max() <= .1, "Humidity outside broad physical range")
                check(raw["cloud_cover"].min() >= 0 and raw["cloud_cover"].max() <= 1, "Cloud outside fraction range")
                check(raw["temperature"].min() > 150 and raw["temperature"].max() < 350, "Temperature outside broad range")
                check(max(np.abs(raw["u"]).max(), np.abs(raw["v"]).max()) < 150, "Wind outside broad range")
                qu, qv = raw["humidity"]*raw["u"], raw["humidity"]*raw["v"]
                div = divergence(qu, qv, geo)
                weighted_div = div * geo["area"]
                integral = np.abs(weighted_div.sum(axis=(-2, -1)))
                relative = integral / np.maximum(np.abs(weighted_div).sum(axis=(-2, -1)), 1e-100)
                qc["global_area_integral_relative_cancellation_max"] = max(qc["global_area_integral_relative_cancellation_max"], float(relative.max()))
                qc["global_area_integral_abs_max_m2_per_s"] = max(qc["global_area_integral_abs_max_m2_per_s"], float(integral.max()))
                check(relative.max() < 1e-12, "Discrete global boundary-flux cancellation failed")
                for ri in range(2):
                    indices, w = core[ri], weights[ri]
                    selected = {f: x.reshape(len(x), -1)[:, indices] for f, x in raw.items()}
                    means = {f: (x*w).sum(axis=1) for f, x in selected.items()}
                    for fi, f in enumerate(SOURCE_FIELDS):
                        local[destination, ri, fi] = means[f]
                        if fi < 4:
                            grid[destination, ri, :, fi] = raw[f].reshape(end-begin, -1)[:, halo[ri]]
                    mean_qu, mean_qv = means["humidity"]*means["u"], means["humidity"]*means["v"]
                    weighted_qu = (qu.reshape(end-begin, -1)[:, indices]*w).sum(axis=1)
                    weighted_qv = (qv.reshape(end-begin, -1)[:, indices]*w).sum(axis=1)
                    cov_qu, cov_qv = weighted_qu-mean_qu, weighted_qv-mean_qv
                    direct_cov = [((selected["humidity"]-means["humidity"][:, None])
                                   * (selected[f]-means[f][:, None])*w).sum(axis=1) for f in ["u", "v"]]
                    cov_difference = max(float(np.abs(cov_qu-direct_cov[0]).max()), float(np.abs(cov_qv-direct_cov[1]).max()))
                    qc["region_covariance_centered_max_abs_difference"] = max(qc["region_covariance_centered_max_abs_difference"], cov_difference)
                    check(cov_difference < 1e-14, "Weighted covariance identity failed")
                    mfc = -(div.reshape(end-begin, -1)[:, indices]*w).sum(axis=1)
                    direct_mfc = direct_region_mfc(qu, qv, indices, geo)
                    difference = float(np.abs(mfc-direct_mfc).max())
                    qc["region_external_boundary_mfc_max_abs_difference"] = max(qc["region_external_boundary_mfc_max_abs_difference"], difference)
                    check(difference < 1e-18, "Regional divergence disagrees with external-boundary flux")
                    flux[destination, ri] = np.stack([mean_qu, mean_qv, cov_qu, cov_qv, mfc], axis=1)
                if begin == 0 or end == count or (begin // args.chunk_size) % 25 == 0:
                    status = {"stage": "processing", "source_segment": si, "done": offset+end, "total": n, "updated_utc": now()}
                    save_json(args.output / "status.json", status)
                    print(json.dumps(status), flush=True)
        by_segment.append({"source_segment": si, "start_index": offset, "stop_index_exclusive": offset+count})
        offset += count
    check(offset == n and np.isfinite(local).all() and np.isfinite(flux).all() and np.isfinite(grid).all(), "Incomplete inputs")
    with np.load(args.controls / "region_controls_trainfit.npz", allow_pickle=False) as d:
        controls = d["physical"][:, REGION_IDS, :].copy()
    check(np.isfinite(controls).all(), "Nonfinite physical controls")
    terrain = []
    for ri, rid in enumerate(REGION_IDS):
        sp = controls[:, ri, CONTROL_NAMES.index("surface_pressure")]
        for part in by_segment:
            x = sp[part["start_index"]:part["stop_index_exclusive"]]
            terrain.append({"region_id": int(rid), "source_segment": part["source_segment"], "samples": int(len(x)),
                            "minimum_region_mean_surface_pressure_pa": float(x.min()),
                            "maximum_region_mean_surface_pressure_pa": float(x.max()),
                            "region_mean_below_850hpa_samples": int((x < 85000).sum()),
                            "fraction": float((x < 85000).mean())})
    temporary = args.output / "input_data.partial.npz"
    np.savez_compressed(temporary, timestamps=timestamps, region_ids=REGION_IDS, region_lat=rlat, region_lon=rlon,
                        local=local, local_names=np.array(LOCAL_NAMES), flux=flux, flux_names=np.array(FLUX_NAMES),
                        grid=grid, grid_names=np.array(GRID_NAMES), grid_indices=halo, area_weights=weights,
                        region_grid_indices=core, source_segment=source_segment, controls=controls,
                        control_names=np.array(CONTROL_NAMES), grid_lat=np.rad2deg(geo["lat"]),
                        grid_lon=np.rad2deg(geo["lon"]), local_units=np.array(["m/s", "m/s", "kg/kg", "K", "fraction"]),
                        flux_units=np.array(["m/s", "m/s", "m/s", "m/s", "s^-1"]),
                        grid_units=np.array(["m/s", "m/s", "kg/kg", "K"]),
                        control_units=np.array(report["controls"]["units"]))
    check(temporary.stat().st_size < 400_000_000, "Input archive exceeds storage budget")
    check(sha256(args.plan) == frozen_plan_hash, "Frozen plan changed while features were computed")
    os.replace(temporary, output)
    report.update({"status": "completed", "completed_utc": now(), "elapsed_seconds": time.perf_counter()-start,
                   "frozen_plan": {"path": str(args.plan.resolve()), "sha256": frozen_plan_hash},
                   "python": platform.python_version(), "numpy": np.__version__, "xarray": xr.__version__,
                   "chunk_size": args.chunk_size, "fields_qc": fields_qc, "operator_data_checks": qc,
                   "coarse_region_surface_pressure_diagnostic": terrain,
                   "terrain_diagnostic_limitation": "Arithmetic regional mean surface pressure only. Even zero counts below 85000 Pa do not establish that every native/source grid cell is above ground.",
                   "source_segments": by_segment,
                   "peak_working_set_bytes": int(getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)),
                   "outputs": {"input_data.npz": {"bytes": output.stat().st_size, "sha256": sha256(output)}}})
    save_json(args.output / "input_manifest.json", report)
    save_json(args.output / "status.json", {"stage": "completed", "updated_utc": now(), "manifest_sha256": sha256(args.output / "input_manifest.json")})
    print(json.dumps({"status": "completed", "output_bytes": output.stat().st_size, "elapsed_seconds": report["elapsed_seconds"], "qc": qc}), flush=True)


def main():
    base = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--metadata-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("D:/Paper2/vipuser/Data"))
    parser.add_argument("--reference", type=Path, default=base / "revision_outputs/inputs")
    parser.add_argument("--controls", type=Path, default=base / "revision_outputs/physical_controls_inputs")
    parser.add_argument("--output", type=Path, default=base / "revision_outputs/transport_consistency_v1/inputs")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    check(1 <= args.chunk_size <= 1024, "Chunk size must be 1--1024")
    args.output.mkdir(parents=True, exist_ok=True)
    inspected = inspect(args)
    save_json(args.output / "metadata_inspection.json", inspected[0])
    print(json.dumps({"metadata": "passed", "shape": inspected[0]["shape"]}), flush=True)
    if args.execute:
        run(args, inspected)


if __name__ == "__main__":
    main()
