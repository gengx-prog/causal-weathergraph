"""Same-date check of the native-conservative CDS route against WB2, beside the coarse route.

On the 152 pre-splice timestamps of run_source_overlap_comparison.py, two routes
for the 2023-01-11--2025 continuation are compared with the WeatherBench 2
conservative product: (i) the CDS 5.625-degree interpolation used so far and
(ii) native 0.25-degree CDS fields remapped with the WB2 conservative method
(download_era5_primary_native.py, regrid_era5_primary_native.py). Differences are
reported in physical units, in discovery-fitted standardized anomalies and after
aggregation to the 66 analysis regions, with the frozen 1979--2018 parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import xarray as xr

from run_source_overlap_comparison import diff_stats, load_cds, load_wb2, regional, sha256, standardize

NATIVE_FILES = {"temperature": "temperature_850", "humidity": "specific_humidity_850",
                "u": "u_component_of_wind_850", "v": "v_component_of_wind_850", "cloud_cover": "total_cloud_cover"}
ROUTES = {"cds_5p625_interpolation": "CDS 5.625-degree interpolation (used so far)",
          "cds_native_conservative": "CDS native 0.25 degrees + WB2 conservative remapping",
          "cds_grid_box_average": "CDS (MARS) grid-box-average onto the WB2 grid, server-side"}


def load_native(directory: Path, times: np.ndarray):
    fields, lat, lon, sources = {}, None, None, []
    for name, var in NATIVE_FILES.items():
        path = directory / f"{var}.nc"
        sources.append(path)
        with xr.open_dataset(path) as ds:
            da = ds[var].sel(time=times).transpose("time", "latitude", "longitude")
            if not np.array_equal(da.time.values.astype("datetime64[ns]"), times):
                raise ValueError(f"Native-route time selection mismatch for {name}")
            lat, lon = ds.latitude.values, ds.longitude.values
            fields[name] = np.asarray(da.values, dtype=np.float64).reshape(len(times), -1)
    return fields, lat, lon, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap-dir", type=Path, default=Path(r"D:\Paper2\Major Revision\supplementary_data\era5_source_overlap_2022_2023"))
    ap.add_argument("--native-dir", type=Path, default=Path(r"D:\Paper2\Major Revision\supplementary_data\era5_primary_native\overlap_64x32"))
    ap.add_argument("--params", type=Path, default=Path(r"D:\Paper2\Major Revision\revision_outputs\inputs\trainfit_parameters.npz"))
    ap.add_argument("--output", type=Path, default=Path(r"D:\Paper2\Major Revision\revision_outputs\source_overlap_native_route"))
    ap.add_argument("--label", choices=["cds_native_conservative", "cds_grid_box_average"], default="cds_native_conservative")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    coarse, times, lat_c, lon_c, coarse_sources = load_cds(args.overlap_dir)
    wb2, lat_w, lon_w, wb2_sources = load_wb2(times)
    native, lat_n, lon_n, native_sources = load_native(args.native_dir, times)
    # The coarse CDS grid matches WB2 to 1e-3 degrees (as in run_source_overlap_comparison.py);
    # the native route is remapped onto the WB2 coordinates themselves.
    for (lat, lon), tol in (((lat_c, lon_c), 1e-3), ((lat_n, lon_n), 1e-9)):
        if np.max(np.abs(lat - lat_w)) > tol or np.max(np.abs(lon - lon_w)) > tol:
            raise ValueError("Grid mismatch with WB2")
    for src in (coarse, wb2, native):
        src["wind"] = np.sqrt(src["u"] ** 2 + src["v"] ** 2)
        if not all(np.isfinite(v).all() for v in src.values()):
            raise ValueError("Nonfinite values")

    params = dict(np.load(args.params))
    mapping = params["node_to_region"]
    months = pd.DatetimeIndex(times).month.to_numpy()
    units = {"temperature": "K", "humidity": "kg kg-1", "wind": "m s-1", "u": "m s-1", "v": "m s-1", "cloud_cover": "fraction"}
    rows = []
    for route, src in (("cds_5p625_interpolation", coarse), (args.label, native)):
        for name in ["temperature", "humidity", "wind", "u", "v", "cloud_cover"]:
            z_c, z_w = standardize(src[name], months, params, name), standardize(wb2[name], months, params, name)
            r_c, r_w = regional(z_c, mapping), regional(z_w, mapping)
            corr = np.array([np.corrcoef(r_c[:, r], r_w[:, r])[0, 1] for r in range(r_c.shape[1])])
            for scale, st in (("physical", diff_stats(src[name], wb2[name])), ("grid_standardized", diff_stats(z_c, z_w)),
                              ("regional_standardized", diff_stats(r_c, r_w))):
                rows.append({"route": route, "variable": name, "scale": scale,
                             "units": units[name] if scale == "physical" else "discovery SD", **st})
            rows[-1].update({"regional_max_abs_bias": float(np.max(np.abs((r_c - r_w).mean(axis=0)))),
                             "regional_min_corr": float(corr.min()), "regional_median_corr": float(np.median(corr))})
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "route_comparison.csv", index=False)
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "routes": {k: ROUTES[k] for k in ("cds_5p625_interpolation", args.label)},
        "n_timestamps": int(len(times)), "first": str(times[0]), "last": str(times[-1]),
        "frozen_parameters": str(args.params), "frozen_parameters_sha256": sha256(args.params),
        "inputs": {str(p): sha256(p) for p in coarse_sources + wb2_sources + native_sources},
        "code_sha256": sha256(Path(__file__)), "python": platform.python_version(),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "limitations": ["All routes derive from ERA5; agreement is processing consistency, not independent validation.",
                        "Five pre-splice windows; homogeneity after the splice is inferred from identical processing."],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(table[["route", "variable", "scale", "mean_difference_cds_minus_wb2", "rmse", "pearson_r", "rmse_over_wb2_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
