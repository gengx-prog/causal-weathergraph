"""Same-date comparison of the two ERA5 acquisition routes used in the main record.

The 1979--2023-01-10 record comes from the WeatherBench 2 (WB2) conservative
64x32 product; the 2023-01-11--2025 continuation was requested from the CDS with
its own 5.625-degree interpolation. Here, both routes are compared on 152
identical six-hourly timestamps (2022 Jan/Apr/Jul/Oct days 1-7 and 2023 Jan
days 1-10) that precede the splice. The comparison is reported in physical
units, in discovery-fitted standardized anomalies, and after aggregation to the
66 analysis regions, i.e. in the units entering the lagged regressions.

Nothing is refitted: climatologies, anomaly means and scales are the frozen
1979--2018 parameters of revision_outputs/inputs/trainfit_parameters.npz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import xarray as xr

WB2_DIR = Path(r"D:\Paper2\vipuser\Data\weatherbench2_era5_6h_64x32_850hPa_2019_2023-01-10")
WB2_FILES = {
    "temperature": ("temperature_850.nc", "temperature_850"),
    "humidity": ("specific_humidity_850.nc", "specific_humidity_850"),
    "u": ("u_component_of_wind_850.nc", "u_component_of_wind_850"),
    "v": ("v_component_of_wind_850.nc", "v_component_of_wind_850"),
    "cloud_cover": ("total_cloud_cover.nc", "total_cloud_cover"),
}
CDS_KEYS = {"temperature": "t", "humidity": "q", "u": "u", "v": "v", "cloud_cover": "tcc"}
WINDOWS = ["202201_01_07", "202204_01_07", "202207_01_07", "202210_01_07", "202301_01_10"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_cds(overlap_dir: Path):
    fields, times, lat, lon, sources = {k: [] for k in CDS_KEYS}, [], None, None, []
    for window in WINDOWS:
        primary = overlap_dir / "files" / f"primary_850_coarse_{window}.nc"
        cloud = overlap_dir / "files" / f"cloud_cover_coarse_{window}.nc"
        sources += [primary, cloud]
        with xr.open_dataset(primary) as p, xr.open_dataset(cloud) as c:
            p = p.sortby("latitude")
            c = c.sortby("latitude")
            if not np.array_equal(p.valid_time.values, c.valid_time.values):
                raise ValueError(f"Primary/cloud time mismatch in {window}")
            times.append(p.valid_time.values)
            if lat is None:
                lat, lon = p.latitude.values, p.longitude.values
            for name, key in CDS_KEYS.items():
                ds = c if key == "tcc" else p
                da = ds[key]
                if "pressure_level" in da.dims:
                    da = da.sel(pressure_level=850.0)
                da = da.transpose("valid_time", "latitude", "longitude")
                fields[name].append(np.asarray(da.values, dtype=np.float64).reshape(da.shape[0], -1))
    t = np.concatenate(times).astype("datetime64[ns]")
    return {k: np.concatenate(v) for k, v in fields.items()}, t, lat, lon, sources


def load_wb2(times: np.ndarray):
    fields, lat, lon, sources = {}, None, None, []
    for name, (fname, key) in WB2_FILES.items():
        path = WB2_DIR / fname
        sources.append(path)
        with xr.open_dataset(path) as ds:
            da = ds[key].sel(time=times).transpose("time", "latitude", "longitude")
            if not np.array_equal(da.time.values.astype("datetime64[ns]"), times):
                raise ValueError(f"WB2 time selection mismatch for {name}")
            lat, lon = ds.latitude.values, ds.longitude.values
            fields[name] = np.asarray(da.values, dtype=np.float64).reshape(len(times), -1)
    return fields, lat, lon, sources


def standardize(x, months, params, name):
    clim = params[f"{name}__climatology"]
    return (x - clim[months - 1] - params[f"{name}__mean"]) / params[f"{name}__std"]


def regional(z, mapping):
    return np.column_stack([z[:, mapping == r].mean(axis=1) for r in np.unique(mapping)])


def diff_stats(a_cds, b_wb2):
    d = a_cds - b_wb2
    a, b = a_cds.ravel(), b_wb2.ravel()
    return {
        "mean_difference_cds_minus_wb2": float(d.mean()),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "mae": float(np.mean(np.abs(d))),
        "max_abs_difference": float(np.max(np.abs(d))),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "wb2_sd": float(b.std()),
        "rmse_over_wb2_sd": float(np.sqrt(np.mean(d ** 2)) / b.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap-dir", type=Path, default=Path(r"D:\Paper2\Major Revision\supplementary_data\era5_source_overlap_2022_2023"))
    ap.add_argument("--params", type=Path, default=Path(r"D:\Paper2\Major Revision\revision_outputs\inputs\trainfit_parameters.npz"))
    ap.add_argument("--regions", type=Path, default=Path(r"D:\Paper2\Major Revision\revision_outputs\inputs\region_trainfit.npz"))
    ap.add_argument("--output", type=Path, default=Path(r"D:\Paper2\Major Revision\revision_outputs\source_overlap_comparison"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    cds, times, lat_c, lon_c, cds_sources = load_cds(args.overlap_dir)
    if len(times) != 152 or len(np.unique(times)) != 152:
        raise ValueError("Expected 152 unique overlap timestamps")
    wb2, lat_w, lon_w, wb2_sources = load_wb2(times)
    if np.max(np.abs(lat_c - lat_w)) > 1e-3 or np.max(np.abs(lon_c - lon_w)) > 1e-3:
        raise ValueError("Grid mismatch between CDS coarse route and WB2")
    for src in (cds, wb2):
        src["wind"] = np.sqrt(src["u"] ** 2 + src["v"] ** 2)
        for k, v in src.items():
            if not np.isfinite(v).all():
                raise ValueError(f"Nonfinite values in {k}")

    params = dict(np.load(args.params))
    mapping = params["node_to_region"]
    months = pd.DatetimeIndex(times).month.to_numpy()
    grid_lat = np.repeat(lat_w, len(lon_w))

    rows, region_rows, state_rows = [], [], []
    units = {"temperature": "K", "humidity": "kg kg-1", "wind": "m s-1", "u": "m s-1", "v": "m s-1", "cloud_cover": "fraction"}
    regional_z = {}
    for name in ["temperature", "humidity", "wind", "u", "v", "cloud_cover"]:
        phys = diff_stats(cds[name], wb2[name])
        z_c, z_w = standardize(cds[name], months, params, name), standardize(wb2[name], months, params, name)
        zst = diff_stats(z_c, z_w)
        r_c, r_w = regional(z_c, mapping), regional(z_w, mapping)
        regional_z[name] = (r_c, r_w)
        rst = diff_stats(r_c, r_w)
        bias_by_region = (r_c - r_w).mean(axis=0)
        rmse_by_region = np.sqrt(((r_c - r_w) ** 2).mean(axis=0))
        corr_by_region = np.array([np.corrcoef(r_c[:, r], r_w[:, r])[0, 1] for r in range(r_c.shape[1])])
        for scale, st in (("physical", phys), ("grid_standardized", zst), ("regional_standardized", rst)):
            rows.append({"variable": name, "scale": scale, "units": units[name] if scale == "physical" else "discovery SD", **st})
        rows[-1].update({
            "regional_median_abs_bias": float(np.median(np.abs(bias_by_region))),
            "regional_max_abs_bias": float(np.max(np.abs(bias_by_region))),
            "regional_median_rmse": float(np.median(rmse_by_region)),
            "regional_min_corr": float(corr_by_region.min()),
            "regional_median_corr": float(np.median(corr_by_region)),
        })
        for band, mask in (("tropical", np.abs(grid_lat) < 23.5), ("midlatitude", (np.abs(grid_lat) >= 23.5) & (np.abs(grid_lat) < 60)), ("polar", np.abs(grid_lat) >= 60)):
            bst = diff_stats(z_c[:, mask], z_w[:, mask])
            rows.append({"variable": name, "scale": f"grid_standardized_{band}", "units": "discovery SD", **bst})
        for r in range(r_c.shape[1]):
            region_rows.append({"variable": name, "region": r, "bias_cds_minus_wb2": float(bias_by_region[r]), "rmse": float(rmse_by_region[r]), "pearson_r": float(corr_by_region[r])})

    # Effect of the route on the regime indices and on high-state membership.
    with np.load(args.regions, allow_pickle=True) as reg:
        data, ts = reg["data"], reg["timestamps"]
    train = ts < np.datetime64("2019-01-01")
    for name, var_idx in (("humidity", 1), ("cloud_cover", 3)):
        g_train = data[train, :, var_idx].mean(axis=1)
        q80 = float(np.quantile(g_train, 0.8))
        r_c, r_w = regional_z[name]
        g_c, g_w = r_c.mean(axis=1), r_w.mean(axis=1)
        state_rows.append({
            "index": name, "discovery_q80": q80,
            "mean_index_cds": float(g_c.mean()), "mean_index_wb2": float(g_w.mean()),
            "mean_index_difference": float((g_c - g_w).mean()), "index_rmse": float(np.sqrt(np.mean((g_c - g_w) ** 2))),
            "index_corr": float(np.corrcoef(g_c, g_w)[0, 1]),
            "high_state_fraction_cds": float((g_c >= q80).mean()), "high_state_fraction_wb2": float((g_w >= q80).mean()),
            "membership_disagreements": int(((g_c >= q80) != (g_w >= q80)).sum()), "n_times": int(len(g_c)),
        })

    pd.DataFrame(rows).to_csv(args.output / "variable_route_differences.csv", index=False)
    pd.DataFrame(region_rows).to_csv(args.output / "regional_route_differences.csv", index=False)
    pd.DataFrame(state_rows).to_csv(args.output / "regime_index_route_differences.csv", index=False)
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "purpose": "Same-date WB2 conservative versus CDS 5.625-degree interpolation comparison on 152 pre-splice timestamps.",
        "n_timestamps": int(len(times)), "first": str(times[0]), "last": str(times[-1]),
        "frozen_parameters": str(args.params), "frozen_parameters_sha256": sha256(args.params),
        "inputs": {str(p): sha256(p) for p in cds_sources + wb2_sources},
        "code_sha256": sha256(Path(__file__)), "python": platform.python_version(),
        "elapsed_seconds": time.perf_counter() - start,
        "limitations": [
            "Both routes derive from ERA5; agreement is processing consistency, not independent validation.",
            "Five one-week windows do not establish homogeneity in every season of 2023-2025.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["variable", "scale", "mean_difference_cds_minus_wb2", "rmse", "pearson_r", "rmse_over_wb2_sd"]].to_string())
    print(pd.DataFrame(state_rows).to_string())


if __name__ == "__main__":
    main()
