"""Regional wind-humidity eddy convergence (WHEC) at 850 hPa.

Exact increment identity of the horizontal moisture-flux operator, with the
calendar-month climatology (ubar, qbar; fitted on 1979-2018) as base state and
the anomalies (u', q') as increment:

    div((ubar+u')(qbar+q')) = div(ubar qbar) + [div(ubar q') + div(u' qbar)] + div(u' q').

WHEC_r(t) is minus the area mean over region r of the quadratic term, i.e. the
net import of moisture by the wind-humidity anomaly covariance across the
region boundary divided by the region area (kg kg-1 s-1). The linear term
(LIN) and a placebo whose humidity anomaly is taken 1,460 six-hour steps
(365 days) earlier (PLAC) use the same finite-volume operator. Face fluxes are
centred averages of cell-centre fluxes, so interior faces cancel exactly and
the global sum vanishes. No temporal interpolation; nonfinite inputs fail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.prepare_inputs import SEGMENTS, region_map, sha256  # noqa: E402

FILES = {"u": "u_component_of_wind_850.nc", "v": "v_component_of_wind_850.nc", "q": "specific_humidity_850.nc"}
EARTH_RADIUS = 6.371e6
PLACEBO_SHIFT = 1460  # six-hour steps = 365 days; keeps hour of day
OUT = ROOT.parent / "revision_outputs"
DATA = ROOT.parent / "supplementary_data" / "era5_primary_native" / "data_root"


def load(data_root, sources):
    fields, times, lat, lon = {}, None, None, None
    for name, fname in FILES.items():
        parts, stamps = [], []
        for segment in SEGMENTS:
            path = data_root / segment / fname
            with xr.open_dataset(path) as ds:
                key = next(k for k in ds.data_vars if {"time", "latitude", "longitude"}.issubset(ds[k].dims))
                da = ds[key].transpose("time", "latitude", "longitude")
                if lat is None:
                    lat, lon = ds.latitude.values.astype(float), ds.longitude.values.astype(float)
                if not (np.allclose(ds.latitude.values, lat, atol=1e-3, rtol=0) and np.allclose(ds.longitude.values, lon, atol=1e-3, rtol=0)):
                    raise ValueError(f"Grid mismatch: {path}")
                values = np.asarray(da.values, dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite values in {path}")
                parts.append(values)
                stamps.append(np.asarray(ds.time.values, dtype="datetime64[ns]"))
                sources.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "variable": key,
                                "units": str(da.attrs.get("units", "unspecified")), "n_timestamps": int(values.shape[0])})
        t = np.concatenate(stamps)
        if times is None:
            times = t
        if not np.array_equal(t, times) or np.any(np.diff(t) != np.timedelta64(6, "h")):
            raise ValueError(f"Calendar mismatch or gap for {name}")
        fields[name] = np.concatenate(parts)
        print(f"loaded {name} {fields[name].shape}", flush=True)
    if not (np.all(np.diff(lat) > 0) and np.allclose(np.diff(lon), 360 / len(lon))):
        raise ValueError("Expected ascending latitude and uniform longitude")
    return fields, times, lat, lon


class FiniteVolume:
    """Centred face fluxes on the regular latitude-longitude grid; poles carry no flux."""

    def __init__(self, lat, lon):
        phi = np.radians(lat)
        edges = np.concatenate([[-np.pi / 2], 0.5 * (phi[1:] + phi[:-1]), [np.pi / 2]])
        self.dlam = 2 * np.pi / len(lon)
        self.area = EARTH_RADIUS ** 2 * self.dlam * (np.sin(edges[1:]) - np.sin(edges[:-1]))  # per latitude row
        self.len_east = EARTH_RADIUS * (edges[1:] - edges[:-1])  # meridional length of zonal faces
        self.len_north = EARTH_RADIUS * np.cos(edges[1:-1]) * self.dlam  # interior north faces, rows j|j+1

    def face_fluxes(self, fx, fy):
        east = 0.5 * (fx + np.roll(fx, -1, axis=2)) * self.len_east[None, :, None]
        north = 0.5 * (fy[:, :-1] + fy[:, 1:]) * self.len_north[None, :, None]
        return east, north

    def net_outward(self, fx, fy):
        east, north = self.face_fluxes(fx, fy)
        out = east - np.roll(east, 1, axis=2)
        out[:, :-1] += north
        out[:, 1:] -= north
        return out


def explicit_boundary_outward(east, north, member):
    """Region outward flux from boundary faces only (independent of the cell telescoping)."""
    total = np.zeros(east.shape[0])
    inside_e = member & ~np.roll(member, -1, axis=1)
    outside_e = ~member & np.roll(member, -1, axis=1)
    total += east[:, inside_e].sum(axis=1) - east[:, outside_e].sum(axis=1)
    inside_n = member[:-1] & ~member[1:]
    outside_n = ~member[:-1] & member[1:]
    total += north[:, inside_n].sum(axis=1) - north[:, outside_n].sum(axis=1)
    return total


def standardize(series, months, training):
    clim = np.stack([series[training & (months == m)].mean(axis=0) for m in range(1, 13)])
    anom = series - clim[months - 1]
    mu, sd = anom[training].mean(axis=0), anom[training].std(axis=0)
    return (anom - mu) / sd, {"climatology": clim, "mean": mu, "std": sd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA)
    ap.add_argument("--output", type=Path, default=OUT / "whec_index")
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sources = []
    f, times, lat, lon = load(args.data_root, sources)
    n_time, n_lat, n_lon = f["q"].shape
    months = pd.DatetimeIndex(times).month.to_numpy()
    training = times < np.datetime64("2019-01-01")
    assert n_time == 68668 and training.sum() == 58440

    mapping, rlat, rlon = region_map(lat, lon)
    ref = np.load(OUT / "inputs" / "trainfit_parameters.npz")
    if not np.array_equal(mapping, ref["node_to_region"]):
        raise ValueError("Region mapping differs from the analysis inputs")
    member = mapping.reshape(n_lat, n_lon)
    n_reg = int(member.max()) + 1
    fv = FiniteVolume(lat, lon)
    cell_area = np.repeat(fv.area[:, None], n_lon, axis=1)
    region_area = np.array([cell_area[member == r].sum() for r in range(n_reg)])
    onehot = np.zeros((n_lat * n_lon, n_reg))
    onehot[np.arange(n_lat * n_lon), mapping] = 1.0

    clim = {k: np.stack([f[k][training & (months == m)].mean(axis=0, dtype=np.float64) for m in range(1, 13)]) for k in FILES}
    print("climatology done", flush=True)

    out_q = np.empty((n_time, n_reg)); out_l = np.empty((n_time, n_reg)); out_p = np.empty((n_time, n_reg))
    global_residual = np.empty((n_time, 3)); global_scale = np.empty((n_time, 3))
    zonal_vq = np.zeros(n_lat); zonal_uq = np.zeros(n_lat)
    boundary_check = {}

    def anomalies(idx):
        m = months[idx] - 1
        return {k: f[k][idx].astype(np.float64) - clim[k][m] for k in FILES}, {k: clim[k][m] for k in FILES}

    for lo in range(0, n_time, args.chunk):
        idx = np.arange(lo, min(lo + args.chunk, n_time))
        a, base = anomalies(idx)
        a_lag, _ = anomalies((idx - PLACEBO_SHIFT) % n_time)
        terms = {
            "q": (a["u"] * a["q"], a["v"] * a["q"]),
            "l": (base["u"] * a["q"] + a["u"] * base["q"], base["v"] * a["q"] + a["v"] * base["q"]),
            "p": (a["u"] * a_lag["q"], a["v"] * a_lag["q"]),
        }
        for k, (name, (fx, fy)) in enumerate(terms.items()):
            net = fv.net_outward(fx, fy)
            global_residual[idx, k] = net.reshape(len(idx), -1).sum(axis=1)
            global_scale[idx, k] = np.abs(net).reshape(len(idx), -1).sum(axis=1)
            region_out = net.reshape(len(idx), -1) @ onehot
            target = {"q": out_q, "l": out_l, "p": out_p}[name]
            target[idx] = -region_out / region_area
            if lo == 0:
                east, north = fv.face_fluxes(fx[:200], fy[:200])
                explicit = np.column_stack([explicit_boundary_outward(east, north, member == r) for r in range(n_reg)])
                boundary_check[name] = float(np.max(np.abs(explicit - region_out[:200])) / np.max(np.abs(region_out[:200])))
        tr = training[idx]
        zonal_vq += (a["v"][tr] * a["q"][tr]).mean(axis=2).sum(axis=0)
        zonal_uq += (a["u"][tr] * a["q"][tr]).mean(axis=2).sum(axis=0)
        print(f"fluxes {idx[-1] + 1}/{n_time} {time.perf_counter() - start:.0f}s", flush=True)
    zonal_vq /= training.sum(); zonal_uq /= training.sum()

    std_q, par_q = standardize(out_q, months, training)
    std_l, par_l = standardize(out_l, months, training)
    std_p, par_p = standardize(out_p, months, training)
    band_index = np.floor((np.clip(rlat, -89.999, 89.999) + 90) / 30).astype(int)
    lon_bin = np.arange(n_reg) - band_index * 11
    remote = band_index * 11 + (lon_bin + 5) % 11
    assert np.all(band_index[remote] == band_index) and np.all(remote != np.arange(n_reg))

    np.savez_compressed(args.output / "whec_regional.npz", timestamps=times, lat=rlat, lon=rlon, remote_region=remote,
                        region_area_m2=region_area, whec_phys=out_q, lin_phys=out_l, placebo_phys=out_p,
                        whec=std_q, lin=std_l, placebo=std_p, node_to_region=mapping)
    np.savez_compressed(args.output / "whec_parameters.npz", **{f"{n}__{k}": v for n, par in (("whec", par_q), ("lin", par_l), ("placebo", par_p)) for k, v in par.items()},
                        grid_lat=lat, grid_lon=lon, cell_area_row=fv.area)
    pd.DataFrame({"latitude": lat, "mean_vq_eddy_m_s_g_kg": zonal_vq * 1e3, "mean_uq_eddy_m_s_g_kg": zonal_uq * 1e3}).to_csv(
        args.output / "zonal_mean_eddy_moisture_flux_1979_2018.csv", index=False)

    bands = {"0_30": np.abs(rlat) < 30, "30_60": (np.abs(rlat) >= 30) & (np.abs(rlat) < 60), "60_90": np.abs(rlat) >= 60}
    def band_sd(x):
        return {b: float(np.median(x[training][:, m].std(axis=0))) for b, m in bands.items()}
    corr = lambda a, b: float(np.median([np.corrcoef(a[training, r], b[training, r])[0, 1] for r in range(n_reg)]))
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "python": platform.python_version(), "numpy": np.__version__,
        "code_sha256": sha256(Path(__file__)), "sources": sources, "n_time": int(n_time), "training_n": int(training.sum()),
        "base_state": "calendar-month climatology per grid cell, 1979-2018", "placebo_shift_steps": PLACEBO_SHIFT,
        "placebo_wrap": "humidity anomaly from 365 days earlier; the first 365 days of 1979 wrap to the end of the record",
        "remote_region": "same 30-degree latitude band, longitude bin +5 of 11 (about 164 degrees away)",
        "global_closure_max_relative": {k: float(np.max(np.abs(global_residual[:, i]) / global_scale[:, i])) for i, k in enumerate(("whec", "lin", "placebo"))},
        "explicit_boundary_check_max_relative": boundary_check,
        "median_regional_sd_kg_kg_s_by_band": {"whec": band_sd(out_q), "lin": band_sd(out_l), "placebo": band_sd(out_p)},
        "median_regional_correlation_training": {"whec_lin": corr(std_q, std_l), "whec_placebo": corr(std_q, std_p), "lin_placebo": corr(std_l, std_p)},
        "zonal_mean_vq_by_hemisphere_midlatitude_g_kg_m_s": {"NH_30_60": float(zonal_vq[(lat > 30) & (lat < 60)].mean() * 1e3),
                                                             "SH_30_60": float(zonal_vq[(lat < -30) & (lat > -60)].mean() * 1e3)},
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("global_closure_max_relative", "explicit_boundary_check_max_relative", "median_regional_sd_kg_kg_s_by_band",
                                                "median_regional_correlation_training", "zonal_mean_vq_by_hemisphere_midlatitude_g_kg_m_s", "elapsed_seconds")}, indent=1))


if __name__ == "__main__":
    main()
