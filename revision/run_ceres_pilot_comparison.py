"""Frozen, descriptive January 2019 CERES/ERA5 preprocessing feasibility pilot.

Run --freeze first (coordinates and metadata only), then --run. No downloads.
This comparison does not test causal graphs or establish external validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from prepare_inputs import region_map

REVISION = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REVISION / "revision_outputs/ceres_pilot_comparison"
PILOT = REVISION / "supplementary_data/cloud_validation/ceres_syn1deg_ed42_pilot_201901"
MANIFEST = REVISION / "revision_outputs/inputs/data_manifest.json"
EARTH_RADIUS_M = 6_371_000.0
REGIONS = (("north_atlantic", 48), ("south_atlantic", 15))
SCHEMES = ("ceres_24_slots_vs_era5_4_times", "ceres_4_halfhour_slots_vs_era5_4_times")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def utc_now():
    return pd.Timestamp.now(tz="UTC").isoformat()


def source_paths():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    era5 = next(s for s in manifest["sources"] if "2019_2023" in s["path"] and "total_cloud_cover" in s["path"])
    audit = json.loads((PILOT / "validation_independent_v2.json").read_text(encoding="utf-8"))
    ceres = {r["region"]: {"path": str(PILOT / r["path"]), "sha256": r["sha256"]} for r in audit["regions"]}
    return era5, ceres


def axis_edges(centers):
    c = np.asarray(centers, dtype=np.float64)
    if np.any(np.diff(c) <= 0):
        raise ValueError("Target axes must be increasing")
    delta = np.diff(c)
    if not np.allclose(delta, delta[0], rtol=0, atol=1e-10):
        raise ValueError("Expected fixed regular target grid")
    return np.r_[c[0] - delta[0] / 2, (c[:-1] + c[1:]) / 2, c[-1] + delta[-1] / 2]


def geometry(lat, lon, source_lat, source_lon, region):
    mapping, _, _ = region_map(lat, lon)
    indices = np.flatnonzero(mapping == region)
    lat_edges, lon_edges = axis_edges(lat), axis_edges(lon)
    source_lat = np.asarray(source_lat, dtype=np.float64)
    source_lon = np.asarray(source_lon, dtype=np.float64) % 360
    if not (np.allclose(np.diff(np.sort(source_lat)), 1) and np.allclose(np.diff(np.sort(source_lon)), 1)):
        raise ValueError("CERES must be a regular one-degree subset")
    rows, weights, target_areas, kept = [], [], [], []
    for index in indices:
        iy, ix = np.unravel_index(index, (len(lat), len(lon)))
        south, north = lat_edges[iy:iy+2]
        west, east = lon_edges[ix:ix+2]
        dy = np.maximum(0.0, np.sin(np.deg2rad(np.minimum(north, source_lat + .5))) -
                        np.sin(np.deg2rad(np.maximum(south, source_lat - .5))))
        dx = np.maximum(0.0, np.minimum(east, source_lon + .5) - np.maximum(west, source_lon - .5))
        overlap = dy[:, None] * np.deg2rad(dx[None, :])
        area_sr = (np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))) * np.deg2rad(east - west)
        coverage = float(overlap.sum() / area_sr)
        included = bool(abs(coverage - 1) <= 1e-10)
        rows.append({"region_id": region, "flat_grid_index": int(index), "latitude_index": int(iy),
                     "longitude_index": int(ix), "latitude": float(lat[iy]), "longitude": float(lon[ix]),
                     "south": float(south), "north": float(north), "west": float(west), "east": float(east),
                     "area_m2": float(area_sr * EARTH_RADIUS_M ** 2), "ceres_coverage_fraction": coverage,
                     "included": included})
        if included:
            weights.append(overlap.ravel() / area_sr)
            target_areas.append(area_sr)
            kept.append(int(index))
    if not kept:
        raise ValueError("No fully covered target cells")
    return rows, np.asarray(weights), np.asarray(target_areas), np.asarray(kept)


LIMITATIONS = [
    "One January 2019 month and one fixed Atlantic mirror pair only; not a global, multiseason, multiyear, hemispheric-effect or causal-graph validation.",
    "Both regions use the same calendar month and therefore different local seasons; no north/south effect test is performed.",
    "Daily quantities are means of available product samples under explicitly different temporal sampling operators, not verified identical 24-hour physical means.",
    "CERES nominal half-hour labels are assigned using the documented Ed4A hourbox convention, transferred to Ed4.2 as an explicit version-qualified design assumption. Exact Ed4.2 within-hour cloud sampling weights remain unverified.",
    "Hourly cloud fields may contain interpolation; no delivered cloud-specific observation/interpolation quality flags identify which samples are direct observations. No independence claim or sample-based p-value is made.",
    "Spatial cell edges are inferred from regular centers and product/request geometry; payloads do not provide explicit spatial bounds. ERA5 provenance identifies an equiangular conservative grid.",
    "No land/ocean mask is applied. Both products use the same complete target grid cells and spherical area weights; the footprint is not asserted to be ocean-only.",
    "Cloud definitions, retrieval physics, and temporal support differ. Bias, MAE and RMSE denote cross-product discrepancies, not error relative to truth.",
    "Raw physical cloud fractions are compared without bias adjustment, standardization, anomaly/climatology fitting, outcome-based region selection or scheme selection.",
    "The manuscript's existing regional series average grid-standardized anomalies; this raw area-weighted feasibility comparison is a different estimand and cannot validate those causal coefficients.",
    "A single month cannot estimate reliable seasonal anomaly climatologies or an independent long-term external validation interval.",
]


def freeze(output):
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "comparison_plan.json"
    if plan_path.exists():
        raise FileExistsError("Frozen plan already exists; it must not be silently overwritten")
    era5, ceres = source_paths()
    with xr.open_dataset(era5["path"]) as ds:
        lat, lon = ds.latitude.values.copy(), ds.longitude.values.copy()
        era5_attrs = dict(ds[era5["variable"]].attrs)
    geo, inputs = {}, []
    for region, region_id in REGIONS:
        info = ceres[region]
        with netCDF4.Dataset(info["path"]) as ds:
            slat, slon = ds["lat"][:], ds["lon"][:]
        rows, _, _, _ = geometry(lat, lon, slat, slon, region_id)
        geo[region] = {"region_id": region_id, "original_cells": len(rows),
                       "fully_covered_cells": sum(r["included"] for r in rows), "cells": rows}
        actual_hash = sha256(info["path"])
        if actual_hash != info["sha256"]:
            raise ValueError("CERES does not match independent acquisition audit")
        inputs.append({"role": region, "path": info["path"], "bytes": Path(info["path"]).stat().st_size,
                       "sha256": actual_hash})
    era_hash = sha256(era5["path"])
    if era_hash != era5["sha256"]:
        raise ValueError("ERA5 does not match existing input manifest")
    inputs.append({"role": "era5", "path": era5["path"], "bytes": Path(era5["path"]).stat().st_size,
                   "sha256": era_hash, "variable": era5["variable"], "attributes": era5_attrs})
    evidence = [MANIFEST, PILOT / "request_plan.json", PILOT / "validation_independent_v2.json",
                PILOT / "metadata/time_semantics_evidence.json", Path(__file__).with_name("prepare_inputs.py")]
    plan = {"frozen_utc": utc_now(), "status": "frozen_before_loading_matched_era5_cloud_values",
            "purpose": "Descriptive data-processing feasibility; not confirmatory scientific external validation",
            "date_range": ["2019-01-01", "2019-01-31"], "regions": geo, "inputs": inputs,
            "evidence": [{"path": str(p), "sha256": sha256(p)} for p in evidence],
            "spatial_operator": "First map CERES 1-degree cell values by spherical overlap area to fully covered ERA5 target cells, then area-weight identical target cells in both products. Keep original region_map identities; list exclusions if any.",
            "earth_radius_m_for_reported_area": EARTH_RADIUS_M,
            "spatial_support_assumption": "CERES centers plus/minus 0.5 degree; ERA5 regular centers with midpoint boundaries. No land/ocean filtering.",
            "schemes": {
                SCHEMES[0]: {"CERES_nominal_hours": list(range(24)), "ERA5_hours": [0, 6, 12, 18],
                             "meaning": "Daily 24 product-slot mean versus daily four synoptic sample mean"},
                SCHEMES[1]: {"CERES_nominal_hours": [0, 6, 12, 18], "ERA5_hours": [0, 6, 12, 18],
                             "meaning": "Daily four CERES half-hour slots versus four ERA5 whole-hour samples; same-hour rule, not exact temporal equivalence"}},
            "ceres_time_assignment": "Assign each numeric timestamp to its containing whole-hour slot and nominal midpoint; require agreement with fixed Jan2019 slot sequence within 15 seconds. Preserve raw numeric and decoded coordinates, do not edit source. Interpret slots with explicit version-qualified assumption only.",
            "missing_policy": "Preserve source masks. No imputation, clipping or outcome-based cell selection. Require all positive-weight source cells for each remapped cell, all fixed cells for each regional sample, and all requested samples for each daily mean. Report invalid days and complete paired denominators.",
            "metrics": {"unit": "percentage points for differences; percent for means", "bias_sign": "CERES minus ERA5",
                        "list": ["n_paired_days", "mean_CERES", "mean_ERA5", "bias", "MAE", "RMSE", "Pearson_r"],
                        "inference": "No p-values, no independent-sample confidence intervals; both schemes retained regardless of outcomes"},
            "limitations": LIMITATIONS}
    write_json(plan_path, plan)
    print(json.dumps({"plan": str(plan_path), "sha256": sha256(plan_path), "frozen_utc": plan["frozen_utc"],
                      "covered_cells": {r: geo[r]["fully_covered_cells"] for r, _ in REGIONS}}), flush=True)


def metric(ceres, era5):
    mask = np.isfinite(ceres) & np.isfinite(era5)
    x, y = np.asarray(ceres)[mask], np.asarray(era5)[mask]
    delta = x - y
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and x.std() > 0 and y.std() > 0 else None
    return {"n_paired_days": int(mask.sum()), "ceres_mean_percent": float(x.mean()) if len(x) else None,
            "era5_mean_percent": float(y.mean()) if len(y) else None,
            "bias_pp_ceres_minus_era5": float(delta.mean()) if len(x) else None,
            "mae_pp": float(np.abs(delta).mean()) if len(x) else None,
            "rmse_pp": float(np.sqrt(np.mean(delta**2))) if len(x) else None, "pearson_r": r}


def run(output):
    start = time.perf_counter()
    plan_path = output / "comparison_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (output / "run_manifest.json").exists():
        raise FileExistsError("Completed output exists; use a separate output directory for a rerun")
    for item in plan["inputs"] + plan["evidence"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input/evidence changed: {item['path']}")
    input_lookup = {x["role"]: x for x in plan["inputs"]}
    nominal = pd.date_range("2019-01-01 00:30", periods=744, freq="h")
    expected_era = pd.date_range("2019-01-01", periods=124, freq="6h")
    dates = pd.date_range("2019-01-01", periods=31, freq="D")
    era_source = input_lookup["era5"]
    with xr.open_dataset(era_source["path"]) as ds:
        lat, lon = ds.latitude.values.copy(), ds.longitude.values.copy()
        da = ds[era_source["variable"]].sel(time=slice("2019-01-01", "2019-01-31T23:59:59"))
        if not np.array_equal(da.time.values, expected_era.values):
            raise ValueError("ERA5 time coverage differs from frozen 124 synoptic times")
        era_raw = np.asarray(da.transpose("time", "latitude", "longitude").values, dtype=np.float64)
    if np.any((era_raw[np.isfinite(era_raw)] < 0) | (era_raw[np.isfinite(era_raw)] > 1)):
        raise ValueError("ERA5 has values outside physical 0 to 1 range; no clipping permitted")
    daily_rows, metric_rows, quality, grid_rows, sample_rows, time_rows = [], [], {}, [], [], []
    for region, region_id in REGIONS:
        source = input_lookup[region]
        with netCDF4.Dataset(source["path"]) as ds:
            ds.set_auto_maskandscale(False)
            source_lat, source_lon = np.asarray(ds["lat"][:]), np.asarray(ds["lon"][:])
            raw_time = np.asarray(ds["time"][:]).copy()
            decoded = pd.DatetimeIndex([str(t) for t in netCDF4.num2date(raw_time.astype(np.float64), ds["time"].units)])
            v = ds["cldarea_total_1h"]
            cloud_raw = np.asarray(v[:], dtype=np.float64)
            fill = float(v.getncattr("_FillValue"))
        error = (decoded - nominal).total_seconds().to_numpy()
        if len(raw_time) != 744 or not np.all(np.abs(error) <= 15):
            raise ValueError("CERES time values do not match the frozen nominal-hour sequence")
        missing = ~np.isfinite(cloud_raw) | (cloud_raw == fill)
        valid = cloud_raw[~missing]
        if np.any((valid < 0) | (valid > 100)):
            raise ValueError("CERES values outside physical percent range; no clipping permitted")
        rows, weights, area, kept = geometry(lat, lon, source_lat, source_lon, region_id)
        if rows != plan["regions"][region]["cells"]:
            raise ValueError("Grid geometry differs from the frozen plan")
        grid_rows.extend(rows)
        area_weights = area / area.sum()
        flat = cloud_raw.reshape(744, -1)
        flat_missing = missing.reshape(744, -1)
        mapped_missing = flat_missing.astype(np.int32) @ (weights.T > 0).astype(np.int32) > 0
        mapped = np.where(flat_missing, 0, flat) @ weights.T
        mapped[mapped_missing] = np.nan
        era_cells = era_raw.reshape(124, -1)[:, kept] * 100
        era_missing = ~np.isfinite(era_cells)
        ceres_mean = mapped @ area_weights
        era_mean = era_cells @ area_weights
        era_day = era_mean.reshape(31, 4).mean(axis=1)
        ceres_day24 = ceres_mean.reshape(31, 24).mean(axis=1)
        ceres_day4 = ceres_mean.reshape(31, 24)[:, [0, 6, 12, 18]].mean(axis=1)
        for scheme, c in zip(SCHEMES, (ceres_day24, ceres_day4)):
            metric_rows.append({"region": region, "region_id": region_id, "scheme": scheme, **metric(c, era_day)})
        for d, date in enumerate(dates):
            daily_rows.append({"date_utc": date.date().isoformat(), "region": region, "region_id": region_id,
                               "era5_4_sample_mean_percent": era_day[d], "ceres_24_slot_mean_percent": ceres_day24[d],
                               "ceres_4_halfhour_slot_mean_percent": ceres_day4[d],
                               "difference_24_vs_4_pp": ceres_day24[d] - era_day[d],
                               "difference_4_vs_4_pp": ceres_day4[d] - era_day[d],
                               "era5_valid_samples_of_4": int(np.isfinite(era_mean.reshape(31,4)[d]).sum()),
                               "ceres_valid_samples_of_24": int(np.isfinite(ceres_mean.reshape(31,24)[d]).sum()),
                               "ceres_valid_samples_of_4": int(np.isfinite(ceres_mean.reshape(31,24)[d,[0,6,12,18]]).sum()),
                               "fully_covered_target_cells": len(kept)})
        for i, timestamp in enumerate(nominal):
            time_rows.append({"region": region, "source_index": i, "raw_time_days_since_20000301": float(raw_time[i]),
                              "decoded_time_utc": decoded[i].isoformat(), "nominal_midpoint_utc": timestamp.isoformat(),
                              "nominal_hour_slot_utc": timestamp.floor("h").isoformat(), "rounding_offset_seconds": error[i],
                              "used_in_4_slot_scheme": timestamp.hour in [0,6,12,18]})
            sample_rows.append({"region": region, "product": "CERES", "sample_index": i,
                                "time_utc": decoded[i].isoformat(), "nominal_slot_utc": timestamp.isoformat(),
                                "regional_cloud_percent": ceres_mean[i]})
        for i, timestamp in enumerate(expected_era):
            sample_rows.append({"region": region, "product": "ERA5", "sample_index": i,
                                "time_utc": timestamp.isoformat(), "nominal_slot_utc": timestamp.isoformat(),
                                "regional_cloud_percent": era_mean[i]})
        # An independent rectangular-footprint integration must equal the two-stage area average.
        included = [row for row in rows if row["included"]]
        south, north = min(x["south"] for x in included), max(x["north"] for x in included)
        west, east = min(x["west"] for x in included), max(x["east"] for x in included)
        if len(kept) != len(np.unique([x["latitude"] for x in included])) * len(np.unique([x["longitude"] for x in included])):
            raise ValueError("Direct rectangle check requires rectangular selected footprint")
        sy = np.maximum(0, np.sin(np.deg2rad(np.minimum(source_lat.astype(float)+.5,north))) -
                        np.sin(np.deg2rad(np.maximum(source_lat.astype(float)-.5,south))))
        sx = np.maximum(0,np.minimum(source_lon.astype(float)+.5,east)-np.maximum(source_lon.astype(float)-.5,west))
        direct_weight = (sy[:,None] * np.deg2rad(sx[None,:])).ravel()
        direct_weight /= direct_weight.sum()
        direct_mean = np.where(flat_missing,0,flat) @ direct_weight
        direct_mean[(flat_missing & (direct_weight[None,:] > 0)).any(axis=1)] = np.nan
        finite = np.isfinite(direct_mean) & np.isfinite(ceres_mean)
        mass_diff = float(np.max(np.abs(direct_mean[finite] - ceres_mean[finite]))) if finite.any() else 0.0
        if mass_diff > 1e-10 or not np.allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError("Conservative area/constant-field verification failed")
        np.savez_compressed(output / f"{region}_processed.npz", source_missing_mask=missing,
                            source_raw_time=raw_time, source_latitude=source_lat, source_longitude=source_lon,
                            ceres_target_cloud_percent=mapped, ceres_target_missing_mask=mapped_missing,
                            era5_target_cloud_percent=era_cells, era5_target_missing_mask=era_missing,
                            target_flat_indices=kept, conservative_weights=weights, target_area_weights=area_weights,
                            target_area_m2=area*EARTH_RADIUS_M**2)
        quality[region] = {"original_target_cells": len(rows), "included_target_cells": len(kept),
                           "excluded_target_indices": [x["flat_grid_index"] for x in rows if not x["included"]],
                           "source_ceres_missing_values": int(missing.sum()), "era5_target_missing_values": int(era_missing.sum()),
                           "ceres_remapped_missing_values": int(mapped_missing.sum()),
                           "ceres_source_value_range_percent": [float(valid.min()), float(valid.max())],
                           "era5_target_value_range_percent": [float(np.nanmin(era_cells)),float(np.nanmax(era_cells))],
                           "nominal_time_error_seconds_max_abs": float(np.max(np.abs(error))),
                           "conservative_weight_row_sum_max_abs_error": float(np.max(np.abs(weights.sum(axis=1)-1))),
                           "two_stage_vs_direct_footprint_max_difference_pp": mass_diff,
                           "footprint_area_m2": float(area.sum()*EARTH_RADIUS_M**2)}
    frames = {"paired_daily.csv": daily_rows, "descriptive_metrics.csv": metric_rows, "target_grid_support.csv": grid_rows,
              "regional_samples.csv": sample_rows, "ceres_time_mapping.csv": time_rows}
    for name, records in frames.items():
        pd.DataFrame(records).to_csv(output/name,index=False,float_format="%.12g")
    daily = pd.DataFrame(daily_rows)
    if len(daily) != 62 or daily.duplicated(["date_utc","region"]).any():
        raise ValueError("Daily output grain is not exactly 31 days by two regions")
    write_json(output/"quality_checks.json",quality)
    evidence_dir = output / "source_evidence"
    evidence_dir.mkdir(exist_ok=True)
    for item in plan["evidence"]:
        source_path = Path(item["path"])
        shutil.copy2(source_path, evidence_dir/source_path.name)
    shutil.copy2(Path(__file__), output/Path(__file__).name)
    make_plot(daily, output)
    manifest = {"created_utc": utc_now(), "comparison_plan_sha256": sha256(plan_path),
                "script_sha256": sha256(__file__), "inputs_hashes_verified": True,
                "python_libraries": {"numpy":np.__version__,"pandas":pd.__version__,"xarray":xr.__version__,"netCDF4":netCDF4.__version__},
                "daily_rows":len(daily_rows), "metric_rows":len(metric_rows), "metrics":metric_rows,
                "quality_checks":quality, "limitations":LIMITATIONS,
                "source_time_semantics_evidence":"source_evidence/time_semantics_evidence.json",
                "elapsed_seconds":time.perf_counter()-start,
                "result_status":"DESCRIPTIVE_PREPROCESSING_FEASIBILITY_COMPLETE_NOT_EXTERNAL_SCIENTIFIC_VALIDATION"}
    write_readme(output, manifest)
    manifest["outputs_sha256"] = {str(p.relative_to(output)):sha256(p) for p in output.rglob("*") if p.is_file() and p.name != "run_manifest.json"}
    write_json(output/"run_manifest.json", manifest)
    print(json.dumps({"output":str(output),"metrics":metric_rows,"elapsed_seconds":manifest["elapsed_seconds"]},indent=2),flush=True)


def make_plot(daily, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(8,5.8),sharex=True,sharey=True,layout="constrained")
    for ax, (region, region_id) in zip(axes,REGIONS):
        d = daily[daily.region == region]
        days = pd.to_datetime(d.date_utc)
        ax.plot(days,d.era5_4_sample_mean_percent,"o-",color="#222222",ms=3,lw=1,label="ERA5: 4 samples/day")
        ax.plot(days,d.ceres_24_slot_mean_percent,"-",color="#0072B2",lw=1.5,label="CERES: 24 slots/day")
        ax.plot(days,d.ceres_4_halfhour_slot_mean_percent,"--",color="#D55E00",lw=1.3,label="CERES: 4 half-hour slots/day")
        ax.set_title(f"{region.replace('_',' ').title()} | fixed region {region_id}",loc="left",fontsize=10)
        ax.set_ylabel("Cloud fraction (%)")
        ax.set_ylim(0,100)
        ax.grid(alpha=.22)
    axes[0].legend(ncol=1,fontsize=8,loc="lower left")
    axes[1].set_xlabel("January 2019 (UTC) | different temporal sampling operators")
    fig.suptitle("CERES / ERA5 processing feasibility pilot",fontsize=12)
    fig.savefig(output/"daily_comparison.png",dpi=200)
    fig.savefig(output/"daily_comparison.pdf")
    plt.close(fig)


def write_readme(output, manifest):
    lines = ["# CERES / ERA5 January 2019 preprocessing feasibility", "",
             "This completed pilot compares raw physical cloud fractions on identical spatial support. It does not establish long-term external validation or validate causal graph edges.", "",
             "The acquisition pair was fixed before submission; comparison_plan.json freezes all pairing and aggregation rules before reading the matched ERA5 cloud values. Both original regions are retained, with all 36 cells covered in each region.", "",
             "CERES one-degree cells are conservatively remapped using spherical overlap areas. The same 36 ERA5 cells then receive identical area weights in both products. The footprints include all surface types and are not claimed to be ocean-only. Missing masks are preserved with strict completeness; no values are clipped or imputed.", "",
             "Two predeclared sample-mean comparisons are retained: (A) 24 nominal CERES hourly slots versus ERA5 00/06/12/18 UTC samples each day; (B) CERES 00:30/06:30/12:30/18:30 nominal slots versus those same four ERA5 samples. These are different sampling operators, not a proof of equivalent averaging windows. Raw CERES time values are preserved in the processed NPZ and ceres_time_mapping.csv. The Ed4A hourbox convention is an explicitly qualified assumption when applied to Ed4.2.", "",
             "| Region | Scheme | Days | CERES mean (%) | ERA5 mean (%) | Bias (pp) | MAE (pp) | RMSE (pp) | Pearson r |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in manifest["metrics"]:
        scheme = "A: 24 vs 4" if row["scheme"] == SCHEMES[0] else "B: 4 vs 4"
        vals = [row[k] for k in ["ceres_mean_percent","era5_mean_percent","bias_pp_ceres_minus_era5","mae_pp","rmse_pp","pearson_r"]]
        lines.append(f"| {row['region']} | {scheme} | {row['n_paired_days']} | " + " | ".join("NA" if v is None else f"{v:.4f}" for v in vals) + " |")
    lines += ["", "Bias is CERES minus ERA5. The discrepancy metrics do not designate either product as ground truth. Correlations are descriptive across 31 temporally dependent days; no p-values or independent-sample confidence intervals are reported.", "",
              "## Files", "", "- paired_daily.csv: exactly 62 rows, one date × region, containing both schemes and completeness counts.",
              "- descriptive_metrics.csv: all four region × scheme results.",
              "- target_grid_support.csv: original cell identities, inferred bounds, covered areas and inclusion decisions.",
              "- *_processed.npz: remapped cloud values, original missing masks, ERA5 selected values, original CERES numeric times, and all spatial weights.",
              "- ceres_time_mapping.csv / regional_samples.csv: inspectable time assignment and physical regional samples.",
              "- source_evidence/: frozen acquisition/validation/time-semantics records and original mapping code.",
              "- run_manifest.json: input/code/output hashes, software versions, metrics and numerical checks.",
              "- daily_comparison.png / .pdf: daily product-sample means with fixed 0–100% axes.", "", "## Scope limits", ""]
    lines += [f"- {s}" for s in LIMITATIONS]
    lines += ["", "## Reproduction", "", "Run the saved Python script with --freeze into a new output directory, then --run using that directory. The existing frozen plan is never overwritten. No network access is used. Input hashes must match the acquisition audit and prior ERA5 manifest before results are calculated.", "",
              "```powershell", "& '..\\.venv\\Scripts\\python.exe' 'revision\\run_ceres_pilot_comparison.py' --freeze --output '<new-output-directory>'", "& '..\\.venv\\Scripts\\python.exe' 'revision\\run_ceres_pilot_comparison.py' --run --output '<new-output-directory>'", "```", ""]
    (output/"README.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--run",action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.output)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
