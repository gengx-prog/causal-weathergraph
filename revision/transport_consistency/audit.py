"""Independent read-only numerical audit of transport experiment artifacts.

This does not import the production feature or experiment helpers.  It writes
only the explicitly requested audit receipt.  Scientific causal identification
and native-resolution terrain validity are outside this numerical audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import traceback

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.ensemble import HistGradientBoostingRegressor
import xarray as xr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT.parent / "revision_outputs/transport_consistency_v1"
R = 6371000.0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def close(a, b, *, atol=1e-12, rtol=1e-11, label="values"):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}")
    difference = float(np.max(np.abs(a - b))) if a.size else 0.0
    require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"{label}: nonfinite")
    require(np.allclose(a, b, atol=atol, rtol=rtol), f"{label}: max difference {difference}")
    return difference


def independent_geometry(lat, lon):
    phi = np.deg2rad(lat)
    edges = np.empty(len(phi) + 1)
    edges[0], edges[-1] = -np.pi / 2, np.pi / 2
    edges[1:-1] = (phi[:-1] + phi[1:]) / 2
    dlambda = 2 * np.pi / len(lon)
    area_row = R * R * dlambda * (np.sin(edges[1:]) - np.sin(edges[:-1]))
    return edges, dlambda, np.repeat(area_row[:, None], len(lon), axis=1)


def boundary_mfc(halo, ids, core, edges, dlambda, area):
    """Sum only four external sides; never call a production divergence helper."""
    nx = area.shape[1]
    at = {int(v): i for i, v in enumerate(ids)}
    rows, cols = np.unique(core // nx), np.unique(core % nx)
    q = halo[..., 2].astype(np.float64)
    qu, qv = q * halo[..., 0], q * halo[..., 1]
    outward = np.zeros(halo.shape[0])
    for y in rows:
        for x, neighbor, sign in [(cols[-1], (cols[-1] + 1) % nx, 1),
                                   (cols[0], (cols[0] - 1) % nx, -1)]:
            flow = (qu[:, at[int(y * nx + x)]] + qu[:, at[int(y * nx + neighbor)]]) / 2
            outward += sign * flow * R * (edges[y + 1] - edges[y])
    for x in cols:
        for y, neighbor, boundary, sign in [(rows[-1], rows[-1] + 1, rows[-1] + 1, 1),
                                           (rows[0], rows[0] - 1, rows[0], -1)]:
            flow = (qv[:, at[int(y * nx + x)]] + qv[:, at[int(neighbor * nx + x)]]) / 2
            outward += sign * flow * R * dlambda * np.cos(edges[boundary])
    return -outward / area.ravel()[core].sum()


def audit_inputs(output, receipt):
    folder = output / "inputs"
    mpath = folder / "input_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf8"))
    source_hashes = []
    for row in manifest["sources"]:
        require(sha(row["path"]) == row["sha256"], f"Source hash changed: {row['path']}")
        source_hashes.append({"path": row["path"], "sha256": row["sha256"]})
    archive = folder / "input_data.npz"
    archive_sha = sha(archive)
    require(archive_sha == manifest["outputs"][archive.name]["sha256"], "Input archive hash")
    require(sha(output / "design.json") == manifest["frozen_plan"]["sha256"], "Frozen design hash")
    require(sha(Path(__file__).with_name("features.py")) == manifest["code_sha256"], "Feature code hash")
    with np.load(archive, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    require(data["grid"].shape == (68668, 2, 60, 4), "Full halo shape")
    require(data["region_ids"].tolist() == [15, 48], "Region order")
    require(data["grid_names"].tolist() == ["u", "v", "q", "temperature"], "Grid field order")
    require(data["local_names"].tolist() == ["u", "v", "q", "temperature", "cloud"], "Local field order")
    require(data["flux_names"].tolist() == ["mean_qu", "mean_qv", "cov_qu", "cov_qv", "mfc"], "Flux order")
    require(np.all(np.diff(data["timestamps"]) == np.timedelta64(6, "h")), "Complete calendar")
    for key in ["local", "grid", "flux", "controls"]:
        require(np.isfinite(data[key]).all(), f"Finite {key}")
    edges, dlambda, area = independent_geometry(data["grid_lat"], data["grid_lon"])
    require(abs(area.sum() / (4 * np.pi * R * R) - 1) < 1e-14, "Global spherical area")
    maxima = {"regional_mean": 0., "flux_covariance": 0., "mfc_boundary_s_inverse": 0.}
    for region in range(2):
        core, halo = data["region_grid_indices"][region], data["grid_indices"][region]
        # Independently reconstruct original 6 x 11 bins; region IDs are sorted.
        la, lo = np.meshgrid(data["grid_lat"], data["grid_lon"], indexing="ij")
        wrapped = (lo + 180) % 360 - 180
        bins = np.clip(np.floor((np.clip(la, -89.999, 89.999) + 90) / 30), 0, 5).astype(int) * 11
        bins += np.minimum(np.floor((wrapped + 180) / (360 / 11)).astype(int), 10)
        expected_core = np.flatnonzero(bins.ravel() == data["region_ids"][region])
        require(np.array_equal(core, expected_core), "Original region membership")
        needed = set(map(int, core))
        for i in core:
            y, x = divmod(int(i), 64)
            needed.update([y * 64 + (x - 1) % 64, y * 64 + (x + 1) % 64,
                           (y - 1) * 64 + x, (y + 1) * 64 + x])
        require(set(map(int, halo)) == needed and len(needed) == 60, "Exact same-information halo")
        weights = area.ravel()[core] / area.ravel()[core].sum()
        close(weights, data["area_weights"][region], atol=1e-15, label="Area weights")
        position = np.array([np.flatnonzero(halo == k)[0] for k in core])
        for begin in range(0, len(data["timestamps"]), 1024):
            end = min(begin + 1024, len(data["timestamps"]))
            selected = data["grid"][begin:end, region, position].astype(np.float64)
            means = np.einsum("tgv,g->tv", selected, weights)
            maxima["regional_mean"] = max(maxima["regional_mean"], close(means, data["local"][begin:end, region, :4], label="Full-series raw means"))
            product = means[:, 2:3] * means[:, :2]
            covariance = np.einsum("tgv,tg,g->tv", selected[:, :, :2] - means[:, None, :2],
                                   selected[:, :, 2] - means[:, None, 2], weights)
            expected_flux = np.column_stack([product, covariance])
            maxima["flux_covariance"] = max(maxima["flux_covariance"], close(expected_flux, data["flux"][begin:end, region, :4], atol=1e-14, label="Full-series covariance"))
            boundary = boundary_mfc(data["grid"][begin:end, region], halo, core, edges, dlambda, area)
            maxima["mfc_boundary_s_inverse"] = max(maxima["mfc_boundary_s_inverse"], close(boundary, data["flux"][begin:end, region, 4], atol=1e-18, rtol=1e-11, label="Full-series external boundary MFC"))
    # Original NetCDF values sampled independently in every source/field.
    rng = np.random.default_rng(846101)
    source_samples, max_cloud = [], 0.
    for part in manifest["source_segments"]:
        si, offset, stop = part["source_segment"], part["start_index"], part["stop_index_exclusive"]
        count = stop - offset
        picks = np.unique(np.r_[0, 1, count // 2, count - 2, count - 1,
                                rng.integers(0, count, size=11)]).astype(int)
        sample = {}
        for row in [v for v in manifest["sources"] if v["segment"] == si]:
            with xr.open_dataset(row["path"]) as ds:
                values = ds[row["variable"]].transpose("time", "latitude", "longitude").isel(time=picks).values
                require(np.array_equal(ds.time.values[picks], data["timestamps"][offset + picks]), "Original time mapping")
                require(ds[row["variable"]].attrs["units"] == row["units"], "Original units")
            sample[row["field"]] = values.astype(np.float64)
            fi = ["u", "v", "humidity", "temperature", "cloud_cover"].index(row["field"])
            for ri in range(2):
                flattened = values.reshape(len(picks), -1)
                if fi < 4:
                    require(np.array_equal(flattened[:, data["grid_indices"][ri]].astype(np.float32), data["grid"][offset + picks, ri, :, fi]), "Original halo values")
                core = data["region_grid_indices"][ri]
                weights = area.ravel()[core] / area.ravel()[core].sum()
                expected = flattened[:, core].astype(np.float64) @ weights
                difference = close(expected, data["local"][offset + picks, ri, fi], atol=1e-12, label="Original local average")
                if fi == 4:
                    max_cloud = max(max_cloud, difference)
        # Alternative full-grid implementation: accumulate every face's equal
        # and opposite contribution to its two neighboring cells.
        qu, qv = sample["humidity"] * sample["u"], sample["humidity"] * sample["v"]
        net = np.zeros_like(qu)
        for y in range(32):
            for x in range(64):
                right = (x + 1) % 64
                crossing = (qu[:, y, x] + qu[:, y, right]) * .5 * R * (edges[y + 1] - edges[y])
                net[:, y, x] += crossing
                net[:, y, right] -= crossing
                if y < 31:
                    crossing = (qv[:, y, x] + qv[:, y + 1, x]) * .5 * R * dlambda * np.cos(edges[y + 1])
                    net[:, y, x] += crossing
                    net[:, y + 1, x] -= crossing
        closure = np.abs(net.sum(axis=(1, 2))) / np.maximum(np.abs(net).sum(axis=(1, 2)), 1e-100)
        require(closure.max() < 1e-12, "Independent full-sphere face cancellation")
        for ri in range(2):
            core = data["region_grid_indices"][ri]
            expected = -net.reshape(len(picks), -1)[:, core].sum(axis=1) / area.ravel()[core].sum()
            close(expected, data["flux"][offset + picks, ri, 4], atol=1e-18, label="Original grid cell-net MFC")
        source_samples.append({"source_segment": int(si), "original_time_indices": picks.tolist(),
                               "global_face_cancellation_relative_max": float(closure.max())})
    controls = manifest["controls"]
    require(sha(controls["path"]) == controls["sha256"], "Physical control source hash")
    with np.load(controls["path"], allow_pickle=False) as z:
        require(np.array_equal(data["controls"], z["physical"][:, [15, 48], :]), "Controls use physical array, not 1979-2018 trainfit data")
        require(np.array_equal(data["source_segment"], z["source_segment"]), "Control source segments")
    receipt["inputs"] = {"passed": True, "input_sha256": archive_sha, "manifest_sha256": sha(mpath),
                         "n_times_full_halo_and_flux_checked": len(data["timestamps"]), "max_abs_differences": maxima,
                         "original_netcdf_subset_checks": source_samples, "cloud_raw_mean_sample_max_difference": max_cloud,
                         "original_source_hashes": source_hashes, "physical_controls_full_equality": True,
                         "limits": "All halo-derived flux/means checked; source values and clouds independently sampled at 16 times per source. Producer checked all global values; this audit does not repeat that full original-grid read or establish native terrain validity."}
    print("Independent input audit passed", flush=True)
    return data


def independent_samples(data):
    times = data["timestamps"]
    group = np.select([times < np.datetime64("2015-01-01"), times < np.datetime64("2019-01-01")],
                      ["train", "validation"], default="test")
    candidates = np.flatnonzero(times.astype("datetime64[h]").astype(np.int64) % 24 == 0)
    retained = []
    for i in candidates:
        if i < 2 or i + 1 >= len(times):
            continue
        addresses = np.array([i - 2, i - 1, i, i + 1])
        if len(set(group[addresses])) == 1 and len(set(data["source_segment"][addresses])) == 1:
            retained.append(i)
    idx = np.array(retained, dtype=int)
    return idx, group[idx]


def columns_from_names(data, idx, region, names, allow_future=False):
    """Resolve stored semantic feature labels without production design helpers."""
    lookups = {k: {str(v): i for i, v in enumerate(data[k + "_names"])}
               for k in ["local", "grid", "flux", "control"]}
    cols = []
    day = pd.DatetimeIndex(data["timestamps"][idx]).dayofyear.to_numpy()
    for label in names:
        name = str(label)
        if name in ["dayofyear_sin", "dayofyear_cos"]:
            cols.append((np.sin if name.endswith("sin") else np.cos)(day * 2 * np.pi / 365.2425))
            continue
        if name.startswith("FUTURE6h:"):
            require(allow_future, "Future information in implementable model")
            cols.append(data["flux"][idx + 1, region, lookups["flux"][name.split(":")[1]]])
            continue
        ri = region
        if name.startswith("other_region:"):
            ri, name = 1 - region, name[len("other_region:"):]
        parts = name.split(":")
        require(parts[0] in ["lag0", "lag1", "lag2"], "Unknown lag")
        addresses = idx - int(parts[0][3:])
        if len(parts) == 3:
            require(parts[1].startswith("cell"), "Unknown grid label")
            cols.append(data["grid"][addresses, ri, int(parts[1][4:]), lookups["grid"][parts[2]]])
        elif parts[1] in lookups["local"]:
            cols.append(data["local"][addresses, ri, lookups["local"][parts[1]]])
        elif parts[1] in lookups["control"]:
            cols.append(data["controls"][addresses, ri, lookups["control"][parts[1]]])
        else:
            cols.append(data["flux"][addresses, ri, lookups["flux"][parts[1]]])
    return np.column_stack(cols).astype(np.float64)


def mask_for(frame, label, region):
    if label == "all":
        return np.ones(len(frame), dtype=bool)
    if label.startswith("source:"):
        return frame.source_segment.to_numpy().astype(str) == label.split(":")[1]
    month = pd.DatetimeIndex(frame.origin_time).month.to_numpy()
    nh = np.array(["winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer", "autumn", "autumn", "autumn", "winter"])
    sh = np.array(["summer", "summer", "autumn", "autumn", "autumn", "winter", "winter", "winter", "spring", "spring", "spring", "summer"])
    return (nh if int(region) == 48 else sh)[month - 1] == label[len("local_"):]


def audit_models(output, data, receipt):
    folder = output / "model"
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf8"))
    require(manifest["status"] == "completed", "Models not complete")
    require(manifest["input_sha256"] == receipt["inputs"]["input_sha256"], "Model input binding")
    require(manifest["code_sha256"] == sha(Path(__file__).with_name("experiment.py")), "Model code binding")
    for file, digest in manifest["output_sha256"].items():
        require(sha(folder / file) == digest, f"Model output binding: {file}")
    formal = pd.read_csv(folder / "predictions.csv.gz")
    diagnostic = pd.read_csv(folder / "diagnostic_predictions.csv.gz")
    require(~formal.model.eq("ridge_futureflux").any(), "Future model in formal predictions")
    require(diagnostic.model.eq("ridge_futureflux").all(), "Mixed diagnostic predictions")
    frame = pd.concat([formal, diagnostic], ignore_index=True)
    idx, splits = independent_samples(data)
    ledger = pd.read_csv(folder / "sample_ledger.csv")
    require(np.array_equal(ledger.loc[ledger.retained, "origin_index"], idx), "Independent sample eligibility")
    require(pd.Series(splits).value_counts().to_dict() == {"train": 13148, "test": 2555, "validation": 1460}, "Expected split sizes")
    tr, va = splits == "train", splits == "validation"
    month = pd.DatetimeIndex(data["timestamps"][idx]).month.to_numpy()
    grouped = {}
    for (rid, task, model), f in frame.groupby(["region_id", "task", "model"]):
        f = f.sort_values("origin_index").reset_index(drop=True)
        grouped[(int(rid), task, model)] = f
        require(np.array_equal(f.origin_index, idx) and np.array_equal(f.split, splits), "Same population across all models")
        ri = list(data["region_ids"]).index(int(rid))
        actual = (data["local"][idx + 1, ri, 2] - data["local"][idx, ri, 2]) * 1000 if task == "humidity_change" else data["local"][idx + 1, ri, 4] * 100
        close(f.actual, actual, atol=1e-12, label="All original targets")
        require(np.array_equal(pd.to_datetime(f.origin_time).to_numpy(), data["timestamps"][idx]), "Origin timestamps")
        require(np.array_equal(pd.to_datetime(f.target_time).to_numpy(), data["timestamps"][idx + 1]), "Six-hour target timestamps")
        require(np.array_equal(f.source_segment, data["source_segment"][idx]), "Prediction source labels")
        clipped = np.clip(f.raw_prediction, 0, 100) if task == "cloud_level" else f.raw_prediction
        close(f.prediction, clipped, atol=1e-12, label="Uniform clipping")
        if model in ["persistence", "monthly_climatology", "horizontal_only_budget"]:
            if model == "persistence":
                expected = np.zeros(len(idx)) if task == "humidity_change" else data["local"][idx, ri, 4] * 100
            elif model == "horizontal_only_budget":
                expected = data["flux"][idx, ri, 4] * 21600000
            else:
                climate = np.array([actual[tr & (month == m)].mean() for m in range(1, 13)])
                expected = climate[month - 1]
            close(f.raw_prediction, expected, atol=1e-11, label="Independent baseline")
    model_checks = []
    for details in manifest["models"]:
        rid, task, model = details["region_id"], details["task"], details["model"]
        ri = list(data["region_ids"]).index(rid)
        saved = grouped[(rid, task, model)]
        y = saved.actual.to_numpy()
        stem = f"region{rid}_{task}_{model}"
        if model.startswith("ridge"):
            with np.load(folder / (stem + ".npz"), allow_pickle=False) as z:
                params = {key: z[key] for key in z.files}
            names = params["feature_names"]
            x = columns_from_names(data, idx, ri, names, model == "ridge_futureflux")
            mean, scale = x[tr].mean(axis=0), x[tr].std(axis=0)
            scale[scale < 1e-12] = 1
            close(params["feature_mean"], mean, atol=1e-10, label="Train-only means")
            close(params["feature_scale"], scale, atol=1e-10, label="Train-only scales")
            z = (x - mean) / scale
            intercept = y[tr].mean()
            close(np.asarray(params["intercept"]), np.asarray(intercept), label="Train-only intercept")
            coefficient = params["coefficient"]
            prediction = z @ coefficient + intercept
            discrepancy = close(saved.raw_prediction, prediction, atol=5e-9, label="All ridge predictions")
            rhs = z[tr].T @ (y[tr] - intercept)
            residue = z[tr].T @ (z[tr] @ coefficient - y[tr] + intercept) + details["alpha"] * coefficient
            normal = float(np.linalg.norm(residue) / max(np.linalg.norm(rhs), 1))
            require(normal < 1e-7, "Independent ridge first-order optimum")
            # Direct SPD solves independently reproduce every validation penalty,
            # including the primary high-dimensional same-information models.
            gram = z[tr].T @ z[tr]
            losses = []
            for alpha in [1., 10., 100., 1000.]:
                co = solve(gram + alpha * np.eye(len(mean)), rhs, assume_a="pos")
                pv = z[va] @ co + intercept
                if task == "cloud_level":
                    pv = np.clip(pv, 0, 100)
                loss = float(np.mean((pv - y[va]) ** 2))
                close(np.asarray(loss), np.asarray(details["validation_mse_by_alpha"][str(alpha)]), atol=1e-8, label="Independent alpha validation")
                losses.append(loss)
            require(details["alpha"] == [1., 10., 100., 1000.][int(np.argmin(losses))], "Validation-chosen alpha")
            model_checks.append({"region": rid, "task": task, "model": model, "max_prediction_difference": discrepancy,
                                 "normal_equation_relative_residual": normal, "all_four_alpha_losses_recomputed_by_direct_solve": True})
        else:
            x = columns_from_names(data, idx, ri, details["feature_names"])
            stored = joblib.load(folder / (stem + ".joblib"))
            close(saved.raw_prediction, stored.predict(x), atol=1e-11, label="Serialized HGB predictions")
            # Independently refit on the audited training dates; evaluate all rows.
            frozen = json.loads((output / "design.json").read_text())["models"]["hgb"]
            hyper = {k: v for k, v in frozen.items() if k != "variants"}
            estimator = HistGradientBoostingRegressor(**hyper).fit(x[tr], y[tr])
            discrepancy = close(saved.raw_prediction, estimator.predict(x), atol=1e-8, label="Independent HGB training reproduction")
            model_checks.append({"region": rid, "task": task, "model": model, "full_training_refit": True,
                                 "max_prediction_difference": discrepancy})
    metric_tables = pd.concat([pd.read_csv(folder / "metrics.csv"), pd.read_csv(folder / "diagnostic_metrics.csv")], ignore_index=True)
    cache = {}
    for row in metric_tables.to_dict("records"):
        regions = [15, 48] if str(row["region_id"]) == "pooled_equal_regions" else [int(row["region_id"])]
        parts = []
        for rid in regions:
            key = (rid, row["task"], row["model"], row["split"], row["slice"])
            if key not in cache:
                f = grouped[key[:3]]
                f = f[f.split == row["split"]]
                f = f[mask_for(f, row["slice"], rid)]
                err = (f.prediction - f.actual).to_numpy()
                raw = (f.raw_prediction - f.actual).to_numpy()
                cache[key] = dict(n=len(f), mse=np.mean(err ** 2), mae=np.mean(abs(err)), bias=np.mean(err),
                                  raw_mse=np.mean(raw ** 2), raw_mae=np.mean(abs(raw)),
                                  clipped_count=int((f.prediction != f.raw_prediction).sum()),
                                  clipped_fraction=float((f.prediction != f.raw_prediction).mean()))
            parts.append(cache[key])
        expected = {name: sum(v[name] for v in parts) if name in ["n", "clipped_count"] else np.mean([v[name] for v in parts]) for name in parts[0]}
        expected["rmse"] = np.sqrt(expected["mse"])
        for key, value in expected.items():
            close(np.asarray(row[key]), np.asarray(value), atol=2e-10, label=f"Metric {key}")
    receipt["models"] = {"passed": True, "manifest_sha256": sha(folder / "run_manifest.json"),
                         "n_prediction_rows": len(frame), "all_prediction_targets_calendars_baselines_checked": True,
                         "fit_checks": model_checks, "all_metrics_rows_recomputed": len(metric_tables),
                         "future_columns_isolated_to_diagnostic": True,
                         "date_counts": {s: int((splits == s).sum()) for s in ["train", "validation", "test"]}}
    print("Independent model and metric audit passed", flush=True)
    return frame, grouped


def audit_bootstrap(output, data, grouped, receipt):
    folder = output / "model"
    with np.load(folder / "bootstrap_indices.npz", allow_pickle=False) as z:
        indices, origins, rows = z["indices"], z["origins"], z["origin_index"]
    require(indices.shape == (500, 2555) and indices.min() >= 0 and indices.max() < len(rows), "Bootstrap dimensions")
    source = data["source_segment"][rows]
    days = origins.astype("datetime64[D]")
    cursor = 0
    for value in np.unique(source):
        count = int((source == value).sum())
        chunk = indices[:, cursor:cursor + count]
        require(np.all(source[chunk] == value), "Bootstrap source-preserving draws")
        for start in range(0, count, 30):
            block = days[chunk[:, start:min(start + 30, count)]]
            require(np.all(np.diff(block, axis=1) == np.timedelta64(1, "D")), "Consecutive 30-calendar-day blocks")
        cursor += count
    table = pd.concat([pd.read_csv(folder / "paired_comparisons.csv"), pd.read_csv(folder / "diagnostic_comparisons.csv")], ignore_index=True)
    cache = {}
    for row in table.to_dict("records"):
        regions = [15, 48] if str(row["region_id"]) == "pooled_equal_regions" else [int(row["region_id"])]
        values = []
        for rid in regions:
            key = (rid, row["task"], row["model"], row["baseline"], row["slice"])
            if key not in cache:
                f = grouped[key[:3]]
                b = grouped[(rid, row["task"], row["baseline"])]
                f, b = f[f.split == "test"], b[b.split == "test"]
                require(np.array_equal(f.origin_index, rows) and np.array_equal(b.origin_index, rows), "Same bootstrap calendar")
                mask = mask_for(f, row["slice"], rid)
                losses = np.column_stack([np.square(b.prediction.to_numpy() - b.actual.to_numpy()) - np.square(f.prediction.to_numpy() - f.actual.to_numpy()),
                                          np.abs(b.prediction.to_numpy() - b.actual.to_numpy()) - np.abs(f.prediction.to_numpy() - f.actual.to_numpy())])
                replicates = np.empty((500, 2))
                for j, sample in enumerate(indices):
                    selected = sample[mask[sample]]
                    replicates[j] = losses[selected].mean(axis=0)
                cache[key] = (int(mask.sum()), losses[mask].mean(axis=0), replicates)
            values.append(cache[key])
        estimates = np.mean([v[1] for v in values], axis=0)
        quantiles = np.percentile(np.mean([v[2] for v in values], axis=0), [2.5, 97.5], axis=0)
        require(row["n"] == sum(v[0] for v in values), "Paired comparison population")
        close(np.array([row["delta_mse_baseline_minus_model"], row["delta_mae_baseline_minus_model"]]), estimates, atol=2e-10, label="Paired point estimates")
        expected = np.array([[row["exploratory_delta_mse_p025"], row["exploratory_delta_mae_p025"]],
                             [row["exploratory_delta_mse_p975"], row["exploratory_delta_mae_p975"]]])
        close(expected, quantiles, atol=2e-10, label="All paired percentile intervals")
    receipt["paired_inference"] = {"passed": True, "all_comparison_rows_checked": len(table),
                                   "shared_draws_checked": 500, "source_preserving_consecutive_calendar_blocks": True,
                                   "limits": "Conditional on fitted models; 500 draws, two selected regions, no simultaneous/FDR coverage or causal identification."}
    print("Independent paired-block interval audit passed", flush=True)


def audit_external(output, data, predictions, receipt):
    folder = output / "external"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
    require(manifest["code_sha256"] == sha(Path(__file__).with_name("external_check.py")), "External code binding")
    require(manifest["design_sha256"] == receipt["design_sha256"], "External design binding")
    for path, digest in manifest["source_hashes"].items():
        require(sha(path) == digest, f"External source hash: {path}")
    for path, digest in manifest["outputs_sha256"].items():
        require(sha(folder / path) == digest, f"External output hash: {path}")
    previous = output.parent / "ceres_annual_comparison_2019"
    annual = json.loads((previous / "run_manifest.json").read_text(encoding="utf8"))
    require(sha(previous / "input_manifest.json") == annual["input_manifest_sha256"], "Annual raw input manifest")
    files = json.loads((previous / "input_manifest.json").read_text(encoding="utf8"))["files"]
    pairs = pd.read_csv(folder / "paired_predictions.csv")
    metrics = pd.read_csv(folder / "metrics.csv")
    require(len(pairs) == 8008 and len(metrics) == 44, "Expected external deliverables")
    require(not pairs.duplicated(["region_id", "model", "target_time"]).any(), "External pairing duplicates")
    year = pd.to_datetime(predictions.target_time).dt.year
    selection = predictions[(predictions.task == "cloud_level") & (predictions.split == "test") & (year == 2019)
                            & (predictions.model != "ridge_futureflux")].copy()
    columns = ["region_id", "model", "origin_index"]
    expected_rows = selection.sort_values(columns).reset_index(drop=True)
    actual_rows = pairs.sort_values(columns).reset_index(drop=True)
    require(np.array_equal(actual_rows[columns], expected_rows[columns]), "No external selection or lost forecast rows")
    for col in ["actual", "prediction", "raw_prediction"]:
        close(actual_rows[col], expected_rows[col], atol=1e-8, label=f"External copied {col}")
    lookup = {}
    direct_details = []
    edges, dlambda, target_area = independent_geometry(data["grid_lat"], data["grid_lon"])
    for rid, label in [(15, "south_atlantic"), (48, "north_atlantic")]:
        ri = list(data["region_ids"]).index(rid)
        core = data["region_grid_indices"][ri]
        ys, xs = core // 64, core % 64
        south, north = np.rad2deg(edges[ys.min()]), np.rad2deg(edges[ys.max() + 1])
        west = data["grid_lon"][xs.min()] - np.rad2deg(dlambda) / 2
        east = data["grid_lon"][xs.max()] + np.rad2deg(dlambda) / 2
        with np.load(previous / f"{label}_processed.npz", allow_pickle=False) as z:
            require(np.array_equal(z["target_flat_indices"], core), "CERES/ERA5 identical cell footprint")
            weights = target_area.ravel()[core] / target_area.ravel()[core].sum()
            close(z["target_area_weights"], weights, atol=1e-14, label="CERES original target area weights")
            processed = z["ceres_target_cloud_percent"] @ weights
            rawtime_saved = [z["source_raw_time_file_000"], z["source_raw_time_file_001"]]
        offset = 0
        errors = []
        count = 0
        for number, row in enumerate([f for f in files if f["region"] == label]):
            require(sha(row["path"]) == row["sha256"], "Original CERES source bytes")
            with xr.open_dataset(row["path"], decode_times=False, mask_and_scale=False) as ds:
                raw_time = ds.time.values
                require(np.array_equal(raw_time, rawtime_saved[number]), "Raw CF coordinates preserved")
                require(ds.time.attrs["units"] == "days since 2000-03-01 00:00:00", "CERES epoch")
                decoded = pd.Timestamp("2000-03-01") + pd.to_timedelta(raw_time.astype(np.float64), unit="D")
                nominal = decoded.floor("h") + pd.Timedelta(minutes=30)
                keep = np.flatnonzero(nominal.hour == 6)
                latitude = ds.lat.values.astype(float)
                longitude = np.mod(ds.lon.values.astype(float), 360)
                low, high = np.maximum(latitude - .5, south), np.minimum(latitude + .5, north)
                lat_overlap = np.where(high > low, np.sin(np.deg2rad(high)) - np.sin(np.deg2rad(low)), 0)
                lon_overlap = np.deg2rad(np.maximum(0, np.minimum(longitude + .5, east) - np.maximum(longitude - .5, west)))
                areas = R * R * np.outer(lat_overlap, lon_overlap)
                require(abs(areas.sum() / target_area.ravel()[core].sum() - 1) < 1e-12, "Direct CERES-footprint covered area")
                values = ds[row["variable"]].isel(time=keep).values.astype(float)
                require(np.isfinite(values).all() and values.min() >= 0 and values.max() <= 100, "Original CERES cloud ranges")
                direct = np.einsum("tyx,yx->t", values, areas) / areas.sum()
                errors.append(close(direct, processed[offset + keep], atol=1e-8, label="Direct original CERES area integral"))
                for j, k in enumerate(keep):
                    timestamp = nominal[k] - pd.Timedelta(minutes=30)
                    require((rid, timestamp) not in lookup, "No duplicate original CERES slots")
                    lookup[(rid, timestamp)] = (float(direct[j]), decoded[k], nominal[k])
                count += len(keep)
                offset += len(raw_time)
        require(count == 365, "All 365 original 06-hour CERES slots")
        direct_details.append({"region_id": rid, "source_06_hour_slots": count,
                               "direct_spherical_integral_vs_prior_processed_max_abs_pp": max(errors)})
    for row in pairs.itertuples(index=False):
        target = pd.Timestamp(row.target_time)
        value, decoded, nominal = lookup[(row.region_id, target)]
        require(pd.Timestamp(row.ceres_decoded_time) == decoded, "Original CF time pairing")
        require(pd.Timestamp(row.nominal_slot_utc) == nominal and nominal - target == pd.Timedelta(minutes=30), "Distinct 06:00/06:30 supports")
        close(np.asarray(row.ceres_percent), np.asarray(value), atol=1e-8, label="External original cloud value")
    for row in metrics.itertuples(index=False):
        f = pairs[(pairs.region_id == row.region_id) & (pairs.model == row.model)]
        target = f.ceres_percent.to_numpy() if row.outcome_product == "CERES_hourbox" else f.actual.to_numpy()
        pred = f.prediction.to_numpy()
        error = pred - target
        require(row.n == len(f) == 364, "External evaluation count")
        values = np.array([abs(error).mean(), np.sqrt(np.mean(error ** 2)), error.mean(), np.corrcoef(pred, target)[0, 1]])
        close(np.array([row.MAE_pp, row.RMSE_pp, row.bias_pp, row.correlation]), values, atol=1e-8, label="Independent external metrics")
    receipt["external"] = {"passed": True, "manifest_sha256": sha(folder / "manifest.json"),
                           "all_paired_prediction_rows_checked": len(pairs), "all_metric_rows_recomputed": len(metrics),
                           "original_ceres_cloud_recomputation": direct_details,
                           "same_36_cell_footprint_and_raw_cf_time_verified": True,
                           "limits": "2019 two-region cross-product comparison only; 06:00 instantaneous ERA5 and nominal 06:30 CERES hourbox differ. No cloud ground truth, graph-edge identification, or old cross-region wind-humidity-cloud chain validation."}
    print("Independent raw-CERES external pairing audit passed", flush=True)


def require_full_artifacts(output):
    """Fail before expensive reads if any mandatory full-audit stage is absent."""
    required = ["model/run_manifest.json", "external/manifest.json", "design.json",
                "inputs/input_manifest.json", "inputs/input_data.npz",
                "model/predictions.csv.gz", "model/diagnostic_predictions.csv.gz",
                "model/metrics.csv", "model/diagnostic_metrics.csv",
                "model/paired_comparisons.csv", "model/diagnostic_comparisons.csv",
                "model/bootstrap_indices.npz", "model/sample_ledger.csv",
                "external/paired_predictions.csv", "external/metrics.csv"]
    missing = [name for name in required if not (output / name).is_file()]
    require(not missing, "Full audit requires all stages; missing: " + ", ".join(missing))
    return required


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    receipt = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
               "audit_code_sha256": sha(__file__),
               "independence": "No production features/experiment helpers imported; external-face transport reconstructed independently."}
    exit_code = 0
    try:
        receipt["required_full_artifacts"] = require_full_artifacts(args.output)
        receipt["design_sha256"] = sha(args.output / "design.json")
        with threadpool_limits(limits=2):
            data = audit_inputs(args.output, receipt)
            frame, grouped = audit_models(args.output, data, receipt)
            audit_bootstrap(args.output, data, grouped, receipt)
            audit_external(args.output, data, frame, receipt)
        require(all(receipt.get(stage, {}).get("passed") is True
                    for stage in ["inputs", "models", "paired_inference", "external"]),
                "Full audit cannot pass without all four completed stage receipts")
        receipt["status"] = "PASS_INPUTS_MODELS_EXTERNAL"
    except Exception as exc:
        receipt["status"] = "FAIL_NEEDS_OWNER_REPAIR"
        receipt["error"] = str(exc)
        receipt["traceback"] = traceback.format_exc()
        exit_code = 1
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    destination = args.output / "independent_audit.json"
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"status": receipt["status"], "receipt": str(destination), "error": receipt.get("error")}), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
