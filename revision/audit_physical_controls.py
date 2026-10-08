"""Independent saved-output audit against sampled real NetCDF source records.

Uses netCDF4 directly and per-region reductions, not the preprocessing reader or
its in-memory raw cache. This supplements the full per-grid training-isolation
and direct-fit checks in prepare_physical_controls.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "revision_outputs/physical_controls_inputs"
    manifest = json.loads((output / "data_manifest.json").read_text(encoding="utf8"))
    assert manifest["status"] == "completed"
    for name, receipt in manifest["outputs"].items():
        assert sha(output / name) == receipt["sha256"]
        assert (output / name).stat().st_size == receipt["bytes"]
    assert sha(output / "source_manifest.json") == manifest["source_manifest_sha256"]
    assert sha(output / "validation.json") == manifest["validation_sha256"]
    inventory = json.loads((output / "source_manifest.json").read_text(encoding="utf8"))
    with np.load(output / "region_controls_trainfit.npz", allow_pickle=False) as saved:
        data, physical = saved["data"], saved["physical"]
        timestamps, names = saved["timestamps"], saved["variable_names"].tolist()
        segments, training = saved["source_segment"], saved["training_mask"]
        out_lat, out_lon, out_nodes = saved["lat"], saved["lon"], saved["node_ids"]
    with np.load(root / "revision_outputs/inputs/region_trainfit.npz", allow_pickle=False) as reference:
        assert np.array_equal(timestamps, reference["timestamps"])
        assert np.array_equal(out_lat, reference["lat"])
        assert np.array_equal(out_lon, reference["lon"])
        assert np.array_equal(out_nodes, reference["node_ids"])
    assert data.shape == physical.shape == (68668, 66, 6)
    assert np.isfinite(data).all() and np.isfinite(physical).all()
    assert np.array_equal(training, timestamps < np.datetime64("2019-01-01"))
    assert np.array_equal(segments, np.where(training, 0, np.where(timestamps < np.datetime64("2023-01-11"), 1, 2)))
    with np.load(output / "trainfit_parameters.npz", allow_pickle=False) as parameters:
        mapping = parameters["node_to_region"]
        with np.load(root / "revision_outputs/inputs/trainfit_parameters.npz", allow_pickle=False) as reference:
            assert np.array_equal(mapping, reference["node_to_region"])
            assert np.array_equal(parameters["grid_lat"], reference["grid_lat"])
            assert np.array_equal(parameters["grid_lon"], reference["grid_lon"])
        rng = np.random.default_rng(20261001)
        sample_ids = np.unique(np.concatenate(([0, 58439, 58440, 64323, 64324, 68667],
                                               rng.choice(len(timestamps), 18, replace=False))))
        months = pd.DatetimeIndex(timestamps).month.to_numpy()
        max_standardized_error = max_physical_error = 0.
        rows = []
        for j, field in enumerate(names):
            climatology = parameters[field + "__climatology"]
            mean, std = parameters[field + "__mean"], parameters[field + "__std"]
            assert climatology.shape == (12, 2048) and mean.shape == std.shape == (2048,)
            assert np.isfinite(climatology).all() and np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
            for i in sample_ids:
                source = [r for r in inventory["sources"] if field in r["fields"]
                          and r["start_index"] <= i < r["stop_index_exclusive"]]
                assert len(source) == 1
                source = source[0]
                offset = int(i - source["start_index"])
                with netCDF4.Dataset(source["path"]) as ds:
                    time_var = ds.variables["time"]
                    decoded = netCDF4.num2date(time_var[offset], time_var.units,
                                              calendar=getattr(time_var, "calendar", "standard"),
                                              only_use_cftime_datetimes=False)
                    assert np.datetime64(str(decoded)) == timestamps[i]
                    field_data = ds.variables[field][offset]
                    assert not np.ma.getmaskarray(field_data).any()
                    raw = np.asarray(field_data, dtype=np.float64).reshape(-1)
                z = (raw - climatology[months[i] - 1] - mean) / std
                # Region loop here is an independent per-row reduction, while
                # the producer reduces vectors over time chunks.
                raw_regions = np.array([raw[mapping == r].mean() for r in range(66)])
                z_regions = np.array([z[mapping == r].mean() for r in range(66)])
                pe = float(np.max(np.abs(raw_regions - physical[i, :, j])))
                ze = float(np.max(np.abs(z_regions - data[i, :, j])))
                assert pe < 1e-8 and ze < 1e-10, (field, int(i), pe, ze)
                max_physical_error, max_standardized_error = max(max_physical_error, pe), max(max_standardized_error, ze)
                rows.append({"field": field, "time_index": int(i), "time": str(timestamps[i]),
                             "source": source["path"], "source_record": offset,
                             "physical_max_abs_error": pe, "standardized_max_abs_error": ze})
    terrain = pd.read_csv(output / "coarse_surface_pressure_terrain_diagnostic.csv")
    assert len(terrain) == 594 and not terrain.duplicated(["source_segment", "region", "pressure_level_hpa"]).any()
    for column in ["mean_coarse_grid_cell_fraction_sp_below_level", "max_coarse_grid_cell_fraction_sp_below_level",
                   "fraction_times_with_any_coarse_grid_sp_below_level"]:
        assert terrain[column].between(0, 1).all()
    assert (terrain["mean_coarse_grid_cell_fraction_sp_below_level"] <= terrain["max_coarse_grid_cell_fraction_sp_below_level"] + 1e-12).all()
    report = {"status": "passed", "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
              "audit_script_sha256": sha(__file__), "manifest_sha256": sha(output / "data_manifest.json"),
              "all_output_hashes_recomputed": True, "all_saved_values_finite": True,
              "reference_time_regions_exactly_equal": True, "source_labels_and_training_mask_exact": True,
              "terrain_diagnostic_table_checks": "passed; diagnostic is coarse and does not mask pressure fields",
              "sampled_real_netcdf_time_field_records": len(rows), "regions_checked_each": 66,
              "sample_seed": 20261001, "standardized_max_abs_error": max_standardized_error,
              "physical_max_abs_error": max_physical_error, "samples": rows,
              "limits": "Source record audit is sampled; all-file hashes/metadata and all-grid finite checks plus independent training-fit tests are in producer provenance. No cross-source homogeneity or causal identification asserted."}
    (output / "validation_saved_sources_independent.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: report[k] for k in ["status", "sampled_real_netcdf_time_field_records", "standardized_max_abs_error", "physical_max_abs_error"]}))


if __name__ == "__main__":
    main()
