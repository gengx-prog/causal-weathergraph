"""Cross-product comparison on the prespecified 2019 CERES footprints.

ERA5 cloud forecasts valid at 06 UTC are paired with CERES's nominal 06:30
hourbox. Neither product is labelled cloud truth. No CERES calibration is fit.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / "revision_outputs"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=BASE / "transport_consistency_v1")
    args = parser.parse_args()
    directory = args.root / "external"
    directory.mkdir(exist_ok=True)
    if (directory / "manifest.json").exists():
        raise FileExistsError("External comparison already complete")
    previous = BASE / "ceres_annual_comparison_2019"
    annual = json.loads((previous / "run_manifest.json").read_text(encoding="utf8"))
    audit = json.loads((previous / "independent_audit.json").read_text(encoding="utf8"))
    assert audit["status"] == "passed"
    assert audit["run_manifest_sha256"] == sha(previous / "run_manifest.json")
    needed = ["regional_samples.csv", "north_atlantic_processed.npz", "south_atlantic_processed.npz"]
    hashes = annual["outputs_sha256"]
    for name in needed:
        assert sha(previous / name) == hashes[name], name
    samples = pd.read_csv(previous / "regional_samples.csv")
    actual = samples[samples["product"] == "CERES"].copy()
    actual["region_id"] = actual["region"].map({"south_atlantic": 15, "north_atlantic": 48})
    actual["nominal_slot_utc"] = pd.to_datetime(actual["nominal_slot_utc"])
    actual["time_utc"] = pd.to_datetime(actual["time_utc"], format="ISO8601")
    actual["target_time"] = actual["nominal_slot_utc"] - pd.Timedelta(minutes=30)
    actual = actual[actual["target_time"].dt.hour == 6].copy()
    assert len(actual) == 730 and not actual.duplicated(["region_id", "target_time"]).any()
    actual = actual.rename(columns={"regional_cloud_percent": "ceres_percent", "time_utc": "ceres_decoded_time"})
    geometry = []
    with np.load(args.root / "inputs/input_data.npz", allow_pickle=False) as inputs:
        regions = inputs["region_ids"]
        for region_id, name in [(15, "south_atlantic"), (48, "north_atlantic")]:
            r = int(np.flatnonzero(regions == region_id)[0])
            with np.load(previous / f"{name}_processed.npz", allow_pickle=False) as z:
                assert np.array_equal(inputs["region_grid_indices"][r], z["target_flat_indices"])
                error = float(np.max(np.abs(inputs["area_weights"][r] - z["target_area_weights"])))
                assert error < 1e-12
                direct = z["ceres_target_cloud_percent"] @ z["target_area_weights"]
                ordered = samples[(samples["product"] == "CERES") & (samples["region"] == name)]
                numeric_error = float(np.max(np.abs(direct - ordered.regional_cloud_percent.to_numpy())))
                assert numeric_error < 1e-8
                geometry.append({"region_id": region_id, "cells": 36, "weight_max_error": error, "annual_sample_max_error_pp": numeric_error})
    prediction_path = args.root / "model/predictions.csv.gz"
    predictions = pd.read_csv(prediction_path)
    required = {"task", "region_id", "origin_time", "target_time", "split", "model", "y_true", "prediction"}
    assert required.issubset(predictions.columns), sorted(predictions.columns)
    pred = predictions[(predictions["task"] == "cloud_level") & (predictions["split"] == "test")].copy()
    pred["origin_time"] = pd.to_datetime(pred["origin_time"])
    pred["target_time"] = pd.to_datetime(pred["target_time"])
    pred = pred[pred.target_time.dt.year == 2019]
    assert not pred.duplicated(["model", "region_id", "target_time"]).any()
    joined = pred.merge(actual[["region_id", "target_time", "nominal_slot_utc", "ceres_decoded_time", "ceres_percent"]],
                        on=["region_id", "target_time"], how="left", validate="many_to_one")
    assert joined.ceres_percent.notna().all() and len(joined) == len(pred)
    assert (joined.target_time - joined.origin_time == pd.Timedelta(hours=6)).all()
    # Recheck new raw-area ERA5 targets against the already audited old series.
    era = samples[samples["product"] == "ERA5"].copy()
    era["region_id"] = era["region"].map({"south_atlantic": 15, "north_atlantic": 48})
    era["target_time"] = pd.to_datetime(era["time_utc"])
    joined = joined.merge(era[["region_id", "target_time", "regional_cloud_percent"]], on=["region_id", "target_time"], validate="many_to_one")
    era_error = float(np.max(abs(joined.y_true - joined.regional_cloud_percent)))
    assert era_error < 1e-6, era_error
    rows = []
    for (region, model), group in joined.groupby(["region_id", "model"], sort=True):
        for outcome, col in [("CERES_hourbox", "ceres_percent"), ("ERA5_instantaneous", "y_true")]:
            truth = group[col].to_numpy(float)
            estimate = group.prediction.to_numpy(float)
            residual = estimate - truth
            rows.append({"region_id": int(region), "model": model, "outcome_product": outcome,
                         "n": len(group), "MAE_pp": float(np.abs(residual).mean()),
                         "RMSE_pp": float(np.sqrt(np.mean(residual**2))), "bias_pp": float(residual.mean()),
                         "correlation": float(np.corrcoef(truth, estimate)[0, 1]) if np.std(estimate) > 0 and np.std(truth) > 0 else None})
    joined.to_csv(directory / "paired_predictions.csv", index=False, float_format="%.12g")
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False, float_format="%.12g")
    manifest = {"status": "COMPLETE_CROSS_PRODUCT_COMPARISON_NOT_CLOUD_TRUTH_OR_CAUSAL_VALIDATION",
                "created_utc": datetime.now(timezone.utc).isoformat(), "design_sha256": sha(args.root / "design.json"),
                "code_sha256": sha(__file__), "source_hashes": {str(p): sha(p) for p in [prediction_path, args.root / "inputs/input_data.npz", previous / "run_manifest.json", previous / "independent_audit.json"] + [previous / n for n in needed]},
                "geometry_checks": geometry, "era5_target_max_abs_error_pp": era_error,
                "model_rows": len(joined), "unique_region_date_pairs": len(joined[["region_id", "target_time"]].drop_duplicates()),
                "limitation": "2019 historical cross-product comparison only. ERA5 target 06:00 and CERES nominal 06:30 have different temporal operators; original decoded times retained. No bias correction or CERES fit. New global CERES not used in this bounded experiment.",
                "outputs_sha256": {n: sha(directory / n) for n in ["paired_predictions.csv", "metrics.csv"]}}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"rows": len(joined), "metric_rows": len(rows), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
