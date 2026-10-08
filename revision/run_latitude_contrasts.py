"""Fixed-candidate latitude contrasts of local winter-minus-summer slopes.

The fitted latitude slope is a fixed linear contrast, in units per ten degrees,
on three observed target-latitude rings. Same-calendar influence aggregation
retains dependence across candidates and hemispheres. HAC64/128 uncertainty is
exploratory, not calibrated causal, spatial-superpopulation, or FDR inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
import psutil
from scipy import stats
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from causal_weathergraph.candidate_edges import build_candidate_edges
from revision.prepare_inputs import sha256
from revision.run_hemisphere_contrasts import (
    edge_id, edge_key, fit_pair_batch, local_season_masks, mirrored_pairs,
)
from revision.inference import score_autocovariances

BANDS = [("0_30", 0., 30.), ("30_60", 30., 60.), ("60_90", 60., 90.)]
VIEWS = ("NH", "SH", "NH_minus_SH")
LAGS = (1, 2, 3)
BANDWIDTHS = (64, 128)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def latitude_weights(latitudes):
    """Seven fixed contrasts, retaining unavailable bands/zero denominators."""
    latitudes = np.asarray(latitudes, dtype=np.float64)
    check(latitudes.ndim == 1 and np.isfinite(latitudes).all(), "Invalid latitude vector")
    check(((latitudes >= 0) & (latitudes <= 90)).all(), "Latitude outside [0,90]")
    n = len(latitudes)
    x = latitudes / 10.
    centered = x - x.mean() if n else x
    denominator = float(centered @ centered)
    trend_valid = n > 1 and denominator > 1e-12
    output = [{"statistic": "linear_trend_per_10deg", "weights": centered / denominator if trend_valid else np.zeros(n),
               "valid": trend_valid, "reason": "ok" if trend_valid else "zero_latitude_variation_or_empty_population",
               "candidate_count": n, "latitude_x_mean": float(x.mean()) if n else None,
               "latitude_centered_sum_squares": denominator}]
    band_weights, band_counts = {}, {}
    for name, lower, upper in BANDS:
        keep = (latitudes >= lower) & ((latitudes < upper) if upper < 90 else (latitudes <= upper))
        count = int(keep.sum())
        band_counts[name] = count
        band_weights[name] = keep.astype(float) / count if count else np.zeros(n)
        output.append({"statistic": "mean_" + name, "weights": band_weights[name], "valid": count > 0,
                       "reason": "ok" if count else "empty_latitude_band", "candidate_count": count,
                       "latitude_x_mean": None, "latitude_centered_sum_squares": None})
    for high, low in [("30_60", "0_30"), ("60_90", "30_60"), ("60_90", "0_30")]:
        valid = band_counts[high] > 0 and band_counts[low] > 0
        output.append({"statistic": f"difference_{high}_minus_{low}",
                       "weights": band_weights[high] - band_weights[low] if valid else np.zeros(n),
                       "valid": valid, "reason": "ok" if valid else "empty_comparison_band",
                       "candidate_count": band_counts[high] + band_counts[low],
                       "latitude_x_mean": None, "latitude_centered_sum_squares": None})
    return output


def contrast_hac(effect, score, bandwidth):
    acov = score_autocovariances(score, bandwidth)
    h = np.arange(1, bandwidth + 1)
    variance = float(acov[0, 0] + 2 * np.sum((1 - h / (bandwidth + 1)) * acov[h, 0]))
    check(variance >= -1e-12 * max(1., float(acov[0, 0])), "Unexpected negative HAC variance")
    se = float(np.sqrt(max(variance, 0)))
    z = effect / se if se else (0. if effect == 0 else np.inf)
    return {f"se_hac{bandwidth}": se, f"ci95_low_hac{bandwidth}": effect - 1.96 * se,
            f"ci95_high_hac{bandwidth}": effect + 1.96 * se,
            f"p_nominal_hac{bandwidth}": float(2 * stats.norm.sf(abs(z)))}


def population(input_path, previous):
    with np.load(input_path, allow_pickle=False) as a:
        lat, lon, names = a["lat"], a["lon"], a["variable_names"].tolist()
        timestamps = a["timestamps"]
    candidates = build_candidate_edges(names, lat, lon, {"candidate_k_nearest": 2})
    pairs, excluded, reflection = mirrored_pairs(candidates, lat, lon)
    check(len(pairs) == 425 and len(excluded) == 8, "Original frozen population changed")
    rows = []
    for i, (north, south) in enumerate(pairs):
        rows.append({"pair_index": i, "pair_id": f"{edge_id(edge_key(north))}|{edge_id(edge_key(south))}",
                     "target_abs_latitude": float(lat[north.target_region]), "edge_type": north.edge_type,
                     **{f"nh_{k}": v for k, v in north.to_dict().items()},
                     **{f"sh_{k}": v for k, v in south.to_dict().items()}})
    table = pd.DataFrame(rows)
    old = pd.read_csv(previous / "matched_candidates.csv")
    check(table.pair_id.tolist() == old.pair_id.tolist(), "Matched candidates/order changed")
    check(np.allclose(table.target_abs_latitude, old.target_abs_latitude, rtol=0, atol=1e-12), "Latitude changed")
    check(len(timestamps) == 68668 and np.all(np.diff(timestamps) == np.timedelta64(6, "h")), "Calendar changed")
    return table, pairs, reflection, len(candidates)


def create_weight_design(table):
    columns, vectors, support = [], [], []
    specs = [(int(row.pair_index), lag) for row in table.itertuples() for lag in LAGS]
    spec_lookup = {spec: i for i, spec in enumerate(specs)}
    for edge_type in sorted(table.edge_type.unique()):
        group = table[table.edge_type == edge_type]
        for label, lower, upper in BANDS:
            band = group[(group.target_abs_latitude >= lower) &
                         ((group.target_abs_latitude < upper) if upper < 90 else (group.target_abs_latitude <= upper))]
            support.append({"edge_type": edge_type, "band": label, "candidate_pairs": len(band),
                            "distinct_nh_target_regions": int(band.nh_target_region.nunique()),
                            "distinct_nh_source_regions": int(band.nh_source_region.nunique()),
                            "actual_target_absolute_latitudes": ";".join(f"{v:.8g}" for v in sorted(band.target_abs_latitude.unique()))})
        contrasts = latitude_weights(group.target_abs_latitude.to_numpy())
        for lag in LAGS:
            ids = [spec_lookup[(int(i), lag)] for i in group.pair_index]
            for contrast in contrasts:
                vec = np.zeros(len(specs))
                vec[ids] = contrast["weights"]
                col = {k: v for k, v in contrast.items() if k != "weights"}
                col.update(edge_type=edge_type, lag=lag, aggregate_index=len(columns),
                           aggregate_id=f"{edge_type}|lag{lag}|{contrast['statistic']}",
                           population_candidate_pairs=len(group), nonzero_candidate_weights=int((vec != 0).sum()))
                columns.append(col)
                vectors.append(vec)
    return specs, pd.DataFrame(columns), np.column_stack(vectors), pd.DataFrame(support)


def freeze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    table, _, _, candidate_count = population(args.input, args.previous)
    specs, columns, weights, support = create_weight_design(table)
    table.to_csv(args.output / "matched_candidates.csv", index=False)
    columns.to_csv(args.output / "contrast_definitions.csv", index=False)
    support.to_csv(args.output / "latitude_support.csv", index=False)
    np.savez_compressed(args.output / "fixed_contrast_weights.npz", weights=weights,
                        pair_index=np.array([v[0] for v in specs]), lag=np.array([v[1] for v in specs]),
                        aggregate_ids=columns.aggregate_id.to_numpy(dtype=str))
    files = [Path(__file__), ROOT / "revision/run_hemisphere_contrasts.py", ROOT / "revision/inference.py",
             ROOT / "revision/prepare_inputs.py", ROOT / "src/causal_weathergraph/candidate_edges.py",
             ROOT / "tests/test_latitude_contrasts.py"]
    inputs = [args.input, args.input.parent / "data_manifest.json", args.previous / "design.json",
              args.previous / "matched_candidates.csv", args.previous / "excluded_candidates.csv",
              args.previous / "hemisphere_pair_contrasts.csv", ROOT.parent / "阅读材料/Reviewer1附件.txt"]
    design = {"frozen_utc": pd.Timestamp.now(tz="UTC").isoformat(), "status": "frozen_before_this_fit_postdiagnostic_extension",
              "reviewer_source": "Reviewer1附件.txt:103", "reviewer_quote": "formally quantify regime, latitude, mediation, and null-model contrasts; and",
              "input_path": str(args.input.resolve()), "previous_results": str(args.previous.resolve()),
              "source_sha256": {str(p.resolve()): sha256(p) for p in files + inputs},
              "candidate_count": candidate_count, "matched_pairs": len(table), "unmatched_candidates": 8,
              "lags": list(LAGS), "periods": {"discovery": "1979-2018", "evaluation": "2019-2025"},
              "first_stage": "Same separate hemisphere local-winter interaction models as frozen mirrored experiment; local-season-specific intercept/target own lags1-3 and source main/interaction. Source and own-history before masking; original six-hour calendar retained.",
              "views": list(VIEWS), "bands_absolute_degrees": ["[0,30)", "[30,60)", "[60,90]"],
              "observed_absolute_target_latitudes": sorted(table.target_abs_latitude.unique().tolist()),
              "continuous_estimand": "Within each fixed edge-type/lag population, unweighted OLS projection of signed seasonal deltas on intercept and abs(target latitude)/10. Equivalent fixed weights (x-xbar)/sum((x-xbar)^2); no random-edge residual variance or inverse-variance weighting.",
              "band_estimands": "Equal-candidate mean in each band, plus middle-low, high-middle and high-low; preserve candidates and all lags, do not select using fitted outcomes.",
              "joint_covariance": "Apply the identical fixed weights to same-calendar first-stage normalized scores before HAC, then derive NH-SH on the same rows. Covariance across edges, bands, hemispheres and serial lags is retained; candidates are not independent replications.",
              "hac_bandwidths": list(BANDWIDTHS), "covariance_correction": "Each first-stage influence is multiplied by sqrt(n/(n-rank_full)), matching the prior mirrored analysis; no additional second-stage correction for random candidate sampling.",
              "multiplicity": "No BH, significance screening, favorable-bandwidth selection, simultaneous-coverage or FDR claim; all pointwise nominal normal intervals/p values are exploratory.",
              "unavailable_rule": "Keep output row with valid=false and NaN effect/uncertainty if an empty band or zero latitude variation prevents the contrast; no renormalization over failed fits.",
              "expected_rows": 756, "expected_pair_rows": 2550,
              "limits": ["Only three target absolute-latitude rings; per-10-degree slope is their linear summary, not dense continuous-latitude evidence or extrapolation.",
                         "Fixed candidate-average estimand, not area-weighted global inference or a sample of independent edges.",
                         "Retrospective coefficients separately refitted in both periods; training-fitted preprocessing does not make evaluation refits training-frozen predictions.",
                         "Local winter/summer means DJF/JJA in NH and JJA/DJF in SH, including geographic tropics; the labels do not imply equivalent tropical seasons.",
                         "Signed conditional predictive slopes, not absolute strength, latitude intervention effect, causal DID, calibrated FDR, or unconfounded mediation.",
                         "Latitude is confounded with geography and candidate composition; controls remain target history, not the newer six-field physical-control model.",
                         "Evaluation combines WB2 and CDS-era main inputs; source-boundary homogeneity and finite-sample HAC coverage remain unproven."],
              "definition_files_sha256": {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
              "blas_threads": 2, "python": platform.python_version()}
    write_json(args.output / "design.json", design)
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for p in files:
        shutil.copy2(p, snapshot / p.name)
    print(json.dumps({"status": "design_frozen", "definition_sha256": sha256(args.output / "design.json"),
                      "expected_rows": design["expected_rows"], "support": support.to_dict("records")}), flush=True)


def run(args):
    started = time.perf_counter()
    plan_path = args.output / "design.json"
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    plan_hash = sha256(plan_path)
    check(not (args.output / "manifest.json").exists(), "Completed results exist; refuse overwrite")
    check(str(args.input.resolve()) == plan["input_path"] and str(args.previous.resolve()) == plan["previous_results"], "Input paths differ from frozen plan")
    for path, digest in plan["source_sha256"].items():
        check(sha256(path) == digest, f"Frozen source changed: {path}")
    for filename, digest in plan["definition_files_sha256"].items():
        check(sha256(args.output / filename) == digest, f"Frozen definition changed: {filename}")
    table, pairs, reflection, _ = population(args.input, args.previous)
    specs, columns, weights, _ = create_weight_design(table)
    with np.load(args.output / "fixed_contrast_weights.npz", allow_pickle=False) as a:
        check(np.array_equal(weights, a["weights"]), "Rebuilt weights differ from frozen plan")
    with np.load(args.input, allow_pickle=False) as a:
        data, names, timestamps = a["data"], a["variable_names"].tolist(), a["timestamps"]
    check(np.isfinite(data).all(), "Nonfinite input; no implicit imputation")
    lookup = {n: i for i, n in enumerate(names)}
    wn, ws, season = local_season_masks(timestamps)
    training = timestamps < np.datetime64("2019-01-01")
    offset, length = 3, len(data)
    spec_index = {s: i for i, s in enumerate(specs)}
    groups = defaultdict(list)
    for i, pair in enumerate(pairs):
        groups[(pair[0].target_region, pair[0].target_var)].append((i, pair))
    results, pair_rows, quality = [], [], {}
    fitting_started = pd.Timestamp.now(tz="UTC").isoformat()
    with threadpool_limits(limits=2):
        for period, period_mask in [("discovery", training), ("evaluation", ~training)]:
            mask = (period_mask & season)[offset:]
            scores = np.zeros((length - offset, len(columns), 3), dtype=np.float64)
            pair_effects = np.zeros((len(specs), 3))
            for (rn, target), group in sorted(groups.items()):
                rs = reflection[rn]
                y = [data[offset:, r, lookup[target]] for r in (rn, rs)]
                controls = [np.column_stack([data[offset-lag:length-lag, r, lookup[target]] for lag in LAGS]) for r in (rn, rs)]
                entries = [(i, pair, lag) for i, pair in group for lag in LAGS]
                sources = [np.column_stack([data[offset-lag:length-lag, pair[side].source_region, lookup[pair[side].source_var]]
                                           for i, pair, lag in entries]) for side in (0, 1)]
                north, south, difference, _ = fit_pair_batch(y[0], controls[0], sources[0], y[1], controls[1], sources[1],
                                                            wn[offset:], ws[offset:], mask)
                indices = np.array([spec_index[(i, lag)] for i, _, lag in entries])
                group_weights = weights[indices]
                active = np.flatnonzero(np.any(group_weights != 0, axis=0))
                for side, fit in enumerate([north, south]):
                    scores[:, active, side] += fit["influence"] @ group_weights[:, active]
                    pair_effects[indices, side] = fit["delta"]
                pair_effects[indices, 2] = difference
                for j, (i, _, lag) in enumerate(entries):
                    pair_rows.append({"period": period, "pair_index": i, "pair_id": table.iloc[i].pair_id,
                                      "edge_type": table.iloc[i].edge_type, "target_abs_latitude": float(table.iloc[i].target_abs_latitude),
                                      "lag": lag, "delta_nh": float(north["delta"][j]), "delta_sh": float(south["delta"][j]),
                                      "delta_nh_minus_sh": float(difference[j]), "n_samples": north["n_samples"],
                                      "nh_n_winter": north["n_winter"], "sh_n_winter": south["n_winter"],
                                      "rank_full": north["rank_full"]})
            scores[:, :, 2] = scores[:, :, 0] - scores[:, :, 1]
            estimates = weights.T @ pair_effects
            check(np.all(scores[~mask] == 0), "Excluded calendar rows have nonzero scores")
            identity_error = float(np.max(np.abs(estimates[:, 0] - estimates[:, 1] - estimates[:, 2])))
            check(identity_error < 1e-12, "Hemisphere contrast identity failed")
            for col in columns.to_dict("records"):
                i = col["aggregate_index"]
                for j, view in enumerate(VIEWS):
                    row = {**col, "period": period, "view": view, "n_response_times": int(mask.sum()),
                           "effect": float(estimates[i, j]) if col["valid"] else np.nan,
                           "units": "seasonal_signed_slope_difference_per_10_degrees" if col["statistic"] == "linear_trend_per_10deg" else "seasonal_signed_slope_difference"}
                    for b in BANDWIDTHS:
                        row.update(contrast_hac(row["effect"], scores[:, i, j], b) if col["valid"] else
                                   {f"{k}_hac{b}": np.nan for k in ["se", "ci95_low", "ci95_high", "p_nominal"]})
                    results.append(row)
            np.savez_compressed(args.output / f"{period}_calendar_influences.npz", influence=scores,
                                timestamps=timestamps[offset:], eligible=mask, aggregate_ids=columns.aggregate_id.to_numpy(dtype=str),
                                views=np.array(VIEWS))
            quality[period] = {"score_zero_outside_calendar_mask": True, "response_times": int(mask.sum()),
                               "hemisphere_linear_identity_max_error": identity_error,
                               "pair_effect_shape": list(pair_effects.shape), "aggregate_score_shape": list(scores.shape)}
            del scores
            print(json.dumps({"period": period, "status": "fitted", "rows": len(results)}), flush=True)
    result = pd.DataFrame(results)
    pair_result = pd.DataFrame(pair_rows)
    check(len(result) == plan["expected_rows"] and len(pair_result) == plan["expected_pair_rows"], "Unexpected row count")
    previous = pd.read_csv(args.previous / "hemisphere_pair_contrasts.csv")
    merged = pair_result.merge(previous, on=["period", "pair_id", "lag"], suffixes=("_new", "_old"), validate="one_to_one")
    errors = {key: float(np.max(np.abs(merged[key + "_new"] - merged[key + "_old"]))) for key in ["delta_nh", "delta_sh", "delta_nh_minus_sh"]}
    check(max(errors.values()) < 1e-11, "First-stage reproduction disagrees with previous result")
    pair_result.to_csv(args.output / "pair_seasonal_slopes.csv", index=False)
    result.to_csv(args.output / "latitude_contrasts.csv", index=False)
    result[result.statistic == "linear_trend_per_10deg"].to_csv(args.output / "latitude_linear_trends.csv", index=False)
    write_json(args.output / "implementation_validation.json", {"status": "passed", "periods": quality,
                                                               "previous_first_stage_max_abs_errors": errors,
                                                               "all_contrasts_available": bool(result.valid.all())})
    check(sha256(plan_path) == plan_hash, "Frozen plan changed during execution")
    manifest = {"status": "completed", "frozen_plan_sha256": plan_hash, "fitting_started_utc": fitting_started,
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "result_rows": len(result),
                "valid_result_rows": int(result.valid.sum()), "pair_rows": len(pair_result),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_working_set_bytes": int(getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)),
                "limits": plan["limits"], "outputs_sha256": {p.name: sha256(p) for p in args.output.iterdir() if p.is_file() and p.name != "manifest.json"}}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["status", "result_rows", "valid_result_rows", "elapsed_seconds", "peak_working_set_bytes"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT.parent / "revision_outputs/inputs/region_trainfit.npz")
    p.add_argument("--previous", type=Path, default=ROOT.parent / "revision_outputs/hemisphere_contrasts")
    p.add_argument("--output", type=Path, default=ROOT.parent / "revision_outputs/latitude_contrasts")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if not (args.freeze or args.run):
        p.error("Use --freeze first, then --run")
    if args.freeze:
        freeze(args)
    if args.run:
        run(args)


if __name__ == "__main__":
    main()
