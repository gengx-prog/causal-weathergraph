"""E5: frozen, temporally aligned path, vector and joint-lag diagnostics.

All HAC probabilities/intervals are exploratory sensitivity statistics: the
revision simulation checks do not establish calibrated finite-sample FDR.
Nothing here identifies a natural indirect effect or a moisture trajectory.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from revision.inference import fit_edge, score_autocovariances, adjust_bh
from revision.graph_nulls import distance_matrix


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str,
                               allow_nan=False), encoding="utf-8")


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing in radians, north=0, east=pi/2."""
    phi1, phi2 = np.deg2rad([lat1, lat2])
    delta = np.deg2rad(lon2 - lon1)
    return float(np.arctan2(np.sin(delta) * np.cos(phi2),
                            np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta)))


def project_east_north(east: np.ndarray, north: np.ndarray, bearing: float) -> np.ndarray:
    return east * np.sin(bearing) + north * np.cos(bearing)


def train_monthly_transform(values: np.ndarray, months: np.ndarray,
                            train: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit climatology, anomaly mean and scale only on training observations."""
    climatology = np.array([values[train & (months == month)].mean() for month in range(1, 13)])
    anomalies = values - climatology[months - 1]
    center = float(anomalies[train].mean())
    scale = float(anomalies[train].std())
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError("Projection has insufficient training variation")
    return (anomalies - center) / scale, {"monthly_climatology": climatology.tolist(),
                                         "anomaly_mean": center, "anomaly_scale": scale}


def path_design(data: np.ndarray, names: list[str], i: int, h: int, j: int,
                lag1: int, lag2: int) -> dict[str, np.ndarray]:
    """W_i(t-l1-l2) -> H_h(t-l2) -> C_j(t), with pre-response controls."""
    index = {v: k for k, v in enumerate(names)}
    t = np.arange(max(lag1 + lag2, lag2 + 3, 3), len(data))
    mediator_time = t - lag2
    target_controls = np.column_stack([data[t - lag, j, index[var]]
                                      for var in ("cloud_cover", "temperature")
                                      for lag in (1, 2, 3)])
    mediator_controls = np.column_stack([data[mediator_time - lag, h, index[var]]
                                        for var in ("humidity", "temperature")
                                        for lag in (1, 2, 3)])
    return {"t": t, "mediator_time": mediator_time, "source_time": t - lag1 - lag2,
            "source": data[t - lag1 - lag2, i, index["wind"]],
            "mediator": data[mediator_time, h, index["humidity"]],
            "target": data[t, j, index["cloud_cover"]],
            "target_controls": target_controls, "mediator_controls": mediator_controls}


def hac_mean_summary(values: np.ndarray, bandwidth: int = 64) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    acov = score_autocovariances(values - mean, bandwidth)[:, 0]
    h = np.arange(1, len(acov))
    variance = max(0., float(acov[0] + 2 * np.sum((1 - h / (bandwidth + 1)) * acov[h]))) / n**2
    se = np.sqrt(variance)
    p = float(2 * stats.norm.sf(abs(mean) / se)) if se > 0 else float(mean == 0)
    return {"mse_gain": mean, "mse_gain_se_hac64": float(se),
            "mse_gain_ci95_low": float(mean - 1.959963984540054 * se),
            "mse_gain_ci95_high": float(mean + 1.959963984540054 * se),
            "p_predictive_gain_hac64": p, "evaluation_n": n}


def prediction_gain(y: np.ndarray, restricted: np.ndarray, full: np.ndarray,
                     train: np.ndarray, evaluate: np.ndarray) -> dict:
    """Freeze least-squares coefficients on training data; compare heldout losses."""
    xr = np.column_stack((np.ones(len(y)), restricted))
    xf = np.column_stack((np.ones(len(y)), full))
    br = np.linalg.lstsq(xr[train], y[train], rcond=None)[0]
    bf = np.linalg.lstsq(xf[train], y[train], rcond=None)[0]
    loss_r = (y[evaluate] - xr[evaluate] @ br)**2
    loss_f = (y[evaluate] - xf[evaluate] @ bf)**2
    result = hac_mean_summary(loss_r - loss_f)
    result.update(training_n=int(train.sum()), mse_restricted=float(loss_r.mean()),
                  mse_full=float(loss_f.mean()),
                  heldout_delta_r2=float((loss_r.mean() - loss_f.mean()) / np.var(y[evaluate])))
    return result


def compact_fit(prefix: str, result: dict) -> dict:
    return {f"{prefix}_{k}": result[k] for k in
            ("effect", "p_hac64", "ci95_low_hac64", "ci95_high_hac64", "partial_r2", "n_samples")}


def select_paths(training: pd.DataFrame, distances: np.ndarray,
                  seed: int, per_stratum: int = 10) -> pd.DataFrame:
    wh = training[(training.source_var == "wind") & (training.target_var == "humidity")]
    hc = training[(training.source_var == "humidity") & (training.target_var == "cloud_cover")]
    left = wh[["source_region", "target_region", "lag", "q_hac64_global", "partial_r2"]].rename(
        columns={"source_region": "i", "target_region": "h", "lag": "lag1", "q_hac64_global": "q_wh", "partial_r2": "r2_wh"})
    right = hc[["source_region", "target_region", "lag", "q_hac64_global", "partial_r2"]].rename(
        columns={"source_region": "h", "target_region": "j", "lag": "lag2", "q_hac64_global": "q_hc", "partial_r2": "r2_hc"})
    possible = left.merge(right, on="h")
    possible["mediator_location"] = np.where(possible.h == possible.j, "target_region", "different_region")
    possible["training_score"] = np.sqrt(possible.r2_wh.clip(lower=0) * possible.r2_hc.clip(lower=0))
    possible["both_selected"] = (possible.q_wh < .05) & (possible.q_hc < .05)
    rng = np.random.default_rng(seed)
    selections = []
    for location, group in possible.groupby("mediator_location", sort=True):
        cutoff = group.loc[group.both_selected, "training_score"].quantile(.75)
        group = group.copy()
        group["stratum"] = np.where(~group.both_selected, "unselected",
                                    np.where(group.training_score >= cutoff, "strong", "ordinary"))
        for stratum, subset in group.groupby("stratum", sort=True):
            chosen = subset.iloc[rng.choice(len(subset), min(per_stratum, len(subset)), replace=False)].copy()
            chosen["eligible_in_stratum"] = len(subset)
            chosen["strong_score_cutoff"] = cutoff
            selections.append(chosen)
    selected = pd.concat(selections, ignore_index=True)
    selected.insert(0, "path_id", np.arange(len(selected)))
    alternatives = []
    errors = []
    for row in selected.itertuples():
        discrepancy = np.abs(distances[int(row.i)] - distances[int(row.i), int(row.h)]) + np.abs(distances[:, int(row.j)] - distances[int(row.h), int(row.j)])
        discrepancy[int(row.h)] = np.inf
        nearest = np.flatnonzero(np.isclose(discrepancy, discrepancy.min()))
        alternative = int(rng.choice(nearest))
        alternatives.append(alternative)
        errors.append(float(discrepancy[alternative]))
    selected["alternative_h"] = alternatives
    selected["alternative_distance_mismatch_km"] = errors
    return selected


def select_joint_pairs(training: pd.DataFrame, seed: int, per_stratum: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    core = training[training.edge_type.isin(["wind_to_humidity", "humidity_to_cloud_cover", "wind_to_cloud_cover"])]
    pairs = core.groupby(["source_region", "target_region", "source_var", "target_var", "edge_type"], as_index=False).agg(
        best_training_q=("q_hac64_global", "min"), best_training_r2=("partial_r2", "max"))
    selected = []
    for edge_type, group in pairs.groupby("edge_type", sort=True):
        group = group.copy()
        discoveries = group.best_training_q < .05
        cutoff = group.loc[discoveries, "best_training_r2"].quantile(.75)
        group["stratum"] = np.where(~discoveries, "unselected", np.where(group.best_training_r2 >= cutoff, "strong", "ordinary"))
        for stratum, subset in group.groupby("stratum", sort=True):
            choice = subset.iloc[rng.choice(len(subset), min(per_stratum, len(subset)), replace=False)].copy()
            choice["eligible_in_stratum"] = len(subset)
            selected.append(choice)
    output = pd.concat(selected, ignore_index=True)
    output.insert(0, "pair_id", np.arange(len(output)))
    return output


def annotate_distance_matches(paths: pd.DataFrame, distances: np.ndarray) -> pd.DataFrame:
    """Geographic-only calipers; an unmatched nearest region is not a matched control."""
    out = paths.copy()
    records = []
    for row in out.itertuples():
        i, h, j, alternative = int(row.i), int(row.h), int(row.j), int(row.alternative_h)
        leg1, leg2 = float(distances[i, h]), float(distances[h, j])
        error1 = float(abs(distances[i, alternative] - leg1))
        error2 = float(abs(distances[alternative, j] - leg2))
        caliper1, caliper2 = max(500., .25 * leg1), max(500., .25 * leg2)
        admissible = (np.abs(distances[i] - leg1) <= caliper1) & (np.abs(distances[:, j] - leg2) <= caliper2)
        admissible[h] = False
        valid = error1 <= caliper1 and error2 <= caliper2
        records.append({"original_leg1_km": leg1, "original_leg2_km": leg2,
                        "alternative_leg1_error_km": error1, "alternative_leg2_error_km": error2,
                        "leg1_caliper_km": caliper1, "leg2_caliper_km": caliper2,
                        "valid_distance_match": bool(valid),
                        "distance_match_status": "valid_distance_match" if valid else "no_adequate_match",
                        "available_caliper_alternatives": int(admissible.sum())})
    return pd.concat((out.reset_index(drop=True), pd.DataFrame(records)), axis=1)


def write_distance_match_comparisons(output: Path, paths: pd.DataFrame,
                                     path_results: pd.DataFrame) -> None:
    paths.to_csv(output / "path_distance_match_flags.csv", index=False)
    evaluated = path_results[path_results.period == "evaluation"]
    paired = evaluated.pivot(index="path_id", columns="mediator_role", values="heldout_delta_r2").reset_index()
    paired = paired.merge(paths[["path_id", "stratum", "mediator_location", "valid_distance_match", "distance_match_status"]], on="path_id")
    paired["specified_minus_alternative"] = paired.specified - paired.alternative
    paired.to_csv(output / "path_alternative_comparison.csv", index=False)
    paired.groupby(["distance_match_status", "stratum", "mediator_location"], as_index=False).agg(
        paths=("path_id", "size"),
        specified_better=("specified_minus_alternative", lambda values: int((values > 0).sum())),
        median_difference=("specified_minus_alternative", "median")
    ).to_csv(output / "path_alternative_comparison_summary.csv", index=False)


def joint_lag_design(data: np.ndarray, names: list[str], source_region: int,
                      target_region: int, source_var: str, target_var: str,
                      max_source_lag: int = 12) -> tuple:
    index = {v: k for k, v in enumerate(names)}
    t = np.arange(max(max_source_lag, 3), len(data))
    y = data[t, target_region, index[target_var]]
    own = np.column_stack([data[t - lag, target_region, index[target_var]] for lag in (1, 2, 3)])
    sources = np.column_stack([data[t - lag, source_region, index[source_var]] for lag in range(1, max_source_lag + 1)])
    return t, y, own, sources


def run(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=True) as z:
        data, physical = z["data"], z["physical"]
        names = [str(v) for v in z["variable_names"]]
        timestamps = pd.to_datetime(z["timestamps"])
        lat, lon = z["lat"], z["lon"]
    training = pd.read_csv(args.discovery)
    train_time = np.asarray(timestamps.year <= 2018)
    eval_time = np.asarray(timestamps.year >= 2019)
    months = np.asarray(timestamps.month)
    distances = distance_matrix(lat, lon)
    paths = annotate_distance_matches(select_paths(training, distances, args.seed), distances)
    joint_pairs = select_joint_pairs(training, args.seed + 1)
    spatial = training[(training.source_var == "wind") & (training.target_var == "humidity") &
                       (training.source_region != training.target_region)][["source_region", "target_region"]].drop_duplicates()
    paths.to_csv(output / "frozen_paths.csv", index=False)
    joint_pairs.to_csv(output / "frozen_joint_lag_pairs.csv", index=False)
    spatial.to_csv(output / "frozen_vector_region_pairs.csv", index=False)
    design = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "input_path": str(args.input.resolve()), "input_sha256": file_hash(args.input),
        "discovery_path": str(args.discovery.resolve()), "discovery_sha256": file_hash(args.discovery),
        "data_shape": list(data.shape), "training": "1979-2018", "evaluation": "2019-2025",
        "training_timestamps": int(train_time.sum()), "evaluation_timestamps": int(eval_time.sum()),
        "selection_uses_only": "training discovery table, coordinates and fixed seed; no evaluation result file is read",
        "path_selection": "10 each of h=j/h!=j x strong/ordinary/unselected; strong=top quartile geometric mean partialR2 among both training-selected edges; unselected=at least one edge not selected",
        "path_count": len(paths), "joint_pair_count": len(joint_pairs), "vector_region_pair_count": len(spatial),
        "joint_pair_selection": "4 each of three types x strong/ordinary/unselected; strong=upper quartile max training partialR2 among selected pairs; unselected=none of 1-3 lags selected. Fewer than4 reported if a stratum is exhausted.",
        "path_alignment": "W_i[t-l1-l2] -> H_h[t-l2] -> C_j[t]",
        "path_controls_outcome": "C_j[t-1:t-3] and T_j[t-1:t-3]; direct adds W at total lag; full also adds H_h at lag2",
        "path_controls_mediator": "H_h and T_h at the 3 timestamps preceding mediator time t-lag2",
        "alternative_mediator": "other region minimizing sum of absolute source-to-h and h-to-target great-circle distance discrepancies; seed resolves exact ties; mismatch explicitly reported",
        "alternative_distance_calipers": "Each leg error must be <= max(500 km,25% original leg distance); both legs must pass. Otherwise nearest-unmatched only, not matched evidence.",
        "vector_projection": "physical regional east/north wind or q*u/q*v at source projected along initial source-to-target great-circle bearing; monthly climate and scale fitted1979-2018 only",
        "vector_baseline": "target H own3lags + scalar sourceW at testedlag; wind and moisture-flux projections each added separately",
        "vector_scope": "850hPa proxy, not column-integrated transport or a trajectory. qu/qv were multiplied at grid cells before regional averaging.",
        "joint_lag_controls": "target own3 fixed; all source1-12 simultaneously; each lag tested controlling other11",
        "joint_windows": [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12]],
        "predictive_gains": "coefficients frozen from training; test mean squared-error reduction, with exploratory HAC64 mean-loss interval",
        "inferential_scope": "all HAC64 intervals/p-values and derived BH values exploratory sensitivity only; calibration not established by finite-sample simulation. a*b is descriptive linear path product, not identified mediation.",
        "threads_max": 2,
        "code_sha256": {"runner": file_hash(Path(__file__)), "inference": file_hash(ROOT / "revision/inference.py")},
        "status": "design_frozen_before_evaluation", "timings_seconds": {},
    }
    dump_json(output / "design.json", design)
    print(f"Frozen {len(paths)} paths, {len(joint_pairs)} lag pairs, {len(spatial)} vector pairs", flush=True)
    phase = time.perf_counter()
    path_results = []
    for path in paths.itertuples():
        for mediator_role, h in [("specified", int(path.h)), ("alternative", int(path.alternative_h))]:
            aligned = path_design(data, names, int(path.i), h, int(path.j), int(path.lag1), int(path.lag2))
            t, w, humidity, y = aligned["t"], aligned["source"], aligned["mediator"], aligned["target"]
            c, ch = aligned["target_controls"], aligned["mediator_controls"]
            train, evaluate = train_time[t], eval_time[t]
            predictive = prediction_gain(y, np.column_stack((c, w)), np.column_stack((c, w, humidity)), train, evaluate)
            for period, mask in [("training", train), ("evaluation", evaluate)]:
                a = fit_edge(humidity, ch, w, mask, bandwidths=(64,))
                direct = fit_edge(y, c, w, mask, bandwidths=(64,))
                adjusted = fit_edge(y, np.column_stack((c, humidity)), w, mask, bandwidths=(64,))
                b = fit_edge(y, np.column_stack((c, w)), humidity, mask, bandwidths=(64,))
                row = {"path_id": path.path_id, "i": path.i, "h": h, "original_h": path.h, "j": path.j,
                       "lag1": path.lag1, "lag2": path.lag2, "total_lag": path.lag1 + path.lag2,
                       "stratum": path.stratum, "mediator_location": path.mediator_location,
                       "mediator_role": mediator_role, "period": period,
                       "linear_ab_diagnostic": a["effect"] * b["effect"],
                       "absolute_w_attenuation": 1 - abs(adjusted["effect"]) / abs(direct["effect"]) if abs(direct["effect"]) > 1e-12 else np.nan}
                for label, fit in [("a", a), ("direct_w", direct), ("adjusted_w", adjusted), ("b", b)]:
                    row.update(compact_fit(label, fit))
                if period == "evaluation":
                    row.update(predictive)
                path_results.append(row)
    path_frame = pd.DataFrame(path_results)
    path_frame.to_csv(output / "aligned_path_results.csv", index=False)
    write_distance_match_comparisons(output, paths, path_frame)
    path_frame[path_frame.period == "evaluation"].groupby(
        ["mediator_role", "stratum", "mediator_location"], as_index=False).agg(
            paths=("path_id", "size"), median_attenuation=("absolute_w_attenuation", "median"),
            median_heldout_delta_r2=("heldout_delta_r2", "median"),
            positive_predictive_gain=("mse_gain", lambda values: int((values > 0).sum())),
            positive_pointwise_hac_interval=("mse_gain_ci95_low", lambda values: int((values > 0).sum()))
        ).to_csv(output / "path_summary.csv", index=False)
    design["timings_seconds"]["paths"] = time.perf_counter() - phase
    print(f"Paths complete {design['timings_seconds']['paths']:.1f}s", flush=True)

    phase = time.perf_counter()
    vectors = []
    transforms = []
    index = {v: k for k, v in enumerate(names)}
    t = np.arange(3, len(data))
    for pair in spatial.itertuples(index=False):
        i, j = int(pair.source_region), int(pair.target_region)
        bearing = initial_bearing(lat[i], lon[i], lat[j], lon[j])
        for label, east_name, north_name in [("wind_projection", "u", "v"), ("moisture_flux_projection", "qu", "qv")]:
            raw = project_east_north(physical[:, i, index[east_name]], physical[:, i, index[north_name]], bearing)
            projection, parameters = train_monthly_transform(raw, months, train_time)
            transforms.append({"source_region": i, "target_region": j, "projection": label,
                               "bearing_radians": bearing, **parameters})
            for lag in (1, 2, 3):
                y = data[t, j, index["humidity"]]
                w = data[t - lag, i, index["wind"]]
                own = np.column_stack([data[t - k, j, index["humidity"]] for k in (1, 2, 3)])
                baseline = np.column_stack((own, w))
                z = projection[t - lag]
                fit = fit_edge(y, baseline, z, eval_time[t], bandwidths=(64,))
                prediction = prediction_gain(y, baseline, np.column_stack((baseline, z)), train_time[t], eval_time[t])
                vectors.append({"source_region": i, "target_region": j, "lag": lag, "projection": label,
                                "bearing_radians": bearing, **compact_fit("projection", fit), **prediction})
    vector_frame = pd.DataFrame(vectors)
    vector_frame["q_projection_hac64_exploratory"] = adjust_bh(vector_frame.projection_p_hac64)
    vector_frame["q_predictive_gain_exploratory"] = adjust_bh(vector_frame.p_predictive_gain_hac64)
    vector_frame.to_csv(output / "vector_projection_results.csv", index=False)
    vector_frame.groupby("projection", as_index=False).agg(
        hypotheses=("lag", "size"), median_heldout_delta_r2=("heldout_delta_r2", "median"),
        positive_predictive_gain=("mse_gain", lambda values: int((values > 0).sum())),
        exploratory_q_coefficient_below05=("q_projection_hac64_exploratory", lambda values: int((values < .05).sum()))
    ).to_csv(output / "vector_summary.csv", index=False)
    dump_json(output / "vector_training_transforms.json", {"transforms": transforms})
    design["timings_seconds"]["vectors"] = time.perf_counter() - phase
    print(f"Vectors complete {design['timings_seconds']['vectors']:.1f}s", flush=True)

    phase = time.perf_counter()
    lag_results, window_results = [], []
    for pair in joint_pairs.itertuples():
        t, y, own, sources = joint_lag_design(data, names, int(pair.source_region), int(pair.target_region), pair.source_var, pair.target_var)
        train, evaluate = train_time[t], eval_time[t]
        full = np.column_stack((own, sources))
        common = {"pair_id": pair.pair_id, "source_region": pair.source_region, "target_region": pair.target_region,
                  "source_var": pair.source_var, "target_var": pair.target_var, "edge_type": pair.edge_type, "stratum": pair.stratum}
        for lag in range(1, 13):
            controls = np.column_stack((own, np.delete(sources, lag - 1, axis=1)))
            fit = fit_edge(y, controls, sources[:, lag - 1], evaluate, bandwidths=(64,))
            predictive = prediction_gain(y, controls, full, train, evaluate)
            lag_results.append({**common, "lag": lag, **compact_fit("conditional_lag", fit), **predictive})
        for window in [(1, 2, 3), (4, 5, 6, 7, 8), (9, 10, 11, 12)]:
            dropped = [lag - 1 for lag in window]
            restricted = np.column_stack((own, np.delete(sources, dropped, axis=1)))
            predictive = prediction_gain(y, restricted, full, train, evaluate)
            window_results.append({**common, "window": f"{min(window)}-{max(window)}", "window_lags": len(window), **predictive})
        window_results.append({**common, "window": "1-12", "window_lags": 12,
                               **prediction_gain(y, own, full, train, evaluate)})
    lag_frame = pd.DataFrame(lag_results)
    lag_frame["q_conditional_lag_hac64_exploratory"] = adjust_bh(lag_frame.conditional_lag_p_hac64)
    lag_frame["q_predictive_gain_exploratory"] = adjust_bh(lag_frame.p_predictive_gain_hac64)
    lag_frame.to_csv(output / "joint_lag_results.csv", index=False)
    window_frame = pd.DataFrame(window_results)
    window_frame["q_predictive_gain_exploratory"] = adjust_bh(window_frame.p_predictive_gain_hac64)
    window_frame.to_csv(output / "joint_window_results.csv", index=False)
    window_frame.groupby("window", as_index=False).agg(
        pairs=("pair_id", "size"), median_heldout_delta_r2=("heldout_delta_r2", "median"),
        positive_predictive_gain=("mse_gain", lambda values: int((values > 0).sum()))
    ).to_csv(output / "joint_window_summary.csv", index=False)
    design["timings_seconds"]["joint_lags"] = time.perf_counter() - phase
    design["total_wall_seconds"] = time.perf_counter() - start
    design["status"] = "complete"
    design["completed_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(output / "design.json", design)
    print(f"E5 COMPLETE {design['total_wall_seconds']:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026092905)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
