"""Mirrored NH-minus-SH contrasts of local winter-minus-summer slopes.

These are retrospective observational slope contrasts, not causal difference-in-
differences estimates. Calendar HAC intervals and BH screens are exploratory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
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
sys.path.insert(0, str(ROOT / "src"))
from causal_weathergraph.candidate_edges import build_candidate_edges
from revision.inference import _basis, adjust_bh, score_autocovariances
from revision.prepare_inputs import sha256

BANDWIDTHS = (32, 64, 128)
LAGS = (1, 2, 3)
CAUTION = (
    "Retrospective conditional predictive slopes under target-own-history controls. "
    "Intervals are exploratory and pointwise; finite-sample HAC calibration remains "
    "unresolved. BH values are screening adjustments, not established FDR control. "
    "No causal difference-in-differences, absolute-strength, or area-weighted-global claim."
)


def edge_key(edge):
    return (edge.source_region, edge.target_region, edge.source_var, edge.target_var)


def edge_id(key):
    source, target, source_var, target_var = key
    return f"r{source}:{source_var}->r{target}:{target_var}"


def mirrored_pairs(candidates, lat, lon, tolerance=1e-8):
    """Exact reflected-coordinate intersection; never replace absent candidates."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Nonfinite region coordinates")
    reflection = {}
    for region in range(len(lat)):
        longitude_difference = (lon - lon[region] + 180.0) % 360.0 - 180.0
        matches = np.flatnonzero(
            (np.abs(lat + lat[region]) <= tolerance)
            & (np.abs(longitude_difference) <= tolerance)
        )
        if len(matches) != 1:
            raise ValueError(f"Region {region} has {len(matches)} unique reflected matches")
        reflection[region] = int(matches[0])
    if any(reflection[reflection[i]] != i for i in reflection):
        raise ValueError("Reflection must be an involution")
    lookup = {edge_key(edge): edge for edge in candidates}
    if len(lookup) != len(candidates):
        raise ValueError("Duplicate candidate key")
    pairs, excluded = [], []
    for candidate_index, edge in enumerate(candidates):
        key = edge_key(edge)
        mirror = (reflection[key[0]], reflection[key[1]], key[2], key[3])
        hemisphere = "NH" if lat[key[1]] > 0 else "SH" if lat[key[1]] < 0 else "equator"
        if hemisphere == "equator" or mirror not in lookup:
            excluded.append(dict(
                candidate_index=candidate_index, candidate_id=edge_id(key),
                hemisphere=hemisphere, expected_mirror_id=edge_id(mirror),
                reason="equatorial_target" if hemisphere == "equator" else "mirror_not_in_candidate_set",
                **edge.to_dict(),
            ))
        elif hemisphere == "NH":
            pairs.append((edge, lookup[mirror]))
    pairs.sort(key=lambda pair: edge_key(pair[0]))
    excluded.sort(key=lambda row: row["candidate_id"])
    return pairs, excluded, reflection


def local_season_masks(timestamps):
    months = pd.DatetimeIndex(timestamps).month.to_numpy()
    nh_winter = np.isin(months, (12, 1, 2))
    sh_winter = np.isin(months, (6, 7, 8))
    return nh_winter, sh_winter, nh_winter | sh_winter


def interaction_influence(y, controls, sources, winter, eligible):
    """E3 saturated seasonal model plus corrected normalized calendar scores.

    Each source column is a separate regression. The score is z*residual/sum(z²)
    times sqrt(n/(n-rank_full)); zero outside eligible timestamps. Taking its
    calendar covariance reproduces E3's finite-sample HAC convention.
    """
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    sources = np.asarray(sources, dtype=float)
    if sources.ndim == 1:
        sources = sources[:, None]
    winter = np.asarray(winter, dtype=float)
    mask = np.asarray(eligible, dtype=bool)
    if not all(np.isfinite(a).all() for a in (y, controls, sources, winter)):
        raise ValueError("Nonfinite regression data; implicit row deletion is forbidden")
    if controls.ndim != 2 or not all(len(a) == len(y) for a in (controls, sources, winter, mask)):
        raise ValueError("Inconsistent regression dimensions")
    n = int(mask.sum())
    nuisance = np.column_stack((np.ones(len(y)), controls, winter, controls * winter[:, None]))
    q, rank = _basis(nuisance[mask])
    rank_full = rank + 2
    if n <= rank_full or not (winter[mask].min() == 0 and winter[mask].max() == 1):
        raise ValueError("Insufficient observations or an empty seasonal group")
    yy = y[mask] - q @ (q.T @ y[mask])
    xx = sources[mask] - q @ (q.T @ sources[mask])
    interaction = (sources * winter[:, None])[mask]
    interaction -= q @ (q.T @ interaction)
    source_ss = np.sum(xx * xx, axis=0)
    if np.any(source_ss < 1e-12):
        raise ValueError("Unidentifiable source")
    ratio = np.sum(xx * interaction, axis=0) / source_ss
    z = interaction - xx * ratio
    baseline_partial = yy @ xx / source_ss
    target = yy[:, None] - xx * baseline_partial
    zss = np.sum(z * z, axis=0)
    if np.any(zss < 1e-12):
        raise ValueError("Unidentifiable source-by-season interaction")
    delta = np.sum(z * target, axis=0) / zss
    residual = target - z * delta
    correction = n / (n - rank_full)
    influence = np.zeros_like(sources)
    influence[mask] = z * residual / zss * np.sqrt(correction)
    beta_summer = baseline_partial - ratio * delta
    return dict(
        delta=delta, beta_summer=beta_summer, beta_winter=beta_summer + delta,
        influence=influence, n_samples=n, n_winter=int(winter[mask].sum()),
        n_summer=int(n - winter[mask].sum()), rank_full=rank_full,
        covariance_correction=correction,
    )


def hac_statistics(effect, influence, bandwidths=BANDWIDTHS):
    """Uncertainty for a coefficient or fixed linear contrast of coefficients."""
    effect = np.atleast_1d(effect).astype(float)
    influence = np.asarray(influence, dtype=float)
    if influence.ndim == 1:
        influence = influence[:, None]
    acov = score_autocovariances(influence, max(bandwidths))
    results = [{} for _ in effect]
    for bandwidth in bandwidths:
        h = np.arange(1, min(bandwidth + 1, len(acov)))
        variance = acov[0] + 2 * np.sum((1 - h[:, None] / (bandwidth + 1)) * acov[h], axis=0)
        se = np.sqrt(np.maximum(variance, 0))
        statistic = np.divide(np.abs(effect), se, out=np.full(len(effect), np.inf), where=se > 0)
        statistic[(se == 0) & (effect == 0)] = 0.0
        p = 2 * stats.norm.sf(statistic)
        for j, result in enumerate(results):
            result.update({
                f"se_hac{bandwidth}": float(se[j]),
                f"ci95_low_hac{bandwidth}": float(effect[j] - 1.96 * se[j]),
                f"ci95_high_hac{bandwidth}": float(effect[j] + 1.96 * se[j]),
                f"p_hac{bandwidth}": float(p[j]),
            })
    return results


def fit_pair_batch(y_nh, controls_nh, x_nh, y_sh, controls_sh, x_sh,
                   winter_nh, winter_sh, eligible):
    north = interaction_influence(y_nh, controls_nh, x_nh, winter_nh, eligible)
    south = interaction_influence(y_sh, controls_sh, x_sh, winter_sh, eligible)
    difference = north["delta"] - south["delta"]
    scores = north["influence"] - south["influence"]
    return north, south, difference, scores


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.input) as archive:
        data = archive["data"]
        names = archive["variable_names"].tolist()
        lat, lon, timestamps = archive["lat"], archive["lon"], archive["timestamps"]
    if not np.isfinite(data).all():
        raise ValueError("Input must be finite; no imputation or row deletion")
    if not np.all(np.diff(timestamps) == np.timedelta64(6, "h")):
        raise ValueError("Input must retain the regular original six-hour calendar")
    training = timestamps < np.datetime64("2019-01-01")
    candidates = build_candidate_edges(names, lat, lon, {"candidate_k_nearest": 2})
    pairs, excluded, reflection = mirrored_pairs(candidates, lat, lon)
    paired_rows = []
    for north, south in pairs:
        row = dict(pair_id=f"{edge_id(edge_key(north))}|{edge_id(edge_key(south))}",
                   nh_candidate_id=edge_id(edge_key(north)), sh_candidate_id=edge_id(edge_key(south)),
                   target_abs_latitude=float(lat[north.target_region]), edge_type=north.edge_type)
        row.update({f"nh_{k}": v for k, v in north.to_dict().items()})
        row.update({f"sh_{k}": v for k, v in south.to_dict().items()})
        paired_rows.append(row)
    pd.DataFrame(paired_rows).to_csv(args.output / "matched_candidates.csv", index=False)
    pd.DataFrame(excluded, columns=[
        "candidate_index", "candidate_id", "hemisphere", "expected_mirror_id", "reason",
        "source_region", "target_region", "source_var", "target_var", "edge_type", "distance_km",
    ]).to_csv(args.output / "excluded_candidates.csv", index=False)
    source_files = [Path(__file__), ROOT / "revision/inference.py",
                    ROOT / "src/causal_weathergraph/candidate_edges.py",
                    ROOT / "tests/test_hemisphere_contrasts.py"]
    design = dict(
        frozen_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        status="Frozen before this analysis fits; post-diagnostic retrospective extension",
        input_path=str(args.input.resolve()), input_sha256=sha256(args.input),
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        source_data_manifest_sha256=sha256(args.input.parent / "data_manifest.json"),
        data_shape=list(data.shape), transformation="Existing training-fitted regional grid-standardized anomalies; not re-fitted here",
        periods={"discovery": "1979-2018", "evaluation": "2019-2025"},
        pairing="Reflect source and target latitude at the same longitude; tolerance 1e-8 degrees; intersect the original k=2 candidate sets",
        reflection={str(k): v for k, v in reflection.items()}, candidate_count=len(candidates),
        matched_candidate_pairs=len(pairs), excluded_candidates=len(excluded),
        matched_candidates_sha256=sha256(args.output / "matched_candidates.csv"),
        excluded_candidates_sha256=sha256(args.output / "excluded_candidates.csv"),
        lags=list(LAGS), expected_pair_lag_rows_per_period=len(pairs) * len(LAGS),
        estimand="(beta_NH_DJF-beta_NH_JJA)-(beta_SH_JJA-beta_SH_DJF); signed slopes in existing standardized-input units",
        eligible_calendar_months=[1, 2, 6, 7, 8, 12],
        controls="Separate hemisphere regressions: local-season-specific intercept and own-history lags1-3; source main effect plus source-by-local-winter interaction",
        covariance="Subtract same-calendar normalized influence scores; retain all cross-hemisphere and serial covariance. Each component is scaled by sqrt(n/(n-rank_full)) to reproduce E3's HAC convention",
        primary_hac_bandwidth=64, sensitivity_hac_bandwidths=[32, 128],
        lag_boundary="Create lagged design before seasonal/period masking; training history supplies evaluation context, no compressed or reassembled calendar",
        aggregate_estimand="Within each period and edge_type, equal weight to EVERY matched candidate-lag contrast; all three lags included; fixed-region population, not area-weighted or a random-region inference",
        aggregate_covariance="Sum normalized same-calendar score vectors with the same equal weights before HAC; no independent-edge approximation",
        multiplicity="Edge-pair BH across all matched candidate-lags separately per period/bandwidth; edge-type aggregate BH across all six types separately per period/bandwidth; additional across-period adjustment exported within each table",
        confidence_intervals="Pointwise exploratory normal intervals, not simultaneous or selection-adjusted",
        caution=CAUTION, blas_threads=2,
        software={name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "threadpoolctl"]},
        python=platform.python_version(), platform=platform.platform(),
    )
    write_json(args.output / "design.json", design)
    design_sha256 = sha256(args.output / "design.json")
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for source in source_files:
        shutil.copy2(source, snapshot / source.name)
    print(f"Design frozen: {len(pairs)} matched candidates; {len(excluded)} exclusions", flush=True)
    fitting_started_utc = pd.Timestamp.now(tz="UTC").isoformat()
    lookup = {name: i for i, name in enumerate(names)}
    winter_nh, winter_sh, season = local_season_masks(timestamps)
    groups = defaultdict(list)
    for pair, metadata in zip(pairs, paired_rows):
        groups[(pair[0].target_region, pair[0].target_var)].append((pair, metadata))
    rows, aggregate_rows = [], []
    full_length, first_response = len(data), max(LAGS)
    edge_types = sorted({pair[0].edge_type for pair in pairs})
    with threadpool_limits(limits=2):
        for period, period_mask in (("discovery", training), ("evaluation", ~training)):
            score_sums = {name: np.zeros(full_length - first_response) for name in edge_types}
            delta_sums = defaultdict(float)
            north_sums, south_sums = defaultdict(float), defaultdict(float)
            counts = defaultdict(int)
            eligible = (season & period_mask)[first_response:]
            for (region_nh, target), pair_group in sorted(groups.items()):
                region_sh = reflection[region_nh]
                target_index = lookup[target]
                ys = [data[first_response:, region, target_index] for region in (region_nh, region_sh)]
                controls = [np.column_stack([
                    data[first_response-lag:full_length-lag, region, target_index] for lag in LAGS
                ]) for region in (region_nh, region_sh)]
                specs = [(pair, metadata, lag) for pair, metadata in pair_group for lag in LAGS]
                sources = [np.column_stack([
                    data[first_response-lag:full_length-lag, pair[side].source_region, lookup[pair[side].source_var]]
                    for pair, _, lag in specs
                ]) for side in (0, 1)]
                north, south, difference, scores = fit_pair_batch(
                    ys[0], controls[0], sources[0], ys[1], controls[1], sources[1],
                    winter_nh[first_response:], winter_sh[first_response:], eligible,
                )
                uncertainty = hac_statistics(difference, scores)
                for j, ((pair, metadata, lag), intervals) in enumerate(zip(specs, uncertainty)):
                    row = dict(metadata, period=period, lag=lag, contrast_id=f"{metadata['pair_id']}|lag{lag}",
                               delta_nh=float(north["delta"][j]), delta_sh=float(south["delta"][j]),
                               delta_nh_minus_sh=float(difference[j]), **intervals)
                    for label, fit in (("nh", north), ("sh", south)):
                        row.update({f"{label}_{key}": fit[key] for key in (
                            "n_samples", "n_winter", "n_summer", "rank_full", "covariance_correction")})
                        row.update({f"{label}_{key}": float(fit[key][j]) for key in ("beta_summer", "beta_winter")})
                    rows.append(row)
                    edge_type = pair[0].edge_type
                    score_sums[edge_type] += scores[:, j]
                    delta_sums[edge_type] += float(difference[j])
                    north_sums[edge_type] += float(north["delta"][j])
                    south_sums[edge_type] += float(south["delta"][j])
                    counts[edge_type] += 1
            for edge_type in edge_types:
                count = counts[edge_type]
                effect = delta_sums[edge_type] / count
                uncertainty = hac_statistics(effect, score_sums[edge_type] / count)[0]
                aggregate_rows.append(dict(
                    period=period, edge_type=edge_type, candidate_lag_pairs=count,
                    candidate_pairs=count // len(LAGS), weight_per_candidate_lag=1 / count,
                    mean_delta_nh=north_sums[edge_type] / count,
                    mean_delta_sh=south_sums[edge_type] / count,
                    mean_delta_nh_minus_sh=effect, n_response_times=int(eligible.sum()), **uncertainty,
                ))
            np.savez_compressed(args.output / f"{period}_aggregate_calendar_scores.npz",
                                timestamps=timestamps[first_response:], edge_types=np.array(edge_types),
                                influence=np.column_stack([score_sums[k] / counts[k] for k in edge_types]))
            print(f"Completed {period}: {sum(counts.values())} pair-lag contrasts", flush=True)
    output = pd.DataFrame(rows)
    aggregates = pd.DataFrame(aggregate_rows)
    for table in (output, aggregates):
        for bandwidth in BANDWIDTHS:
            table[f"q_hac{bandwidth}_period"] = table.groupby("period")[f"p_hac{bandwidth}"].transform(adjust_bh)
            table[f"q_hac{bandwidth}_all_periods"] = adjust_bh(table[f"p_hac{bandwidth}"])
    output.to_csv(args.output / "hemisphere_pair_contrasts.csv", index=False)
    aggregates.to_csv(args.output / "hemisphere_edge_type_averages.csv", index=False)
    summary = output.groupby(["period", "edge_type"]).agg(
        tested=("contrast_id", "size"), median_delta_nh_minus_sh=("delta_nh_minus_sh", "median"),
        positive_contrasts=("delta_nh_minus_sh", lambda values: int((values > 0).sum())),
        exploratory_hac64_bh_crossings=("q_hac64_period", lambda values: int((values < .05).sum())),
    ).reset_index()
    summary.to_csv(args.output / "hemisphere_screen_summary.csv", index=False)
    if sha256(args.output / "design.json") != design_sha256:
        raise RuntimeError("Frozen design changed during fitting")
    manifest = dict(design, frozen_design_sha256=design_sha256,
                    fitting_started_utc=fitting_started_utc,
                    completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                    elapsed_seconds=time.perf_counter()-start,
                    peak_working_set_bytes=int(getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)),
                    pair_contrast_rows=len(output), edge_type_aggregate_rows=len(aggregates))
    manifest["output_sha256"] = {path.name: sha256(path) for path in sorted(args.output.iterdir())
                                 if path.is_file() and path.name != "manifest.json"}
    write_json(args.output / "manifest.json", manifest)
    print(f"Complete: {manifest['elapsed_seconds']:.2f} seconds; peak {manifest['peak_working_set_bytes']} bytes", flush=True)


if __name__ == "__main__":
    main()
