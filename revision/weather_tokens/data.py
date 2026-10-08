"""Training-only data transforms for the bounded weather-token prototype.

Physical regional means are read directly; the older 1979--2018 standardized
arrays are never inverted or used. No network access and no model fitting.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT.parent / "revision_outputs"
NODE_IDS = np.array([2, 8, 13, 19, 24, 30, 35, 41, 46, 52, 57, 63], dtype=np.int64)
VARIABLES = ("u", "v", "humidity", "temperature", "omega_700", "cloud_cover")
UNITS = ("m s-1", "m s-1", "kg kg-1", "K", "Pa s-1", "fraction")
HORIZONS = np.array([1, 2, 4], dtype=np.int64)
FEATURES = tuple(f"{kind}:{name}" for kind in ("abs", "anom", "delta") for name in VARIABLES)
TRAIN_END = np.datetime64("2015-01-01", "ns")
VAL_END = np.datetime64("2019-01-01", "ns")
EXTENSION_START = np.datetime64("2023-01-11", "ns")
PHYSICAL_EDGES = np.array([
    [-20, -10, -5, -1, 1, 5, 10, 20],
    [-20, -10, -5, -1, 1, 5, 10, 20],
    [.001, .002, .004, .006, .008, .010, .014, .018],
    [240, 250, 260, 270, 280, 290, 300, 310],
    [-.3, -.1, -.03, -.005, .005, .03, .1, .3],
    [.05, .15, .3, .45, .55, .7, .85, .95],
], dtype=np.float64)
ANOMALY_EDGES = np.array([-2, -1, -.5, -.15, .15, .5, 1, 2], dtype=np.float64)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def load_raw():
    """Return physical values (T,12,6), with ordered node IDs and provenance.

    NPZ compression prevents memory mapping; each physical array is loaded once,
    selected into the small float32 result and released before reading the next.
    """
    vectors = OUTPUTS / "inputs/region_trainfit_vectors.npz"
    controls = OUTPUTS / "physical_controls_inputs/region_controls_trainfit.npz"
    with np.load(vectors, allow_pickle=False) as archive:
        node_ids = archive["node_ids"]
        selected = np.array([np.flatnonzero(node_ids == node)[0] for node in NODE_IDS])
        if len(np.unique(node_ids)) != len(node_ids):
            raise ValueError("Duplicate region IDs")
        timestamps = archive["timestamps"].astype("datetime64[ns]")
        lat, lon = archive["lat"][selected], archive["lon"][selected]
        names = archive["variable_names"].tolist()
        raw = np.empty((len(timestamps), len(NODE_IDS), len(VARIABLES)), dtype=np.float32)
        physical = archive["physical"]
        for column, name in enumerate(VARIABLES):
            if name != "omega_700":
                raw[:, :, column] = physical[:, selected, names.index(name)]
        del physical
    with np.load(controls, allow_pickle=False) as archive:
        if not np.array_equal(archive["timestamps"], timestamps):
            raise ValueError("Physical-control and vector time axes differ")
        if not np.array_equal(archive["node_ids"], node_ids):
            raise ValueError("Physical-control and vector region axes differ")
        if not (np.array_equal(archive["lat"][selected], lat) and np.array_equal(archive["lon"][selected], lon)):
            raise ValueError("Physical-control and vector coordinates differ")
        column = archive["variable_names"].tolist().index("omega_700")
        if str(archive["variable_units"][column]) != "Pa s**-1":
            raise ValueError("Unexpected omega units")
        physical = archive["physical"]
        raw[:, :, 4] = physical[:, selected, column]
        del physical
        acquisition_segment = archive["source_segment"].copy()
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite physical data; no imputation is permitted")
    if len(np.unique(timestamps)) != len(timestamps) or not np.all(np.diff(timestamps) == np.timedelta64(6, "h")):
        raise ValueError("Missing, duplicate, or non-six-hourly timestamps")
    if raw[:, :, 5].min() < 0 or raw[:, :, 5].max() > 1:
        raise ValueError("Cloud cover is not a fraction in [0,1]")
    return {
        "raw": raw, "timestamps": timestamps, "lat": lat, "lon": lon,
        "node_ids": NODE_IDS.copy(), "variable_names": np.array(VARIABLES),
        "variable_units": np.array(UNITS), "source_segment": acquisition_segment,
        "source_hashes": {str(path): _sha(path) for path in (vectors, controls)},
        "source_notes": [
            "physical is an arithmetic mean of raw values within each original region; no temporal fitting was used to compute it.",
            "float64 stored regional means are converted to float32 for this prototype; source grid values were float32.",
            "Regions are coarse 6-latitude by 11-longitude bins; these are not point measurements or area-weighted regional averages.",
            "The old data arrays and 1979-2018 fit parameters/training_mask are not used; 1979-2014 parameters are fitted separately for each case.",
            "WeatherBench2 conservative 64x32 data cover 1979-2023-01-10. The 2023-01-11 onward CDS extension is a distinct acquisition segment.",
            "Original u/v/q850/t850/tcc extension used a CDS 5.625-degree grid request; omega700 extension used native-grid conservative aggregation. Cross-variable/cross-source kernel equivalence is not established.",
            "All source boundaries, including the 2019 file-segment boundary within WeatherBench2, are excluded from context/target windows by design.",
            "Pressure-level values were not masked below terrain; a coarse surface-pressure diagnostic does not establish above-ground validity.",
            "ERA5 reanalysis is retrospective and cannot demonstrate operational real-time forecasting performance.",
        ],
    }


def _calendar(timestamps, lat):
    months = (timestamps.astype("datetime64[M]").astype(np.int64) % 12).astype(np.int64)
    local_months = (months[:, None] + 6 * (lat[None, :] < 0)) % 12
    abs_lat = np.round(np.abs(lat), 6)
    bands, band_index = np.unique(abs_lat, return_inverse=True)
    return months, local_months, bands, band_index


def _metadata(timestamps, lat, lon):
    metadata = np.empty((len(timestamps), len(lat), 6), dtype=np.float32)
    for k, values in enumerate((np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
                                np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)))):
        metadata[:, :, k] = values[None, :]
    year_start = timestamps.astype("datetime64[Y]").astype("datetime64[ns]")
    next_year = (timestamps.astype("datetime64[Y]") + np.timedelta64(1, "Y")).astype("datetime64[ns]")
    phase = np.asarray((timestamps - year_start) / (next_year - year_start), dtype=np.float64)
    angle = 2*np.pi*(phase[:, None] + .5*(lat[None, :] < 0))
    metadata[:, :, 4] = np.sin(angle)
    metadata[:, :, 5] = np.cos(angle)
    return metadata


def _centers(values, codes, boundaries):
    """Training bin means; finite, data-independent fallback for empty bins."""
    counts = np.bincount(codes, minlength=9).astype(np.int64)
    totals = np.bincount(codes, weights=values.astype(np.float64), minlength=9)
    centers = np.empty(9, dtype=np.float64)
    centers[1:8] = (boundaries[:-1] + boundaries[1:])/2
    gaps = np.diff(boundaries)
    positive = gaps[gaps > 0]
    fallback_gap = float(np.median(positive)) if len(positive) else 1.
    centers[0] = boundaries[0] - .5*(gaps[0] if gaps[0] > 0 else fallback_gap)
    centers[8] = boundaries[-1] + .5*(gaps[-1] if gaps[-1] > 0 else fallback_gap)
    nonempty = counts > 0
    centers[nonempty] = totals[nonempty]/counts[nonempty]
    return centers, counts


def _indices(timestamps, lat, source_regions, target_regions):
    hours = timestamps.astype("datetime64[h]").astype(np.int64) % 24
    t = np.arange(len(timestamps), dtype=np.int64)
    base = (hours == 0) & (t >= 8) & (t+4 < len(timestamps))
    segment = np.where(timestamps < VAL_END, 0, np.where(timestamps < EXTENSION_START, 1, 2))
    valid_source = np.zeros(len(timestamps), dtype=bool)
    candidates = t[base]
    # On the verified regular time axis, monotonic segment IDs make endpoints
    # sufficient to check all 13 six-hour steps t-8,...,t+4.
    valid_source[candidates] = segment[candidates-8] == segment[candidates+4]
    masks = {"train": timestamps < TRAIN_END,
             "val": (timestamps >= TRAIN_END) & (timestamps < VAL_END),
             "test": timestamps >= VAL_END}
    answer, dropped = {}, {}
    for split, mask in masks.items():
        eligible = base & mask
        ids = t[eligible]
        eligible[ids] &= mask[ids+4]
        before = np.flatnonzero(eligible)
        region_ids = target_regions if split == "test" else source_regions
        final_times = np.flatnonzero(eligible & valid_source)
        answer[split] = np.column_stack((np.repeat(final_times, len(region_ids)), np.tile(region_ids, len(final_times)))).astype(np.int64)
        dropped[split] = {"origin_times_before_source_filter": int(len(before)),
                          "origin_times_dropped_cross_source": int(np.count_nonzero(eligible & ~valid_source)),
                          "samples_dropped_cross_source": int(np.count_nonzero(eligible & ~valid_source)*len(region_ids)),
                          "origin_times_retained": int(len(final_times)), "region_count": int(len(region_ids)),
                          "samples_retained": int(len(answer[split]))}
    return answer, dropped


def _diagnose(indices, F, Q, S, centers_q, centers_s, labels):
    answer = {}
    for split, ids in indices.items():
        if not len(ids):
            answer[split] = {"samples": 0, "class_counts": [[0]*3 for _ in HORIZONS], "Q": None, "S": None}
            continue
        t, r = ids.T
        values = F[t, r].astype(np.float64)
        counts = [np.bincount(labels[t, r, j], minlength=3).astype(int).tolist() for j in range(3)]
        answer[split] = {"samples": int(len(ids)), "class_counts": counts}
        for name, codes, centers in (("Q", Q, centers_q), ("S", S, centers_s)):
            chosen = codes[t, r]
            reconstructed = centers[np.arange(18)[None, :], chosen]
            answer[split][name] = {
                "rmse_per_feature_in_F_units": np.sqrt(np.mean((values-reconstructed)**2, axis=0)).tolist(),
                "mean_rmse_in_F_units": float(np.sqrt(np.mean((values-reconstructed)**2))),
                "outer_bin_fraction_per_feature": np.mean((chosen == 0) | (chosen == 8), axis=0).tolist(),
                "lower_outer_bin_fraction_per_feature": np.mean(chosen == 0, axis=0).tolist(),
                "upper_outer_bin_fraction_per_feature": np.mean(chosen == 8, axis=0).tolist(),
            }
    return answer


def prepare_case(rawdict, case, outdir):
    """Fit a frozen case transform and return CPU NumPy arrays and sample indices.

    Transfer cases train/validate on the named source hemisphere and test only
    on the opposite hemisphere. Quantizer/representative fitting uses all six-
    hourly source training observations, not only the daily forecast origins.
    outdir is the exact case directory (no additional case suffix is appended).
    """
    if case not in ("pooled", "nh_to_sh", "sh_to_nh"):
        raise ValueError(f"Unknown case: {case}")
    raw = np.asarray(rawdict["raw"])
    times = np.asarray(rawdict["timestamps"], dtype="datetime64[ns]")
    lat, lon = np.asarray(rawdict["lat"]), np.asarray(rawdict["lon"])
    nodes = np.asarray(rawdict["node_ids"])
    if raw.shape != (len(times), 12, 6) or len(lat) != 12 or not np.array_equal(nodes, NODE_IDS):
        raise ValueError("Unexpected raw shape or region order")
    if not np.isfinite(raw).all() or not np.all(np.diff(times) == np.timedelta64(6, "h")):
        raise ValueError("Input values/time continuity failed")
    if times[0] != np.datetime64("1979-01-01T00", "ns"):
        raise ValueError("Expected full training series starting 1979-01-01")
    training = times < TRAIN_END
    if not training.any():
        raise ValueError("Empty training period")
    ntrain = int(training.sum())
    if case == "pooled": source = target = np.arange(12)
    elif case == "nh_to_sh": source, target = np.flatnonzero(lat > 0), np.flatnonzero(lat < 0)
    else: source, target = np.flatnonzero(lat < 0), np.flatnonzero(lat > 0)
    _, local_month, bands, band_id = _calendar(times, lat)
    if len(bands) != 3 or len(source) not in (6, 12):
        raise ValueError("Expected three absolute-latitude bands and nonempty source hemisphere")
    source_training = raw[:ntrain, source].astype(np.float64)
    mean = source_training.mean(axis=(0, 1))
    std_before = source_training.std(axis=(0, 1), ddof=0)
    std = np.where(std_before < 1e-8, 1., std_before)
    climatology = np.empty((3, 12, 6), dtype=np.float64)
    climatology_counts = np.empty((3, 12), dtype=np.int64)
    for band in range(3):
        local_regions = source[band_id[source] == band]
        for month in range(12):
            keep = local_month[:ntrain, local_regions] == month
            subset = raw[:ntrain, local_regions][keep]
            if not len(subset):
                raise ValueError("A source-training seasonal/latitude stratum is empty")
            climatology[band, month] = subset.mean(axis=0, dtype=np.float64)
            climatology_counts[band, month] = len(subset)
    delta_training = source_training[1:] - source_training[:-1]
    delta_std_before = delta_training.std(axis=(0, 1), ddof=0)
    delta_std = np.where(delta_std_before < 1e-8, 1., delta_std_before)
    del source_training, delta_training
    F = np.empty((len(times), 12, 18), dtype=np.float32)
    for j in range(6):
        values = raw[:, :, j].astype(np.float64)
        F[:, :, j] = (values-mean[j])/std[j]
        climate = climatology[band_id[None, :], local_month, j]
        F[:, :, j+6] = (values-climate)/std[j]
        F[0, :, j+12] = 0.
        F[1:, :, j+12] = np.diff(values, axis=0)/delta_std[j]
    boundaries_q = np.empty((18, 8), dtype=np.float64)
    boundaries_s = np.tile(ANOMALY_EDGES, (18, 1))
    boundaries_s[:6] = (PHYSICAL_EDGES-mean[:, None])/std[:, None]
    Q, S = np.empty(F.shape, dtype=np.int64), np.empty(F.shape, dtype=np.int64)
    centers_q, centers_s = np.empty((18,9), dtype=np.float64), np.empty((18,9), dtype=np.float64)
    counts_q, counts_s = np.empty((18,9), dtype=np.int64), np.empty((18,9), dtype=np.int64)
    for j in range(18):
        first = 1 if j >= 12 else 0
        fitting = F[first:ntrain, source, j].reshape(-1)
        boundaries_q[j] = np.quantile(fitting, np.arange(1,9)/9, method="linear")
        Q[:, :, j] = np.searchsorted(boundaries_q[j], F[:, :, j], side="right")
        S[:, :, j] = np.searchsorted(boundaries_s[j], F[:, :, j], side="right")
        centers_q[j], counts_q[j] = _centers(fitting, Q[first:ntrain, source, j].reshape(-1), boundaries_q[j])
        centers_s[j], counts_s[j] = _centers(fitting, S[first:ntrain, source, j].reshape(-1), boundaries_s[j])
    metadata = _metadata(times, lat, lon)
    labels = np.full((len(times), 12, 3), -1, dtype=np.int64)
    cloud = raw[:, :, 5].astype(np.float64)
    for j, step in enumerate(HORIZONS):
        change = cloud[step:] - cloud[:-step]
        labels[:-step, :, j] = np.where(change < -.05, 0, np.where(change > .05, 2, 1))
    indices, excluded = _indices(times, lat, source, target)
    test_times = times[indices["test"][:, 0]]
    early = test_times < EXTENSION_START
    late = test_times >= EXTENSION_START + np.timedelta64(48, "h")
    if not np.all(early | late):
        raise AssertionError("An unclassified source-transition test origin survived")
    for split, ids in indices.items():
        if len(ids) and (labels[ids[:, 0], ids[:, 1]] < 0).any():
            raise AssertionError(f"Unresolved targets in {split}")
    diagnostic_indices = dict(indices, test_early=indices["test"][early], test_late=indices["test"][late])
    diagnostics = _diagnose(diagnostic_indices, F, Q, S, centers_q, centers_s, labels)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    transform_path = destination/"transform.npz"
    np.savez_compressed(transform_path, abs_mean=mean, abs_std=std, abs_std_before_floor=std_before,
                        climatology=climatology, climatology_counts=climatology_counts,
                        absolute_latitude_bands=bands, delta_std=delta_std,
                        delta_std_before_floor=delta_std_before, boundaries_q=boundaries_q,
                        boundaries_s=boundaries_s, centers_q=centers_q, centers_s=centers_s,
                        training_bin_counts_q=counts_q, training_bin_counts_s=counts_s,
                        source_local_region_indices=source, target_local_region_indices=target,
                        node_ids=nodes, variable_names=np.array(VARIABLES), variable_units=np.array(UNITS),
                        feature_names=np.array(FEATURES), horizons_steps=HORIZONS,
                        physical_absolute_bin_boundaries=PHYSICAL_EDGES)
    manifest = {
        "status":"prepared", "case":case, "created_utc":datetime.now(timezone.utc).isoformat(),
        "data_module_sha256":_sha(__file__), "source_hashes":rawdict.get("source_hashes", {}),
        "source_notes":rawdict.get("source_notes", []), "transform_sha256":_sha(transform_path),
        "variables":list(VARIABLES), "units":list(UNITS), "feature_names":list(FEATURES),
        "shape":{"raw":list(raw.shape), "F":list(F.shape), "Q":list(Q.shape), "S":list(S.shape), "meta":list(metadata.shape), "labels":list(labels.shape)},
        "dtypes":{"raw":str(raw.dtype), "F":str(F.dtype), "Q":str(Q.dtype), "S":str(S.dtype), "meta":str(metadata.dtype), "labels":str(labels.dtype)},
        "fit_period":[str(times[0]),str(times[ntrain-1])], "fit_time_count":ntrain,
        "source_node_ids":nodes[source].tolist(), "target_node_ids":nodes[target].tolist(),
        "node_ids":nodes.tolist(), "lat":lat.tolist(), "lon":lon.tolist(),
        "split_rules":{"train":"origin and t+4 < 2015-01-01", "val":"origin >= 2015-01-01 and t+4 < 2019-01-01", "test":"origin >= 2019-01-01 and all targets exist; target hemisphere only for transfer", "validation_hemisphere":"source hemisphere only for transfer"},
        "fitting":"All 6-hourly source training values; delta std/quantiles/centers exclude t=0. No validation or destination-hemisphere values used in parameter fitting.",
        "anomaly":"Source-training raw monthly means pooled over the two source regions in each absolute-latitude band (four in pooled). SH season-aligned month=(calendar_month_zero_based+6)%12; NH unshifted; subtract mean then divide by source pooled raw-variable std.",
        "metadata":"sin/cos geographic latitude, sin/cos longitude, sin/cos elapsed-calendar-year fraction (leap-aware) with +0.5 phase in SH. Season-aligned month does not rename the actual calendar month.",
        "delta":"Six-hour raw difference divided by source-training population std of differences, without subtracting a difference mean; F[0,:,12:]=0 is never a forecast token.",
        "history":"8 tokens t-7..t; the first difference needs t-8 (48 hours before origin). Forecast origins are 00 UTC. Entire t-8..t+4 remains within one acquisition segment.",
        "source_boundaries":[str(VAL_END),str(EXTENSION_START)], "excluded_and_retained":excluded,
        "label_definition":{"horizon_steps":HORIZONS.tolist(),"horizon_hours":[6,12,24],"down":0,"neutral":1,"up":2,"down_threshold":"raw tcc change < -0.05", "up_threshold":"raw tcc change > 0.05", "otherwise":"neutral, inclusive at +/-0.05", "unavailable_array_tail":-1},
        "quantization":{"Q":"Eight linear empirical quantiles 1/9..8/9 from source training F; tied boundaries retained.","S":"Fixed raw physical boundaries converted to abs F; fixed standardized boundaries for anom/delta.","intervals":"np.searchsorted side=right: an exact boundary enters the upper bin; codes 0..8; outer bins unbounded.","representatives":"Source-training bin means; empty internal bin midpoint; empty tail finite half-spacing extrapolation, median positive spacing or 1 if needed.","no_clipping":True,"empty_bins_q":int((counts_q==0).sum()),"empty_bins_s":int((counts_s==0).sum()),"tied_q_boundaries_per_feature":np.sum(np.diff(boundaries_q,axis=1)==0,axis=1).tolist()},
        "diagnostic_scope":"Reconstruction RMSE and outer-bin frequencies evaluated at retained forecast origins in F units; these are quantization diagnostics, not forecast skill. Class-count rows follow horizons 6/12/24h; columns down/neutral/up.",
        "diagnostics":diagnostics,
        "test_subsets":{"early":"origin < 2023-01-11; WB2 source", "late":"origin >= 2023-01-13; at least 48-hour context wholly CDS extension", "early_samples":int(early.sum()),"late_samples":int(late.sum())},
        "limitations":["This is a regional token prototype, not a gridded weather forecast or causal identification experiment.","Transfer learns normalization and climatology from the source hemisphere only; destination anomalies are relative to source seasonal expectations.","Cross-source equivalence is unestablished, so early/late test results must remain separated.","Coarse pressure-level inputs are unmasked below terrain; no above-ground validity claim.","Physical region-first normalization differs from older grid-standardize-then-average causal input preprocessing."],
    }
    _json(destination/"case_manifest.json",manifest)
    examples = []
    for split, ids in indices.items():
        for selection in np.unique(np.linspace(0,len(ids)-1,min(3,len(ids)),dtype=int)) if len(ids) else []:
            t,r=map(int,ids[selection])
            examples.append({"split":split,"time_index":t,"timestamp":str(times[t]),"local_region_index":r,"node_id":int(nodes[r]),
                             "raw":raw[t,r].astype(float).tolist(),"F":F[t,r].astype(float).tolist(),"Q":Q[t,r].tolist(),"S":S[t,r].tolist(),
                             "labels_6_12_24h":labels[t,r].tolist(),"text":" ".join(f"{name}={value:.6g} {unit}" for name,value,unit in zip(VARIABLES,raw[t,r],UNITS))})
    _json(destination/"examples.json",examples)
    return {"F":F,"Q":Q,"S":S,"centers_s":centers_s,"centers_q":centers_q,"meta":metadata,"labels":labels,
            "indices":indices,"train_idx":indices["train"],"val_idx":indices["val"],"test_idx":indices["test"],
            "test_early_mask":early,"test_late_mask":late,"test_early_idx":indices["test"][early],"test_late_idx":indices["test"][late],
            "source_regions":source,"target_regions":target,"timestamps":times,"lat":lat,"lon":lon,"node_ids":nodes,
            "variable_names":np.array(VARIABLES),"feature_names":np.array(FEATURES),"manifest":manifest}
