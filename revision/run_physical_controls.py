"""Frozen-sample predictive control sensitivity, not causal identification.

The new physical fields are standardized using 1979--2018 only. Existing
training-selected pairs and paths are reused without selecting on new outcomes.
Future-source fits are reverse-time diagnostics and never operational forecasts.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil
from scipy import linalg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.inference import _finish, score_autocovariances
from revision.prepare_inputs import sha256


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def regress_basis(base, values):
    """Pivoted rank-aware OLS; explicit coefficients reproduce the same QR span."""
    q, r, piv = linalg.qr(base, mode='economic', pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    tol = max(base.shape) * np.finfo(float).eps * diagonal.max()
    rank = int(np.count_nonzero(diagonal > tol))
    q = q[:, :rank]
    coefficients = np.zeros((base.shape[1], values.shape[1]))
    coefficients[piv[:rank]] = linalg.solve_triangular(r[:rank, :rank], q.T @ values, check_finite=False)
    residual = values - q @ (q.T @ values)
    return coefficients, residual, rank, float(np.linalg.cond(r[:rank, :rank]))


def analyze_sources(y, controls, sources, training, evaluation_masks, refit_mask=None):
    """Each source is added separately to the common restricted controls.

The source columns are NOT jointly included. Evaluation coefficients are
separate retrospective refits. Losses always use frozen training coefficients.
"""
    y, controls, sources = map(np.asarray, (y, controls, sources))
    if sources.ndim == 1:
        sources = sources[:, None]
    if not all(np.isfinite(a).all() for a in (y, controls, sources)):
        raise ValueError('Nonfinite input; no implicit row deletion permitted')
    base = np.column_stack((np.ones(len(y)), controls))
    joined = np.column_stack((y, sources))
    coefficients, residual, rank, condition = regress_basis(base[training], joined[training])
    fits = {'training': _finish(y[training], residual[:, 0], residual[:, 1:], training, rank, (64,))}
    if refit_mask is not None:
        _, re, erank, econdition = regress_basis(base[refit_mask], joined[refit_mask])
        fits['evaluation_refit'] = _finish(y[refit_mask], re[:, 0], re[:, 1:], refit_mask, erank, (64,))
        for row in fits['evaluation_refit']:
            row['control_condition_number'] = econdition
            row['control_columns_with_intercept'] = base.shape[1]
    for row in fits['training']:
        row['control_condition_number'] = condition
        row['control_columns_with_intercept'] = base.shape[1]
    beta = np.array([row['effect'] for row in fits['training']])
    restricted = base @ coefficients[:, 0]
    full = restricted[:, None] + (sources - base @ coefficients[:, 1:]) * beta
    forecasts = {}
    for period, mask in evaluation_masks.items():
        positions = np.flatnonzero(mask)
        if len(positions) < 2 or not np.all(np.diff(positions) == 1):
            raise ValueError('Evaluation must have contiguous actual-calendar rows')
        er = y[mask] - restricted[mask]
        ef = y[mask, None] - full[mask]
        gain = er[:, None] ** 2 - ef ** 2
        mean = gain.mean(axis=0)
        covariance = score_autocovariances(gain - mean, 64)
        h = np.arange(1, len(covariance))
        meat = covariance[0] + 2 * np.sum((1 - h[:, None] / 65) * covariance[h], axis=0)
        se = np.sqrt(np.maximum(meat, 0) * len(gain) / (len(gain) - 1)) / len(gain)
        train_denominator = np.mean((y[mask] - y[training].mean()) ** 2)
        test_denominator = np.var(y[mask])
        forecasts[period] = [dict(valid=bool(fits['training'][k]['valid']), status=fits['training'][k]['status'], training_n=int(training.sum()), evaluation_n=int(mask.sum()),
            frozen_source_beta=float(beta[k]), mse_restricted=float(np.mean(er ** 2)),
            mse_full=float(np.mean(ef[:, k] ** 2)), mse_gain=float(mean[k]),
            mse_gain_se_hac64=float(se[k]), mse_gain_ci95_low=float(mean[k] - 1.959963984540054 * se[k]),
            mse_gain_ci95_high=float(mean[k] + 1.959963984540054 * se[k]),
            delta_r2_train_centered=float(mean[k] / train_denominator),
            delta_r2_test_centered=float(mean[k] / test_denominator)) for k in range(sources.shape[1])]
    return fits, forecasts


def physical_history(physical, times, regions, lags):
    regions = sorted(set(int(r) for r in regions))
    return np.column_stack([physical[times - lag, region, field]
        for region in regions for field in range(physical.shape[2]) for lag in lags])


def evaluation_periods(timestamps, earliest=None, latest=None):
    evaluation = timestamps >= np.datetime64('2019-01-01')
    earliest = timestamps if earliest is None else earliest
    latest = timestamps if latest is None else latest
    return {'evaluation_all': evaluation,
            'evaluation_wb2': evaluation & (latest < np.datetime64('2023-01-11')),
            'evaluation_cds': earliest >= np.datetime64('2023-01-11')}


def append_results(fits, predictions, metadata, descriptors, fit_rows, prediction_rows):
    for period, records in fits.items():
        for descriptor, row in zip(descriptors, records):
            fit_rows.append({**metadata, **descriptor, 'period': period, **row})
    for period, records in predictions.items():
        for descriptor, row in zip(descriptors, records):
            prediction_rows.append({**metadata, **descriptor, 'period': period, **row})


def run_directionality(data, physical, timestamps, names, pairs, output):
    index = {n: i for i, n in enumerate(names)}
    # Common support for all forward and future sources, and both physical lags.
    t = np.arange(6, len(data) - 3)
    training = timestamps[t + 3] < np.datetime64('2019-01-01')
    masks = evaluation_periods(timestamps[t], timestamps[t-6], timestamps[t+3])
    fit_rows, prediction_rows = [], []
    for number, row in enumerate(pairs.itertuples(), 1):
        for orientation in ('original', 'reversed'):
            i, j, sv, tv = int(row.source_region), int(row.target_region), row.source_var, row.target_var
            if orientation == 'reversed':
                i, j, sv, tv = j, i, tv, sv
            y = data[t, j, index[tv]]
            own = np.column_stack([data[t - lag, j, index[tv]] for lag in (1, 2, 3)])
            signed_lags = [1, 2, 3, -1, -2, -3]
            sources = np.column_stack([data[t - lag, i, index[sv]] for lag in signed_lags])
            descriptors = [dict(signed_source_lag=lag,
                source_timing='past' if lag > 0 else 'future_diagnostic') for lag in signed_lags]
            for model, lags in [('own_history', ()), ('physical_response_history', (1, 2, 3)),
                                ('physical_presource_history', (4, 5, 6))]:
                controls = np.column_stack((own, physical_history(physical, t, [i, j], lags))) if lags else own
                fits, predictions = analyze_sources(y, controls, sources, training, masks, masks['evaluation_all'])
                metadata = dict(pair_id=int(row.pair_id), stratum=row.stratum, orientation=orientation,
                    source_region=i, target_region=j, source_var=sv, target_var=tv,
                    original_edge_type=row.edge_type, model=model)
                append_results(fits, predictions, metadata, descriptors, fit_rows, prediction_rows)
        save(output / 'status.json', dict(status='running_directionality', pairs_completed=number,
             total_pairs=len(pairs), updated_utc=utcnow()))
        print(f'directionality {number}/{len(pairs)}', flush=True)
    pd.DataFrame(fit_rows).to_csv(output / 'direction_coefficients.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(output / 'direction_frozen_losses.csv', index=False)


def run_paths(data, physical, timestamps, names, paths, output):
    index = {n: i for i, n in enumerate(names)}
    # Shared row start for every path: max(l1+l2)+3 = 9.
    t = np.arange(9, len(data))
    training = timestamps[t] < np.datetime64('2019-01-01')
    masks = evaluation_periods(timestamps[t], timestamps[t-9], timestamps[t])
    fit_rows, prediction_rows = [], []
    for number, row in enumerate(paths.itertuples(), 1):
        i, h, j = int(row.i), int(row.h), int(row.j)
        lag1, lag2 = int(row.lag1), int(row.lag2)
        st, mt = t - lag1 - lag2, t - lag2
        w, humidity, cloud = data[st, i, index['wind']], data[mt, h, index['humidity']], data[t, j, index['cloud_cover']]
        cm = np.column_stack([data[mt - lag, h, index[name]] for name in ('humidity', 'temperature') for lag in (1, 2, 3)])
        cy = np.column_stack([data[t - lag, j, index[name]] for name in ('cloud_cover', 'temperature') for lag in (1, 2, 3)])
        for model in ('baseline', 'physical_before_path_source'):
            mediator_controls, target_controls = cm, cy
            if model != 'baseline':
                extra = physical_history(physical, st, [i, h, j], (1, 2, 3))
                mediator_controls, target_controls = np.column_stack((cm, extra)), np.column_stack((cy, extra))
            cases = [
                ('a_wind_to_humidity', humidity, mediator_controls, w),
                ('total_wind_to_cloud', cloud, target_controls, w),
                ('b_humidity_to_cloud_given_wind', cloud, np.column_stack((target_controls, w)), humidity),
                ('direct_wind_to_cloud_given_humidity', cloud, np.column_stack((target_controls, humidity)), w),
            ]
            for stage, y, controls, source in cases:
                stage_masks = masks
                if stage == 'a_wind_to_humidity':
                    # Avoid assigning a 2018 mediator response to a 2019 refit.
                    stage_masks = {key: value & (timestamps[mt] >= np.datetime64('2019-01-01'))
                                   for key, value in masks.items()}
                fits, predictions = analyze_sources(y, controls, source, training, stage_masks,
                                                    stage_masks['evaluation_all'])
                metadata = dict(path_id=int(row.path_id), stratum=row.stratum, mediator_location=row.mediator_location,
                    i=i, h=h, j=j, lag1=lag1, lag2=lag2, model=model, stage=stage)
                append_results(fits, predictions, metadata, [{}], fit_rows, prediction_rows)
        save(output / 'status.json', dict(status='running_paths', paths_completed=number,
             total_paths=len(paths), updated_utc=utcnow()))
        if number % 5 == 0:
            print(f'paths {number}/{len(paths)}', flush=True)
    pd.DataFrame(fit_rows).to_csv(output / 'path_coefficients.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(output / 'path_frozen_losses.csv', index=False)


def summarize(output):
    for prefix in ('direction', 'path'):
        table = pd.read_csv(output / f'{prefix}_frozen_losses.csv')
        group = ['model', 'period'] + (['orientation', 'original_edge_type', 'source_timing'] if prefix == 'direction' else ['stage', 'stratum'])
        summary = table.groupby(group, as_index=False).agg(models=('mse_gain', 'size'),
            valid_models=('valid', 'sum'),
            positive_incremental_gain=('mse_gain', lambda v: int((v > 0).sum())),
            median_mse_gain=('mse_gain', 'median'), median_delta_r2=('delta_r2_test_centered', 'median'))
        summary.to_csv(output / f'{prefix}_summary.csv', index=False)
        baseline_name = 'own_history' if prefix == 'direction' else 'baseline'
        keys = ['period'] + (['pair_id', 'orientation', 'signed_source_lag'] if prefix == 'direction' else ['path_id', 'stage'])
        baseline = table[table.model == baseline_name][keys + ['mse_full', 'mse_restricted', 'mse_gain', 'delta_r2_test_centered']]
        comparisons = table[table.model != baseline_name].merge(baseline, on=keys, validate='many_to_one', suffixes=('', '_baseline'))
        comparisons['whole_model_mse_improvement'] = comparisons.mse_full_baseline - comparisons.mse_full
        comparisons['source_increment_change'] = comparisons.mse_gain - comparisons.mse_gain_baseline
        comparisons.to_csv(output / f'{prefix}_control_comparisons.csv', index=False)
        coefficients = pd.read_csv(output / f'{prefix}_coefficients.csv')
        base_coef = coefficients[coefficients.model == baseline_name][keys + ['effect']]
        changes = coefficients[coefficients.model != baseline_name].merge(base_coef, on=keys, validate='many_to_one', suffixes=('', '_baseline'))
        changes['same_sign_as_baseline'] = np.sign(changes.effect) == np.sign(changes.effect_baseline)
        changes['absolute_coefficient_change'] = np.abs(changes.effect) - np.abs(changes.effect_baseline)
        changes.to_csv(output / f'{prefix}_coefficient_comparisons.csv', index=False)
    paths = pd.read_csv(output / 'path_coefficients.csv')
    products = paths.pivot(index=['path_id', 'model', 'period'], columns='stage', values='effect').reset_index()
    products['descriptive_a_times_b'] = products.a_wind_to_humidity * products.b_humidity_to_cloud_given_wind
    products.to_csv(output / 'path_descriptive_products.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, default=ROOT.parent / 'revision_outputs/inputs/region_trainfit.npz')
    parser.add_argument('--physical', type=Path, default=ROOT.parent / 'revision_outputs/physical_controls_inputs/region_controls_trainfit.npz')
    parser.add_argument('--selections', type=Path, default=ROOT.parent / 'revision_outputs/path_diagnostics')
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'revision_outputs/physical_controls_experiments')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'design.json').exists():
        raise ValueError('Use a fresh output directory; never overwrite an executed design')
    paths_file, pairs_file = args.selections / 'frozen_paths.csv', args.selections / 'frozen_joint_lag_pairs.csv'
    pairs, paths = pd.read_csv(pairs_file), pd.read_csv(paths_file)
    if args.smoke:
        pairs, paths = pairs.head(1), paths.head(1)
    with np.load(args.primary, allow_pickle=False) as a, np.load(args.physical, allow_pickle=False) as b:
        data, physical, timestamps = a['data'], b['data'], a['timestamps']
        names, physical_names = a['variable_names'].tolist(), b['variable_names'].tolist()
        for key in ('timestamps', 'lat', 'lon'):
            np.testing.assert_array_equal(a[key], b[key])
    assert physical_names == ['omega_500', 'omega_700', 'geopotential_500', 'temperature_700', 'surface_pressure', 'mean_sea_level_pressure']
    assert len(timestamps) == 68668 and data.shape[:2] == physical.shape[:2] == (68668, 66)
    assert np.all(np.diff(timestamps) == np.timedelta64(6, 'h'))
    inputs = {str(p): sha256(p) for p in (args.primary, args.physical, paths_file, pairs_file)}
    design = dict(frozen_utc=utcnow(), analysis='post-diagnostic physical-driver sensitivity', smoke=bool(args.smoke),
        pair_count=len(pairs), path_count=len(paths), training='1979-2018', evaluation='2019-2025; split at 2023-01-11 source change',
        physical_names=physical_names, inputs_sha256=inputs,
        code_sha256={name: sha256(ROOT / 'revision' / name) for name in ('run_physical_controls.py', 'inference.py')},
        pair_selection='Existing 36 training-selected pairs; retain original and reciprocal orientation and signed lags +1,+2,+3,-1,-2,-3; no reselection.',
        pair_response_support='t=6 through N-4 inclusive; training requires t+3 before 2019 to exclude cross-split future sources; all models use same rows.',
        segmented_evaluation='WB2/CDS segment results exclude every design row whose potential inputs straddle 2023-01-11. Direction bounds t-6,t+3; path bounds t-9,t. Full-period results retain boundary rows.',
        pair_models={'own_history': 'target history1:3', 'physical_response_history': '+six physical fields in unique source/target regions at t-1:t-3; some may follow tested past source',
                     'physical_presource_history': '+same fields at t-4:t-6, strictly before every tested past source'},
        source_fits='Six source lags are tested separately, not jointly; future-source models are reverse-time diagnostics, not implementable forecasts.',
        path_selection='Reuse all existing60 paths, no outcome reselection and no new matched-alternative claim.',
        path_models='Original H/T and C/T autoregressive baselines; augmented six fields in unique i,h,j at1:3 steps BEFORE W[t-l1-l2]. Common target row start9.',
        path_stages='a W->H; total W->C; b H->C|W; direct W->C|H. Product a*b is descriptive, not identified mediation.',
        coefficient_periods='Training fits; separately refitted evaluation coefficients. First-stage 2018 mediator responses excluded from evaluation refit.',
        prediction='Coefficients always training-frozen; exact deletion and refit of added source, both train-centered and evaluation-centered deltaR2 denominators.',
        inference='Calendar Bartlett HAC64 exploratory intervals only; no calibrated FDR/causal-identification claim, no significance-based filtering.',
        limitations=['Previously examined evaluation period, not untouched prospective validation.', 'Regional mean pressure fields can include below-terrain/extrapolated levels; coarse surface-pressure diagnostics cannot repair native terrain masks.',
                     'Cross-source grid processing of existing late primary variables differs; source segments reported separately, no measured overlap bias.',
                     'A fixed training-selected subset does not establish whole-graph robustness or resolve omitted/unmeasured confounding.'])
    save(output / 'design.json', design)
    snapshot = output / 'code_snapshot'
    snapshot.mkdir()
    for name in design['code_sha256']:
        (snapshot / name).write_bytes((ROOT / 'revision' / name).read_bytes())
    start = time.perf_counter()
    try:
        run_directionality(data, physical, timestamps, names, pairs, output)
        run_paths(data, physical, timestamps, names, paths, output)
        summarize(output)
        manifest = dict(status='completed', completed_utc=utcnow(), elapsed_seconds=time.perf_counter()-start,
            peak_working_set_bytes=psutil.Process().memory_info().peak_wset,
            software={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'psutil')},
            outputs_sha256={p.name: sha256(p) for p in output.glob('*.csv')}, design_sha256=sha256(output / 'design.json'))
        save(output / 'manifest.json', manifest)
        save(output / 'status.json', manifest)
        print(json.dumps(manifest), flush=True)
    except Exception as exc:
        save(output / 'status.json', dict(status='failed_review_required', error_type=type(exc).__name__, error=str(exc), updated_utc=utcnow()))
        raise


if __name__ == '__main__':
    main()
