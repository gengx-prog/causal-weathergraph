"""Fixed-pair source-window sensitivity with joint lag deletion and frozen losses.

Retrospective descriptive diagnostics. No window or history length is selected
using evaluation outcomes, and HAC intervals are not claimed to be calibrated.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import psutil
from scipy import linalg, stats
from revision.inference import score_autocovariances

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
WINDOWS = (3, 6, 12)
MODELS = ('own3', 'own12', 'own3_physical_before12')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def qr_fit(x, y):
    q, r = linalg.qr(x, mode='economic', check_finite=False)
    singular = linalg.svdvals(r, check_finite=False)
    rank = int((singular > max(x.shape) * np.finfo(float).eps * singular[0]).sum())
    if rank != x.shape[1]:
        raise ValueError(f'Rank deficient design: {rank}/{x.shape[1]}; no silent deletion')
    beta = linalg.solve_triangular(r, q.T @ y, check_finite=False)
    invr = linalg.solve_triangular(r, np.eye(len(r)), check_finite=False)
    return beta, q, invr, float(singular[0] / singular[-1])


def fit_joint(y, base, source, mask, inference=True):
    """All source columns jointly enter; source deletion uses exact OLS refit.

    The stored dual design uses QR rather than inversion of X'X. The deletion
    identity is independently checked against explicit restricted least squares.
    """
    x = np.column_stack((base, source))
    y, mask = np.asarray(y), np.asarray(mask, bool)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite inputs; no implicit row deletion')
    beta, q, invr, condition = qr_fit(x[mask], y[mask])
    p, n = x.shape[1], int(mask.sum())
    source_start = base.shape[1]
    indices = np.arange(source_start, p)
    pseudoinverse = invr @ q.T
    dual = pseudoinverse[indices].T
    diagonal = np.sum(pseudoinverse[indices] ** 2, axis=1)
    # R^-1 R^-T[:,j] constructed from QR, never from the normal equations.
    deletion_coefficient = invr @ invr[indices].T / diagonal
    residualized_source = x @ deletion_coefficient
    predicted = x @ beta
    restricted_prediction = predicted[:, None] - residualized_source * beta[indices]
    restricted_beta, _, _, _ = qr_fit(base[mask], y[mask])
    out = {'beta': beta[indices], 'prediction': predicted,
           'deleted_prediction': restricted_prediction,
           'no_source_prediction': base @ restricted_beta,
           'n': n, 'rank': p, 'condition': condition,
           'source_residual_variance_fraction': 1 / (diagonal * np.sum((source[mask] - source[mask].mean(axis=0))**2, axis=0))}
    if inference:
        errors = y[mask] - x[mask] @ beta
        scores = np.zeros((len(mask), len(indices)))
        scores[mask] = dual * errors[:, None]
        ac = score_autocovariances(scores, 64)
        h = np.arange(1, len(ac))
        variance = (ac[0] + 2 * np.sum((1-h[:, None]/65) * ac[1:], axis=0)) * n/(n-p)
        se = np.sqrt(np.maximum(variance, 0))
        out.update(se_hac64=se, p_hac64=2*stats.norm.sf(np.abs(out['beta'])/se),
                   ci95_low=out['beta']-stats.norm.ppf(.975)*se,
                   ci95_high=out['beta']+stats.norm.ppf(.975)*se)
    return out


def loss_contrast(y, restricted, full, mask, train_mean):
    position = np.flatnonzero(mask)
    if len(position) < 2 or not np.all(np.diff(position) == 1):
        raise ValueError('Loss supports must be contiguous calendar intervals')
    er, ef = y[mask] - restricted[mask], y[mask] - full[mask]
    gain = er*er - ef*ef
    mean = float(gain.mean())
    ac = score_autocovariances(gain-mean, 64)[:, 0]
    h = np.arange(1, len(ac))
    variance_sum = (ac[0]+2*np.sum((1-h/65)*ac[1:]))*len(gain)/(len(gain)-1)
    se = float(np.sqrt(max(0, variance_sum)) / len(gain))
    return dict(evaluation_n=len(gain), mse_restricted=float(np.mean(er*er)), mse_full=float(np.mean(ef*ef)),
                mse_gain=mean, se_hac64=se, ci95_low=mean-stats.norm.ppf(.975)*se,
                ci95_high=mean+stats.norm.ppf(.975)*se,
                delta_r2_test_centered=mean/float(np.var(y[mask])),
                delta_r2_train_centered=mean/float(np.mean((y[mask]-train_mean)**2)))


def make_inputs(data, physical, names, pair, times, model):
    index = {v: i for i, v in enumerate(names)}
    i, j = int(pair.source_region), int(pair.target_region)
    y = data[times, j, index[pair.target_var]]
    history = 12 if model == 'own12' else 3
    controls = [np.ones(len(times))] + [data[times-k, j, index[pair.target_var]] for k in range(1, history+1)]
    if model == 'own3_physical_before12':
        controls += [physical[times-k, r, f] for r in sorted({i, j})
                     for f in range(6) for k in (13, 14, 15)]
    source = np.column_stack([data[times-k, i, index[pair.source_var]] for k in range(1, 13)])
    return y, np.column_stack(controls), source


def freeze(output, smoke=False):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'design.json').exists():
        raise FileExistsError('Frozen design exists; use a new output directory')
    inputs = [BASE/'revision_outputs/inputs/region_trainfit.npz',
              BASE/'revision_outputs/physical_controls_inputs/region_controls_trainfit.npz',
              BASE/'revision_outputs/path_diagnostics/frozen_joint_lag_pairs.csv']
    code = [Path(__file__), Path(__file__).with_name('inference.py')]
    plan = dict(frozen_utc=now(), smoke=smoke, reviewer_ids=['R2-4', 'R1-D5', 'R1-D6', 'R3-6'],
        analysis='post-diagnostic source search-window and target-history sensitivity',
        inputs_sha256={str(p):sha(p) for p in inputs}, code_sha256={p.name:sha(p) for p in code},
        pair_count=1 if smoke else 36, source_windows=list(WINDOWS), control_models=list(MODELS),
        source_joint=True, common_response_start=15, training='1979-2018', evaluation='2019-2025',
        physical_lags=[13,14,15], selection='Reuse frozen pairs; no new outcome-based selection',
        prediction='All coefficients train-frozen; each deleted lag refits all other coefficients; whole source-block versus no source',
        extended_window='MSE of shorter window minus MSE of longer window; same target history, same response samples',
        coefficient_fits='Training and separate retrospective evaluation fits; no p/q filtering',
        inference='Calendar Bartlett HAC64, normal reference, n/(n-rank) correction; exploratory and uncalibrated',
        uniform_lag_reference='(L+1)/2, descriptive reference if all tested lags receive equal weight; not a fitted null distribution',
        segmentation='Full evaluation; WB2 segment with target before2023-01-11; CDS segment with earliest input t-15 on/after2023-01-11',
        limitations=['No physical travel-time or mediation identification.',
                    'Three source windows were specified after earlier diagnostics; no best-window selection.',
                    'Fixed36pair subset, not whole-graph robustness.',
                    'Primary source processing differs across the acquisition boundary.',
                    'No finite-sample FDR calibration from these intervals.'])
    save(output/'design.json', plan)
    snap=output/'code_snapshot';snap.mkdir()
    for p in code:shutil.copy2(p,snap/p.name)
    return plan


def run_real(output, smoke=False):
    plan=freeze(output,smoke)
    start=time.perf_counter()
    with np.load(BASE/'revision_outputs/inputs/region_trainfit.npz',allow_pickle=False) as z:
        data=z['data']; timestamps=z['timestamps'].astype('datetime64[ns]'); names=z['variable_names'].tolist()
        lat,lon,node=z['lat'],z['lon'],z['node_ids']
    with np.load(BASE/'revision_outputs/physical_controls_inputs/region_controls_trainfit.npz',allow_pickle=False) as z:
        physical=z['data']
        for key,expected in [('timestamps',timestamps),('lat',lat),('lon',lon),('node_ids',node)]:
            assert np.array_equal(z[key],expected),key
    assert len(data)==68668 and physical.shape==(68668,66,6)
    assert np.all(np.diff(timestamps)==np.timedelta64(6,'h'))
    pairs=pd.read_csv(BASE/'revision_outputs/path_diagnostics/frozen_joint_lag_pairs.csv').head(plan['pair_count'])
    t=np.arange(15,len(data));tr=timestamps[t]<np.datetime64('2019-01-01');ev=~tr
    masks={'evaluation_all':ev,'evaluation_wb2':ev&(timestamps[t]<np.datetime64('2023-01-11')),
           'evaluation_cds':timestamps[t-15]>=np.datetime64('2023-01-11')}
    coefficients=[];losses=[];windows=[];centroids=[];checks=[]
    for number,pair in enumerate(pairs.itertuples(index=False),1):
        common={k:getattr(pair,k) for k in ['pair_id','source_region','target_region','source_var','target_var','edge_type','stratum']}
        for model in MODELS:
            y,base,sources=make_inputs(data,physical,names,pair,t,model)
            stored={}
            for window in WINDOWS:
                source=sources[:,:window]
                train=fit_joint(y,base,source,tr)
                evaluation=fit_joint(y,base,source,ev)
                meta={**common,'control_model':model,'max_source_lag':window}
                stored[window]=train['prediction']
                for period,fit in [('training',train),('evaluation_refit',evaluation)]:
                    for lag in range(1,window+1):
                        k=lag-1
                        coefficients.append({**meta,'period':period,'source_lag':lag,'valid':True,'n':fit['n'],
                            'rank':fit['rank'],'condition':fit['condition'],'beta':fit['beta'][k],
                            'se_hac64':fit['se_hac64'][k],'p_hac64_uncalibrated':fit['p_hac64'][k],
                            'ci95_low':fit['ci95_low'][k],'ci95_high':fit['ci95_high'][k],
                            'source_residual_variance_fraction':fit['source_residual_variance_fraction'][k]})
                    weight=np.abs(fit['beta']);lags=np.arange(1,window+1)
                    centroids.append({**meta,'period':period,'uniform_reference':(window+1)/2,
                        'absolute_coefficient_centroid':float(lags@weight/weight.sum()),
                        'largest_absolute_coefficient_lag':int(lags[weight.argmax()]),
                        'centroid_note':'Descriptive jointly fitted standardized coefficients, not a physical travel-time estimator'})
                for period,mask in masks.items():
                    for lag in range(window+1):
                        restricted=train['no_source_prediction'] if lag==0 else train['deleted_prediction'][:,lag-1]
                        losses.append({**meta,'period':period,'deleted_source':'all' if lag==0 else str(lag),
                            'training_n':int(tr.sum()),**loss_contrast(y,restricted,train['prediction'],mask,y[tr].mean())})
                if number in (1,12,24,36) and window==12:
                    x=np.column_stack((base,source))
                    # Separate SVD least squares, explicit refits, no deletion identity.
                    bf=np.linalg.lstsq(x[tr],y[tr],rcond=None)[0]
                    maxerr=float(np.max(np.abs((x@bf)-train['prediction'])))
                    for k in (0,5,11):
                        reduced=np.delete(x,base.shape[1]+k,axis=1)
                        br=np.linalg.lstsq(reduced[tr],y[tr],rcond=None)[0]
                        err=float(np.max(np.abs(reduced@br-train['deleted_prediction'][:,k])))
                        checks.append({**meta,'deleted_lag':k+1,'full_prediction_max_error':maxerr,'deleted_prediction_max_error':err})
                        if max(maxerr,err)>1e-7:raise ValueError('Explicit refit prediction disagreement')
            for shorter,longer in ((3,6),(6,12),(3,12)):
                for period,mask in masks.items():
                    windows.append({**common,'control_model':model,'shorter_window':shorter,'longer_window':longer,
                        'period':period,'training_n':int(tr.sum()),**loss_contrast(y,stored[shorter],stored[longer],mask,y[tr].mean())})
        save(output/'status.json',dict(status='running',pairs_completed=number,pairs_total=len(pairs),elapsed_seconds=time.perf_counter()-start))
        if number%4==0 or number==len(pairs):print(f'Lag-window pairs {number}/{len(pairs)}',flush=True)
    tables={'joint_coefficients':pd.DataFrame(coefficients),'frozen_deletion_losses':pd.DataFrame(losses),
            'extended_window_losses':pd.DataFrame(windows),'coefficient_centroids':pd.DataFrame(centroids),
            'independent_restricted_refit_checks':pd.DataFrame(checks)}
    for name,frame in tables.items():frame.to_csv(output/(name+'.csv'),index=False)
    summary=tables['frozen_deletion_losses'].query("deleted_source == 'all'").groupby(
        ['period','control_model','max_source_lag','edge_type'],as_index=False).agg(
        models=('pair_id','size'),positive_source_gain=('mse_gain',lambda s:int((s>0).sum())),
        median_delta_r2=('delta_r2_test_centered','median'))
    summary.to_csv(output/'source_block_summary.csv',index=False)
    extension=tables['extended_window_losses'].groupby(['period','control_model','shorter_window','longer_window'],as_index=False).agg(
        models=('pair_id','size'),longer_improves=('mse_gain',lambda s:int((s>0).sum())),median_delta_r2=('delta_r2_test_centered','median'))
    extension.to_csv(output/'window_extension_summary.csv',index=False)
    manifest=dict(status='completed',completed_utc=now(),elapsed_seconds=time.perf_counter()-start,
        peak_working_set_bytes=int(getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)),
        row_counts={k:len(v) for k,v in tables.items()},design_sha256=sha(output/'design.json'),
        software={'numpy':np.__version__,'pandas':pd.__version__},
        outputs_sha256={p.name:sha(p) for p in output.glob('*.csv')})
    save(output/'manifest.json',manifest);save(output/'status.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=BASE/'revision_outputs/lag_window_sensitivity')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    run_real(args.output,args.smoke)


if __name__=='__main__':main()
