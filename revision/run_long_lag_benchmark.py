"""Finite linear benchmark for source-window truncation, not FDR calibration."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import psutil
from revision.run_lag_window_sensitivity import fit_joint, sha, save, now

BASE=Path(__file__).resolve().parents[2]


def simulate(true_lag, repetition, seed=2026093019, ntrain=4096, ntest=2048, burn=1024):
    # Identical innovations across lag scenarios in the same repetition.
    rng=np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(repetition,)))
    innovations=rng.normal(size=(burn+ntrain+ntest,2))
    x=np.zeros(len(innovations));y=np.zeros_like(x)
    for t in range(12,len(x)):
        x[t]=.75*x[t-1]+innovations[t,0]
        y[t]=.6*y[t-1]+.25*x[t-true_lag]+innovations[t,1]
    data=np.column_stack((x[burn:],y[burn:]))
    mean=data[:ntrain].mean(0);scale=data[:ntrain].std(0)
    return (data-mean)/scale, float(.25*scale[0]/scale[1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'revision_outputs/long_lag_benchmark')
    parser.add_argument('--repetitions',type=int,default=100)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'design.json').exists():raise FileExistsError('Choose a fresh output directory')
    files=[Path(__file__),Path(__file__).with_name('run_lag_window_sensitivity.py'),Path(__file__).with_name('inference.py')]
    plan=dict(frozen_utc=now(),reviewer_ids=['R2-4','R2-5','R3-2'],role='Post-diagnostic finite linear long-lag benchmark',
        repetitions=args.repetitions,seed=2026093019,true_source_lags=[2,6,10],source_windows=[3,6,12],target_histories=[3,12],
        n_training_calendar=4096,n_evaluation=2048,burn=1024,common_response_start=12,
        dgp='x_t=.75*x_(t-1)+e_x; y_t=.6*y_(t-1)+.25*x_(t-d)+e_y; independent N(0,1) innovations',
        paired_scenarios='Each repetition reuses innovations across true lags and fitted models',
        standardization='Fit both means/scales using first4096 post-burn observations only',
        lag_recovery='Largest absolute jointly fitted training coefficient; report whether true lag lies in search window separately',
        uncertainty='Wilson95 Monte Carlo intervals for peak recovery across independent repetitions; no edge significance/FDR claims',
        performance='All models training-frozen; source-block MSE increment and window extensions on identical2048 evaluation responses',
        limitations=['Only this linear two-variable DGP and effect size, not ERA5 accuracy or broad method superiority.',
                    'Missing true lag creates model misspecification; out-of-window peaks are not direct-causal recovery.',
                    'Lag maxima and coefficient centroids are descriptive, not transport times.',
                    'No selection of source window by best evaluation outcome.'],
        code_sha256={p.name:sha(p) for p in files})
    save(out/'design.json',plan);snap=out/'code_snapshot';snap.mkdir()
    for p in files:shutil.copy2(p,snap/p.name)
    start=time.perf_counter();rows=[];coef=[];extension=[]
    t=np.arange(12,6144);train=t<4096;evaluate=~train
    for repetition in range(args.repetitions):
        for true_lag in (2,6,10):
            data,true_standard_beta=simulate(true_lag,repetition)
            target=data[t,1];source=np.column_stack([data[t-k,0] for k in range(1,13)])
            for history in (3,12):
                base=np.column_stack([np.ones(len(t))]+[data[t-k,1] for k in range(1,history+1)])
                predictions={}
                for window in (3,6,12):
                    fitted=fit_joint(target,base,source[:,:window],train,inference=False)
                    beta=fitted['beta'];predictions[window]=fitted['prediction']
                    peak=int(np.argmax(np.abs(beta)))+1
                    er=target[evaluate]-fitted['no_source_prediction'][evaluate]
                    ef=target[evaluate]-fitted['prediction'][evaluate]
                    gain=float(np.mean(er**2-ef**2))
                    common=dict(repetition=repetition,true_lag=true_lag,target_history=history,max_source_lag=window)
                    rows.append({**common,'true_lag_in_window':true_lag<=window,'peak_training_lag':peak,
                        'peak_matches_truth':peak==true_lag,'training_n':int(train.sum()),'evaluation_n':int(evaluate.sum()),
                        'mse_gain':gain,'delta_r2':gain/float(np.var(target[evaluate])),
                        'true_standardized_beta':true_standard_beta,
                        'estimated_true_lag_beta':float(beta[true_lag-1]) if true_lag<=window else None,
                        'absolute_beta_centroid':float(np.arange(1,window+1)@np.abs(beta)/np.abs(beta).sum())})
                    coef += [{**common,'source_lag':k+1,'beta':float(b)} for k,b in enumerate(beta)]
                for shorter,longer in ((3,6),(6,12),(3,12)):
                    gain=float(np.mean((target[evaluate]-predictions[shorter][evaluate])**2-(target[evaluate]-predictions[longer][evaluate])**2))
                    extension.append(dict(repetition=repetition,true_lag=true_lag,target_history=history,
                        shorter_window=shorter,longer_window=longer,mse_gain= gain,delta_r2=gain/float(np.var(target[evaluate]))))
        if (repetition+1)%10==0:
            print(f'Long-lag paired repetition {repetition+1}/{args.repetitions}',flush=True)
            save(out/'status.json',dict(status='running',repetitions_completed=repetition+1))
    frame=pd.DataFrame(rows);frame.to_csv(out/'repetition_results.csv',index=False)
    pd.DataFrame(coef).to_csv(out/'training_coefficients.csv',index=False)
    ext=pd.DataFrame(extension);ext.to_csv(out/'window_extensions.csv',index=False)
    summary=frame.groupby(['true_lag','target_history','max_source_lag'],as_index=False).agg(
        repetitions=('repetition','size'),true_lag_in_window=('true_lag_in_window','all'),peak_correct=('peak_matches_truth','sum'),
        positive_prediction_gain=('mse_gain',lambda x:int((x>0).sum())),median_delta_r2=('delta_r2','median'),
        median_coefficient_centroid=('absolute_beta_centroid','median'))
    n=summary.repetitions.to_numpy();phat=summary.peak_correct.to_numpy()/n;z=1.959963984540054
    center=(phat+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
    summary['peak_recovery_fraction']=phat;summary['wilson95_low']=center-half;summary['wilson95_high']=center+half
    summary.to_csv(out/'summary.csv',index=False)
    ext.groupby(['true_lag','target_history','shorter_window','longer_window'],as_index=False).agg(
        repetitions=('repetition','size'),longer_improves=('mse_gain',lambda x:int((x>0).sum())),
        median_delta_r2=('delta_r2','median')).to_csv(out/'extension_summary.csv',index=False)
    manifest=dict(status='completed',completed_utc=now(),elapsed_seconds=time.perf_counter()-start,
        peak_working_set_bytes=int(getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)),
        independent_repetitions=args.repetitions,paired_lag_scenarios=3,model_rows=len(frame),coefficient_rows=len(coef),
        design_sha256=sha(out/'design.json'),outputs_sha256={p.name:sha(p) for p in out.glob('*.csv')})
    save(out/'manifest.json',manifest);save(out/'status.json',manifest);print(manifest,flush=True)


if __name__=='__main__':main()
