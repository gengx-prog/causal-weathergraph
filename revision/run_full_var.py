"""Shared dense, unpenalised VAR(3) benchmark and frozen source ablations.

Every one of the 264 observed region-variable nodes enters all three lags.
The 132 humidity/cloud outcomes share one QR decomposition per period.
No sparsity penalty or hidden ridge is used. Inference remains a sensitivity
analysis because covariance calibration and causal identification are limited.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='4'
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import psutil
from scipy import linalg,stats
from threadpoolctl import threadpool_limits,threadpool_info

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from causal_weathergraph.candidate_edges import build_candidate_edges
from revision.inference import score_autocovariances,adjust_bh,_cluster_statistics
from revision.run_inference import sha256


KEY=['source_region','target_region','source_var','target_var','lag']


def shared_qr_fit(x,y,selected_columns):
    """Exact full-rank OLS plus inverse-Gram score directions, shared over Y."""
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    if y.ndim==1:
        y=y[:,None]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Full VAR requires finite complete design; missingness must be handled explicitly')
    n,p=x.shape
    if n<=p:
        raise ValueError(f'OLS not identified: n={n}, p={p}')
    q,r=linalg.qr(x,mode='economic',check_finite=False)
    singular=linalg.svdvals(r,check_finite=False)
    tolerance=max(x.shape)*np.finfo(float).eps*singular[0]
    rank=int(np.count_nonzero(singular>tolerance))
    diagnostics=dict(n_samples=n,n_parameters=p,n_predictors_excluding_intercept=p-1,rank=rank,
        rank_tolerance=float(tolerance),largest_singular_value=float(singular[0]),
        smallest_singular_value=float(singular[-1]),condition_number=float(singular[0]/singular[-1]),
        estimator='unpenalised full-rank QR OLS',regularization=None)
    if rank!=p:
        raise ValueError('Rank-deficient full VAR; no implicit ridge or pseudoinverse: '+json.dumps(diagnostics))
    coefficients=linalg.solve_triangular(r,q.T@y,check_finite=False)
    residual=y-x@coefficients
    inverse_r=linalg.solve_triangular(r,np.eye(p),check_finite=False)
    inverse_gram=inverse_r@inverse_r.T
    directions=q@inverse_r.T[:,selected_columns]
    diagnostics['max_abs_residual_normal_equation']=float(np.max(np.abs(x.T@residual)))
    diagnostics['relative_residual_orthogonality']=float(np.linalg.norm(q.T@residual)/max(np.linalg.norm(residual),1e-300))
    diagnostics['inverse_gram_diagonal_min']=float(np.min(np.diag(inverse_gram)))
    diagnostics['inverse_gram_diagonal_max']=float(np.max(np.diag(inverse_gram)))
    return dict(coefficients=coefficients,residual=residual,inverse_gram=inverse_gram,
                directions=directions,diagnostics=diagnostics)


def score_inference(scores,coefficients,n_parameters,calendar_offset=0,bandwidth=64,block_length=128):
    """Covariance for coefficient influence scores (already multiplied by bread)."""
    scores=np.asarray(scores,dtype=float)
    coefficients=np.asarray(coefficients,dtype=float)
    n=len(scores)
    acov=score_autocovariances(scores,bandwidth)
    h=np.arange(1,len(acov))
    meat=acov[0]+2*np.sum((1-h[:,None]/(bandwidth+1))*acov[h],axis=0)
    se=np.sqrt(np.maximum(meat*n/(n-n_parameters),0))
    p=2*stats.norm.sf(np.divide(np.abs(coefficients),se,out=np.full_like(se,np.inf),where=se>0))
    cluster=_cluster_statistics(scores,np.ones(n,dtype=bool),np.ones(scores.shape[1]),
        coefficients,n_parameters,(block_length,),calendar_offset)[block_length]
    return se,p,cluster


def frozen_ablation_gain(x_evaluation,y_evaluation,coefficients,inverse_gram,column,outcome):
    """Exact restricted training refit after removing ONE predictor coefficient."""
    full_prediction=x_evaluation@coefficients[:,outcome]
    removed_prediction=(x_evaluation@inverse_gram[:,column])*coefficients[column,outcome]/inverse_gram[column,column]
    full_error=y_evaluation[:,outcome]-full_prediction
    restricted_error=full_error+removed_prediction
    return restricted_error**2-full_error**2,full_error,restricted_error


def make_design(data,names,candidates,history_lags=3,source_lags=3,start_lag=None):
    t,r,v=data.shape
    start_lag=history_lags if start_lag is None else int(start_lag)
    if start_lag<history_lags or source_lags>history_lags:
        raise ValueError('Require start_lag >= history_lags >= source_lags')
    vi={name:i for i,name in enumerate(names)}
    x=np.column_stack([np.ones(t-start_lag)]+[data[start_lag-l:t-l].reshape(t-start_lag,-1) for l in range(1,history_lags+1)])
    targets=sorted({(e.target_region,e.target_var) for e in candidates})
    target_index={key:i for i,key in enumerate(targets)}
    y=np.column_stack([data[start_lag:,region,vi[name]] for region,name in targets])
    descriptors=[]
    for edge in candidates:
        for lag in range(1,source_lags+1):
            row=edge.to_dict()
            row.update(lag=lag,column=1+(lag-1)*r*v+edge.source_region*v+vi[edge.source_var],
                       outcome=target_index[(edge.target_region,edge.target_var)])
            descriptors.append(row)
    return x,y,targets,pd.DataFrame(descriptors)


def infer_period(x,y,fit,specs,selected_columns,period,calendar_offset,progress):
    rows=[]
    n,p=x.shape
    lookup={col:j for j,col in enumerate(selected_columns)}
    rss=np.sum(fit['residual']**2,axis=0)
    tss=np.sum((y-y.mean(axis=0))**2,axis=0)
    diag=np.diag(fit['inverse_gram'])
    for count,(target,group) in enumerate(specs.groupby('outcome',sort=True),1):
        cols=group.column.to_numpy(dtype=int)
        direction=fit['directions'][:,[lookup[col] for col in cols]]
        score=direction*fit['residual'][:,target,None]
        beta=fit['coefficients'][cols,target]
        hac_se,hac_p,(cluster_se,cluster_p,g,correction)=score_inference(score,beta,p,calendar_offset)
        ols_se=np.sqrt(rss[target]/(n-p)*diag[cols])
        drop=beta**2/diag[cols]
        ols_p=2*stats.t.sf(np.abs(beta)/ols_se,n-p)
        for j,(_,descriptor) in enumerate(group.iterrows()):
            row=descriptor.to_dict()
            row.update(period=period,effect=float(beta[j]),effect_abs=float(abs(beta[j])),
                p_ols=float(ols_p[j]),se_ols=float(ols_se[j]),
                p_hac64=float(hac_p[j]),se_hac64=float(hac_se[j]),
                ci95_low_hac64=float(beta[j]-stats.norm.ppf(.975)*hac_se[j]),
                ci95_high_hac64=float(beta[j]+stats.norm.ppf(.975)*hac_se[j]),
                p_cluster128=float(cluster_p[j]),se_cluster128=float(cluster_se[j]),
                ci95_low_cluster128=float(beta[j]-stats.t.ppf(.975,g-1)*cluster_se[j]),
                ci95_high_cluster128=float(beta[j]+stats.t.ppf(.975,g-1)*cluster_se[j]),
                n_samples=n,n_parameters=p,n_clusters128=g,cluster_correction128=float(correction),
                partial_r2=float(drop[j]/(rss[target]+drop[j])),delta_r2=float(drop[j]/tss[target]),
                r2_full=float(1-rss[target]/tss[target]))
            rows.append(row)
        if count%22==0:
            progress(f'{period} covariance {count}/132')
    table=pd.DataFrame(rows)
    for method in ('ols','hac64','cluster128'):
        table[f'q_{method}_global']=adjust_bh(table[f'p_{method}'])
        table[f'q_{method}_target_family']=table.groupby('target_var')[f'p_{method}'].transform(adjust_bh)
    return table


def predictive_table(x_test,y_test,train_fit,specs,progress):
    predictions=x_test@train_fit['coefficients']
    errors=y_test-predictions
    selected=sorted(specs.column.unique())
    directions=x_test@train_fit['inverse_gram'][:,selected]
    lookup={col:j for j,col in enumerate(selected)}
    rows=[]
    for count,(target,group) in enumerate(specs.groupby('outcome',sort=True),1):
        cols=group.column.to_numpy(dtype=int)
        beta=train_fit['coefficients'][cols,target]
        delta=directions[:,[lookup[col] for col in cols]]*(beta/np.diag(train_fit['inverse_gram'])[cols])
        err=errors[:,target,None]
        gains=(err+delta)**2-err**2
        mean=gains.mean(axis=0)
        centered=gains-mean
        acov=score_autocovariances(centered,64)
        h=np.arange(1,len(acov))
        meat=acov[0]+2*np.sum((1-h[:,None]/65)*acov[h],axis=0)
        se=np.sqrt(np.maximum(meat*len(gains)/(len(gains)-1),0))/len(gains)
        p=2*stats.norm.sf(np.abs(mean)/np.maximum(se,1e-300))
        for j,(_,descriptor) in enumerate(group.iterrows()):
            row=descriptor.to_dict()
            row.update(frozen_training_effect=float(beta[j]),mse_gain=float(mean[j]),
                mse_gain_se_hac64=float(se[j]),mse_gain_ci95_low=float(mean[j]-1.96*se[j]),
                mse_gain_ci95_high=float(mean[j]+1.96*se[j]),p_predictive_gain=float(p[j]),
                mse_full=float(np.mean(err**2)),mse_drop_one_source_lag=float(np.mean((err[:,0]+delta[:,j])**2)),
                mean_relative_mse_gain=float(mean[j]/np.mean(err**2)))
            rows.append(row)
        if count%22==0:
            progress(f'frozen ablation covariance {count}/132')
    result=pd.DataFrame(rows)
    result['q_predictive_gain_global']=adjust_bh(result.p_predictive_gain)
    return result,predictions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT.parent/'revision_outputs'/'inputs'/'region_trainfit.npz')
    parser.add_argument('--output',type=Path,default=ROOT.parent/'revision_outputs'/'full_var')
    parser.add_argument('--history-lags',type=int,default=3)
    parser.add_argument('--source-lags',type=int,default=3)
    parser.add_argument('--start-lag',type=int,default=None)
    args=parser.parse_args()
    if args.start_lag is None:
        args.start_lag=args.history_lags
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    process=psutil.Process()
    def progress(message):
        status=dict(message=message,elapsed_seconds=time.perf_counter()-start,rss_mb=process.memory_info().rss/1024**2)
        print(json.dumps(status),flush=True)
        (args.output/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
        analysis_class='Post-diagnostic dense conditional VAR benchmark',
        rationale='Known-graph simulation showed own-history models mistake indirect/common-driver associations for direct links.',
        input=str(args.input),input_sha256=sha256(args.input),history_lags=args.history_lags,
        source_lags=args.source_lags,start_lag=args.start_lag,
        predictors=f'All 264 observed region-variable nodes at lags1..{args.history_lags} plus intercept; no current-time covariates.',
        outcomes='132 humidity/cloud_cover region outcomes; shared QR OLS per period.',
        training_end_exclusive='2019-01-01',evaluation='2019-01-01 through 2025-12-31',
        inference=f'Original 858 candidate edges x source lags1..{args.source_lags}; HAC64 and exploratory cluster128.',
        predictive_comparison='Fit full model on training, freeze coefficients, compare heldout squared loss against exact training restricted fit dropping one source-lag coefficient.',
        regularization=None,
        limitations=['This conditional model includes all observed regional variables, not unobserved atmospheric drivers.',
            'HAC64 and cluster128 calibration failed some simulations; adjusted p-values are sensitivity statistics, not proof of nominal FDR control.',
            'Seven-year holdout is retrospective and previously examined; not an untouched prospective validation sample.',
            'Individual lag coefficients condition on all other lags, unlike original single-source-lag additions.',
            'Drop-one-predictor ablation is conditional predictive contribution, not intervention or removal of a physical variable.',
            'No regime-stratified VAR fit in this benchmark.'],
        software={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','statsmodels','psutil']})
    source_paths=[Path(__file__),ROOT/'revision'/'inference.py',ROOT/'src'/'causal_weathergraph'/'candidate_edges.py']
    manifest['source_hashes']={str(p.relative_to(ROOT)):sha256(p) for p in source_paths}
    for relative in manifest['source_hashes']:
        destination=args.output/'code_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/relative).read_bytes())
    (args.output/'protocol.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with np.load(args.input,allow_pickle=False) as archive:
        data=archive['data']; names=archive['variable_names'].tolist()
        ts=pd.DatetimeIndex(archive['timestamps']); lat=archive['lat'];lon=archive['lon']
    if not np.all(np.diff(ts.as_unit('ns').asi8)==pd.Timedelta(hours=6).value):
        raise ValueError('Complete six-hour calendar required')
    candidates=build_candidate_edges(names,lat,lon,{'candidate_k_nearest':2})
    manifest['alignment']='Both history3/start12 and history12/start12 use identical original target timestamps t>=12; earliest test lags can use observed pre-2019 history, as in the original temporal holdout.'
    manifest['post_diagnostic_lag_expansion']=bool(args.history_lags>3)
    if args.history_lags>3:
        manifest['lag_expansion_trigger']='Residual daily autocorrelation remained after VAR3; fixed history12 examines longer measured histories without selecting history length for more discoveries.'
        manifest['limitations'].append('Many regressors relative to test-period sample size can impair covariance finite-sample calibration; HC1/CR1 corrections do not guarantee control.')
    (args.output/'protocol.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    x,y,targets,specs=make_design(data,names,candidates,args.history_lags,args.source_lags,args.start_lag)
    assert x.shape[1]==1+264*args.history_lags and y.shape[1]==132 and len(specs)==858*args.source_lags
    train=ts[args.start_lag:]<pd.Timestamp('2019-01-01')
    test=~train
    if not np.all(np.diff(np.flatnonzero(train))==1) or not np.all(np.diff(np.flatnonzero(test))==1):
        raise ValueError('Training and test must each be contiguous calendar periods')
    selected=sorted(specs.column.unique())
    fit_diagnostics={}
    tables={}
    residual_diagnostics=[]
    residual_variance_rows=[]
    residual_variance_summaries=[]
    with threadpool_limits(limits=4):
        manifest['threadpools']=threadpool_info()
        for period,mask in [('discovery',train),('evaluation',test)]:
            progress(f'{period} shared QR start, n={int(mask.sum())}, p={x.shape[1]}')
            try:
                fit=shared_qr_fit(x[mask],y[mask],selected)
            except Exception as error:
                manifest.update(status='failed',error=str(error))
                (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                raise
            fit_diagnostics[period]=fit['diagnostics']
            (args.output/'fit_diagnostics.json').write_text(json.dumps(fit_diagnostics,indent=2),encoding='utf-8')
            progress(f'{period} QR done; condition={fit["diagnostics"]["condition_number"]:.4g}')
            offset=int(np.flatnonzero(mask)[0])+args.start_lag
            table=infer_period(x[mask],y[mask],fit,specs,selected,period,offset,progress)
            table.to_csv(args.output/f'{period}_full_var_edges.csv',index=False)
            tables[period]=table
            for lag in (1,4,12,28):
                a=fit['residual'][lag:]
                b=fit['residual'][:-lag]
                correlation=np.sum(a*b,axis=0)/np.sqrt(np.sum(a*a,axis=0)*np.sum(b*b,axis=0))
                for target,(region,variable) in enumerate(targets):
                    residual_diagnostics.append(dict(period=period,target_region=region,target_var=variable,
                        calendar_lag_steps=lag,calendar_lag_hours=6*lag,residual_acf=float(correlation[target]),
                        n_pairs=len(a),description='Full dense VAR residual, refit within this period'))
            residual_frame=pd.DataFrame(fit['residual'])
            dates=ts[args.start_lag:][mask]
            for frequency,name in [('M','calendar_month'),('Y','calendar_year')]:
                variances=residual_frame.groupby(dates.to_period(frequency)).var(ddof=1)
                for target,(region,variable) in enumerate(targets):
                    values=variances[target].to_numpy()
                    residual_variance_summaries.append(dict(period=period,target_region=region,target_var=variable,
                        grouping=name,n_periods=len(values),variance_min=float(values.min()),variance_max=float(values.max()),
                        variance_max_min_ratio=float(values.max()/values.min()),
                        variance_p90_p10_ratio=float(np.quantile(values,.9)/np.quantile(values,.1))))
                    for label,value in variances[target].items():
                        residual_variance_rows.append(dict(period=period,target_region=region,target_var=variable,
                            grouping=name,calendar_period=str(label),residual_variance=float(value)))
            np.savez_compressed(args.output/f'{period}_full_var_model.npz',coefficients=fit['coefficients'],
                inverse_gram=fit['inverse_gram'],targets=np.array([f'{r}:{v}' for r,v in targets]),
                variable_names=np.asarray(names),history_lags=args.history_lags,source_lags=args.source_lags,start_lag=args.start_lag)
            if period=='discovery':
                frozen={k:fit[k] for k in ('coefficients','inverse_gram')}
            del fit
        progress('Frozen source ablations start')
        prediction,predictions=predictive_table(x[test],y[test],frozen,specs,progress)
        prediction.to_csv(args.output/'frozen_prediction_full_var_all_candidates.csv',index=False)
        performance=[]
        vi={name:i for i,name in enumerate(names)}
        n_nodes=data.shape[1]*data.shape[2]
        for target,(region,variable) in enumerate(targets):
            own_columns=[0]+[1+(lag-1)*n_nodes+region*len(names)+vi[variable] for lag in range(1,args.history_lags+1)]
            own_coefficients=np.linalg.lstsq(x[np.ix_(train,own_columns)],y[train,target],rcond=None)[0]
            own_error=y[test,target]-x[np.ix_(test,own_columns)]@own_coefficients
            full_error=y[test,target]-predictions[:,target]
            own_mse=float(np.mean(own_error**2))
            full_mse=float(np.mean(full_error**2))
            denominator=float(np.mean((y[test,target]-y[train,target].mean())**2))
            performance.append(dict(target_region=region,target_var=variable,own_history_lags=args.history_lags,
                heldout_mse_frozen_full_var=full_mse,heldout_mse_frozen_own_history=own_mse,
                relative_mse_improvement_vs_own_history=1-full_mse/own_mse,
                heldout_r2_vs_training_mean=1-full_mse/denominator))
        pd.DataFrame(performance).to_csv(args.output/'frozen_model_performance.csv',index=False)
    residual_table=pd.DataFrame(residual_diagnostics)
    residual_table.to_csv(args.output/'full_var_residual_calendar_acf.csv',index=False)
    residual_table.groupby(['period','calendar_lag_steps']).residual_acf.agg(['median','min','max']).to_csv(args.output/'full_var_residual_calendar_acf_summary.csv')
    pd.DataFrame(residual_variance_rows).to_csv(args.output/'full_var_residual_variance_by_month_year.csv',index=False)
    pd.DataFrame(residual_variance_summaries).to_csv(args.output/'full_var_residual_variance_ratios.csv',index=False)
    combined=tables['discovery'][KEY+['effect','q_hac64_global','q_cluster128_global','partial_r2']].rename(
        columns={c:'train_'+c for c in ['effect','q_hac64_global','q_cluster128_global','partial_r2']})
    combined=combined.merge(tables['evaluation'][KEY+['effect','q_hac64_global','q_cluster128_global']],on=KEY,validate='one_to_one').rename(
        columns={c:'test_'+c for c in ['effect','q_hac64_global','q_cluster128_global']})
    combined=combined.merge(prediction,on=KEY,validate='one_to_one')
    combined.to_csv(args.output/'full_var_holdout_all_candidates.csv',index=False)
    summaries=[]
    for period,table in tables.items():
        for target,g in [('ALL',table)]+list(table.groupby('target_var')):
            for method in ('ols','hac64','cluster128'):
                for family in ('global','target_family'):
                    keep=g[f'q_{method}_{family}']<.05
                    summaries.append(dict(period=period,target_var=target,method=method,bh_family=family,
                        n_tests=len(g),n_q_below_05=int(keep.sum()),median_abs_effect=float(g[keep].effect_abs.median()),
                        median_partial_r2=float(g[keep].partial_r2.median()),median_delta_r2=float(g[keep].delta_r2.median())))
    pd.DataFrame(summaries).to_csv(args.output/'inference_summary.csv',index=False)
    own=pd.read_csv(args.output.parent/'holdout'/'holdout_all_candidates.csv')
    matched=combined.merge(own[KEY+['train_effect','train_q_hac64_global','test_effect','test_q_hac64_global','mse_gain']],on=KEY,suffixes=('_fullvar','_ownhistory'),validate='one_to_one')
    matched.to_csv(args.output/'own_history_vs_full_var_all_candidates.csv',index=False)
    comparisons=[]
    for name,g in [('ALL',matched)]+list(matched.groupby('edge_type')):
        comparisons.append(dict(edge_type=name,n=len(g),
            train_q05_ownhistory=int((g.train_q_hac64_global_ownhistory<.05).sum()),
            train_q05_fullvar=int((g.train_q_hac64_global_fullvar<.05).sum()),
            test_q05_ownhistory=int((g.test_q_hac64_global_ownhistory<.05).sum()),
            test_q05_fullvar=int((g.test_q_hac64_global_fullvar<.05).sum()),
            train_effect_same_sign_models=int((np.sign(g.train_effect_fullvar)==np.sign(g.train_effect_ownhistory)).sum()),
            fullvar_train_test_same_sign=int((np.sign(g.train_effect_fullvar)==np.sign(g.test_effect_fullvar)).sum()),
            positive_frozen_gain_ownhistory=int((g.mse_gain_ownhistory>0).sum()),
            positive_frozen_gain_fullvar=int((g.mse_gain_fullvar>0).sum()),
            positive_frozen_gain_q05_fullvar=int(((g.mse_gain_fullvar>0)&(g.q_predictive_gain_global<.05)).sum()),
            median_fullvar_frozen_relative_mse_gain=float(g.mean_relative_mse_gain.median())))
    pd.DataFrame(comparisons).to_csv(args.output/'own_history_vs_full_var_summary.csv',index=False)
    manifest.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-start,fit_diagnostics=fit_diagnostics,
        peak_working_set_bytes=int(getattr(process.memory_info(),'peak_wset',process.memory_info().rss)))
    manifest['output_sha256']={p.name:sha256(p) for pattern in ('*.csv','*.npz') for p in args.output.glob(pattern)}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__=='__main__':main()
