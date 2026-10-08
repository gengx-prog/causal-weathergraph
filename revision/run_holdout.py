"""Frozen-training predictive evaluation and all-candidate held-period agreement."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from causal_weathergraph.candidate_edges import build_candidate_edges
from revision.inference import run_graph_discovery, score_autocovariances, adjust_bh
from revision.prepare_inputs import sha256


KEY = ['source_region','target_region','source_var','target_var','lag']


def predictive_losses(data, names, candidates, training, evaluation, lag_max=3, bandwidth=64):
    """Coefficients frozen on discovery observations; evaluate every candidate."""
    vi = {n:i for i,n in enumerate(names)}
    groups = defaultdict(list)
    for edge in candidates:
        groups[(edge.target_region,edge.target_var)].append(edge)
    tr, te = training[lag_max:], evaluation[lag_max:]
    t = len(data)
    records=[]
    for (region,target),edges in groups.items():
        y=data[lag_max:,region,vi[target]]
        c=np.column_stack([np.ones(len(y))]+[data[lag_max-l:t-l,region,vi[target]] for l in range(1,lag_max+1)])
        specs=[(e,l) for e in edges for l in range(1,lag_max+1)]
        x=np.column_stack([data[lag_max-l:t-l,e.source_region,vi[e.source_var]] for e,l in specs])
        target_fit=np.linalg.lstsq(c[tr],y[tr],rcond=None)[0]
        source_fit=np.linalg.lstsq(c[tr],x[tr],rcond=None)[0]
        yr=y[tr]-c[tr]@target_fit
        xr=x[tr]-c[tr]@source_fit
        beta=(yr@xr)/np.sum(xr*xr,axis=0)
        err_r=y[te]-c[te]@target_fit
        incremental=(x[te]-c[te]@source_fit)*beta
        err_f=err_r[:,None]-incremental
        gain=err_r[:,None]**2-err_f**2
        mean=gain.mean(axis=0)
        centered=gain-mean
        # Evaluation is an actual contiguous period; no time compression across gaps.
        if not np.all(np.diff(np.flatnonzero(te))==1):
            raise ValueError('Evaluation period must be contiguous')
        cov=score_autocovariances(centered,bandwidth)
        h=np.arange(1,len(cov))
        meat=cov[0]+2*np.sum((1-h[:,None]/(bandwidth+1))*cov[h],axis=0)
        se=np.sqrt(np.maximum(meat,0)*len(gain)/(len(gain)-1))/len(gain)
        p=2*stats.norm.sf(np.abs(mean)/np.maximum(se,1e-30))
        tss=float(np.sum((y[te]-y[tr].mean())**2))
        tss_test_centered=float(np.sum((y[te]-y[te].mean())**2))
        for j,(edge,lag) in enumerate(specs):
            row=edge.to_dict()
            row.update(lag=lag, training_n=int(tr.sum()),evaluation_n=int(te.sum()),frozen_source_beta=float(beta[j]),mse_gain=float(mean[j]),mse_gain_se_hac64=float(se[j]),mse_gain_ci95_low=float(mean[j]-1.96*se[j]),mse_gain_ci95_high=float(mean[j]+1.96*se[j]),p_predictive_gain=float(p[j]),heldout_delta_r2=float(gain[:,j].sum()/tss),mse_restricted=float(np.mean(err_r**2)),mse_full=float(np.mean(err_f[:,j]**2)))
            row['heldout_delta_r2_test_centered']=float(gain[:,j].sum()/tss_test_centered)
            records.append(row)
    frame=pd.DataFrame(records)
    frame['q_predictive_gain_global']=adjust_bh(frame.p_predictive_gain)
    return frame


def pcmci_all_candidates(data, names, candidates, evaluation, lag_max=3):
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    vi={n:i for i,n in enumerate(names)}
    nvars=len(names)
    selected=np.flatnonzero(evaluation)
    if not np.all(np.diff(selected)==1) or selected[0]<2*lag_max:
        raise ValueError('PCMCI evaluation requires a contiguous period and observed history prefix')
    # Tigramite's default cutoff is 2*tau_max. Retain that many already observed
    # training points as predictor history so every response lies in evaluation.
    matrix=data[selected[0]-2*lag_max:selected[-1]+1].reshape(len(selected)+2*lag_max,-1)
    node=lambda r,v: int(r)*nvars+vi[v]
    assumptions={j:{(j,-l):'-?>' for l in range(1,lag_max+1)} for j in range(matrix.shape[1])}
    for e in candidates:
        i,j=node(e.source_region,e.source_var),node(e.target_region,e.target_var)
        assumptions[j].update({(i,-l):'-?>' for l in range(1,lag_max+1)})
    pcmci=PCMCI(dataframe=pp.DataFrame(matrix),cond_ind_test=ParCorr(significance='analytic'),verbosity=0)
    audit_array,_,_=pcmci.dataframe.construct_array(X=[(0,-1)],Y=[(1,0)],Z=[],tau_max=lag_max,cut_off='2xtau_max')
    if audit_array.shape[1]!=len(selected):
        raise ValueError('Unexpected PCMCI response sample count')
    result=pcmci.run_pcmci(tau_min=1,tau_max=lag_max,pc_alpha=.05,alpha_level=.05,max_conds_dim=3,max_combinations=1,max_conds_py=3,max_conds_px=3,link_assumptions=assumptions,fdr_method='none')
    rows=[]
    for e in candidates:
        i,j=node(e.source_region,e.source_var),node(e.target_region,e.target_var)
        for lag in range(1,lag_max+1):
            row=e.to_dict()
            row.update(lag=lag,p_pcmci=float(result['p_matrix'][i,j,lag]),pcmci_partial_correlation=float(result['val_matrix'][i,j,lag]),pcmci_response_n=len(selected),pcmci_observed_history_prefix=2*lag_max)
            rows.append(row)
    table=pd.DataFrame(rows)
    table['q_pcmci_global']=adjust_bh(table.p_pcmci)
    table['q_pcmci_target_family']=table.groupby('target_var').p_pcmci.transform(adjust_bh)
    return table


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pcmci',action='store_true')
    p.add_argument('--reuse-granger',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with np.load(args.input,allow_pickle=False) as z:
        data=z['data']; names=z['variable_names'].tolist(); lat=z['lat'];lon=z['lon'];ts=z['timestamps']
    tr=ts<np.datetime64('2019-01-01');te=~tr
    candidates=build_candidate_edges(names,lat,lon,{'candidate_k_nearest':2})
    fingerprint={'input_sha256':sha256(args.input),'code_sha256':sha256(__file__),'inference_sha256':sha256(Path(__file__).with_name('inference.py'))}
    settings={'train':[str(ts[tr][0]),str(ts[tr][-1])],'test':[str(ts[te][0]),str(ts[te][-1])],'candidate_count':len(candidates),'lag':3,'hac_bandwidth':64,'fdr_family':'all 2574 candidate-lag hypotheses within each period; target-specific BH also retained','predictive_gain':'MSE_R - MSE_F using coefficients frozen in training; positive is better','heldout_delta_r2_denominator':'test sum of squared deviations from training target mean; additional test-centered column also exported','pcmci':bool(args.pcmci),'pcmci_max_conds_dim':3,'pcmci_max_conds_px':3,'pcmci_max_conds_py':3,'pcmci_observed_history_prefix':6,'pcmci_effective_response_n':int(te.sum()),'interpretation':'retrospective temporal holdout; previously examined period; PCMCI agreement is not graph accuracy',**fingerprint}
    (args.output/'design.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    tables=[]
    for period,mask in [('discovery',tr),('evaluation',te)]:
        target=args.output/f'{period}_edges.csv'
        if args.reuse_granger and target.exists():
            table=pd.read_csv(target)
            expected=json.loads((args.output/'granger_inputs.json').read_text())
            if expected!=fingerprint:
                raise ValueError('Reuse fingerprint mismatch')
        else:
            table=run_graph_discovery(data,names,candidates,{period:mask},bandwidths=(64,),progress=lambda done,total,*_:print(f'{period} {done}/{total}',flush=True) if done%22==0 else None)
            table.to_csv(target,index=False)
        tables.append(table)
    (args.output/'granger_inputs.json').write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
    losses=predictive_losses(data,names,candidates,tr,te)
    losses.to_csv(args.output/'frozen_prediction_all_candidates.csv',index=False)
    train,test=tables
    selected=train[KEY+['effect','p_hac64','q_hac64_global','partial_r2']].rename(columns={c:'train_'+c for c in ['effect','p_hac64','q_hac64_global','partial_r2']})
    merged=selected.merge(test[KEY+['effect','p_hac64','q_hac64_global']],on=KEY,validate='one_to_one').rename(columns={c:'test_'+c for c in ['effect','p_hac64','q_hac64_global']})
    merged=merged.merge(losses,on=KEY,validate='one_to_one')
    merged['stratum']='unselected'
    sig=merged.train_q_hac64_global<.05
    merged.loc[sig,'stratum']='ordinary_discovery'
    # A fixed top quartile among significant training effects, within relation type.
    for _,idx in merged[sig].groupby('edge_type').groups.items():
        cutoff=merged.loc[idx,'train_effect'].abs().quantile(.75)
        merged.loc[idx[merged.loc[idx,'train_effect'].abs()>=cutoff],'stratum']='strong_discovery'
    if args.pcmci:
        print('Starting full candidate PCMCI on held-period data',flush=True)
        pcmci=pcmci_all_candidates(data,names,candidates,te)
        pcmci.to_csv(args.output/'heldout_pcmci_all_candidates.csv',index=False)
        merged=merged.merge(pcmci[KEY+['p_pcmci','q_pcmci_global','pcmci_partial_correlation']],on=KEY,validate='one_to_one')
    merged.to_csv(args.output/'holdout_all_candidates.csv',index=False)
    rows=[]
    for (edge_type,stratum),g in merged.groupby(['edge_type','stratum']):
        row={'edge_type':edge_type,'stratum':stratum,'n':len(g),'test_hac_significant':int((g.test_q_hac64_global<.05).sum()),'same_sign':int((np.sign(g.train_effect)==np.sign(g.test_effect)).sum()),'positive_frozen_mse_gain':int((g.mse_gain>0).sum()),'positive_gain_global_q05':int(((g.mse_gain>0)&(g.q_predictive_gain_global<.05)).sum()),'median_frozen_delta_r2':float(g.heldout_delta_r2.median())}
        if args.pcmci: row['pcmci_significant']=int((g.q_pcmci_global<.05).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output/'holdout_summary.csv',index=False)
    settings.update(elapsed_seconds=time.perf_counter()-start,peak_working_set_bytes=int(getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)),completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),software={x:importlib.metadata.version(x) for x in ['numpy','scipy','pandas','tigramite']})
    (args.output/'manifest.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print('Holdout complete',flush=True)


if __name__=='__main__':main()
