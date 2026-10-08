"""Direct slope contrasts using training-defined current and antecedent regimes."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import psutil
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from causal_weathergraph.candidate_edges import build_candidate_edges
from revision.inference import _basis, score_autocovariances, adjust_bh
from revision.prepare_inputs import sha256


def contrast_batch(y, controls, sources, high, eligible, bandwidth=64):
    """Saturated two-group intercept/own-history with source main and interaction.

    Return high-minus-reference source slopes and calendar-HAC uncertainty.
    """
    y=np.asarray(y); c=np.asarray(controls); x=np.asarray(sources)
    if x.ndim==1:x=x[:,None]
    state=np.asarray(high,dtype=float)
    mask=np.asarray(eligible,dtype=bool)&np.isfinite(y)&np.isfinite(c).all(1)&np.isfinite(x).all(1)
    n=int(mask.sum())
    c_full=np.column_stack([np.ones(len(y)),c,state,c*state[:,None]])
    q,rank=_basis(c_full[mask])
    yy=y[mask]; yy=yy-q@(q.T@yy)
    xx=x[mask];xx=xx-q@(q.T@xx)
    v=(x*state[:,None])[mask];v=v-q@(q.T@v)
    sxx=np.sum(xx*xx,axis=0)
    if np.any(sxx<1e-12):raise ValueError('Unidentifiable source in interaction model')
    ratio=np.sum(xx*v,axis=0)/sxx
    z=v-xx*ratio
    base_beta=yy@xx/sxx
    target=yy[:,None]-xx*base_beta
    zz=np.sum(z*z,axis=0)
    if np.any(zz<1e-12):raise ValueError('Unidentifiable interaction or empty regime')
    delta=np.sum(z*target,axis=0)/zz
    residual=target-z*delta
    score=np.zeros_like(x);score[mask]=z*residual
    acov=score_autocovariances(score,bandwidth)
    h=np.arange(1,len(acov))
    meat=acov[0]+2*np.sum((1-h[:,None]/(bandwidth+1))*acov[h],axis=0)
    se=np.sqrt(np.maximum(meat,0)*n/(n-rank-2))/zz
    lo=base_beta-ratio*delta
    return [dict(delta_beta=float(delta[j]),beta_reference=float(lo[j]),beta_high=float(lo[j]+delta[j]),se_hac64=float(se[j]),ci95_low=float(delta[j]-1.96*se[j]),ci95_high=float(delta[j]+1.96*se[j]),p_hac64=float(2*stats.norm.sf(abs(delta[j])/max(se[j],1e-30))),n_samples=n,n_high=int((mask&(state>0)).sum()),n_reference=int((mask&(state==0)).sum())) for j in range(x.shape[1])]


def state_definitions(data,names,timestamps,training):
    months=pd.DatetimeIndex(timestamps).month.to_numpy()
    defs=[];thresholds={}
    for name,var in [('humidity','humidity'),('cloud','cloud_cover')]:
        index=data[:,:,names.index(var)].mean(axis=1)
        low,upper=np.quantile(index[training],[.2,.8])
        thresholds[name]=dict(q20=float(low),q80=float(upper))
        for shift in [0,4]:
            g=np.full_like(index,np.nan)
            if shift:g[shift:]=index[:-shift]
            else:g=index.copy()
            high=g>=upper
            normal=((g>low)&(g<upper)) if name=='humidity' else g<upper
            defs.append((f'{name}_lag{shift}',high,high|normal,None))
    season=np.isin(months,[12,1,2,6,7,8])
    defs.append(('NH_local_winter_minus_summer',np.isin(months,[12,1,2]),season,'NH'))
    defs.append(('SH_local_winter_minus_summer',np.isin(months,[6,7,8]),season,'SH'))
    return defs,thresholds


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    with np.load(a.input) as z:
        data=z['data'];names=z['variable_names'].tolist();lat=z['lat'];lon=z['lon'];ts=z['timestamps']
    training=ts<np.datetime64('2019-01-01');vi={n:i for i,n in enumerate(names)}
    candidates=build_candidate_edges(names,lat,lon,{'candidate_k_nearest':2})
    definitions,thresholds=state_definitions(data,names,ts,training)
    design={'input_sha256':sha256(a.input),'code_sha256':sha256(__file__),'thresholds_training_only':thresholds,'state_lead_steps':4,'lag_max':3,'hac_bandwidth':64,'controls':'regime-specific intercept and target own lags1-3; source main effect and source-by-regime interaction','families':'BH across all tested candidate-lag interactions within each period and contrast; also across all contrasts/periods sensitivity','caution':'Lagged endogenous index is a sensitivity analysis, not guaranteed exogenous. Hemisphere seasons are calendar-defined.'}
    (a.output/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    groups=defaultdict(list)
    for e in candidates:groups[(e.target_region,e.target_var)].append(e)
    rows=[];t=len(data);L=3
    for period,period_mask in [('discovery',training),('evaluation',~training)]:
        for label,high,eligible,hemisphere in definitions:
            for (region,target),edges in groups.items():
                if hemisphere=='NH' and lat[region]<=0:continue
                if hemisphere=='SH' and lat[region]>=0:continue
                y=data[L:,region,vi[target]]
                c=np.column_stack([data[L-l:t-l,region,vi[target]] for l in range(1,L+1)])
                specs=[(e,l) for e in edges for l in range(1,L+1)]
                x=np.column_stack([data[L-l:t-l,e.source_region,vi[e.source_var]] for e,l in specs])
                fitted=contrast_batch(y,c,x,high[L:],(eligible&period_mask)[L:])
                for (e,l),fit in zip(specs,fitted):
                    fit.update(e.to_dict());fit.update(lag=l,period=period,contrast=label,target_latitude=float(lat[region]));rows.append(fit)
            print(f'Completed {period} {label}',flush=True)
    out=pd.DataFrame(rows)
    out['q_hac64_contrast']=out.groupby(['period','contrast']).p_hac64.transform(adjust_bh)
    out['q_hac64_all_contrasts']=adjust_bh(out.p_hac64)
    out.to_csv(a.output/'regime_interaction_edges.csv',index=False)
    summary=out.groupby(['period','contrast','edge_type']).agg(tested=('p_hac64','size'),significant=('q_hac64_contrast',lambda s:int((s<.05).sum())),median_delta=('delta_beta','median'),median_abs_delta=('delta_beta',lambda s:s.abs().median()),n_high=('n_high','min'),n_reference=('n_reference','min')).reset_index()
    summary.to_csv(a.output/'regime_interaction_summary.csv',index=False)
    design.update(elapsed_seconds=time.perf_counter()-start,peak_working_set_bytes=int(getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)),completed_utc=pd.Timestamp.now(tz='UTC').isoformat())
    (a.output/'manifest.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    print('Regime contrasts complete',flush=True)


if __name__=='__main__':main()
