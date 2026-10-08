"""Comparable indexed-edge evaluation of official CaStLe on local grid SCMs.

Load only the official PCMCI and concatenation function ASTs to avoid importing
the unrelated optional DYNOTEARS dependency. Their function bodies are unedited.
Apply the same BH family to raw p values from each method for reported metrics.
Local regressions and local PCMCI share responses t=2,...,T-1. Official CaStLe
retains its upstream pooled concatenation and its different sample handling.
"""
from __future__ import annotations
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import psutil
from scipy import stats
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.independence_tests_base import CondIndTest
from revision.inference import adjust_bh
from revision.prepare_inputs import sha256

OFFSETS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
NONSELF=[0,1,2,3,5,6,7,8]


def generate_grid(seed,size=6,length=256,heterogeneous=False):
    rng=np.random.default_rng(seed);burn=256
    arr=np.zeros((size,size,length+burn));truth=[]
    for t in range(1,length+burn):
        previous=arr[:,:,t-1]
        padded=np.pad(previous,1)
        left=padded[1:-1,:-2];right=padded[1:-1,2:];above=padded[:-2,1:-1]
        source=left.copy()
        if heterogeneous:source[:,size//2:]=right[:,size//2:]
        arr[:,:,t]=.45*previous+.20*source+.12*above+rng.normal(size=(size,size))
    for r in range(1,size-1):
        for c in range(1,size-1):
            true=np.zeros(9,bool);true[1]=True;true[5 if heterogeneous and c>=size//2 else 3]=True
            truth.append(true[NONSELF])
    return arr[:,:,burn:],np.array(truth)


def load_official(source):
    captures={}
    class RecordingPCMCI(PCMCI):
        def run_pcmci(self,*args,**kwargs):
            result=super().run_pcmci(*args,**kwargs)
            captures['p']=result['p_matrix'].copy()
            return result
    tree=ast.parse(Path(source).read_text(encoding='utf-8'))
    selected={'concatenate_timeseries_wrapping','concatenate_timeseries_nonwrapping','CaStLe_PCMCI'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in selected]
    if len(functions)!=3:raise ValueError('Unexpected official CaStLe API')
    scope={'np':np,'pp':pp,'PCMCI':RecordingPCMCI,'CondIndTest':CondIndTest}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),scope)
    return scope,captures


def neighborhoods(data):
    size=data.shape[0]
    for r in range(1,size-1):
        for c in range(1,size-1):
            yield np.column_stack([data[r+dr,c+dc] for dr,dc in OFFSETS])


def local_methods(data):
    own=[];full=[];pcmci=[]
    assumptions={j:({(i,-1):'-?>' for i in range(9)} if j==4 else {}) for j in range(9)}
    elapsed={'local_own_history':0.,'local_multivariable':0.,'local_pcmci':0.}
    for x in neighborhoods(data):
        # Match Tigramite's default cut_off='2xtau_max' at tau_max=1.
        # Preserve the full x for PCMCI, which may use lag-two conditions.
        y=x[2:,4];history=x[1:-1]
        tick=time.perf_counter()
        controls=np.column_stack([np.ones(len(y)),history[:,4]])
        q,_=np.linalg.qr(controls,mode='reduced')
        yr=y-q@(q.T@y);z=history[:,NONSELF]-q@(q.T@history[:,NONSELF])
        ss=np.sum(z*z,axis=0);beta=yr@z/ss;res=yr[:,None]-z*beta
        se=np.sqrt(np.sum(res*res,axis=0)/(len(y)-3)/ss)
        own.append(2*stats.t.sf(np.abs(beta/se),len(y)-3));elapsed['local_own_history']+=time.perf_counter()-tick
        tick=time.perf_counter()
        design=np.column_stack([np.ones(len(y)),history]);coef=np.linalg.lstsq(design,y,rcond=None)[0]
        residual=y-design@coef;var=float(residual@residual)/(len(y)-10)
        se=np.sqrt(var*np.diag(np.linalg.inv(design.T@design)))
        full.append((2*stats.t.sf(np.abs(coef/se),len(y)-10))[np.array(NONSELF)+1]);elapsed['local_multivariable']+=time.perf_counter()-tick
        tick=time.perf_counter()
        m=PCMCI(pp.DataFrame(x),ParCorr(significance='analytic'),verbosity=0)
        result=m.run_pcmci(tau_min=1,tau_max=1,pc_alpha=.05,alpha_level=.05,link_assumptions=assumptions,fdr_method='none')
        pcmci.append(result['p_matrix'][NONSELF,4,1]);elapsed['local_pcmci']+=time.perf_counter()-tick
    return {'local_own_history':np.array(own),'local_multivariable':np.array(full),'local_pcmci':np.array(pcmci)},elapsed


def response_window_metadata(length):
    """State exact zero-based response windows, including upstream differences."""
    return {
        'index_convention':'Zero-based time indices after the 256-step generator burn-in; all end indices inclusive',
        'local_methods':['local_own_history','local_multivariable','local_pcmci'],
        'local_response_start':2,
        'local_response_end':length-1,
        'local_response_count_per_target':length-2,
        'local_lag_one_predictor_start':1,
        'local_lag_one_predictor_end':length-2,
        'local_pcmci_tau_min':1,
        'local_pcmci_tau_max':1,
        'local_pcmci_cut_off':'2xtau_max (installed Tigramite default)',
        'local_conditioning_difference':'Only response windows are aligned. Conditioning sets differ by method; PCMCI can use lag-two conditions.',
        'previous_local_ols_window':{'response_start':1,'response_end':length-1,'response_count_per_target':length-1},
        'official_castle':{
            'unchanged_function_bodies':True,
            'concatenation_order':'Interior targets in row-major order, all T records per target',
            'response_start_in_concatenated_series':2,
            'response_count_formula':'(size-2)^2 * T - 2',
            'first_target_response_window':[2,length-1],
            'later_target_response_window':[0,length-1],
            'boundary_mask':False,
            'lag_one_seam_response_count_formula':'(size-2)^2 - 1',
            'samples_by_grid_size':{
                str(size):{'interior_targets':(size-2)**2,
                           'concatenated_records':(size-2)**2*length,
                           'response_count':(size-2)**2*length-2,
                           'lag_one_seam_responses':(size-2)**2-1}
                for size in [6,10]
            },
            'disclosure':'Official CaStLe drops two records once after pooling, not separately per target. Later targets include t=0 and t=1 responses; t=0 lag-one predictors cross cell boundaries. Lag-two conditions may also cross boundaries. This upstream handling is retained, so response-window alignment applies only to the three local methods.'
        }
    }


def graph_metrics(truth,pvalues):
    true=np.asarray(truth).ravel();q=adjust_bh(np.asarray(pvalues).ravel());pred=q<.05
    tp=int((true&pred).sum());fp=int((~true&pred).sum());fn=int((true&~pred).sum())
    return dict(tested=len(true),true_edges=int(true.sum()),tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),fdp=fp/max(tp+fp,1),shd=fp+fn)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repetitions',type=int,default=25);p.add_argument('--length',type=int,default=256)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    source=a.source/'src'/'stencil_functions.py';scope,captures=load_official(source)
    commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
    design={'source_repo':'https://github.com/jjakenichol/CaStLe','source_commit':commit,'official_source_sha256':sha256(source),'adapter_sha256':sha256(__file__),'algorithm':'official CaStLe_PCMCI function bodies loaded without unrelated causalnex imports; only raw-p recording wrapper and common evaluation BH are added','scope':'Univariate lag-one stencil benchmark, not a direct multivariate ERA5 regional graph comparison','grid_sizes':[6,10],'length':a.length,'repetitions':a.repetitions,'scenarios':['homogeneous_left_and_above','heterogeneous_left_or_right_and_above'],'seed_base':20260929,'truth':'self coefficient .45, horizontal parent .20, upper parent .12; row sum .77 bounds spectral radius below1','candidate_family':'All eight nonself neighbors of every interior target at lag1; BH over the same indexed-edge family for all methods','method_difference':'CaStLe pools repeated local stencil observations and assumes a shared template; local methods fit each region separately. Spatial averaging is not applied.','official_concatenation_boundaries':'Upstream function concatenates cell records without lag-boundary masks; this is retained and disclosed (at most one seam per cell, length256).','precision_no_discoveries':0,'directed_shd':'FP+FN on indexed source-target-lag entries; direction reversal counts2','software':{x:importlib.metadata.version(x) for x in ['numpy','scipy','tigramite']}}
    design.update(design_frozen_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                  response_windows=response_window_metadata(a.length),
                  correction_scope='Rerun identical seeds, grids, scenarios, length and indexed-edge BH family after aligning local OLS responses to local PCMCI; official CaStLe is unchanged.')
    (a.output/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    snapshot=a.output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    shutil.copy2(source,snapshot/'stencil_functions.py');shutil.copy2(a.source/'LICENSE',snapshot/'LICENSE');shutil.copy2(__file__,snapshot/'run_castle_benchmark.py')
    start=time.perf_counter();rows=[]
    with (a.output/'repetitions.jsonl').open('w',encoding='utf-8') as stream:
        for size in [6,10]:
            for scenario,hetero in [('homogeneous_left_and_above',False),('heterogeneous_left_or_right_and_above',True)]:
                for rep in range(a.repetitions):
                    seed=20260929+size*10000+int(hetero)*1000+rep
                    data,truth=generate_grid(seed,size,a.length,hetero)
                    values,timings=local_methods(data)
                    tick=time.perf_counter()
                    native_graph,_=scope['CaStLe_PCMCI'](data,ParCorr(significance='analytic'),pc_alpha=.05,rows_inverted=True,dependence_threshold=.05,dependencies_wrap=False)
                    timings['castle_pcmci']=time.perf_counter()-tick
                    values['castle_pcmci']=np.tile(captures['p'][NONSELF,4,1],(len(truth),1))
                    for method,pvals in values.items():
                        row=dict(scenario=scenario,size=size,replicate=rep,seed=seed,method=method,wall_seconds=timings[method],**graph_metrics(truth,pvals))
                        rows.append(row);stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    print(f'{size} {scenario} {rep+1}/{a.repetitions}',flush=True)
    df=pd.DataFrame(rows);df.to_csv(a.output/'repetitions.csv',index=False)
    out=[]
    for (scenario,size,method),g in df.groupby(['scenario','size','method']):
        row=dict(scenario=scenario,size=int(size),method=method,repetitions=len(g))
        for metric in ['precision','recall','f1','fdp','shd','wall_seconds']:
            mean=float(g[metric].mean());se=float(g[metric].std(ddof=1)/np.sqrt(len(g))) if len(g)>1 else np.nan
            row.update({f'mean_{metric}':mean,f'{metric}_mc_se':se,f'{metric}_ci95_low':mean-1.96*se,f'{metric}_ci95_high':mean+1.96*se})
        out.append(row)
    pd.DataFrame(out).to_csv(a.output/'summary.csv',index=False)
    design.update(elapsed_seconds=time.perf_counter()-start,peak_working_set_bytes=int(getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)),completed_utc=pd.Timestamp.now(tz='UTC').isoformat())
    (a.output/'manifest.json').write_text(json.dumps(design,indent=2),encoding='utf-8')


if __name__=='__main__':main()
