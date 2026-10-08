"""Failure-triggered exploratory calendar-block CR1/t sensitivity.

This does not overwrite or replace the frozen E1 OLS/HAC results.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import psutil
import yaml
from threadpoolctl import threadpool_limits,threadpool_info

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from causal_weathergraph.candidate_edges import build_candidate_edges
from causal_weathergraph.regimes import define_regimes
from revision.inference import run_graph_discovery
from revision.run_inference import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path(r'D:\Paper2\vipuser\论文2026_516\Causal\causal-weathergraph\outputs\processed\region_series.npz'))
    parser.add_argument('--output',type=Path,default=ROOT.parent/'revision_outputs'/'inference_exploratory_cluster')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter()
    protocol=dict(status='running',analysis_class='failure-triggered exploratory calibration sensitivity',
        trigger='Fixed HAC64 failed finite-sample all-null BH calibration at N=2048; exploratory method specified after that result, before cluster results.',
        started_utc=datetime.now(timezone.utc).isoformat(),
        primary_block_length_steps=128,sensitivity_block_lengths_steps=[64,256],
        rationale='128 is the already specified calendar-design bootstrap primary block length, not chosen from exploratory cluster outcomes.',
        covariance='CR1: G/(G-1)*(n-1)/(n-rank_full)*sum_g(S_g^2)/(z_residual^T z_residual)^2; S_g is sum of full-model FWL scores in calendar block g.',
        reference_distribution='Student t with G-1 df; G counts populated original-calendar blocks.',
        calendar='block label floor(original six-hour row index / block_length); no regime time compression; phase anchored at first dataset timestamp.',
        effect_model='Unchanged original own-target-lag3 model; each source lag tested separately.',
        tests=18018,alpha=.05,bh_families=['regime x target variable','all 18018 tests'],
        limitations=['Adjacent calendar blocks can remain dependent, so cluster independence is approximate.',
                     'CR1 and t(G-1) are finite-sample adjustments, not universal finite-sample guarantees.',
                     'Exploratory simulations must be reported even if they fail; no block-length choice based on best rejection rate.',
                     'Regime selection and omitted common drivers remain unaddressed by covariance changes.',
                     'This analysis does not supersede the frozen primary HAC results.'],
        sources=['https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_cluster.html',
                 'https://escholarship.org/uc/item/1jq5d0pq'])
    protocol['input']=str(args.input)
    protocol['input_sha256']=sha256(args.input)
    protocol['source_hashes']={str(p.relative_to(ROOT)):sha256(p) for p in
        [Path(__file__),ROOT/'revision'/'inference.py',ROOT/'configs'/'local_full_period_fast.yaml']}
    snapshots=args.output/'code_snapshot'
    for relative in protocol['source_hashes']:
        target=snapshots/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    (args.output/'protocol_amendment.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    config=yaml.safe_load((ROOT/'configs'/'local_full_period_fast.yaml').read_text())
    with np.load(args.input,allow_pickle=True) as archive:
        data=archive['data']; names=archive['variable_names'].tolist()
        lat,lon=archive['lat'],archive['lon']
        timestamps=pd.DatetimeIndex(archive['timestamps'])
    if not np.all(np.diff(timestamps.as_unit('ns').asi8)==pd.Timedelta(hours=6).value):
        raise ValueError('Six-hour complete calendar required')
    candidates=build_candidate_edges(names,lat,lon,config['causal'],data)
    regimes=define_regimes(data,names,timestamps,config['regimes'])
    peak=0
    process=psutil.Process()
    def progress(done,total,regime,region):
        nonlocal peak
        peak=max(peak,process.memory_info().rss)
        if done%66==0:
            print(json.dumps(dict(completed=done,total=total,regime=regime,elapsed_seconds=time.perf_counter()-began)),flush=True)
    with threadpool_limits(limits=2):
        protocol['threadpools']=threadpool_info()
        table=run_graph_discovery(data,names,candidates,regimes,max_lag=3,
            controls=config['causal']['controls'],bandwidths=(),cluster_lengths=(64,128,256),progress=progress)
    assert len(table)==18018
    table.to_csv(args.output/'all_tests_exploratory_cluster.csv',index=False)
    summaries=[]
    for regime,part in [('TOTAL',table)]+list(table.groupby('regime',sort=False)):
        for method in ('ols','cluster64','cluster128','cluster256'):
            for scope in ('within','global'):
                yes=part[f'significant_{method}_{scope}']
                selected=part[yes]
                summaries.append(dict(regime=regime,method=method,bh_family=scope,n_tests=len(part),
                    n_significant=int(yes.sum()),fraction_significant=float(yes.mean()),
                    median_abs_coefficient=float(selected.effect_abs.median()),
                    median_partial_r2=float(selected.partial_r2.median()),
                    median_delta_r2=float(selected.delta_r2.median())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(args.output/'significance_summary_exploratory.csv',index=False)
    original=pd.read_csv(args.output.parent/'inference'/'all_tests_ols_hac.csv')
    keys=['source_region','target_region','source_var','target_var','lag','regime']
    matched=table.merge(original,on=keys,validate='one_to_one',suffixes=('_cluster','_frozen'))
    assert np.max(np.abs(matched.effect_cluster-matched.effect_frozen))<1e-12
    assert np.array_equal(matched.significant_ols_within_cluster,matched.significant_ols_within_frozen)
    protocol.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-began,peak_sampled_rss_mb=peak/1024**2,
        actual_n_tests=len(table),valid_tests=int(table.valid.sum()),
        effect_matches_frozen=True,ols_significance_matches_frozen=True)
    protocol['output_sha256']={p.name:sha256(p) for p in args.output.glob('*.csv')}
    (args.output/'manifest.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(summary[summary.regime=='TOTAL'].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
