"""Reproduce 18,018 original tests and run fixed calendar-time HAC inference."""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='2'

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import psutil
import yaml
from threadpoolctl import threadpool_limits, threadpool_info

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
from causal_weathergraph.candidate_edges import build_candidate_edges
from causal_weathergraph.regimes import define_regimes
from revision.inference import run_graph_discovery, bootstrap_edge_effects


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path(r'D:\Paper2\vipuser\论文2026_516\Causal\causal-weathergraph\outputs\processed\region_series.npz'))
    parser.add_argument('--legacy-edges',type=Path,default=Path(r'D:\Paper2\vipuser\论文2026_516\Causal\causal-weathergraph\outputs\causal_edges\all_edges.csv'))
    parser.add_argument('--output',type=Path,default=ROOT.parent/'revision_outputs'/'inference')
    parser.add_argument('--bootstrap-replicates',type=int,default=200)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    clock=time.perf_counter()
    config=yaml.safe_load((ROOT/'configs'/'local_full_period_fast.yaml').read_text())
    with np.load(args.input,allow_pickle=True) as z:
        data=z['data']
        names=z['variable_names'].tolist()
        lat,lon=z['lat'],z['lon']
        timestamps=pd.DatetimeIndex(z['timestamps'])
    gaps=np.diff(timestamps.as_unit('ns').asi8)
    expected_gap=int(pd.Timedelta(hours=6).value)
    if not np.all(gaps==expected_gap):
        raise ValueError('Input timeline is not complete regular six-hour calendar; explicitly reindex before fitting')
    candidates=build_candidate_edges(names,lat,lon,config['causal'],data)
    regimes=define_regimes(data,names,timestamps,config['regimes'])
    assert len(candidates)==858 and len(regimes)==7
    manifest=dict(experiment='E1 calendar-time HAC',status='running',started_utc=started,
                  input=str(args.input),input_sha256=sha256(args.input),shape=list(data.shape),
                  time_first=str(timestamps[0]),time_last=str(timestamps[-1]),calendar_step_hours=6,
                  configuration=config,primary_hac_bandwidth_steps=64,sensitivity_hac_bandwidth_steps=[32,128],
                  bandwidth_selection='Fixed before inspecting results; 8,16,32 calendar days',
                  covariance='Bartlett score autocovariances on full calendar; excluded timestamps have zero scores',
                  covariance_correction='n_selected / (n_selected - rank_full)',
                  hac_test_distribution='asymptotic standard normal; two-sided',
                  ols_test_distribution='nested F with 1 numerator df and n-rank_full denominator df',
                  families={'within':'regime x target variable, all candidate region pairs and lags',
                            'global':'all 18,018 original tests'},
                  alpha=.05,regime_sizes={name:int(mask.sum()) for name,mask in regimes.items()},
                  n_candidate_edges=len(candidates),expected_n_tests=18018,
                  residual_acf='full source-added model, pair-normalized product using selected pairs at actual calendar lag; not compressed-index ACF',
                  limitations=['HAC adjusts serial/heteroskedastic covariance, not omitted-variable confounding or regime-selection bias.',
                               'State thresholds retain original full-period definitions for exact baseline comparison.',
                               'BH validity requires suitable dependence conditions; global BH is a separate family sensitivity, not a guaranteed arbitrary-dependence correction.',
                               'Effect coefficients refer to the processed region series and are not interventional effects.',
                               'Lag-specific models test one added source lag at a time, not a joint source-lag block.'],
                  software={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','statsmodels','psutil','threadpoolctl']},
                  python=sys.version,platform=platform.platform(),
                  source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in
                    [Path(__file__),ROOT/'revision'/'inference.py',ROOT/'configs'/'local_full_period_fast.yaml',
                     ROOT/'src'/'causal_weathergraph'/'regimes.py',ROOT/'src'/'causal_weathergraph'/'candidate_edges.py']},
                  method_sources=['https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html',
                                  'https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.fdrcorrection.html'])
    manifest['bootstrap_sensitivity']=dict(replicates=args.bootstrap_replicates,block_lengths=[64,128,256],
        selection='Fixed before effect inspection: region indices 0,22,44; wind->humidity and humidity->cloud_cover; lag1; all and high_humidity regimes.',
        interpretation='Limited 12-edge diagnostic of coefficient stability, not all-edge bootstrap or FDR calibration; percentile intervals are pointwise.',
        lag_handling='Construct all original calendar lags once; sample contiguous original design-row blocks; then select sampled regime mask.',
        thresholds='Original thresholds held fixed; conditional sensitivity only')
    snapshots=args.output/'code_snapshot'
    snapshots.mkdir(exist_ok=True)
    for relative in manifest['source_hashes']:
        target=snapshots/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    manifest_path=args.output/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    process=psutil.Process()
    peak=process.memory_info().rss
    def progress(done,total,regime,region):
        nonlocal peak
        peak=max(peak,process.memory_info().rss)
        if done%22==0 or done==total:
            line=dict(completed_target_groups=done,total_target_groups=total,regime=regime,
                      elapsed_seconds=round(time.perf_counter()-clock,2),rss_mb=round(process.memory_info().rss/1024**2,1))
            print(json.dumps(line),flush=True)
            (args.output/'progress.json').write_text(json.dumps(line,indent=2),encoding='utf-8')
    with threadpool_limits(limits=2):
        manifest['threadpools']=threadpool_info()
        results=run_graph_discovery(data,names,candidates,regimes,max_lag=3,
                    controls=config['causal']['controls'],bandwidths=(32,64,128),progress=progress)
    assert len(results)==18018
    results.to_csv(args.output/'all_tests_ols_hac.csv',index=False)
    primary=results[results.significant_hac64_within]
    primary.to_csv(args.output/'significant_hac64_within.csv',index=False)
    results[results.significant_hac64_global].to_csv(args.output/'significant_hac64_global.csv',index=False)
    summaries=[]
    for regime,part in [('TOTAL',results)]+list(results.groupby('regime',sort=False)):
        for method in ('ols','hac32','hac64','hac128'):
            for scope in ('within','global'):
                yes=part[f'significant_{method}_{scope}']
                selected=part[yes]
                summaries.append(dict(regime=regime,method=method,bh_family=scope,n_tests=len(part),
                    n_significant=int(yes.sum()),fraction_significant=float(yes.mean()),
                    median_abs_coefficient=float(selected.effect_abs.median()),
                    median_partial_r2=float(selected.partial_r2.median()),
                    median_delta_r2=float(selected.delta_r2.median())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(args.output/'significance_summary.csv',index=False)
    diagnostics=[]
    for regime,part in results.groupby('regime',sort=False):
        for lag in (1,4,12,28):
            val=part[f'residual_acf_{lag}']
            diagnostics.append(dict(regime=regime,calendar_lag_steps=lag,calendar_lag_hours=6*lag,
                n_models=len(val),median_acf=float(val.median()),median_abs_acf=float(val.abs().median()),
                p90_abs_acf=float(val.abs().quantile(.9)),fraction_abs_acf_above_01=float((val.abs()>.1).mean()),
                median_selected_pairs=float(part[f'residual_pairs_{lag}'].median())))
    pd.DataFrame(diagnostics).to_csv(args.output/'residual_calendar_acf_summary.csv',index=False)
    legacy=pd.read_csv(args.legacy_edges)
    keys=['regime','source_region','target_region','source_var','target_var','lag']
    joined=results.merge(legacy,on=keys,suffixes=('_new','_legacy'),validate='one_to_one')
    comparison=dict(legacy_input=str(args.legacy_edges),legacy_sha256=sha256(args.legacy_edges),
        matching_tests=len(joined),legacy_tests=len(legacy),new_tests=len(results),
        legacy_significant=int(legacy.significant.sum()),new_ols_significant=int(results.significant_ols_within.sum()),
        significance_disagreements=int((joined.significant_ols_within!=joined.significant).sum()),
        max_abs_coefficient_difference=float(np.max(np.abs(joined.effect_coefficient_new-joined.effect_coefficient_legacy))),
        max_abs_p_difference=float(np.max(np.abs(joined.p_ols-joined.p_value))),
        max_abs_q_difference=float(np.max(np.abs(joined.q_ols_within-joined.q_value))))
    (args.output/'baseline_reproduction.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    assert comparison['matching_tests']==18018
    legacy_input=Path(r'D:\Paper2\vipuser\论文2026_516\Causal\causal-weathergraph\outputs\processed\region_series.npz')
    if Path(args.input).resolve()==legacy_input.resolve():
        # Exact reproduction is required only for the legacy input; with the re-acquired 2023-2025 segment the
        # differences are the effect of the corrected data and are recorded in baseline_reproduction.json.
        assert comparison['significance_disagreements']==0
    if args.bootstrap_replicates:
        boot_summary=[]
        boot_rows=[]
        names_to_index={v:i for i,v in enumerate(names)}
        with threadpool_limits(limits=2):
            for regime in ('all','high_humidity'):
                for region in (0,22,44):
                    for source,target in (('wind','humidity'),('humidity','cloud_cover')):
                        ti,si=names_to_index[target],names_to_index[source]
                        y=data[3:,region,ti]
                        c=np.column_stack([data[3-lag:len(data)-lag,region,ti] for lag in (1,2,3)])
                        z=data[2:-1,region,si]
                        for length in (64,128,256):
                            boot=bootstrap_edge_effects(y,c,z,regimes[regime][3:],length,args.bootstrap_replicates,seed=20260929+length)
                            metadata=dict(regime=regime,source_region=region,target_region=region,
                                          source_var=source,target_var=target,lag=1,block_length_steps=length)
                            boot_summary.append(dict(**metadata,replicates=args.bootstrap_replicates,
                                n_valid=boot['n_valid'],ci95_low=boot['ci95_low'],ci95_high=boot['ci95_high'],
                                median_coefficient=float(np.nanmedian(boot['effects'])),
                                positive_fraction=float(np.nanmean(boot['effects']>0))))
                            boot_rows.extend(dict(**metadata,replicate=j+1,effect=float(beta),n_samples=int(n))
                                for j,(beta,n) in enumerate(zip(boot['effects'],boot['sample_sizes'])))
                        print(f'Bootstrap fixed diagnostic: {regime} region={region} {source}->{target}',flush=True)
        pd.DataFrame(boot_summary).to_csv(args.output/'fixed_edge_calendar_bootstrap_summary.csv',index=False)
        pd.DataFrame(boot_rows).to_csv(args.output/'fixed_edge_calendar_bootstrap_replicates.csv',index=False)
    manifest.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter()-clock,peak_sampled_rss_mb=peak/1024**2,
                    actual_n_tests=len(results),valid_tests=int(results.valid.sum()),baseline_reproduction=comparison)
    manifest['output_sha256']={p.name:sha256(p) for p in args.output.glob('*.csv')}
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary[summary.regime=='TOTAL'].to_string(index=False),flush=True)
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':
    main()
