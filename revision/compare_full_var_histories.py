"""Compare dense VAR history3/history12 on identical target calendar rows."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revision.run_inference import sha256
from revision.run_full_var import KEY


def main():
    outputs=ROOT.parent/'revision_outputs'
    paths={3:outputs/'full_var_lag3_aligned12',12:outputs/'full_var_lag12'}
    target=outputs/'full_var_history_comparison'
    target.mkdir(exist_ok=True)
    rows=[]
    diagnostics=[]
    manifests={h:json.loads((p/'manifest.json').read_text()) for h,p in paths.items()}
    for period in ('discovery','evaluation'):
        assert manifests[3]['fit_diagnostics'][period]['n_samples']==manifests[12]['fit_diagnostics'][period]['n_samples']
    for history,path in paths.items():
        combined=pd.read_csv(path/'full_var_holdout_all_candidates.csv')
        for edge_type,g in [('ALL',combined)]+list(combined.groupby('edge_type')):
            rows.append(dict(history_lags=history,source_lags=3,edge_type=edge_type,n=len(g),
                train_hac_q05=int((g.train_q_hac64_global<.05).sum()),test_hac_q05=int((g.test_q_hac64_global<.05).sum()),
                train_cluster_q05=int((g.train_q_cluster128_global<.05).sum()),test_cluster_q05=int((g.test_q_cluster128_global<.05).sum()),
                frozen_positive_gain=int((g.mse_gain>0).sum()),
                frozen_positive_gain_hac_q05=int(((g.mse_gain>0)&(g.q_predictive_gain_global<.05)).sum()),
                median_frozen_relative_mse_gain=float(g.mean_relative_mse_gain.median())))
        acf=pd.read_csv(path/'full_var_residual_calendar_acf_summary.csv')
        variances=pd.read_csv(path/'full_var_residual_variance_ratios.csv')
        for period,fit in manifests[history]['fit_diagnostics'].items():
            d=dict(history_lags=history,period=period,**fit)
            for lag in (1,4,12,28):
                d[f'median_residual_acf_lag{lag}']=float(acf[(acf.period==period)&(acf.calendar_lag_steps==lag)]['median'].iloc[0])
            for grouping in ('calendar_month','calendar_year'):
                chosen=variances[(variances.period==period)&(variances.grouping==grouping)]
                d[f'median_{grouping}_variance_max_min_ratio']=float(chosen.variance_max_min_ratio.median())
                d[f'median_{grouping}_variance_p90_p10_ratio']=float(chosen.variance_p90_p10_ratio.median())
            diagnostics.append(d)
    pd.DataFrame(rows).to_csv(target/'edge_inference_and_prediction.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(target/'fit_residual_and_heteroskedasticity_summary.csv',index=False)
    perf={h:pd.read_csv(p/'frozen_model_performance.csv') for h,p in paths.items()}
    paired=perf[3].merge(perf[12],on=['target_region','target_var'],validate='one_to_one',suffixes=('_lag3','_lag12'))
    paired['relative_mse_improvement_lag12_vs_lag3']=1-paired.heldout_mse_frozen_full_var_lag12/paired.heldout_mse_frozen_full_var_lag3
    paired.to_csv(target/'frozen_whole_model_prediction_by_target.csv',index=False)
    summary=[]
    for name,g in [('ALL',paired)]+list(paired.groupby('target_var')):
        summary.append(dict(target_var=name,n_targets=len(g),lag12_improves_count=int((g.relative_mse_improvement_lag12_vs_lag3>0).sum()),
            median_relative_mse_improvement=float(g.relative_mse_improvement_lag12_vs_lag3.median()),
            min_relative_mse_improvement=float(g.relative_mse_improvement_lag12_vs_lag3.min()),
            max_relative_mse_improvement=float(g.relative_mse_improvement_lag12_vs_lag3.max())))
    pd.DataFrame(summary).to_csv(target/'frozen_whole_model_prediction_summary.csv',index=False)
    edges={h:pd.read_csv(p/'full_var_holdout_all_candidates.csv') for h,p in paths.items()}
    same=edges[3].merge(edges[12],on=KEY,validate='one_to_one',suffixes=('_lag3','_lag12'))
    concordance=dict(n_tests=len(same),train_same_sign=int((np.sign(same.train_effect_lag3)==np.sign(same.train_effect_lag12)).sum()),
        test_same_sign=int((np.sign(same.test_effect_lag3)==np.sign(same.test_effect_lag12)).sum()),
        source_lags='1..3 in both models',target_timestamps='identical t>=12; same 2019 boundary',
        note='Adjusted-p threshold counts are sensitivity summaries; simulated FDR calibration failures remain unresolved.')
    (target/'comparison_checks.json').write_text(json.dumps(concordance,indent=2),encoding='utf-8')
    provenance=dict(code_sha256=sha256(__file__),inputs={str(p/'manifest.json'):sha256(p/'manifest.json') for p in paths.values()},
        outputs={p.name:sha256(p) for p in target.glob('*.csv')},interpretation=concordance['note'])
    (target/'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).query('edge_type == "ALL"').to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(diagnostics).to_string(index=False))


if __name__=='__main__':main()
