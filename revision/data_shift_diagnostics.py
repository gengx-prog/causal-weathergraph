"""Read-only descriptive diagnostics after detecting regime-distribution drift.

This script reads saved train-fitted and physical regional data, retains the
existing training thresholds, and changes no model, input, threshold or result.
Spatial summaries are equal-weight means over 66 regions, not area-weighted
planetary averages. Boundary-window comparisons are descriptive, not proof of
an ingestion defect or a causal source discontinuity.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revision.prepare_inputs import sha256


def summarize(values,selection):
    selected=np.asarray(values)[selection]
    return dict(n=int(len(selected)),mean=float(selected.mean()),variance=float(selected.var(ddof=1)),
                standard_deviation=float(selected.std(ddof=1)),minimum=float(selected.min()),
                maximum=float(selected.max()),q10=float(np.quantile(selected,.1)),
                median=float(np.median(selected)),q90=float(np.quantile(selected,.9)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,default=ROOT.parent/'revision_outputs'/'inputs')
    parser.add_argument('--output',type=Path,default=ROOT.parent/'revision_outputs'/'data_shift_diagnostics')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    input_paths=[args.inputs/name for name in ('region_trainfit.npz','region_trainfit_vectors.npz','data_manifest.json')]
    source_manifest=json.loads((args.inputs/'data_manifest.json').read_text(encoding='utf-8'))
    input_hashes={p.name:sha256(p) for p in input_paths}
    for name in ('region_trainfit.npz','region_trainfit_vectors.npz'):
        if input_hashes[name]!=source_manifest['outputs'][name]:
            raise ValueError(f'Input differs from recorded preprocessing manifest: {name}')
    with np.load(args.inputs/'region_trainfit.npz',allow_pickle=False) as archive:
        original=archive['data']; core_names=archive['variable_names'].tolist()
        timestamps=pd.DatetimeIndex(archive['timestamps']); lat=archive['lat'];lon=archive['lon']
    with np.load(args.inputs/'region_trainfit_vectors.npz',allow_pickle=False) as archive:
        processed=archive['data']; physical=archive['physical']; names=archive['variable_names'].tolist()
        if not np.array_equal(archive['timestamps'],timestamps.values):
            raise ValueError('Vector timestamp mismatch')
        if not np.array_equal(archive['lat'],lat) or not np.array_equal(archive['lon'],lon):
            raise ValueError('Vector region coordinates mismatch')
    core_indices=[names.index(name) for name in core_names]
    core_max_difference=float(np.max(np.abs(original-processed[:,:,core_indices])))
    if core_max_difference!=0:
        raise ValueError('Vector core variables differ from trainfit regional input')
    del original
    if not np.isfinite(processed).all() or not np.isfinite(physical).all():
        raise ValueError('Nonfinite saved input')
    gaps=np.diff(timestamps.as_unit('ns').asi8)
    if not np.all(gaps==pd.Timedelta(hours=6).value):
        raise ValueError('Timeline contains gap, duplicate or irregular step')
    train=np.asarray(timestamps<pd.Timestamp('2019-01-01'))
    test=~train
    global_series={'processed':processed.mean(axis=1),'physical':physical.mean(axis=1)}
    indices={name:names.index(name) for name in names}
    state={}
    thresholds={}
    for variable in ('humidity','cloud_cover'):
        values=global_series['processed'][:,indices[variable]]
        low,high=np.quantile(values[train],[.2,.8])
        thresholds[variable]={'q20':float(low),'q80':float(high)}
        state[variable]=values>=high
    regime_design=ROOT.parent/'revision_outputs'/'regime_contrasts'/'design.json'
    threshold_comparison={}
    if regime_design.exists():
        existing=json.loads(regime_design.read_text())['thresholds_training_only']
        for name,reference in [('humidity','humidity'),('cloud_cover','cloud')]:
            for quantile in ('q20','q80'):
                difference=abs(thresholds[name][quantile]-existing[reference][quantile])
                threshold_comparison[f'{name}_{quantile}_absolute_difference']=float(difference)
                if difference>1e-12:
                    raise ValueError('Threshold differs from existing E3 definition')
    units={'temperature':'K','humidity':'kg kg**-1','wind':'m s**-1','cloud_cover':'(0 - 1)',
           'u':'m s**-1','v':'m s**-1','qu':'kg kg**-1 m s**-1','qv':'kg kg**-1 m s**-1'}
    periods={
        'training_1979_2018':train,
        'evaluation_2019_2025':test,
        'source1_weatherbench2_1979_2018':train,
        'source2_weatherbench2_2019_2023jan10':np.asarray((timestamps>=pd.Timestamp('2019-01-01'))&(timestamps<pd.Timestamp('2023-01-11'))),
        'source3_cds_2023jan11_2025':np.asarray(timestamps>=pd.Timestamp('2023-01-11')),
    }
    source_labels=np.where(train,'source1_weatherbench2',np.where(timestamps<pd.Timestamp('2023-01-11'),'source2_weatherbench2','source3_cds'))
    timeline=pd.DataFrame({'timestamp':timestamps.astype(str),'source_segment':source_labels,
        'high_humidity_training_threshold':state['humidity'],'high_cloud_training_threshold':state['cloud_cover']})
    for representation,values in global_series.items():
        for variable in names:
            timeline[f'{representation}_{variable}_equal_region_mean']=values[:,indices[variable]]
    timeline.to_csv(args.output/'global_regional_mean_timeseries.csv',index=False)
    annual=[]
    for year in np.unique(timestamps.year):
        selection=timestamps.year==year
        for representation,values in global_series.items():
            for variable in names:
                row=dict(year=int(year),representation=representation,variable=variable,
                    units=units[variable] if representation=='physical' else 'grid-standardized monthly-anomaly units',
                    **summarize(values[:,indices[variable]],selection))
                row['high_state_fraction_using_processed_training_threshold']=float(state[variable][selection].mean()) if variable in state else np.nan
                row['high_humidity_fraction']=float(state['humidity'][selection].mean())
                row['high_cloud_fraction']=float(state['cloud_cover'][selection].mean())
                annual.append(row)
    pd.DataFrame(annual).to_csv(args.output/'annual_global_regional_mean_statistics.csv',index=False)
    period_rows=[]
    occupancy=[]
    for label,selection in periods.items():
        for variable in state:
            occupancy.append(dict(period=label,variable=variable,n=int(selection.sum()),
                n_high=int(state[variable][selection].sum()),high_fraction=float(state[variable][selection].mean()),
                threshold_q80_processed=thresholds[variable]['q80']))
        for representation,values in global_series.items():
            for variable in names:
                period_rows.append(dict(period=label,representation=representation,variable=variable,
                    units=units[variable] if representation=='physical' else 'grid-standardized monthly-anomaly units',
                    **summarize(values[:,indices[variable]],selection)))
    pd.DataFrame(period_rows).to_csv(args.output/'period_source_global_regional_mean_statistics.csv',index=False)
    occupancy_table=pd.DataFrame(occupancy)
    occupancy_table.to_csv(args.output/'period_source_high_state_occupancy.csv',index=False)
    regional=[]
    for year in np.unique(timestamps.year):
        selection=timestamps.year==year
        for representation,array in [('processed',processed),('physical',physical)]:
            for variable in ('humidity','cloud_cover'):
                values=array[selection,:,indices[variable]]
                means,variances=values.mean(axis=0),values.var(axis=0,ddof=1)
                for region in range(len(lat)):
                    regional.append(dict(year=int(year),representation=representation,variable=variable,
                        region=region,latitude=float(lat[region]),longitude=float(lon[region]),n=int(selection.sum()),
                        mean=float(means[region]),variance=float(variances[region])))
    pd.DataFrame(regional).to_csv(args.output/'annual_region_humidity_cloud_statistics.csv',index=False)
    boundaries=[]
    jumps=[]
    for date in ('2019-01-01','2023-01-11'):
        boundary=pd.Timestamp(date)
        position=int(np.searchsorted(timestamps.asi8,boundary.value))
        assert timestamps[position]==boundary
        for representation,values in global_series.items():
            for variable in names:
                series=values[:,indices[variable]]
                training_sd=series[train].std(ddof=1)
                step=series[position]-series[position-1]
                historical_differences=np.diff(series)[train[1:]&train[:-1]]
                # January-to-January adjacent six-hour differences provide a
                # descriptive seasonal reference for these January seams.
                january=(timestamps.month[1:]==1)&(timestamps.month[:-1]==1)&train[1:]&train[:-1]
                january_differences=np.diff(series)[january]
                same_transition=(timestamps.month[1:]==boundary.month)&(timestamps.day[1:]==boundary.day)&(timestamps.hour[1:]==boundary.hour)&train[1:]&train[:-1]
                same_transition_differences=np.diff(series)[same_transition]
                jumps.append(dict(boundary=date,representation=representation,variable=variable,
                    before_timestamp=str(timestamps[position-1]),after_timestamp=str(timestamps[position]),
                    before_value=float(series[position-1]),after_value=float(series[position]),
                    step_change=float(step),step_change_over_training_sd=float(step/training_sd),
                    abs_step_empirical_percentile_training=float(np.mean(np.abs(historical_differences)<=abs(step))),
                    abs_step_empirical_percentile_training_january=float(np.mean(np.abs(january_differences)<=abs(step))),
                    crosses_month_boundary=bool(timestamps[position-1].month!=boundary.month),
                    same_calendar_transition_reference_n=len(same_transition_differences),
                    abs_step_empirical_percentile_same_calendar_transition=float(np.mean(np.abs(same_transition_differences)<=abs(step)))))
                for days in (30,365):
                    before=np.asarray((timestamps>=boundary-pd.Timedelta(days=days))&(timestamps<boundary))
                    after=np.asarray((timestamps>=boundary)&(timestamps<boundary+pd.Timedelta(days=days)))
                    before_stats,after_stats=summarize(series,before),summarize(series,after)
                    row=dict(boundary=date,window_days=days,representation=representation,variable=variable,
                        pre_n=before_stats['n'],post_n=after_stats['n'],pre_mean=before_stats['mean'],post_mean=after_stats['mean'],
                        mean_change=after_stats['mean']-before_stats['mean'],
                        mean_change_over_training_sd=(after_stats['mean']-before_stats['mean'])/training_sd,
                        pre_variance=before_stats['variance'],post_variance=after_stats['variance'],
                        variance_ratio_post_pre=after_stats['variance']/before_stats['variance'])
                    if variable in state:
                        row.update(pre_high_fraction=float(state[variable][before].mean()),post_high_fraction=float(state[variable][after].mean()))
                    boundaries.append(row)
    pd.DataFrame(boundaries).to_csv(args.output/'source_boundary_windows.csv',index=False)
    pd.DataFrame(jumps).to_csv(args.output/'source_boundary_adjacent_steps.csv',index=False)
    unit_rows=[]
    units_consistent={}
    for variable in sorted({row['variable'] for row in source_manifest['sources']}):
        sources=[row for row in source_manifest['sources'] if row['variable']==variable]
        units_consistent[variable]=len({row['units'] for row in sources})==1
        for source in sources:
            unit_rows.append(dict(variable=variable,units=source['units'],path=source['path'],
                first=source['first'],last=source['last'],n_timestamps=source['n_timestamps'],
                latitude_max_abs_difference=source['latitude_max_abs_difference'],
                units_consistent_across_segments=units_consistent[variable]))
    pd.DataFrame(unit_rows).to_csv(args.output/'source_units_and_boundaries_from_manifest.csv',index=False)
    annual_table=pd.DataFrame(annual)
    h_annual=annual_table[(annual_table.variable=='humidity')&(annual_table.representation=='processed')]
    c_annual=annual_table[(annual_table.variable=='cloud_cover')&(annual_table.representation=='processed')]
    state_examples={str(year):float(h_annual.loc[h_annual.year==year,'high_humidity_fraction'].iloc[0]) for year in (2016,2018,2019,2020,2023,2024,2025)}
    physical_humidity_train=global_series['physical'][train,indices['humidity']].mean()
    physical_humidity_test=global_series['physical'][test,indices['humidity']].mean()
    humidity_seams=[row for row in jumps if row['variable']=='humidity' and row['representation']=='physical']
    summary=dict(analysis_class='Post-diagnostic descriptive audit after high-state distribution drift was observed',
        completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,
        no_model_refit=True,no_threshold_change=True,no_input_change=True,
        input_sha256=input_hashes,code_sha256=sha256(__file__),
        spatial_aggregation='Equal-weight arithmetic mean of 66 regional means; not area-weighted global atmospheric mean.',
        processed_definition='Original grid-specific monthly climatology and scaling fitted on1979-2018, aggregated to66regions.',
        physical_definition='Original regional means in declared source units, before trainfit anomaly/scaling.',
        thresholds_training_only=thresholds,threshold_match_existing_e3=threshold_comparison,
        integrity=dict(n_timestamps=len(timestamps),n_regions=len(lat),n_variables=len(names),
            first=str(timestamps[0]),last=str(timestamps[-1]),six_hour_calendar_complete=True,
            nonfinite_values=0,vector_core_max_abs_difference=core_max_difference,
            source_declared_units_consistent=units_consistent,
            physical_humidity_min=float(physical[:,:,indices['humidity']].min()),
            physical_humidity_max=float(physical[:,:,indices['humidity']].max()),
            physical_cloud_min=float(physical[:,:,indices['cloud_cover']].min()),
            physical_cloud_max=float(physical[:,:,indices['cloud_cover']].max())),
        occupancy=occupancy,
        descriptive_findings=dict(
            humidity_high_fraction_examples=state_examples,
            humidity_physical_mean_training_kg_per_kg=float(physical_humidity_train),
            humidity_physical_mean_evaluation_kg_per_kg=float(physical_humidity_test),
            humidity_physical_mean_relative_change=float(physical_humidity_test/physical_humidity_train-1),
            humidity_physical_boundary_steps=humidity_seams,
            pre_2019_drift_evidence='High-humidity occupancy was already elevated in2016 and2018, before the2019 source boundary; the evaluation shift is not uniquely aligned to that boundary.',
            boundary_caution='The2023 humidity step is unusually large relative to historical January six-hour steps and is negative; this warrants retaining source-comparability uncertainty, not attributing the longer-run increase to a positive unit step.'),
        annual_high_state_range={
            'humidity':dict(min_fraction=float(h_annual.high_humidity_fraction.min()),max_fraction=float(h_annual.high_humidity_fraction.max()),
                min_year=int(h_annual.loc[h_annual.high_humidity_fraction.idxmin(),'year']),max_year=int(h_annual.loc[h_annual.high_humidity_fraction.idxmax(),'year'])),
            'cloud':dict(min_fraction=float(c_annual.high_cloud_fraction.min()),max_fraction=float(c_annual.high_cloud_fraction.max()),
                min_year=int(c_annual.loc[c_annual.high_cloud_fraction.idxmin(),'year']),max_year=int(c_annual.loc[c_annual.high_cloud_fraction.idxmax(),'year']))},
        interpretation=[
            'High-state occupancy under fixed historical thresholds can change under genuine distribution shift; 20% training occupancy does not imply20% evaluation occupancy.',
            'Consistent unit metadata and plausible ranges provide no evidence of a simple declared-unit change, but do not prove source homogenization.',
            'Source changes, weather variability, climate trends and preprocessing baseline effects are confounded in these descriptive comparisons.',
            'Boundary-window contrasts and adjacent-step empirical ranks are descriptive; no formal discontinuity hypothesis test or source-error classification was performed.',
            'January1 also changes the monthly-climatology subtraction; processed-series step ranks against within-January steps are not directly comparable. Same calendar-day/hour transition ranks are provided with their small reference counts.',
            'The consequence is a changed regime population and weak comparability of high-state counts across periods; mechanism attribution remains unresolved.'])
    summary['output_sha256']={p.name:sha256(p) for p in args.output.glob('*.csv')}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (args.output/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(occupancy_table.to_string(index=False))
    print(pd.DataFrame(boundaries).query('representation=="processed" and variable in ["humidity","cloud_cover"]')[
        ['boundary','window_days','variable','mean_change_over_training_sd','pre_high_fraction','post_high_fraction']].to_string(index=False))


if __name__=='__main__':main()
