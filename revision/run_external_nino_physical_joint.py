"""Frozen external-month-state contrasts with antecedent physical controls.

All fits are retrospective sensitivity analyses. Group OLS is used for speed;
contrast influence scores are combined on the original six-hour calendar before
HAC, preserving cross-state covariance. No input preprocessing is refitted.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import psutil
from scipy import linalg, stats
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from revision.inference import score_autocovariances
from revision.run_external_nino_regimes import (
    BASE, INPUT, PAIRS, MONTHLY, CONTEXT, INDEX_MANIFEST, BANDWIDTHS,
    CONTRASTS, PERIODS, LIMITS, load_index, thresholds, digest, save_json, now,
)

PHYSICAL_DIR=BASE/"revision_outputs/physical_controls_inputs"
PHYSICAL=PHYSICAL_DIR/"region_controls_trainfit.npz"
PRIOR=BASE/"revision_outputs/external_nino_regimes/frozen_plan.json"
MODELS=("own_history_same_sample","own_history_plus_physical_tminus4_6")
L=6
EXTRA_LIMITS=[
    "Both models use the same six-step initial cutoff and identical state/period samples; the baseline is not silently copied from an experiment with a different cutoff.",
    "Physical controls are six fields at t-4/t-5/t-6 six-hour steps, for unique source/target regions. They precede source lags t-1/t-2/t-3, but temporal ordering alone does not establish a sufficient adjustment set.",
    "Each state has separate intercept, own-history, physical-control and source coefficients. A physical-control-adjusted change in a state contrast is sensitivity evidence, not proof of mediation, mechanism or causal identification.",
    "Surface and sea-level pressures and adjacent physical lags can be nearly collinear. Rank, raw and column-normalized condition numbers, source residual fractions and VIF are retained; unstable or unidentifiable results are flagged rather than selected away.",
    "The antecedent monthly state precedes all tested source lags but may overlap some older physical-control observations at a month boundary. It is not asserted to be exogenous to all model history.",
    "Source-segment population counts are descriptive. Models are not additionally fitted separately by acquisition segment in this experiment; source homogeneity is not established.",
]


def freeze(output):
    output.mkdir(parents=True,exist_ok=True)
    path=output/"frozen_plan.json"
    if path.exists():raise FileExistsError("Frozen joint plan already exists")
    prior=json.loads(PRIOR.read_text(encoding="utf-8"))
    preprocessing=json.loads((PHYSICAL_DIR/"preprocessing_plan.json").read_text(encoding="utf-8"))
    pm=json.loads((PHYSICAL_DIR/"data_manifest.json").read_text(encoding="utf-8"))
    if pm["status"]!="completed":raise ValueError("Physical preprocessing incomplete")
    if digest(PHYSICAL)!=pm["outputs"][PHYSICAL.name]["sha256"]:raise ValueError("Physical file differs from verified manifest")
    if preprocessing["definition"]["fit_period"] != ["1979-01-01T00","2018-12-31T18"]:raise ValueError("Unexpected physical fit period")
    native,_=load_index()
    if not np.array_equal(thresholds(native),[prior["thresholds"]["q20"],prior["thresholds"]["q80"]]):raise ValueError("Prior thresholds changed")
    paths=[INPUT,PAIRS,MONTHLY,CONTEXT,INDEX_MANIFEST,PHYSICAL,PRIOR,PHYSICAL_DIR/"preprocessing_plan.json",
           PHYSICAL_DIR/"data_manifest.json",PHYSICAL_DIR/"validation.json",PHYSICAL_DIR/"trainfit_parameters.npz",
           Path(__file__),Path(__file__).with_name("inference.py"),Path(__file__).with_name("run_external_nino_regimes.py")]
    plan={"frozen_utc":now(),"status":"frozen_before_joint_result_fits","inputs":[{"path":str(p),"sha256":digest(p),"bytes":p.stat().st_size} for p in paths],
          "prior_state_design_sha256":digest(PRIOR),"thresholds":prior["thresholds"],"state_assignment":prior["state_assignment"],
          "state_boundaries":prior["state_boundaries"],"native_index_units":"degree_C_anomaly","models":list(MODELS),
          "source_pairs":36,"source_lags_six_hour_steps":[1,2,3],"physical_control_lags_six_hour_steps":[4,5,6],
          "physical_fields":list(preprocessing["definition"]["variables"]),"control_regions":"sorted unique source_region and target_region; same-region pairs include that region once",
          "own_history_lags":[1,2,3],"initial_calendar_cutoff_rows":L,"periods":prior["periods"],"contrasts":[x[0] for x in CONTRASTS],
          "hac_bandwidths_six_hour_rows":list(BANDWIDTHS),"expected_results":2592,
          "model_definition":"State-specific intercept, target own lags 1-3, all physical controls when present, and tested source slope. Separate source-lag fits; every baseline and physical fit uses the same eligible rows.",
          "estimation":"Rank-aware pivoted QR residualization separately by low/middle/high state. Contrast beta_extreme-beta_middle. Normalized source influence scores extreme minus middle are combined on the full six-hour calendar; excluded rows remain zero. Bartlett HAC applied to combined scores preserves cross-state terms.",
          "finite_sample_correction":"For each contrast, n_selected/(n_selected-rank_middle_full-rank_extreme_full); not separate per-state variance corrections.",
          "numerical_diagnostics":{"rank_tolerance":"max(n,p)*float64_eps*largest_pivoted_R_diagonal","near_collinear_normalized_condition_warning":1e8,"source_residual_fraction_warning":1e-8,"unidentifiable_source_residual_sum_squares":"<= 1e-12 or nonfinite","unstable_policy":"retain flagged results; no outcome-driven dropping, regularization or bandwidth selection"},
          "verification":"Pairs with minimum and maximum pair_id, all three state groups and both periods/models/three lags: independently solve full group designs with SVD least squares. First pair, lag1, all contrast/period/model combinations: form full saturated interaction design with pseudoinverse; verify source contrast and direct calendar Bartlett HAC for all bandwidths.",
          "training_isolation":"Reuse existing training-fitted physical/original arrays and their verified hashes; fit no new climatology/scaling. Reuse original 480-native-month thresholds, with evaluation-index perturbation check. Both period-specific regressions remain retrospective.",
          "reporting":"All positive, negative, null and flagged fits retained; no BH/significance filter. Report matched baseline-versus-physical contrast changes and original-source segment sample/month/episode counts.",
          "limitations":LIMITS+EXTRA_LIMITS}
    save_json(path,plan)
    print(json.dumps({"plan":str(path),"sha256":digest(path),"frozen_utc":plan["frozen_utc"]}),flush=True)


def group_fit(y, controls, sources, mask):
    selected=np.flatnonzero(mask)
    base=np.column_stack([np.ones(len(selected)),controls[mask]])
    yy=y[mask]; xx=sources[mask]
    q,r,piv=linalg.qr(base,mode="economic",pivoting=True,check_finite=False)
    diagonal=np.abs(np.diag(r)); tol=max(base.shape)*np.finfo(float).eps*diagonal.max()
    rank=int(np.count_nonzero(diagonal>tol)); q=q[:,:rank]
    values=np.column_stack([yy,xx]); residual=values-q@(q.T@values)
    yr,z=residual[:,0],residual[:,1:]
    zz=np.sum(z*z,axis=0); valid=np.isfinite(zz)&(zz>1e-12)
    beta=np.divide(yr@z,zz,out=np.full(xx.shape[1],np.nan),where=valid)
    error=yr[:,None]-z*beta
    influence=np.zeros_like(sources)
    influence[mask]=np.divide(z*error,zz,out=np.full_like(z,np.nan),where=valid)
    sst=np.sum((xx-xx.mean(axis=0))**2,axis=0)
    fraction=np.divide(zz,sst,out=np.zeros_like(zz),where=sst>0)
    # R has the same singular values as the original base. Column normalization
    # here is only a dimensionless diagnostic, never a learned transformation.
    singular=linalg.svdvals(r,check_finite=False)
    norm=np.linalg.norm(base,axis=0)
    normalized=linalg.svdvals(r/np.where(norm[piv]>0,norm[piv],1)[None,:],check_finite=False)
    raw_cond=float(singular[0]/singular[-1]) if singular[-1]>0 else float("inf")
    normalized_cond=float(normalized[0]/normalized[-1]) if normalized[-1]>0 else float("inf")
    return {"beta":beta,"influence":influence,"rank_controls":rank,"rank_full":rank+1,"n":len(selected),
            "n_columns_controls":base.shape[1],"control_rank_deficient":rank<base.shape[1],"control_condition_raw":raw_cond,
            "control_condition_column_normalized":normalized_cond,"source_residual_fraction":fraction,
            "source_vif":np.divide(1,fraction,out=np.full_like(fraction,np.inf),where=fraction>0),"valid":valid,
            "mask":mask,"base":base,"selected_y":yy,"selected_sources":xx}


def population(mask,months):
    return {"six_hour_rows":int(mask.sum()),"native_months":int(pd.Index(months[mask]).nunique()),
            "state_episodes":int((mask & ~np.r_[False,mask[:-1]]).sum())}


def build_states(ts,plan):
    native,context=load_index(); low=plan["thresholds"]["q20"];high=plan["thresholds"]["q80"]
    changed=native.copy();changed.loc[changed.date>="2019-01-01","value"]=1e8
    if not np.array_equal(thresholds(changed),[low,high]):raise ValueError("Evaluation index affects thresholds")
    table=pd.concat([context,native],ignore_index=True)
    lookup=pd.Series(table.value.to_numpy(),index=table.date.dt.to_period("M"))
    month=(ts-pd.Timedelta(hours=24)).to_period("M")-1
    values=lookup.reindex(month).to_numpy();states=np.full(len(ts),2,dtype=np.int8)
    states[np.isfinite(values)]=0;states[values<=low]=-1;states[values>=high]=1
    ok=np.isfinite(values)
    if not np.all(month.end_time[ok]<ts[ok]-pd.Timedelta(hours=18)):raise ValueError("State does not precede source")
    return states,month.astype(str).to_numpy(),values


def full_design_audit(y,controls,source,reference_mask,extreme_mask,delta,se_by_bw):
    mask=reference_mask|extreme_mask; state=extreme_mask[mask]
    base=np.column_stack([np.ones(int(mask.sum())),controls[mask],source[mask]])
    design=np.column_stack([base*(~state)[:,None],base*state[:,None]])
    inverse=np.linalg.pinv(design,rcond=max(design.shape)*np.finfo(float).eps)
    coef=inverse@y[mask]
    residual=y[mask]-design@coef
    k=base.shape[1]; contrast=float(coef[-1]-coef[k-1])
    score=np.zeros(len(y));score[mask]=(inverse[-1]-inverse[k-1])*residual
    rank=np.linalg.matrix_rank(design)
    correction=int(mask.sum())/(int(mask.sum())-rank)
    products=np.array([float(np.sum(score[h:]*score[:len(score)-h])) for h in range(max(BANDWIDTHS)+1)])
    records=[]
    for b in BANDWIDTHS:
        h=np.arange(1,b+1);meat=products[0]+2*np.sum((1-h/(b+1))*products[1:b+1])
        direct=float(np.sqrt(max(0,meat*correction)))
        records.append({"bandwidth":b,"delta_full_svd":contrast,"delta_group_fwl":float(delta),"delta_abs_difference":abs(contrast-delta),
                        "se_direct_full_svd":direct,"se_group_fwl_fft":float(se_by_bw[b]),"se_abs_difference":abs(direct-se_by_bw[b]),
                        "excluded_calendar_score_zero":bool(np.all(score[~mask]==0)),"full_svd_rank":int(rank)})
    return records


def run(output):
    started=now(); start=time.perf_counter();plan_file=output/"frozen_plan.json"
    plan=json.loads(plan_file.read_text(encoding="utf-8"))
    if (output/"manifest.json").exists():raise FileExistsError("Completed output exists")
    for item in plan["inputs"]:
        if digest(item["path"])!=item["sha256"]:raise ValueError("Frozen input changed: "+item["path"])
    with np.load(INPUT) as z:
        data=z["data"];names=z["variable_names"].tolist();ts=pd.DatetimeIndex(z["timestamps"]);lat=z["lat"];lon=z["lon"]
    with np.load(PHYSICAL) as z:
        physical=z["data"];phys_names=z["variable_names"].tolist();segments=z["source_segment"];segment_names=z["source_segment_names"].tolist()
        if not np.array_equal(z["timestamps"],ts.values) or not np.array_equal(z["lat"],lat) or not np.array_equal(z["lon"],lon):raise ValueError("Coordinate mismatch")
        if not np.array_equal(z["training_mask"],ts<"2019-01-01"):raise ValueError("Training mask mismatch")
    if physical.shape!=(len(ts),66,6) or phys_names!=plan["physical_fields"]:raise ValueError("Wrong physical field shape/order")
    if not np.isfinite(data).all() or not np.isfinite(physical).all() or np.any(np.diff(ts.values)!=np.timedelta64(6,"h")):raise ValueError("Input missingness/calendar mismatch")
    states,months,index_value=build_states(ts,plan)
    state_table=pd.DataFrame({"target_time_utc":ts,"antecedent_native_month":months,"nino34_value_degC":index_value,
                              "state_code":states,"source_segment":segments,"passes_six_lag_cutoff":np.arange(len(ts))>=L})
    state_table.to_csv(output/"state_assignment.csv",index=False)
    segment_counts=[]
    for i,label in enumerate(segment_names):
        for state,name in [(-1,"low"),(0,"middle"),(1,"high"),(2,"missing_context")]:
            mask=(segments==i)&(states==state)&(np.arange(len(ts))>=L)
            counts=population(mask,months)
            counts["available_native_months"]=counts["native_months"] if state!=2 else 0
            segment_counts.append({"source_segment":label,"state":name,**counts})
    pd.DataFrame(segment_counts).to_csv(output/"source_segment_population.csv",index=False)
    cross_source_window=np.zeros(len(ts),dtype=bool)
    for lag in range(1,L+1):cross_source_window[L:]|=segments[L:]!=segments[L-lag:len(ts)-lag]
    pairs=pd.read_csv(PAIRS);vi={name:i for i,name in enumerate(names)}
    audit_pairs={int(pairs.pair_id.min()),int(pairs.pair_id.max())}
    rows=[];diagnostics=[];ols_audits=[];bartlett_audits=[];control_map=[]
    calendar_month=months[L:];trimmed_states=states[L:];calendar_n=len(ts)-L
    for period in PERIODS:
        period_mask=((ts<"2019-01-01") if period=="training" else (ts>="2019-01-01"))[L:]
        for pair_number,pair in enumerate(pairs.to_dict("records")):
            sr,tr=pair["source_region"],pair["target_region"]
            y=data[L:,tr,vi[pair["target_var"]]]
            own=np.column_stack([data[L-l:len(ts)-l,tr,vi[pair["target_var"]]] for l in [1,2,3]])
            sources=np.column_stack([data[L-l:len(ts)-l,sr,vi[pair["source_var"]]] for l in [1,2,3]])
            regions=sorted({sr,tr})
            terms=[(region,field,lag) for region in regions for field in phys_names for lag in [4,5,6]]
            controls=np.column_stack([physical[L-l:len(ts)-l,region,phys_names.index(field)] for region,field,l in terms])
            if period=="training":control_map.extend({"pair_id":pair["pair_id"],"region":r,"field":f,"lag_six_hour_steps":l} for r,f,l in terms)
            for model in MODELS:
                c=own if model==MODELS[0] else np.column_stack([own,controls])
                fits={}
                for code,state_name in [(-1,"low"),(0,"middle"),(1,"high")]:
                    mask=period_mask&(trimmed_states==code)
                    fit=group_fit(y,c,sources,mask);fits[code]=fit
                    pop=population(mask,calendar_month)
                    for j,lag in enumerate([1,2,3]):
                        diagnostics.append({"pair_id":pair["pair_id"],"period":period,"model":model,"state":state_name,"lag":lag,
                                            **pop,**{k:fit[k] for k in ["rank_controls","rank_full","n_columns_controls","control_rank_deficient","control_condition_raw","control_condition_column_normalized"]},
                                            "source_residual_fraction":float(fit["source_residual_fraction"][j]),"source_vif":float(fit["source_vif"][j]),
                                            "source_identifiable":bool(fit["valid"][j]),"near_collinearity_warning":bool(fit["control_condition_column_normalized"]>1e8 or fit["source_residual_fraction"][j]<1e-8 or fit["control_rank_deficient"])})
                        if pair["pair_id"] in audit_pairs:
                            direct_design=np.column_stack([fit["base"],fit["selected_sources"][:,j]])
                            coef,_,rank,_=linalg.lstsq(direct_design,fit["selected_y"],cond=max(direct_design.shape)*np.finfo(float).eps,lapack_driver="gelsd",check_finite=False)
                            difference=abs(float(coef[-1])-fit["beta"][j])
                            ols_audits.append({"pair_id":pair["pair_id"],"period":period,"model":model,"state":state_name,"lag":lag,
                                               "beta_svd":float(coef[-1]),"beta_fwl":float(fit["beta"][j]),"absolute_difference":difference,"rank_svd":int(rank),"rank_fwl":fit["rank_full"],
                                               "pass":bool(np.isclose(coef[-1],fit["beta"][j],atol=1e-8,rtol=1e-7) and rank==fit["rank_full"])})
                for contrast,code in CONTRASTS:
                    middle,extreme=fits[0],fits[code]
                    delta=extreme["beta"]-middle["beta"]
                    score=extreme["influence"]-middle["influence"]
                    covariance=score_autocovariances(score,max(BANDWIDTHS))
                    selected=middle["mask"]|extreme["mask"];n=int(selected.sum());rank=middle["rank_full"]+extreme["rank_full"]
                    correction=n/(n-rank); se_by_bw={}
                    for b in BANDWIDTHS:
                        h=np.arange(1,b+1);meat=covariance[0]+2*np.sum((1-h[:,None]/(b+1))*covariance[1:b+1],axis=0)
                        se_by_bw[b]=np.sqrt(np.maximum(meat,0)*correction)
                        for j,lag in enumerate([1,2,3]):
                            valid=bool(middle["valid"][j] and extreme["valid"][j]); se=float(se_by_bw[b][j]);d=float(delta[j])
                            diag_flag=bool(middle["control_condition_column_normalized"]>1e8 or extreme["control_condition_column_normalized"]>1e8 or middle["source_residual_fraction"][j]<1e-8 or extreme["source_residual_fraction"][j]<1e-8 or middle["control_rank_deficient"] or extreme["control_rank_deficient"])
                            row={**pair,"lag":lag,"period":period,"contrast":contrast,"model":model,"hac_bandwidth_six_hour_rows":b,
                                 "valid":valid,"near_collinearity_warning":diag_flag,"delta_beta":d,"beta_middle":float(middle["beta"][j]),"beta_extreme":float(extreme["beta"][j]),
                                 "se_hac":se,"ci95_low_normal_uncalibrated":d-1.96*se,"ci95_high_normal_uncalibrated":d+1.96*se,
                                 "p_normal_uncalibrated":float(2*stats.norm.sf(abs(d)/max(se,1e-300))) if valid else np.nan,
                                 "n_six_hour_model_rows":n,"rank_full":rank,"control_columns_per_state":c.shape[1]+1,"n_physical_controls_per_state":0 if model==MODELS[0] else len(terms),
                                 "hac_finite_sample_correction":correction,"calendar_cross_source_window_rows":int((selected&cross_source_window[L:]).sum()),
                                 "middle_source_residual_fraction":float(middle["source_residual_fraction"][j]),"extreme_source_residual_fraction":float(extreme["source_residual_fraction"][j]),
                                 "max_control_condition_column_normalized":max(middle["control_condition_column_normalized"],extreme["control_condition_column_normalized"]),
                                 **{"middle_"+k:v for k,v in population(middle["mask"],calendar_month).items()},**{"extreme_"+k:v for k,v in population(extreme["mask"],calendar_month).items()}}
                            rows.append(row)
                    if pair["pair_id"]==pairs.pair_id.min():
                        audits=full_design_audit(y,c,sources[:,0],middle["mask"],extreme["mask"],delta[0],{b:v[0] for b,v in se_by_bw.items()})
                        for record in audits:
                            record.update(pair_id=pair["pair_id"],lag=1,period=period,contrast=contrast,model=model)
                            record["pass"]=bool(record["delta_abs_difference"]<1e-8 and record["se_abs_difference"]<1e-8 and record["excluded_calendar_score_zero"] and record["full_svd_rank"]==rank)
                            bartlett_audits.append(record)
            if (pair_number+1)%6==0:
                status={"updated_utc":now(),"period":period,"completed_pairs_in_period":pair_number+1,"of_pairs":len(pairs),"result_rows":len(rows)}
                save_json(output/"status.json",status);print(json.dumps(status),flush=True)
    result=pd.DataFrame(rows);key=["pair_id","lag","period","contrast","hac_bandwidth_six_hour_rows"]
    if len(result)!=2592 or result.duplicated(key+["model"]).any():raise ValueError("Expected 2592 distinct model/bandwidth rows")
    result.to_csv(output/"joint_nino_contrasts.csv",index=False,float_format="%.15g")
    base=result[result.model==MODELS[0]].set_index(key);joint=result[result.model==MODELS[1]].set_index(key)
    if not np.array_equal(base.n_six_hour_model_rows,joint.n_six_hour_model_rows):raise ValueError("Baseline and joint samples differ")
    comparison=pd.DataFrame({"baseline_delta_beta":base.delta_beta,"physical_delta_beta":joint.delta_beta,"physical_minus_baseline_delta":joint.delta_beta-base.delta_beta,
                             "baseline_se_hac":base.se_hac,"physical_se_hac":joint.se_hac,"sign_changed":np.sign(base.delta_beta)!=np.sign(joint.delta_beta),
                             "baseline_valid":base.valid,"physical_valid":joint.valid,"physical_near_collinearity_warning":joint.near_collinearity_warning,
                             "n_six_hour_model_rows":base.n_six_hour_model_rows}).reset_index()
    comparison.to_csv(output/"matched_model_comparisons.csv",index=False,float_format="%.15g")
    result.groupby(["period","contrast","model","edge_type","hac_bandwidth_six_hour_rows"]).agg(models=("delta_beta","size"),valid_models=("valid","sum"),flagged_models=("near_collinearity_warning","sum"),median_delta_beta=("delta_beta","median"),median_abs_delta_beta=("delta_beta",lambda x:float(np.nanmedian(np.abs(x)))),median_se_hac=("se_hac","median")).reset_index().to_csv(output/"descriptive_summary.csv",index=False,float_format="%.15g")
    pd.DataFrame(diagnostics).to_csv(output/"rank_collinearity_diagnostics.csv",index=False,float_format="%.15g")
    pd.DataFrame(control_map).to_csv(output/"physical_control_mapping.csv",index=False)
    pd.DataFrame(ols_audits).to_csv(output/"independent_group_svd_checks.csv",index=False,float_format="%.15g")
    pd.DataFrame(bartlett_audits).to_csv(output/"full_design_direct_bartlett_checks.csv",index=False,float_format="%.15g")
    d=pd.DataFrame(diagnostics)
    audit={"result_rows":len(result),"all_model_pairs_same_samples":True,"all_inputs_hash_verified":True,
           "state_thresholds_evaluation_perturbation_unchanged":True,"input_preprocessing_refitted":False,
           "missing_state_rows_full_calendar":int((states==2).sum()),"missing_state_rows_after_six_lag_cutoff":int((trimmed_states==2).sum()),
           "independent_group_svd_checks":len(ols_audits),"all_group_svd_checks_pass":all(a["pass"] for a in ols_audits),
           "group_svd_max_abs_beta_difference":float(max(a["absolute_difference"] for a in ols_audits)),
           "full_design_bartlett_checks":len(bartlett_audits),"all_full_design_bartlett_checks_pass":all(a["pass"] for a in bartlett_audits),
           "full_design_max_abs_delta_difference":float(max(a["delta_abs_difference"] for a in bartlett_audits)),
           "full_design_max_abs_se_difference":float(max(a["se_abs_difference"] for a in bartlett_audits)),
           "invalid_result_rows":int((~result.valid).sum()),"near_collinearity_flagged_result_rows":int(result.near_collinearity_warning.sum()),
           "rank_deficient_group_lag_diagnostic_rows":int(d.control_rank_deficient.sum()),
           "maximum_control_condition_column_normalized":float(d.control_condition_column_normalized.max()),
           "minimum_source_residual_fraction":float(d.source_residual_fraction.min()),
           "calendar_cross_source_window_rows_after_cutoff":int(cross_source_window[L:].sum()),
           "all_results_retained":True,"pvalues_and_intervals_calibrated":False,"bh_applied":False}
    save_json(output/"implementation_audit.json",audit)
    snapshots=output/"code_snapshot";snapshots.mkdir(exist_ok=True)
    for p in [Path(__file__),Path(__file__).with_name("inference.py"),Path(__file__).with_name("run_external_nino_regimes.py")]:shutil.copy2(p,snapshots/p.name)
    readme=["# Antecedent Niño3.4 states with physical controls", "", "All 2592 predeclared result rows are retained: 108 directed source-lag models × two state contrasts × two retrospectively fitted periods × three calendar HAC bandwidths × two adjustment models. This extends the external-index sensitivity experiment; it does not establish causal identification.","",
            "The own-history baseline and joint physical model use identical rows after the six-step history cutoff. The joint model adds six fields at t−4/t−5/t−6 for the unique source/target regions (18 or 36 physical columns per state). Every state has its own intercept, target own-lag coefficients, physical-control coefficients and tested-source coefficient. State fitting is separate computationally, but extreme-minus-middle influence scores are combined on the original six-hour calendar before HAC; cross-state covariance is retained.","",
            "The previous q20/q80 thresholds (−0.66/+0.66 °C anomaly, 480 native training months) and antecedent-month rule are unchanged. No original or physical preprocessing parameters are fitted anew. The initial six-row cutoff removes all four unavailable November-1978 state rows; this is explicitly different from the earlier own-history-only experiment's three-row cutoff.","",
            "joint_nino_contrasts.csv contains all fits with explicitly labeled HAC bandwidths and exploratory uncalibrated normal intervals/p-values. matched_model_comparisons.csv gives every same-sample adjustment change. rank_collinearity_diagnostics.csv retains per-group rank, condition numbers and source residual fractions; physical_control_mapping.csv lists every lagged field/region. source_segment_population.csv reports available month labels, state episodes and six-hour rows separately for the three existing acquisition segments. No extra source-segment regressions are fitted.","",
            f"Implementation checks: {len(ols_audits)} independent SVD group solves, maximum source-slope difference {audit['group_svd_max_abs_beta_difference']:.3g}; {len(bartlett_audits)} full saturated-design pseudoinverse and direct time-domain Bartlett checks, maximum contrast difference {audit['full_design_max_abs_delta_difference']:.3g} and SE difference {audit['full_design_max_abs_se_difference']:.3g}. These verify arithmetic and calendar handling, not statistical calibration.","","## Limitations",""]+["- "+s for s in LIMITS+EXTRA_LIMITS]
    (output/"README.md").write_text("\n".join(readme)+"\n",encoding="utf-8")
    manifest={"started_utc":started,"completed_utc":now(),"frozen_plan_sha256":digest(plan_file),"script_sha256":digest(__file__),
              "elapsed_seconds":time.perf_counter()-start,"peak_working_set_bytes":int(getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)),
              "software":{name:importlib.metadata.version(name) for name in ["numpy","pandas","scipy","psutil","threadpoolctl"]},
              "audit":audit,"limitations":LIMITS+EXTRA_LIMITS,"outputs_sha256":{str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file() and p.name not in ["manifest.json","status.json"]}}
    save_json(output/"manifest.json",manifest)
    save_json(output/"status.json",{"status":"completed","completed_utc":manifest["completed_utc"],"result_rows":len(result)})
    print(json.dumps({"output":str(output),"audit":audit,"elapsed_seconds":manifest["elapsed_seconds"],"peak_working_set_bytes":manifest["peak_working_set_bytes"]},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=BASE/"revision_outputs/external_nino_physical_joint")
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--freeze",action="store_true");mode.add_argument("--run",action="store_true")
    args=p.parse_args()
    with threadpool_limits(limits=1):(freeze if args.freeze else run)(args.output)


if __name__=="__main__":main()
