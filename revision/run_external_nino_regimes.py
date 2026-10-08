"""Retrospective slope contrasts under antecedent monthly Nino3.4 quantile states.

This is an external-index sensitivity experiment, not official ENSO-event
classification, calibrated inference, causal identification, or prediction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import psutil
from scipy import stats
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.inference import _basis, score_autocovariances

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "revision_outputs/inputs/region_trainfit.npz"
PAIRS = BASE / "revision_outputs/path_diagnostics/frozen_joint_lag_pairs.csv"
MONTHLY = BASE / "supplementary_data/circulation_indices/nino34_1979_2025_monthly.csv"
CONTEXT = MONTHLY.with_name("nino34_preceding_context.csv")
INDEX_MANIFEST = MONTHLY.with_name("manifest.json")
BANDWIDTHS = (64, 128, 512)
PERIODS = ("training", "evaluation")
CONTRASTS = (("high_minus_middle", 1), ("low_minus_middle", -1))
LIMITS = [
    "Monthly Nino3.4 quantile states are not official ENSO episodes and do not classify all atmospheric circulation regimes.",
    "Broadcasting one monthly value to six-hour rows does not create independent six-hour index observations; native-month and contiguous-episode counts are reported separately.",
    "Index values are published retrospective estimates; temporal precedence of the represented month does not establish real-time release availability.",
    "External index construction does not guarantee causal exogeneity, no unmeasured confounding, or identification of an ENSO causal effect.",
    "Both periods are fitted retrospectively. Evaluation-period slope fitting is not a training-frozen forecast or prospective validation.",
    "The 36 pairs were already selected using training information; these contrasts do not remove that selection or create an exhaustive graph-wide test.",
    "HAC bandwidths 64/128/512 six-hour rows (16/32/128 days) are exploratory sensitivity settings. Monthly-state inference is not calibrated; nominal normal intervals/p-values do not establish valid FDR or familywise claims.",
    "No BH correction, significance filtering, favorable-bandwidth selection, or official ENSO-event labels are used.",
    "The evaluation interval combines the existing WeatherBench2 ERA5 and later CDS ERA5 input segments; this experiment does not isolate their source-boundary contribution.",
    "Thresholds use 480 unique native months in 1979-2018; December 1978 supplies preceding context but never enters threshold fitting. Unavailable November 1978 is not imputed.",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def now():
    return pd.Timestamp.now(tz="UTC").isoformat()


def load_index():
    native = pd.read_csv(MONTHLY, parse_dates=["date"])
    context = pd.read_csv(CONTEXT, parse_dates=["date"])
    if native.date.duplicated().any() or context.date.duplicated().any():
        raise ValueError("Native monthly index has duplicate month labels")
    expected = pd.date_range("1979-01-01", "2025-12-01", freq="MS")
    if not np.array_equal(native.date.values, expected.values):
        raise ValueError("Native index does not have the exact expected 564 months")
    if context.date.dt.strftime("%Y-%m").tolist() != ["1978-12"]:
        raise ValueError("Only the frozen December 1978 context is allowed")
    if not np.isfinite(native.value).all() or not np.isfinite(context.value).all():
        raise ValueError("Missing native index values; no imputation allowed")
    return native, context


def thresholds(native):
    train = native[(native.date >= "1979-01-01") & (native.date < "2019-01-01")]
    if len(train) != 480 or train.date.nunique() != 480:
        raise ValueError("Threshold fitting requires exactly 480 unique months")
    return np.quantile(train.value.to_numpy(), [.2, .8], method="linear")


def freeze(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "frozen_plan.json"
    if path.exists():
        raise FileExistsError("Existing frozen plan will not be overwritten")
    native, context = load_index()
    low, high = thresholds(native)
    pairs = pd.read_csv(PAIRS)
    if len(pairs) != 36 or pairs.pair_id.nunique() != 36:
        raise ValueError("Expected the pre-existing 36 unique directed pairs")
    source_index = next(x for x in json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))["indices"] if x["id"] == "nino34")
    paths = [INPUT, PAIRS, MONTHLY, CONTEXT, INDEX_MANIFEST, Path(__file__), Path(__file__).with_name("inference.py")]
    plan = {"frozen_utc": now(), "status": "frozen_before_regime_fits", "inputs": [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in paths],
            "pairs": 36, "source_lags_six_hour_steps": [1,2,3], "directed_models": 108,
            "contrasts": [c for c,_ in CONTRASTS], "periods": {"training":["1979-01-01","2018-12-31"],"evaluation":["2019-01-01","2025-12-31"]},
            "expected_long_result_rows":1296,"hac_bandwidths_six_hour_rows":list(BANDWIDTHS),
            "thresholds": {"q20":float(low),"q80":float(high),"method":"numpy linear quantile", "native_months":480,"fit_first":"1979-01","fit_last":"2018-12","context_used_for_thresholds":False},
            "index_source": {"url":source_index["download"]["url"],"documentation":source_index["download"]["documentation"],"units":source_index["units"],"retrospective_vintage_downloaded_utc":source_index["download"]["downloaded_at_utc"]},
            "state_assignment": "For target timestamp t, use monthly value for period(t - 24 hours) - 1 calendar month. Require represented month end strictly before t - 18 hours, hence before every source lag 1..3.",
            "state_boundaries": "low <= q20; high >= q80; middle strictly between. Apply frozen training-month thresholds to both periods.",
            "missing_context": "All four target rows on 1979-01-01 require unavailable November 1978, retained missing. Lag cutoff removes the first three model rows; one additional model-calendar row remains missing.",
            "model": "Separate high-vs-middle and low-vs-middle saturated two-group models: each group has its own intercept, target own lags 1/2/3, and tested source-lag slope. Exclude the other extreme group. Full rank normally 10. FWL batching is computational only; each source-lag is a separate model.",
            "calendar_hac": "Full original six-hour calendar positions retained after the common initial three-lag cutoff. Excluded periods/states/missing rows carry score zero. Bartlett weights 1-h/(bandwidth+1); covariance correction n/(n-rank_full). All bandwidths retained, not selected.",
            "reporting": "Long table with explicit bandwidth, delta slope, both group slopes, exploratory SE/normal CI/nominal uncalibrated p; no BH or significance flags. Separate six-hour row, unique antecedent native month, and contiguous-state episode counts.",
            "verification": "All contrasts independently checked as two separately fitted group OLS slopes. At least first pair/lag in each contrast and period checked using direct time-domain Bartlett score cross-products for all bandwidths. Perturb evaluation index values to verify threshold isolation.",
            "limitations":LIMITS}
    save_json(path,plan)
    print(json.dumps({"frozen_plan":str(path),"sha256":digest(path),"thresholds":plan["thresholds"]}),flush=True)


def fit_batch(y, controls, sources, extreme, eligible):
    x = np.asarray(sources, dtype=float)
    state = np.asarray(extreme, dtype=float)
    mask = np.asarray(eligible,dtype=bool) & np.isfinite(y) & np.isfinite(controls).all(1) & np.isfinite(x).all(1)
    c = np.column_stack([np.ones(len(y)), controls, state, controls*state[:,None]])
    q, rank = _basis(c[mask])
    yy = y[mask] - q @ (q.T @ y[mask])
    xx = x[mask] - q @ (q.T @ x[mask])
    vv = (x*state[:,None])[mask]
    vv = vv - q @ (q.T @ vv)
    sxx = np.sum(xx*xx,axis=0)
    if np.any(sxx < 1e-12):
        raise ValueError("Source not identifiable")
    ratio = np.sum(xx*vv,axis=0)/sxx
    z = vv - xx*ratio
    zz = np.sum(z*z,axis=0)
    if np.any(zz < 1e-12):
        raise ValueError("Interaction not identifiable")
    base = yy @ xx / sxx
    target = yy[:,None]-xx*base
    delta = np.sum(z*target,axis=0)/zz
    resid = target-z*delta
    influence = np.zeros_like(x)
    influence[mask] = z*resid/zz
    acov = score_autocovariances(influence,max(BANDWIDTHS))
    n = int(mask.sum())
    full_rank = rank+2
    correction = n/(n-full_rank)
    reference = base-ratio*delta
    errors = {}
    for b in BANDWIDTHS:
        h = np.arange(1,b+1)
        meat = acov[0]+2*np.sum((1-h[:,None]/(b+1))*acov[1:b+1],axis=0)
        if np.any(meat < -1e-14):
            raise ValueError("Negative Bartlett variance outside numerical tolerance")
        errors[b] = np.sqrt(np.maximum(meat,0)*correction)
    return {"delta":delta,"reference":reference,"extreme":reference+delta,"se":errors,
            "mask":mask,"influence":influence,"n":n,"rank":full_rank,"correction":correction}


def count_population(mask, state_mask, antecedent_month):
    selected = np.asarray(mask) & np.asarray(state_mask)
    previous = np.r_[False, selected[:-1]]
    return {"six_hour_rows":int(selected.sum()),"native_months":int(pd.Index(antecedent_month[selected]).nunique()),
            "contiguous_episodes":int((selected & ~previous).sum())}


def run(output):
    start = time.perf_counter()
    plan_file = output/"frozen_plan.json"
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if (output/"manifest.json").exists():
        raise FileExistsError("Completed output exists; use a new output directory")
    for info in plan["inputs"]:
        if digest(info["path"]) != info["sha256"]:
            raise ValueError(f"Frozen input changed: {info['path']}")
    native, context = load_index()
    low, high = plan["thresholds"]["q20"], plan["thresholds"]["q80"]
    mutated = native.copy()
    mutated.loc[mutated.date >= "2019-01-01", "value"] = 1e8
    if not np.array_equal(thresholds(mutated),np.array([low,high])):
        raise ValueError("Thresholds leak evaluation index data")
    with np.load(INPUT) as z:
        data=np.asarray(z["data"],dtype=float); names=z["variable_names"].tolist(); timestamps=pd.DatetimeIndex(z["timestamps"])
    if np.any(np.diff(timestamps.values) != np.timedelta64(6,"h")) or not np.isfinite(data).all():
        raise ValueError("Regional input must have complete, finite six-hour data")
    lookup = pd.concat([context,native],ignore_index=True).set_index(pd.concat([context,native],ignore_index=True).date.dt.to_period("M")).value
    antecedent = (timestamps - pd.Timedelta(hours=24)).to_period("M")-1
    index_value = lookup.reindex(antecedent).to_numpy()
    states = np.full(len(timestamps),2,dtype=np.int8)
    states[np.isfinite(index_value)] = 0
    states[index_value <= low] = -1
    states[index_value >= high] = 1
    eligible_time = np.isfinite(index_value)
    month_ends = antecedent.end_time
    if not np.all(month_ends[eligible_time] < timestamps[eligible_time]-pd.Timedelta(hours=18)):
        raise ValueError("Antecedent month is not strictly earlier than every tested source")
    period_masks={"training":timestamps<"2019-01-01","evaluation":timestamps>="2019-01-01"}
    state_labels = np.array([{-1:"low",0:"middle",1:"high",2:"missing_context"}[int(s)] for s in states])
    state_table=pd.DataFrame({"target_time_utc":timestamps,"antecedent_native_month":antecedent.astype(str),
                              "antecedent_month_end_utc":month_ends,"nino34_value_degC":index_value,"state":state_labels,
                              "period":np.where(period_masks["training"],"training","evaluation"),
                              "passes_initial_three_lag_cutoff":np.arange(len(timestamps))>=3,
                              "has_native_index":eligible_time})
    state_table.to_csv(output/"state_assignment.csv",index=False)
    counts=[]
    for period,p_mask in period_masks.items():
        for scope, scope_mask in [("all_target_rows",np.ones(len(timestamps),dtype=bool)),("after_three_lag_cutoff",np.arange(len(timestamps))>=3)]:
            for label,value in [("low",-1),("middle",0),("high",1),("missing_context",2)]:
                counts.append({"period":period,"scope":scope,"state":label,**count_population(p_mask & scope_mask, states==value,antecedent.astype(str).to_numpy())})
    pd.DataFrame(counts).to_csv(output/"population_counts.csv",index=False)
    native_out=native.copy(); native_out["used_to_fit_thresholds"]=native_out.date<"2019-01-01"
    native_out["quantile_state"]=np.select([native.value<=low,native.value>=high],["low","high"],default="middle")
    native_out.to_csv(output/"native_month_states.csv",index=False)
    pairs=pd.read_csv(PAIRS)
    groups=defaultdict(list)
    for pair in pairs.to_dict("records"):
        for lag in [1,2,3]:
            groups[(pair["target_region"],pair["target_var"])].append((pair,lag))
    vi={name:i for i,name in enumerate(names)}; ntime=len(data); L=3
    rows=[]; group_audits=[]; hac_audits=[]
    for period in PERIODS:
        for contrast,code in CONTRASTS:
            selected=(period_masks[period] & np.isin(states,[0,code]))[L:]
            extreme=(states==code)[L:]
            for group_number,((target_region,target_var),specs) in enumerate(groups.items()):
                y=data[L:,target_region,vi[target_var]]
                own=np.column_stack([data[L-l:ntime-l,target_region,vi[target_var]] for l in [1,2,3]])
                source=np.column_stack([data[L-l:ntime-l,p["source_region"],vi[p["source_var"]]] for p,l in specs])
                fit=fit_batch(y,own,source,extreme,selected)
                pop_ref=count_population(fit["mask"],~extreme,antecedent.astype(str).to_numpy()[L:])
                pop_ext=count_population(fit["mask"],extreme,antecedent.astype(str).to_numpy()[L:])
                # Independent OLS check: separate regressions, no FWL or interaction design.
                for j,(pair,lag) in enumerate(specs):
                    slopes=[]
                    for state_flag in [False,True]:
                        m=fit["mask"] & (extreme==state_flag)
                        design=np.column_stack([np.ones(int(m.sum())),own[m],source[m,j]])
                        coef,_,rank,_=np.linalg.lstsq(design,y[m],rcond=None)
                        if rank!=5:raise ValueError("Independent group OLS rank differs from five")
                        slopes.append(float(coef[-1]))
                    delta_check=slopes[1]-slopes[0]
                    difference=max(abs(slopes[0]-fit["reference"][j]),abs(slopes[1]-fit["extreme"][j]),abs(delta_check-fit["delta"][j]))
                    if difference>1e-9:raise ValueError("Grouped OLS does not match interaction contrast")
                    group_audits.append({"pair_id":pair["pair_id"],"lag":lag,"period":period,"contrast":contrast,"max_absolute_coefficient_difference":difference})
                    for b in BANDWIDTHS:
                        delta=float(fit["delta"][j]); se=float(fit["se"][b][j])
                        row={**pair,"lag":lag,"period":period,"contrast":contrast,"hac_bandwidth_six_hour_rows":b,
                             "delta_beta":delta,"beta_middle":float(fit["reference"][j]),"beta_extreme":float(fit["extreme"][j]),
                             "se_hac":se,"ci95_low_normal_uncalibrated":delta-1.96*se,"ci95_high_normal_uncalibrated":delta+1.96*se,
                             "p_normal_uncalibrated":float(2*stats.norm.sf(abs(delta)/max(se,1e-300))),
                             "n_six_hour_model_rows":fit["n"],"rank_full":fit["rank"],"hac_finite_sample_correction":fit["correction"],
                             **{"middle_"+k:v for k,v in pop_ref.items()},**{"extreme_"+k:v for k,v in pop_ext.items()}}
                        rows.append(row)
                if group_number==0:
                    score=fit["influence"][:,0]
                    direct=np.array([float(np.sum(score[h:]*score[:len(score)-h])) for h in range(max(BANDWIDTHS)+1)])
                    for b in BANDWIDTHS:
                        h=np.arange(1,b+1)
                        se=float(np.sqrt(max(0,(direct[0]+2*np.sum((1-h/(b+1))*direct[1:b+1]))*fit["correction"])))
                        error=abs(se-fit["se"][b][0])
                        if error>1e-10:raise ValueError("Direct Bartlett check failed")
                        hac_audits.append({"pair_id":specs[0][0]["pair_id"],"lag":specs[0][1],"period":period,"contrast":contrast,"bandwidth":b,
                                           "se_direct":se,"se_fft":float(fit["se"][b][0]),"absolute_difference":error,
                                           "excluded_calendar_scores_exact_zero":bool(np.all(score[~fit["mask"]]==0))})
            print(f"Completed {period} {contrast}",flush=True)
    result=pd.DataFrame(rows)
    if len(result)!=1296 or result.duplicated(["pair_id","lag","period","contrast","hac_bandwidth_six_hour_rows"]).any():
        raise ValueError("Output is not the frozen 1296 distinct result rows")
    result.to_csv(output/"nino_regime_contrasts.csv",index=False,float_format="%.15g")
    summary=result.groupby(["period","contrast","edge_type","hac_bandwidth_six_hour_rows"],sort=False).agg(models=("delta_beta","size"),median_delta_beta=("delta_beta","median"),median_abs_delta_beta=("delta_beta",lambda x:float(np.median(np.abs(x)))),median_exploratory_se=("se_hac","median"),min_model_rows=("n_six_hour_model_rows","min")).reset_index()
    summary.to_csv(output/"descriptive_summary.csv",index=False,float_format="%.15g")
    pd.DataFrame(group_audits).to_csv(output/"independent_group_ols_checks.csv",index=False,float_format="%.15g")
    pd.DataFrame(hac_audits).to_csv(output/"direct_bartlett_checks.csv",index=False,float_format="%.15g")
    audits={"all_input_hashes_match_frozen_plan":True,"threshold_evaluation_perturbation_unchanged":True,
            "threshold_native_months":480,"native_index_months":len(native),"context_months":len(context),
            "full_calendar_rows":len(timestamps),"initial_three_lag_cutoff":3,"missing_index_rows_before_lag_cutoff":int((~eligible_time).sum()),
            "missing_index_rows_after_lag_cutoff":int((~eligible_time[L:]).sum()),"no_native_missing_value_imputed":True,
            "antecedent_month_end_strictly_before_all_source_times":True,
            "group_ols_contrasts_checked":len(group_audits),"group_ols_max_abs_difference":float(max(a["max_absolute_coefficient_difference"] for a in group_audits)),
            "direct_bartlett_comparisons":len(hac_audits),"direct_bartlett_max_se_abs_difference":float(max(a["absolute_difference"] for a in hac_audits)),
            "calendar_score_zero_checks_pass":all(a["excluded_calendar_scores_exact_zero"] for a in hac_audits),
            "bandwidths_correctly_labeled":result.hac_bandwidth_six_hour_rows.value_counts().sort_index().to_dict(),
            "science_inference_calibrated":False,"multiplicity_correction_applied":False,"all_results_retained":True}
    save_json(output/"implementation_audit.json",audits)
    snapshot=output/"code_snapshot";snapshot.mkdir(exist_ok=True)
    shutil.copy2(__file__,snapshot/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("inference.py"),snapshot/"inference.py")
    manifest={"started_after_plan_frozen_utc":plan["frozen_utc"],"completed_utc":now(),"code_sha256":digest(__file__),"frozen_plan_sha256":digest(plan_file),
              "result_rows":len(result),"thresholds":plan["thresholds"],"audit":audits,"limitations":LIMITS,
              "elapsed_seconds":time.perf_counter()-start,"peak_working_set_bytes":int(getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)),
              "numpy_version":np.__version__,"pandas_version":pd.__version__}
    readme=["# Antecedent monthly Nino3.4 state sensitivity", "", "The frozen 36 directed pairs and three source lags yield 108 models. High-minus-middle and low-minus-middle contrasts are fitted retrospectively in training and evaluation periods, and all three HAC bandwidths are retained (1296 rows). No significance filtering or BH correction is applied.","",
            f"Thresholds from 480 unique native training months: q20={low:.6g} and q80={high:.6g} degrees C anomaly. The antecedent state is period(target time minus 24 hours) minus one month. Its month end strictly precedes every tested source. December 1978 is context only; four full-calendar rows need unavailable November 1978. Three overlap the initial lag cutoff, leaving one additional excluded model-calendar row.","",
            "Each contrast model permits regime-specific intercepts, target own-history slopes (lags 1–3), and the tested source-lag slope. Coefficients are in the existing training-fitted standardized-variable units. Distinct source lags are fitted separately, with target own-history controls; this does not add a joint source-history adjustment.","",
            "The HAC time axis keeps all original six-hour positions; unused rows have zero scores. Bandwidths of 64, 128, and 512 rows correspond to 16, 32, and 128 days. Stored intervals and normal p-values are explicitly uncalibrated exploratory quantities. More broadcast rows do not imply more independent monthly observations.","",
            "Files: nino_regime_contrasts.csv contains every model/bandwidth; descriptive_summary.csv gives medians without significance counts; state_assignment.csv preserves the full mapping and missing context; population_counts.csv separates before/after lag cutoff, six-hour rows, native months, and contiguous episodes; native_month_states.csv preserves the 564 native-month values and threshold eligibility; implementation_audit.json and independent_group_ols_checks.csv / direct_bartlett_checks.csv contain numerical verification; frozen_plan.json and manifest.json record provenance and hashes.","",
            f"Numerical checks: {len(group_audits)} contrasts independently reproduce two separate group OLS regressions (maximum coefficient difference {audits['group_ols_max_abs_difference']:.3g}); {len(hac_audits)} direct time-domain Bartlett calculations agree with FFT-based standard errors (maximum difference {audits['direct_bartlett_max_se_abs_difference']:.3g}). Replacing every evaluation index value with 1e8 leaves training thresholds unchanged.","","## Interpretation limits",""]
    readme += ["- "+s for s in LIMITS]
    (output/"README.md").write_text("\n".join(readme)+"\n",encoding="utf-8")
    manifest["outputs_sha256"]={str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file() and p.name!="manifest.json"}
    save_json(output/"manifest.json",manifest)
    print(json.dumps({"output":str(output),"rows":len(result),"thresholds":plan["thresholds"],"audit":audits,"elapsed_seconds":manifest["elapsed_seconds"]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=BASE/"revision_outputs/external_nino_regimes")
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze",action="store_true");modes.add_argument("--run",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        (freeze if args.freeze else run)(args.output)


if __name__=="__main__":
    main()
