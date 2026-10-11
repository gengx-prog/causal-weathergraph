"""Read-only independent audit of the bounded weather-token pilot.

Does not invoke preparation, training, or production scoring helpers. Results
are printed to stdout and optionally saved to an explicitly named audit report.
Production artifacts are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
NODES = np.array([2, 8, 13, 19, 24, 30, 35, 41, 46, 52, 57, 63])
CASES = ("pooled", "nh_to_sh", "sh_to_nh")
VARIANTS = ("continuous", "quantile_tokens", "physical_tokens", "physical_tokens_residual")
PHYSICAL = np.array([[-20,-10,-5,-1,1,5,10,20],[-20,-10,-5,-1,1,5,10,20],
                    [.001,.002,.004,.006,.008,.010,.014,.018],[240,250,260,270,280,290,300,310],
                    [-.3,-.1,-.03,-.005,.005,.03,.1,.3],[.05,.15,.3,.45,.55,.7,.85,.95]])
STANDARD = np.array([-2,-1,-.5,-.15,.15,.5,1,2])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def close(actual, expected, name, atol=2e-7):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    error = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
    assert np.allclose(a, b, rtol=2e-6, atol=atol), (name, error)
    return error


def raw_inputs():
    files = [BASE/"revision_outputs/inputs/region_trainfit_vectors.npz",
             BASE/"revision_outputs/physical_controls_inputs/region_controls_trainfit.npz"]
    with np.load(files[0]) as z:
        selected = np.array([np.where(z["node_ids"] == n)[0][0] for n in NODES])
        ts, lat, lon = z["timestamps"].astype("datetime64[ns]"), z["lat"][selected], z["lon"][selected]
        names = z["variable_names"].tolist()
        physical = z["physical"]
        values = np.empty((len(ts),12,6), dtype=np.float32)
        for j, name in enumerate(("u","v","humidity","temperature",None,"cloud_cover")):
            if name is not None:
                values[:,:,j] = physical[:,selected,names.index(name)]
        del physical
    with np.load(files[1]) as z:
        assert np.array_equal(z["timestamps"], ts)
        assert np.array_equal(z["node_ids"][selected], NODES)
        assert np.array_equal(z["lat"][selected], lat)
        physical = z["physical"]
        values[:,:,4] = physical[:,selected,z["variable_names"].tolist().index("omega_700")]
        del physical
    assert np.isfinite(values).all()
    assert np.all(np.diff(ts) == np.timedelta64(6,"h"))
    assert values[:,:,5].min() >= 0 and values[:,:,5].max() <= 1
    return values, ts, lat, lon, {str(p):sha(p) for p in files}


def independent_indices(ts, region_indices, split):
    start, end = {"train":("1979-01-01","2015-01-01"),
                  "val":("2015-01-01","2019-01-01"),
                  "test":("2019-01-01","2100-01-01")}[split]
    candidate = np.where((ts >= np.datetime64(start)) & (ts < np.datetime64(end)) &
                         (ts.astype("datetime64[h]").astype(np.int64)%24 == 0))[0]
    candidate = candidate[(candidate>=8) & (candidate+4<len(ts))]
    candidate = candidate[ts[candidate+4]<np.datetime64(end)]
    offsets = np.arange(-8,5)
    w = ts[candidate[:,None]+offsets]
    segments = (w>=np.datetime64("2019-01-01")).astype(int)+(w>=np.datetime64("2023-01-11")).astype(int)
    candidate = candidate[np.all(segments == segments[:,[0]],axis=1)]
    return np.array([(t,r) for t in candidate for r in region_indices],dtype=np.int64)


def independent_labels(raw, index):
    t,r = index.T
    diff = raw[t[:,None]+np.array([1,2,4]),r[:,None],5].astype(float)-raw[t,r,5,None].astype(float)
    return np.where(diff<-.05,0,np.where(diff>.05,2,1))


def independent_metrics(labels, probability):
    assert probability.shape == (len(labels),3,3)
    assert np.isfinite(probability).all() and probability.min()>=0 and probability.max()<=1
    close(probability.sum(-1),np.ones(labels.shape),"probability normalization")
    p = probability.astype(float)
    one = np.eye(3)[labels]
    brier = np.einsum("nhc,nhc->nh",p-one,p-one)
    logloss = -np.log(np.clip(p[np.arange(len(p))[:,None],np.arange(3)[None,:],labels],1e-12,None))
    result = []
    for h in range(3):
        prediction = p[:,h].argmax(1)
        conf = p[:,h].max(1)
        correct = prediction == labels[:,h]
        f1=[]
        for c in range(3):
            tp=np.sum((prediction==c)&(labels[:,h]==c))
            fp=np.sum((prediction==c)&(labels[:,h]!=c))
            fn=np.sum((prediction!=c)&(labels[:,h]==c))
            f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        # Reproduce documented ten top-label bins using integer bin assignment.
        bins = np.minimum(np.floor(conf*10).astype(int),9)
        ece=sum(np.mean(bins==k)*abs(conf[bins==k].mean()-correct[bins==k].mean())
                for k in range(10) if np.any(bins==k))
        result.append({"brier":brier[:,h].mean(),"log_loss":logloss[:,h].mean(),
                       "accuracy":correct.mean(),"macro_f1":np.mean(f1),"ece10":ece})
    return result,brier


def audit_case(out, case, raw, ts, lat, source_hashes):
    folder=out/case
    if not (folder/"transform.npz").exists():
        return {"status":"not_yet_prepared"},{}
    m=json.loads((folder/"case_manifest.json").read_text(encoding="utf8"))
    assert m["source_hashes"] == source_hashes
    assert m["transform_sha256"] == sha(folder/"transform.npz")
    source=np.arange(12) if case=="pooled" else np.where(lat>0 if case=="nh_to_sh" else lat<0)[0]
    target=np.arange(12) if case=="pooled" else np.where(lat<0 if case=="nh_to_sh" else lat>0)[0]
    train=ts<np.datetime64("2015-01-01")
    n=int(train.sum())
    values=raw[train][:,source].astype(float)
    mean=values.reshape(-1,6).mean(0)
    std=values.reshape(-1,6).std(0)
    std=np.where(std<1e-8,1.,std)
    ds=np.diff(values,axis=0).reshape(-1,6).std(0)
    ds=np.where(ds<1e-8,1.,ds)
    month=np.array([int(str(t.astype("datetime64[M]"))[-2:])-1 for t in ts])
    month=(month[:,None]+np.where(lat<0,6,0)[None,:])%12
    bands,band=np.unique(np.round(abs(lat),6),return_inverse=True)
    climate=np.empty((3,12,6))
    counts=np.empty((3,12),int)
    for b in range(3):
        for mo in range(12):
            r=source[band[source]==b]
            chosen=raw[:n,r][month[:n,r]==mo].astype(float)
            climate[b,mo]=chosen.mean(0)
            counts[b,mo]=len(chosen)
    F=np.empty((*raw.shape[:2],18),np.float32)
    F[:,:,:6]=(raw.astype(float)-mean)/std
    F[:,:,6:12]=(raw.astype(float)-climate[band[None,:],month])/std
    F[0,:,12:]=0
    F[1:,:,12:]=np.diff(raw.astype(float),axis=0)/ds
    max_error=0.
    with np.load(folder/"transform.npz") as z:
        for name,expected in (("abs_mean",mean),("abs_std",std),("delta_std",ds),
            ("climatology",climate),("climatology_counts",counts),
            ("source_local_region_indices",source),("target_local_region_indices",target)):
            max_error=max(max_error,close(z[name],expected,name))
        bq=np.empty((18,8)); bs=np.tile(STANDARD,(18,1)); bs[:6]=(PHYSICAL-mean[:,None])/std[:,None]
        for j in range(18):
            first=1 if j>=12 else 0
            v=F[first:n,source,j].ravel().astype(float)
            ordered=np.sort(v); location=(len(ordered)-1)*np.arange(1,9)/9
            lower=np.floor(location).astype(int); frac=location-lower
            bq[j]=ordered[lower]+(ordered[np.ceil(location).astype(int)]-ordered[lower])*frac
            for scheme,boundary in (("q",bq[j]),("s",bs[j])):
                code=np.sum(v[:,None]>=boundary,axis=1)
                number=np.bincount(code,minlength=9)
                close(z[f"training_bin_counts_{scheme}"][j],number,f"{scheme} counts")
                for c in range(9):
                    if number[c]:
                        max_error=max(max_error,close(z[f"centers_{scheme}"][j,c],v[code==c].mean(),f"{scheme} center"))
        close(z["boundaries_q"],bq,"quantile edges")
        close(z["boundaries_s"],bs,"physical edges")
    expected={split:independent_indices(ts,target if split=="test" else source,split)
              for split in ("train","val","test")}
    for split,idx in expected.items():
        assert len(idx)==m["excluded_and_retained"][split]["samples_retained"]
    examples=json.loads((folder/"examples.json").read_text(encoding="utf8"))
    for example in examples:
        t,r=example["time_index"],example["local_region_index"]
        close(example["F"],F[t,r],"example F")
        assert np.array_equal(example["Q"],np.sum(F[t,r,:,None]>=bq,axis=1))
        assert np.array_equal(example["S"],np.sum(F[t,r,:,None]>=bs,axis=1))
        assert np.array_equal(example["labels_6_12_24h"],independent_labels(raw,np.array([[t,r]]))[0])
    del F,values
    predictions={}; run_checks=[]
    for predpath in sorted(folder.glob("*/predictions.npz")):
        with np.load(predpath) as z:
            idx,labels,p=z["index"],z["label"],z["probability"]
        runfile=predpath.parent/"run.json"
        run=json.loads(runfile.read_text()) if runfile.exists() else None
        smoke=bool(run and run.get("smoke"))
        want=expected["test"][:2048] if smoke else expected["test"]
        assert np.array_equal(idx,want), str(predpath)+" sample indices"
        assert np.array_equal(labels,independent_labels(raw,idx)), str(predpath)+" labels"
        result,loss=independent_metrics(labels,p)
        table=pd.read_csv(predpath.parent/"metrics.csv")
        for j,row in enumerate(result):
            for metric,value in row.items():
                close(table.iloc[j][metric],value,"metric "+metric,atol=2e-6)
        name=predpath.parent.name.split("_seed")[0]
        predictions.setdefault(name,[]).append((idx,loss))
        if run and "best_epoch" in run:
            history=pd.read_csv(predpath.parent/"history.csv")
            best=float("inf"); chosen=-1
            for row in history.itertuples():
                if row.validation_brier<best-1e-5:
                    best=row.validation_brier; chosen=int(row.epoch)
            assert run["best_epoch"]==chosen
            close(run["best_validation_brier"],best,"checkpoint validation")
            assert run["train_n"]==(min(2048,len(expected["train"])) if smoke else len(expected["train"]))
            assert run["validation_n"]==(min(2048,len(expected["val"])) if smoke else len(expected["val"]))
            assert run["source_hashes"]==source_hashes
            assert run["code_sha256"]==sha(Path(__file__).with_name("train.py"))
            assert run["data_code_sha256"]==sha(Path(__file__).with_name("data.py"))
            assert run["protocol_sha256"]==sha(out/"protocol.json")
        if name=="training_frequency":
            train_labels=independent_labels(raw,expected["train"])
            prior=np.array([(np.bincount(train_labels[:,h],minlength=3)+1)/(len(train_labels)+3) for h in range(3)])
            close(p,np.broadcast_to(prior,p.shape),"training-only frequency baseline")
        if name=="persistence":
            close(p,np.broadcast_to(np.array([0.,1.,0.]),p.shape),"persistence baseline")
        run_checks.append({"run":predpath.parent.name,"n":len(idx),"smoke":smoke})
    return {"status":"PASS","samples":{k:len(v) for k,v in expected.items()},
            "max_transform_error":max_error,"examples_checked":len(examples),"prediction_runs":run_checks},predictions


def audit_bootstrap(out,ts,predictions):
    path=out/"paired_brier_differences.csv"
    if not path.exists():return {"status":"not_yet_available"}
    table=pd.read_csv(path); maximum=0.; checked=0
    for case,pred in predictions.items():
        if "continuous" not in pred:continue
        idx=pred["continuous"][0][0]
        for runs in pred.values():
            assert all(np.array_equal(index,idx) for index,loss in runs)
        avg={name:np.mean([loss for index,loss in runs],axis=0) for name,runs in pred.items()}
        _,inverse=np.unique(ts[idx[:,0]].astype("datetime64[D]"),return_inverse=True)
        nd=int(inverse.max())+1
        size=np.bincount(inverse)
        assert np.all(size==size[0]),"Unequal region counts per date would change bootstrap weighting"
        for name,loss in avg.items():
            if name=="continuous":continue
            delta=loss-avg["continuous"]
            daily=np.column_stack([np.bincount(inverse,weights=delta[:,h])/size for h in range(3)])
            rng=np.random.default_rng(20261001)
            boot=[]
            for rep in range(500):
                start=rng.integers(nd,size=(nd+29)//30)
                selected=np.concatenate([np.arange(k,k+30)%nd for k in start])[:nd]
                boot.append(daily[selected].mean(0))
            bounds=np.percentile(boot,[2.5,97.5],axis=0)
            rows=table[(table.case==case)&(table.variant==name)].sort_values("horizon_hours")
            if len(rows)==0:continue
            assert len(rows)==3
            maximum=max(maximum,close(rows.delta_brier,delta.mean(0),"paired delta"),
                        close(rows.exploratory_block_p025,bounds[0],"bootstrap p025"),
                        close(rows.exploratory_block_p975,bounds[1],"bootstrap p975"))
            checked+=3
    return {"status":"PASS","rows_checked":checked,"maximum_error":maximum,
            "scope":"circular blocks of 30 retained dates; seam gaps compressed, no calibrated coverage or seed-population inference"}


def audit_summary(out,ts):
    allfile=out/"metrics_all_runs.csv"
    meanfile=out/"metrics_seed_mean.csv"
    if not allfile.exists() or not meanfile.exists():return {"status":"not_yet_available"}
    actual=pd.read_csv(allfile); collected=[]
    keys=["case","variant","run","slice","horizon_hours"]
    assert not actual.duplicated(keys).any()
    for case in CASES:
        for path in sorted((out/case).glob("*/predictions.npz")):
            with np.load(path) as z:
                index,label,p=z["index"],z["label"],z["probability"]
            name=path.parent.name; variant=name.split("_seed")[0]
            times=ts[index[:,0]]
            for segment,mask in (("all",np.ones(len(label),bool)),
                ("pre_extension",times<np.datetime64("2023-01-11")),
                ("extension",times>=np.datetime64("2023-01-13"))):
                if not mask.any():continue
                results,_=independent_metrics(label[mask],p[mask])
                for h,result in zip((6,12,24),results):
                    collected.append(dict(case=case,variant=variant,run=name,slice=segment,horizon_hours=h,**result))
    want=pd.DataFrame(collected).set_index(keys).sort_index()
    actual=actual.set_index(keys).sort_index()
    assert actual.index.equals(want.index)
    maximum=0.
    metric_names=["brier","log_loss","accuracy","macro_f1","ece10"]
    for name in metric_names:maximum=max(maximum,close(actual[name],want[name],"summary "+name,atol=2e-6))
    mean_keys=["case","variant","slice","horizon_hours"]
    expected=want.reset_index().groupby(mean_keys)[metric_names].mean().sort_index()
    means=pd.read_csv(meanfile).set_index(mean_keys).sort_index()
    assert means.index.equals(expected.index)
    for name in metric_names:maximum=max(maximum,close(means[name],expected[name],"seed mean "+name,atol=2e-6))
    return {"status":"PASS","run_metric_rows":len(actual),"seed_mean_rows":len(means),"maximum_error":maximum}


def audit_transfer_gaps(out):
    path=out/"transfer_gap_all_seeds.csv"
    if not path.exists():return {"status":"not_yet_available"}
    actual=pd.read_csv(path)
    keys=["case","variant","seed","horizon_hours"]
    assert len(actual)==72 and not actual.duplicated(keys).any()
    rows=[]
    for case in ("nh_to_sh","sh_to_nh"):
        for variant in VARIANTS:
            for seed in (17,29,43):
                name=f"{variant}_seed{seed}"
                with np.load(out/case/name/"predictions.npz") as z:
                    index,label,p=z["index"],z["label"],z["probability"]
                with np.load(out/"pooled"/name/"predictions.npz") as z:
                    positions={tuple(pair):i for i,pair in enumerate(z["index"])}
                    selected=np.array([positions[tuple(pair)] for pair in index])
                    assert np.array_equal(z["index"][selected],index)
                    assert np.array_equal(z["label"][selected],label)
                    pooled=z["probability"][selected]
                one=np.eye(3)[label]
                source_loss=((p.astype(float)-one)**2).sum(2).mean(0)
                pooled_loss=((pooled.astype(float)-one)**2).sum(2).mean(0)
                for h,hours in enumerate((6,12,24)):
                    rows.append(dict(case=case,variant=variant,seed=seed,horizon_hours=hours,
                        n_target_samples=len(index),source_only_brier=source_loss[h],
                        pooled_on_same_target_brier=pooled_loss[h],transfer_minus_pooled=source_loss[h]-pooled_loss[h]))
    expected=pd.DataFrame(rows).set_index(keys).sort_index()
    actual=actual.set_index(keys).sort_index()
    assert actual.index.equals(expected.index)
    maximum=0.
    for name in expected.columns:maximum=max(maximum,close(actual[name],expected[name],"transfer "+name))
    mean_keys=["case","variant","horizon_hours"]
    cols=["source_only_brier","pooled_on_same_target_brier","transfer_minus_pooled"]
    expected_mean=expected.reset_index().groupby(mean_keys)[cols].mean().sort_index()
    means=pd.read_csv(out/"transfer_gap_seed_mean.csv").set_index(mean_keys).sort_index()
    assert means.index.equals(expected_mean.index) and len(means)==24
    for name in cols:maximum=max(maximum,close(means[name],expected_mean[name],"mean transfer "+name))
    return {"status":"PASS","all_seed_rows":72,"seed_mean_rows":24,"maximum_error":maximum,
        "scope":"Identical target examples; pooled versus transfer also changes training sample size, hemisphere composition, preprocessing, and validation population, so gap is not an isolated causal effect of geographic transfer."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=BASE/"revision_outputs/weather_token_pilot")
    parser.add_argument("--require-complete",action="store_true")
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    raw,ts,lat,lon,hashes=raw_inputs()
    report={"production_artifacts_read_only":True,"audit_sha256":sha(__file__),"cases":{},"source_hashes":hashes}
    predictions={}
    for case in CASES:
        report["cases"][case],predictions[case]=audit_case(args.output,case,raw,ts,lat,hashes)
    report["bootstrap"]=audit_bootstrap(args.output,ts,predictions)
    report["summary"]=audit_summary(args.output,ts)
    report["transfer_gaps"]=audit_transfer_gaps(args.output)
    if args.require_complete:
        for case in CASES:
            runs=report["cases"][case]["prediction_runs"]
            assert len(runs)==15 and not any(run["smoke"] for run in runs)
            for variant in VARIANTS:assert len(predictions[case][variant])==3
        assert report["bootstrap"]["rows_checked"]==54
        assert report["summary"]["run_metric_rows"]==405
        assert report["summary"]["seed_mean_rows"]==189
    report["complete_batch_requested"]=args.require_complete
    if args.report:
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
