"""Small, retrospective ERA5 tokenization pilot; no language-model pretraining.

Run from any directory with the revision venv. All artifacts are new and scoped
to weather_token_pilot. Input preparation is deliberately separate from models.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score

from data import load_raw, prepare_case

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "revision_outputs" / "weather_token_pilot"
VARIANTS = ["continuous", "quantile_tokens", "physical_tokens", "physical_tokens_residual"]
CASES = ["pooled", "nh_to_sh", "sh_to_nh"]
SEEDS = [17, 29, 43]
HORIZONS = [6, 12, 24]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf8")


def protocol():
    return {
        "question": "Do predefined atmospheric bins, with or without numerical residuals, improve cloud-change probability forecasts over matched continuous or quantile-token inputs?",
        "status": "new prospective computation protocol on historically inspected retrospective data; not preregistered research",
        "training": "1979-2014 source regions", "validation": "2015-2018 source regions", "evaluation": "2019-2025; historically inspected; all target regions in transfer cases",
        "origins": "00 UTC daily; every variant uses identical origins within each case",
        "region_ids": [2,8,13,19,24,30,35,41,46,52,57,63],
        "context": "8 state tokens t-42h,...,t; differences require ninth raw observation t-48h",
        "horizons_hours": HORIZONS, "target": "raw regional total cloud cover change: decrease < -0.05; stable [-0.05,0.05]; increase >0.05",
        "inputs": ["u850", "v850", "q850", "T850", "omega700", "total_cloud_cover"],
        "features": "each variable: standardized absolute value, source-training seasonal-reference anomaly, standardized preceding 6h change; 18 identical features in all arms",
        "cases": CASES, "variants": VARIANTS, "seeds": SEEDS,
        "semantic_scope": "fixed physical/anomaly binning; same embedding architecture as quantile bins; no claimed natural-language understanding or new causal identification",
        "source_reference": "source-training absolute-latitude band x local month; SH month shifted by 6; no target-hemisphere fitted statistics; this is a seasonal-transfer assumption",
        "model": {"width":64, "layers":2, "heads":4, "ff_width":128, "dropout":0.1, "feature_embedding_width":4, "classes_per_horizon":3},
        "optimization": {"optimizer":"AdamW", "lr":0.001, "weight_decay":0.01, "batch_size":1024, "max_epochs":10, "early_stopping_patience":3, "minimum_validation_brier_improvement":0.00001, "loss":"unweighted multiclass cross entropy, mean over horizons", "selection":"lowest source-validation mean multiclass Brier"},
        "torch_execution": "deterministic algorithms required; flash and memory-efficient SDPA disabled; float32, TF32 disabled; four CPU threads",
        "metrics": ["multiclass Brier, sum across classes; primary", "log loss", "accuracy", "macro F1", "10-bin top-label ECE; descriptive"],
        "uncertainty": "paired differences after averaging per-example losses across 3 seeds, all regions retained together by origin date; circular blocks of 30 retained dates, 500 draws; gaps at source seams mean these are not always 30 consecutive calendar days; exploratory, no simultaneous or causal guarantee",
        "evaluation_slices": ["all", "before CDS extension 2023-01-11", "CDS extension after 48h seam exclusion"],
        "limitations": ["12 prechosen coarse regions; not global or operational weather validation", "uniform region weighting, not area-weighted global error", "reanalysis available retrospectively, not real-time observations", "source preprocessing discontinuity is not resolved by tokenization", "SH seasonal shift does not guarantee physical symmetry", "three random seeds and a fixed small model do not establish SOTA", "same backbone and total allocated parameters; active input parameters differ by arm", "no natural-language pretrained model, actual text is a deterministic rendering", "no atmospheric causal-effect identification"],
    }


class Forecaster(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.numeric = nn.Linear(18, 64)
        self.embedding = nn.Embedding(18 * 9, 4)
        self.token_projection = nn.Linear(18 * 4, 64)
        self.metadata = nn.Linear(6, 64)
        self.position = nn.Parameter(torch.empty(8,64))
        nn.init.normal_(self.position, std=.02)
        self.register_buffer("offsets", torch.arange(18)*9)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4,
            dim_feedforward=128, dropout=.1, activation="gelu", batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64,9)

    def forward(self, f, q, s, residual, meta):
        if self.variant == "continuous":
            h = self.numeric(f)
        else:
            code = q if self.variant == "quantile_tokens" else s
            h = self.token_projection(self.embedding(code+self.offsets).flatten(-2))
            if self.variant == "physical_tokens_residual":
                h = h + self.numeric(residual)
        # Every time is observed by the forecast origin; no future tokens exist.
        # Bidirectional attention within observed history is therefore allowed.
        h = h + self.position + self.metadata(meta)[:,None,:]
        h = self.transformer(h)
        return self.head(self.norm(h[:,-1])).reshape(-1,3,3)


class DeviceData:
    def __init__(self, d, device):
        self.F = torch.as_tensor(d["F"], device=device, dtype=torch.float32)
        self.Q = torch.as_tensor(d["Q"], device=device, dtype=torch.int64)
        self.S = torch.as_tensor(d["S"], device=device, dtype=torch.int64)
        self.meta = torch.as_tensor(d["meta"], device=device, dtype=torch.float32)
        self.y = torch.as_tensor(d["labels"], device=device, dtype=torch.long)
        centers = torch.as_tensor(d["centers_s"], device=device, dtype=torch.float32)
        feat = torch.arange(18, device=device)
        self.residual = self.F - centers[feat[None,None,:], self.S]
        self.lags = torch.arange(-7,1,device=device)
        self.device = device

    def batch(self, index):
        i = torch.as_tensor(index,device=self.device,dtype=torch.long)
        t, r = i[:,0], i[:,1]
        tt, rr = t[:,None]+self.lags, r[:,None]
        return (self.F[tt,rr],self.Q[tt,rr],self.S[tt,rr],
                self.residual[tt,rr],self.meta[t,r]), self.y[t,r]


@torch.no_grad()
def predict(model, data, idx, batch=2048):
    model.eval()
    probs = []
    for start in range(0,len(idx),batch):
        x, _ = data.batch(idx[start:start+batch])
        probs.append(model(*x).softmax(-1).cpu().numpy())
    return np.concatenate(probs)


def scores(y, p):
    p = np.asarray(p, dtype=np.float64)
    one = np.eye(3)[y]
    brier = ((p-one)**2).sum(-1)
    ll = -np.log(np.maximum(np.take_along_axis(p,y[...,None],axis=-1)[...,0],1e-12))
    return brier, ll


def metrics(y,p):
    b, ll = scores(y,p)
    rows = []
    for j,h in enumerate(HORIZONS):
        pred = p[:,j].argmax(-1)
        conf = p[:,j].max(-1)
        corr = pred == y[:,j]
        ece = 0.
        for low in np.arange(10)/10:
            mask = (conf>=low) & (conf<low+.1 if low<.9 else conf<=1.)
            if mask.any():
                ece += mask.mean()*abs(conf[mask].mean()-corr[mask].mean())
        rows.append({"horizon_hours":h,"n":len(y),"brier":float(b[:,j].mean()),
            "log_loss":float(ll[:,j].mean()),"accuracy":float(corr.mean()),
            "macro_f1":float(f1_score(y[:,j],pred,labels=[0,1,2],average="macro",zero_division=0)),
            "ece10":float(ece),"decrease_rate":float((y[:,j]==0).mean()),
            "stable_rate":float((y[:,j]==1).mean()),"increase_rate":float((y[:,j]==2).mean())})
    return rows


def train_one(d, gpu, case, variant, seed, out, epochs=10, smoke=False):
    run = out / case / f"{variant}_seed{seed}"
    run.mkdir(parents=True,exist_ok=True)
    complete = run/"run.json"
    if complete.exists() and (run/"predictions.npz").exists():
        previous=json.loads(complete.read_text(encoding="utf8"))
        expected={"smoke":smoke,"code_sha256":sha(__file__),
            "data_code_sha256":sha(Path(__file__).with_name("data.py")),
            "protocol_sha256":sha(out/"protocol.json"),"source_hashes":d.get("source_hashes")}
        if any(previous.get(k)!=v for k,v in expected.items()):
            raise ValueError(f"Existing run provenance differs: {run}; preserve it and use a new output directory")
        print(f"REUSE {case} {variant} seed={seed}",flush=True)
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = Forecaster(variant).to(gpu.device)
    optim = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]
    if smoke:
        train_idx=train_idx[:2048]
        val_idx=val_idx[:2048]
    vy = d["labels"][val_idx[:,0],val_idx[:,1]]
    rng = np.random.default_rng(seed)
    best = np.inf
    best_epoch = -1
    stale = 0
    history = []
    start = time.perf_counter()
    updates = 0
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(train_idx))
        loss_sum = 0.
        for first in range(0,len(perm),1024):
            chosen = train_idx[perm[first:first+1024]]
            x,y = gpu.batch(chosen)
            optim.zero_grad(set_to_none=True)
            logits = model(*x)
            loss = nn.functional.cross_entropy(logits.flatten(0,1),y.flatten())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
            loss_sum += loss.item()*len(chosen)
            updates += 1
        vp = predict(model,gpu,val_idx)
        vb = float(scores(vy,vp)[0].mean())
        history.append({"epoch":epoch+1,"train_cross_entropy":loss_sum/len(train_idx),
            "validation_brier":vb,"elapsed_seconds":time.perf_counter()-start,"updates":updates})
        print(json.dumps({"case":case,"variant":variant,"seed":seed,**history[-1]}),flush=True)
        if vb < best-1e-5:
            best=vb
            best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            stale=0
        else:
            stale+=1
        if stale>=3:
            break
    training_seconds=time.perf_counter()-start
    model.load_state_dict(best_state)
    torch.save(best_state,run/"checkpoint.pt")
    test_idx=d["test_idx"] if not smoke else d["test_idx"][:2048]
    pp=predict(model,gpu,test_idx)
    yy=d["labels"][test_idx[:,0],test_idx[:,1]]
    np.savez_compressed(run/"predictions.npz",index=test_idx,probability=pp,label=yy)
    pd.DataFrame(metrics(yy,pp)).to_csv(run/"metrics.csv",index=False)
    pd.DataFrame(history).to_csv(run/"history.csv",index=False)
    unused = (sum(p.numel() for p in model.embedding.parameters())+sum(p.numel() for p in model.token_projection.parameters()) if variant=="continuous" else sum(p.numel() for p in model.numeric.parameters()) if variant!="physical_tokens_residual" else 0)
    dump(complete,{"case":case,"variant":variant,"seed":seed,"best_epoch":best_epoch,
        "best_validation_brier":best,"epochs_executed":len(history),"optimizer_updates":updates,
        "training_plus_validation_seconds":training_seconds,"total_seconds":time.perf_counter()-start,
        "allocated_parameters":sum(p.numel() for p in model.parameters()),"inactive_input_parameters":unused,
        "train_n":len(train_idx),"validation_n":len(val_idx),"test_n":len(test_idx),
        "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "code_sha256":sha(__file__),"data_code_sha256":sha(Path(__file__).with_name("data.py")),
        "protocol_sha256":sha(out/"protocol.json"),"source_hashes":d.get("source_hashes"),"smoke":smoke})
    print(f"DONE {case} {variant} seed={seed}, {time.perf_counter()-start:.1f}s",flush=True)


def baselines(d, out, case):
    idx=d["train_idx"]
    y=d["labels"][idx[:,0],idx[:,1]]
    test=d["test_idx"]
    yy=d["labels"][test[:,0],test[:,1]]
    prior=np.stack([(np.bincount(y[:,h],minlength=3)+1)/(len(y)+3) for h in range(3)])
    for name,p in [("training_frequency",np.broadcast_to(prior,(len(test),3,3)).copy()),
                   ("persistence",np.broadcast_to(np.array([0.,1.,0.]),(len(test),3,3)).copy())]:
        dest=out/case/name
        dest.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(dest/"predictions.npz",index=test,probability=p.astype(np.float32),label=yy)
        pd.DataFrame(metrics(yy,p)).to_csv(dest/"metrics.csv",index=False)


def summarize(out, raw):
    all_rows=[]
    diff_rows=[]
    for case in CASES:
        predicted={}
        for variant in VARIANTS+["training_frequency","persistence","linear_logistic"]:
            paths=sorted((out/case).glob(f"{variant}_seed*/predictions.npz")) if variant in VARIANTS else [out/case/variant/"predictions.npz"]
            paths=[p for p in paths if p.exists()]
            if not paths: continue
            bs=[]
            reference_index=reference_label=None
            for path in paths:
                with np.load(path) as z:
                    idx=z["index"].copy(); yy=z["label"].copy(); pp=z["probability"].copy()
                if reference_index is None:
                    reference_index,reference_label=idx,yy
                elif not (np.array_equal(idx,reference_index) and np.array_equal(yy,reference_label)):
                    raise ValueError(f"Seed predictions do not align: {path}")
                if pp.shape!=(len(idx),3,3) or not np.isfinite(pp).all() or not np.allclose(pp.sum(-1),1,atol=1e-5) or (pp<0).any():
                    raise ValueError(f"Invalid probabilities: {path}")
                dates=raw["timestamps"][idx[:,0]]
                masks={"all":np.ones(len(idx),dtype=bool),"pre_extension":dates<np.datetime64("2023-01-11"),"extension":dates>=np.datetime64("2023-01-13")}
                for split,mask in masks.items():
                    if not mask.any(): continue
                    for row in metrics(yy[mask],pp[mask]):
                        all_rows.append({"case":case,"variant":variant,"run":path.parent.name,"slice":split,**row})
                b,_=scores(yy,pp)
                bs.append(b)
            predicted[variant]=(idx,np.mean(bs,axis=0))
        if "continuous" not in predicted: continue
        idx,base=predicted["continuous"]
        dates=raw["timestamps"][idx[:,0]].astype("datetime64[D]")
        # One calendar date is one cluster, retaining all available regions.
        unique,inv=np.unique(dates,return_inverse=True)
        for variant,(otheridx,b) in predicted.items():
            if variant=="continuous":continue
            if not np.array_equal(idx,otheridx):raise ValueError("Unpaired evaluation indices")
            diff=b-base
            daily=np.zeros((len(unique),3))
            count=np.bincount(inv)
            for h in range(3):daily[:,h]=np.bincount(inv,weights=diff[:,h])/count
            rng=np.random.default_rng(20261001)
            boots=[]
            # Blocks may cross the small excluded seam gap: explicitly an approximate
            # calendar-cluster sensitivity interval, not a certified coverage claim.
            for rep in range(500):
                starts=rng.integers(0,len(daily),size=int(np.ceil(len(daily)/30)))
                take=((starts[:,None]+np.arange(30))%len(daily)).ravel()[:len(daily)]
                boots.append(daily[take].mean(0))
            bounds=np.quantile(boots,[.025,.975],axis=0)
            for h,hours in enumerate(HORIZONS):
                diff_rows.append({"case":case,"variant":variant,"reference":"continuous","horizon_hours":hours,
                    "delta_brier":float(diff[:,h].mean()),"exploratory_block_p025":bounds[0,h],"exploratory_block_p975":bounds[1,h],
                    "n_calendar_days":len(daily),"block_days":30,"bootstrap_draws":500,
                    "meaning":"negative favors variant; average individual-seed losses, not probability ensemble"})
    if all_rows:
        frame=pd.DataFrame(all_rows)
        frame.to_csv(out/"metrics_all_runs.csv",index=False)
        frame.groupby(["case","variant","slice","horizon_hours"],as_index=False)[["brier","log_loss","accuracy","macro_f1","ece10"]].mean().to_csv(out/"metrics_seed_mean.csv",index=False)
    if diff_rows:pd.DataFrame(diff_rows).to_csv(out/"paired_brier_differences.csv",index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--cases",nargs="+",default=CASES,choices=CASES)
    ap.add_argument("--variants",nargs="+",default=VARIANTS,choices=VARIANTS)
    ap.add_argument("--seeds",nargs="+",type=int,default=SEEDS)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--summarize-only",action="store_true")
    args=ap.parse_args()
    out=args.output
    if args.smoke and out==DEFAULT_OUT:
        out=out.with_name(out.name+"_smoke")
    out.mkdir(parents=True,exist_ok=True)
    if not (out/"protocol.json").exists():dump(out/"protocol.json",protocol())
    else:
        old=json.loads((out/"protocol.json").read_text(encoding="utf8"))
        if old!=protocol():raise ValueError("Protocol differs from frozen file; use a new output directory")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dump(out/"environment.json",{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"device":str(device),"gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,"deterministic_algorithms":True, "sdpa":"math; flash and memory-efficient disabled", "code_sha256":sha(__file__)})
    raw=load_raw()
    if not args.summarize_only:
        for case in args.cases:
            d=prepare_case(raw,case,out/case)
            d["source_hashes"]=raw.get("source_hashes")
            gpu=DeviceData(d,device)
            for variant in args.variants:
                for seed in args.seeds:
                    train_one(d,gpu,case,variant,seed,out,epochs=1 if args.smoke else 10,smoke=args.smoke)
            if not args.smoke:baselines(d,out,case)
            del gpu,d
            if torch.cuda.is_available():torch.cuda.empty_cache()
    summarize(out,raw)
    print("FINISHED "+str(out),flush=True)


if __name__=="__main__":main()
