"""Read-only forecast-error diagnosis; no training or test-set recalibration."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[3]/"revision_outputs/weather_token_pilot"
DEST=OUT/"diagnostics_explanation"
MODELS=["continuous","quantile_tokens","physical_tokens","physical_tokens_residual"]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    DEST.mkdir(exist_ok=True)
    decomposed=[];regions=[];probabilities=[];reliability=[];hashes={}
    for case in ["pooled","nh_to_sh","sh_to_nh"]:
        manifest=json.loads((OUT/case/"case_manifest.json").read_text(encoding="utf8"))
        with np.load(OUT/case/"training_frequency/predictions.npz") as z:
            ids=z["index"].copy(); y=z["label"].copy(); base=z["probability"].astype(float)
        target=np.eye(3)[y]
        base_loss=((base-target)**2).sum(-1)
        for model in MODELS:
            for seed in [17,29,43]:
                path=OUT/case/f"{model}_seed{seed}/predictions.npz"
                hashes[str(path.relative_to(OUT))]=sha(path)
                with np.load(path) as z:
                    assert np.array_equal(z["index"],ids) and np.array_equal(z["label"],y)
                    p=z["probability"].astype(float)
                loss=((p-target)**2).sum(-1)
                for j,h in enumerate([6,12,24]):
                    pj=p[:,j];oj=target[:,j];bj=base[:,j]
                    truth=oj.mean(0);mean=pj.mean(0)
                    var=((pj-mean)**2).sum(-1).mean()
                    cov_credit=2*((pj-mean)*(oj-truth)).sum(-1).mean()
                    uncertainty=1-(truth**2).sum()
                    bias=((mean-truth)**2).sum()
                    departure=((pj-bj)**2).sum(-1).mean()
                    alignment_credit=2*((pj-bj)*(oj-bj)).sum(-1).mean()
                    direct=loss[:,j].mean()
                    assert abs(direct-(uncertainty+bias+var-cov_credit))<1e-12
                    gap=(loss[:,j]-base_loss[:,j]).mean()
                    assert abs(gap-(departure-alignment_credit))<1e-12
                    conf=pj.max(-1);correct=pj.argmax(-1)==y[:,j]
                    high=conf>=.8
                    record={"case":case,"model":model,"seed":seed,"hours":h,"n":len(y),
                        "brier":direct,"frequency_brier":base_loss[:,j].mean(),"excess_brier":gap,
                        "test_label_uncertainty_diagnostic_only":uncertainty,"squared_mean_probability_bias":bias,
                        "probability_variation":var,"observation_covariance_credit":cov_credit,
                        "departure_from_training_frequency":departure,"alignment_credit_relative_to_frequency":alignment_credit,
                        "mean_max_probability":conf.mean(),"accuracy":correct.mean(),
                        "confidence_minus_accuracy":conf.mean()-correct.mean(),
                        "high_confidence_n":int(high.sum()),"high_confidence_fraction":high.mean(),
                        "high_confidence_mean_probability":conf[high].mean() if high.any() else None,
                        "high_confidence_accuracy":correct[high].mean() if high.any() else None}
                    decomposed.append(record)
                    for k,cl in enumerate(["decrease","stable","increase"]):
                        probabilities.append({"case":case,"model":model,"seed":seed,"hours":h,"class":cl,
                            "training_frequency":bj[0,k],"test_observed_frequency":truth[k],"model_mean_probability":mean[k],
                            "squared_mean_bias":(mean[k]-truth[k])**2})
                        binid=np.minimum((pj[:,k]*10).astype(int),9)
                        for b in range(10):
                            keep=binid==b
                            if keep.any():reliability.append({"case":case,"model":model,"seed":seed,"hours":h,"class":cl,
                                "bin":b,"n":int(keep.sum()),"mean_probability":pj[keep,k].mean(),"observed_fraction":oj[keep,k].mean()})
                for r in np.unique(ids[:,1]):
                    keep=ids[:,1]==r
                    for j,h in enumerate([6,12,24]):
                        conf=p[keep,j].max(-1);correct=p[keep,j].argmax(-1)==y[keep,j]
                        regions.append({"case":case,"model":model,"seed":seed,"region_id":manifest["node_ids"][r],
                            "latitude":manifest["lat"][r],"longitude":manifest["lon"][r],"hours":h,"n":int(keep.sum()),
                            "brier":loss[keep,j].mean(),"frequency_brier":base_loss[keep,j].mean(),
                            "excess_brier":(loss[keep,j]-base_loss[keep,j]).mean(),
                            "contribution_to_total_excess":(loss[keep,j]-base_loss[keep,j]).sum()/len(y),
                            "confidence_minus_accuracy":conf.mean()-correct.mean()})
    a=pd.DataFrame(decomposed);b=pd.DataFrame(regions);c=pd.DataFrame(probabilities)
    a.to_csv(DEST/"probability_diagnostics_all_seeds.csv",index=False)
    a.groupby(["case","model","hours"],as_index=False).mean(numeric_only=True).drop(columns="seed").to_csv(DEST/"probability_diagnostics_seed_mean.csv",index=False)
    a.groupby(["case","model"],as_index=False).mean(numeric_only=True).drop(columns=["seed","hours"]).to_csv(DEST/"probability_diagnostics_horizon_mean.csv",index=False)
    b.to_csv(DEST/"regional_losses_all_seeds.csv",index=False)
    b.groupby(["case","model","region_id","latitude","longitude"],as_index=False).mean(numeric_only=True).drop(columns=["seed","hours"]).to_csv(DEST/"regional_losses_horizon_seed_mean.csv",index=False)
    c.to_csv(DEST/"class_probabilities_all_seeds.csv",index=False)
    c.groupby(["case","model","hours","class"],as_index=False).mean(numeric_only=True).drop(columns="seed").to_csv(DEST/"class_probabilities_seed_mean.csv",index=False)
    pd.DataFrame(reliability).to_csv(DEST/"reliability_bins_all_seeds.csv",index=False)
    metadata={"question":"Why did token forecasts not beat the training-frequency baseline?",
        "mode":"read-only descriptive diagnosis of existing predictions; no retraining, threshold selection, or recalibration",
        "identity_1":"BS = test label uncertainty + squared mean probability bias + variance of probabilities - 2*forecast/observation covariance; this algebra is not causal attribution or binned reliability/resolution decomposition",
        "identity_2":"BS_model - BS_fixed_frequency = mean squared departure from fixed frequency - 2*mean inner product of departure and observed departure",
        "checks":"Both identities checked for all 108 model/seed/horizon combinations to 1e-12; input label/index equality checked for all 36 model prediction files",
        "caveats":["Test label proportions are diagnostic summaries, not a new fitted operational baseline","Seed and horizon means are averages of single-model diagnostics, not an ensemble","All findings are retrospective and exploratory","High-confidence condition is frozen here at max probability >=0.8; correlated examples are not independent trials"],
        "code_sha256":sha(__file__),"prediction_hashes":hashes}
    (DEST/"probability_diagnostics_manifest.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf8")
    selected=a.groupby(["case","model"])[["brier","frequency_brier","excess_brier","squared_mean_probability_bias","probability_variation","observation_covariance_credit","mean_max_probability","accuracy","confidence_minus_accuracy","high_confidence_fraction","high_confidence_mean_probability","high_confidence_accuracy"]].mean()
    print(selected.round(5).to_string())
    print("REGIONAL EXCESS (physical tokens, equal horizon/seed mean)")
    print(b[b.model=="physical_tokens"].groupby(["case","region_id","latitude","longitude"])[["excess_brier","contribution_to_total_excess"]].mean().round(5).to_string())


if __name__=="__main__":main()
