"""Create standalone scientific figures and a source-linked experiment record."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"revision_outputs/weather_token_pilot"
CASES=["pooled","nh_to_sh","sh_to_nh"]
MODELS=["continuous","quantile_tokens","physical_tokens","physical_tokens_residual"]
LABELS={"continuous":"Continuous", "quantile_tokens":"Quantile tokens",
        "physical_tokens":"Physical tokens", "physical_tokens_residual":"Tokens + residual",
        "linear_logistic":"Linear logistic", "training_frequency":"Training frequency"}
COLORS={"continuous":"#244b7c", "quantile_tokens":"#af790e", "physical_tokens":"#a74579",
        "physical_tokens_residual":"#347969", "linear_logistic":"#696969", "training_frequency":"#999999"}
CASE_NAMES={"pooled":"12 regions pooled", "nh_to_sh":"North to South", "sh_to_nh":"South to North"}
CASE_ZH={"pooled":"12区域合并", "nh_to_sh":"北训南测", "sh_to_nh":"南训北测"}


def hash_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transfer_gaps():
    records=[]
    for case in CASES[1:]:
        for variant in MODELS:
            for seed in [17,29,43]:
                with np.load(OUT/case/f"{variant}_seed{seed}/predictions.npz") as a, np.load(OUT/"pooled"/f"{variant}_seed{seed}/predictions.npz") as b:
                    ids=a["index"]; pp=a["probability"]; yy=a["label"]
                    mask=np.isin(b["index"][:,1],np.unique(ids[:,1]))
                    if not np.array_equal(b["index"][mask],ids) or not np.array_equal(b["label"][mask],yy):
                        raise ValueError("Unmatched transfer and pooled target observations")
                    one=np.eye(3)[yy]
                    source_loss=((pp-one)**2).sum(-1).mean(0)
                    pooled_loss=((b["probability"][mask]-one)**2).sum(-1).mean(0)
                for j,h in enumerate([6,12,24]):
                    records.append({"case":case,"variant":variant,"seed":seed,"horizon_hours":h,
                        "n_target_samples":len(ids),"source_only_brier":source_loss[j],
                        "pooled_on_same_target_brier":pooled_loss[j],"transfer_minus_pooled":source_loss[j]-pooled_loss[j]})
    frame=pd.DataFrame(records)
    frame.to_csv(OUT/"transfer_gap_all_seeds.csv",index=False)
    frame.groupby(["case","variant","horizon_hours"],as_index=False)[["source_only_brier","pooled_on_same_target_brier","transfer_minus_pooled"]].mean().to_csv(OUT/"transfer_gap_seed_mean.csv",index=False)
    return frame


def main():
    run_paths=[OUT/c/f"{m}_seed{s}/run.json" for c in CASES for m in MODELS for s in [17,29,43]]
    missing=[str(p) for p in run_paths if not p.exists()]
    if missing:raise RuntimeError(f"Cannot report incomplete 36-run experiment: {missing}")
    for case in CASES:
        if not (OUT/case/"linear_logistic/run.json").exists():raise RuntimeError("Linear baseline unfinished")
    metrics=pd.read_csv(OUT/"metrics_seed_mean.csv")
    differences=pd.read_csv(OUT/"paired_brier_differences.csv")
    frame=metrics[metrics["slice"]=="all"]
    transfer=transfer_gaps()
    runs=[json.loads(p.read_text(encoding="utf8")) for p in run_paths]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.labelcolor":"#303030","text.color":"#303030"})
    fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey="row")
    fig.subplots_adjust(left=.08,right=.99,bottom=.15,top=.82,hspace=.18,wspace=.08)
    styles={"continuous":("o","-"),"quantile_tokens":("s","--"),"physical_tokens":("D","-"),
            "physical_tokens_residual":("^","-"),"linear_logistic":("x",":"),"training_frequency":("+",":")}
    for col,case in enumerate(CASES):
        ax=axes[0,col]
        ax.set_title(CASE_NAMES[case])
        for variant in LABELS:
            s=frame[(frame.case==case)&(frame.variant==variant)].sort_values("horizon_hours")
            marker,ls=styles[variant]
            ax.plot(s.horizon_hours,s.brier,label=LABELS[variant],color=COLORS[variant],marker=marker,ls=ls,lw=1.5,ms=5)
        ax.grid(axis="y",color="#eeeeee")
        ax=axes[1,col]
        ax.axhline(0,color="#555555",lw=1)
        for variant,shift,marker in [("physical_tokens",-.4,"D"),("physical_tokens_residual",.4,"^")]:
            s=differences[(differences.case==case)&(differences.variant==variant)].sort_values("horizon_hours")
            # Draw endpoint intervals directly, including possible asymmetric
            # bootstrap bounds that do not bracket the point estimate.
            x=s.horizon_hours.to_numpy()+shift
            ax.vlines(x,s.exploratory_block_p025,s.exploratory_block_p975,color=COLORS[variant],lw=1.6)
            ax.scatter(x,s.delta_brier,color=COLORS[variant],marker=marker,s=32,label=LABELS[variant])
        ax.grid(axis="y",color="#eeeeee")
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_xticks([6,12,24])
    axes[0,0].set_ylabel("Multiclass Brier score (lower is better)")
    axes[1,0].set_ylabel("Brier difference from continuous\nNegative favors token variant")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.suptitle("Cloud-change probability forecast: retrospective 2019–2025 pilot",fontsize=14,y=.99)
    fig.text(.5,.035,"Top: mean of 3 single-model seed scores; baselines deterministic. Bottom: exploratory 2.5–97.5 percentiles,\n500 circular bootstrap draws of 30 retained dates; all regions kept together. 00 UTC origins; coarse regions; no causal claim.",ha="center",fontsize=9)
    dest=OUT/"figures";dest.mkdir(exist_ok=True)
    fig.savefig(dest/"forecast_comparison.png",dpi=180,bbox_inches="tight")
    fig.savefig(dest/"forecast_comparison.pdf",bbox_inches="tight")
    plt.close(fig)
    # Preserve the full numerical source behind the figure.
    frame.to_csv(dest/"forecast_comparison_source.csv",index=False)
    means=frame.groupby(["case","variant"]).brier.mean().unstack("variant")
    means=means.reindex(CASES)
    order=MODELS+["linear_logistic","training_frequency","persistence"]
    means[order].to_csv(OUT/"mean_brier_over_three_horizons.csv")
    rows=[]
    for case in CASES:
        rows.append("| "+CASE_ZH[case]+" | "+" | ".join(f"{means.loc[case,m]:.5f}" for m in order)+" |")
    pure=differences[differences.variant=="physical_tokens"]
    hybrid=differences[differences.variant=="physical_tokens_residual"]
    total_seconds=sum(r["training_plus_validation_seconds"] for r in runs)
    allocated=runs[0]["allocated_parameters"]
    peak=max(r["gpu_peak_allocated_bytes"] for r in runs)
    size=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    audit_path=OUT/"independent_audit.json"
    audit=json.loads(audit_path.read_text(encoding="utf8")) if audit_path.exists() else None
    report=f"""# 气象状态词元预测：首轮原型结果

已完成36次Transformer训练（3种场景×4种表示×3个种子），以及每场景的线性分类、类别频率和持续性基线。本轮只评价固定物理分箱及其数值残差；未训练自然语言大模型，也未识别物理因果效应。

**当前结论：原型实现完成，但尚未达到支持新方法优势的证据要求。** 合并区域的三个时距均由连续输入模型优于两种物理词元模型；跨半球有相对改善，但两种物理词元模型在两个方向、三个时距上的Brier分数均差于训练期类别频率基线。不能把“优于退化较大的连续模型”写成已经具备可靠跨域预测能力。

## 核心结果

下表为6/12/24小时三个时距等权平均的多类别Brier分数；Transformer先对三次单模型损失取均值，**没有用概率集成制造额外优势**。分数越低越好。不同场景的目标地区不同，不宜直接比较其绝对难度。

| 场景 | 连续数值 | 分位词元 | 物理词元 | 物理词元+残差 | 线性分类 | 类别频率 | 持续性 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

纯物理词元相对连续输入，在9个“场景×时距”组合中有{int((pure.delta_brier<0).sum())}个点估计更低；物理词元加残差有{int((hybrid.delta_brier<0).sum())}个。完整差值及探索性区间见 `paired_brier_differences.csv`。这些计数不是独立检验次数，也不是统计显著性结论；不据评价结果重新选择分箱、地区或模型参数。

![预测比较](figures/forecast_comparison.png)

## 如何理解这一轮

- 物理词元和分位词元使用相同的嵌入结构，区别仅在箱边界。因此可归因的比较是分箱规则，不是汉字含义或预训练语言知识。
- 物理词元加残差保留相同18项连续输入的信息；量化词元单独使用存在有损压缩。检查 `case_manifest.json` 中的重构RMSE与尾箱频率。
- 三个迁移/合并场景具有各自固定的源区训练统计。源域季节参照异常不是未见目标地区自己的气候异常；南北月份平移六个月是待检验假设。
- `transfer_gap_seed_mean.csv` 将迁移模型与合并模型在完全相同的目标半球样本上比较。合并模型拥有两倍训练样本、目标半球历史以及不同的归一化和验证来源；这个描述差值不能解释成仅由跨半球分布变化造成的纯迁移代价。
- 两个迁移方向的macro-F1部分高于类别频率基线，但预先指定的主要指标是概率质量Brier，不能在看到结果后更换成功标准。
- 在2023年来源扩展前、后的分段平均Brier中，迁移的两种物理词元模型也都未超过类别频率基线。不能把本轮失败简单归因于唯一的拼接边界。
- 尚可进一步研究的假设是离散化是否缓和了域外外推，以及物理边界是否比普通分位箱更稳健；本轮没有建立这种机制或原创性。新的改动需要独立记录为后续探索，不能继续把同一评价期视为未见数据。

## 任务、资源与范围

固定12个区域（六个纬带、每带两个经度扇区），六个原单位变量。过去48小时资料形成8个状态词元，预测总云量相对起报时刻下降超过5个百分点、稳定或上升超过5个百分点。只评估每日00 UTC起报。

训练1979—2014，验证2015—2018，回顾评价2019—2025。合并场景样本数157752/17520/30612；每个迁移方向78876/8760/15306。全部参数与箱代表值只由训练源地区拟合。排除输入或目标跨来源边界的窗口；所有方法使用相同样本。

每模型总分配参数{allocated:,}；各表示的活跃输入参数不同，详见每次 `run.json`。36次训练与验证计时合计{total_seconds:.1f}秒，不能理解成全流程墙钟时间。PyTorch最大记录GPU分配内存约{peak/2**20:.1f} MiB，不包含其他软件或全部驱动内存。生成本记录前正式输出约{size/2**20:.1f} MiB；无新增外部数据下载。

## 限制与审计

1. 2019—2025资料在原论文工作中已经查看。这是新的冻结计算方案下的回顾评价，不是从未接触数据的前瞻确认。
2. 2023年1月后的主变量与垂直速度来源处理不同，同质性未建立。已另报扩展前/后结果；剔除边界不能消除系统差异。
3. 12个粗区域不是全球高分辨率预测。区域算术平均、部分压力层可能位于地形以下等限制保留。
4. 这里只测试一种小模型、最大10轮和三个种子；尚无广泛调参、样本效率曲线、极端事件专项或真实缺测实验。
5. 循环块bootstrap使用30个保留起报日期，跨来源缺口时不等于30个连续日历日；区间仅作依赖敏感性的探索性摘要。
6. 独立实现核验结果以 `independent_audit.json` 为准。数值核验通过不代表预测方法优越、区间已校准或机制成立。

## 复算入口

- 固定设计：`protocol.json`；运行环境：`environment.json`。
- 全部逐模型/分段/时距指标：`metrics_all_runs.csv`；三种子均值：`metrics_seed_mean.csv`。
- 每次模型目录保存检查点、训练日志、逐样本概率、标签、索引及代码/数据哈希。
- 每场景保存 `transform.npz`、`case_manifest.json`、原始例子及可读气象词元例子。
- 源码与命令：`../../causal-weathergraph/revision/weather_tokens/README.md`。

本原型的结果尚未自动写入论文正文或作为审稿回复中的创新性结论。
"""
    (OUT/"RESULTS_ZH.md").write_text(report,encoding="utf8")
    evidence={"status":"complete_computation_audit_available" if audit else "complete_computation_pending_independent_audit", "trained_models":len(runs),
        "training_plus_validation_seconds_sum":total_seconds,"allocated_parameters":allocated,
        "peak_torch_gpu_allocated_bytes":peak,"output_bytes_before_report":size,
        "physical_tokens_lower_brier_cells_of_9":int((pure.delta_brier<0).sum()),
        "hybrid_lower_brier_cells_of_9":int((hybrid.delta_brier<0).sum()),
        "artifacts":{str(p.relative_to(OUT)):hash_file(p) for p in [OUT/"protocol.json",OUT/"metrics_all_runs.csv",OUT/"metrics_seed_mean.csv",OUT/"paired_brier_differences.csv",OUT/"transfer_gap_seed_mean.csv",OUT/"RESULTS_ZH.md",dest/"forecast_comparison.png",dest/"forecast_comparison.pdf"]}}
    (OUT/"completion_manifest.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf8")
    print(means[order].to_string())
    print(json.dumps({k:v for k,v in evidence.items() if k!="artifacts"},indent=2))


if __name__=="__main__":main()
