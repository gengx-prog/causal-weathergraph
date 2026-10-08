"""Create the manuscript experiment supplement from audited saved outputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
OUT = BASE / "revision_outputs/transport_consistency_v1"
NAMES = {"ridge_local": "局地历史 Ridge", "ridge_meanflux": "加平均输送", "ridge_meancov": "再加协方差", "ridge_flux_mfc": "再加通量辐合",
         "ridge_samegrid": "同格点 Ridge", "ridge_samegrid_flux": "同格点 Ridge 加物理", "hgb_samegrid": "同格点 HGB", "hgb_samegrid_flux": "同格点 HGB 加物理",
         "persistence": "持续性", "monthly_climatology": "训练月气候态", "horizontal_only_budget": "仅水平趋势", "ridge_spatial_mismatch": "另一半球输送敏感性"}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def table(columns, rows):
    def esc(v): return str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(esc, columns)) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"] + ["| " + " | ".join(map(esc, r)) + " |" for r in rows])


def main():
    audit_path = OUT / "independent_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    if audit["status"] != "PASS_INPUTS_MODELS_EXTERNAL":
        raise RuntimeError("Full independent input/model/external audit must pass before delivery")
    model = json.loads((OUT / "model/run_manifest.json").read_text(encoding="utf8"))
    assert model["status"] == "completed"
    metrics = pd.read_csv(OUT / "model/metrics.csv", dtype={"region_id": str})
    comparisons = pd.read_csv(OUT / "model/paired_comparisons.csv", dtype={"region_id": str})
    external = pd.read_csv(OUT / "external/metrics.csv")
    def m(task, name, region="pooled_equal_regions", sl="all"):
        rows = metrics[(metrics.task == task) & (metrics.model == name) & (metrics.region_id == region) & (metrics["slice"] == sl) & (metrics.split == "test")]
        assert len(rows) == 1
        return rows.iloc[0]
    def c(task, name, baseline, region="pooled_equal_regions", sl="all"):
        rows = comparisons[(comparisons.task == task) & (comparisons.model == name) & (comparisons.baseline == baseline) & (comparisons.region_id == region) & (comparisons["slice"] == sl)]
        assert len(rows) == 1
        return rows.iloc[0]
    main_rows = []
    for task, task_name in [("humidity_change", "6小时比湿变化"), ("cloud_level", "6小时后总云量")]:
        for method in ["ridge", "hgb"]:
            before, after = f"{method}_samegrid", f"{method}_samegrid_flux"
            b, a, contrast = m(task, before), m(task, after), c(task, after, before)
            reduction = 100 * (b.mse - a.mse) / b.mse
            main_rows.append([task_name, method.upper(), f"{b.mse:.6f}", f"{a.mse:.6f}", f"{reduction:+.2f}%", f"[{contrast.exploratory_delta_mse_p025:.6f}, {contrast.exploratory_delta_mse_p975:.6f}]"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.4))
    plot_rows = []
    for ax, task, title, unit in zip(axes, ["humidity_change", "cloud_level"], ["6-hour humidity change", "6-hour cloud cover"], [r"MSE reduction $(g\,kg^{-1})^2$", r"MSE reduction $(percentage\ points)^2$"]):
        for method, color, marker, shift in [("ridge", "#245B80", "o", -.10), ("hgb", "#B86F2A", "s", .10)]:
            for y, region in enumerate(["15", "48", "pooled_equal_regions"]):
                row = c(task, f"{method}_samegrid_flux", f"{method}_samegrid", region)
                estimate = row.delta_mse_baseline_minus_model
                lower, upper = row.exploratory_delta_mse_p025, row.exploratory_delta_mse_p975
                ax.plot([lower, upper], [y + shift, y + shift], color=color, linewidth=1.7)
                ax.scatter([estimate], [y + shift], color=color, marker=marker, s=34, label=method.upper() if y == 0 else None, zorder=3)
                plot_rows.append({"task": task, "method": method, "region_id": region, "estimate": estimate, "lower": lower, "upper": upper})
        ax.axvline(0, color="#555555", linewidth=.8)
        ax.set_yticks(range(3), ["South Atlantic", "North Atlantic", "Equal-region mean"])
        ax.invert_yaxis(); ax.set_xlabel(unit); ax.set_title(title, loc="left", fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="x", alpha=.18); ax.legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle("Added transport features versus the same raw grid information", fontsize=13)
    fig.text(.5, .01, "2019–2025 retrospective evaluation; 2,555 origins per region. Positive favors added physics.\nExploratory 95% paired 30-day block intervals; not causal effects or simultaneous intervals.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(OUT / "transport_skill_comparison.png", dpi=180)
    fig.savefig(OUT / "transport_skill_comparison.pdf")
    plt.close(fig)
    pd.DataFrame(plot_rows).to_csv(OUT / "figure_values.csv", index=False)
    ablation_rows = []
    for name in ["persistence", "monthly_climatology", "horizontal_only_budget", "ridge_local", "ridge_meanflux", "ridge_meancov", "ridge_flux_mfc", "ridge_samegrid", "ridge_samegrid_flux", "hgb_samegrid", "hgb_samegrid_flux", "ridge_spatial_mismatch"]:
        q = m("humidity_change", name)
        cloud = None if name == "horizontal_only_budget" else m("cloud_level", name)
        ablation_rows.append([NAMES[name], f"{q.mse:.6f}", f"{q.mae:.6f}", "不适用" if cloud is None else f"{cloud.mse:.6f}", "不适用" if cloud is None else f"{cloud.mae:.6f}"])
    source_rows = []
    for region, region_name in [("15", "南大西洋"), ("48", "北大西洋")]:
        for method in ["ridge", "hgb"]:
            for sl, sl_name in [("source:1", "WB2段"), ("source:2", "CDS段")]:
                row = c("humidity_change", f"{method}_samegrid_flux", f"{method}_samegrid", region, sl)
                source_rows.append([region_name, method.upper(), sl_name, f"{row.delta_mse_baseline_minus_model:.6f}", f"[{row.exploratory_delta_mse_p025:.6f}, {row.exploratory_delta_mse_p975:.6f}]"])
    external_rows = []
    for region, label in [(15, "南大西洋"), (48, "北大西洋")]:
        for name in ["ridge_samegrid", "ridge_samegrid_flux", "hgb_samegrid", "hgb_samegrid_flux"]:
            e = external[(external.region_id == region) & (external.model == name) & (external.outcome_product == "CERES_hourbox")].iloc[0]
            external_rows.append([label, NAMES[name], int(e.n), f"{e.MAE_pp:.4f}", f"{e.bias_pp:.4f}"])
    q_gain = 100*(m("humidity_change", "ridge_samegrid").mse-m("humidity_change", "ridge_samegrid_flux").mse)/m("humidity_change", "ridge_samegrid").mse
    h_gain = 100*(m("humidity_change", "hgb_samegrid").mse-m("humidity_change", "hgb_samegrid_flux").mse)/m("humidity_change", "hgb_samegrid").mse
    text = rf'''# 水汽输送约束补充实验与审稿回复依据

本文件记录借鉴 Navier–Stokes 文稿中非线性通量与完整残差核查思路后，实际完成的两区域实验。核心发现是：在提供相同原始格点信息的对照下，输送特征改善了未来6小时比湿变化预测；云量预测没有一致改善。实验支持有限的物理特征价值，没有识别风—湿度—云干预效应，也没有验证原论文全部跨区域路径。

## 已完成工作及结果

设计在计算新输送诊断和拟合新模型前冻结。完整处理1979—2025年68,668个六小时时次，在南大西洋与北大西洋两个既有区域分别训练。每区域保留13,148个训练起点、1,460个验证起点和2,555个评价起点。训练截至2014年，验证为2015—2018年，回顾评价为2019—2025年。每个起点为00 UTC，输入仅使用当前、前6小时和前12小时数据，目标为未来6小时。共完成40个拟合模型；未来信息诊断单独保存。

两区域等权汇总中，同格点Ridge加入物理特征后比湿变化MSE降低{q_gain:.2f}%，同格点梯度提升树HGB降低{h_gain:.2f}%。这两个比较以完全相同的原始格点和历史信息为基础；增益不能解释为因果效应，也不能推广为所有海区、季节或模型均改善。云量Ridge改善很小，HGB稍有恶化，两者差异区间均包含零。

{table(["任务", "模型", "基线MSE", "加物理MSE", "MSE相对降低", "MSE减少量探索性95%区间"], main_rows)}

比湿变化MSE单位为(g/kg)²，云量MSE单位为百分点²。相对降低为(基线MSE−加入特征MSE)/基线MSE；正值表示改善。区间来自500次配对30日历日块重采样，在资料来源段内抽样、两区域同步使用相同日历块。区间条件于已拟合模型，不涵盖调参不确定性，不是同时置信区间，也没有证明名义FDR控制。

![同格点信息下增加物理特征的预测比较](transport_skill_comparison.png)

图中全部正负结果均保留。两个区域等权汇总并非面积加权全球表现。

## 数学思路与大气适配

用户提供的166页文稿研究具有特制光滑外力的三维不可压缩Navier–Stokes有限时间爆破。第11—14页通过背景流、扰动、平均二次通量和完整残差重算组织证明。这里借鉴的是保留非线性输送项及逐项核查的原则；没有实现其高频脉冲构造，没有将其爆破定理作为大气因果定理，也没有把区域均值模型称为Navier–Stokes求解器。

在850 hPa格点计算水平水汽通量辐合：

$$M_{{850}}=-\nabla_h\cdot(q\mathbf v_h)=-\frac{{1}}{{a\cos\phi}}\left[\frac{{\partial(qu)}}{{\partial\lambda}}+\frac{{\partial(qv\cos\phi)}}{{\partial\phi}}\right].$$

其中q为比湿(kg/kg)，u、v为水平风(m/s)，a=6,371,000 m，M单位为s⁻¹。采用经度周期边界、相邻格点乘积通量的面插值以及球面有限体积面积，先在原始网格计算，再对区域做面积平均。使用参考格点中心推断边界，记录后段CDS坐标约0.0005°舍入差；没有据此宣称两个来源的重网格算子相同。

进一步采用同一球面面积平均算子分解：

$$\overline{{q\mathbf v_h}}=\bar q\,\overline{{\mathbf v_h}}+\overline{{(q-\bar q)(\mathbf v_h-\overline{{\mathbf v_h}})}}.$$

后一项是本次已解析粗格点间的区域内协方差，不是观测到的亚网格湍流通量，也不是PDF中人为构造的振荡应力。本实验没有对区域平均量直接求空间导数；MFC始终独立由原格点边界面通量计算。

只使用当前MFC构造未拟合比较量21600×1000×MFC(t)，相当于假设水平瞬时趋势维持6小时。它不包含垂直输送、凝结蒸发或其他源汇，不能作为完整湿度预算。该比较量MSE为{m('humidity_change','horizontal_only_budget').mse:.6f}，高于持续性基线{m('humidity_change','persistence').mse:.6f}。实际数据明确显示：不能把单层水平辐合直接当作全部湿度变化。

## 模型公平性与完整消融

两个区域均沿用此前CERES实验的36个完整目标格点，按球面面积平均。每区用于同信息对照的格点为核心36格及东西南北相邻24格，共60格。它们覆盖了计算区域边界MFC所需的全部原始u/v/q/T输入。所有模型共有局地u/v/q/T/云历史、六项物理控制历史和年内日历正余弦项。六控制仍沿用原66区算术均值，此差异明确记录且各模型相同；没有使用旧1979—2018标准化控制数组。

Ridge的中心和尺度仅在1979—2014拟合，alpha在1、10、100、1000中按验证MSE选择，评价期不重新拟合。HGB固定120轮、15叶、最少叶样本50、学习率0.05、L2为1、64个直方图箱，不早停、不用评价资料调参。云量预测统一限制在0—100%，原始未截断预测及截断比例另行保留。模型之间的行完全匹配；跨训练/验证/评价边界和数据来源拼接点的历史或目标窗口均排除。

{table(["模型或比较量", "比湿MSE", "比湿MAE g/kg", "云量MSE", "云量MAE 百分点"], ablation_rows)}

这次目标是下一时刻变化或水平，不与此前同刻风湿云估计器的MAE直接比较。局地、同格点和物理特征模型的比较也不能称为原有图方法已恢复真实物理链。

## 半球和资料来源差异

{table(["区域", "模型", "资料段", "增加物理后的比湿MSE减少量", "探索性95%区间"], source_rows)}

Ridge的比湿改善在两个区域和两个评价资料段的点估计均为正。HGB在北大西洋CDS段的点估计为负，区间包含零；因此不能宣称所有模型和来源都改善。WB2段为2019年至2023年1月10日，CDS段从2023年1月11日起。二者日期与天气条件不同，差异不能归因于获取来源本身。各本地季节结果完整保存在逐组CSV；单一区域对比不构成普遍半球效应。

消融还存在具体反例：北区比湿模型在平均输送之后增加协方差，全期MSE减少量为−0.000161，春季为−0.001014。因此“完整物理特征组合有效”不能改写成每个组成部分均有效。云量同格点Ridge比较的全部区域、来源和季节区间均包含零；HGB北区秋季和两区等权夏季还出现不利区间。全部细项保存在配对比较表，不按结果挑选季节。

## 敏感性诊断的含义

将另一半球同时刻的输送特征加入局地模型，属于空间对应敏感性。另一半球仍可能携带共同强迫、季节或遥相关信息，不能作为可交换的因果零分布。全期两区域等权比湿MSE增益约为{c('humidity_change','ridge_spatial_mismatch','ridge_local').delta_mse_baseline_minus_model:.6f}，未显示与正确局地输送相当的改善；这不是对真实输送路径的认证。

另有显式使用未来6小时通量的诊断，全部输出与正式预测分开。该通量含目标时刻q，因此对湿度变化任务存在直接目标信息泄漏，不能用其改善证明反向因果、作为阴性对照，或作为公平预测性能上界。它没有解决原审稿意见中的反时序关联问题。输入维数也不同于正式三时刻物理模型，相关限制已保留。

## CERES跨产品比较

复用已独立核验的2019年CERES小时数据，严格保持原36格足迹和球面面积权重。每区匹配364个有效预测起点：ERA5云预测有效时间06:00 UTC，对应CERES名义06:30小时标签。2019年1月1日起点因其输入历史跨划分边界而排除。两产品的瞬时值与小时采样算子不同，原始CF解码时刻已保存；没有把30分钟标签差解释为误差已消除。

{table(["区域", "预测模型", "样本数", "对CERES的MAE 百分点", "预测减CERES偏差 百分点"], external_rows)}

所列同格点模型对CERES的差异明显大于对ERA5自身目标的误差，偏差约为−8至−9.6个百分点。加入物理项只带来很小且依指标变化的差异，不能解决产品差异。北区Ridge对CERES的MAE略有恶化；南区月气候态对CERES的MAE为8.2596个百分点，低于所有学习模型。外部表是描述比较，没有配对区间，不能把极小差异称为已证实提升。表内的差异不是相对于真实云量的测量误差；2019资料已用于前期研究，不是从未查看过的确认集。全部11个可实施云预测及敏感性模型均已保存，合计8,008行匹配和44行分产品指标。新增全球CERES数据尚未用于本次两区实验。

## 对审稿意见的直接回答

**R1-I2与R3-5：标量风缺少输送方向与物理支持。** 已新增由格点u/v/q形成的实际方向通量和水平辐合，而非只对风速重新命名。该诊断有明确单位、球面算子及守恒核查；仍为850 hPa代理，与整层云量存在垂直支持差异。

**R2-4：拼接两条统计边不足以证明传播或中介。** 本轮结果提供局地湿度变化预测与输送量的一致性检验，但未重建空气轨迹、未闭合整层水汽预算，也没有重新认证旧60条跨区域链。云量未一致改善进一步限制了完整WHC机制主张。

**R2-5：基线、消融与验证。** 已加入原始格点及历史信息完全一致的Ridge/HGB对照，冻结训练与验证规则，保留正负结果、来源分段及配对不确定性。CERES提供限定年度和足迹的跨产品检查，不能提供真实因果图标签。

**R3-6：共同驱动与方向识别。** 六物理控制在各模型中相同，但不构成充分混杂控制证明。本轮未来通量诊断含目标时刻信息，不能替代原问题要求的有效反证或识别条件；原方向性限制继续保留。

可用于回复的英文补充如下：

> I have added a bounded transport-consistency experiment on the two previously fixed Atlantic footprints. Horizontal moisture-flux convergence is computed on the original resolved grid using a spherical finite-volume operator before area averaging. The experiment distinguishes mean-flow transport from resolved within-region humidity–wind covariance and compares these features against models supplied with the same raw grid cells and histories. This is a diagnostic of local six-hour humidity and cloud prediction, not a validation of every previously enumerated cross-region pathway.

> In the retrospective 2019–2025 evaluation, adding the transport features reduces the equally weighted humidity-change MSE by {q_gain:.2f}% for ridge regression and {h_gain:.2f}% for the fixed gradient-boosting baseline. The cloud comparisons do not show a consistent improvement. A horizontal-tendency-only comparator performs worse than persistence, which demonstrates the inadequacy of treating the single-level horizontal term as a closed moisture budget. All regional, acquisition-segment and seasonal results are retained. The paired block intervals are exploratory and do not establish graph-wide error control or intervention effects.

> The supplementary 2019 CERES comparison uses identical spatial footprints but explicitly different temporal operators: ERA5 forecast validity at 06:00 UTC and the CERES nominal 06:30 hourly slot. It shows substantial cross-product offsets and only small changes after adding transport features. This is not an assessment against cloud truth or independent causal graph validation. The physical diagnostics therefore strengthen the description of the predictive information while preserving the limitations on vertical support, unobserved processes and causal identification.

## 核验范围与交付状态

独立审计状态：{audit['status']}。生产输入检查覆盖全部68668时次；独立复算按审计清单区分派生量全量核对与原始NetCDF抽样核对，不以相同函数重复调用代替独立证明。核验报告和可运行脚本随结果保存。数值通过仅证明受检实现及计算一致，不证明物理因果、预算闭合或名义推断校准。

英文方法片段已保存为 `manuscript/revised/transport_consistency_methods.tex`，配套结果片段为 `transport_consistency_results.tex`。本轮没有将全部主文稿改成新方法论文；现有主文稿、完整参考文献、全篇语言润色及最终修订标记仍需整体整合。新增材料中的所有实验表述均以实际完成输出为准。

此前新增下载已完成：全球CERES共23文件、22,814,957,704字节；ERA5同日期重叠包共30项、目录占用1,437,305,693字节。CERES后台文件校验通过，ERA5重叠包仍待全值审计和网格处理；下载完成不等于这些全球/来源对照实验已经完成。

## 可复现证据与文献

- `design.json`：计算前冻结的实验设计，SHA256 `{sha(OUT/'design.json')}`。
- `inputs/input_manifest.json` 与 `inputs/input_data.npz`：原始文件哈希、单位、区域、面积与通量检查。
- `model/sample_ledger.csv`、`model/run_manifest.json`：全部候选起点、排除原因、模型配置与输出哈希。
- `model/metrics.csv`、`model/paired_comparisons.csv`：全部正式与空间敏感性结果；`model/diagnostic_*.csv` 单独保存未来信息诊断。
- `model/predictions.csv.gz`：全部正式预测；`model/bootstrap_indices.npz`：固定配对块抽样。
- `external/paired_predictions.csv` 与 `external/manifest.json`：逐样本跨产品配对、原始CERES解码时刻及几何检查。
- `independent_audit.json`：独立复算范围、误差与代码哈希；`figure_values.csv`：图中全部数值。
- 本实验采用的水汽通量辐合依据：[Banacos and Schultz (2005)](https://doi.org/10.1175/WAF858.1)。该诊断本身不是本研究提出的新物理定律。
- 云形成和云量还涉及输送、对流、凝结及蒸发等过程：[ECMWF 云过程说明](https://confluence.ecmwf.int/spaces/FUG/pages/673550490/Section%2B2A.1.5.2%2BClouds)。
- 数学思路出处：[OpenAI Navier–Stokes研究](https://openai.com/index/navier-stokes-solution/)。未将其有限时爆破结论用于本研究的现实大气因果识别，也未默认加入最终参考文献。
'''
    # Write the result module from the same numeric source as the report.
    result_tex = ROOT / "manuscript/revised/transport_consistency_results.tex"
    result_tex.write_text(r"\subsection{Bounded transport-consistency results}" + "\n" +
        f"The retrospective evaluation retained 2,555 daily origins in each of the two Atlantic regions. With identical raw-grid information and histories, adding transport features reduced the equally weighted six-hour specific-humidity-change MSE from {m('humidity_change','ridge_samegrid').mse:.6f} to {m('humidity_change','ridge_samegrid_flux').mse:.6f} $(\\mathrm{{g}}\\,\\mathrm{{kg}}^{{-1}})^2$ for ridge regression ({q_gain:.2f}\\%), and from {m('humidity_change','hgb_samegrid').mse:.6f} to {m('humidity_change','hgb_samegrid_flux').mse:.6f} for gradient boosting ({h_gain:.2f}\\%). The corresponding paired reductions had exploratory 95\\% block intervals [0.001389, 0.002315] and [0.000330, 0.001041]. These are local forecast comparisons, not recovery rates for the previously enumerated cross-region pathways.\n\n" +
        "Cloud prediction did not improve consistently: the pooled MSE change favored the augmented ridge model by 0.073355 percentage-point squared but favored the unaugmented gradient-boosting model by 0.027166; both exploratory intervals included zero. The unfitted horizontal-tendency comparator had humidity-change MSE 0.043341, exceeding persistence at 0.036201. Thus the horizontal term alone is not an adequate moisture budget. Gradient-boosting gains also failed to persist in the northern later-acquisition segment.\n\n" +
        "The 2019 CERES comparison retained 364 origins per region over unchanged area-weighted footprints. Forecasts valid at 06:00 UTC were compared with nominal 06:30 CERES hourly slots, preserving the difference in temporal operators. The same-grid models showed approximately 8--9.6 percentage-point negative offsets relative to CERES and only small changes after augmentation. These cross-product differences are not errors against cloud truth or independent causal validation. Missing vertical moisture transport, source--sink terms, acquisition differences and incomplete causal adjustment remain limitations.\n", encoding="utf8")
    md = OUT / "水汽输送补充实验与审稿回复依据.md"
    md.write_text(text, encoding="utf8")
    # Retain the existing scientific-document typography; no cloud publication.
    sys.path.insert(0, str(ROOT))
    from revision.build_response_document import create_reference
    reference = OUT / "document_style.docx"
    create_reference(reference)
    docx = OUT / "水汽输送补充实验与审稿回复依据.docx"
    subprocess.run([str(Path.home()/"AppData/Local/Pandoc/pandoc.exe"), str(md), "-o", str(docx),
                    "--from=markdown+tex_math_dollars", "--standalone", f"--reference-doc={reference}", f"--resource-path={OUT}"], check=True)
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.oxml import OxmlElement
    doc = Document(docx)
    doc.core_properties.title = "水汽输送约束补充实验与审稿回复依据"
    doc.core_properties.subject = "Bounded physical transport experiment for the major revision"
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading"):
            p.paragraph_format.keep_with_next = True
    for t in doc.tables:
        for row_index, row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(3)
                    if row_index == 0:
                        p.paragraph_format.keep_with_next = True
                    for run in p.runs: run.font.size = Pt(9)
    doc.save(docx)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "independent_audit_sha256": sha(audit_path),
                "script_sha256": sha(__file__), "files": {p.name: sha(p) for p in [md, docx, OUT/"transport_skill_comparison.png", OUT/"transport_skill_comparison.pdf", OUT/"figure_values.csv"]},
                "source_files": {str(p.relative_to(OUT)): sha(p) for p in [OUT/"model/run_manifest.json", OUT/"model/metrics.csv", OUT/"model/paired_comparisons.csv", OUT/"external/metrics.csv"]},
                "pdf_export_status": "pending", "main_manuscript_integration": "Separate methods/results modules only; full manuscript and bibliography integration pending"}
    (OUT/"document_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"docx": str(docx), "md": str(md), "figure": str(OUT/"transport_skill_comparison.png")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
