#!/usr/bin/env python3
"""Build a paper-structured Chinese report ordered by standard paper sections."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "outputs" / "reports"
FIGURE_DIR = REPO_ROOT / "outputs" / "figures" / "top_conference"
OUTPUT_PDF = REPORT_DIR / "Causal_WeatherGraph_论文六段式结构化报告_1979_2025.pdf"
OUTPUT_MD = REPORT_DIR / "causal_weathergraph_structured_paper_report_1979_2025_zh.md"

FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else FontProperties()

LANDSCAPE = (13.8, 8.5)
INK = "#17202A"
MUTED = "#667085"
BLUE = "#2C7FB8"
ORANGE = "#D95F02"
GREEN = "#009E73"
PURPLE = "#7570B3"
RED = "#B94C5C"
PAPER = "#FBFCFE"


SECTION_COLORS = {
    "Introduction": BLUE,
    "Related Work": PURPLE,
    "Methodology": GREEN,
    "Experiments": ORANGE,
    "Discussion": RED,
    "Conclusion": INK,
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.family": "sans-serif",
        }
    )


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / name)


def visual_width(char: str) -> int:
    return 1 if ord(char) < 128 else 2


def wrap_mixed(text: str, max_width: int) -> list[str]:
    """Wrap mixed Chinese/English text by approximate visual width."""
    lines: list[str] = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            lines.append("")
            continue
        line = ""
        width = 0
        for token in re.findall(r"[A-Za-z0-9_\-\./:]+|\s+|.", raw):
            token_width = sum(visual_width(ch) for ch in token)
            if token.isspace():
                if line and not line.endswith(" "):
                    token = " "
                    token_width = 1
                else:
                    continue
            if width + token_width > max_width and line:
                lines.append(line.rstrip())
                line = token.lstrip()
                width = sum(visual_width(ch) for ch in line)
            else:
                line += token
                width += token_width
        if line:
            lines.append(line.rstrip())
    return lines


def add_wrapped(
    ax: plt.Axes,
    text: str,
    x: float,
    y: float,
    *,
    max_width: int,
    fontsize: float = 8.8,
    color: str = INK,
    line_height: float = 0.043,
    weight: str | None = None,
) -> float:
    cur_y = y
    for line in wrap_mixed(text, max_width):
        if line == "":
            cur_y -= line_height * 0.65
            continue
        ax.text(
            x,
            cur_y,
            line,
            ha="left",
            va="top",
            fontsize=fontsize,
            color=color,
            weight=weight,
            fontproperties=FONT,
            transform=ax.transAxes,
        )
        cur_y -= line_height
    return cur_y


def add_bullets(
    ax: plt.Axes,
    bullets: list[str],
    x: float,
    y: float,
    *,
    max_width: int,
    color: str = INK,
    bullet_color: str = BLUE,
    fontsize: float = 8.8,
) -> float:
    cur_y = y
    for bullet in bullets:
        ax.text(x, cur_y, "■", ha="left", va="top", fontsize=6.3, color=bullet_color, fontproperties=FONT, transform=ax.transAxes)
        cur_y = add_wrapped(ax, bullet, x + 0.025, cur_y + 0.002, max_width=max_width, fontsize=fontsize, color=color, line_height=0.039)
        cur_y -= 0.012
    return cur_y


def add_header(fig: plt.Figure, section: str, title: str, subtitle: str | None = None) -> None:
    color = SECTION_COLORS.get(section, BLUE)
    fig.text(0.045, 0.955, section, ha="left", va="top", fontsize=11, weight="bold", color=color, fontproperties=FONT)
    fig.text(0.045, 0.918, title, ha="left", va="top", fontsize=18, weight="bold", color=INK, fontproperties=FONT)
    if subtitle:
        fig.text(0.045, 0.875, subtitle, ha="left", va="top", fontsize=9.0, color=MUTED, fontproperties=FONT)
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.852, 0.852], transform=fig.transFigure, color="#D0D7E2", linewidth=0.8))


def add_footer(fig: plt.Figure, label: str) -> None:
    fig.text(0.045, 0.035, "Causal WeatherGraph | paper-structured report | 1979-2025", ha="left", va="bottom", fontsize=7.6, color=MUTED, fontproperties=FONT)
    fig.text(0.955, 0.035, label, ha="right", va="bottom", fontsize=7.6, color=MUTED, fontproperties=FONT)


def metrics() -> dict[str, float | int | str]:
    summary = read_csv("summary_metrics.csv")
    chain = read_csv("causal_chain_summary.csv")
    lat = read_csv("latitude_band_chain_summary.csv")
    null = read_csv("null_model_summary.csv")
    enrich = read_csv("null_model_variable_preserved_summary.csv")
    boot = read_csv("bootstrap_stability_summary.csv")
    period = read_csv("period_stability_summary.csv")
    direction = read_csv("directionality_control_summary.csv")
    lag = read_csv("targeted_maxlag12_lag_summary.csv")

    physical = null.loc[null["edge_set"] == "physical_candidates"].iloc[0]
    random = null.loc[null["edge_set"] == "random_edges"].iloc[0]
    variable = null.loc[null["edge_set"] == "variable_preserved_random"].iloc[0]
    all_summary = summary.loc[summary["regime"] == "all"].iloc[0]
    all_chain = chain.loc[chain["regime"] == "all"].iloc[0]
    mid = lat.loc[lat["latitude_band"] == "midlatitude"].iloc[0]
    wh_enrich = enrich.loc[enrich["edge_type"] == "wind->humidity", "enrichment"].iloc[0]
    hc_enrich = enrich.loc[enrich["edge_type"] == "humidity->cloud", "enrichment"].iloc[0]
    wh_boot = boot.loc[boot["edge_type"] == "wind_to_humidity"].iloc[0]
    hc_boot = boot.loc[boot["edge_type"] == "humidity_to_cloud_cover"].iloc[0]
    shuffled = direction.loc[direction["test_type"] == "time_shuffled", "significant_ratio"].mean()
    shifted = direction.loc[direction["test_type"] == "circular_shift", "significant_ratio"].mean()
    lag_48 = lag["lag_4_8_frac"].mean()

    return {
        "all_edges": int(all_summary["n_significant_edges"]),
        "all_chains": int(all_chain["n_chains"]),
        "physical_ratio": float(physical["significant_ratio"]),
        "random_ratio": float(random["significant_ratio"]),
        "variable_ratio": float(variable["significant_ratio"]),
        "wh_enrich": float(wh_enrich),
        "hc_enrich": float(hc_enrich),
        "stable_wh": int(wh_boot["stable_edges"]),
        "stable_hc": int(hc_boot["stable_edges"]),
        "mid_chains": int(mid["WHC_chains"]),
        "mid_stable_chains": int(mid["stable_WHC_chains"]),
        "mid_score": float(mid["mean_chain_score"]),
        "period_overlap_min": float(period["overlap_with_full"].min()),
        "period_overlap_max": float(period["overlap_with_full"].max()),
        "time_shuffled_ratio": float(shuffled),
        "circular_shift_ratio": float(shifted),
        "lag_4_8_frac": float(lag_48),
    }


def cover_page(pdf: PdfPages, m: dict[str, float | int | str]) -> None:
    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor("#F7F9FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0.045, 0.16), 0.018, 0.68, color=BLUE, transform=fig.transFigure, clip_on=False))
    fig.text(0.09, 0.78, "Causal WeatherGraph", ha="left", va="top", fontsize=33, weight="bold", color=INK, fontproperties=FONT)
    fig.text(0.09, 0.69, "论文六段式结构化图文报告", ha="left", va="top", fontsize=23, weight="bold", color=INK, fontproperties=FONT)
    fig.text(
        0.09,
        0.59,
        "按 Introduction, Related Work, Methodology, Experiments,\nDiscussion, Conclusion 的顺序组织当前实验图片与结果叙事。",
        ha="left",
        va="top",
        fontsize=12.5,
        color=MUTED,
        linespacing=1.55,
        fontproperties=FONT,
    )

    tiles = [
        ("1979-2025", "data range", BLUE),
        (f"{int(m['all_edges']):,}", "all-regime significant edges", ORANGE),
        (f"{int(m['all_chains']):,}", "all-regime WHC chains", GREEN),
        (f"{float(m['physical_ratio']):.3f}", "physical candidate sig. ratio", PURPLE),
    ]
    for idx, (value, label, color) in enumerate(tiles):
        x = 0.09 + idx * 0.205
        ax.add_patch(plt.Rectangle((x, 0.28), 0.165, 0.125, facecolor="white", edgecolor="#D0D7E2", linewidth=0.8, transform=fig.transFigure))
        ax.add_patch(plt.Rectangle((x, 0.28), 0.01, 0.125, facecolor=color, edgecolor="none", transform=fig.transFigure))
        fig.text(x + 0.022, 0.365, value, ha="left", va="top", fontsize=17, weight="bold", color=INK, fontproperties=FONT)
        fig.text(x + 0.022, 0.317, label, ha="left", va="top", fontsize=8.0, color=MUTED, fontproperties=FONT)

    fig.text(0.09, 0.13, f"PDF: {OUTPUT_PDF.relative_to(REPO_ROOT)}", ha="left", va="bottom", fontsize=8.2, color=MUTED, fontproperties=FONT)
    pdf.savefig(fig)
    plt.close(fig)


def section_text_page(
    pdf: PdfPages,
    section: str,
    title: str,
    subtitle: str,
    paragraphs: list[str],
    bullets_title: str,
    bullets: list[str],
    page_label: str,
) -> None:
    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor("white")
    add_header(fig, section, title, subtitle)
    ax = fig.add_axes([0.055, 0.10, 0.89, 0.71])
    ax.axis("off")

    y = 0.96
    for paragraph in paragraphs:
        y = add_wrapped(ax, paragraph, 0.0, y, max_width=94, fontsize=9.0, color=INK, line_height=0.043)
        y -= 0.035

    ax.add_patch(plt.Rectangle((0.0, 0.02), 1.0, 0.25, facecolor=PAPER, edgecolor="#D0D7E2", linewidth=0.8, transform=ax.transAxes))
    ax.text(0.025, 0.235, bullets_title, ha="left", va="top", fontsize=10.2, weight="bold", color=INK, fontproperties=FONT, transform=ax.transAxes)
    add_bullets(ax, bullets, 0.025, 0.185, max_width=82, bullet_color=SECTION_COLORS.get(section, BLUE), fontsize=8.25)
    add_footer(fig, page_label)
    pdf.savefig(fig)
    plt.close(fig)


def figure_page(
    pdf: PdfPages,
    section: str,
    title: str,
    subtitle: str,
    image_file: str,
    notes: list[str],
    page_label: str,
) -> None:
    path = FIGURE_DIR / image_file
    if not path.exists():
        raise FileNotFoundError(path)

    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor("white")
    add_header(fig, section, title, subtitle)

    img = Image.open(path).convert("RGB")
    ax_img = fig.add_axes([0.045, 0.235, 0.91, 0.59])
    ax_img.axis("off")
    ax_img.imshow(img)

    ax_note = fig.add_axes([0.055, 0.085, 0.89, 0.11])
    ax_note.axis("off")
    ax_note.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=PAPER, edgecolor="#D0D7E2", linewidth=0.8, transform=ax_note.transAxes))
    ax_note.text(0.02, 0.78, "本节使用方式", ha="left", va="top", fontsize=9.2, weight="bold", color=INK, fontproperties=FONT, transform=ax_note.transAxes)
    y = 0.76
    for note in notes:
        ax_note.text(0.16, y, "■", ha="left", va="top", fontsize=5.8, color=SECTION_COLORS.get(section, BLUE), fontproperties=FONT, transform=ax_note.transAxes)
        y = add_wrapped(ax_note, note, 0.18, y + 0.002, max_width=111, fontsize=7.7, color=MUTED, line_height=0.26)
    add_footer(fig, page_label)
    pdf.savefig(fig)
    plt.close(fig)


def experiment_table_page(pdf: PdfPages, m: dict[str, float | int | str]) -> None:
    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor("white")
    add_header(fig, "Experiments", "实验设计与结果摘要", "把实验组织成主结果、反证、稳健性和机制扩展四组证据。")
    ax = fig.add_axes([0.055, 0.12, 0.89, 0.68])
    ax.axis("off")

    rows = [
        ("Main graph", f"all regime: {int(m['all_edges']):,} significant edges; {int(m['all_chains']):,} WHC chains", "证明核心 pathway 存在足够强的统计支撑"),
        ("Null models", f"physical ratio {float(m['physical_ratio']):.3f}; random {float(m['random_ratio']):.3f}; variable-preserved {float(m['variable_ratio']):.3f}", "排除完全随机边和变量类型保持随机配对"),
        ("Path enrichment", f"WH {float(m['wh_enrich']):.2f}x; HC {float(m['hc_enrich']):.2f}x", "说明核心物理路径相对随机区域配对显著富集"),
        ("Bootstrap", f"stable WH {int(m['stable_wh']):,}; stable HC {int(m['stable_hc']):,}", "强调跨时间块可复现性"),
        ("Latitude bands", f"midlatitude WHC {int(m['mid_chains']):,}; stable {int(m['mid_stable_chains']):,}; score {float(m['mid_score']):.3f}", "支撑中纬度天气尺度过程解释"),
        ("Robustness", f"period overlap {float(m['period_overlap_min']):.3f}-{float(m['period_overlap_max']):.3f}", "说明不是单一时期、预处理或区域划分偶然性"),
        ("Temporal controls", f"time-shuffled mean ratio {float(m['time_shuffled_ratio']):.3f}; circular-shift {float(m['circular_shift_ratio']):.3f}", "证明真实时间对齐对结果重要"),
        ("Long lag", f"lag 4-8 mean fraction {float(m['lag_4_8_frac']):.3f}", "支持 24-48 小时尺度的长尾依赖"),
    ]
    df = pd.DataFrame(rows, columns=["Experiment block", "Key number", "Paper role"])
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left", bbox=[0, 0.02, 1, 0.92])
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7E2")
        cell.set_linewidth(0.65)
        cell.PAD = 0.13
        cell.get_text().set_fontproperties(FONT)
        if row == 0:
            cell.set_facecolor(ORANGE)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F9FC")
        else:
            cell.set_facecolor("white")
        if col == 0 and row > 0:
            cell.get_text().set_weight("bold")
            cell.get_text().set_color(INK)
    add_footer(fig, "Experiments summary")
    pdf.savefig(fig)
    plt.close(fig)


def markdown_report(m: dict[str, float | int | str]) -> str:
    return f"""# Causal WeatherGraph 论文六段式结构化报告（1979-2025）

## 1. Introduction

全球大气系统具有强烈的时空耦合，风场、水汽和云量之间的滞后依赖既包含局地响应，也包含区域输送和天气尺度系统传播。本文关注的问题不是单纯提高天气预报精度，而是从 1979-2025 年 6 小时再分析数据中构建一个 regime-aware 的 Causal WeatherGraph，用图结构描述 Wind -> Humidity -> Cloud Cover 的稳定滞后 directed dependency pathway。

本文核心贡献可以概括为三点：第一，把全球大气变量转化为区域级滞后依赖图；第二，在不同 regime 下分析 WHC pathway 的结构变化；第三，通过 null model、bootstrap、direction controls、preprocessing/region robustness 和 long-lag tests 构成反证与稳健性证据链。

## 2. Related Work

相关工作可以分为四类。第一类是 precipitation nowcasting 和 earth-system forecasting，例如 ConvLSTM、Earthformer、PreDiff、spatiotemporal video diffusion 等，它们强调从时空序列中获得高质量预测。第二类是图神经天气预报和 hierarchical graph weather models，它们证明图结构适合表达球面大气状态之间的空间依赖。第三类是 Transformer 和 diffusion weather forecasting，它们重视多尺度时空建模与不确定性。第四类是气候/天气中的因果发现和滞后依赖分析，关注变量之间是否存在具有物理一致性的方向性关系。

本文区别于多数 forecasting work：本文不把预测分数作为唯一目标，而是把可解释的区域级 lagged directed dependency graph 作为主要对象，并通过 regime-aware 和 robustness analyses 说明 WHC pathway 的稳定性与边界。

## 3. Methodology

数据范围为 1979-2025 年，时间分辨率为 6 小时。核心变量包括 temperature、humidity、wind 和 cloud_cover。方法流程为：区域聚合 -> 候选边构造 -> Granger-style OLS 滞后依赖检验 -> FDR 校正 -> pathway 统计 -> 稳健性与反证检验。

主实验使用 66 个区域，candidate_k_nearest=2，max_lag=3，即 6-18 小时。边的含义是：在控制目标变量自身滞后后，源变量滞后项对目标变量仍提供增量预测信息。本文应把这些边称为 lagged directed dependencies，避免写成严格干预因果。

对应图：`fig01_evidence_stack` 和 `fig02_spatial_pathway`。

## 4. Experiments

主图结果显示 all regime 中共有 {int(m['all_edges']):,} 条显著边和 {int(m['all_chains']):,} 条 WHC chains。physical candidate graph 的显著比例为 {float(m['physical_ratio']):.3f}，高于 random edges 的 {float(m['random_ratio']):.3f} 和 variable-preserved random 的 {float(m['variable_ratio']):.3f}。变量保持随机对照下，Wind -> Humidity 富集为 {float(m['wh_enrich']):.2f}x，Humidity -> Cloud 富集为 {float(m['hc_enrich']):.2f}x。

Bootstrap 结果保留 stable WH 边 {int(m['stable_wh']):,} 条、stable HC 边 {int(m['stable_hc']):,} 条。纬度带分析显示中纬度 WHC chains 为 {int(m['mid_chains']):,}，stable WHC chains 为 {int(m['mid_stable_chains']):,}，mean chain score 为 {float(m['mid_score']):.3f}。历史时期 overlap with full graph 为 {float(m['period_overlap_min']):.3f}-{float(m['period_overlap_max']):.3f}。

对应图：`fig03_regime_mechanism`、`fig04_controls`、`fig05_robustness` 和 `fig06_temporal_mechanism`。

## 5. Discussion

这些结果支持一个谨慎但有力的解释：Wind -> Humidity -> Cloud pathway 在观测再分析数据中具有稳定、非随机、跨时期可复现的滞后 directed dependency 结构。中纬度增强与天气尺度系统中的水汽输送和云量响应相一致。

同时，本文必须保留解释边界。reverse 和 future-to-past controls 提示风、湿度、云量存在强耦合或共同驱动背景，因此结果不能写成“证明 wind 干预导致 humidity，再导致 cloud cover”。更准确的表达是：该图识别出与物理过程一致的 lagged directed dependencies，并通过多组对照证明它不是简单随机边、空间近邻或时间打乱造成的伪结构。

## 6. Conclusion

本文提出 Causal WeatherGraph，用于从 1979-2025 全球 6 小时大气再分析数据中识别 regime-aware 的区域级滞后依赖图。结果表明，Wind -> Humidity -> Cloud pathway 在显著性、富集性、bootstrap 稳定性、历史时期稳健性、预处理/区域划分稳健性和 long-lag 扩展中均得到支持。

未来工作可以进一步引入 PCMCI/Tigramite 完整验证、非线性条件独立检验、真实地形/海陆分层、季节内尺度机制分析，以及更严格的干预或准实验设计。
"""


def build_report() -> None:
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    m = metrics()

    with PdfPages(OUTPUT_PDF) as pdf:
        cover_page(pdf, m)
        section_text_page(
            pdf,
            "Introduction",
            "研究动机与问题定义",
            "从预测导向转向可解释的全球大气滞后依赖图。",
            [
                "全球大气变量之间的关系具有强时空耦合：风场影响水汽输送，湿度变化影响云量响应，而这些过程会随着季节、湿度状态和云量状态发生 regime-dependent 的结构变化。",
                "本文的目标不是只追求预测精度，而是构建一个可以解释区域级滞后 directed dependencies 的 Causal WeatherGraph，重点回答 Wind -> Humidity -> Cloud Cover pathway 是否稳定、是否非随机、是否跨 regime 和跨时期保持一致。",
            ],
            "Introduction 中建议强调",
            [
                "研究对象：1979-2025 年全球 6 小时大气再分析数据。",
                "科学问题：能否从观测数据中识别物理一致的 WHC 滞后依赖链条。",
                "贡献表达：regime-aware graph、pathway enrichment、bootstrap stability、robustness suite。",
            ],
            "1 Introduction",
        )
        section_text_page(
            pdf,
            "Related Work",
            "相关工作与本文定位",
            "把本文放在 weather forecasting、graph weather models 和 causal discovery 的交叉处。",
            [
                "已有 precipitation nowcasting 和 earth-system forecasting 工作，如 ConvLSTM、Earthformer、PreDiff、spatiotemporal video diffusion 等，主要展示模型从时空序列中预测未来天气状态的能力。",
                "图神经天气预报和 hierarchical graph neural weather models 证明了图结构适合表达球面大气变量之间的空间依赖。Transformer 和 diffusion weather forecasting 进一步强调多尺度建模和不确定性表达。",
                "与这些工作不同，本文把区域级 lagged dependency graph 本身作为研究对象，关注可解释 pathway、regime 差异和稳健性反证，而不是把预测指标作为唯一贡献。",
            ],
            "Related Work 中建议分组写",
            [
                "Spatiotemporal forecasting：ConvLSTM、Earthformer、PreDiff、STVD 等。",
                "Graph-based weather models：GraphCast / hierarchical graph forecasting 一类工作。",
                "Causal discovery for climate/weather：Granger-style、PCMCI、条件独立与滞后依赖分析。",
                "本文定位：从 forecasting representation 转向 interpretable regime-aware causal weather graph。",
            ],
            "2 Related Work",
        )
        figure_page(
            pdf,
            "Methodology",
            "方法总览：从数据到证据链",
            "Fig.1 展示 Causal WeatherGraph 的数据处理、图构造、检验和稳健性证据栈。",
            "fig01_evidence_stack.png",
            [
                "正文可先解释数据、区域聚合、候选边和 Granger-style 滞后检验，再说明 null/bootstrapping/direction controls 如何形成证据链。",
                "注意把 edge 定义为 lagged directed dependency，不直接写成干预因果。",
            ],
            "3 Methodology | Fig.1",
        )
        figure_page(
            pdf,
            "Methodology",
            "空间图与 pathway 统计",
            "Fig.2 连接方法输出和物理解释：稳定边在空间上形成 WH 与 HC 两步路径。",
            "fig02_spatial_pathway.png",
            [
                "这里可以介绍区域节点、稳定边、纬度带分层和 band-to-band transfer 的计算方式。",
                "中纬度链条数量与得分最高，是后续 discussion 的物理解释入口。",
            ],
            "3 Methodology | Fig.2",
        )
        experiment_table_page(pdf, m)
        figure_page(
            pdf,
            "Experiments",
            "Regime-aware 实验结果",
            "Fig.3 说明不同 regime 下边类型密度、WHC 链条和图结构距离均存在可辨别变化。",
            "fig03_regime_mechanism.png",
            [
                "正文应把 regime analysis 作为主实验之一，而不是附加探索。",
                "样本量平衡结果用于回应 high/normal regime 差异是否仅由样本数造成。",
            ],
            "4 Experiments | Fig.3",
        )
        figure_page(
            pdf,
            "Experiments",
            "Null、bootstrap 与时间方向对照",
            "Fig.4 汇总非随机性、路径富集、bootstrap 稳定性和 temporal-direction controls。",
            "fig04_controls.png",
            [
                "这张图适合作为主文强证据图，用来回应随机边、变量类型、时间打乱等审稿问题。",
                f"可在文字中点出 WH/HC enrichment 分别为 {float(m['wh_enrich']):.2f}x 和 {float(m['hc_enrich']):.2f}x。",
            ],
            "4 Experiments | Fig.4",
        )
        figure_page(
            pdf,
            "Experiments",
            "稳健性实验",
            "Fig.5 覆盖 candidate-k、历史时期、预处理和区域聚合方式。",
            "fig05_robustness.png",
            [
                "这张图说明核心 pathway 不是某个 candidate_k、某个时期、某个预处理方案或某个区域划分的偶然产物。",
                f"历史时期 overlap with full graph 为 {float(m['period_overlap_min']):.3f}-{float(m['period_overlap_max']):.3f}。",
            ],
            "4 Experiments | Fig.5",
        )
        figure_page(
            pdf,
            "Discussion",
            "时间尺度、中介解释与极端事件前兆",
            "Fig.6 支持对 WHC pathway 的机制讨论，同时保留解释边界。",
            "fig06_temporal_mechanism.png",
            [
                "long-lag 结果说明 lag 1 峰值和 24-48 小时长尾可以共同存在。",
                "humidity mediation 结果支持部分中介，但不支持把 Wind -> Cloud 完全解释为单一湿度中介路径。",
            ],
            "5 Discussion | Fig.6",
        )
        section_text_page(
            pdf,
            "Discussion",
            "解释边界与论文表述策略",
            "强结论需要稳健性支撑，同时必须避免过度因果化。",
            [
                "结果支持的强表述是：Causal WeatherGraph 在 1979-2025 年全球再分析数据中识别出稳定、非随机、跨时期可复现的 Wind -> Humidity -> Cloud lagged directed dependency pathway。",
                "结果不支持的过强表述是：本文已经证明 wind 对 humidity 或 humidity 对 cloud cover 的干预因果效应。reverse 和 future-to-past controls 说明变量之间存在强耦合或共同驱动，因此需要在 discussion 中明确解释边界。",
                "最适合写入论文的表述是：这些边是 physically consistent lagged directed dependencies，在 null models、bootstrap stability、temporal controls 和 robustness analyses 下保持一致。",
            ],
            "Discussion 中建议主动说明",
            [
                "观测数据不等同于干预实验。",
                "空间近邻相关是强背景，因此 distance-matched random 是必要对照。",
                "time-shuffled 几乎破坏结构，说明真实时间对齐很重要。",
                "future work 可加入 PCMCI、非线性检验、海陆/地形分层和准实验设计。",
            ],
            "5 Discussion",
        )
        section_text_page(
            pdf,
            "Conclusion",
            "结论与未来工作",
            "用一段清晰结论收束方法、实验和物理解释。",
            [
                "本文提出 Causal WeatherGraph，用于从全球 6 小时大气再分析数据中构建 regime-aware 的区域级滞后依赖图。实验显示 Wind -> Humidity -> Cloud pathway 在主图、null model、bootstrap、direction controls、robustness checks 和 long-lag extension 中均得到支持。",
                f"关键实验证据包括：all regime 中 {int(m['all_edges']):,} 条显著边、{int(m['all_chains']):,} 条 WHC chains、physical candidate significant ratio {float(m['physical_ratio']):.3f}，以及中纬度 mean chain score {float(m['mid_score']):.3f}。",
                "未来工作可以扩展到更严格的条件独立因果发现、非线性滞后依赖、地理分层、极端天气案例研究，以及与图神经天气模型的可解释性连接。",
            ],
            "Conclusion 中建议保留三句话",
            [
                "一句话概括方法：regime-aware Causal WeatherGraph。",
                "一句话概括发现：稳定的 WHC lagged directed dependency pathway。",
                "一句话概括边界和未来：观测依赖不是干预因果，未来需要更强验证。",
            ],
            "6 Conclusion",
        )

    OUTPUT_MD.write_text(markdown_report(m), encoding="utf-8")
    print(f"Saved PDF: {OUTPUT_PDF}")
    print(f"Saved Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    build_report()
