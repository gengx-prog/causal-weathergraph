#!/usr/bin/env python
"""Build paper-ready experiment tables, composite figures, and a PDF bundle."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "outputs" / "reports"
FIGURE_DIR = REPO_ROOT / "outputs" / "figures"
OUTPUT_PDF = REPORT_DIR / "Causal_WeatherGraph_论文实验图表汇编_1979_2025.pdf"

LANDSCAPE = (11.69, 8.27)
PORTRAIT = (8.27, 11.69)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else FontProperties()

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]


EDGE_LABELS = {
    "wind_to_humidity": "Wind -> Humidity",
    "humidity_to_cloud_cover": "Humidity -> Cloud Cover",
    "wind_to_cloud_cover": "Wind -> Cloud Cover",
    "humidity_to_humidity": "Humidity -> Humidity",
    "temperature_to_humidity": "Temperature -> Humidity",
    "temperature_to_cloud_cover": "Temperature -> Cloud Cover",
}

REGIME_ORDER = [
    "all",
    "DJF",
    "JJA",
    "high_humidity",
    "normal_humidity",
    "high_cloud",
    "normal_cloud",
]

REGIME_LABELS = {
    "all": "all",
    "DJF": "DJF",
    "JJA": "JJA",
    "high_humidity": "high humidity",
    "normal_humidity": "normal humidity",
    "high_cloud": "high cloud",
    "normal_cloud": "normal cloud",
}


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / name)


def fmt_int(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def fmt_float(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def short_text(value: object, width: int = 26) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def display_regime(value: object) -> str:
    return REGIME_LABELS.get(str(value), str(value))


def add_header(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.035, 0.955, title, fontproperties=FONT, fontsize=18, weight="bold", va="top")
    if subtitle:
        fig.text(0.035, 0.915, subtitle, fontproperties=FONT, fontsize=9.5, color="#444444", va="top")


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8dee8", linewidth=0.8, alpha=0.7)
    ax.tick_params(labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT)


def render_table_page(
    pdf: PdfPages,
    title: str,
    df: pd.DataFrame,
    subtitle: str | None = None,
    *,
    font_size: float = 7.2,
    wrap_width: int = 24,
    page_size: tuple[float, float] = LANDSCAPE,
) -> None:
    fig, ax = plt.subplots(figsize=page_size)
    ax.axis("off")
    add_header(fig, title, subtitle)

    display = df.astype(str).copy()
    for col in display.columns:
        display[col] = display[col].map(lambda x: short_text(x, wrap_width))
    headers = [short_text(col, 18) for col in display.columns]

    table = ax.table(
        cellText=display.values,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        bbox=[0.018, 0.045, 0.964, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#c7cfdb")
        cell.set_linewidth(0.55)
        cell.PAD = 0.12
        cell.get_text().set_fontproperties(FONT)
        if row == 0:
            cell.set_facecolor("#23395d")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f7fb")
        else:
            cell.set_facecolor("white")

    fig.text(0.035, 0.025, "Causal WeatherGraph | 1979-2025", fontproperties=FONT, fontsize=8, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def image_page(
    pdf: PdfPages,
    title: str,
    image_paths: list[Path],
    captions: list[str],
    subtitle: str | None = None,
    *,
    cols: int = 1,
    page_size: tuple[float, float] = LANDSCAPE,
) -> None:
    rows = math.ceil(len(image_paths) / cols)
    fig = plt.figure(figsize=page_size)
    add_header(fig, title, subtitle)
    top = 0.86 if subtitle else 0.89
    grid = fig.add_gridspec(rows, cols, left=0.04, right=0.97, top=top, bottom=0.07, wspace=0.08, hspace=0.16)
    for idx, path in enumerate(image_paths):
        ax = fig.add_subplot(grid[idx // cols, idx % cols])
        ax.axis("off")
        img = Image.open(path)
        ax.imshow(img)
        ax.text(0.0, -0.04, captions[idx], transform=ax.transAxes, fontproperties=FONT, fontsize=9, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def table_dataset_setup() -> pd.DataFrame:
    with open(REPO_ROOT / "outputs" / "processed" / "preprocess_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    audit = read_csv("data_time_audit.csv")
    rows = [
        ("Data source", "/home/vipuser/Data"),
        ("Time range", f"{audit['time_min'].min()} to {audit['time_max'].max()}"),
        ("Temporal resolution", "6 hours"),
        ("Time steps", fmt_int(summary["region_shape"][0])),
        ("Raw array shape", str(tuple(summary["raw_shape"]))),
        ("Region-level array shape", str(tuple(summary["region_shape"]))),
        ("Variables", ", ".join(summary["variables"])),
        ("Main region aggregation", "latlon_bins, target 64, actual 66"),
        ("Main method", "Granger-style OLS"),
        ("Main max lag", "3 steps, i.e., 6-18 h"),
        ("Candidate neighborhood", "candidate_k_nearest = 2"),
        ("Significance control", "Benjamini-Hochberg FDR, q < 0.05"),
        ("Data audit", "no missing 6-hour steps; no duplicate timestamps"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Setting"])


def table_regime_summary() -> pd.DataFrame:
    df = read_csv("summary_metrics.csv").copy()
    df["regime"] = pd.Categorical(df["regime"], REGIME_ORDER, ordered=True)
    df = df.sort_values("regime")
    return pd.DataFrame(
        {
            "Regime": df["regime"].astype(str).map(display_regime),
            "Tested edges": df["n_tested_edges"].map(fmt_int),
            "Significant edges": df["n_significant_edges"].map(fmt_int),
            "WH frac.": df["fraction_wind_to_humidity"].map(fmt_float),
            "HC frac.": df["fraction_humidity_to_cloud"].map(fmt_float),
            "Cross-region frac.": df["fraction_cross_region"].map(fmt_float),
            "Mean WH lag": df["mean_lag_wind_to_humidity"].map(fmt_float),
            "Mean HC lag": df["mean_lag_humidity_to_cloud"].map(fmt_float),
        }
    )


def table_chain_summary() -> pd.DataFrame:
    df = read_csv("causal_chain_summary.csv").copy()
    df["regime"] = pd.Categorical(df["regime"], REGIME_ORDER, ordered=True)
    df = df.sort_values("regime")
    return pd.DataFrame(
        {
            "Regime": df["regime"].astype(str).map(display_regime),
            "WHC chains": df["n_chains"].map(fmt_int),
            "Humidity regions involved": df["n_unique_humidity_regions"].map(fmt_int),
            "Mean total lag": df["mean_total_lag"].map(fmt_float),
            "Mean chain score": df["mean_chain_score"].map(fmt_float),
        }
    )


def table_null_summary() -> pd.DataFrame:
    df = read_csv("null_model_summary.csv")
    return pd.DataFrame(
        {
            "Edge set": df["edge_set"],
            "Tested edges": df["tested_edges"].map(fmt_int),
            "Significant edges": df["significant_edges"].map(fmt_int),
            "Significant ratio": df["significant_ratio"].map(fmt_float),
            "Mean abs. coef.": df["mean_abs_coef"].map(lambda x: fmt_float(x, 4)),
            "Mean effect score": df["mean_effect_score"].map(fmt_float),
        }
    )


def table_null_enrichment() -> pd.DataFrame:
    df = read_csv("null_model_variable_preserved_summary.csv")
    return pd.DataFrame(
        {
            "Edge type": df["edge_type"],
            "True sig. ratio": df["true_sig_ratio"].map(fmt_float),
            "Random-region sig. ratio": df["random_region_sig_ratio"].map(fmt_float),
            "Enrichment": df["enrichment"].map(fmt_float),
        }
    )


def table_bootstrap() -> pd.DataFrame:
    df = read_csv("bootstrap_stability_summary.csv")
    return pd.DataFrame(
        {
            "Edge type": df["edge_type"].map(lambda x: EDGE_LABELS.get(x, x)),
            "Significant edges": df["significant_edges"].map(fmt_int),
            "Stable edges": df["stable_edges"].map(fmt_int),
            "Stable ratio": df["stable_ratio"].map(fmt_float),
            "Mean stability": df["mean_stability"].map(fmt_float),
        }
    )


def table_directionality() -> pd.DataFrame:
    df = read_csv("directionality_control_summary.csv").copy()
    order = ["forward", "reverse", "future_to_past", "circular_shift", "time_shuffled"]
    path_order = ["Wind -> Humidity", "Humidity -> Wind", "Humidity -> Cloud", "Cloud -> Humidity", "Wind -> Cloud", "Cloud -> Wind"]
    df["test_type"] = pd.Categorical(df["test_type"], order, ordered=True)
    df["edge_rank"] = df["edge_type"].map(lambda x: path_order.index(x) if x in path_order else 99)
    df = df.sort_values(["edge_rank", "test_type"])
    return pd.DataFrame(
        {
            "Control": df["test_type"].astype(str),
            "Path": df["edge_type"],
            "Tested edges": df["tested_edges"].map(fmt_int),
            "Significant ratio": df["significant_ratio"].map(fmt_float),
            "Mean effect score": df["mean_effect_score"].map(fmt_float),
        }
    )


def table_latitude_mechanism() -> pd.DataFrame:
    edge = read_csv("latitude_band_edge_summary.csv")
    chain = read_csv("latitude_band_chain_summary.csv")
    df = edge.merge(chain, on="latitude_band", how="left")
    order = ["tropical", "midlatitude", "polar"]
    df["latitude_band"] = pd.Categorical(df["latitude_band"], order, ordered=True)
    df = df.sort_values("latitude_band")
    return pd.DataFrame(
        {
            "Latitude band": df["latitude_band"].astype(str),
            "WH edges": df["WH_edges"].map(fmt_int),
            "HC edges": df["HC_edges"].map(fmt_int),
            "WHC chains": df["WHC_chains"].map(fmt_int),
            "Stable WHC chains": df["stable_WHC_chains"].map(fmt_int),
            "Mean WH lag": df["mean_WH_lag"].map(fmt_float),
            "Mean HC lag": df["mean_HC_lag"].map(fmt_float),
            "Mean chain score": df["mean_chain_score"].map(fmt_float),
        }
    )


def table_robustness_summary() -> pd.DataFrame:
    k_df = read_csv("candidate_k_robustness_summary.csv")
    period = read_csv("period_stability_summary.csv")
    pre = read_csv("preprocessing_robustness_summary.csv")
    pre_overlap = read_csv("preprocessing_robustness_overlap.csv")
    region = read_csv("region_robustness_summary.csv")
    balance = read_csv("regime_sample_balance_summary.csv")
    pcmci = read_csv("pcmci_validation_summary.csv")

    pre_ov = pre_overlap.loc[pre_overlap["preprocess_setting"] != "P1_monthly", "WHC_chain_overlap_with_P1"]
    rows = [
        (
            "candidate-k",
            f"k=2-8 all recover WH/HC/WHC; k=8 WHC chains={fmt_int(k_df['WHC_chains'].max())}",
            "not dependent on candidate_k_nearest=2",
        ),
        (
            "period split",
            f"overlap with full graph={period['overlap_with_full'].min():.3f}-{period['overlap_with_full'].max():.3f}",
            "stable across historical periods",
        ),
        (
            "preprocessing",
            f"P1-P5 all recover WH/HC/WHC; WHC overlap={pre_ov.min():.3f}-{pre_ov.max():.3f}",
            "not caused by one preprocessing choice",
        ),
        (
            "region aggregation",
            f"{', '.join(region['setting'])} all recover WH/HC/WHC",
            "not an artifact of 66-region binning",
        ),
        (
            "sample balance",
            f"matched Jaccard={balance['matched_jaccard_mean'].min():.3f}-{balance['matched_jaccard_mean'].max():.3f}",
            "high/normal regime differences remain distinguishable",
        ),
        (
            "PCMCI validation",
            f"selected {int(pcmci['granger_top_edges'].sum())} top stable edges; status={pcmci['status'].iloc[0]}",
            "optional validation ready; current environment lacks Tigramite",
        ),
    ]
    return pd.DataFrame(rows, columns=["Robustness test", "Main finding", "Interpretation"])


def table_preprocessing() -> pd.DataFrame:
    df = read_csv("preprocessing_robustness_summary.csv")
    return pd.DataFrame(
        {
            "Setting": df["preprocess_setting"],
            "Tested edges": df["tested_edges"].map(fmt_int),
            "Significant edges": df["significant_edges"].map(fmt_int),
            "Sig. ratio": df["significant_ratio"].map(fmt_float),
            "WH": df["WH_edges"].map(fmt_int),
            "HC": df["HC_edges"].map(fmt_int),
            "WC": df["WC_edges"].map(fmt_int),
            "WHC chains": df["WHC_chains"].map(fmt_int),
            "Stable WH": df["stable_WH"].map(fmt_int),
            "Stable HC": df["stable_HC"].map(fmt_int),
        }
    )


def table_region() -> pd.DataFrame:
    df = read_csv("region_robustness_summary.csv")
    return pd.DataFrame(
        {
            "Setting": df["setting"],
            "Method": df["method"],
            "Actual regions": df["actual_regions"].map(fmt_int),
            "Tested edges": df["tested_edges"].map(fmt_int),
            "Significant edges": df["significant_edges"].map(fmt_int),
            "WH": df["WH_edges"].map(fmt_int),
            "HC": df["HC_edges"].map(fmt_int),
            "WC": df["WC_edges"].map(fmt_int),
            "WHC chains": df["WHC_chains"].map(fmt_int),
            "DJF WH > JJA": df["DJF_WH_stronger_than_JJA"].map(lambda x: "yes" if bool(x) else "no"),
            "Midlat strongest": df["midlatitude_strongest"].map(lambda x: "yes" if bool(x) else "no"),
        }
    )


def table_long_lag() -> pd.DataFrame:
    df = read_csv("targeted_maxlag12_lag_summary.csv")
    return pd.DataFrame(
        {
            "Edge type": df["edge_type"].map(lambda x: EDGE_LABELS.get(x, x)),
            "Peak lag": df["peak_lag"].map(fmt_int),
            "Mean lag": df["mean_lag"].map(fmt_float),
            "Median lag": df["median_lag"].map(fmt_float),
            "Lag 1 frac.": df["lag_1_frac"].map(fmt_float),
            "Lag 2-3 frac.": df["lag_2_3_frac"].map(fmt_float),
            "Lag 4-8 frac.": df["lag_4_8_frac"].map(fmt_float),
            "Lag 9-12 frac.": df["lag_9_12_frac"].map(fmt_float),
        }
    )


def table_mediation() -> pd.DataFrame:
    df = read_csv("wind_cloud_humidity_mediation_summary.csv").copy()
    df["regime"] = pd.Categorical(df["regime"], REGIME_ORDER, ordered=True)
    df = df.sort_values("regime")
    return pd.DataFrame(
        {
            "Regime": df["regime"].astype(str).map(display_regime),
            "WC edges": df["edge_count"].map(fmt_int),
            "Mean beta direct": df["mean_beta_direct"].map(lambda x: fmt_float(x, 4)),
            "Mean beta controlled": df["mean_beta_controlled"].map(lambda x: fmt_float(x, 4)),
            "Mean reduction": df["mean_reduction"].map(fmt_float),
            "Reduction >30%": df["fraction_reduction_gt_30"].map(fmt_float),
            "Reduction >50%": df["fraction_reduction_gt_50"].map(fmt_float),
        }
    )


def table_pcmci() -> pd.DataFrame:
    df = read_csv("pcmci_validation_summary.csv")
    return pd.DataFrame(
        {
            "Edge type": df["edge_type"].map(lambda x: EDGE_LABELS.get(x, x)),
            "Granger top edges": df["granger_top_edges"].map(fmt_int),
            "PCMCI confirmed": df["pcmci_confirmed"].map(fmt_int),
            "Confirmed ratio": df["confirmed_ratio"].map(fmt_float),
            "Mean Granger effect": df["mean_granger_effect_score"].map(fmt_float),
            "Status": df["status"],
        }
    )


def draw_framework() -> Path:
    path = FIGURE_DIR / "paper_fig1_framework.png"
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        ("Local global atmospheric\nreanalysis data\n1979-2025, 6-hourly", 0.08, 0.72, "#dfeef7"),
        ("Variable extraction\ntemperature, humidity,\nwind, cloud_cover", 0.30, 0.72, "#e7f4e2"),
        ("Spherical region\naggregation\n66 region series", 0.52, 0.72, "#fff1cc"),
        ("Regime definition\nall / DJF / JJA /\nhigh humidity / high cloud", 0.74, 0.72, "#fde2e2"),
        ("Candidate edge\nconstruction\nWH / HC / WC / HH / TH / TC", 0.19, 0.34, "#eadff7"),
        ("Lagged Granger-style\ndependency testing\nmax_lag=3, FDR q<0.05", 0.41, 0.34, "#e6edf9"),
        ("Bootstrap stability\nNull models\nDirection controls", 0.63, 0.34, "#e9f7ef"),
        ("Regime-aware\nCausal WeatherGraph\nWHC pathway", 0.80, 0.34, "#fff0e6"),
    ]
    box_w, box_h = 0.18, 0.18
    centers = []
    for text, x, y, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.1,
            edgecolor="#40526b",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + box_w / 2, y + box_h / 2, text, ha="center", va="center", fontproperties=FONT, fontsize=10)
        centers.append((x + box_w / 2, y + box_h / 2))

    arrow_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    for src, dst in arrow_pairs:
        sx, sy = centers[src]
        dx, dy = centers[dst]
        if src == 3:
            start = (sx, sy - box_h / 2)
            end = (centers[4][0], centers[4][1] + box_h / 2)
            connection = "arc3,rad=-0.28"
        else:
            start = (sx + box_w / 2, sy) if dx > sx else (sx, sy - box_h / 2)
            end = (dx - box_w / 2, dy) if dx > sx else (dx, dy + box_h / 2)
            connection = "arc3,rad=0.0"
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color="#40526b",
                connectionstyle=connection,
            )
        )

    ax.text(
        0.5,
        0.08,
        "Output: physically consistent lagged directed dependency pathway, interpreted below interventional causality.",
        ha="center",
        va="center",
        fontproperties=FONT,
        fontsize=10,
        color="#444444",
    )
    fig.suptitle("Figure 1. Overview of the Causal WeatherGraph framework", fontproperties=FONT, fontsize=16, y=0.97)
    save_figure(fig, path)
    return path


def draw_regime_chain_combo() -> Path:
    path = FIGURE_DIR / "paper_fig4_regime_chain_summary.png"
    images = [
        FIGURE_DIR / "regime_comparison_heatmap.png",
        FIGURE_DIR / "causal_chain_summary.png",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    captions = ["A. Regime graph overlap / distance", "B. WHC chain statistics by regime"]
    for ax, img_path, cap in zip(axes, images, captions):
        ax.axis("off")
        ax.imshow(Image.open(img_path))
        ax.text(0.01, 0.99, cap, transform=ax.transAxes, va="top", ha="left", fontproperties=FONT, fontsize=11, weight="bold")
    fig.suptitle("Figure 4. Regime-aware graph differences and WHC chain summary", fontproperties=FONT, fontsize=15)
    save_figure(fig, path)
    return path


def draw_controls_combo() -> Path:
    path = FIGURE_DIR / "paper_fig5_controls.png"
    null = read_csv("null_model_summary.csv")
    boot = read_csv("bootstrap_stability_summary.csv")
    direction = read_csv("directionality_control_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0), gridspec_kw={"width_ratios": [1.05, 1.35, 1.25]})
    fig.subplots_adjust(wspace=0.55)
    palette = ["#4666a8", "#6a9f58", "#d2913d", "#b94c5c", "#6f5ea8"]

    ax = axes[0]
    labels = ["physical", "random", "dist.-matched", "var.-preserved"]
    ax.bar(labels, null["significant_ratio"], color=palette[:4])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Significant ratio", fontproperties=FONT)
    ax.set_title("A. Null model comparison", fontproperties=FONT, fontsize=11, weight="bold")
    style_axis(ax)
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1]
    short_edge = {
        "wind_to_humidity": "Wind -> Humidity",
        "humidity_to_cloud_cover": "Humidity -> Cloud",
        "wind_to_cloud_cover": "Wind -> Cloud",
        "humidity_to_humidity": "Humidity -> Humidity",
        "temperature_to_humidity": "Temp -> Humidity",
        "temperature_to_cloud_cover": "Temp -> Cloud",
    }
    boot_labels = [short_edge.get(x, x) for x in boot["edge_type"]]
    ax.barh(boot_labels, boot["stable_edges"], color="#5b8fc4")
    ax.set_xlabel("Stable edges", fontproperties=FONT)
    ax.set_title("B. Bootstrap stable edges", fontproperties=FONT, fontsize=11, weight="bold")
    style_axis(ax)

    def path_group(label: str) -> str:
        items = set(label.replace(" ", "").split("->"))
        if items == {"Wind", "Humidity"}:
            return "Wind-Humidity"
        if items == {"Humidity", "Cloud"}:
            return "Humidity-Cloud"
        if items == {"Wind", "Cloud"}:
            return "Wind-Cloud"
        return label

    direction["path_group"] = direction["edge_type"].map(path_group)
    pivot = direction.pivot_table(index="test_type", columns="path_group", values="significant_ratio", aggfunc="mean")
    order = ["forward", "reverse", "future_to_past", "circular_shift", "time_shuffled"]
    pivot = pivot.reindex(order)
    ax = axes[2]
    x = np.arange(len(pivot.index))
    cols = ["Wind-Humidity", "Humidity-Cloud", "Wind-Cloud"]
    width = 0.23
    for i, col in enumerate(cols):
        ax.bar(x + (i - 1) * width, pivot[col].values, width, label=col, color=palette[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["forward", "reverse", "future", "shift", "shuffle"], rotation=25)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Significant ratio", fontproperties=FONT)
    ax.set_title("C. Temporal-direction controls", fontproperties=FONT, fontsize=11, weight="bold")
    ax.legend(prop=FONT, fontsize=7, frameon=False)
    style_axis(ax)

    save_figure(fig, path)
    return path


def draw_mechanism_robustness_combo() -> Path:
    path = FIGURE_DIR / "paper_fig6_mechanism_robustness.png"
    lat_chain = read_csv("latitude_band_chain_summary.csv")
    pre_overlap = read_csv("preprocessing_robustness_overlap.csv")
    region = read_csv("region_robustness_summary.csv")

    order = ["tropical", "midlatitude", "polar"]
    lat_chain["latitude_band"] = pd.Categorical(lat_chain["latitude_band"], order, ordered=True)
    lat_chain = lat_chain.sort_values("latitude_band")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#4b7bb8", "#6aa56a", "#d99b4c"]

    ax = axes[0, 0]
    x = np.arange(len(lat_chain))
    ax.bar(x - 0.18, lat_chain["WHC_chains"], width=0.36, label="WHC chains", color=colors[0])
    ax.bar(x + 0.18, lat_chain["stable_WHC_chains"], width=0.36, label="Stable WHC", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(lat_chain["latitude_band"])
    ax.set_title("A. Latitude-band chain counts", fontproperties=FONT, fontsize=11, weight="bold")
    ax.legend(prop=FONT, fontsize=8, frameon=False)
    style_axis(ax)

    ax = axes[0, 1]
    ax.bar(lat_chain["latitude_band"].astype(str), lat_chain["mean_chain_score"], color=colors)
    ax.set_ylim(0, max(lat_chain["mean_chain_score"]) * 1.25)
    ax.set_title("B. Mean chain score", fontproperties=FONT, fontsize=11, weight="bold")
    style_axis(ax)

    ax = axes[1, 0]
    sub = pre_overlap[pre_overlap["preprocess_setting"] != "P1_monthly"]
    ax.bar(sub["preprocess_setting"], sub["WHC_chain_overlap_with_P1"], color="#8a6fb3")
    ax.set_ylim(0.8, 0.9)
    ax.set_ylabel("WHC overlap with P1", fontproperties=FONT)
    ax.set_title("C. Preprocessing robustness", fontproperties=FONT, fontsize=11, weight="bold")
    ax.tick_params(axis="x", rotation=25)
    style_axis(ax)

    ax = axes[1, 1]
    ax.bar(region["setting"], region["WHC_chains"], color="#c56d5f")
    ax.set_ylabel("WHC chains", fontproperties=FONT)
    ax.set_title("D. Region aggregation robustness", fontproperties=FONT, fontsize=11, weight="bold")
    style_axis(ax)

    fig.suptitle("Figure 6. Latitude-band mechanism and robustness of the WHC pathway", fontproperties=FONT, fontsize=14)
    save_figure(fig, path)
    return path


def cover_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor("#f7f9fc")
    fig.text(0.08, 0.86, "Causal WeatherGraph", fontproperties=FONT, fontsize=30, weight="bold", color="#23395d")
    fig.text(0.08, 0.80, "论文实验表与图汇编", fontproperties=FONT, fontsize=24, weight="bold", color="#23395d")
    fig.text(
        0.08,
        0.72,
        "Regime-Aware Spatio-Temporal Causal Discovery\nfor Global Atmospheric Dynamics",
        fontproperties=FONT,
        fontsize=14,
        color="#44546a",
        linespacing=1.5,
    )
    fig.text(0.08, 0.58, "Data range: 1979-01-01 00:00:00 to 2025-12-31 18:00:00", fontproperties=FONT, fontsize=11)
    fig.text(0.08, 0.54, "Temporal resolution: 6 hours | Main method: Granger-style OLS", fontproperties=FONT, fontsize=11)
    fig.text(0.08, 0.50, "Main pathway: Wind -> Humidity -> Cloud Cover", fontproperties=FONT, fontsize=11)
    fig.text(
        0.08,
        0.34,
        "This PDF collects paper-ready experimental tables and figures generated\nfrom the current repository outputs. Edges are interpreted as physically\nconsistent lagged directed dependencies, not interventional causal effects.",
        fontproperties=FONT,
        fontsize=11,
        color="#4d4d4d",
        linespacing=1.45,
    )
    fig.text(0.08, 0.12, f"Output file: {OUTPUT_PDF.relative_to(REPO_ROOT)}", fontproperties=FONT, fontsize=9, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def index_page(pdf: PdfPages) -> None:
    items = [
        ("Main Table 1", "Dataset summary and experimental configuration"),
        ("Main Table 2", "Regime-aware Granger-style directed dependency summary"),
        ("Main Table 3", "Wind -> Humidity -> Cloud Cover chain statistics"),
        ("Main Table 4", "Null-model comparison and pathway enrichment"),
        ("Main Table 5", "Bootstrap stability and temporal-direction controls"),
        ("Main Table 6", "Mechanistic stratification and robustness summary"),
        ("Figure 1", "Overall framework of Causal WeatherGraph"),
        ("Figure 2", "Global stable Wind -> Humidity dependency map"),
        ("Figure 3", "Global stable Humidity -> Cloud dependency map"),
        ("Figure 4", "Regime-aware graph and WHC chain summary"),
        ("Figure 5", "Null model, stability, and directionality controls"),
        ("Figure 6", "Latitude-band mechanism and robustness"),
        ("Supplement", "Long-lag, mediation, preprocessing, region, PCMCI status tables and figures"),
    ]
    df = pd.DataFrame(items, columns=["Item", "Content"])
    render_table_page(pdf, "图表目录", df, "Main paper tables, main figures, and selected supplementary outputs.", font_size=9, wrap_width=54, page_size=PORTRAIT)


def build_pdf() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig1 = draw_framework()
    fig4 = draw_regime_chain_combo()
    fig5 = draw_controls_combo()
    fig6 = draw_mechanism_robustness_combo()

    with PdfPages(OUTPUT_PDF) as pdf:
        cover_page(pdf)
        index_page(pdf)

        render_table_page(pdf, "Table 1. Dataset summary and experimental configuration", table_dataset_setup(), font_size=8.0, wrap_width=42)
        render_table_page(pdf, "Table 2. Regime-aware Granger-style directed dependency summary", table_regime_summary(), font_size=7.0, wrap_width=18)
        render_table_page(pdf, "Table 3. Wind -> Humidity -> Cloud Cover chain statistics across regimes", table_chain_summary(), font_size=8.0, wrap_width=22)
        render_table_page(pdf, "Table 4A. Null-model comparison", table_null_summary(), font_size=8.0, wrap_width=24)
        render_table_page(pdf, "Table 4B. Core pathway enrichment under variable-preserved randomization", table_null_enrichment(), font_size=9.0, wrap_width=24)
        render_table_page(pdf, "Table 5A. Bootstrap stability", table_bootstrap(), font_size=8.0, wrap_width=26)
        render_table_page(pdf, "Table 5B. Temporal-direction controls", table_directionality(), font_size=6.4, wrap_width=18)
        render_table_page(pdf, "Table 6A. Latitude-band mechanistic stratification", table_latitude_mechanism(), font_size=7.2, wrap_width=18)
        render_table_page(pdf, "Table 6B. Robustness summary", table_robustness_summary(), font_size=7.6, wrap_width=38)

        image_page(pdf, "Figure 1. Overview of the Causal WeatherGraph framework", [fig1], ["Framework from data ingestion to regime-aware graph and WHC pathway."], cols=1)
        image_page(
            pdf,
            "Figure 2. Spatial distribution of stable Wind -> Humidity dependencies",
            [FIGURE_DIR / "causal_edge_map_wind_to_humidity.png"],
            ["Top spatial Wind -> Humidity dependencies; arrows show region-level lagged directed dependencies."],
        )
        image_page(
            pdf,
            "Figure 3. Spatial distribution of stable Humidity -> Cloud Cover dependencies",
            [FIGURE_DIR / "causal_edge_map_humidity_to_cloud.png"],
            ["Top spatial Humidity -> Cloud Cover dependencies; lagged edges support the second step of the WHC pathway."],
        )
        image_page(
            pdf,
            "Figure 4. Regime-aware graph differences and WHC chain summary",
            [fig4],
            ["Combined view of regime graph differences and Wind -> Humidity -> Cloud Cover chain statistics."],
        )
        image_page(
            pdf,
            "Figure 5. Control analyses for non-randomness, stability, and temporal directionality",
            [fig5],
            ["Null models, bootstrap stable edges, and temporal-direction controls summarized from CSV outputs."],
        )
        image_page(
            pdf,
            "Figure 6. Latitude-band mechanism and robustness of the WHC pathway",
            [fig6],
            ["Midlatitude strengthening plus preprocessing and region aggregation robustness."],
        )

        render_table_page(pdf, "Supplementary Table S1. Long-lag max_lag=12 results", table_long_lag(), font_size=7.5, wrap_width=24)
        render_table_page(pdf, "Supplementary Table S2. Wind -> Cloud humidity mediation", table_mediation(), font_size=7.0, wrap_width=18)
        render_table_page(pdf, "Supplementary Table S3. Preprocessing robustness", table_preprocessing(), font_size=6.8, wrap_width=18)
        render_table_page(pdf, "Supplementary Table S4. Region aggregation robustness", table_region(), font_size=6.3, wrap_width=16)
        render_table_page(pdf, "Supplementary Table S5. PCMCI validation status", table_pcmci(), font_size=8.0, wrap_width=24)

        image_page(
            pdf,
            "Supplementary Figure S1-S3. Additional robustness and application figures",
            [
                FIGURE_DIR / "lag_distribution.png",
                FIGURE_DIR / "preprocessing_robustness_heatmap.png",
                FIGURE_DIR / "region_robustness_summary.png",
                FIGURE_DIR / "extreme_precursor_heatmap.png",
            ],
            [
                "S1. Long-lag distribution under max_lag=12.",
                "S2. Preprocessing robustness heatmap.",
                "S3. Region aggregation robustness.",
                "S4. Extreme precursor analysis heatmap.",
            ],
            cols=2,
        )

    print(f"Saved PDF: {OUTPUT_PDF}")
    print("Saved composite figures:")
    for path in [fig1, fig4, fig5, fig6]:
        print(f"  {path}")


if __name__ == "__main__":
    build_pdf()
