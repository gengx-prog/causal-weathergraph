#!/usr/bin/env python
"""Build a print-friendly PDF with one table or one figure per page."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from make_paper_figures_pdf import (
    EDGE_LABELS,
    FIGURE_DIR,
    FONT,
    LANDSCAPE,
    PORTRAIT,
    REGIME_ORDER,
    REPORT_DIR,
    display_regime,
    draw_framework,
    fmt_float,
    fmt_int,
    image_page,
    read_csv,
    render_table_page,
    save_figure,
    style_axis,
    table_bootstrap,
    table_chain_summary,
    table_dataset_setup,
    table_directionality,
    table_latitude_mechanism,
    table_long_lag,
    table_mediation,
    table_null_enrichment,
    table_null_summary,
    table_pcmci,
    table_preprocessing,
    table_regime_summary,
    table_region,
    table_robustness_summary,
)


OUTPUT_PDF = REPORT_DIR / "Causal_WeatherGraph_论文实验图表单图多样化版_1979_2025.pdf"


def set_common_text(ax: plt.Axes, title: str, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_title(title, fontproperties=FONT, fontsize=13, weight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=FONT, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=FONT, fontsize=10)
    style_axis(ax)


def add_value_labels(ax: plt.Axes, bars, fmt="{:.3f}", *, horizontal: bool = False) -> None:
    for bar in bars:
        if horizontal:
            value = bar.get_width()
            ax.text(value, bar.get_y() + bar.get_height() / 2, "  " + fmt.format(value), va="center", fontproperties=FONT, fontsize=8)
        else:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value, fmt.format(value), ha="center", va="bottom", fontproperties=FONT, fontsize=8)


def standalone_null_model() -> Path:
    path = FIGURE_DIR / "paper_varied_null_model_significant_ratio.png"
    df = read_csv("null_model_summary.csv")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    labels = ["physical", "random", "distance\nmatched", "variable\npreserved"]
    colors = ["#4666a8", "#6a9f58", "#d2913d", "#b94c5c"]
    x = np.arange(len(df))
    ax.vlines(x, 0, df["significant_ratio"], color="#94a3b8", linewidth=2.0, alpha=0.85)
    ax.scatter(x, df["significant_ratio"], s=220, color=colors, edgecolor="white", linewidth=1.3, zorder=3)
    for xi, value in zip(x, df["significant_ratio"]):
        ax.text(xi, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontproperties=FONT, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.9)
    set_common_text(ax, "Null model comparison", "Significant ratio")
    save_figure(fig, path)
    return path


def standalone_null_enrichment() -> Path:
    path = FIGURE_DIR / "paper_varied_null_enrichment_radar.png"
    df = read_csv("null_model_variable_preserved_summary.csv")
    labels = df["edge_type"].astype(str).str.replace("->", " -> ", regex=False).tolist()
    values = df["enrichment"].to_numpy(dtype=float)
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    values_closed = np.r_[values, values[0]]
    angles_closed = np.r_[angles, angles[0]]

    fig, ax = plt.subplots(figsize=(8.0, 5.8), subplot_kw={"projection": "polar"})
    ax.plot(angles_closed, values_closed, color="#4666a8", linewidth=2.4)
    ax.fill(angles_closed, values_closed, color="#4666a8", alpha=0.18)
    ax.scatter(angles, values, s=95, color=["#4666a8", "#6a9f58", "#d2913d"], edgecolor="white", linewidth=1.1, zorder=3)
    for angle, value in zip(angles, values):
        ax.text(angle, value + 0.08, f"{value:.2f}x", ha="center", va="center", fontproperties=FONT, fontsize=9)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontproperties=FONT, fontsize=9)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_yticks(np.linspace(0.5, 2.0, 4))
    ax.set_yticklabels([f"{v:.1f}x" for v in np.linspace(0.5, 2.0, 4)], fontproperties=FONT, fontsize=8, color="#555555")
    ax.grid(color="#d8dee8", linewidth=0.8)
    ax.spines["polar"].set_color("#c7cfdb")
    ax.set_title("Core pathway enrichment", fontproperties=FONT, fontsize=13, weight="bold", pad=22)
    save_figure(fig, path)
    return path


def standalone_bootstrap() -> Path:
    path = FIGURE_DIR / "paper_varied_bootstrap_stability_bubble.png"
    df = read_csv("bootstrap_stability_summary.csv").copy()
    short_edge = {
        "wind_to_humidity": "Wind -> Humidity",
        "humidity_to_cloud_cover": "Humidity -> Cloud",
        "wind_to_cloud_cover": "Wind -> Cloud",
        "humidity_to_humidity": "Humidity -> Humidity",
        "temperature_to_humidity": "Temp -> Humidity",
        "temperature_to_cloud_cover": "Temp -> Cloud",
    }
    df["label"] = df["edge_type"].map(lambda x: short_edge.get(x, x))
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    y = np.arange(len(df))
    sizes = 420 + 2400 * df["stable_ratio"].to_numpy(dtype=float)
    sc = ax.scatter(
        df["stable_edges"],
        y,
        s=sizes,
        c=df["mean_stability"],
        cmap="viridis",
        vmin=df["mean_stability"].min(),
        vmax=df["mean_stability"].max(),
        alpha=0.86,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.hlines(y, 0, df["stable_edges"], color="#d8dee8", linewidth=1.0, zorder=0)
    for xi, yi, ratio in zip(df["stable_edges"], y, df["stable_ratio"]):
        ax.text(xi + 25, yi, f"{int(xi)} | {ratio:.2f}", va="center", fontproperties=FONT, fontsize=8.2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_ylim(len(df) - 0.25, -0.75)
    set_common_text(ax, "Bootstrap stability", "Edge type", "Stable edges")
    ax.set_xlim(0, df["stable_edges"].max() * 1.28)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Mean stability", fontproperties=FONT, fontsize=9)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(FONT)
    save_figure(fig, path)
    return path


def standalone_directionality() -> Path:
    path = FIGURE_DIR / "paper_varied_directionality_controls_heatmap.png"
    df = read_csv("directionality_control_summary.csv").copy()

    def path_group(label: str) -> str:
        items = set(label.replace(" ", "").split("->"))
        if items == {"Wind", "Humidity"}:
            return "Wind-Humidity"
        if items == {"Humidity", "Cloud"}:
            return "Humidity-Cloud"
        if items == {"Wind", "Cloud"}:
            return "Wind-Cloud"
        return label

    df["path_group"] = df["edge_type"].map(path_group)
    pivot = df.pivot_table(index="test_type", columns="path_group", values="significant_ratio", aggfunc="mean")
    order = ["forward", "reverse", "future_to_past", "circular_shift", "time_shuffled"]
    pivot = pivot.reindex(order)
    cols = ["Wind-Humidity", "Humidity-Cloud", "Wind-Cloud"]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=0.9, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(["forward", "reverse", "future-to-past", "circular-shift", "time-shuffled"])
    for i in range(len(pivot.index)):
        for j in range(len(cols)):
            value = pivot.iloc[i, j]
            color = "white" if value > 0.55 else "#23395d"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontproperties=FONT, fontsize=8.5, color=color)
    ax.set_title("Temporal-direction controls", fontproperties=FONT, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Path group", fontproperties=FONT, fontsize=10)
    ax.set_ylabel("Control type", fontproperties=FONT, fontsize=10)
    ax.tick_params(labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Significant ratio", fontproperties=FONT, fontsize=9)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(FONT)
    save_figure(fig, path)
    return path


def standalone_whc_chain_count() -> Path:
    path = FIGURE_DIR / "paper_varied_whc_chain_count_by_regime.png"
    df = read_csv("causal_chain_summary.csv").copy()
    df["regime"] = pd.Categorical(df["regime"], REGIME_ORDER, ordered=True)
    df = df.sort_values("regime")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    labels = df["regime"].astype(str).map(display_regime).to_numpy()
    x = np.arange(len(df))
    y = df["n_chains"].to_numpy(dtype=float)
    ax.fill_between(x, y, y.min() * 0.92, color="#4b7bb8", alpha=0.16)
    ax.plot(x, y, color="#4b7bb8", linewidth=2.5, marker="o", markersize=7)
    for xi, value in zip(x, y):
        ax.text(xi, value + 80, f"{int(value)}", ha="center", va="bottom", fontproperties=FONT, fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", rotation=20)
    set_common_text(ax, "WHC chain count by regime", "Number of chains")
    ax.set_ylim(y.min() * 0.9, y.max() * 1.12)
    save_figure(fig, path)
    return path


def standalone_whc_chain_score() -> Path:
    path = FIGURE_DIR / "paper_varied_whc_chain_score_by_regime.png"
    df = read_csv("causal_chain_summary.csv").copy()
    df["regime"] = pd.Categorical(df["regime"], REGIME_ORDER, ordered=True)
    df = df.sort_values("regime")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(df))
    values = df["mean_chain_score"].to_numpy(dtype=float)
    mean_value = values.mean()
    ax.axvline(mean_value, color="#7a869a", linestyle="--", linewidth=1.2, label=f"Mean {mean_value:.3f}")
    ax.hlines(y, mean_value, values, color="#9fbc8f", linewidth=3, alpha=0.8)
    ax.scatter(values, y, s=120, color="#6aa56a", edgecolor="white", linewidth=1.1, zorder=3)
    for value, yi in zip(values, y):
        ax.text(value + 0.006, yi, f"{value:.3f}", va="center", fontproperties=FONT, fontsize=8.2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["regime"].astype(str).map(display_regime))
    ax.invert_yaxis()
    ax.legend(prop=FONT, fontsize=8.5, frameon=False, loc="lower right")
    set_common_text(ax, "Mean WHC chain score by regime", xlabel="Mean chain score")
    ax.set_xlim(values.min() * 0.9, values.max() * 1.12)
    save_figure(fig, path)
    return path


def standalone_latitude_chains() -> Path:
    path = FIGURE_DIR / "paper_varied_latitude_whc_chains_dumbbell.png"
    df = read_csv("latitude_band_chain_summary.csv").copy()
    order = ["tropical", "midlatitude", "polar"]
    df["latitude_band"] = pd.Categorical(df["latitude_band"], order, ordered=True)
    df = df.sort_values("latitude_band")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    y = np.arange(len(df))
    ax.hlines(y, df["stable_WHC_chains"], df["WHC_chains"], color="#c7cfdb", linewidth=3)
    ax.scatter(df["WHC_chains"], y, s=120, color="#4b7bb8", label="WHC chains", edgecolor="white", linewidth=1.1, zorder=3)
    ax.scatter(df["stable_WHC_chains"], y, s=120, color="#6aa56a", label="Stable WHC", edgecolor="white", linewidth=1.1, zorder=3)
    for yi, total, stable in zip(y, df["WHC_chains"], df["stable_WHC_chains"]):
        ax.text(total + 160, yi, f"{int(total)}", va="center", fontproperties=FONT, fontsize=8.2)
        ax.text(stable - 160, yi, f"{int(stable)}", ha="right", va="center", fontproperties=FONT, fontsize=8.2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["latitude_band"])
    ax.invert_yaxis()
    ax.legend(prop=FONT, fontsize=9, frameon=False)
    set_common_text(ax, "Latitude-band WHC chain counts", "Latitude band", "Number of chains")
    ax.set_xlim(0, df["WHC_chains"].max() * 1.18)
    save_figure(fig, path)
    return path


def standalone_latitude_score() -> Path:
    path = FIGURE_DIR / "paper_varied_latitude_chain_score_bubbles.png"
    df = read_csv("latitude_band_chain_summary.csv").copy()
    order = ["tropical", "midlatitude", "polar"]
    df["latitude_band"] = pd.Categorical(df["latitude_band"], order, ordered=True)
    df = df.sort_values("latitude_band")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    sizes = 260 + 1600 * df["WHC_chains"].to_numpy(dtype=float) / df["WHC_chains"].max()
    sc = ax.scatter(
        df["mean_total_lag"],
        df["mean_chain_score"],
        s=sizes,
        c=np.arange(len(df)),
        cmap="Set2",
        alpha=0.88,
        edgecolor="white",
        linewidth=1.2,
    )
    for _, row in df.iterrows():
        ax.text(
            row["mean_total_lag"] + 0.004,
            row["mean_chain_score"],
            f"{row['latitude_band']} ({row['mean_chain_score']:.3f})",
            va="center",
            fontproperties=FONT,
            fontsize=8.7,
        )
    set_common_text(ax, "Latitude-band mean chain score", "Mean chain score", "Mean total lag")
    ax.set_xlim(df["mean_total_lag"].min() - 0.02, df["mean_total_lag"].max() + 0.08)
    ax.set_ylim(df["mean_chain_score"].min() * 0.9, df["mean_chain_score"].max() * 1.12)
    save_figure(fig, path)
    return path


def standalone_candidate_k() -> Path:
    path = FIGURE_DIR / "paper_varied_candidate_k_robustness.png"
    df = read_csv("candidate_k_robustness_summary.csv")
    fig, ax1 = plt.subplots(figsize=(8.8, 5.3))
    ax1.plot(df["k"], df["WHC_chains"], marker="o", color="#4b7bb8", linewidth=2.2, label="WHC chains")
    ax1.set_ylabel("WHC chains", fontproperties=FONT, color="#4b7bb8")
    ax1.tick_params(axis="y", labelcolor="#4b7bb8")
    ax2 = ax1.twinx()
    ax2.plot(df["k"], df["stable_WHC_chains"], marker="s", color="#b94c5c", linewidth=2.2, label="Stable WHC chains")
    ax2.set_ylabel("Stable WHC chains", fontproperties=FONT, color="#b94c5c")
    ax2.tick_params(axis="y", labelcolor="#b94c5c")
    ax1.set_xticks(df["k"])
    set_common_text(ax1, "Candidate-k robustness", xlabel="candidate_k_nearest")
    for label in ax2.get_yticklabels():
        label.set_fontproperties(FONT)
    save_figure(fig, path)
    return path


def standalone_period_stability() -> Path:
    path = FIGURE_DIR / "paper_varied_period_stability.png"
    df = read_csv("period_stability_summary.csv")
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    labels = df["period"] + "\n" + df["time_range"]
    x = np.arange(len(df))
    y = df["overlap_with_full"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", markersize=8, color="#6a9f58", linewidth=2.4)
    ax.fill_between(x, y, y.min() - 0.01, color="#6a9f58", alpha=0.15)
    for xi, value in zip(x, y):
        ax.text(xi, value + 0.0035, f"{value:.3f}", ha="center", va="bottom", fontproperties=FONT, fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.84, 0.93)
    set_common_text(ax, "Historical-period stability", "Overlap with full graph")
    save_figure(fig, path)
    return path


def standalone_preprocessing_chains() -> Path:
    path = FIGURE_DIR / "paper_varied_preprocessing_whc_chains.png"
    df = read_csv("preprocessing_robustness_summary.csv")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(df))
    chains = df["WHC_chains"].to_numpy(dtype=float)
    ratio = df["significant_ratio"].to_numpy(dtype=float)
    ax.plot(x, chains, marker="D", color="#8a6fb3", linewidth=2.3, markersize=7, label="WHC chains")
    for xi, value in zip(x, chains):
        ax.text(xi, value + 700, f"{int(value)}", ha="center", va="bottom", fontproperties=FONT, fontsize=8.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["preprocess_setting"])
    ax.tick_params(axis="x", rotation=20)
    set_common_text(ax, "Preprocessing robustness: WHC chains", "WHC chains")
    ax.set_ylim(chains.min() * 0.88, chains.max() * 1.15)
    ax2 = ax.twinx()
    ax2.plot(x, ratio, marker="o", color="#d2913d", linewidth=1.9, markersize=5.5, label="Significant ratio")
    ax2.set_ylabel("Significant ratio", fontproperties=FONT, color="#9a6b1f")
    ax2.tick_params(axis="y", labelcolor="#9a6b1f", labelsize=8)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(FONT)
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, prop=FONT, fontsize=8.5, frameon=False, loc="upper left")
    save_figure(fig, path)
    return path


def standalone_preprocessing_overlap() -> Path:
    path = FIGURE_DIR / "paper_varied_preprocessing_overlap_heatmap.png"
    df = read_csv("preprocessing_robustness_overlap.csv")
    df = df[df["preprocess_setting"] != "P1_monthly"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    cols = ["edge_overlap_with_P1", "WH_overlap_with_P1", "HC_overlap_with_P1", "WHC_chain_overlap_with_P1"]
    labels = ["All edges", "Wind-Humidity", "Humidity-Cloud", "WHC chain"]
    values = df[cols].to_numpy(dtype=float)
    image = ax.imshow(values, cmap="YlOrBr", vmin=0.84, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels, rotation=12)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["preprocess_setting"])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            color = "white" if value < 0.9 else "#23395d"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontproperties=FONT, fontsize=8.4, color=color)
    ax.set_title("Preprocessing robustness: overlap with P1", fontproperties=FONT, fontsize=13, weight="bold", pad=12)
    ax.tick_params(labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Overlap", fontproperties=FONT, fontsize=9)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(FONT)
    save_figure(fig, path)
    return path


def standalone_region_chains() -> Path:
    path = FIGURE_DIR / "paper_varied_region_robustness_whc_chains.png"
    df = read_csv("region_robustness_summary.csv")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    markers = {"latlon_bins": "o", "kmeans_sphere": "s"}
    for method, sub in df.groupby("method"):
        ax.scatter(
            sub["actual_regions"],
            sub["WHC_chains"],
            s=2200 * sub["significant_ratio"],
            c=sub["mean_effect_score"] if "mean_effect_score" in sub else sub["significant_ratio"],
            cmap="plasma",
            vmin=df["significant_ratio"].min(),
            vmax=df["significant_ratio"].max(),
            marker=markers.get(method, "o"),
            alpha=0.82,
            edgecolor="white",
            linewidth=1.2,
            label=method,
        )
    for _, row in df.iterrows():
        ax.text(row["actual_regions"] + 1.6, row["WHC_chains"], f"{row['setting']}\n{int(row['WHC_chains'])}", va="center", fontproperties=FONT, fontsize=8.1)
    set_common_text(ax, "Region aggregation robustness: WHC chains", "WHC chains", "Actual regions")
    ax.legend(prop=FONT, fontsize=8.5, frameon=False, loc="upper left")
    ax.set_xlim(df["actual_regions"].min() - 8, df["actual_regions"].max() + 18)
    ax.set_ylim(df["WHC_chains"].min() * 0.82, df["WHC_chains"].max() * 1.12)
    save_figure(fig, path)
    return path


def standalone_long_lag() -> Path:
    path = FIGURE_DIR / "paper_varied_long_lag_distribution_heatmap.png"
    df = read_csv("targeted_maxlag12_lag_summary.csv").copy()
    df["label"] = df["edge_type"].map(lambda x: EDGE_LABELS.get(x, x).replace(" Cover", ""))
    cols = ["lag_1_frac", "lag_2_3_frac", "lag_4_8_frac", "lag_9_12_frac"]
    labels = ["lag 1", "lag 2-3", "lag 4-8", "lag 9-12"]
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    values = df[cols].to_numpy(dtype=float)
    image = ax.imshow(values, cmap="PuBuGn", vmin=0, vmax=0.45, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["label"])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            color = "white" if value > 0.25 else "#23395d"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontproperties=FONT, fontsize=8.5, color=color)
    ax.set_title("Long-lag distribution under max_lag=12", fontproperties=FONT, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Lag group", fontproperties=FONT, fontsize=10)
    ax.set_ylabel("Edge type", fontproperties=FONT, fontsize=10)
    ax.tick_params(labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Fraction of significant edges", fontproperties=FONT, fontsize=9)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(FONT)
    save_figure(fig, path)
    return path


def standalone_mediation() -> Path:
    path = FIGURE_DIR / "paper_varied_mediation_reduction_slope.png"
    df = read_csv("wind_cloud_humidity_mediation_summary.csv").copy()
    df["regime"] = pd.Categorical(df["regime"], REGIME_ORDER, ordered=True)
    df = df.sort_values("regime")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.array([0, 1])
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    label_rows = []
    for color, (_, row) in zip(colors, df.iterrows()):
        y = [row["mean_beta_direct"], row["mean_beta_controlled"]]
        ax.plot(x, y, marker="o", linewidth=1.8, color=color, alpha=0.9)
        label_rows.append((y[1], color, f"{display_regime(row['regime'])} ({row['mean_reduction']:.3f})"))

    label_rows = sorted(label_rows, key=lambda item: item[0])
    min_gap = 0.00013
    label_positions = [item[0] for item in label_rows]
    for idx in range(1, len(label_positions)):
        if label_positions[idx] - label_positions[idx - 1] < min_gap:
            label_positions[idx] = label_positions[idx - 1] + min_gap
    overflow = label_positions[-1] - max(item[0] for item in label_rows)
    if overflow > 0:
        label_positions = [pos - overflow * 0.45 for pos in label_positions]
    for (point_y, color, label), label_y in zip(label_rows, label_positions):
        ax.plot([1.0, 1.025], [point_y, label_y], color=color, linewidth=0.9, alpha=0.65)
        ax.text(1.035, label_y, label, va="center", fontproperties=FONT, fontsize=8.0)
    ax.axhline(0, color="#7a869a", linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["Direct W -> C", "Humidity controlled"])
    ax.set_xlim(-0.08, 1.55)
    ymin = min(df["mean_beta_direct"].min(), df["mean_beta_controlled"].min()) * 1.22
    ymax = max(df["mean_beta_direct"].max(), df["mean_beta_controlled"].max()) * 0.72
    ax.set_ylim(ymin, ymax)
    set_common_text(ax, "Wind -> Cloud humidity mediation", "Mean beta", "Model specification")
    save_figure(fig, path)
    return path


def cover_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor("#f7f9fc")
    fig.text(0.08, 0.86, "Causal WeatherGraph", fontproperties=FONT, fontsize=30, weight="bold", color="#23395d")
    fig.text(0.08, 0.80, "论文实验表与图汇编：单图单表多样化图形版", fontproperties=FONT, fontsize=20, weight="bold", color="#23395d")
    fig.text(
        0.08,
        0.71,
        "每一页只放一张表或一张图，并混合使用点线图、热力图、雷达图、气泡图和坡度图。",
        fontproperties=FONT,
        fontsize=12,
        color="#44546a",
    )
    fig.text(0.08, 0.58, "Data range: 1979-01-01 00:00:00 to 2025-12-31 18:00:00", fontproperties=FONT, fontsize=11)
    fig.text(0.08, 0.54, "Temporal resolution: 6 hours | Main method: Granger-style OLS", fontproperties=FONT, fontsize=11)
    fig.text(0.08, 0.50, "Main pathway: Wind -> Humidity -> Cloud Cover", fontproperties=FONT, fontsize=11)
    fig.text(
        0.08,
        0.34,
        "All figures and tables are generated from current repository outputs.\nEdges should be interpreted as physically consistent lagged directed\ndependencies, not interventional causal effects.",
        fontproperties=FONT,
        fontsize=11,
        color="#4d4d4d",
        linespacing=1.45,
    )
    fig.text(0.08, 0.12, f"Output file: {OUTPUT_PDF.name}", fontproperties=FONT, fontsize=9, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def index_page(pdf: PdfPages, figure_items: list[tuple[str, Path]]) -> None:
    table_items = [
        "Dataset summary",
        "Main regime-aware graph summary",
        "WHC chain summary",
        "Null-model comparison",
        "Core pathway enrichment",
        "Bootstrap stability",
        "Temporal-direction controls",
        "Latitude-band mechanism",
        "Robustness summary",
        "Long-lag results",
        "Wind -> Cloud mediation",
        "Preprocessing robustness",
        "Region aggregation robustness",
        "PCMCI status",
    ]
    render_table_page(
        pdf,
        "表目录：单图单表多样化图形版",
        pd.DataFrame([(f"Table {idx + 1}", item) for idx, item in enumerate(table_items)], columns=["Item", "Content"]),
        "Each following table page contains exactly one table.",
        font_size=8.0,
        wrap_width=42,
        page_size=PORTRAIT,
    )
    render_table_page(
        pdf,
        "图目录：单图单表多样化图形版",
        pd.DataFrame([(f"Figure {idx + 1}", title.split(". ", 1)[-1]) for idx, (title, _path) in enumerate(figure_items)], columns=["Item", "Content"]),
        "Each following figure page contains exactly one figure.",
        font_size=7.2,
        wrap_width=46,
        page_size=PORTRAIT,
    )


def build_standalone_figures() -> list[tuple[str, Path]]:
    framework = draw_framework()
    figure_items = [
        ("Figure 1. Overview of the Causal WeatherGraph framework", framework),
        ("Figure 2. Region network used by Causal WeatherGraph", FIGURE_DIR / "region_network.png"),
        ("Figure 3. Stable Wind -> Humidity dependency map", FIGURE_DIR / "causal_edge_map_wind_to_humidity.png"),
        ("Figure 4. Stable Humidity -> Cloud Cover dependency map", FIGURE_DIR / "causal_edge_map_humidity_to_cloud.png"),
        ("Figure 5. Regime comparison heatmap", FIGURE_DIR / "regime_comparison_heatmap.png"),
        ("Figure 6. WHC chain count by regime", standalone_whc_chain_count()),
        ("Figure 7. Mean WHC chain score by regime", standalone_whc_chain_score()),
        ("Figure 8. Null model significant-ratio comparison", standalone_null_model()),
        ("Figure 9. Core pathway enrichment", standalone_null_enrichment()),
        ("Figure 10. Bootstrap stable edge counts", standalone_bootstrap()),
        ("Figure 11. Temporal-direction controls", standalone_directionality()),
        ("Figure 12. Latitude-band WHC chain counts", standalone_latitude_chains()),
        ("Figure 13. Latitude-band mean chain score", standalone_latitude_score()),
        ("Figure 14. Candidate-k robustness", standalone_candidate_k()),
        ("Figure 15. Historical-period stability", standalone_period_stability()),
        ("Figure 16. Preprocessing robustness: WHC chains", standalone_preprocessing_chains()),
        ("Figure 17. Preprocessing robustness: WHC overlap with P1", standalone_preprocessing_overlap()),
        ("Figure 18. Region aggregation robustness: WHC chains", standalone_region_chains()),
        ("Figure 19. Long-lag distribution under max_lag=12", standalone_long_lag()),
        ("Figure 20. Wind -> Cloud humidity mediation", standalone_mediation()),
        ("Figure 21. Extreme precursor heatmap", FIGURE_DIR / "extreme_precursor_heatmap.png"),
    ]
    return figure_items


def build_pdf() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figure_items = build_standalone_figures()

    with PdfPages(OUTPUT_PDF) as pdf:
        cover_page(pdf)
        index_page(pdf, figure_items)

        render_table_page(pdf, "Table 1. Dataset summary and experimental configuration", table_dataset_setup(), font_size=8.0, wrap_width=42)
        render_table_page(pdf, "Table 2. Regime-aware Granger-style directed dependency summary", table_regime_summary(), font_size=7.0, wrap_width=18)
        render_table_page(pdf, "Table 3. Wind -> Humidity -> Cloud Cover chain statistics across regimes", table_chain_summary(), font_size=8.0, wrap_width=22)
        render_table_page(pdf, "Table 4A. Null-model comparison", table_null_summary(), font_size=8.0, wrap_width=24)
        render_table_page(pdf, "Table 4B. Core pathway enrichment under variable-preserved randomization", table_null_enrichment(), font_size=9.0, wrap_width=24)
        render_table_page(pdf, "Table 5A. Bootstrap stability", table_bootstrap(), font_size=8.0, wrap_width=26)
        render_table_page(pdf, "Table 5B. Temporal-direction controls", table_directionality(), font_size=6.4, wrap_width=18)
        render_table_page(pdf, "Table 6A. Latitude-band mechanistic stratification", table_latitude_mechanism(), font_size=7.2, wrap_width=18)
        render_table_page(pdf, "Table 6B. Robustness summary", table_robustness_summary(), font_size=7.6, wrap_width=38)
        render_table_page(pdf, "Supplementary Table S1. Long-lag max_lag=12 results", table_long_lag(), font_size=7.5, wrap_width=24)
        render_table_page(pdf, "Supplementary Table S2. Wind -> Cloud humidity mediation", table_mediation(), font_size=7.0, wrap_width=18)
        render_table_page(pdf, "Supplementary Table S3. Preprocessing robustness", table_preprocessing(), font_size=6.8, wrap_width=18)
        render_table_page(pdf, "Supplementary Table S4. Region aggregation robustness", table_region(), font_size=6.3, wrap_width=16)
        render_table_page(pdf, "Supplementary Table S5. PCMCI validation status", table_pcmci(), font_size=8.0, wrap_width=24)

        for title, path in figure_items:
            image_page(pdf, title, [path], [""], cols=1, page_size=LANDSCAPE)

    print(f"Saved single-figure PDF: {OUTPUT_PDF}")
    print("Saved standalone figures:")
    for _title, path in figure_items:
        if path.name.startswith("paper_varied_"):
            print(f"  {path}")


if __name__ == "__main__":
    build_pdf()
