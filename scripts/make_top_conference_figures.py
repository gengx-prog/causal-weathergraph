#!/usr/bin/env python3
"""Generate journal-style multi-panel figures for Causal WeatherGraph.

Figure style follows Ma et al., "HiSTGNN: Hierarchical spatio-temporal graph
neural network for weather forecasting", Information Sciences 648 (2023):
white panels boxed by black spines, no gridlines, marker-coded line series,
muted blue-gray bars, red sequential heatmaps with in-cell annotations,
framed legends, and serif "(a) ..." sub-captions below each panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "outputs" / "reports"
EDGE_DIR = REPO_ROOT / "outputs" / "causal_edges"
REGION_DIR = REPO_ROOT / "outputs" / "regions"
OUT_DIR = REPO_ROOT / "outputs" / "figures" / "top_conference"
OUT_PDF = REPORT_DIR / "Causal_WeatherGraph_top_conference_figures_1979_2025.pdf"

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
    "all": "All",
    "DJF": "DJF",
    "JJA": "JJA",
    "high_humidity": "High humidity",
    "normal_humidity": "Normal humidity",
    "high_cloud": "High cloud",
    "normal_cloud": "Normal cloud",
}

BAND_ORDER = ["tropical", "midlatitude", "polar"]
BAND_LABELS = {"tropical": "Tropical", "midlatitude": "Midlatitude", "polar": "Polar"}

EDGE_ORDER = [
    "wind_to_humidity",
    "humidity_to_cloud_cover",
    "wind_to_cloud_cover",
    "humidity_to_humidity",
    "temperature_to_humidity",
    "temperature_to_cloud_cover",
]

EDGE_LABELS = {
    "wind_to_humidity": "Wind -> Humidity",
    "humidity_to_cloud_cover": "Humidity -> Cloud",
    "wind_to_cloud_cover": "Wind -> Cloud",
    "humidity_to_humidity": "Humidity -> Humidity",
    "temperature_to_humidity": "Temp -> Humidity",
    "temperature_to_cloud_cover": "Temp -> Cloud",
    "wind->humidity": "Wind -> Humidity",
    "humidity->cloud": "Humidity -> Cloud",
    "wind->cloud": "Wind -> Cloud",
}

# Categorical line palette (CVD-validated Okabe-Ito variant); every multi-line
# panel additionally separates series by marker shape and direct labels.
LINE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#8250C4"]
LINE_MARKERS = ["o", "s", "D", "^", "v", "*", "P"]

# Muted blue-gray bar family in the spirit of HiSTGNN Fig. 4.
BAR_LIGHT = "#b9cfe4"
BAR_MID = "#7d9cc4"
BAR_GRAYS = ["#7d9cc4", "#c9ced6", "#a8b4c4", "#dae3ec"]

INK = "#000000"
MUTED = "#555555"

HEAT_CMAP = "Reds"

# Pastel fills for the schematic overview, sampled from HiSTGNN Fig. 2.
DIAGRAM = {
    "teal": "#8ed3cf",
    "orange": "#f5c08a",
    "salmon": "#eda3a3",
    "yellow": "#f7eeb2",
    "lavender": "#d8d3ea",
    "edge": "#5f6b76",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "black",
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7.6,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "0.6",
            "legend.fancybox": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def read_report(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / name)


def read_edges(name: str) -> pd.DataFrame:
    return pd.read_csv(EDGE_DIR / name)


def read_regions() -> pd.DataFrame:
    return pd.read_csv(REGION_DIR / "region_metadata.csv")


def regime_label(value: object) -> str:
    return REGIME_LABELS.get(str(value), str(value))


def edge_label(value: object) -> str:
    return EDGE_LABELS.get(str(value), str(value).replace("_", " "))


def fmt_count(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def short_label(text: str, width: int = 18) -> str:
    if len(text) <= width:
        return text
    parts: list[str] = []
    cur: list[str] = []
    for token in text.split():
        trial = " ".join(cur + [token])
        if len(trial) > width and cur:
            parts.append(" ".join(cur))
            cur = [token]
        else:
            cur.append(token)
    if cur:
        parts.append(" ".join(cur))
    return "\n".join(parts)


def style_axis(ax: plt.Axes) -> None:
    """HiSTGNN look: all four black spines, no grid."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(length=3.0, width=0.8, color="black")


def panel_caption(ax: plt.Axes, tag: str, text: str, y: float = -0.24) -> None:
    """Serif sub-caption below the panel, mimicking LaTeX subcaptions."""
    ax.text(
        0.5,
        y,
        f"({tag}) {text}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.0,
        color="black",
        family="serif",
        clip_on=False,
    )


def save_figure(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{name}.png"
    pdf_path = OUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=360, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return png_path


def add_metric_tile(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    value: str,
    label: str,
    color: str,
    note: str | None = None,
) -> None:
    x, y = xy
    w, h = size
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="white",
            edgecolor=DIAGRAM["edge"],
            linewidth=0.8,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.010, y + 0.012),
            0.016,
            h - 0.024,
            boxstyle="round,pad=0.0,rounding_size=0.005",
            facecolor=color,
            edgecolor="none",
        )
    )
    ax.text(x + 0.042, y + h * 0.64, value, fontsize=15.5, weight="bold", color=INK)
    ax.text(x + 0.042, y + h * 0.38, label, fontsize=7.8, color=MUTED)
    if note:
        ax.text(x + 0.042, y + h * 0.16, note, fontsize=6.9, color="#777777")


def add_pipeline_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    lines: list[str],
    color: str,
) -> tuple[float, float]:
    x, y = xy
    w, h = size
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=color,
            edgecolor=DIAGRAM["edge"],
            linewidth=0.9,
        )
    )
    ax.text(x + w / 2, y + h - 0.035, title, va="center", ha="center", color=INK, fontsize=9.0, weight="bold")
    for idx, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.085 - idx * 0.045, line, va="top", ha="center", color=INK, fontsize=7.6)
    return x + w, y + h / 2


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color="black",
            shrinkA=8,
            shrinkB=8,
            connectionstyle="arc3,rad=0.0",
        )
    )


def make_figure_1_overview() -> Path:
    summary = read_report("summary_metrics.csv")
    null = read_report("null_model_summary.csv")
    enrich = read_report("null_model_variable_preserved_summary.csv")
    boot = read_report("bootstrap_stability_summary.csv")
    lat = read_report("latitude_band_chain_summary.csv")
    period = read_report("period_stability_summary.csv")

    physical_ratio = null.loc[null["edge_set"] == "physical_candidates", "significant_ratio"].iloc[0]
    wh_enrich = enrich.loc[enrich["edge_type"] == "wind->humidity", "enrichment"].iloc[0]
    hc_enrich = enrich.loc[enrich["edge_type"] == "humidity->cloud", "enrichment"].iloc[0]
    wh_stable = boot.loc[boot["edge_type"] == "wind_to_humidity", "stable_edges"].iloc[0]
    hc_stable = boot.loc[boot["edge_type"] == "humidity_to_cloud_cover", "stable_edges"].iloc[0]
    mid_score = lat.loc[lat["latitude_band"] == "midlatitude", "mean_chain_score"].iloc[0]
    period_min = period["overlap_with_full"].min()
    period_max = period["overlap_with_full"].max()
    all_sig = summary.loc[summary["regime"] == "all", "n_significant_edges"].iloc[0]

    fig = plt.figure(figsize=(13.8, 6.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (
            (0.04, 0.70),
            "Data",
            ["ERA5-style fields", "1979-2025, 6-hourly", "temperature, humidity,", "wind, cloud cover"],
            DIAGRAM["teal"],
        ),
        (
            (0.285, 0.70),
            "Region graph",
            ["66 spherical regions", "candidate_k = 2", "local plus nearest", "physical candidates"],
            DIAGRAM["orange"],
        ),
        (
            (0.53, 0.70),
            "Lagged testing",
            ["Granger-style OLS", "max_lag = 3", "FDR q < 0.05", "effect score retained"],
            DIAGRAM["salmon"],
        ),
        (
            (0.775, 0.70),
            "Evidence filters",
            ["null models", "block bootstrap", "direction controls", "robustness suite"],
            DIAGRAM["yellow"],
        ),
    ]
    centers = []
    for xy, title, lines, color in boxes:
        centers.append(add_pipeline_box(ax, xy, (0.185, 0.25), title, lines, color))
    for i in range(len(centers) - 1):
        add_arrow(ax, centers[i], (boxes[i + 1][0][0], centers[i + 1][1]))

    ax.text(0.04, 0.615, "Core claim translated into figure grammar", fontsize=9.5, weight="bold", color=INK)
    ax.text(
        0.04,
        0.573,
        "Stable Wind -> Humidity -> Cloud dependencies are stronger than random controls, reproducible across time blocks, and most concentrated in midlatitudes.",
        fontsize=8.0,
        color=MUTED,
    )

    tile_w, tile_h = 0.215, 0.15
    tile_data = [
        (f"{physical_ratio:.3f}", "physical candidate sig. ratio", DIAGRAM["teal"], f"{fmt_count(all_sig)} significant edges in all regime"),
        (f"{wh_enrich:.2f}x / {hc_enrich:.2f}x", "WH / HC enrichment", DIAGRAM["orange"], "vs variable-preserved random regions"),
        (f"{fmt_count(wh_stable)} / {fmt_count(hc_stable)}", "stable WH / HC edges", DIAGRAM["salmon"], "temporal block bootstrap"),
        (f"{mid_score:.3f}", "midlatitude mean chain score", DIAGRAM["yellow"], f"period overlap {period_min:.3f}-{period_max:.3f}"),
    ]
    for idx, item in enumerate(tile_data):
        add_metric_tile(ax, (0.04 + idx * 0.24, 0.365), (tile_w, tile_h), *item)

    x0, y0, w, h = 0.07, 0.15, 0.86, 0.14
    ax.plot([x0, x0 + w], [y0 + h / 2, y0 + h / 2], color="#9aa4ae", linewidth=1.0)
    stages = [
        ("WH", "Wind", "Humidity", DIAGRAM["teal"]),
        ("HC", "Humidity", "Cloud", DIAGRAM["orange"]),
        ("CTRL", "Null / bootstrap", "direction checks", DIAGRAM["salmon"]),
        ("ROB", "k / period /", "preprocess / region", DIAGRAM["yellow"]),
    ]
    for idx, (tag, line1, line2, color) in enumerate(stages):
        cx = x0 + (idx + 0.5) * w / len(stages)
        ax.scatter([cx], [y0 + h / 2], s=520, color=color, edgecolor=DIAGRAM["edge"], linewidth=0.9, zorder=3)
        ax.text(cx, y0 + h / 2, tag, ha="center", va="center", color=INK, fontsize=8.4, weight="bold")
        ax.text(cx, y0 - 0.008, line1, ha="center", va="top", color=INK, fontsize=8.0)
        ax.text(cx, y0 - 0.048, line2, ha="center", va="top", color=MUTED, fontsize=7.2)

    ax.text(
        0.04,
        0.032,
        "Interpretation boundary: edges are observational lagged directed dependencies, not interventional causal effects.",
        fontsize=7.7,
        style="italic",
        color=MUTED,
    )
    return save_figure(fig, "fig01_evidence_stack")


def draw_world_frame(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for lon in np.arange(-180, 181, 60):
        ax.axvline(lon, color="#dddddd", linewidth=0.5, zorder=0)
    for lat in np.arange(-60, 61, 30):
        ax.axhline(lat, color="#dddddd", linewidth=0.5, zorder=0)
    for lat in (-60.0, -23.5, 23.5, 60.0):
        ax.axhline(lat, color="#999999", linewidth=0.7, linestyle=(0, (4, 3)), zorder=0)
    ax.axhline(0, color="#bbbbbb", linewidth=0.7, zorder=0)
    ax.set_xlim(-185, 185)
    ax.set_ylim(-86, 86)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    style_axis(ax)


def aggregate_edges(edges: pd.DataFrame, edge_type: str) -> pd.DataFrame:
    sub = edges.loc[edges["edge_type"] == edge_type].copy()
    if sub.empty:
        return sub
    sub["abs_effect"] = sub["effect_score"].abs()
    agg_spec: dict[str, object] = {
        "source_lat": "mean",
        "source_lon": "mean",
        "target_lat": "mean",
        "target_lon": "mean",
        "distance_km": "mean",
        "lag": "mean",
        "abs_effect": "mean",
        "effect_score": "mean",
        "regime": "nunique",
    }
    if "stability" in sub.columns:
        agg_spec["stability"] = "mean"
    grouped = sub.groupby(["source_region", "target_region"], as_index=False).agg(agg_spec)
    grouped = grouped.rename(columns={"abs_effect": "strength", "regime": "n_regimes"})
    return grouped.sort_values("strength", ascending=False)


def draw_curved_edge(
    ax: plt.Axes,
    row: pd.Series,
    color: str,
    strength_max: float,
    idx: int,
) -> bool:
    x1, y1 = float(row["source_lon"]), float(row["source_lat"])
    x2, y2 = float(row["target_lon"]), float(row["target_lat"])
    if abs(x2 - x1) > 185:
        return False
    lw = 0.35 + 2.0 * min(1.0, float(row["strength"]) / max(strength_max, 1e-9))
    alpha = 0.25 + 0.40 * min(1.0, float(row["strength"]) / max(strength_max, 1e-9))
    rad = 0.08 if idx % 2 == 0 else -0.08
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=7.0 + lw * 1.6,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=4,
    )
    ax.add_patch(patch)
    return True


def draw_edge_map(ax: plt.Axes, stable_edges: pd.DataFrame, regions: pd.DataFrame, edge_type: str, color: str) -> None:
    draw_world_frame(ax)
    ax.scatter(
        regions["centroid_lon"],
        regions["centroid_lat"],
        s=9,
        c="#555555",
        edgecolor="none",
        alpha=0.9,
        zorder=3,
    )

    grouped = aggregate_edges(stable_edges, edge_type)
    cross = grouped.loc[grouped["distance_km"] > 100].head(95)
    local = grouped.loc[grouped["distance_km"] <= 100].head(45)
    strength_max = max(float(grouped["strength"].quantile(0.96)), 1e-9) if not grouped.empty else 1.0

    for idx, (_, row) in enumerate(cross.iterrows()):
        draw_curved_edge(ax, row, color, strength_max, idx)
    if not local.empty:
        sizes = 16 + 95 * np.clip(local["strength"].to_numpy(dtype=float) / strength_max, 0, 1)
        ax.scatter(
            local["target_lon"],
            local["target_lat"],
            s=sizes,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            alpha=0.6,
            zorder=5,
        )

    med_lag = grouped["lag"].median() if not grouped.empty else np.nan
    ax.text(
        0.985,
        0.03,
        f"{len(grouped):,} stable region links, median lag {med_lag:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "0.6", "linewidth": 0.6, "boxstyle": "square,pad=0.25"},
        zorder=6,
    )


def draw_small_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    fmt: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
) -> mpl.image.AxesImage:
    image = ax.imshow(values, cmap=HEAT_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if len(col_labels) > 3:
        ax.tick_params(axis="x", rotation=28)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    finite = values[np.isfinite(values)]
    lo = float(finite.min()) if vmin is None else float(vmin)
    hi = float(finite.max()) if vmax is None else float(vmax)
    threshold = lo + 0.55 * (hi - lo) if finite.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                text = "-"
                color = MUTED
            else:
                text = fmt.format(value)
                color = "white" if value > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.6, color=color)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)
    if cbar_label:
        cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(cbar_label, fontsize=8.0)
        cbar.ax.tick_params(labelsize=7.4)
        cbar.outline.set_linewidth(0.8)
    return image


def make_figure_2_spatial_pathway() -> Path:
    regions = read_regions()
    stable_edges = read_edges("stable_edges.csv")
    lat_chain = read_report("latitude_band_chain_summary.csv").copy()
    transfer = read_report("latitude_band_transfer_summary.csv").copy()

    lat_chain["latitude_band"] = pd.Categorical(lat_chain["latitude_band"], BAND_ORDER, ordered=True)
    lat_chain = lat_chain.sort_values("latitude_band")

    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.975, bottom=0.10, top=0.975, hspace=0.52, wspace=0.26, height_ratios=[1.12, 0.9])

    ax_a = fig.add_subplot(gs[0, 0])
    draw_edge_map(ax_a, stable_edges, regions, "wind_to_humidity", LINE_COLORS[0])
    panel_caption(ax_a, "a", "Wind $\\rightarrow$ Humidity stable region-level links", y=-0.22)

    ax_b = fig.add_subplot(gs[0, 1])
    draw_edge_map(ax_b, stable_edges, regions, "humidity_to_cloud_cover", LINE_COLORS[1])
    panel_caption(ax_b, "b", "Humidity $\\rightarrow$ Cloud stable region-level links", y=-0.22)

    ax_c = fig.add_subplot(gs[1, 0])
    x = np.arange(len(lat_chain))
    width = 0.34
    bars_total = ax_c.bar(x - width / 2, lat_chain["WHC_chains"], width, color=BAR_LIGHT, label="WHC chains")
    bars_stable = ax_c.bar(x + width / 2, lat_chain["stable_WHC_chains"], width, color=BAR_MID, label="Stable WHC chains")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([BAND_LABELS[str(v)] for v in lat_chain["latitude_band"]])
    ax_c.set_ylabel("Chain count")
    ax_c.set_ylim(0, lat_chain["WHC_chains"].max() * 1.22)
    ax_c2 = ax_c.twinx()
    (line_score,) = ax_c2.plot(x, lat_chain["mean_chain_score"], color="black", marker="D", linewidth=1.3, markersize=4, label="Mean chain score (right)")
    ax_c2.set_ylabel("Mean chain score")
    ax_c2.set_ylim(0, lat_chain["mean_chain_score"].max() * 1.32)
    ax_c.legend(handles=[bars_total, bars_stable, line_score], loc="upper left")
    style_axis(ax_c)
    panel_caption(ax_c, "c", "Latitude-band chain counts and mean chain score", y=-0.20)

    ax_d = fig.add_subplot(gs[1, 1])
    core_types = ["wind_to_humidity", "humidity_to_cloud_cover"]
    sub = transfer[transfer["edge_type"].isin(core_types)].copy()
    pivot = sub.pivot_table(index="source_band", columns="target_band", values="edge_count", aggfunc="sum").reindex(index=BAND_ORDER, columns=BAND_ORDER).fillna(0)
    draw_small_heatmap(
        ax_d,
        pivot.to_numpy(dtype=float),
        [BAND_LABELS[v] for v in pivot.index],
        [BAND_LABELS[v] for v in pivot.columns],
        fmt="{:.0f}",
        cbar_label="WH + HC edge count",
    )
    ax_d.set_xlabel("Target latitude band")
    ax_d.set_ylabel("Source latitude band")
    panel_caption(ax_d, "d", "Band-to-band edge counts for WH and HC", y=-0.26)

    return save_figure(fig, "fig02_spatial_pathway")


def edge_count_matrix(edges: pd.DataFrame) -> pd.DataFrame:
    counts = edges.groupby(["regime", "edge_type"]).size().rename("count").reset_index()
    pivot = counts.pivot(index="regime", columns="edge_type", values="count").reindex(index=REGIME_ORDER, columns=EDGE_ORDER).fillna(0)
    return pivot


def regime_distance_matrix() -> pd.DataFrame:
    comp = read_report("regime_comparison.csv")
    labels = REGIME_ORDER
    matrix = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels)
    for row in comp.itertuples(index=False):
        matrix.loc[row.regime_a, row.regime_b] = row.jaccard_distance
        matrix.loc[row.regime_b, row.regime_a] = row.jaccard_distance
    return matrix


def make_figure_3_regime() -> Path:
    sig_edges = read_edges("significant_edges.csv")
    chain = read_report("causal_chain_summary.csv").copy()
    balance = read_report("regime_sample_balance_jaccard.csv")
    balance_summary = read_report("regime_sample_balance_summary.csv")

    chain["regime"] = pd.Categorical(chain["regime"], REGIME_ORDER, ordered=True)
    chain = chain.sort_values("regime")

    fig = plt.figure(figsize=(13.8, 8.8))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.965, bottom=0.115, top=0.975, hspace=0.62, wspace=0.26)

    ax_a = fig.add_subplot(gs[0, 0])
    pivot = edge_count_matrix(sig_edges)
    row_labels = [regime_label(v) for v in pivot.index]
    col_labels = [short_label(edge_label(v), 14) for v in pivot.columns]
    draw_small_heatmap(ax_a, pivot.to_numpy(dtype=float), row_labels, col_labels, fmt="{:.0f}", cbar_label="Significant edges")
    panel_caption(ax_a, "a", "Significant edge counts per regime and edge type", y=-0.40)

    ax_b = fig.add_subplot(gs[0, 1])
    x = np.arange(len(chain))
    labels = [regime_label(v) for v in chain["regime"].astype(str)]
    bars = ax_b.bar(x, chain["n_chains"], color=BAR_LIGHT, label="WHC chain count")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([short_label(v, 12) for v in labels])
    ax_b.tick_params(axis="x", rotation=18)
    ax_b.set_ylabel("WHC chain count")
    ax_b.set_ylim(0, chain["n_chains"].max() * 1.18)
    ax_b2 = ax_b.twinx()
    (line_score,) = ax_b2.plot(x, chain["mean_chain_score"], color="black", marker="D", linewidth=1.3, markersize=4, label="Mean chain score (right)")
    ax_b2.set_ylabel("Mean chain score")
    ax_b2.set_ylim(0, chain["mean_chain_score"].max() * 1.25)
    ax_b.legend(handles=[bars, line_score], loc="upper right")
    style_axis(ax_b)
    panel_caption(ax_b, "b", "WHC chain count and mean chain score per regime", y=-0.34)

    ax_c = fig.add_subplot(gs[1, 0])
    dist = regime_distance_matrix()
    draw_small_heatmap(
        ax_c,
        dist.to_numpy(dtype=float),
        [regime_label(v) for v in dist.index],
        [short_label(regime_label(v), 12) for v in dist.columns],
        fmt="{:.2f}",
        cbar_label="Jaccard distance",
    )
    panel_caption(ax_c, "c", "Graph dissimilarity across regimes", y=-0.44)

    ax_d = fig.add_subplot(gs[1, 1])
    comparisons = list(balance_summary["comparison"])
    data = [balance.loc[balance["comparison"] == comp, "matched_jaccard"].to_numpy(dtype=float) for comp in comparisons]
    box = ax_d.boxplot(data, patch_artist=True, widths=0.5, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for key in ["whiskers", "caps"]:
        for item in box[key]:
            item.set_color("black")
            item.set_linewidth(1.0)
    for item in box["medians"]:
        item.set_color(LINE_COLORS[1])
        item.set_linewidth(1.3)
    originals = balance_summary["original_jaccard"].to_numpy(dtype=float)
    ax_d.scatter(np.arange(1, len(comparisons) + 1), originals, s=58, marker="D", color=LINE_COLORS[0], edgecolor="white", linewidth=0.7, zorder=3, label="Original Jaccard")
    ax_d.set_xticks(np.arange(1, len(comparisons) + 1))
    ax_d.set_xticklabels([short_label(c.replace(" vs ", "\nvs "), 19) for c in comparisons])
    ax_d.set_ylabel("Jaccard distance")
    ax_d.legend(loc="upper left")
    style_axis(ax_d)
    panel_caption(ax_d, "d", "Matched-sample balance stress test", y=-0.30)

    return save_figure(fig, "fig03_regime_mechanism")


def make_directionality_pivot() -> pd.DataFrame:
    df = read_report("directionality_control_summary.csv").copy()

    def path_group(label: str) -> str:
        items = set(label.replace(" ", "").split("->"))
        if items == {"Wind", "Humidity"}:
            return "Wind-Humidity"
        if items == {"Humidity", "Cloud"}:
            return "Humidity-Cloud"
        if items == {"Wind", "Cloud"}:
            return "Wind-Cloud"
        return label

    order = ["forward", "reverse", "future_to_past", "circular_shift", "time_shuffled"]
    df["path_group"] = df["edge_type"].map(path_group)
    pivot = df.pivot_table(index="test_type", columns="path_group", values="significant_ratio", aggfunc="mean")
    pivot = pivot.reindex(order)
    return pivot[["Wind-Humidity", "Humidity-Cloud", "Wind-Cloud"]]


def make_figure_4_controls() -> Path:
    null = read_report("null_model_summary.csv")
    enrich = read_report("null_model_variable_preserved_summary.csv")
    boot = read_report("bootstrap_stability_summary.csv").copy()
    direction = make_directionality_pivot()

    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.965, bottom=0.105, top=0.975, hspace=0.55, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    labels = ["Physical", "Random", "Distance\nmatched", "Variable\npreserved"]
    x = np.arange(len(null))
    bars = ax_a.bar(x, null["significant_ratio"], color=BAR_GRAYS, width=0.62)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("Significant ratio")
    ax_a.set_ylim(0, max(null["significant_ratio"]) * 1.18)
    ax_a2 = ax_a.twinx()
    (line_eff,) = ax_a2.plot(x, null["mean_effect_score"], color="black", marker="D", linewidth=1.3, markersize=4, label="Mean effect score (right)")
    ax_a2.set_ylabel("Mean effect score")
    ax_a2.set_ylim(0, max(null["mean_effect_score"]) * 1.35)
    ax_a.legend(handles=[line_eff], loc="upper right")
    style_axis(ax_a)
    panel_caption(ax_a, "a", "Null-model comparison of edge-set constructions", y=-0.26)

    ax_b = fig.add_subplot(gs[0, 1])
    y = np.arange(len(enrich))
    true = enrich["true_sig_ratio"].to_numpy(dtype=float)
    random = enrich["random_region_sig_ratio"].to_numpy(dtype=float)
    ax_b.hlines(y, random, true, color="#bbbbbb", linewidth=2.4, zorder=1)
    ax_b.scatter(random, y, s=64, facecolors="white", edgecolors="#555555", linewidth=1.0, label="Random-region", zorder=3)
    ax_b.scatter(true, y, s=72, color=LINE_COLORS[0], edgecolor="white", linewidth=0.7, label="Physical", zorder=4)
    for yi, row in enumerate(enrich.itertuples(index=False)):
        ax_b.text(max(row.true_sig_ratio, row.random_region_sig_ratio) + 0.022, yi, f"{row.enrichment:.2f}x", va="center", fontsize=8.0, color=INK)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([edge_label(v) for v in enrich["edge_type"]])
    ax_b.set_ylim(2.65, -0.95)
    ax_b.set_xlabel("Significant ratio")
    ax_b.set_xlim(0.35, 0.92)
    ax_b.legend(loc="upper right")
    style_axis(ax_b)
    panel_caption(ax_b, "b", "Enrichment against variable-preserved random regions", y=-0.30)

    ax_c = fig.add_subplot(gs[1, 0])
    boot["label"] = boot["edge_type"].map(edge_label)
    boot = boot.sort_values("stable_edges")
    y = np.arange(len(boot))
    ax_c.barh(y, boot["stable_edges"], color=BAR_MID, height=0.62)
    for yi, row in zip(y, boot.itertuples(index=False)):
        ax_c.text(row.stable_edges + 42, yi, f"{fmt_count(row.stable_edges)} ({row.stable_ratio:.2f})", va="center", fontsize=7.6, color=INK)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(boot["label"])
    ax_c.set_xlabel("Stable edge count")
    ax_c.set_xlim(0, boot["stable_edges"].max() * 1.30)
    style_axis(ax_c)
    panel_caption(ax_c, "c", "Temporal block-bootstrap stability (stable ratio in parentheses)", y=-0.30)

    ax_d = fig.add_subplot(gs[1, 1])
    draw_small_heatmap(
        ax_d,
        direction.to_numpy(dtype=float),
        ["forward", "reverse", "future-to-past", "circular-shift", "time-shuffled"],
        list(direction.columns),
        fmt="{:.2f}",
        vmin=0,
        vmax=0.9,
        cbar_label="Significant ratio",
    )
    panel_caption(ax_d, "d", "Temporal-direction controls", y=-0.34)

    return save_figure(fig, "fig04_controls")


def make_figure_5_robustness() -> Path:
    k_df = read_report("candidate_k_robustness_summary.csv")
    period = read_report("period_stability_summary.csv")
    pre_overlap = read_report("preprocessing_robustness_overlap.csv")
    region = read_report("region_robustness_summary.csv")

    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.965, bottom=0.105, top=0.975, hspace=0.55, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    x = k_df["k"].to_numpy(dtype=float)
    (line_chain,) = ax_a.plot(x, k_df["WHC_chains"], color=LINE_COLORS[0], marker=LINE_MARKERS[0], linewidth=1.4, markersize=4.5, label="WHC chains (left)")
    ax_a.set_xlabel("candidate_k_nearest")
    ax_a.set_ylabel("WHC chains")
    ax_a.set_xticks(x)
    ax_a2 = ax_a.twinx()
    (line_overlap,) = ax_a2.plot(x, k_df["overlap_with_k2"], color=LINE_COLORS[1], marker=LINE_MARKERS[1], linewidth=1.4, markersize=4.5, label="Overlap with k = 2 (right)")
    ax_a2.set_ylabel("Overlap with k = 2")
    ax_a2.set_ylim(0, 1.08)
    ax_a.legend(handles=[line_chain, line_overlap], loc="center right")
    style_axis(ax_a)
    panel_caption(ax_a, "a", "Candidate-neighborhood sensitivity", y=-0.26)

    ax_b = fig.add_subplot(gs[0, 1])
    era_bounds = [(1979, 1990), (1990, 2000), (2000, 2010), (2010, 2019), (2019, 2026)]
    mids = [(lo + hi) / 2 for lo, hi in era_bounds]
    for idx, (lo, hi) in enumerate(era_bounds):
        if idx % 2 == 1:
            ax_b.axvspan(lo, hi, color="#f2f2f2", zorder=0)
    labels = period["period"] + "\n" + period["time_range"]
    (line_full,) = ax_b.plot(mids, period["overlap_with_full"], color=LINE_COLORS[0], marker=LINE_MARKERS[0], linewidth=1.6, markersize=4.5, zorder=3, label="Overlap with full graph (left)")
    for xi, value in zip(mids, period["overlap_with_full"]):
        ax_b.text(xi, value + 0.0025, f"{value:.3f}", ha="center", va="bottom", fontsize=6.8, color=INK, zorder=4)
    ax_b.set_xticks(mids)
    ax_b.set_xticklabels(labels)
    ax_b.set_xlim(1979, 2026)
    ax_b.set_ylabel("Overlap with full graph")
    ax_b.set_ylim(0.86, 0.928)
    ax_b.axvline(2023, color=LINE_COLORS[5], linewidth=0.9, linestyle=(0, (2, 2)), zorder=2)
    ax_b.text(2022.2, 0.8985, "CDS continuation\nfrom 2023", fontsize=6.6, color=LINE_COLORS[5], ha="right", va="center")
    ax_b2 = ax_b.twinx()
    (line_sig,) = ax_b2.plot(mids, period["significant_ratio"], color=LINE_COLORS[1], marker=LINE_MARKERS[1], linewidth=1.6, markersize=4.5, zorder=3, label="Significant ratio (right)")
    ax_b2.set_ylabel("Significant ratio")
    ax_b2.set_ylim(0.64, 0.77)
    ax_b.legend(handles=[line_full, line_sig], loc="lower left")
    style_axis(ax_b)
    panel_caption(ax_b, "b", "Historical-period stability across five eras", y=-0.30)

    ax_c = fig.add_subplot(gs[1, 0])
    sub = pre_overlap.set_index("preprocess_setting")
    cols = ["edge_overlap_with_P1", "WH_overlap_with_P1", "HC_overlap_with_P1", "WHC_chain_overlap_with_P1"]
    values = sub[cols].to_numpy(dtype=float)
    row_labels = [s.replace("_", "\n") for s in sub.index]
    col_labels = ["All edges", "WH", "HC", "WHC chain"]
    draw_small_heatmap(
        ax_c,
        values,
        row_labels,
        col_labels,
        fmt="{:.2f}",
        vmin=0.84,
        vmax=1.0,
        cbar_label="Overlap with P1",
    )
    panel_caption(ax_c, "c", "Preprocessing robustness (overlap with the P1 baseline)", y=-0.40)

    ax_d = fig.add_subplot(gs[1, 1])
    method_styles = {
        "latlon_bins": (LINE_COLORS[0], LINE_MARKERS[0]),
        "kmeans_sphere": (LINE_COLORS[1], LINE_MARKERS[1]),
    }
    for method, sub_df in region.groupby("method"):
        color, marker = method_styles.get(method, (LINE_COLORS[2], LINE_MARKERS[2]))
        ax_d.scatter(
            sub_df["actual_regions"],
            sub_df["WHC_chains"],
            s=46,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=method.replace("_", " "),
            zorder=3,
        )
    for row in region.itertuples(index=False):
        dy = 900 if row.method == "latlon_bins" else -900
        ax_d.text(row.actual_regions + 2.4, row.WHC_chains + dy, f"{row.setting} ({fmt_count(row.WHC_chains)})", va="center", fontsize=7.2, color=INK)
    ax_d.set_xlabel("Actual regions")
    ax_d.set_ylabel("WHC chains")
    ax_d.set_xlim(region["actual_regions"].min() - 12, region["actual_regions"].max() + 34)
    ax_d.set_ylim(region["WHC_chains"].min() * 0.75, region["WHC_chains"].max() * 1.14)
    ax_d.legend(loc="upper left")
    style_axis(ax_d)
    panel_caption(ax_d, "d", "Region-aggregation sensitivity", y=-0.26)

    return save_figure(fig, "fig05_robustness")


def per_lag_fractions() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-lag share of significant edges (lag 1..12 -> 6..72 h) for the four
    core edge types; mean and std are taken across regimes so the shaded band
    reflects regime-to-regime variability."""
    edges = read_edges("targeted_maxlag12_significant_edges.csv")
    core = EDGE_ORDER[:4]
    lags = np.arange(1, 13)
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for etype in core:
        sub = edges.loc[edges["edge_type"] == etype]
        per_regime = []
        for _, grp in sub.groupby("regime"):
            counts = grp["lag"].value_counts().reindex(lags, fill_value=0).to_numpy(dtype=float)
            per_regime.append(counts / max(counts.sum(), 1.0))
        arr = np.vstack(per_regime)
        means[etype] = arr.mean(axis=0)
        stds[etype] = arr.std(axis=0)
    return lags * 6.0, means, stds


def make_figure_6_temporal_mechanism() -> Path:
    lag = read_report("targeted_maxlag12_lag_summary.csv")
    long_chain = read_report("targeted_maxlag12_causal_chain_summary.csv").copy()
    mediation = read_report("wind_cloud_humidity_mediation_summary.csv").copy()
    extreme = read_report("extreme_precursor_summary.csv")

    long_chain["regime"] = pd.Categorical(long_chain["regime"], REGIME_ORDER, ordered=True)
    long_chain = long_chain.sort_values("regime")
    mediation["regime"] = pd.Categorical(mediation["regime"], REGIME_ORDER, ordered=True)
    mediation = mediation.sort_values("regime")

    fig = plt.figure(figsize=(13.8, 8.8))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.965, bottom=0.115, top=0.975, hspace=0.62, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    hours, lag_means, lag_stds = per_lag_fractions()
    ylo = 0.04
    ymax = max(float((lag_means[e] + lag_stds[e]).max()) for e in lag_means) * 1.14
    ax_a.axvspan(3, 12, color="#eef3f8", zorder=0)
    ax_a.axvspan(12, 48, color="#fdf6ec", zorder=0)
    ax_a.axvspan(48, 75, color="#fdefee", zorder=0)
    for xz, zone in ((7.5, "$\\leq$12 h"), (30, "12-48 h"), (61, "48-72 h")):
        ax_a.text(xz, ymax - 0.0025, zone, ha="center", va="top", fontsize=6.8, color="#888888")
    lag_series = [
        ("wind_to_humidity", LINE_COLORS[0], "o", "-", 1.6, 1.0),
        ("humidity_to_cloud_cover", LINE_COLORS[1], "s", "-", 1.6, 1.0),
        ("wind_to_cloud_cover", LINE_COLORS[2], "^", "--", 1.1, 0.7),
        ("humidity_to_humidity", "#888888", "v", "--", 1.1, 0.7),
    ]
    for etype, color, marker, ls, lw, alpha in lag_series:
        mean, std = lag_means[etype], lag_stds[etype]
        if ls == "-":
            ax_a.fill_between(hours, mean - std, mean + std, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax_a.plot(hours, mean, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=3.8, alpha=alpha, zorder=3, label=edge_label(etype).replace(" Cover", ""))
    med_hours = float(lag["median_lag"].median()) * 6
    ax_a.axvline(med_hours, color=LINE_COLORS[5], linewidth=0.9, linestyle=(0, (2, 2)), zorder=2)
    ax_a.text(med_hours + 1.3, ylo + (ymax - ylo) * 0.13, f"median lag = {med_hours:.0f} h", fontsize=6.8, color=LINE_COLORS[5], rotation=90, va="center")
    ax_a.set_xlim(3, 75)
    ax_a.set_xticks(np.arange(6, 73, 6))
    ax_a.set_ylim(ylo, ymax)
    ax_a.set_xlabel("Lag (hours)")
    ax_a.set_ylabel("Fraction of significant edges")
    ax_a.legend(loc="upper right", ncols=2, fontsize=6.9)
    style_axis(ax_a)
    panel_caption(ax_a, "a", "Per-lag distribution under max_lag = 12 ($\\pm$1$\\sigma$ across regimes)", y=-0.26)

    ax_b = fig.add_subplot(gs[0, 1])
    x = np.arange(len(long_chain))
    bars = ax_b.bar(x, long_chain["n_chains"], color=BAR_LIGHT, label="WHC chains")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([short_label(regime_label(v), 12) for v in long_chain["regime"].astype(str)])
    ax_b.tick_params(axis="x", rotation=18)
    ax_b.set_ylabel("WHC chains, max_lag = 12")
    ax_b.set_ylim(0, long_chain["n_chains"].max() * 1.38)
    ax_b2 = ax_b.twinx()
    (line_lag,) = ax_b2.plot(x, long_chain["mean_total_lag"] * 6, color="black", marker="D", linewidth=1.3, markersize=4, label="Mean total lag, hours (right)")
    ax_b2.set_ylabel("Mean total lag (hours)")
    ax_b2.set_ylim(0, (long_chain["mean_total_lag"] * 6).max() * 1.38)
    ax_b.legend(handles=[bars, line_lag], loc="upper center", ncols=2)
    style_axis(ax_b)
    panel_caption(ax_b, "b", "Long-lag WHC chains and mean total lag by regime", y=-0.34)

    ax_c = fig.add_subplot(gs[1, 0])
    x_pair = np.array([0, 1], dtype=float)
    final_points: list[tuple[float, str, str, float]] = []
    for idx, row in enumerate(mediation.itertuples(index=False)):
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        y = np.array([row.mean_beta_direct, row.mean_beta_controlled], dtype=float)
        ax_c.plot(x_pair, y, color=color, marker="o", linewidth=1.3, markersize=4)
        final_points.append((y[1], regime_label(row.regime), color, row.mean_reduction))
    final_points.sort(key=lambda item: item[0])
    point_y_values = np.array([item[0] for item in final_points], dtype=float)
    min_gap = 0.00016
    low = float(point_y_values.min() - 0.00004)
    high = float(point_y_values.max() + 0.00004)
    needed_span = min_gap * max(0, len(final_points) - 1)
    if high - low < needed_span:
        center = (high + low) / 2
        low = center - needed_span / 2
        high = center + needed_span / 2
    label_y = np.linspace(low, high, len(final_points))
    for (point_y, label, color, reduction), ypos in zip(final_points, label_y):
        ax_c.plot([1.0, 1.055], [point_y, ypos], color=color, linewidth=0.7, alpha=0.8)
        ax_c.text(1.075, ypos, f"{label} ({reduction:.3f})", va="center", fontsize=7.0, color=INK)
    ax_c.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax_c.set_xticks(x_pair)
    ax_c.set_xticklabels(["Direct W $\\rightarrow$ C", "Humidity controlled"])
    all_y = np.r_[mediation["mean_beta_direct"].to_numpy(dtype=float), mediation["mean_beta_controlled"].to_numpy(dtype=float), label_y]
    ax_c.set_xlim(-0.08, 1.78)
    ax_c.set_ylim(all_y.min() - 0.00010, max(0.00005, all_y.max() + 0.00010))
    ax_c.set_ylabel("Mean beta")
    style_axis(ax_c)
    panel_caption(ax_c, "c", "Humidity mediation (mean reduction in parentheses)", y=-0.24)

    ax_d = fig.add_subplot(gs[1, 1])
    windows = [6, 12, 24, 48]
    event_styles = [
        ("high_cloud", "High cloud", LINE_COLORS[0], "o"),
        ("high_humidity", "High humidity", LINE_COLORS[1], "s"),
    ]
    peak = 0.0
    for event, label, color, marker in event_styles:
        sub = extreme.loc[extreme["event_type"] == event].set_index("precursor_window_hours").reindex(windows)
        y = sub["WHC_chain_count"].fillna(0).to_numpy(dtype=float)
        peak = max(peak, float(y.max()))
        ax_d.plot(windows, y, color=color, marker=marker, linewidth=1.6, markersize=4.5, label=label)
        for xi, yi in zip(windows, y):
            if yi == 0:
                ax_d.text(xi, yi - peak * 0.035, fmt_count(yi), ha="center", va="top", fontsize=7.0, color=INK)
            else:
                ax_d.text(xi, yi + peak * 0.035, fmt_count(yi), ha="center", va="bottom", fontsize=7.0, color=INK)
    ax_d.set_xlim(2, 52)
    ax_d.set_xticks(windows)
    ax_d.set_ylim(-peak * 0.13, peak * 1.18)
    ax_d.set_xlabel("Precursor window (hours)")
    ax_d.set_ylabel("WHC precursor chains")
    ax_d.legend(loc="upper left")
    style_axis(ax_d)
    panel_caption(ax_d, "d", "WHC precursor response ahead of extreme regimes", y=-0.26)

    return save_figure(fig, "fig06_temporal_mechanism")


def build_contact_sheet(paths: list[Path]) -> Path:
    from PIL import Image, ImageDraw

    thumb_w, thumb_h = 520, 340
    cols = 2
    rows = int(np.ceil(len(paths) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w - 20, thumb_h - 42), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w + 10
        y = (idx // cols) * thumb_h + 32
        canvas.paste(image, (x, y))
        draw.text((x, y - 22), path.stem, fill=(0, 0, 0))
    out = OUT_DIR / "top_conference_contact_sheet.png"
    canvas.save(out)
    return out


def build_pdf(paths: list[Path]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for path in paths:
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(13.8, 8.5))
            ax.axis("off")
            ax.imshow(image)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        make_figure_1_overview(),
        make_figure_2_spatial_pathway(),
        make_figure_3_regime(),
        make_figure_4_controls(),
        make_figure_5_robustness(),
        make_figure_6_temporal_mechanism(),
    ]
    contact = build_contact_sheet(paths)
    build_pdf(paths)
    print("Saved journal-style figures:")
    for path in paths:
        print(f"  {path}")
    print(f"  {contact}")
    print(f"Saved PDF bundle: {OUT_PDF}")


if __name__ == "__main__":
    main()
