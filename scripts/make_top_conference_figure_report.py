#!/usr/bin/env python3
"""Build a Chinese PDF report from the standalone top-conference figures."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "outputs" / "figures" / "top_conference"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"
OUTPUT_PDF = REPORT_DIR / "Causal_WeatherGraph_顶会风格单图报告_1979_2025.pdf"

FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else FontProperties()

LANDSCAPE = (13.8, 8.5)
INK = "#17202A"
MUTED = "#667085"
BLUE = "#2C7FB8"
ORANGE = "#D95F02"
GREEN = "#009E73"
PURPLE = "#7570B3"
PAPER = "#FBFCFE"


FIGURE_ITEMS = [
    {
        "file": "fig01_evidence_stack.png",
        "title": "Figure 1. Causal WeatherGraph 证据链总览",
        "caption": "从数据、区域图、滞后依赖检验到 null model、bootstrap、方向性和稳健性检验，概括本文核心证据链。",
        "takeaway": "适合作为论文方法与结果之间的总览图，强调本文报告的是观测数据中的 lagged directed dependencies。",
    },
    {
        "file": "fig02_spatial_pathway.png",
        "title": "Figure 2. Wind -> Humidity -> Cloud 空间路径组织",
        "caption": "展示稳定 Wind -> Humidity 与 Humidity -> Cloud 区域链接、纬度带链条强度以及 band-to-band transfer。",
        "takeaway": "适合作为主结果图，突出中纬度链条数量、稳定链条数量和链条得分更高的物理解释。",
    },
    {
        "file": "fig03_regime_mechanism.png",
        "title": "Figure 3. Regime-aware 路径行为",
        "caption": "比较不同 regime 下的边类型密度、WHC 链条数与链条得分、图结构距离和样本量平衡检查。",
        "takeaway": "适合支撑 regime-aware 是必要设计，而不是简单全样本图的附属分析。",
    },
    {
        "file": "fig04_controls.png",
        "title": "Figure 4. 非随机性、稳定性与时间方向对照",
        "caption": "综合 null model、变量保持随机对照富集、bootstrap 稳定边和 temporal-direction controls。",
        "takeaway": "适合作为审稿人最关心的反证图：排除任意边选择、变量类型偏差和伪时间对齐。",
    },
    {
        "file": "fig05_robustness.png",
        "title": "Figure 5. 图构造与预处理稳健性",
        "caption": "覆盖 candidate-k、历史时期切分、预处理方案和区域聚合方式的敏感性分析。",
        "takeaway": "适合作为主文或补充主图，说明核心 WHC 路径不是某个超参数或区域划分的偶然结果。",
    },
    {
        "file": "fig06_temporal_mechanism.png",
        "title": "Figure 6. 时间尺度、中介与极端事件前兆",
        "caption": "展示 max_lag=12 长滞后分布、长滞后链条、湿度中介坡度图和 extreme-regime precursor 响应。",
        "takeaway": "适合作为机制扩展图，说明短滞后峰值与 24-48 小时长尾可以共同存在。",
    },
]


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


def add_report_header(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.045, 0.95, title, ha="left", va="top", fontsize=18, weight="bold", color=INK, fontproperties=FONT)
    if subtitle:
        fig.text(0.045, 0.905, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED, fontproperties=FONT)


def add_footer(fig: plt.Figure, page_label: str) -> None:
    fig.text(0.045, 0.035, "Causal WeatherGraph | 1979-2025", ha="left", va="bottom", fontsize=8, color=MUTED, fontproperties=FONT)
    fig.text(0.955, 0.035, page_label, ha="right", va="bottom", fontsize=8, color=MUTED, fontproperties=FONT)


def wrapped_text(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def read_metric(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / name)


def cover_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor("#F7F9FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0.045, 0.17), 0.015, 0.67, color=BLUE, transform=fig.transFigure, clip_on=False))
    fig.text(0.085, 0.78, "Causal WeatherGraph", ha="left", va="top", fontsize=33, weight="bold", color=INK, fontproperties=FONT)
    fig.text(0.085, 0.69, "顶会风格单图实验报告", ha="left", va="top", fontsize=24, weight="bold", color=INK, fontproperties=FONT)
    fig.text(
        0.085,
        0.59,
        "基于当前 standalone figures 自动生成\n每张主图单页展示，并附中文图注与论文写作要点",
        ha="left",
        va="top",
        fontsize=13,
        color=MUTED,
        linespacing=1.55,
        fontproperties=FONT,
    )

    summary = read_metric("summary_metrics.csv")
    chain = read_metric("causal_chain_summary.csv")
    null = read_metric("null_model_summary.csv")
    physical_ratio = null.loc[null["edge_set"] == "physical_candidates", "significant_ratio"].iloc[0]
    all_edges = summary.loc[summary["regime"] == "all", "n_significant_edges"].iloc[0]
    all_chains = chain.loc[chain["regime"] == "all", "n_chains"].iloc[0]

    metrics = [
        ("1979-2025", "data range"),
        (f"{int(all_edges):,}", "all-regime significant edges"),
        (f"{int(all_chains):,}", "all-regime WHC chains"),
        (f"{physical_ratio:.3f}", "physical candidate sig. ratio"),
    ]
    for idx, (value, label) in enumerate(metrics):
        x = 0.085 + idx * 0.21
        ax.add_patch(
            plt.Rectangle((x, 0.275), 0.17, 0.13, facecolor="white", edgecolor="#D0D7E2", linewidth=0.8, transform=fig.transFigure)
        )
        fig.text(x + 0.018, 0.365, value, ha="left", va="top", fontsize=18, weight="bold", color=INK, fontproperties=FONT)
        fig.text(x + 0.018, 0.315, label, ha="left", va="top", fontsize=8.4, color=MUTED, fontproperties=FONT)

    fig.text(
        0.085,
        0.14,
        f"Output: {OUTPUT_PDF.relative_to(REPO_ROOT)}",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
        fontproperties=FONT,
    )
    pdf.savefig(fig)
    plt.close(fig)


def index_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor("white")
    add_report_header(fig, "图目录", "本报告使用 outputs/figures/top_conference 下的当前单独图片生成。")
    ax = fig.add_axes([0.055, 0.12, 0.89, 0.72])
    ax.axis("off")

    y = 0.94
    colors = [BLUE, ORANGE, GREEN, PURPLE, "#7B3294", "#A6761D"]
    for idx, item in enumerate(FIGURE_ITEMS, start=1):
        color = colors[(idx - 1) % len(colors)]
        ax.add_patch(plt.Rectangle((0.0, y - 0.055), 0.018, 0.052, color=color, transform=ax.transAxes))
        ax.text(0.03, y, item["title"], ha="left", va="top", fontsize=11, weight="bold", color=INK, fontproperties=FONT, transform=ax.transAxes)
        ax.text(
            0.03,
            y - 0.04,
            wrapped_text(item["caption"], 95),
            ha="left",
            va="top",
            fontsize=8.3,
            color=MUTED,
            fontproperties=FONT,
            transform=ax.transAxes,
        )
        y -= 0.145

    add_footer(fig, "Index")
    pdf.savefig(fig)
    plt.close(fig)


def figure_page(pdf: PdfPages, item: dict[str, str], page_number: int, total_pages: int) -> None:
    image_path = FIGURE_DIR / item["file"]
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor("white")
    add_report_header(fig, item["title"], item["caption"])

    img = Image.open(image_path).convert("RGB")
    ax = fig.add_axes([0.045, 0.19, 0.91, 0.66])
    ax.axis("off")
    ax.imshow(img)

    note_ax = fig.add_axes([0.045, 0.075, 0.91, 0.075])
    note_ax.axis("off")
    note_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#F7F9FC", edgecolor="#D0D7E2", linewidth=0.8, transform=note_ax.transAxes))
    note_ax.text(0.018, 0.68, "论文写作要点", ha="left", va="top", fontsize=8.6, weight="bold", color=INK, fontproperties=FONT)
    note_ax.text(0.15, 0.68, wrapped_text(item["takeaway"], 118), ha="left", va="top", fontsize=8.1, color=MUTED, fontproperties=FONT)

    add_footer(fig, f"{page_number}/{total_pages}")
    pdf.savefig(fig)
    plt.close(fig)


def build_report() -> None:
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    total_pages = len(FIGURE_ITEMS) + 2
    with PdfPages(OUTPUT_PDF) as pdf:
        cover_page(pdf)
        index_page(pdf)
        for idx, item in enumerate(FIGURE_ITEMS, start=3):
            figure_page(pdf, item, idx, total_pages)
    print(f"Saved figure report: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_report()
