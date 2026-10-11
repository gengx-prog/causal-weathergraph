"""Reproduce circular experiment networks from the packaged, unchanged candidate-level data.

Run: python render_networks.py
Dependencies: matplotlib, numpy, pandas. No reanalysis arrays are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Wedge
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STEM = "circular_networks"
BLUE, ORANGE = "#1974AA", "#CE7025"
INK, GREY = "#243746", "#6C7680"
RAMP = ["#E1E8ED", "#ADC9DA", "#639CBF", "#24658E"]
ORDER = ["wind_to_humidity", "humidity_to_cloud_cover", "wind_to_cloud_cover",
         "humidity_to_humidity", "temperature_to_humidity", "temperature_to_cloud_cover"]
LABELS = ["W\u2192H", "H\u2192C", "W\u2192C", "H\u2192H", "T\u2192H", "T\u2192C"]
KEY = ["source_region", "target_region", "source_var", "target_var", "lag"]
WIDTHS = {1: 0.65, 2: 1.10, 3: 1.55}
LAT_LABELS = ["60\u201390\u00b0S", "30\u201360\u00b0S", "0\u201330\u00b0S",
              "0\u201330\u00b0N", "30\u201360\u00b0N", "60\u201390\u00b0N"]


def load_and_validate():
    t = pd.read_csv(DATA / "three_stage_all_candidates.csv")
    r = pd.read_csv(DATA / "regions.csv").sort_values("region")
    s = pd.read_csv(DATA / "three_stage_summary.csv").set_index("edge_type")
    assert t.shape == (2574, 72) and r.shape == (66, 4)
    assert not t.duplicated(KEY).any()
    assert r.region.tolist() == list(range(66))
    assert set(t.lag) == {1, 2, 3}
    assert t[KEY].notna().all().all()
    assert t.source_region.isin(r.region).all() and t.target_region.isin(r.region).all()
    assert (t.s2_pass <= t.s1_pass).all()
    assert (t.eval_wb2_replicated <= t.s2_pass).all()
    for et in ORDER + ["ALL"]:
        g = t if et == "ALL" else t[t.edge_type == et]
        expected = {"tested": len(g), "stage1_screened": int(g.s1_pass.sum()),
                    "stage2_confirmed": int(g.s2_pass.sum()),
                    "eval_wb2_replicated": int(g.eval_wb2_replicated.sum()),
                    "eval_cds_replicated": int(g.eval_cds_replicated.sum()),
                    "eval_wb2_negctrl_n": int((~g.s2_pass).sum()),
                    "eval_wb2_negctrl_replicated": int(((~g.s2_pass) & (g.eval_wb2_negctrl_q < .05)).sum())}
        for field, val in expected.items():
            assert val == int(s.loc[et, field]), (et, field, val, s.loc[et, field])
    assert t.eval_wb2_replicated.sum() == 760
    assert t.eval_cds_replicated.sum() == 667
    return t, r, s


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": "#74818C",
        "axes.linewidth": .65, "axes.labelsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    })


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=.6)
    ax.grid(False)


def panel_title(ax, tag, title):
    ax.text(0, 1.055, f"({tag})  {title}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.3, fontweight="bold")


def plot_stage_counts(ax, summary):
    stages = [("tested", "Tested"), ("stage1_screened", "Screened"),
              ("stage2_confirmed", "Confirmed"), ("eval_wb2_replicated", "Replicated")]
    w = .19
    for k, (field, label) in enumerate(stages):
        ax.bar(np.arange(6) + (k - 1.5) * w,
               [summary.loc[e, field] for e in ORDER], width=w * .94,
               color=RAMP[k], linewidth=.35, edgecolor="#7291A5", label=label)
    ax.set_xticks(range(6), LABELS)
    ax.set_yticks([0, 200, 400, 600])
    ax.set_ylim(0, 820)
    ax.set_ylabel("Edge\u2013lag hypotheses", labelpad=3)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=7.5,
              handlelength=1.1, handletextpad=.45, columnspacing=1.0,
              borderaxespad=.05, labelspacing=.45)
    panel_title(ax, "a", "Hypotheses retained at each stage")
    style(ax)


def plot_rates(ax, summary):
    families = [("stage2_confirmed", "eval_wb2_replicated", "Confirmed", BLUE, "o", -.12),
                ("eval_wb2_negctrl_n", "eval_wb2_negctrl_replicated", "Unconfirmed comparison", ORANGE, "s", .12)]
    for denominator, numerator, name, color, marker, offset in families:
        vals = [summary.loc[e, numerator] / summary.loc[e, denominator] for e in ORDER + ["ALL"]]
        ax.plot(np.arange(7) + offset, vals, linestyle="none", marker=marker,
                markersize=4.2, markeredgewidth=.65, color=color, label=name)
    ax.axvline(5.5, color="#C6CDD3", linewidth=.7)
    ax.set_xticks(range(7), LABELS + ["All"])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, .2, .4, .6, .8, 1], ["0", "20", "40", "60", "80", "100"])
    ax.set_ylabel("Replication rate (%)", labelpad=3)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=7.5,
              handlelength=1.0, handletextpad=.35, columnspacing=.7,
              borderaxespad=.1)
    ax.text(6.10, .53, "49.0", fontsize=7.5, color=BLUE, ha="center")
    ax.text(6.10, .145, "9.3", fontsize=7.5, color=ORANGE, ha="center")
    panel_title(ax, "b", "Descriptive rates (2019\u20132023)")
    style(ax)


def region_positions(regions):
    """Six labelled latitude sectors; longitude increases clockwise within each."""
    angles = {}
    starts = {}
    for group, (_, rows) in enumerate(regions.groupby("centroid_lat_deg", sort=True)):
        rows = rows.sort_values("centroid_lon_deg")
        assert len(rows) == 11
        start = 90 - group * 60 - 3
        starts[group] = start
        for j, r in enumerate(rows.itertuples()):
            angles[int(r.region)] = np.deg2rad(start - (j + .5) * 54 / 11)
    return angles, starts


def polar(radius, angle):
    return np.array([radius * np.cos(angle), radius * np.sin(angle)])


def aggregate_network(t, regions, edge_type):
    selected = t[(t.edge_type == edge_type) & t.eval_wb2_replicated].copy()
    agg = selected.groupby(["source_region", "target_region"], as_index=False).agg(
        replicated_lags=("lag", "size"), lag_steps=("lag", lambda a: ";".join(map(str, sorted(a)))))
    agg["edge_type"] = edge_type
    coords = regions.set_index("region")
    for side in ("source", "target"):
        for short, original in (("lat", "centroid_lat_deg"), ("lon", "centroid_lon_deg")):
            agg[f"{side}_{short}_deg"] = agg[f"{side}_region"].map(coords[original])
    agg["local"] = agg.source_region == agg.target_region
    agg["crosses_dateline_in_old_map"] = (~agg.local) & ((agg.source_lon_deg - agg.target_lon_deg).abs() > 185)
    assert agg.replicated_lags.between(1, 3).all()
    assert int(agg.replicated_lags.sum()) == len(selected)
    return selected, agg


def plot_network(ax, t, regions, edge_type, color, tag, title):
    selected, agg = aggregate_network(t, regions, edge_type)
    angles, starts = region_positions(regions)
    ax.set_aspect("equal")
    ax.set_xlim(-1.40, 1.40)
    ax.set_ylim(-1.43, 1.43)
    ax.axis("off")
    band_colors = ["#E5EBEF", "#F1F4F6", "#E5EBEF", "#F1F4F6", "#E5EBEF", "#F1F4F6"]
    for group, start in starts.items():
        ax.add_patch(Wedge((0, 0), 1.285, start - 54, start, width=.055,
                           facecolor=band_colors[group], edgecolor="white", linewidth=.3))
        mid = np.deg2rad(start - 27)
        pos = polar(1.34, mid)
        sector_rotation = (np.rad2deg(mid) - 90 + 180) % 360 - 180
        if sector_rotation < -90:
            sector_rotation += 180
        elif sector_rotation > 90:
            sector_rotation -= 180
        ax.text(*pos, LAT_LABELS[group], ha="center", va="center", fontsize=7.5,
                color="#536775", rotation=sector_rotation, rotation_mode="anchor")
    cross = agg[~agg.local]
    for row in cross.itertuples():
        p0 = polar(.925, angles[row.source_region])
        p3 = polar(.895, angles[row.target_region])
        # Curves indicate source-target association only; no geographic geometry.
        chord = p3 - p0
        unit_normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
        bend = .018 * unit_normal
        vertices = [p0, .32 * p0 + bend, .32 * p3 + bend, p3]
        path = MplPath(vertices, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=5.2,
                                linewidth=WIDTHS[int(row.replicated_lags)], color=color,
                                alpha=.78, zorder=2)
        ax.add_patch(patch)
    for row in agg[agg.local].itertuples():
        p = polar(.95, angles[row.target_region])
        ax.scatter(*p, s=27, facecolor="white", edgecolor=color,
                   linewidth=WIDTHS[int(row.replicated_lags)], zorder=4)
    for region in regions.region:
        p = polar(.95, angles[region])
        ax.scatter(*p, s=3.3, facecolor=INK, edgecolor="none", zorder=5)
        # Alternate radial positions and keep individual labels outward-facing.
        # Text height, rather than two-digit width, occupies the angular spacing.
        label_pos = polar(1.065 if region % 2 == 0 else 1.145, angles[region])
        rotation = np.rad2deg(angles[region])
        if rotation < -90 or rotation > 90:
            rotation += 180
        label = ax.text(*label_pos, str(region), ha="center", va="center", rotation=rotation,
                        fontsize=7.2, color=INK, rotation_mode="anchor")
        label.set_gid(f"region-label-{tag}-{region:02d}")
    confirmed = int(((t.edge_type == edge_type) & t.s2_pass).sum())
    ax.text(0, 1.39, f"{len(selected)} / {confirmed} confirmed edge\u2013lags replicate",
            ha="center", va="center", color=GREY, fontsize=7.5)
    ax.text(0, 1.505, f"({tag})  {title}", ha="center", va="bottom",
            fontsize=8.4, fontweight="bold", color=color)
    ax.text(0, -1.38, f"{len(agg)} pairs: {int(agg.local.sum())} local + {len(cross)} cross-region",
            ha="center", va="top", fontsize=7.5)
    return agg


def legend(fig):
    ax = fig.add_axes([.10, .023, .83, .053])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0, .69, "Replicated lags per pair", fontsize=7.5, va="center")
    for n in (1, 2, 3):
        x = .325 + (n - 1) * .115
        ax.plot([x, x + .048], [.69, .69], color=INK, linewidth=WIDTHS[n])
        ax.text(x + .06, .69, str(n), fontsize=7.5, va="center")
    ax.scatter(.715, .69, s=27, facecolor="white", edgecolor=INK, linewidth=.9)
    ax.text(.735, .69, "local", fontsize=7.5, va="center")
    ax.add_patch(FancyArrowPatch((.82, .69), (.879, .69), arrowstyle="-|>",
                                mutation_scale=6, color=INK, linewidth=.9))
    ax.text(.89, .69, "cross-region", fontsize=7.5, va="center")
    ax.text(.50, -.03, "Region IDs run clockwise within latitude sectors; circles are schematic.",
            fontsize=7.5, color=GREY, ha="center", va="center")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    t, regions, summary = load_and_validate()
    configure()
    # The exact canvas gives a height/width ratio below 0.9 at 6.3-inch width.
    fig = plt.figure(figsize=(6.3, 5.3))
    axa = fig.add_axes([.085, 3.9 / 5.3, .392, 1.1 / 5.3])
    axb = fig.add_axes([.593, 3.9 / 5.3, .375, 1.1 / 5.3])
    plot_stage_counts(axa, summary)
    plot_rates(axb, summary)
    axc = fig.add_axes([.015, .55 / 5.3, .475, 2.85 / 5.3])
    axd = fig.add_axes([.510, .55 / 5.3, .475, 2.85 / 5.3])
    c = plot_network(axc, t, regions, ORDER[0], BLUE, "c", "Wind \u2192 humidity")
    d = plot_network(axd, t, regions, ORDER[1], ORANGE, "d", "Humidity \u2192 cloud")
    legend(fig)
    HERE.mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(HERE / f"{STEM}.{ext}", **kw)
    # Explicit 6.3-inch rendition for checking the journal-width figure.
    fig.savefig(HERE / "preview_6p3in.png", dpi=160)
    plt.close(fig)
    pd.concat([c, d], ignore_index=True).to_csv(HERE / "data/network_pairs_primary.csv", index=False)
    shown = summary.loc[ORDER + ["ALL"]].copy()
    shown["confirmed_replication_rate"] = shown.eval_wb2_replicated / shown.stage2_confirmed
    shown["unconfirmed_replication_rate"] = shown.eval_wb2_negctrl_replicated / shown.eval_wb2_negctrl_n
    shown.to_csv(HERE / "data/panel_ab_summary.csv")
    print("Wrote", HERE / (STEM + ".pdf"))


if __name__ == "__main__":
    main()
