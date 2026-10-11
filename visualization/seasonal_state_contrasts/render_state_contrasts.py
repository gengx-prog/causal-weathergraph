"""Seasonal and state contrasts: circular views of the unchanged seasonal/state results.

Run from any working directory: python render_state_contrasts.py [--output-dir PATH]
The two input CSVs are resolved relative to this file. No analysis is refitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Wedge, Circle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TYPES = ["wind_to_humidity", "humidity_to_cloud_cover", "wind_to_cloud_cover"]
TYPE_LABELS = ["W→H", "H→C", "W→C"]
PERIODS = ["discovery", "evaluation"]
RINGS = [("discovery", "nh"), ("discovery", "sh"),
         ("evaluation", "nh"), ("evaluation", "sh")]
CONTRASTS = ["cloud_lag0", "cloud_lag4", "humidity_lag0", "humidity_lag4",
             "NH_local_winter_minus_summer", "SH_local_winter_minus_summer"]
STATE_LABELS = ["Cloud, now", "Cloud, −24 h", "Humid., now", "Humid., −24 h",
                "NH season", "SH season"]
FONT = 10.4  # 8.62 pt when a 7.6-inch original is reproduced at 6.3 inches.
INK = "#202B34"
MUTED = "#51616C"
PERIOD_COLORS = ["#32678B", "#BC7255"]
SLOPE_MAP = colors.LinearSegmentedColormap.from_list(
    "signed_slope", ["#326A93", "#F6F6F3", "#BA674C"])
RATE_MAP = colors.LinearSegmentedColormap.from_list(
    "state_rate", ["#F1F7F5", "#8EBFB6", "#226D69"])


def normalized_luminance(rgba):
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    return float(np.dot(linear, [.2126, .7152, .0722]))


def text_color(rgba):
    # Maximize black/white text contrast, rather than relying on palette roots.
    lum = normalized_luminance(rgba)
    black = (lum + .05) / .05
    white = 1.05 / (lum + .05)
    return "#FFFFFF" if white > black else "#17232B"


def circle_axes(fig, position):
    ax = fig.add_axes(position)
    ax.set_aspect("equal")
    ax.set_xlim(-1.43, 1.43)
    ax.set_ylim(-1.38, 1.38)
    ax.axis("off")
    return ax


def annular_cells(ax, values, angles, inner, width, cmap, norm, formatter, tangent=False):
    """Category and ring geometry is fixed. Only fill color represents value."""
    nsectors, nrings = values.shape
    angle_width = 360 / nsectors
    labels = []
    for sector, angle in enumerate(angles):
        for ring in range(nrings):
            value = float(values[sector, ring])
            rgba = cmap(norm(value))
            radius = inner + (ring + 1) * width
            ax.add_patch(Wedge((0, 0), radius, angle - angle_width / 2,
                               angle + angle_width / 2, width=width,
                               facecolor=rgba, edgecolor="white", linewidth=1.0))
            middle = radius - width / 2
            theta = np.deg2rad(angle)
            label = ax.text(middle * np.cos(theta), middle * np.sin(theta),
                            formatter(value), ha="center", va="center",
                            fontsize=FONT, color=text_color(rgba), fontweight="medium",
                            rotation=(angle % 180 - 90) if tangent else 0,
                            rotation_mode="anchor")
            labels.append(label)
    return labels


def title(fig, x, letter, name):
    fig.text(x, .958, f"({letter}) {name}", ha="left", va="top",
             fontsize=11.2, fontweight="bold", color=INK)


def horizontal_colorbar(fig, cmap, norm, rect, ticks, labels, caption):
    cax = fig.add_axes(rect)
    bar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                          orientation="horizontal", ticks=ticks)
    bar.set_ticklabels(labels)
    bar.solids.set_rasterized(False)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=FONT, length=2.2, pad=2, colors=INK)
    fig.text(rect[0] + rect[2] / 2, .164, caption, ha="center", va="top",
             fontsize=FONT, color=INK)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hpath = HERE / "source_data/hemisphere_edge_type_averages.csv"
    spath = HERE / "source_data/regime_interaction_summary.csv"
    h = pd.read_csv(hpath)
    s = pd.read_csv(spath)
    provenance = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    for entry in provenance["copied_inputs"]:
        assert hashlib.sha256((HERE / entry["relative_copy"]).read_bytes()).hexdigest() == entry["sha256"]

    slope = np.empty((3, 4))
    intervals = []
    for i, edge in enumerate(TYPES):
        for j, (period, hemisphere) in enumerate(RINGS):
            row = h[(h.edge_type == edge) & (h.period == period)].iloc[0]
            slope[i, j] = row[f"mean_delta_{hemisphere}"]
        for period in PERIODS:
            row = h[(h.edge_type == edge) & (h.period == period)].iloc[0]
            intervals.append({"edge_type": edge, "period": period,
                              "estimate": float(row.mean_delta_nh_minus_sh),
                              "low": float(row.ci95_low_hac64), "high": float(row.ci95_high_hac64),
                              "n_response_times": int(row.n_response_times),
                              "candidate_pairs": int(row.candidate_pairs),
                              "candidate_lag_pairs": int(row.candidate_lag_pairs)})
    counts = np.empty((6, 2), dtype=int)
    totals = np.empty((6, 2), dtype=int)
    for i, contrast in enumerate(CONTRASTS):
        for j, period in enumerate(PERIODS):
            selected = s[(s.period == period) & (s.contrast == contrast)]
            counts[i, j] = int(selected.significant.sum())
            totals[i, j] = int(selected.tested.sum())
    rates = counts / totals
    assert np.array_equal(totals[:4], np.full((4, 2), 2574))
    assert np.array_equal(totals[4:], np.full((2, 2), 1287))
    assert np.max(np.abs(slope)) < .016
    assert np.max(rates) < .65

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT,
                         "text.color": INK, "axes.labelcolor": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": True,
                         "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.6, 3.5), facecolor="white")
    title(fig, .018, "a", "Seasonal contrasts")
    title(fig, .374, "b", "NH−SH contrast")
    title(fig, .686, "c", "State interactions")

    # A: 12 exact source estimates; labels use explicit 10^-3 units for legibility.
    axa = circle_axes(fig, [.012, .278, .327, .605])
    angles_a = [90, 210, 330]
    slope_norm = colors.TwoSlopeNorm(vmin=-.016, vcenter=0, vmax=.016)
    annular_cells(axa, slope, angles_a, .285, .20, SLOPE_MAP, slope_norm,
                  lambda value: f"{1000*value:+.2f}".replace("-", "−"), tangent=True)
    axa.add_patch(Circle((0, 0), .285, facecolor="white", edgecolor="none"))
    axa.text(0, 0, "Δβ", ha="center", va="center", fontsize=FONT+1, color=MUTED)
    for (x, y), label in zip([(0, 1.26), (-.78, -1.23), (.78, -1.23)], TYPE_LABELS):
        axa.text(x, y, label,
                 ha="center", va="center", fontsize=FONT, color=INK)
    horizontal_colorbar(fig, SLOPE_MAP, slope_norm, [.045, .243, .267, .031],
                        [-.016, 0, .016], ["−16", "0", "+16"],
                        "Slope contrast (×10⁻³)")
    fig.text(.177, .101, "Inner → outer rings", ha="center", va="top", fontsize=FONT)
    fig.text(.177, .050, "NH·D / SH·D / NH·E / SH·E", ha="center", va="top", fontsize=FONT)

    # B: unchanged signed contrast and original covariance-aware HAC64 bounds.
    axb = fig.add_axes([.411, .315, .236, .480])
    for i, edge in enumerate(TYPES):
        for j, period in enumerate(PERIODS):
            row = next(x for x in intervals if x["edge_type"] == edge and x["period"] == period)
            estimate = row["estimate"]
            axb.errorbar(estimate, 2-i + (.115 if j==0 else -.115),
                         xerr=[[estimate-row["low"]], [row["high"]-estimate]],
                         color=PERIOD_COLORS[j], marker=["o", "s"][j],
                         markersize=4.6, markerfacecolor=[PERIOD_COLORS[0], "white"][j],
                         markeredgewidth=1.2, capsize=2.5, linewidth=1.15, linestyle="none")
    axb.axvline(0, color="#7A858C", linewidth=.8, zorder=0)
    axb.set_ylim(-.5, 2.5)
    axb.set_yticks([2, 1, 0], TYPE_LABELS)
    axb.tick_params(axis="y", length=0, pad=5, labelsize=FONT)
    axb.set_xlim(-.013, .031)
    axb.set_xticks([-.01, 0, .01, .02, .03], ["−.01", "0", ".01", ".02", ".03"])
    axb.tick_params(axis="x", labelsize=FONT, length=2.5, color="#A8B0B5")
    for name in ["top", "right", "left"]:
        axb.spines[name].set_visible(False)
    axb.spines["bottom"].set_color("#A8B0B5")
    axb.set_xlabel("Slope difference", fontsize=FONT, labelpad=6)
    fig.text(.518, .883, "95% calendar-HAC intervals", ha="center", va="top", fontsize=FONT)
    fig.text(.518, .827, "n = 29,157 / 5,104", ha="center", va="top", fontsize=FONT, color=MUTED)
    for j, (letter, years) in enumerate([("D", "1979–2018"), ("E", "2019–2025")]):
        y = .108 - .054*j
        fig.text(.431, y, ["●", "□"][j], color=PERIOD_COLORS[j], fontsize=FONT+2, ha="right", va="top")
        fig.text(.442, y, f"{letter}  {years}", fontsize=FONT, ha="left", va="top")

    # C: fixed geometry; color is the rate, labels are exact integer numerators.
    axc = circle_axes(fig, [.666, .278, .329, .605])
    angles_c = [90, 30, -30, -90, -150, -210]
    rate_norm = colors.Normalize(vmin=0, vmax=.65)
    annular_cells(axc, rates, angles_c, .43, .285, RATE_MAP, rate_norm,
                  lambda value: "")
    for i, angle in enumerate(angles_c):
        theta = np.deg2rad(angle)
        for j in range(2):
            radius = .43+(j+.5)*.285
            color = RATE_MAP(rate_norm(rates[i, j]))
            axc.text(radius*np.cos(theta), radius*np.sin(theta), f"{counts[i,j]}",
                     ha="center", va="center", fontsize=FONT, fontweight="medium", color=text_color(color))
        rotation = angle % 180 - 90
        axc.text(1.175*np.cos(theta), 1.175*np.sin(theta), STATE_LABELS[i],
                 rotation=rotation, rotation_mode="anchor", ha="center", va="center",
                 fontsize=FONT, color=INK)
    axc.add_patch(Circle((0, 0), .43, facecolor="white", edgecolor="none"))
    axc.text(0, 0, "q < 0.05", ha="center", va="center", fontsize=FONT, color=MUTED)
    horizontal_colorbar(fig, RATE_MAP, rate_norm, [.703, .243, .255, .031],
                        [0, .325, .65], ["0", "32.5", "65"], "Share of tests (%)")
    fig.text(.831, .101, "Counts: inner D / outer E", ha="center", va="top", fontsize=FONT)
    fig.text(.831, .050, "State /2,574; season /1,287", ha="center", va="top", fontsize=FONT)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    beyond = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            beyond.append(artist.get_text())
    if beyond:
        raise RuntimeError(f"Text outside canvas: {beyond}")

    stem = args.output_dir / "seasonal_state_contrasts"
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(stem.with_suffix("."+extension), dpi=400, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    records = []
    for i, edge in enumerate(TYPES):
        for j, (period, hemisphere) in enumerate(RINGS):
            records.append({"panel":"a", "row":edge, "period":period, "hemisphere":hemisphere,
                            "value":float(slope[i,j]), "display":f"{1000*slope[i,j]:+.2f}", "display_multiplier":1000})
    for row in intervals:
        records.append({"panel":"b", **row})
    for i, contrast in enumerate(CONTRASTS):
        for j, period in enumerate(PERIODS):
            records.append({"panel":"c", "row":contrast, "period":period,
                            "significant":int(counts[i,j]), "tested":int(totals[i,j]),
                            "rate":float(rates[i,j])})
    pd.DataFrame(records).to_csv(args.output_dir / "plotted_values.csv", index=False)
    audit = {"canvas_inches":[7.6,3.5], "minimum_authored_font_pt":FONT,
             "minimum_font_pt_at_6_3_inches":FONT*6.3/7.6,
             "periods":{"D":"1979–2018", "E":"2019–2025"},
             "a_matrix_shape":[3,4], "a_ring_order_inner_to_outer":RINGS,
             "a_unit":"Signed local winter-minus-summer slope contrast on the fitted regional-anomaly scale",
             "a_display_unit":"10^-3 of the fitted slope; explicit scale cue on colorbar; color binds unscaled full-precision values",
             "a_color_limits":[-.016,.016], "b_interval":"Unchanged pointwise 95% calendar-HAC64",
             "c_matrix_shape":[6,2], "c_color_limits":[0,.65],
             "c_labels":"Exact significant-hypothesis numerators; denominators explicitly shown",
             "uncertainty_moved_or_invented":False, "magnitudes_encoded_by_radius":False,
             "outside_canvas_text":beyond, "records":records,
             "files":{ext:hashlib.sha256(stem.with_suffix('.'+ext).read_bytes()).hexdigest() for ext in ['pdf','svg','png']}}
    (args.output_dir / "value_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Rendered {stem}; {len(records)} source-backed values; no text outside canvas.")


if __name__ == "__main__":
    main()
