"""Result figures for the revised manuscript (HiSTGNN-like house style).

White panels boxed by four black spines, no gridlines, serif "(a) ..." captions
under panels, CVD-checked Okabe-Ito colours with marker coding, framed legends.
Every plotted number is read from audited CSV/NPZ outputs in revision_outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT.parent / "revision_outputs"
# 2023-01-11--2025 segment: native CDS fields with the WeatherBench 2 conservative regridding.
EXT = ROOT.parent / "supplementary_data/era5_primary_native/era5_cds_native_conservative_6h_64x32_850hPa_2023-01-11_2025"
FIG = OUT / "paper_figures_20261002"
FIG.mkdir(parents=True, exist_ok=True)

C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#8250C4"]
M = ["o", "s", "D", "^", "v", "*", "P"]
RAMP = ["#dae3ec", "#b9cfe4", "#7d9cc4", "#3f5f8f"]
ET = {"wind_to_humidity": "W→H", "humidity_to_cloud_cover": "H→C", "wind_to_cloud_cover": "W→C",
      "humidity_to_humidity": "H→H", "temperature_to_humidity": "T→H", "temperature_to_cloud_cover": "T→C"}
ORDER = ["wind_to_humidity", "humidity_to_cloud_cover", "wind_to_cloud_cover", "humidity_to_humidity", "temperature_to_humidity", "temperature_to_cloud_cover"]


def setup():
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": 7.2, "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "0.6",
        "legend.fancybox": False, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
        "axes.unicode_minus": False})


def style(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black"); s.set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(length=3.0, width=0.8, color="black")


def caption(ax, tag, text, y=-0.30):
    ax.text(0.5, y, f"({tag}) {text}", transform=ax.transAxes, ha="center", va="top", fontsize=9.2, family="serif", clip_on=False)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------- Figure 2
def fig_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.45), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 1.3]})
    ax = axes[0]
    lags = [1, 4, 12, 28]
    for k, (d, lab) in enumerate((("three_stage", "Monthly clim."), ("three_stage_diurnal", "Month × hour"))):
        s = pd.read_csv(OUT / d / "stage1_residual_acf_summary.csv", index_col=0)
        v = [s.loc["50%", f"residual_acf_{l}"] for l in lags]
        ax.plot(range(4), v, color=C[k], marker=M[k], ms=5, lw=1.4, label=lab)
    ax.axhline(0, color="0.55", lw=0.7)
    ax.set_xticks(range(4)); ax.set_xticklabels(["6", "24", "72", "168"])
    ax.set_xlabel("Calendar lag (h)"); ax.set_ylabel("Median residual autocorrelation")
    ax.set_ylim(-0.03, 0.40); ax.legend(loc="upper right", fontsize=6.6, ncol=1); style(ax)
    caption(ax, "a", "Screening residuals")

    ax = axes[1]
    s = pd.read_csv(OUT / "simulations_post_diagnostic" / "all_null_summary.csv")
    s = s[(s.regime == "all") & (s.estimand == "regression_null_calibration")]
    combos = [("own_history_ols", "ar1_homoskedastic", "OLS F, AR(1)"), ("own_history_hac64", "ar1_homoskedastic", "HAC64, AR(1)"),
              ("own_history_ols", "ar4_heteroskedastic", "OLS F, AR(4)-het."), ("own_history_hac64", "ar4_heteroskedastic", "HAC64, AR(4)-het.")]
    for k, (meth, sc, lab) in enumerate(combos):
        g = s[(s.method == meth) & (s.scenario == sc)].sort_values("sample_length").drop_duplicates("sample_length")
        x = np.log2(g.sample_length.to_numpy())
        ax.errorbar(x + (k - 1.5) * 0.06, g.empirical_fdr, yerr=[g.empirical_fdr - g.fdr_mc_ci95_low, g.fdr_mc_ci95_high - g.empirical_fdr],
                    color=C[[0, 0, 5, 5][k]], marker=M[k], ms=4.5, lw=1.2, capsize=2, ls="-" if k % 2 else "--", label=lab)
    ax.axhline(0.05, color="black", lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks([11, 13, 14]); ax.set_xticklabels(["2,048", "8,192", "16,384"])
    ax.set_xlabel("Series length N"); ax.set_ylabel("Empirical FDR (all-null)"); ax.set_ylim(0, 1.0)
    ax.legend(loc="center right", fontsize=6.4); style(ax)
    caption(ax, "b", "Known-null calibration")

    ax = axes[2]
    g = pd.read_csv(OUT / "simulations" / "graph_summary.csv")
    g = g[g.regime == "state_0"]
    scen = [("weak_ar", "Weak\nAR"), ("strong_ar", "Strong\nAR"), ("common_driver_observed", "Obs.\ndriver"),
            ("common_driver_omitted", "Hid.\ndriver"), ("known_external_regime_switching", "Regime\nswitch")]
    meths = [("own_history_ols", "Own-history"), ("all_observed_history_ols", "All histories"), ("pcmci_parcorr", "PCMCI")]
    w = 0.26
    for k, (meth, lab) in enumerate(meths):
        vals = [g[(g.method == meth) & (g.scenario == sc)].mean_f1.iloc[0] for sc, _ in scen]
        ax.bar(np.arange(5) + (k - 1) * w, vals, width=w * 0.92, color=[RAMP[1], C[0], C[2]][k], edgecolor="none", label=lab)
    ax.set_xticks(range(5)); ax.set_xticklabels([l for _, l in scen], fontsize=6.3)
    ax.set_ylabel("Mean F1 (50 runs)"); ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper center", ncol=3, fontsize=6.3, columnspacing=0.8, handlelength=1.2); style(ax)
    caption(ax, "c", "Known-structure recovery")
    save(fig, "fig02_calibration")


# ---------------------------------------------------------------- Figure 3
def draw_world(ax, ylim=(-104, 88)):
    for lon in np.arange(-180, 181, 60):
        ax.axvline(lon, color="#e2e2e2", lw=0.5, zorder=0)
    for lat in (-60, -30, 0, 30, 60):
        ax.axhline(lat, color="#e2e2e2", lw=0.5, zorder=0)
    ax.set_xlim(-185, 185); ax.set_ylim(*ylim)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180]); ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)"); style(ax)


def edge_map(ax, t, edge_type, color, lat, lon, label):
    draw_world(ax)
    ax.scatter(lon, lat, s=7, c="#555555", zorder=3)
    sub = t[(t.edge_type == edge_type) & t.eval_wb2_replicated]
    agg = sub.groupby(["source_region", "target_region"]).agg(n=("lag", "size"), eff=("s2_effect", lambda v: np.abs(v).max())).reset_index()
    cross, local = agg[agg.source_region != agg.target_region], agg[agg.source_region == agg.target_region]
    for i, r in enumerate(cross.itertuples()):
        x1, y1, x2, y2 = lon[r.source_region], lat[r.source_region], lon[r.target_region], lat[r.target_region]
        if abs(x2 - x1) > 185:
            continue
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6.5, lw=0.6 + 0.35 * r.n,
                                     color=color, alpha=0.75, connectionstyle=f"arc3,rad={0.12 if i % 2 else -0.12}", shrinkA=3, shrinkB=3, zorder=4))
    ax.scatter(lon[local.target_region], lat[local.target_region], s=18 + 14 * local.n, facecolors="none", edgecolors=color, lw=1.0, zorder=5)
    n_tests = int(len(sub)); n_conf = int(((t.edge_type == edge_type) & t.s2_pass).sum())
    ax.text(0.985, 0.035, f"{label}: {n_tests}/{n_conf} confirmed edge–lags replicate", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.6, bbox={"facecolor": "white", "edgecolor": "0.6", "lw": 0.6, "boxstyle": "square,pad=0.25"}, zorder=6)


def fig_three_stage():
    t = pd.read_csv(OUT / "three_stage" / "three_stage_all_candidates.csv")
    s = pd.read_csv(OUT / "three_stage" / "three_stage_summary.csv").set_index("edge_type")
    with np.load(OUT / "inputs" / "region_trainfit.npz") as z:
        lat, lon = z["lat"], ((z["lon"] + 180) % 360) - 180
    fig = plt.figure(figsize=(7.4, 5.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.62, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    stages = [("tested", "Tested"), ("stage1_screened", "Screened (S1)"), ("stage2_confirmed", "Confirmed (S2)"), ("eval_wb2_replicated", "Replicated (S3)")]
    w = 0.2
    for k, (col, lab) in enumerate(stages):
        ax.bar(np.arange(6) + (k - 1.5) * w, [s.loc[e, col] for e in ORDER], width=w * 0.9, color=RAMP[k], edgecolor="none", label=lab)
    ax.set_xticks(range(6)); ax.set_xticklabels([ET[e] for e in ORDER])
    ax.set_ylabel("Edge–lag hypotheses"); ax.set_ylim(0, 820)
    ax.legend(loc="upper right", ncol=2, fontsize=6.3, columnspacing=0.8, handlelength=1.1); style(ax)
    caption(ax, "a", "Hypotheses retained at each stage", y=-0.2)

    ax = fig.add_subplot(gs[0, 1])
    for k, (col_n, col_k, lab, off) in enumerate((("stage2_confirmed", "eval_wb2_replicated", "Confirmed (S2)", -0.12),
                                                  ("eval_wb2_negctrl_n", "eval_wb2_negctrl_replicated", "Not confirmed (control)", 0.12))):
        for i, e in enumerate(ORDER + ["ALL"]):
            n, kk = s.loc[e, col_n], s.loc[e, col_k]
            lo, hi = wilson(kk, n)
            ax.errorbar(i + off, kk / n, yerr=[[kk / n - lo], [hi - kk / n]], color=C[[0, 5][k]], marker=M[k], ms=4.5, capsize=2, lw=1.0,
                        label=lab if i == 0 else None)
    ax.axvline(5.5, color="0.6", lw=0.7)
    ax.set_xticks(range(7)); ax.set_xticklabels([ET[e] for e in ORDER] + ["All"])
    ax.set_ylabel("Replication rate, 2019–2023"); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=6.6); style(ax)
    caption(ax, "b", "Held-out replication versus control", y=-0.2)

    ax = fig.add_subplot(gs[1, 0]); edge_map(ax, t, "wind_to_humidity", C[0], lat, lon, "W→H"); caption(ax, "c", "Replicated wind→humidity edges", y=-0.24)
    ax = fig.add_subplot(gs[1, 1]); edge_map(ax, t, "humidity_to_cloud_cover", C[5], lat, lon, "H→C"); caption(ax, "d", "Replicated humidity→cloud edges", y=-0.24)
    save(fig, "fig03_three_stage")


# ---------------------------------------------------------------- Figure 4
def fig_direction_nulls():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), gridspec_kw={"wspace": 0.4, "width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    era = pd.read_csv(OUT / "three_stage" / "symmetric_direction_asymmetry.csv")
    cer = pd.read_csv(OUT / "ceres_cloud_substitution" / "substitution_direction_month.csv")
    rows = [("humidity->cloud_cover", "H→C vs C→H"), ("wind->humidity", "W→H vs H→W"), ("wind->cloud_cover", "W→C vs C→W")]
    series = [("discovery", "ERA5 1979–2018"), ("eval_wb2", "ERA5 2019–2023"), ("eval_cds", "ERA5 2023–2025"), ("ceres", "CERES cloud 2017–2023")]
    for k, (per, lab) in enumerate(series):
        for i, (fwd, _) in enumerate(rows):
            if per == "ceres":
                g = cer[(cer.model == "dense_var3") & (cer.period == "wb2") & (cer["product"] == "ceres") & (cer.forward == fwd)]
            else:
                g = era[(era.period == per) & (era.forward == fwd)]
            if g.empty:
                continue
            f, n = float(g.forward_stronger_fraction.iloc[0]), int(g.pairs.iloc[0])
            lo, hi = wilson(f * n, n)
            ax.errorbar(f, i + (k - 1.5) * 0.17, xerr=[[f - lo], [hi - f]], color=C[k if k < 3 else 5], marker=M[k], ms=4.5, capsize=1.8, lw=1.0,
                        label=lab if i == 0 or (per == "ceres" and i == 0) else None, ls="none")
    ax.axvline(0.5, color="black", lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(range(3)); ax.set_yticklabels([r for _, r in rows]); ax.set_ylim(3.35, -0.45)
    ax.set_xlabel("Share of pairs with larger forward |t|"); ax.set_xlim(0.38, 0.66)
    ax.legend(loc="lower center", ncol=2, fontsize=5.6, handlelength=1.0, columnspacing=0.6); style(ax)
    caption(ax, "a", "Direction under full conditioning")

    s = pd.read_csv(OUT / "graph_nulls" / "candidate_null_summary.csv")
    s = s[(s.family == "legacy858") & (s.correction == "q_hac64_global") & (s.edge_type == "ALL")]
    e = pd.read_csv(OUT / "graph_nulls" / "candidate_null_enrichment.csv")
    e = e[(e.family == "legacy858") & (e.correction == "q_hac64_global") & (e.edge_type == "ALL")].set_index("metric")
    for ax, metric, xl, tag, cap in ((axes[1], "significant_fraction", "Fraction of tests with q < 0.05", "b", "Edge-level reference"),
                                     (axes[2], "whc_lag_resolved_chains", "W→H→C chain count", "c", "Chain-level reference")):
        ax.hist(s[metric], bins=14, color=RAMP[1], edgecolor="white", lw=0.6, label="100 matched draws")
        obs = e.loc[metric, "observed"]
        ax.axvline(obs, color=C[5], lw=1.6, label="Physical candidates")
        ax.set_xlabel(xl); ax.set_ylabel("Draws"); style(ax)
        ax.legend(loc="upper left" if metric == "whc_lag_resolved_chains" else "upper center", fontsize=6.2)
        caption(ax, tag, cap)
    axes[1].set_xlim(0.79, 0.86)
    save(fig, "fig04_direction_nulls")


# ---------------------------------------------------------------- Figure 5
def fig_regimes():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1, 1.1]})
    h = pd.read_csv(OUT / "hemisphere_contrasts" / "hemisphere_edge_type_averages.csv")
    types = ["wind_to_humidity", "humidity_to_cloud_cover", "wind_to_cloud_cover"]
    ax = axes[0]
    for k, per in enumerate(("discovery", "evaluation")):
        for i, et in enumerate(types):
            r = h[(h.period == per) & (h.edge_type == et)].iloc[0]
            ax.errorbar(r.mean_delta_nh, i + (k - 0.5) * 0.25, color=C[0], marker=M[k], ms=4.5, ls="none", label=f"NH, {per}" if i == 0 else None,
                        mfc=C[0] if k == 0 else "white")
            ax.errorbar(r.mean_delta_sh, i + (k - 0.5) * 0.25, color=C[5], marker=M[k], ms=4.5, ls="none", label=f"SH, {per}" if i == 0 else None,
                        mfc=C[5] if k == 0 else "white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(3)); ax.set_yticklabels([ET[t] for t in types]); ax.set_ylim(3.6, -0.5)
    ax.set_xlabel("Local winter − summer slope"); ax.legend(loc="lower center", ncol=2, fontsize=5.6, handlelength=0.9, columnspacing=0.5); style(ax)
    caption(ax, "a", "Seasonal modulation by hemisphere", y=-0.42)

    ax = axes[1]
    for k, per in enumerate(("discovery", "evaluation")):
        for i, et in enumerate(types):
            r = h[(h.period == per) & (h.edge_type == et)].iloc[0]
            ax.errorbar(r.mean_delta_nh_minus_sh, i + (k - 0.5) * 0.25, xerr=[[r.mean_delta_nh_minus_sh - r.ci95_low_hac64], [r.ci95_high_hac64 - r.mean_delta_nh_minus_sh]],
                        color=C[2], marker=M[k], ms=4.5, capsize=2, lw=1.0, ls="none", mfc=C[2] if k == 0 else "white",
                        label=f"{per.capitalize()} (n = {int(r.n_response_times):,})" if i == 0 else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(3)); ax.set_yticklabels([ET[t] for t in types]); ax.set_ylim(3.6, -0.5)
    ax.set_xlabel("NH − SH contrast (95% CI)"); ax.legend(loc="lower center", fontsize=5.6, handlelength=0.9); style(ax)
    caption(ax, "b", "Mirrored-region difference", y=-0.42)

    ax = axes[2]
    r = pd.read_csv(OUT / "regime_contrasts" / "regime_interaction_summary.csv")
    cats = [("cloud_lag0", "Cloud, now"), ("cloud_lag4", "Cloud, −24 h"), ("humidity_lag0", "Humid., now"),
            ("humidity_lag4", "Humid., −24 h"), ("NH_local_winter_minus_summer", "NH season"), ("SH_local_winter_minus_summer", "SH season")]
    w = 0.38
    for k, per in enumerate(("discovery", "evaluation")):
        vals = []
        for c, _ in cats:
            g = r[(r.period == per) & (r.contrast == c)]
            vals.append(g.significant.sum() / g.tested.sum())
        ax.bar(np.arange(6) + (k - 0.5) * w, vals, width=w * 0.9, color=[RAMP[2], RAMP[3]][k], edgecolor="none",
               label="1979–2018" if k == 0 else "2019–2025")
    ax.set_xticks(range(6)); ax.set_xticklabels([l for _, l in cats], fontsize=6.6, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Share of tests with q < 0.05"); ax.set_ylim(0, 0.75)
    ax.text(0.03, 0.70, "2,574 tests per state\n1,287 per hemisphere", transform=ax.transAxes, va="top", fontsize=6.0)
    ax.legend(loc="upper left", fontsize=6.4); style(ax)
    caption(ax, "c", "Direct state-interaction tests", y=-0.42)
    save(fig, "fig05_regimes")


# ---------------------------------------------------------------- Figure 6
def fig_timing_transport():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"wspace": 0.62})
    ax = axes[0]
    c = pd.read_csv(OUT / "lag_window_sensitivity" / "coefficient_centroids.csv")
    c = c[c.period == "training"]
    windows = [3, 6, 12]
    ax.plot(windows, [(L + 1) / 2 for L in windows], color="black", ls=(0, (4, 3)), lw=1.0, label="Uniform, (L+1)/2")
    for k, (mdl, lab) in enumerate((("own3", "ERA5, own 3 lags"), ("own12", "ERA5, own 12 lags"))):
        v = [c[(c.control_model == mdl) & (c.max_source_lag == L)].absolute_coefficient_centroid.median() for L in windows]
        ax.plot(windows, v, color=C[k], marker=M[k], ms=4.5, lw=1.3, label=lab)
    b = pd.read_csv(OUT / "long_lag_benchmark" / "summary.csv")
    b = b[(b.target_history == 3) & (b.true_lag == 2)].sort_values("max_source_lag")
    ax.plot(b.max_source_lag, b.median_coefficient_centroid, color=C[2], marker="D", ms=4.5, lw=1.3, label="Simulated, true lag 2")
    ax.set_xticks(windows); ax.set_xlabel("Maximum source lag L (6-h steps)"); ax.set_ylabel("Coefficient lag centroid")
    ax.legend(loc="upper left", fontsize=6.1); style(ax)
    caption(ax, "a", "Lag summaries track the window")

    ax = axes[1]
    j = pd.read_csv(OUT / "path_diagnostics" / "joint_window_summary.csv").set_index("window")
    wins = ["1-3", "4-8", "9-12"]
    vals = [j.loc[w_, "median_heldout_delta_r2"] * 1e4 for w_ in wins]
    ax.bar(range(3), vals, color=RAMP[2], width=0.6, edgecolor="none")
    for i, w_ in enumerate(wins):
        ax.text(i, vals[i] + 0.12, f"{int(j.loc[w_, 'positive_predictive_gain'])}/36", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(["6–18 h", "24–48 h", "54–72 h"])
    ax.set_ylabel(r"Median held-out $\Delta R^2$ ($\times10^{-4}$)"); ax.set_ylim(0, max(vals) * 1.35); style(ax)
    caption(ax, "b", "Joint source-lag windows")

    ax = axes[2]
    lab = {("humidity_change", "ridge"): "Humidity\nridge", ("humidity_change", "hgb"): "Humidity\nboosting",
           ("cloud_level", "ridge"): "Cloud\nridge", ("cloud_level", "hgb"): "Cloud\nboosting"}
    for i, key in enumerate(lab):
        e, lo, hi = transport_reduction(*key)
        ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], color=C[0] if key[0] == "humidity_change" else C[5], marker=M[0 if key[1] == "ridge" else 1],
                    ms=5, capsize=2, lw=1.1, ls="none")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(4)); ax.set_yticklabels(list(lab.values()), fontsize=6.8); ax.invert_yaxis()
    ax.set_xlabel("MSE reduction (%)"); style(ax)
    caption(ax, "c", "Moisture-transport features")
    save(fig, "fig06_timing_transport")


def transport_reduction(task, method):
    """Percent reduction of pooled held-out MSE when the transport summaries are added (estimate, 2.5%, 97.5%)."""
    folder = OUT / "transport_consistency_v1" / "model"
    metrics = pd.read_csv(folder / "metrics.csv", dtype={"region_id": str})
    comps = pd.read_csv(folder / "paired_comparisons.csv", dtype={"region_id": str})
    base = metrics[(metrics.task == task) & (metrics.model == f"{method}_samegrid") & (metrics.region_id == "pooled_equal_regions")
                   & (metrics["slice"] == "all") & (metrics.split == "test")]
    r = comps[(comps.task == task) & (comps.model == f"{method}_samegrid_flux") & (comps.baseline == f"{method}_samegrid")
              & (comps.region_id == "pooled_equal_regions") & (comps["slice"] == "all")]
    assert len(base) == 1 and len(r) == 1
    mse, r = float(base.mse.iloc[0]), r.iloc[0]
    return tuple(100 * r[x] / mse for x in ("delta_mse_baseline_minus_model", "exploratory_delta_mse_p025", "exploratory_delta_mse_p975"))


# ---------------------------------------------------------------- Figure 7
def fig_ceres():
    from revision.run_ceres_cloud_substitution import load_era5_cloud, standardize_window, regional
    with np.load(OUT / "ceres_cloud_substitution" / "ceres_cloud_6h_64x32.npz") as z:
        ceres, ct = z["cloud_percent"].reshape(len(z["timestamps"]), -1).astype(float), pd.DatetimeIndex(z["timestamps"])
    era, _ = load_era5_cloud(EXT)
    keep = np.isfinite(ceres).all(axis=1)
    mapping = np.load(OUT / "inputs" / "trainfit_parameters.npz")["node_to_region"]
    months = ct.month.to_numpy()[keep]
    ze, zs = standardize_window(era[keep], months), standardize_window(ceres[keep], months)
    wb2 = ct[keep] < pd.Timestamp("2023-01-11")
    re_, rs = regional(ze[wb2], mapping), regional(zs[wb2], mapping)
    r = np.array([np.corrcoef(re_[:, k], rs[:, k])[0, 1] for k in range(66)])
    pd.DataFrame({"region": range(66), "anomaly_corr_wb2": r}).to_csv(FIG / "fig07_regional_correlation.csv", index=False)
    with np.load(OUT / "inputs" / "region_trainfit.npz") as z:
        lat, lon = z["lat"], ((z["lon"] + 180) % 360) - 180
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.45), gridspec_kw={"wspace": 0.62, "width_ratios": [1.45, 1, 1]})
    ax = axes[0]; draw_world(ax, ylim=(-86, 86))
    sc = ax.scatter(lon, lat, c=r, cmap="Blues", vmin=0.3, vmax=0.95, s=48, edgecolors="0.35", linewidths=0.4, zorder=4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02); cb.set_label("Anomaly r", fontsize=7.0, labelpad=2); cb.ax.tick_params(labelsize=6.8); cb.outline.set_linewidth(0.8)
    caption(ax, "a", "Regional anomaly agreement")
    t = pd.read_csv(OUT / "ceres_cloud_substitution" / "substitution_tests_month.csv")
    t = t[(t.model == "dense_var3") & (t.period == "wb2")]
    K = ["source_region", "target_region", "source_var", "target_var", "lag"]
    a = t[t["product"] == "era5"].set_index(K); b = t[t["product"] == "ceres"].set_index(K)
    jn = a.join(b, lsuffix="_e", rsuffix="_c")
    hc = jn[jn.edge_type_e == "humidity_to_cloud_cover"]
    ax = axes[1]
    te, tc = hc.effect_e / hc.se_hac64_e, hc.effect_c / hc.se_hac64_c
    both = (hc.q_hac64_global_e < .05) & (hc.q_hac64_global_c < .05)
    ax.scatter(te[~both], tc[~both], s=6, color="0.65", lw=0, label="Other tests")
    ax.scatter(te[both], tc[both], s=8, color=C[5], lw=0, label="q < 0.05 with both")
    lim = max(np.abs(te).max(), np.abs(tc).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.7, ls=(0, (4, 3)))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("t, ERA5 cloud target"); ax.set_ylabel("t, CERES cloud target")
    ax.text(0.04, 0.96, f"r = {np.corrcoef(te, tc)[0, 1]:.2f}", transform=ax.transAxes, va="top", fontsize=7.2)
    ax.legend(loc="lower right", fontsize=6.0, markerscale=1.6); style(ax)
    caption(ax, "b", "H→C tests, 2017–2023")
    ax = axes[2]
    s = pd.read_csv(OUT / "ceres_cloud_substitution" / "substitution_summary_month.csv")
    s = s[(s.model == "dense_var3") & (s.period == "wb2")].set_index("edge_type")
    tys = [("humidity_to_cloud_cover", "H→C"), ("cloud_cover_to_humidity", "C→H"), ("wind_to_cloud_cover", "W→C"), ("cloud_cover_to_wind", "C→W")]
    w = 0.38
    for k, (col, lab) in enumerate((("sig_era5", "ERA5 cloud"), ("sig_ceres", "CERES cloud"))):
        ax.bar(np.arange(4) + (k - 0.5) * w, [s.loc[e, col] / s.loc[e, "tested"] for e, _ in tys], width=w * 0.9, color=[RAMP[2], C[5]][k], edgecolor="none", label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in tys]); ax.set_ylabel("Share of 636 tests, q < 0.05"); ax.set_ylim(0, 0.42)
    ax.legend(loc="upper right", fontsize=6.4); style(ax)
    caption(ax, "c", "Same tests, two cloud products")
    save(fig, "fig07_ceres")


if __name__ == "__main__":
    setup()
    which = sys.argv[1:] or ["2", "3", "4", "5", "6", "7"]
    for w_ in which:
        {"2": fig_calibration, "3": fig_three_stage, "4": fig_direction_nulls, "5": fig_regimes, "6": fig_timing_transport, "7": fig_ceres}[w_]()
        print("figure", w_, "done", flush=True)
