"""Generate LaTeX table bodies for the revised manuscript directly from audited CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "revision_outputs"
TAB = ROOT / "manuscript" / "revision_v2_20261002" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
ET = {"wind_to_humidity": r"$W\rightarrow H$", "humidity_to_cloud_cover": r"$H\rightarrow C$", "wind_to_cloud_cover": r"$W\rightarrow C$",
      "humidity_to_humidity": r"$H\rightarrow H$", "temperature_to_humidity": r"$T\rightarrow H$", "temperature_to_cloud_cover": r"$T\rightarrow C$",
      "cloud_cover_to_humidity": r"$C\rightarrow H$", "humidity_to_wind": r"$H\rightarrow W$", "cloud_cover_to_wind": r"$C\rightarrow W$"}
ORDER = ["wind_to_humidity", "humidity_to_cloud_cover", "wind_to_cloud_cover", "humidity_to_humidity", "temperature_to_humidity", "temperature_to_cloud_cover"]


def c(x):
    return f"{int(round(x)):,}"


def f3(x):
    return f"{x:.3f}"


def regime_table():
    a = pd.read_csv(OUT / "inference" / "all_tests_ols_hac.csv")
    rows = []
    for reg, lab in [("all", "All"), ("DJF", "DJF"), ("JJA", "JJA"), ("high_humidity", "High humidity"), ("normal_humidity", "Normal humidity"),
                     ("high_cloud", "High cloud"), ("normal_cloud", "Normal cloud")]:
        g = a[a.regime == reg]
        s = g[g.significant_hac64_within]
        cross = (s.source_region != s.target_region).mean()
        rows.append(f"{lab} & {c(g.n_samples.iloc[0])} & {c(len(g))} & {c(g.significant_ols_within.sum())} & {c(len(s))} & "
                    f"{f3((s.edge_type == 'wind_to_humidity').mean())} & {f3((s.edge_type == 'humidity_to_cloud_cover').mean())} & {f3(cross)} \\\\")
    tot = a
    rows.append(r"\midrule")
    rows.append(f"Total & -- & {c(len(tot))} & {c(tot.significant_ols_within.sum())} & {c(tot.significant_hac64_within.sum())} & -- & -- & -- \\\\")
    (TAB / "tab_regime_screening.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def three_stage_table():
    s = pd.read_csv(OUT / "three_stage" / "three_stage_summary.csv").set_index("edge_type")
    d = pd.read_csv(OUT / "three_stage_diurnal" / "three_stage_summary.csv").set_index("edge_type")
    rows = []
    for e in ORDER + ["ALL"]:
        r = s.loc[e]
        lab = ET.get(e, r"\textit{All}")
        if e == "ALL":
            rows.append(r"\midrule")
        # Both held-out segments: replicated count, replication rate, and the control rate under the same rule.
        rows.append(f"{lab} & {c(r.tested)} & {c(r.stage1_screened)} & {c(r.stage2_confirmed)} & {c(r.eval_wb2_replicated)} & "
                    f"{f3(r.eval_wb2_replicated / r.stage2_confirmed)} & {f3(r.eval_wb2_negctrl_replicated / r.eval_wb2_negctrl_n)} & "
                    f"{c(r.eval_cds_replicated)} & {f3(r.eval_cds_replicated / r.stage2_confirmed)} & "
                    f"{f3(r.eval_cds_negctrl_replicated / r.eval_cds_negctrl_n)} \\\\")
    (TAB / "tab_three_stage.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def direction_table():
    s = pd.read_csv(OUT / "three_stage" / "symmetric_full_var_summary.csv")
    p = pd.read_csv(OUT / "three_stage" / "symmetric_direction_asymmetry.csv")
    pdiu = pd.read_csv(OUT / "three_stage_diurnal" / "symmetric_direction_asymmetry.csv")
    cer = pd.read_csv(OUT / "ceres_cloud_substitution" / "substitution_direction_month.csv")
    pairs = [("humidity_to_cloud_cover", "cloud_cover_to_humidity", "humidity->cloud_cover"),
             ("wind_to_humidity", "humidity_to_wind", "wind->humidity"),
             ("wind_to_cloud_cover", "cloud_cover_to_wind", "wind->cloud_cover")]
    def cnt(per, et):
        return int(s[(s.period == per) & (s.edge_type == et)].q_below_05.iloc[0])
    def fmt(fr, pv):
        return f"{fr:.3f} ({'$<$0.001' if pv < 0.001 else f'{pv:.3f}'})"
    rows = []
    rows.append("Significant, 1979--2018 & " + " & ".join(f"{cnt('discovery', f)} / {cnt('discovery', r)}" for f, r, _ in pairs) + " \\\\")
    rows.append("Significant, 2019--2023 & " + " & ".join(f"{cnt('eval_wb2', f)} / {cnt('eval_wb2', r)}" for f, r, _ in pairs) + " \\\\")
    rows.append("Significant, 2023--2025 & " + " & ".join(f"{cnt('eval_cds', f)} / {cnt('eval_cds', r)}" for f, r, _ in pairs) + " \\\\")
    for lab, tab, per in (("Stronger forward, 1979--2018", p, "discovery"), ("Stronger forward, 2019--2023", p, "eval_wb2"),
                          ("Stronger forward, 2023--2025", p, "eval_cds"), ("Stronger forward, diurnal, 2019--2023", pdiu, "eval_wb2")):
        cells = []
        for _, _, key in pairs:
            r = tab[(tab.period == per) & (tab.forward == key)].iloc[0]
            cells.append(fmt(r.forward_stronger_fraction, r.wilcoxon_p_abs_t_difference))
        rows.append(lab + " & " + " & ".join(cells) + " \\\\")
    for lab, per in (("Stronger forward, CERES, 2017--2023", "wb2"), ("Stronger forward, CERES, 2023--2025", "cds")):
        cells = []
        for _, _, key in pairs:
            r = cer[(cer.model == "dense_var3") & (cer.period == per) & (cer["product"] == "ceres") & (cer.forward == key)]
            cells.append("--" if r.empty else fmt(r.forward_stronger_fraction.iloc[0], r.wilcoxon_p.iloc[0]))
        rows.append(lab + " & " + " & ".join(cells) + " \\\\")
    (TAB / "tab_direction.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def benchmark_table():
    g = pd.read_csv(OUT / "simulations" / "graph_summary.csv")
    g = g[g.regime == "state_0"]
    scen = ["weak_ar", "strong_ar", "common_driver_observed", "common_driver_omitted", "known_external_regime_switching"]
    meth = [("own_history_ols", "Own-history OLS"), ("own_history_hac64", "Own-history HAC"),
            ("all_observed_history_ols", "All-history OLS"), ("all_observed_history_hac64", "All-history HAC"),
            ("pcmci_parcorr", "PCMCI")]
    rows = []
    for m, lab in meth:
        vals = [g[(g.method == m) & (g.scenario == sc)].mean_f1.iloc[0] for sc in scen]
        secs = g[g.method == m].mean_wall_seconds.mean()
        rows.append(f"{lab} & " + " & ".join(f3(v) for v in vals) + f" & {secs:.2f} \\\\")
    (TAB / "tab_benchmark_graph.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    r = pd.read_csv(OUT / "rpcmci" / "summary.csv").set_index("method")
    rows = []
    for m, lab, info in [("pooled_pcmci", "Pooled PCMCI", "none"), ("regime_pcmci_hidden", "Regime-PCMCI", "state count"),
                         ("oracle_own_history_ols", "Own-history OLS", "true masks"), ("oracle_all_observed_history_hac64", "All-history HAC", "true masks"),
                         ("oracle_all_observed_history_ols", "All-history OLS", "true masks"), ("oracle_pcmci", "PCMCI", "true masks")]:
        x = r.loc[m]
        rows.append(f"{lab} & {info} & {f3(x.mean_precision)} & {f3(x.mean_recall)} & {f3(x.mean_f1)} ({x.sd_f1:.3f}) & {x.mean_wall_seconds:.2f} & {x.mean_sampled_peak_rss_mib:.0f} \\\\")
    (TAB / "tab_benchmark_regime.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    k = pd.read_csv(OUT / "castle_response_aligned" / "summary.csv")
    rows = []
    for m, lab in [("castle_pcmci", "CaStLe (official)"), ("local_own_history", "Local own-history OLS"), ("local_multivariable", "Local multivariable OLS"), ("local_pcmci", "Local PCMCI")]:
        cells = []
        for sc in ["homogeneous_left_and_above", "heterogeneous_left_or_right_and_above"]:
            for size in (6, 10):
                x = k[(k.method == m) & (k.scenario == sc) & (k["size"] == size)].iloc[0]
                cells.append(f"{f3(x.mean_f1)}")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    (TAB / "tab_benchmark_castle.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def ceres_table():
    s = pd.read_csv(OUT / "ceres_cloud_substitution" / "substitution_summary_month.csv")
    h = pd.read_csv(OUT / "ceres_cloud_substitution" / "substitution_summary_month_hour.csv")
    rows = []
    for et in ["humidity_to_cloud_cover", "cloud_cover_to_humidity", "wind_to_cloud_cover", "cloud_cover_to_wind"]:
        r = s[(s.model == "dense_var3") & (s.period == "wb2") & (s.edge_type == et)].iloc[0]
        o = s[(s.model == "own_history") & (s.period == "wb2") & (s.edge_type == et)].iloc[0]
        d = h[(h.model == "dense_var3") & (h.period == "wb2") & (h.edge_type == et)].iloc[0]
        rows.append(f"{ET[et]} & {int(o.sig_era5)} / {int(o.sig_ceres)} & {int(r.sig_era5)} / {int(r.sig_ceres)} & {int(r.sig_both)} & "
                    f"{int(r.same_sign_when_both)} & {r.t_pearson:.2f} & {int(d.sig_era5)} / {int(d.sig_ceres)} \\\\")
    (TAB / "tab_ceres.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def sci(x, signed=False):
    """Plain three decimals unless the value is tiny, then a compact power of ten."""
    if x == 0 or abs(x) >= 0.001:
        return f"{x:+.3f}" if signed else f"{x:.3f}"
    m, e = f"{x:.1e}".split("e")
    sign = "+" if signed and x > 0 else ""
    return f"${sign}{m}\\times10^{{{int(e)}}}$"


def route_rows(v, route=None):
    rows = []
    if route is not None:
        v = v[v.route == route]
    for var, lab in [("temperature", "Temperature $T$"), ("humidity", "Specific humidity $q$"), ("wind", "Wind speed $W$"), ("u", "Zonal wind $u$"), ("v", "Meridional wind $v$"), ("cloud_cover", "Total cloud cover $C$")]:
        ph = v[(v.variable == var) & (v.scale == "physical")].iloc[0]
        gs = v[(v.variable == var) & (v.scale == "grid_standardized")].iloc[0]
        rs = v[(v.variable == var) & (v.scale == "regional_standardized")].iloc[0]
        unit = {"temperature": "K", "humidity": r"g\,kg$^{-1}$", "wind": r"m\,s$^{-1}$", "u": r"m\,s$^{-1}$", "v": r"m\,s$^{-1}$", "cloud_cover": "\\%"}[var]
        scale = 1000 if var == "humidity" else (100 if var == "cloud_cover" else 1)
        rows.append(f"{lab} & {sci(ph.mean_difference_cds_minus_wb2 * scale, True)} {unit} & {sci(ph.rmse * scale)} & {gs.pearson_r:.3f} & "
                    f"{sci(rs.mean_difference_cds_minus_wb2, True)} & {sci(rs.rmse)} & {rs.pearson_r:.3f} \\\\")
    return rows


def route_table():
    # Panel A: the CDS 5.625-degree interpolation route of the original continuation.
    v = pd.read_csv(OUT / "source_overlap_comparison" / "variable_route_differences.csv")
    (TAB / "tab_route.tex").write_text("\n".join(route_rows(v)) + "\n", encoding="utf-8")
    # Panel B: native CDS fields with the WB2 conservative remapping, used for 2023-01-11 to 2025-12-31.
    n = pd.read_csv(OUT / "source_overlap_native_route" / "route_comparison.csv")
    (TAB / "tab_route_native.tex").write_text("\n".join(route_rows(n, "cds_native_conservative")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    regime_table(); three_stage_table(); direction_table(); benchmark_table(); ceres_table(); route_table()
    for p in sorted(TAB.glob("*.tex")):
        print("=====", p.name); print(p.read_text(encoding="utf-8"))
