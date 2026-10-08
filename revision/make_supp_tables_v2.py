"""LaTeX bodies for new Supplementary Tables, generated from audited outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "revision_outputs"
TAB = ROOT / "manuscript" / "revision_v2_20261002" / "tables"
ET = {"wind_to_humidity": r"$W\rightarrow H$", "humidity_to_cloud_cover": r"$H\rightarrow C$", "wind_to_cloud_cover": r"$W\rightarrow C$",
      "humidity_to_humidity": r"$H\rightarrow H$", "temperature_to_humidity": r"$T\rightarrow H$", "temperature_to_cloud_cover": r"$T\rightarrow C$",
      "cloud_cover_to_humidity": r"$C\rightarrow H$", "humidity_to_wind": r"$H\rightarrow W$", "cloud_cover_to_wind": r"$C\rightarrow W$"}
VN = {"wind": "W", "humidity": "H", "cloud_cover": "C", "temperature": "T"}
ORDER = ["wind_to_humidity", "humidity_to_cloud_cover", "wind_to_cloud_cover", "humidity_to_humidity", "temperature_to_humidity", "temperature_to_cloud_cover"]


def w(name, rows):
    (TAB / name).write_text("\n".join(rows) + "\n", encoding="utf-8")


def s7_nino():
    d = pd.read_csv(OUT / "external_nino_physical_joint" / "descriptive_summary.csv")
    m = pd.read_csv(OUT / "external_nino_physical_joint" / "matched_model_comparisons.csv")
    rows = []
    for contrast, lab in (("high_minus_middle", "High $-$ middle"), ("low_minus_middle", "Low $-$ middle")):
        for et in ["wind_to_humidity", "humidity_to_cloud_cover", "wind_to_cloud_cover"]:
            g = d[(d.period == "evaluation") & (d.hac_bandwidth_six_hour_rows == 64) & (d.contrast == contrast) & (d.edge_type == et)]
            b = g[g.model == "own_history_same_sample"].iloc[0]
            p = g[g.model == "own_history_plus_physical_tminus4_6"].iloc[0]
            rows.append(f"{lab} & {ET[et]} & {b.median_delta_beta:+.4f} & {b.median_abs_delta_beta:.4f} & {p.median_delta_beta:+.4f} & {p.median_abs_delta_beta:.4f} & {p.median_se_hac:.4f} \\\\")
        mm = m[(m.period == "evaluation") & (m.hac_bandwidth_six_hour_rows == 64) & (m.contrast == contrast)]
        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\quad Sign changes after adding physical controls: {int(mm.sign_changed.sum())} of {len(mm)} contrasts}} \\")
        if contrast == "high_minus_middle":
            rows.append(r"\midrule")
    w("supp_s7_nino.tex", rows)


def s8_bootstrap():
    b = pd.read_csv(OUT / "inference" / "fixed_edge_calendar_bootstrap_summary.csv")
    rows = []
    for (reg, sr, tr, sv, tv, lag), g in b.groupby(["regime", "source_region", "target_region", "source_var", "target_var", "lag"], sort=False):
        cells = []
        for L in (64, 128, 256):
            r = g[g.block_length_steps == L].iloc[0]
            cells.append(f"[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]")
        r = g[g.block_length_steps == 128].iloc[0]
        rows.append(f"{reg.replace('_', ' ')} & {sr}$\\rightarrow${tr} & ${VN[sv]}\\rightarrow {VN[tv]}$ & {lag} & {r.median_coefficient:+.4f} & " + " & ".join(cells) + f" & {r.positive_fraction:.3f} \\\\")
    w("supp_s8_bootstrap.tex", rows)


def s11_diurnal():
    d = pd.read_csv(OUT / "three_stage_diurnal" / "three_stage_summary.csv").set_index("edge_type")
    m = pd.read_csv(OUT / "three_stage" / "three_stage_summary.csv").set_index("edge_type")
    rows = []
    for e in ORDER + ["ALL"]:
        r, q = d.loc[e], m.loc[e]
        if e == "ALL":
            rows.append(r"\midrule")
        lab = ET.get(e, r"\textit{All}")
        rows.append(f"{lab} & {int(r.stage1_screened):,} & {int(r.stage2_confirmed):,} & {int(r.eval_wb2_replicated):,} & {r.eval_wb2_replicated / r.stage2_confirmed:.3f} & "
                    f"{r.eval_wb2_negctrl_replicated / r.eval_wb2_negctrl_n:.3f} & {int(q.eval_wb2_replicated):,} & {q.eval_wb2_replicated / q.stage2_confirmed:.3f} \\\\")
    w("supp_s11_diurnal.tex", rows)


def s12_paths():
    p = pd.read_csv(OUT / "physical_controls_experiments" / "path_summary.csv")
    p = p[p.period == "evaluation_all"]
    stages = [("a_wind_to_humidity", r"$W_i\rightarrow H_h$"), ("b_humidity_to_cloud_given_wind", r"$H_h\rightarrow C_j\mid W_i$"),
              ("direct_wind_to_cloud_given_humidity", r"$W_i\rightarrow C_j\mid H_h$"), ("total_wind_to_cloud", r"$W_i\rightarrow C_j$ total")]
    rows = []
    for st, lab in stages:
        cells = []
        for model in ("baseline", "physical_before_path_source"):
            for stratum in ("strong", "ordinary", "unselected"):
                r = p[(p.model == model) & (p.stage == st) & (p.stratum == stratum)].iloc[0]
                cells.append(f"{int(r.positive_incremental_gain)} ({r.median_delta_r2 * 1e4:.2f})")
        rows.append(lab + " & " + " & ".join(cells) + " \\\\")
    w("supp_s12_paths.tex", rows)


def s13_graph():
    g = pd.read_csv(OUT / "simulations" / "graph_summary.csv")
    g = g[g.regime == "state_0"]
    scen = [("weak_ar", "Weak AR"), ("strong_ar", "Strong AR"), ("common_driver_observed", "Observed driver"),
            ("common_driver_omitted", "Hidden driver"), ("known_external_regime_switching", "Regime switching")]
    meth = [("own_history_ols", "Own-history OLS"), ("own_history_hac64", "Own-history HAC"), ("all_observed_history_ols", "All-history OLS"),
            ("all_observed_history_hac64", "All-history HAC"), ("pcmci_parcorr", "PCMCI")]
    rows = []
    for sc, slab in scen:
        for k, (m, mlab) in enumerate(meth):
            r = g[(g.method == m) & (g.scenario == sc)].iloc[0]
            rows.append(f"{slab if k == 0 else ''} & {mlab} & {r.mean_precision:.3f} & {r.mean_recall:.3f} & {r.mean_f1:.3f} & {r.mean_shd:.2f} & {r.empirical_fdr:.3f} \\\\")
        if sc != scen[-1][0]:
            rows.append(r"\addlinespace")
    w("supp_s13_graph.tex", rows)


def s14_degree():
    rows = []
    for case, lab in (("symmetric_full_topology", "Full record, symmetric support"), ("symmetric_late_vs_early_overlap", "2002--2025 versus 1979--2001 overlap")):
        ex = json.loads((OUT / "graph_overlap_null_round2" / f"{case}_original_support_exact.json").read_text(encoding="utf-8"))
        dg = json.loads((OUT / "graph_overlap_null_round2" / f"{case}_diagnostics.json").read_text(encoding="utf-8"))
        acc = np.mean([c["productive_acceptance_fraction"] for c in dg["chain_records"]])
        rows.append(f"{lab} & {ex['feasible_graphs']} & {dg['expanded_distinct_graphs']} & {100 * dg['expanded_mean_changed_fraction']:.2f} & {100 * acc:.3f} \\\\")
    inv = json.loads((OUT / "graph_overlap_null_round2" / "whc_chain_invariant_proof.json").read_text(encoding="utf-8"))
    tot = {c["case"]: c["total_unique_edge_chains"] for c in inv["cases"]}
    w("supp_s14_degree.tex", rows)
    return tot


def s15_ceres():
    a = pd.read_csv(OUT / "ceres_cloud_substitution" / "product_agreement_month.csv")
    h = pd.read_csv(OUT / "ceres_cloud_substitution" / "product_agreement_month_hour.csv")
    rows = []
    lab = {"2017_2025": "2017--2025", "wb2_2017_2023-01-10": "2017-01-01 to 2023-01-10 (WB2 route)", "cds_2023-01-11_2025": "2023-01-11 to 2025-12-31 (CDS route)"}
    for _, r in a.iterrows():
        hh = h[h.segment == r.segment].iloc[0]
        rows.append(f"{lab[r.segment]} & {int(r.n_times):,} & {r.cell_corr_median:.3f} & {r.cell_corr_tropics_median:.3f} & {r.cell_corr_midlat_median:.3f} & {r.cell_corr_polar_median:.3f} & "
                    f"{r.regional_corr_median:.3f} [{r.regional_corr_min:.2f}, {r.regional_corr_max:.2f}] & {hh.regional_corr_median:.3f} & {r.global_area_weighted_bias_pp:+.2f} \\\\")
    w("supp_s15_ceres.tex", rows)


if __name__ == "__main__":
    s7_nino(); s8_bootstrap(); s11_diurnal(); s12_paths(); s13_graph(); tot = s14_degree(); s15_ceres()
    print("WHC invariant totals:", tot)
    for p in sorted(TAB.glob("supp_*.tex")):
        print("=====", p.name); print(p.read_text(encoding="utf-8")[:1500])
