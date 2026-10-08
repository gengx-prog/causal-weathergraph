"""Compare analyses using the CDS 5.625-degree route with those using a conservative route.

Only the 2023-01-11--2025 segment differs between the two input sets, so the
discovery and 2019--2023 WB2 results must be identical; this is checked first.
Tables are written to revision_outputs/<suffix>_route_summary/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(r"D:\Paper2\Major Revision\revision_outputs")
SPLICE = np.datetime64("2023-01-11")
SUFFIX = "native"


def check_inputs(old_dir, new_dir):
    with np.load(old_dir / "region_trainfit.npz") as a, np.load(new_dir / "region_trainfit.npz") as b:
        if not np.array_equal(a["timestamps"], b["timestamps"]):
            raise ValueError("Timestamp axes differ")
        before = a["timestamps"] < SPLICE
        diff_before = float(np.max(np.abs(a["data"][before] - b["data"][before])))
        d_after = a["data"][~before] - b["data"][~before]
        names = [str(n) for n in a["variable_names"]]
        result = {"max_abs_difference_before_splice": diff_before,
                  "after_splice_rmse_by_variable": {n: float(np.sqrt(np.mean(d_after[:, :, k] ** 2))) for k, n in enumerate(names)},
                  "after_splice_mean_difference_old_minus_new": {n: float(d_after[:, :, k].mean()) for k, n in enumerate(names)}}
    old_p, new_p = old_dir / "trainfit_parameters.npz", new_dir / "trainfit_parameters.npz"
    if old_p.exists() and new_p.exists():
        with np.load(old_p) as a, np.load(new_p) as b:
            result["trainfit_parameters_identical"] = all(np.array_equal(a[k], b[k]) for k in a.files)
    return result


def three_stage(old, new, label):
    a, b = pd.read_csv(old / "three_stage_summary.csv"), pd.read_csv(new / "three_stage_summary.csv")
    same = ["tested", "stage1_screened", "stage2_confirmed", "eval_wb2_replicated", "eval_wb2_negctrl_replicated"]
    if not a[["edge_type"] + same].equals(b[["edge_type"] + same]):
        raise ValueError(f"{label}: discovery or WB2 results differ although their inputs are identical")
    rows = []
    for (_, ra), (_, rb) in zip(a.iterrows(), b.iterrows()):
        row = {"climatology": label, "edge_type": ra["edge_type"], "confirmed": int(ra["stage2_confirmed"]),
               "eval_wb2_replicated": int(ra["eval_wb2_replicated"]), "negctrl_n": int(ra["eval_wb2_negctrl_n"]),
               "eval_wb2_negctrl_replicated": int(ra["eval_wb2_negctrl_replicated"])}
        for period in ("eval_cds", "eval_all"):
            for col in ("replicated", "predictive", "negctrl_replicated"):
                row[f"{period}_{col}_old"] = int(ra[f"{period}_{col}"])
                row[f"{period}_{col}_new"] = int(rb[f"{period}_{col}"])
            row[f"{period}_replication_rate_old"] = ra[f"{period}_replicated"] / ra["stage2_confirmed"]
            row[f"{period}_replication_rate_new"] = rb[f"{period}_replicated"] / rb["stage2_confirmed"]
            row[f"{period}_negctrl_rate_old"] = ra[f"{period}_negctrl_replicated"] / ra[f"{period}_negctrl_n"]
            row[f"{period}_negctrl_rate_new"] = rb[f"{period}_negctrl_replicated"] / rb[f"{period}_negctrl_n"]
        rows.append(row)
    return pd.DataFrame(rows)


def direction(old, new, label):
    a, b = pd.read_csv(old / "symmetric_direction_asymmetry.csv"), pd.read_csv(new / "symmetric_direction_asymmetry.csv")
    keys = ["period", "forward"]
    m = a.merge(b, on=keys, suffixes=("_old", "_new"))
    m.insert(0, "climatology", label)
    cols = ["forward_only", "reverse_only", "forward_stronger_fraction", "median_log_partial_r2_ratio", "wilcoxon_p_abs_t_difference"]
    return m[["climatology"] + keys + [f"{c}_{s}" for c in cols for s in ("old", "new")]]


def ceres(old, new, clim):
    agree = pd.read_csv(old / f"product_agreement_{clim}.csv").merge(
        pd.read_csv(new / f"product_agreement_{clim}.csv"), on="segment", suffixes=("_old", "_new"))
    agree.insert(0, "climatology", clim)
    keys = ["model", "period", "edge_type"]
    subst = pd.read_csv(old / f"substitution_summary_{clim}.csv").merge(
        pd.read_csv(new / f"substitution_summary_{clim}.csv"), on=keys, suffixes=("_old", "_new"))
    subst.insert(0, "climatology", clim)
    keys = ["model", "period", "product", "forward"]
    dirn = pd.read_csv(old / f"substitution_direction_{clim}.csv").merge(
        pd.read_csv(new / f"substitution_direction_{clim}.csv"), on=keys, suffixes=("_old", "_new"))
    dirn.insert(0, "climatology", clim)
    return agree, subst, dirn


def main():
    global SUFFIX
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suffix", default="native", help="Suffix of the rerun output directories (native or conservative).")
    SUFFIX = ap.parse_args().suffix
    DST = OUT / f"{SUFFIX}_route_summary"
    DST.mkdir(parents=True, exist_ok=True)
    checks = {"monthly": check_inputs(OUT / "inputs", OUT / f"inputs_{SUFFIX}"),
              "diurnal": check_inputs(OUT / "inputs_diurnal", OUT / f"inputs_diurnal_{SUFFIX}")}
    for label, c in checks.items():
        if c["max_abs_difference_before_splice"] != 0.0:
            raise ValueError(f"{label} inputs differ before the splice")
    ts = pd.concat([three_stage(OUT / "three_stage", OUT / f"three_stage_{SUFFIX}", "month"),
                    three_stage(OUT / "three_stage_diurnal", OUT / f"three_stage_diurnal_{SUFFIX}", "month_x_hour")])
    ts.to_csv(DST / "three_stage_route_comparison.csv", index=False)
    dr = pd.concat([direction(OUT / "three_stage", OUT / f"three_stage_{SUFFIX}", "month"),
                    direction(OUT / "three_stage_diurnal", OUT / f"three_stage_diurnal_{SUFFIX}", "month_x_hour")])
    dr.to_csv(DST / "direction_route_comparison.csv", index=False)
    parts = [ceres(OUT / "ceres_cloud_substitution", OUT / f"ceres_cloud_substitution_{SUFFIX}", c) for c in ("month", "month_hour")]
    for i, name in enumerate(["ceres_agreement", "ceres_substitution", "ceres_direction"]):
        pd.concat([p[i] for p in parts]).to_csv(DST / f"{name}_route_comparison.csv", index=False)
    (DST / "input_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(json.dumps(checks, indent=1))
    cols = ["climatology", "edge_type", "confirmed", "eval_wb2_replicated", "eval_cds_replicated_old", "eval_cds_replicated_new",
            "eval_all_replicated_old", "eval_all_replicated_new", "eval_cds_negctrl_replicated_old", "eval_cds_negctrl_replicated_new",
            "eval_all_negctrl_replicated_old", "eval_all_negctrl_replicated_new"]
    print(ts[cols].to_string(index=False))
    print(dr[dr.period.isin(["eval_cds", "eval_all"])].to_string(index=False))
    agree = pd.concat([p[0] for p in parts])
    print(agree[["climatology", "segment", "cell_corr_median_old", "cell_corr_median_new",
                 "regional_corr_median_old", "regional_corr_median_new",
                 "global_area_weighted_bias_pp_old", "global_area_weighted_bias_pp_new"]].to_string(index=False))
    dirn = pd.concat([p[2] for p in parts])
    print(dirn[["climatology", "period", "product", "forward", "forward_sig_old", "forward_sig_new", "reverse_sig_old",
                "reverse_sig_new", "wilcoxon_p_old", "wilcoxon_p_new"]].to_string(index=False))


if __name__ == "__main__":
    main()
