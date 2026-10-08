"""Screen -> confirm -> replicate evaluation of the regional lagged-dependency graph.

Stage 1 (screen, 1979-2018): own-history nested test, HAC64, BH over all 2,574
candidate-lag hypotheses (recall-oriented; values reused from holdout_aligned).
Stage 2 (confirm, 1979-2018): the same hypotheses inside a dense VAR(3) that
conditions on all 264 observed regional variables, HAC64, BH over 2,574.
Stage 3 (replicate, held-out): dense VAR(3) refitted on a held-out segment;
directional (same-sign, one-sided) HAC64 p-values, BH within the confirmed
family; and frozen-coefficient predictive gains from the 1979-2018 fit.

Held-out segments: WB2 2019-01-01..2023-01-10 (same acquisition route as the
discovery period; primary), CDS 2023-01-11..2025-12-31 (different 5.625-degree
route, see source_overlap_comparison), and their union.

A symmetric-direction family (1,272 candidates, W<->H, H<->C, W<->C, 3,816
tests) is evaluated with the same dense VAR in every period.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import psutil
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from causal_weathergraph.candidate_edges import build_candidate_edges  # noqa: E402
from revision.full_var_tests import build_design, fit_tests, frozen_gains, attach_specs, adjust_bh  # noqa: E402
from revision.inference import run_graph_discovery  # noqa: E402
from revision.run_holdout import predictive_losses  # noqa: E402

KEY = ["source_region", "target_region", "source_var", "target_var", "lag"]
OUT = ROOT.parent / "revision_outputs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for b in iter(lambda: s.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def one_sided_same_sign(p_two, ref_effect, new_effect):
    same = np.sign(ref_effect) == np.sign(new_effect)
    return np.where(same, p_two / 2, 1 - p_two / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=OUT / "inputs" / "region_trainfit.npz")
    ap.add_argument("--output", type=Path, default=OUT / "three_stage")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with np.load(args.input) as z:
        data, names = z["data"], z["variable_names"].tolist()
        ts = pd.DatetimeIndex(z["timestamps"]); lat, lon = z["lat"], z["lon"]
    cand_objs = build_candidate_edges(names, lat, lon, {"candidate_k_nearest": 2})
    legacy = pd.DataFrame([c.to_dict() for c in cand_objs])
    symmetric = pd.read_csv(OUT / "graph_nulls" / "candidates_symmetric_core.csv")
    lg = attach_specs(pd.concat([legacy.assign(lag=l) for l in (1, 2, 3)], ignore_index=True), names, 66)
    sy = attach_specs(pd.concat([symmetric.assign(lag=l) for l in (1, 2, 3)], ignore_index=True), names, 66)
    assert len(lg) == 2574 and len(sy) == 3816

    t_resp = ts[3:]
    periods = {
        "discovery": t_resp < pd.Timestamp("2019-01-01"),
        "eval_wb2": (t_resp >= pd.Timestamp("2019-01-01")) & (t_resp < pd.Timestamp("2023-01-11")),
        "eval_cds": t_resp >= pd.Timestamp("2023-01-11"),
        "eval_all": t_resp >= pd.Timestamp("2019-01-01"),
    }
    x, y = build_design(data, 3)
    var_tables, sym_tables, fits, diag = {}, {}, {}, {}
    for name, mask in periods.items():
        rows = np.flatnonzero(mask)
        assert np.all(np.diff(rows) == 1)
        both = pd.concat([lg.assign(family="legacy"), sy.assign(family="symmetric")], ignore_index=True)
        t_fit = time.perf_counter()
        res, fit = fit_tests(x[mask], y[mask], both)
        fit_seconds = time.perf_counter() - t_fit
        fits[name] = fit
        diag[name] = {"n": int(mask.sum()), "first_response": str(t_resp[mask][0]), "last_response": str(t_resp[mask][-1]),
                      "condition_number": fit["condition_number"], "dense_var_fit_and_test_seconds": fit_seconds}
        for fam, store in (("legacy", var_tables), ("symmetric", sym_tables)):
            t = res[res.family == fam].drop(columns="family").reset_index(drop=True)
            t["q_hac64_global"] = adjust_bh(t.p_hac64)
            store[name] = t
        print(f"VAR {name}: n={mask.sum()} elapsed={time.perf_counter()-start:.0f}s", flush=True)

    # Frozen predictive gains of the discovery dense VAR on each held-out segment.
    gains = {}
    for name in ("eval_wb2", "eval_cds", "eval_all"):
        m = periods[name]
        g = frozen_gains(fits["discovery"], x[m], y[m], lg)
        gains[name] = g

    # Own-history (screening model) on each held-out segment, and frozen own-history gains.
    full_mask = np.zeros(len(ts), dtype=bool)
    own_eval, own_gain = {}, {}
    disc_cal = ts < pd.Timestamp("2019-01-01")
    for name, lo, hi in (("eval_wb2", "2019-01-01", "2023-01-11"), ("eval_cds", "2023-01-11", "2026-01-01"), ("eval_all", "2019-01-01", "2026-01-01")):
        cal = (ts >= pd.Timestamp(lo)) & (ts < pd.Timestamp(hi))
        t = run_graph_discovery(data, names, cand_objs, {name: cal}, bandwidths=(64,))
        own_eval[name] = t
        own_gain[name] = predictive_losses(data, names, cand_objs, disc_cal, cal)
        print(f"own-history {name} elapsed={time.perf_counter()-start:.0f}s", flush=True)

    screen = run_graph_discovery(data, names, cand_objs, {"discovery": disc_cal}, bandwidths=(64,))
    acf_cols = [c for c in screen.columns if c.startswith("residual_acf_")]
    screen[KEY + acf_cols].describe().to_csv(args.output / "stage1_residual_acf_summary.csv")
    reference = OUT / "holdout_aligned" / "discovery_edges.csv"
    if args.input.resolve() == (OUT / "inputs" / "region_trainfit.npz").resolve():
        ref = pd.read_csv(reference).merge(screen[KEY + ["effect", "q_hac64_global"]], on=KEY, suffixes=("", "_new"))
        if not (np.allclose(ref.effect, ref.effect_new, atol=1e-12) and np.allclose(ref.q_hac64_global, ref.q_hac64_global_new, atol=1e-12)):
            raise ValueError("Stage-1 screen does not reproduce holdout_aligned discovery results")
    base = screen[KEY + ["edge_type", "distance_km", "effect", "p_hac64", "q_hac64_global", "partial_r2"]].rename(
        columns={"effect": "s1_effect", "p_hac64": "s1_p", "q_hac64_global": "s1_q", "partial_r2": "s1_partial_r2"})
    conf = var_tables["discovery"][KEY + ["effect", "p_hac64", "q_hac64_global", "partial_r2", "se_hac64"]].rename(
        columns={"effect": "s2_effect", "p_hac64": "s2_p", "q_hac64_global": "s2_q", "partial_r2": "s2_partial_r2", "se_hac64": "s2_se"})
    table = base.merge(conf, on=KEY, validate="one_to_one")
    table["s1_pass"] = table.s1_q < 0.05
    table["s2_pass"] = table.s1_pass & (table.s2_q < 0.05)
    for name in ("eval_wb2", "eval_cds", "eval_all"):
        v = var_tables[name][KEY + ["effect", "p_hac64", "partial_r2"]].rename(columns={"effect": f"{name}_effect", "p_hac64": f"{name}_p", "partial_r2": f"{name}_partial_r2"})
        g = gains[name][KEY + ["mse_gain", "mse_gain_se_hac64", "p_gain_one_sided", "relative_gain"]].rename(
            columns={"mse_gain": f"{name}_gain", "mse_gain_se_hac64": f"{name}_gain_se", "p_gain_one_sided": f"{name}_gain_p1", "relative_gain": f"{name}_relative_gain"})
        o = own_eval[name][KEY + ["effect", "p_hac64"]].rename(columns={"effect": f"{name}_own_effect", "p_hac64": f"{name}_own_p"})
        og = own_gain[name][KEY + ["mse_gain", "p_predictive_gain"]].rename(columns={"mse_gain": f"{name}_own_gain", "p_predictive_gain": f"{name}_own_gain_p"})
        table = table.merge(v, on=KEY, validate="one_to_one").merge(g, on=KEY, validate="one_to_one").merge(o, on=KEY, validate="one_to_one").merge(og, on=KEY, validate="one_to_one")
        table[f"{name}_same_sign"] = np.sign(table[f"{name}_effect"]) == np.sign(table.s2_effect)
        table[f"{name}_p1"] = one_sided_same_sign(table[f"{name}_p"], table.s2_effect, table[f"{name}_effect"])
        fam = table.s2_pass
        q = np.ones(len(table)); q[fam.to_numpy()] = adjust_bh(table.loc[fam, f"{name}_p1"])
        table[f"{name}_q_rep"] = q
        qg = np.ones(len(table)); qg[fam.to_numpy()] = adjust_bh(table.loc[fam, f"{name}_gain_p1"])
        table[f"{name}_gain_q_rep"] = qg
        table[f"{name}_replicated"] = fam & (table[f"{name}_q_rep"] < 0.05)
        table[f"{name}_predictive"] = fam & (table[f"{name}_gain"] > 0) & (table[f"{name}_gain_q_rep"] < 0.05)
        # Negative-control family: screened-out and unconfirmed hypotheses, same rule.
        neg = ~table.s2_pass
        qn = np.ones(len(table)); qn[neg.to_numpy()] = adjust_bh(one_sided_same_sign(table.loc[neg, f"{name}_p"], table.loc[neg, "s2_effect"], table.loc[neg, f"{name}_effect"]))
        table[f"{name}_negctrl_q"] = qn
    table.to_csv(args.output / "three_stage_all_candidates.csv", index=False)

    rows = []
    for et, g in [("ALL", table)] + list(table.groupby("edge_type")):
        r = {"edge_type": et, "tested": len(g), "stage1_screened": int(g.s1_pass.sum()), "stage2_confirmed": int(g.s2_pass.sum()),
             "stage2_alone": int((g.s2_q < 0.05).sum()),
             "s1_s2_same_sign": int((np.sign(g.s1_effect) == np.sign(g.s2_effect))[g.s2_pass].sum()),
             "median_abs_s2_effect_confirmed": float(g.loc[g.s2_pass, "s2_effect"].abs().median()) if g.s2_pass.any() else np.nan,
             "median_s2_partial_r2_confirmed": float(g.loc[g.s2_pass, "s2_partial_r2"].median()) if g.s2_pass.any() else np.nan}
        for name in ("eval_wb2", "eval_cds", "eval_all"):
            c = g[g.s2_pass]
            r[f"{name}_same_sign"] = int(c[f"{name}_same_sign"].sum())
            r[f"{name}_replicated"] = int(c[f"{name}_replicated"].sum())
            r[f"{name}_positive_gain"] = int((c[f"{name}_gain"] > 0).sum())
            r[f"{name}_predictive"] = int(c[f"{name}_predictive"].sum())
            r[f"{name}_replicated_and_predictive"] = int((c[f"{name}_replicated"] & c[f"{name}_predictive"]).sum())
            n = g[~g.s2_pass]
            r[f"{name}_negctrl_n"] = len(n)
            r[f"{name}_negctrl_replicated"] = int((n[f"{name}_negctrl_q"] < 0.05).sum())
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "three_stage_summary.csv", index=False)

    srows = []
    for name, t in sym_tables.items():
        t.to_csv(args.output / f"symmetric_full_var_{name}.csv", index=False)
        for et, g in t.groupby("edge_type"):
            srows.append({"period": name, "edge_type": et, "tested": len(g), "q_below_05": int((g.q_hac64_global < 0.05).sum()),
                          "median_abs_effect_sig": float(g.loc[g.q_hac64_global < 0.05, "effect"].abs().median()) if (g.q_hac64_global < 0.05).any() else np.nan,
                          "median_partial_r2_sig": float(g.loc[g.q_hac64_global < 0.05, "partial_r2"].median()) if (g.q_hac64_global < 0.05).any() else np.nan})
    pd.DataFrame(srows).to_csv(args.output / "symmetric_full_var_summary.csv", index=False)
    # Direction asymmetry on matched pairs (same region pair and lag, reversed variables).
    pairs = []
    for name, t in sym_tables.items():
        k = t.set_index(["source_region", "target_region", "source_var", "target_var", "lag"])
        for a, b in (("wind", "humidity"), ("humidity", "cloud_cover"), ("wind", "cloud_cover")):
            f = t[(t.source_var == a) & (t.target_var == b)]
            for _, e in f.iterrows():
                rev_key = (e.target_region, e.source_region, b, a, e.lag)
                if rev_key in k.index:
                    rv = k.loc[rev_key]
                    pairs.append({"period": name, "forward": f"{a}->{b}", "lag": e.lag, "source_region": e.source_region, "target_region": e.target_region,
                                  "forward_abs_t": abs(e.effect) / e.se_hac64, "reverse_abs_t": abs(rv.effect) / rv.se_hac64,
                                  "forward_sig": e.q_hac64_global < 0.05, "reverse_sig": rv.q_hac64_global < 0.05,
                                  "forward_partial_r2": e.partial_r2, "reverse_partial_r2": rv.partial_r2})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(args.output / "symmetric_matched_pairs.csv", index=False)
    prow = []
    for (name, fwd), g in pairs.groupby(["period", "forward"]):
        d = np.log(g.forward_partial_r2 / g.reverse_partial_r2)
        w = stats.wilcoxon(g.forward_abs_t - g.reverse_abs_t)
        prow.append({"period": name, "forward": fwd, "pairs": len(g), "forward_only": int((g.forward_sig & ~g.reverse_sig).sum()),
                     "reverse_only": int((~g.forward_sig & g.reverse_sig).sum()), "both": int((g.forward_sig & g.reverse_sig).sum()),
                     "neither": int((~g.forward_sig & ~g.reverse_sig).sum()), "forward_stronger_fraction": float((g.forward_abs_t > g.reverse_abs_t).mean()),
                     "median_log_partial_r2_ratio": float(d.median()), "wilcoxon_p_abs_t_difference": float(w.pvalue)})
    pd.DataFrame(prow).to_csv(args.output / "symmetric_direction_asymmetry.csv", index=False)

    manifest = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "periods": diag, "input_sha256": sha256(args.input),
                "code_sha256": {p: sha256(ROOT / p) for p in ("revision/run_three_stage.py", "revision/full_var_tests.py", "revision/inference.py", "revision/run_holdout.py")},
                "python": platform.python_version(), "elapsed_seconds": time.perf_counter() - start, "peak_working_set_bytes": int(getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)), "cpu": platform.processor(),
                "rules": {"stage1": "own-history HAC64 BH(2574) q<0.05 on 1979-2018 (holdout_aligned)",
                          "stage2": "stage1 and dense VAR(3) HAC64 BH(2574) q<0.05 on 1979-2018",
                          "stage3_replicated": "stage2 and held-out dense VAR(3) refit: same sign, one-sided HAC64 p, BH within stage-2 family < 0.05",
                          "stage3_predictive": "stage2 and frozen 1979-2018 dense VAR(3) drop-one gain > 0 with one-sided HAC64 p, BH within stage-2 family < 0.05",
                          "negative_control": "same replication rule applied to hypotheses failing stage 2, BH within that family"},
                "interpretation": "Retrospective temporal holdout of previously examined years; conditional linear predictive dependence, not intervention effects."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(summary.to_string())
    print(pd.DataFrame(prow).to_string())
    print(pd.DataFrame(srows).to_string())


if __name__ == "__main__":
    main()
