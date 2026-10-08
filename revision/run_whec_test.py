"""Pre-specified test of the wind-humidity eddy convergence (WHEC) index.

The design (revision_outputs/whec_test/design.json) was frozen before any
result below was computed; its SHA-256 is checked. For every target region r
and target variable (humidity, cloud cover) the Stage-2 dense VAR(3) of the
main analysis (264 regional T, H, W, C anomalies) is augmented with lags 1-3
of u_r, v_r, LIN_r and one test series K:
    A  K = WHEC_r (local eddy convergence)
    B  K = WHEC of the region ~164 degrees away in the same latitude band
    C  K = placebo, humidity anomaly taken 365 days earlier
Exact OLS by Frisch-Waugh-Lovell on the shared dense design; Bartlett HAC64.
Frozen gains use the 1979-2018 fit; the restricted model (K dropped) is the
exact restricted 1979-2018 refit, identical for the three families.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy import linalg, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from revision.full_var_tests import build_design, adjust_bh  # noqa: E402
from revision.prepare_inputs import sha256  # noqa: E402
from revision.run_ceres_cloud_substitution import load_era5_cloud, standardize_window, regional  # noqa: E402

OUT = ROOT.parent / "revision_outputs"
EXT = ROOT.parent / "supplementary_data" / "era5_primary_native" / "era5_cds_native_conservative_6h_64x32_850hPa_2023-01-11_2025"
DESIGN_SHA = "0507157c5a8ec0b248a36bca29818af6623ba86b1e19576f57fd743d92beee92"
LAGS = (1, 2, 3)
BW = 64
TARGETS = ("humidity", "cloud_cover")
FAMILIES = ("A_local_whec", "B_remote_whec", "C_placebo")
SERIES = ("u", "v", "lin", "whec", "plac")
EVALS = ("eval_wb2", "eval_cds", "eval_all")


def hac_cov(scores, n_params, bw=BW):
    """Bartlett HAC covariance for coefficients with influence rows `scores` (n x k), factor n/(n-p)."""
    n = len(scores)
    v = scores.T @ scores
    for h in range(1, min(bw, n - 1) + 1):
        g = scores[h:].T @ scores[:-h]
        v += (1 - h / (bw + 1)) * (g + g.T)
    return v * n / (n - n_params)


def mean_test(g):
    """Mean of a series with a HAC64 standard error; one-sided p for a positive mean."""
    m = float(g.mean())
    se = float(np.sqrt(hac_cov((g - m)[:, None], 1)[0, 0])) / len(g)
    return m, se, float(stats.norm.sf(m / se))


def lagged(series):
    t = len(series)
    return np.concatenate([series[3 - lag:t - lag] for lag in LAGS], axis=1)


def band_of(lat):
    a = np.abs(lat)
    return np.where(a < 30, "tropical", np.where(a < 60, "midlatitude", "polar"))


class Augmented:
    """Dense VAR(3) design X0 shared by all outcomes; per-outcome extra columns Z by FWL."""

    def __init__(self, x, y, z):
        self.n, self.p0 = x.shape
        q, r = linalg.qr(x, mode="economic", check_finite=False)
        sv = linalg.svdvals(r, check_finite=False)
        if np.count_nonzero(sv > max(x.shape) * np.finfo(float).eps * sv[0]) != x.shape[1]:
            raise ValueError("Rank-deficient dense design")
        self.cond = float(sv[0] / sv[-1])
        qz, qy = q.T @ z, q.T @ y
        self.zr = z - q @ qz
        self.yr = y - q @ qy
        self.b0 = linalg.solve_triangular(r, qy, check_finite=False)
        self.proj = linalg.solve_triangular(r, qz, check_finite=False)

    def fit(self, cols, o):
        zs = self.zr[:, cols]
        ginv = linalg.inv(zs.T @ zs)
        gamma = ginv @ (zs.T @ self.yr[:, o])
        e = self.yr[:, o] - zs @ gamma
        return zs, ginv, gamma, e

    def k_tests(self, cols, o, k_idx):
        zs, ginv, gamma, e = self.fit(cols, o)
        p = self.p0 + len(cols)
        rss = float(e @ e)
        scores = (zs @ ginv[:, k_idx]) * e[:, None]
        v = hac_cov(scores, p)
        gk = gamma[k_idx]
        se = np.sqrt(np.diag(v))
        drop = gk ** 2 / np.diag(ginv)[k_idx]
        block_drop = float(gk @ linalg.solve(ginv[np.ix_(k_idx, k_idx)], gk))
        wald = float(gk @ linalg.solve(v, gk))
        return {"effect": gk, "se": se, "p": 2 * stats.norm.sf(np.abs(gk / se)),
                "se_ols": np.sqrt(rss / (self.n - p) * np.diag(ginv)[k_idx]), "partial_r2": drop / (rss + drop),
                "block_partial_r2": block_drop / (rss + block_drop), "wald": wald, "wald_p": float(stats.chi2.sf(wald, len(k_idx))),
                "gamma": gamma, "ginv": ginv}

    def frozen_setup(self, x_test, z_test, y_test):
        """Held-out extra columns residualized with the 1979-2018 projection, and dense-VAR-only errors."""
        return z_test - x_test @ self.proj, y_test - x_test @ self.b0

    @staticmethod
    def frozen_errors(setup, cols, o, gamma, ginv, drop_idx):
        """Held-out errors of the full and the exact restricted (columns drop_idx removed) 1979-2018 fits."""
        zt = setup[0][:, cols]
        base = setup[1][:, o]
        keep = [k for k in range(len(cols)) if k not in drop_idx]
        g_keep = gamma[keep] - ginv[np.ix_(keep, drop_idx)] @ linalg.solve(ginv[np.ix_(drop_idx, drop_idx)], gamma[drop_idx])
        return base - zt @ gamma, base - zt[:, keep] @ g_keep


def own_history_test(y, k):
    """Target on intercept and its lags 1-3 plus one K lag; HAC64 for the K coefficient (n x 1 inputs)."""
    t = len(y)
    base = np.column_stack([np.ones(t - 3)] + [y[3 - lag:t - lag] for lag in LAGS])
    yy, kk = y[3:], k
    q, _ = linalg.qr(base, mode="economic", check_finite=False)
    yr = yy - q @ (q.T @ yy)
    kr = kk - q @ (q.T @ kk)
    beta = float(kr @ yr / (kr @ kr))
    e = yr - kr * beta
    se = float(np.sqrt(hac_cov((kr / (kr @ kr) * e)[:, None], 5)[0, 0]))
    return beta, se, float(2 * stats.norm.sf(abs(beta / se)))


def one_sided_same_sign(p_two, ref, new):
    return np.where(np.sign(ref) == np.sign(new), p_two / 2, 1 - p_two / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT / "whec_test")
    args = ap.parse_args()
    start = time.perf_counter()
    design_path = args.output / "design.json"
    if sha256(design_path) != DESIGN_SHA:
        raise ValueError("design.json differs from the frozen design")

    with np.load(OUT / "inputs" / "region_trainfit.npz") as z:
        data, names, ts, lat = z["data"], z["variable_names"].tolist(), pd.DatetimeIndex(z["timestamps"]), z["lat"]
    with np.load(OUT / "inputs" / "region_trainfit_vectors.npz") as z:
        vec, vnames = z["data"], z["variable_names"].tolist()
    with np.load(OUT / "whec_index" / "whec_regional.npz") as z:
        if not np.array_equal(pd.DatetimeIndex(z["timestamps"]).values, ts.values):
            raise ValueError("WHEC calendar differs")
        series = {"u": vec[:, :, vnames.index("u")], "v": vec[:, :, vnames.index("v")], "lin": z["lin"], "whec": z["whec"], "plac": z["placebo"]}
        remote = z["remote_region"]
    n_reg = data.shape[1]
    band = band_of(lat)
    x, y_all = build_design(data, 3)
    outcomes = [(r, tv) for tv in TARGETS for r in range(n_reg)]
    y = y_all[:, [r * len(names) + names.index(tv) for r, tv in outcomes]]
    o_index = {key: i for i, key in enumerate(outcomes)}
    zall = np.concatenate([lagged(series[s]) for s in SERIES], axis=1)
    offset = {s: i * n_reg * len(LAGS) for i, s in enumerate(SERIES)}

    def cols_for(fam, r):
        c = lambda s, reg: [offset[s] + (lag - 1) * n_reg + reg for lag in LAGS]
        k = {"A_local_whec": c("whec", r), "B_remote_whec": c("whec", int(remote[r])), "C_placebo": c("plac", r)}[fam]
        return c("u", r) + c("v", r) + c("lin", r) + k

    def k_source(fam, r):
        return {"A_local_whec": ("whec", r), "B_remote_whec": ("whec", int(remote[r])), "C_placebo": ("plac", r)}[fam]

    k_idx = [9, 10, 11]
    t_resp = ts[3:]
    periods = {"discovery": t_resp < pd.Timestamp("2019-01-01"),
               "eval_wb2": (t_resp >= pd.Timestamp("2019-01-01")) & (t_resp < pd.Timestamp("2023-01-11")),
               "eval_cds": t_resp >= pd.Timestamp("2023-01-11"),
               "eval_all": t_resp >= pd.Timestamp("2019-01-01")}
    models = {}
    for name, m in periods.items():
        models[name] = Augmented(x[m], y[m], zall[m])
        print(f"{name}: n={m.sum()} cond={models[name].cond:.1f} {time.perf_counter() - start:.0f}s", flush=True)

    # Independent check of the FWL route against a direct least-squares fit of one augmented equation.
    d = models["discovery"]; m = periods["discovery"]
    n_disc = int(m.sum())
    assert np.array_equal(np.flatnonzero(m), np.arange(n_disc))
    cols = cols_for("A_local_whec", 47); o = o_index[(47, "cloud_cover")]
    direct = np.linalg.lstsq(np.column_stack([x[m], zall[m][:, cols]]), y[m][:, o], rcond=None)[0]
    fwl = d.fit(cols, o)[2]
    fwl_check = float(np.max(np.abs(direct[-12:] - fwl)))
    if fwl_check > 1e-8:
        raise ValueError(f"FWL coefficients differ from direct fit by {fwl_check}")

    # Per-lag hypotheses, block tests and frozen gains.
    rows, blocks, gain_series = [], [], {}
    setups = {e: d.frozen_setup(x[periods[e]], zall[periods[e]], y[periods[e]]) for e in EVALS}
    for fam in FAMILIES:
        for r in range(n_reg):
            cols = cols_for(fam, r)
            src, src_reg = k_source(fam, r)
            for tv in TARGETS:
                o = o_index[(r, tv)]
                yt = data[:, r, names.index(tv)]
                rec = {"family": fam, "target_region": r, "target_var": tv, "band": band[r], "target_lat": float(lat[r]),
                       "k_series": src, "k_region": src_reg}
                res = {name: models[name].k_tests(cols, o, k_idx) for name in periods}
                for i, lag in enumerate(LAGS):
                    row = dict(rec, lag=lag)
                    kcol = zall[:, cols[k_idx[i]]][periods["discovery"]]
                    b1, s1, p1 = own_history_test(yt[:n_disc + 3], kcol)
                    row.update(s1_effect=b1, s1_se=s1, s1_p=p1)
                    for name, rr in res.items():
                        pre = "s2" if name == "discovery" else name
                        row.update({f"{pre}_effect": rr["effect"][i], f"{pre}_se": rr["se"][i], f"{pre}_p": rr["p"][i],
                                    f"{pre}_partial_r2": rr["partial_r2"][i]})
                    rows.append(row)
                disc = res["discovery"]
                for name, rr in res.items():
                    blocks.append(dict(rec, period=name, wald=rr["wald"], wald_p=rr["wald_p"], block_partial_r2=rr["block_partial_r2"],
                                       sum_effect=float(rr["effect"].sum())))
                for e in EVALS:
                    e_full, e_red = d.frozen_errors(setups[e], cols, o, disc["gamma"], disc["ginv"], k_idx)
                    g = e_red ** 2 - e_full ** 2
                    mse = float(np.mean(e_full ** 2))
                    mg, se, pg = mean_test(g)
                    blocks.append(dict(rec, period=f"frozen_{e}", block_gain=mg, block_gain_se=se, block_gain_p1=pg,
                                       relative_block_gain=mg / mse, mse_full=mse))
                    gain_series[(fam, tv, e, r)] = g / mse
                    for i, lag in enumerate(LAGS):
                        ef, er = d.frozen_errors(setups[e], cols, o, disc["gamma"], disc["ginv"], [k_idx[i]])
                        gl = er ** 2 - ef ** 2
                        mgl, sel, pgl = mean_test(gl)
                        target_row = rows[-3 + i]
                        target_row.update({f"{e}_gain": mgl, f"{e}_gain_se": sel, f"{e}_gain_p1": pgl, f"{e}_relative_gain": mgl / float(np.mean(ef ** 2))})
        print(f"family {fam} done {time.perf_counter() - start:.0f}s", flush=True)

    table = pd.DataFrame(rows)
    for fam, g in table.groupby("family"):
        idx = g.index
        table.loc[idx, "s1_q"] = adjust_bh(g.s1_p)
        table.loc[idx, "s2_q"] = adjust_bh(g.s2_p)
    table["s1_pass"] = table.s1_q < 0.05
    table["s2_pass"] = table.s1_pass & (table.s2_q < 0.05)
    for e in EVALS:
        table[f"{e}_same_sign"] = np.sign(table[f"{e}_effect"]) == np.sign(table.s2_effect)
        table[f"{e}_p1"] = one_sided_same_sign(table[f"{e}_p"], table.s2_effect, table[f"{e}_effect"])
        table[f"{e}_q_rep"] = 1.0; table[f"{e}_gain_q_rep"] = 1.0; table[f"{e}_negctrl_q"] = 1.0
        for fam, g in table.groupby("family"):
            conf = g[g.s2_pass]; neg = g[~g.s2_pass]
            if len(conf):
                table.loc[conf.index, f"{e}_q_rep"] = adjust_bh(conf[f"{e}_p1"])
                table.loc[conf.index, f"{e}_gain_q_rep"] = adjust_bh(conf[f"{e}_gain_p1"])
            if len(neg):
                table.loc[neg.index, f"{e}_negctrl_q"] = adjust_bh(neg[f"{e}_p1"])
        table[f"{e}_replicated"] = table.s2_pass & (table[f"{e}_q_rep"] < 0.05)
        table[f"{e}_predictive"] = table.s2_pass & (table[f"{e}_gain"] > 0) & (table[f"{e}_gain_q_rep"] < 0.05)
    table.to_csv(args.output / "whec_tests_per_lag.csv", index=False)
    blocks = pd.DataFrame(blocks)
    blocks.to_csv(args.output / "whec_block_tests.csv", index=False)

    summary = []
    for (fam, tv), g in table.groupby(["family", "target_var"]):
        for b, h in [("all", g)] + list(g.groupby("band")):
            s = {"family": fam, "target_var": tv, "band": b, "tested": len(h), "stage1": int(h.s1_pass.sum()), "confirmed": int(h.s2_pass.sum()),
                 "confirmed_positive": int((h.s2_pass & (h.s2_effect > 0)).sum()),
                 "median_partial_r2_confirmed": float(h.loc[h.s2_pass, "s2_partial_r2"].median()) if h.s2_pass.any() else np.nan}
            for e in EVALS:
                c, n = h[h.s2_pass], h[~h.s2_pass]
                s.update({f"{e}_replicated": int(c[f"{e}_replicated"].sum()), f"{e}_predictive": int(c[f"{e}_predictive"].sum()),
                          f"{e}_rate": float(c[f"{e}_replicated"].mean()) if len(c) else np.nan,
                          f"{e}_negctrl_n": len(n), f"{e}_negctrl_replicated": int((n[f"{e}_negctrl_q"] < 0.05).sum())})
            summary.append(s)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output / "whec_summary.csv", index=False)

    # Band endpoints from region-normalized frozen block gains.
    endpoint_rows = []
    for fam in FAMILIES:
        for tv in TARGETS:
            for e in EVALS:
                bseries = {b: np.mean([gain_series[(fam, tv, e, r)] for r in range(n_reg) if band[r] == b], axis=0)
                           for b in ("tropical", "midlatitude", "polar")}
                stats_list = [("midlatitude", bseries["midlatitude"]), ("tropical", bseries["tropical"]), ("polar", bseries["polar"]),
                              ("midlatitude_minus_tropical", bseries["midlatitude"] - bseries["tropical"]),
                              ("midlatitude_minus_polar", bseries["midlatitude"] - bseries["polar"]),
                              ("polar_minus_tropical", bseries["polar"] - bseries["tropical"])]
                for stat, sseries in stats_list:
                    mval, se, p1 = mean_test(sseries)
                    endpoint_rows.append({"family": fam, "target_var": tv, "period": e, "statistic": stat, "estimate": mval,
                                          "se_hac64": se, "z": mval / se, "p_one_sided": p1, "n": len(sseries)})
    endpoints = pd.DataFrame(endpoint_rows)
    endpoints.to_csv(args.output / "whec_band_endpoints.csv", index=False)

    def ep(fam, tv, e, stat):
        r = endpoints[(endpoints.family == fam) & (endpoints.target_var == tv) & (endpoints.period == e) & (endpoints.statistic == stat)].iloc[0]
        return {"estimate": float(r.estimate), "z": float(r.z), "p_one_sided": float(r.p_one_sided), "holds": bool(r.p_one_sided < 0.05)}
    decision = {}
    for tv in TARGETS:
        e1 = {fam: {e: ep(fam, tv, e, "midlatitude") for e in ("eval_wb2", "eval_cds")} for fam in FAMILIES}
        e2 = {e: ep("A_local_whec", tv, e, "midlatitude_minus_tropical") for e in ("eval_wb2", "eval_cds")}
        a1 = all(v["holds"] for v in e1["A_local_whec"].values())
        a2 = all(v["holds"] for v in e2.values())
        ctrl = {f: all(v["holds"] for v in e1[f].values()) for f in ("B_remote_whec", "C_placebo")}
        if not a1:
            verdict = "not_supported"
        elif any(ctrl.values()):
            verdict = "not_specific_to_covariance"
        elif not a2:
            verdict = "not_specific_to_midlatitudes"
        else:
            verdict = "supported"
        decision[tv] = {"role": "primary" if tv == "cloud_cover" else "secondary", "E1": e1, "E2_family_A": e2, "verdict": verdict}

    # Secondary: CERES SYN1deg cloud replacing ERA5 cloud (as predictor and outcome) in window refits.
    with np.load(OUT / "ceres_cloud_substitution" / "ceres_cloud_6h_64x32.npz") as z:
        ceres, ct = z["cloud_percent"].reshape(len(z["timestamps"]), -1).astype(np.float64), z["timestamps"]
    era, et = load_era5_cloud(EXT)
    if not np.array_equal(ct, et):
        raise ValueError("CERES/ERA5 time mismatch")
    params = np.load(OUT / "inputs" / "trainfit_parameters.npz")
    mapping = params["node_to_region"]
    keep = np.isfinite(ceres).all(axis=1)
    win_t = pd.DatetimeIndex(ct)[keep]
    months = win_t.month.to_numpy()
    idx = np.searchsorted(ts.values, win_t.values)
    if not (np.array_equal(ts.values[idx], win_t.values) and np.all(np.diff(idx) == 1)):
        raise ValueError("CERES window not contiguous in the main calendar")
    arrays = {"era5": data[idx].copy(), "ceres": data[idx].copy()}
    arrays["era5"][:, :, 3] = regional(standardize_window(era[keep], months), mapping)
    arrays["ceres"][:, :, 3] = regional(standardize_window(ceres[keep], months), mapping)
    z_win = zall[idx[3:] - 3]
    seg_wb2 = win_t[3:] < pd.Timestamp("2023-01-11")
    disc_effects = table.set_index(["family", "target_region", "target_var", "lag"]).s2_effect
    crow = []
    for product, arr in arrays.items():
        xw, yw_all = build_design(arr, 3)
        yw = yw_all[:, [r * len(names) + names.index("cloud_cover") for r in range(n_reg)]]
        for pname, msk in (("wb2_2017_2023-01-10", seg_wb2), ("cds_2023-01-11_2025", ~seg_wb2)):
            mod = Augmented(xw[msk], yw[msk], z_win[msk])
            for fam in FAMILIES:
                for r in range(n_reg):
                    rr = mod.k_tests(cols_for(fam, r), r, k_idx)
                    for i, lag in enumerate(LAGS):
                        crow.append({"product": product, "period": pname, "family": fam, "target_region": r, "band": band[r], "lag": lag,
                                     "effect": rr["effect"][i], "se": rr["se"][i], "p": rr["p"][i], "wald_p": rr["wald_p"],
                                     "ref_effect": disc_effects[(fam, r, "cloud_cover", lag)]})
            print(f"CERES check {product} {pname} {time.perf_counter() - start:.0f}s", flush=True)
    ct_tab = pd.DataFrame(crow)
    ct_tab["p1"] = one_sided_same_sign(ct_tab.p, ct_tab.ref_effect, ct_tab.effect)
    ct_tab["q"] = 1.0
    for _, g in ct_tab.groupby(["product", "period", "family"]):
        ct_tab.loc[g.index, "q"] = adjust_bh(g.p1)
    ct_tab.to_csv(args.output / "whec_ceres_check.csv", index=False)
    csum = []
    for (product, pname, fam), g in ct_tab.groupby(["product", "period", "family"]):
        for b, h in [("all", g)] + list(g.groupby("band")):
            csum.append({"product": product, "period": pname, "family": fam, "band": b, "tests": len(h),
                         "same_sign_q05": int((h.q < 0.05).sum()), "positive_effects": int((h.effect > 0).sum()),
                         "median_t": float(np.median(h.effect / h.se))})
    pd.DataFrame(csum).to_csv(args.output / "whec_ceres_summary.csv", index=False)

    manifest = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "design_sha256": DESIGN_SHA, "python": platform.python_version(),
                "code_sha256": sha256(Path(__file__)), "inputs": {p: sha256(OUT / p) for p in ("inputs/region_trainfit.npz", "inputs/region_trainfit_vectors.npz", "whec_index/whec_regional.npz", "ceres_cloud_substitution/ceres_cloud_6h_64x32.npz")},
                "periods": {k: int(v.sum()) for k, v in periods.items()}, "condition_numbers": {k: v.cond for k, v in models.items()},
                "fwl_vs_direct_max_abs_difference": fwl_check, "decision": decision, "elapsed_seconds": time.perf_counter() - start}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(json.dumps(decision, indent=1))
    print(summary.to_string())
    print(endpoints[endpoints.period != "eval_all"].to_string())
    print(pd.DataFrame(csum).query("band != 'all' or family == 'A_local_whec'").to_string())


if __name__ == "__main__":
    main()
