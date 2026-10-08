"""Replace ERA5 total cloud cover by CERES SYN1deg satellite cloud area (2017-2025).

ERA5 cloud cover is a model diagnostic that depends on the model's humidity and
cloud scheme, so a humidity->cloud dependence could partly be built into the
reanalysis. Here the identical regressions are repeated with an independently
retrieved cloud product. Both cloud products are processed identically on the
common window (cell-wise monthly climatology and scale fitted on 2017-2025,
arithmetic regional means of standardized cells); T, H and W are the frozen
discovery-fitted ERA5 regional anomalies of the main analysis.

Outputs: product agreement (cell and regional anomaly correlations, biases),
and own-history and dense VAR(3) tests for the symmetric cloud candidate set
(H<->C, W<->C; 2,544 tests) with each cloud product, by acquisition segment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from causal_weathergraph.candidate_edges import CandidateEdge  # noqa: E402
from revision.full_var_tests import build_design, fit_tests, attach_specs, adjust_bh  # noqa: E402
from revision.inference import run_graph_discovery  # noqa: E402

OUT = ROOT.parent / "revision_outputs"
DATA = Path(r"D:\Paper2\vipuser\Data")
SEGMENTS = [("weatherbench2_era5_6h_64x32_850hPa_1979_2018", "2017-01-01", "2018-12-31T18"),
            ("weatherbench2_era5_6h_64x32_850hPa_2019_2023-01-10", "2019-01-01", "2023-01-10T18"),
            ("era5_cds_6h_64x32_850hPa_2023-01-11_2025", "2023-01-11", "2025-12-31T18")]
KEY = ["source_region", "target_region", "source_var", "target_var", "lag"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for b in iter(lambda: s.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_era5_cloud(extension_dir=None):
    parts, times = [], []
    for seg, lo, hi in SEGMENTS:
        folder = extension_dir if (extension_dir is not None and seg == SEGMENTS[-1][0]) else DATA / seg
        with xr.open_dataset(folder / "total_cloud_cover.nc") as ds:
            key = next(k for k in ds.data_vars)
            da = ds[key].sel(time=slice(lo, hi)).transpose("time", "latitude", "longitude")
            if da.latitude.values[0] > da.latitude.values[-1]:
                da = da.sortby("latitude")
            parts.append(np.asarray(da.values, dtype=np.float64).reshape(da.shape[0], -1) * 100.0)
            times.append(da.time.values.astype("datetime64[ns]"))
    return np.concatenate(parts), np.concatenate(times)


def standardize_window(x, months, by_hour=None):
    keys = months if by_hour is None else months * 10 + by_hour
    a = np.empty_like(x)
    for k in np.unique(keys):
        m = keys == k
        a[m] = x[m] - np.nanmean(x[m], axis=0)
    sd = np.nanstd(a, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return a / sd


def regional(z, mapping):
    return np.column_stack([z[:, mapping == r].mean(axis=1) for r in range(mapping.max() + 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT / "ceres_cloud_substitution")
    ap.add_argument("--climatology", choices=["month", "month_hour"], default="month")
    ap.add_argument("--extension-dir", type=Path, help="Directory replacing the 2023-01-11--2025 CDS segment, e.g. the native-conservative route.")
    ap.add_argument("--inputs", type=Path, default=OUT / "inputs", help="Directory with region_trainfit.npz and trainfit_parameters.npz.")
    ap.add_argument("--ceres-npz", type=Path, default=OUT / "ceres_cloud_substitution" / "ceres_cloud_6h_64x32.npz")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tag = args.climatology
    start = time.perf_counter()
    with np.load(args.ceres_npz) as z:
        ceres, ct = z["cloud_percent"].reshape(len(z["timestamps"]), -1).astype(np.float64), z["timestamps"]
    era, et = load_era5_cloud(args.extension_dir)
    if not np.array_equal(ct, et):
        raise ValueError("CERES/ERA5 time mismatch")
    with np.load(args.inputs / "region_trainfit.npz") as z:
        data, names, ts = z["data"], z["variable_names"].tolist(), pd.DatetimeIndex(z["timestamps"])
    params = np.load(args.inputs / "trainfit_parameters.npz")
    mapping = params["node_to_region"]
    grid_lat = np.repeat(params["grid_lat"], len(params["grid_lon"]))
    # Drop the first window time (no preceding CERES hour box).
    keep = np.isfinite(ceres).all(axis=1)
    if keep.sum() != len(keep) - 1 or keep[0]:
        raise ValueError("Unexpected CERES missingness pattern")
    win_t = pd.DatetimeIndex(ct)
    months = win_t.month.to_numpy()
    hours = win_t.hour.to_numpy() if tag == "month_hour" else None
    z_e = standardize_window(era[keep], months[keep], None if hours is None else hours[keep])
    z_s = standardize_window(ceres[keep], months[keep], None if hours is None else hours[keep])
    t_keep = win_t[keep]
    seg_wb2 = t_keep < pd.Timestamp("2023-01-11")

    # Product agreement at grid and regional scale.
    agree = []
    area = np.cos(np.radians(grid_lat))
    for seg_name, m in (("2017_2025", np.ones(len(t_keep), bool)), ("wb2_2017_2023-01-10", seg_wb2), ("cds_2023-01-11_2025", ~seg_wb2)):
        a, b = z_e[m], z_s[m]
        r_cell = np.array([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(a.shape[1])])
        ra, rb = regional(a, mapping), regional(b, mapping)
        r_reg = np.array([np.corrcoef(ra[:, k], rb[:, k])[0, 1] for k in range(ra.shape[1])])
        bias = ceres[keep][m] - era[keep][m]
        agree.append({"segment": seg_name, "n_times": int(m.sum()),
                      "cell_corr_median": float(np.median(r_cell)), "cell_corr_area_weighted_mean": float(np.sum(r_cell * area) / area.sum()),
                      "cell_corr_tropics_median": float(np.median(r_cell[np.abs(grid_lat) < 23.5])),
                      "cell_corr_midlat_median": float(np.median(r_cell[(np.abs(grid_lat) >= 23.5) & (np.abs(grid_lat) < 60)])),
                      "cell_corr_polar_median": float(np.median(r_cell[np.abs(grid_lat) >= 60])),
                      "regional_corr_median": float(np.median(r_reg)), "regional_corr_min": float(r_reg.min()), "regional_corr_max": float(r_reg.max()),
                      "global_area_weighted_bias_pp": float(np.sum(bias.mean(axis=0) * area) / area.sum())})
    pd.DataFrame(agree).to_csv(args.output / f"product_agreement_{tag}.csv", index=False)

    # Analysis arrays over the kept window: T,H,W from discovery-fitted ERA5; C from each product.
    idx = np.searchsorted(ts.values, t_keep.values)
    if not np.array_equal(ts.values[idx], t_keep.values):
        raise ValueError("Window not aligned with main calendar")
    base = data[idx]
    arrays = {"era5": base.copy(), "ceres": base.copy()}
    arrays["era5"][:, :, 3] = regional(z_e, mapping)
    arrays["ceres"][:, :, 3] = regional(z_s, mapping)
    sym = pd.read_csv(OUT / "graph_nulls" / "candidates_symmetric_core.csv")
    sym = sym[(sym.source_var == "cloud_cover") | (sym.target_var == "cloud_cover")].reset_index(drop=True)
    cand_objs = [CandidateEdge(int(r.source_region), int(r.target_region), r.source_var, r.target_var, r.edge_type, float(r.distance_km)) for r in sym.itertuples()]
    tests = attach_specs(pd.concat([sym.assign(lag=l) for l in (1, 2, 3)], ignore_index=True), names, 66)
    periods_cal = {"2017_2025": np.ones(len(t_keep), bool), "wb2": seg_wb2, "cds": ~seg_wb2}
    results = []
    for product, arr in arrays.items():
        x, y = build_design(arr, 3)
        for pname, cal in periods_cal.items():
            m = cal[3:]
            res, fit = fit_tests(x[m], y[m], tests)
            res["q_hac64_global"] = adjust_bh(res.p_hac64)
            res = res.assign(product=product, model="dense_var3", period=pname)
            results.append(res)
            own = run_graph_discovery(arr, names, cand_objs, {pname: cal}, bandwidths=(64,))
            own = own[KEY + ["edge_type", "effect", "se_hac64", "p_hac64", "q_hac64_global", "partial_r2"]].assign(product=product, model="own_history", period=pname)
            results.append(own)
            print(f"{product} {pname} done {time.perf_counter()-start:.0f}s", flush=True)
    allres = pd.concat(results, ignore_index=True)
    allres.to_csv(args.output / f"substitution_tests_{tag}.csv", index=False)

    rows, pair_rows = [], []
    for (model, pname), g in allres.groupby(["model", "period"]):
        e = g[g["product"] == "era5"].set_index(KEY)
        s = g[g["product"] == "ceres"].set_index(KEY)
        j = e.join(s, lsuffix="_era5", rsuffix="_ceres")
        for et, h in j.groupby("edge_type_era5"):
            te, tc = h.effect_era5 / h.se_hac64_era5, h.effect_ceres / h.se_hac64_ceres
            both = (h.q_hac64_global_era5 < .05) & (h.q_hac64_global_ceres < .05)
            rows.append({"model": model, "period": pname, "edge_type": et, "tested": len(h),
                         "sig_era5": int((h.q_hac64_global_era5 < .05).sum()), "sig_ceres": int((h.q_hac64_global_ceres < .05).sum()),
                         "sig_both": int(both.sum()), "same_sign_when_both": int((np.sign(h.effect_era5) == np.sign(h.effect_ceres))[both].sum()),
                         "t_pearson": float(np.corrcoef(te, tc)[0, 1]), "t_spearman": float(stats.spearmanr(te, tc).statistic),
                         "median_abs_effect_sig_ceres": float(h.loc[h.q_hac64_global_ceres < .05, "effect_ceres"].abs().median()) if (h.q_hac64_global_ceres < .05).any() else np.nan})
        for product in ("era5", "ceres"):
            k = g[g["product"] == product].set_index(KEY)
            for a, b in (("humidity", "cloud_cover"), ("wind", "cloud_cover")):
                f = g[(g["product"] == product) & (g.source_var == a) & (g.target_var == b)]
                fa, ra = [], []
                for r in f.itertuples():
                    rk = (r.target_region, r.source_region, b, a, r.lag)
                    if rk in k.index:
                        rv = k.loc[rk]
                        fa.append((abs(r.effect) / r.se_hac64, r.q_hac64_global < .05, r.partial_r2))
                        ra.append((abs(rv.effect) / rv.se_hac64, rv.q_hac64_global < .05, rv.partial_r2))
                fa, ra = np.array(fa, dtype=float), np.array(ra, dtype=float)
                pair_rows.append({"model": model, "period": pname, "product": product, "forward": f"{a}->{b}", "pairs": len(fa),
                                  "forward_sig": int(fa[:, 1].sum()), "reverse_sig": int(ra[:, 1].sum()),
                                  "forward_only": int(((fa[:, 1] == 1) & (ra[:, 1] == 0)).sum()), "reverse_only": int(((fa[:, 1] == 0) & (ra[:, 1] == 1)).sum()),
                                  "forward_stronger_fraction": float((fa[:, 0] > ra[:, 0]).mean()),
                                  "median_log_partial_r2_ratio": float(np.median(np.log(fa[:, 2] / ra[:, 2]))),
                                  "wilcoxon_p": float(stats.wilcoxon(fa[:, 0] - ra[:, 0]).pvalue)})
    pd.DataFrame(rows).to_csv(args.output / f"substitution_summary_{tag}.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.output / f"substitution_direction_{tag}.csv", index=False)
    manifest = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "climatology": tag, "n_window_times": int(len(t_keep)),
                "first": str(t_keep[0]), "last": str(t_keep[-1]), "n_wb2_times": int(seg_wb2.sum()), "n_cds_times": int((~seg_wb2).sum()),
                "ceres_input_sha256": sha256(args.ceres_npz), "inputs": str(args.inputs), "extension_dir": str(args.extension_dir) if args.extension_dir else None, "code_sha256": sha256(Path(__file__)),
                "elapsed_seconds": time.perf_counter() - start,
                "notes": ["CERES hour boxes centred on each ERA5 analysis time; product and temporal-operator differences remain.",
                          "Both cloud products standardized identically on the common 2017-2025 window; sources are discovery-fitted ERA5 anomalies.",
                          "Agreement with an independent cloud retrieval does not identify intervention effects."]}
    (args.output / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(pd.DataFrame(agree).to_string()); print(pd.DataFrame(rows).to_string()); print(pd.DataFrame(pair_rows).to_string())


if __name__ == "__main__":
    main()
