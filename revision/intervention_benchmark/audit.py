"""Independent arithmetic audit of the bounded intervention benchmark.

This module never imports simulation or estimator helpers.  It reconstructs
finite-volume face fluxes, local equilibrium partition, known assignments,
linear fits, HC3 covariance, oracle scores, and reported summary statistics.
HGB fitted objects are used only to verify their stored predictions; this is
not a second independent HGB fit or a proof of atmospheric model validity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

import joblib
import numpy as np
import pandas as pd
from scipy.special import stdtrit
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


DEFAULT_OUT = Path(__file__).resolve().parents[3] / "revision_outputs/intervention_benchmark_v1"
Z975 = 1.959963984540054


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def close(actual, expected, message, atol=1e-9, rtol=2e-8):
    aa, bb = np.asarray(actual), np.asarray(expected)
    require(aa.shape == bb.shape, message + ": incompatible shape")
    if not np.allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=True):
        difference = np.abs(aa - bb)
        finite = difference[np.isfinite(difference)]
        maximum = float(np.max(finite)) if finite.size else float("nan")
        raise AssertionError(message + ": maximum finite absolute difference " + str(maximum))


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def saturated_ratio(temperature):
    # Independent log-vapour-pressure expression, fixed reference pressure.
    latent_over_rv = 2500000.0 / 461.0
    es = np.exp(np.log(611.2) + latent_over_rv / 273.15 - latent_over_rv / temperature)
    require(np.all(es < 85000.0), "Saturation formula singular at es>=reference pressure")
    return 0.622 / (85000.0 / es - 1.0)


def equilibrium_partition(vapour, liquid, temperature):
    # Equivalent thermodynamic equilibrium derived from total water, rather
    # than reproducing production's signed condensation/evaporation formula.
    water = vapour + liquid
    vapour_new = np.minimum(water, saturated_ratio(temperature))
    return vapour_new, water - vapour_new


def face_flux_step(field, speed, dt, dx):
    # Flux through the right face of each cell; periodic finite-volume balance.
    velocity = np.asarray(speed)[:, None]
    flux_right = np.where(velocity >= 0, velocity * field,
                          velocity * np.roll(field, -1, axis=-1))
    return field - dt / dx * (flux_right - np.roll(flux_right, 1, axis=-1))


def independent_initial(data, cells):
    c = np.asarray(data["initial_coefficients"])
    theta = (np.arange(cells) + 0.5) * (2 * np.pi / cells)
    modes = np.array([np.ones(cells), np.cos(theta), np.sin(theta)])
    vapour = .010 + .0015 * np.cos(theta) + .0004 * (c[:, 0:3] @ modes)
    liquid = np.maximum(.0003 + .00015 * (c[:, 3:6] @ modes), 0)
    temperature = 282 + 3 * np.sin(theta + .5) + .5 * (c[:, 6:9] @ modes)
    vapour, liquid = equilibrium_partition(vapour, liquid, temperature)
    return vapour, liquid, temperature, theta


def check_analytic_operators():
    rng = np.random.default_rng(724501)
    field = rng.uniform(.0001, .02, (12, 48))
    speed = np.linspace(-18, 18, 12)
    moved = face_flux_step(field, speed, 300, 25000)
    close(moved.sum(axis=1), field.sum(axis=1), "Periodic advection conservation", atol=1e-14)
    require(moved.min() >= field.min() - 1e-14 and moved.max() <= field.max() + 1e-14,
            "Monotone advection bound")
    constant = np.full((12, 48), .0123)
    close(face_flux_step(constant, speed, 300, 25000), constant, "Constant-field advection", atol=1e-15)
    temperature = rng.uniform(271, 294, field.shape)
    liquid = rng.uniform(0, .012, field.shape)
    vapour, condensate = equilibrium_partition(field, liquid, temperature)
    close(vapour + condensate, field + liquid, "Equilibrium water conservation", atol=1e-15)
    require(vapour.min() >= 0 and condensate.min() >= -1e-15, "Equilibrium positivity")
    require(np.max(vapour - saturated_ratio(temperature)) < 1e-14, "No remaining supersaturation")
    wet = condensate > 1e-12
    close(vapour[wet], saturated_ratio(temperature)[wet], "Saturated cells containing condensate", atol=1e-14)
    dry = np.full((12, 48), .0001)
    zero = np.zeros_like(dry)
    vv, ll = equilibrium_partition(dry, zero, temperature)
    close(vv, dry, "Unsaturated dry parcels unchanged", atol=1e-15)
    close(ll, zero, "No negative condensate from evaporation", atol=1e-15)
    return {"status": "passed", "checks": ["periodic total-water conservation",
            "positive/negative wind fluxes", "constant-state preservation", "monotone bounds",
            "saturation and complete evaporation", "local total-water conservation"]}


def audit_dataset(data, scenario, design, replicate):
    n = sum(design["sample_sizes"].values())
    require(len(data["A"]) == n, "Dataset sample count")
    require(np.array_equal(data["episode_id"], np.arange(n)), "Unique ordered episode IDs")
    expected_split = np.repeat(np.arange(3), list(design["sample_sizes"].values()))
    require(np.array_equal(data["split"], expected_split), "Frozen train/validation/test split")
    seed = int(design["seed"]) + int(replicate) * 1000
    rng = np.random.default_rng(seed)
    expected_z, expected_eta = rng.normal(size=(2, n))
    coefficients = rng.normal(size=(n, 9))
    uniform = rng.random(n)
    close(data["initial_z"], expected_z, "Initial Z random stream", atol=0, rtol=0)
    close(data["initial_eta"], expected_eta, "Initial eta random stream", atol=0, rtol=0)
    close(data["initial_coefficients"], coefficients, "Initial field coefficients", atol=0, rtol=0)
    probability = (np.full(n, .5) if scenario["randomized"] else
                   np.clip(1.0 / (1.0 + np.exp(-1.5 * expected_z)), .15, .85))
    close(data["propensity"], probability, "Known treatment propensity", atol=0, rtol=0)
    require(np.array_equal(data["A"], uniform < probability), "Random treatment assignment")
    require(np.array_equal(data["Y"], np.where(data["A"][:, None, None] == 1, data["Y1"], data["Y0"])),
            "One factual outcome from assigned potential outcome")
    require(np.array_equal(data["horizons"], [6, 12]), "Horizon definitions")
    require(np.array_equal(data["outcomes"], ["vapor_g_per_kg", "cloud_proxy_pp"]), "Outcome names")
    for outcome_array in (data["Y"], data["Y0"], data["Y1"]):
        require(np.isfinite(outcome_array).all(), "Finite outcomes")
        require(np.min(outcome_array[:, :, 0]) >= 0, "Nonnegative vapour output")
        require(np.min(outcome_array[:, :, 1]) >= 0 and np.max(outcome_array[:, :, 1]) <= 100,
                "Cloud proxy within [0,100]")
    require(bool(data["exact_null"]) == bool(scenario["exact_null"]), "Null flag matches scenario")
    if scenario["exact_null"]:
        require(np.array_equal(data["Y0"], data["Y1"]), "Exact null holds for every episode/horizon/outcome")
    vapour, liquid, temperature, theta = independent_initial(data, design["cells"])
    x = (np.arange(design["cells"]) + .5) * design["domain_m"] / design["cells"]
    window = (x >= design["window_m"][0]) & (x < design["window_m"][1])
    upstream = (x >= 0) & (x < 200000)
    mv, ml, mt = [f[:, window].mean(axis=1) for f in (vapour, liquid, temperature)]
    upv = vapour[:, upstream].mean(axis=1)
    prewind = 8 + 2 * np.tanh(expected_z) + 1.5 * np.tanh(expected_eta)
    if scenario["feedback"]:
        prewind += 2 * np.tanh(ml / .001 - 1)
    proxy = (100 * (1 - np.exp(-liquid[:, window] / .001))).mean(axis=1)
    local = np.column_stack((mv * 1000, ml * 1000, mt, proxy, upv * 1000,
                             (mv - upv) * 1000 / 200, prewind))
    observed = np.column_stack((local, expected_z))
    full = np.column_stack((vapour * 1000, liquid * 1000, temperature, expected_z, prewind))
    for name, expected in (("X_local", local), ("X_observed", observed), ("X_full", full)):
        close(data[name], expected, "Pretreatment-only feature reconstruction " + name, atol=2e-10)
    return {"status": "passed", "episodes": n, "split_counts": list(design["sample_sizes"].values()),
            "propensity_min": float(probability.min()), "propensity_max": float(probability.max()),
            "null_all_potential_outcomes_equal": bool(np.array_equal(data["Y0"], data["Y1"]))}


def audit_trace(data, trace, scenario, design, replicate):
    dt, cells = int(design["dt_seconds"]), int(design["cells"])
    n = len(data["A"])
    retained = trace["state0"].shape[1]
    require(retained == 8, "Eight complete trajectories must be retained")
    vapour, liquid, temperature, theta = independent_initial(data, cells)
    initial_fields = [a[:retained].copy() for a in (vapour, liquid, temperature)]
    # Independently rebuild the common OU paths from generator seeds, not from
    # production exogenous_paths. This also binds saved paths to frozen seeds.
    noise = np.random.default_rng(int(design["seed"]) + int(replicate) * 1000 + 1).normal(
        size=(43200 // 150, 2, n))
    z = data["initial_z"][:retained].copy()
    eta = data["initial_eta"][:retained].copy()
    zs, etas = [z.copy()], [eta.copy()]
    rz, re = math.exp(-150 / 21600), math.exp(-150 / 10800)
    for innovation in noise:
        z = rz * z + math.sqrt(-math.expm1(-300 / 21600)) * innovation[0, :retained]
        eta = re * eta + math.sqrt(-math.expm1(-300 / 10800)) * innovation[1, :retained]
        zs.append(z.copy()); etas.append(eta.copy())
    close(trace["z"], np.array(zs), "Common saved 150-second Z paths", atol=2e-12)
    close(trace["eta"], np.array(etas), "Common saved 150-second eta paths", atol=2e-12)
    dx = design["domain_m"] / cells
    x = (np.arange(cells) + .5) * dx
    window = (x >= design["window_m"][0]) & (x < design["window_m"][1])
    summary = {"status": "passed", "retained_episodes": retained,
               "steps_per_arm": 43200 // dt, "maximum_state_abs_error": 0.,
               "maximum_wind_abs_error": 0., "maximum_water_budget_residual": 0.}
    for arm in (0, 1):
        vapour, liquid, temperature = [f.copy() for f in initial_fields]
        initial_water = (vapour + liquid).mean(axis=1)
        source, sink = np.zeros(retained), np.zeros(retained)
        expected_state_shape = (43200 // dt + 1, retained, 3, cells)
        require(trace[f"state{arm}"].shape == expected_state_shape, "Complete trajectory dimensions")
        close(trace[f"state{arm}"][0], np.stack(initial_fields, axis=1), "Initial trajectory state", atol=2e-11)
        for step in range(43200 // dt):
            clock = step * dt
            z, eta = trace["z"][clock // 150], trace["eta"][clock // 150]
            wind = 8 + 2 * np.tanh(z) + 1.5 * np.tanh(eta)
            if scenario["feedback"]:
                wind += 2 * np.tanh(liquid[:, window].mean(axis=1) / .001 - 1)
            if clock < 3600:
                wind += 4 * arm - 2
            close(trace[f"wind{arm}"][step], wind, "Prescribed wind and feedback at each step", atol=2e-11)
            summary["maximum_wind_abs_error"] = max(summary["maximum_wind_abs_error"],
                float(np.max(np.abs(trace[f"wind{arm}"][step] - wind))))
            speed = wind if scenario["transport"] else np.zeros(retained)
            require(np.max(np.abs(speed)) * dt / dx <= 1, "Upwind CFL bound")
            previous_water = (vapour + liquid).mean(axis=1)
            vapour, liquid, temperature = [face_flux_step(f, speed, dt, dx)
                                          for f in (vapour, liquid, temperature)]
            close((vapour + liquid).mean(axis=1), previous_water, "Every-step periodic water flux", atol=2e-14)
            humidity_target = .010 + .0015 * np.cos(theta) + .0012 * np.tanh(z[:, None])
            temperature_target = 282 + 3 * np.sin(theta + .5) - 1.5 * np.tanh(z[:, None])
            vapour_increment = (humidity_target - vapour) * (-math.expm1(-dt / 43200))
            source += vapour_increment.mean(axis=1)
            vapour += vapour_increment
            temperature += (temperature_target - temperature) * (-math.expm1(-dt / 21600))
            previous_water = vapour + liquid
            vapour, liquid = equilibrium_partition(vapour, liquid, temperature)
            close(vapour + liquid, previous_water, "Every-step condensation water balance", atol=2e-14)
            precipitation = liquid * (-math.expm1(-dt / 21600))
            liquid -= precipitation
            sink += precipitation.mean(axis=1)
            residual = (vapour + liquid).mean(axis=1) - initial_water - source + sink
            summary["maximum_water_budget_residual"] = max(summary["maximum_water_budget_residual"], float(np.max(np.abs(residual))))
            require(np.max(np.abs(residual)) < 1e-12, "Complete water budget")
            require(vapour.min() >= -1e-14 and liquid.min() >= -1e-14, "Every-step moisture positivity")
            rebuilt = np.stack((vapour, liquid, temperature), axis=1)
            stored = trace[f"state{arm}"][step + 1]
            close(stored, rebuilt, "Independently reconstructed complete trajectory", atol=2e-10, rtol=2e-12)
            summary["maximum_state_abs_error"] = max(summary["maximum_state_abs_error"], float(np.max(np.abs(stored - rebuilt))))
            if clock + dt in (21600, 43200):
                hi = 0 if clock + dt == 21600 else 1
                q = 1000 * vapour[:, window].mean(axis=1)
                cloud = 100 * np.mean(1 - np.exp(-liquid[:, window] / .001), axis=1)
                close(data[f"Y{arm}"][:retained, hi], np.column_stack((q, cloud)),
                      "Truth endpoints from complete trajectories", atol=2e-8)
    if scenario["exact_null"]:
        require(np.array_equal(trace["state0"], trace["state1"]), "Exact-null complete paired trajectories identical")
        require(not np.array_equal(trace["wind0"], trace["wind1"]), "Null retains the distinct assigned wind policies")
    return summary


def independent_ols(data, hi, oi, method):
    train, test = data["split"] == 0, data["split"] == 2
    action, y = data["A"], data["Y"][train, hi, oi]
    names = {"local_ols": "X_local", "observed_z_ols": "X_observed"}
    controls = data[names[method]] if method in names else np.empty((len(action), 0))
    mean, scale = controls[train].mean(axis=0), controls[train].std(axis=0)
    scale[scale <= 1e-14] = 1.
    standard = (controls - mean) / scale
    matrix = np.column_stack((np.ones(train.sum()), standard[train], action[train]))
    cutoff = max(matrix.shape) * np.finfo(float).eps
    coefficient = np.linalg.lstsq(matrix, y, rcond=cutoff)[0]
    inverse = np.linalg.pinv(matrix, rcond=cutoff)
    residual = y - matrix @ coefficient
    leverage = np.einsum("ij,ji->i", matrix, inverse)
    scaled = residual / (1 - leverage)
    influence = inverse * scaled[None, :]
    covariance = influence @ influence.T
    se = float(np.sqrt(covariance[-1, -1]))
    test0 = np.column_stack((np.ones(test.sum()), standard[test], np.zeros(test.sum())))
    test1 = test0.copy(); test1[:, -1] = 1
    return dict(estimate=float(coefficient[-1]), se=se,
                m0=test0 @ coefficient, m1=test1 @ coefficient,
                coefficient=coefficient, mean=mean, scale=scale, covariance=covariance)


def audit_model_directory(data, directory, scenario, replicate, design, verify_hgb=True):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    require(manifest["status"] == "completed", "Completed estimator manifest")
    for relative, info in manifest["artifacts"].items():
        path = directory / relative
        require(path.is_file(), "Missing bound estimator artifact " + relative)
        require(path.stat().st_size == info["bytes"] and sha(path) == info["sha256"],
                "Estimator artifact hash mismatch " + relative)
    predictions = pd.read_csv(directory / "test_predictions.csv.gz")
    metrics = pd.read_csv(directory / "metrics.csv")
    require(len(metrics) == 32 and len(predictions) == 32 * np.sum(data["split"] == 2), "Complete estimator rows")
    require(not metrics.duplicated(["method", "horizon_hours", "outcome"]).any(), "Unique metric keys")
    require(not predictions.duplicated(["method", "horizon_hours", "outcome", "episode_index"]).any(), "Unique prediction keys")
    require(set(metrics.method) == set(design["estimators"]), "All frozen methods present")
    test = np.flatnonzero(data["split"] == 2)
    outcomes = list(data["outcomes"])
    fresh_rows = []
    independent_hgb_refits = 0
    for row in metrics.to_dict("records"):
        method, horizon, outcome = row["method"], int(row["horizon_hours"]), row["outcome"]
        hi, oi = [6, 12].index(horizon), outcomes.index(outcome)
        pframe = predictions[(predictions.method == method) &
                            (predictions.horizon_hours == horizon) & (predictions.outcome == outcome)].sort_values("episode_index")
        require(np.array_equal(pframe.episode_index, test), "Full untouched evaluation episode list")
        require(set(pframe.scenario) == {scenario["name"]} and set(pframe.replicate) == {replicate}, "Prediction scenario/replicate provenance")
        a, propensity = data["A"][test].astype(float), data["propensity"][test]
        factual = data["Y"][test, hi, oi]
        y0, y1 = data["Y0"][test, hi, oi], data["Y1"][test, hi, oi]
        truth = y1 - y0
        for column, expected in (("A", a), ("propensity", propensity), ("factual_y", factual),
                                 ("truth_y0", y0), ("truth_y1", y1), ("paired_intervention_effect", truth)):
            close(pframe[column].to_numpy(), expected, "Prediction-column binding " + column, atol=2e-10)
        m0, m1 = pframe.m0.to_numpy(), pframe.m1.to_numpy()
        standard_error = float("nan")
        if method.endswith("ols"):
            fitted = independent_ols(data, hi, oi, method)
            estimate, standard_error = fitted["estimate"], fitted["se"]
            close(m0, fitted["m0"], "OLS independently refitted m0", atol=2e-8)
            close(m1, fitted["m1"], "OLS independently refitted m1", atol=2e-8)
            saved = load_npz(directory / "models" / f"h{horizon}_y{oi}_{method}.npz")
            for key, result in (("control_mean", fitted["mean"]), ("control_scale", fitted["scale"]),
                                ("hc3_covariance", fitted["covariance"])):
                close(saved[key], result, "Independent OLS parameter " + key, atol=2e-8)
        elif method.endswith("gcomp"):
            estimate = float(np.mean(m1 - m0))
            if verify_hgb:
                key = {"local_hgb_gcomp": "X_local", "observed_z_hgb_gcomp": "X_observed",
                       "full_hgb_gcomp": "X_full"}[method]
                estimator = joblib.load(directory / "models" / f"h{horizon}_y{oi}_{method}.joblib")
                for parameter, value in design["hgb"].items():
                    require(estimator.get_params()[parameter] == value, "Frozen HGB parameter " + parameter)
                x = data[key][test]
                close(m0, estimator.predict(np.column_stack((np.zeros(len(test)), x))), "Stored HGB m0 prediction", atol=2e-9)
                close(m1, estimator.predict(np.column_stack((np.ones(len(test)), x))), "Stored HGB m1 prediction", atol=2e-9)
                if replicate == 0 and horizon == 6 and method == "observed_z_hgb_gcomp":
                    # Bounded factual-only training audit: two outcomes in all
                    # four scenarios, eight of the 960 HGB fits. No training
                    # potential-outcome arrays enter this independent refit.
                    train = data["split"] == 0
                    raw_x = data[key]
                    independent = HistGradientBoostingRegressor(**design["hgb"])
                    independent.fit(np.column_stack((data["A"][train].astype(float), raw_x[train])),
                                    data["Y"][train, hi, oi])
                    close(m0, independent.predict(np.column_stack((np.zeros(len(test)), x))),
                          "Independent factual-only HGB refit m0", atol=2e-9)
                    close(m1, independent.predict(np.column_stack((np.ones(len(test)), x))),
                          "Independent factual-only HGB refit m1", atol=2e-9)
                    independent_hgb_refits += 1
        else:
            if method == "oracle_ipw":
                pseudo = a * factual / propensity - (1 - a) * factual / (1 - propensity)
                require(np.isnan(m0).all() and np.isnan(m1).all(), "IPW has no outcome nuisance predictions")
            elif method == "oracle_aipw":
                reference = predictions[(predictions.method == "observed_z_hgb_gcomp") &
                    (predictions.horizon_hours == horizon) & (predictions.outcome == outcome)].sort_values("episode_index")
                close(m0, reference.m0.to_numpy(), "AIPW uses heldout observed-Z nuisance m0")
                close(m1, reference.m1.to_numpy(), "AIPW uses heldout observed-Z nuisance m1")
                pseudo = m1 - m0 + a / propensity * (factual - m1) - (1 - a) / (1 - propensity) * (factual - m0)
            else:
                raise AssertionError("Unexpected method " + method)
            close(pframe.influence_pseudo_outcome.to_numpy(), pseudo, "Oracle estimating score", atol=2e-8)
            estimate = float(np.mean(pseudo))
            standard_error = float(np.std(pseudo, ddof=1) / np.sqrt(len(pseudo)))
        expected = dict(effect_estimate=estimate, test_bank_truth_mean=float(truth.mean()),
            truth_mc_standard_error=float(truth.std(ddof=1) / np.sqrt(len(truth))),
            signed_error_vs_test_bank=estimate - float(truth.mean()),
            absolute_error_vs_test_bank=abs(estimate - float(truth.mean())), standard_error=standard_error,
            n_train=int(np.sum(data["split"] == 0)), n_test=len(test))
        if np.isfinite(m0).all():
            prediction = (1 - a) * m0 + a * m1
            close(pframe.factual_prediction.to_numpy(), prediction, "Factual model prediction")
            expected["factual_test_mse"] = float(np.mean((prediction - factual) ** 2))
        else:
            expected["factual_test_mse"] = float("nan")
            require(pframe.factual_prediction.isna().all(), "No fabricated oracle IPW forecasts")
        if method.endswith("ols") or method.endswith("gcomp"):
            close(pframe.effect_prediction.to_numpy(), m1 - m0, "Conditional-mean model contrast")
            expected["episode_twin_effect_rmse"] = float(np.sqrt(np.mean((m1 - m0 - truth) ** 2)))
        else:
            expected["episode_twin_effect_rmse"] = float("nan")
            require(pframe.effect_prediction.isna().all(), "No fabricated oracle episode treatment effects")
        if np.isfinite(standard_error):
            expected["ci95_low"] = estimate - Z975 * standard_error
            expected["ci95_high"] = estimate + Z975 * standard_error
            expected["p_value_zero"] = (math.erfc(abs(estimate) / standard_error / math.sqrt(2))
                                        if standard_error > 0 else (1. if estimate == 0 else 0.))
        else:
            expected.update(ci95_low=float("nan"), ci95_high=float("nan"), p_value_zero=float("nan"))
        for key, value in expected.items():
            close(np.asarray(row[key]), np.asarray(value), "Recomputed metric " + key, atol=3e-8)
        require(bool(row["exact_structural_null"]) == bool(scenario["exact_null"]), "Metric structural-null flag")
        rejection = bool(expected["ci95_low"] > 0 or expected["ci95_high"] < 0)
        if scenario["exact_null"] and np.isfinite(standard_error):
            recorded = row["reject_zero_on_exact_null"]
            require(recorded is True or recorded is False or str(recorded).lower() in ("true", "false"), "Defined null rejection")
            require((str(recorded).lower() == "true") == rejection, "Exact-null interval rejection")
        else:
            require(pd.isna(row["reject_zero_on_exact_null"]), "No rejection claims outside structural null/defined interval")
        fresh_rows.append({**row, **expected})
    return {"status": "passed", "metric_rows": len(metrics), "prediction_rows": len(predictions),
            "ols_refits": 12, "hgb_saved_model_prediction_checks": 12 if verify_hgb else 0,
            "hgb_independent_factual_only_refits": independent_hgb_refits,
            "oracle_score_vectors": 8}, pd.DataFrame(fresh_rows)


def locate_input(out, scenario, replicate, trace=False):
    """Resolve one unambiguous generated artifact, allowing runner layout choice."""
    rep = f"rep_{int(replicate):03d}"
    candidates = []
    wanted = ("trace.npz", "traces.npz", "trajectories.npz") if trace else ("episodes.npz", "data.npz", "dataset.npz")
    for name in wanted:
        candidates.extend(out.glob(f"**/{scenario}/{rep}/{name}"))
    candidates = list(dict.fromkeys(candidates))
    require(len(candidates) == 1, f"Expected one {'trace' if trace else 'dataset'} for {scenario}/{rep}; got {candidates}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--numerics-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INCOMPLETE", "scope": "Independent numeric reconstruction and metric verification; not physical causal validation",
        "audit_source_sha256": sha(__file__), "design_sha256": sha(out / "design.json"),
        "numerics_only_requested": bool(args.numerics_only),
        "hgb_independent_refit_scope": "Eight of 960 fits: observed-Z, both six-hour outcomes, replicate 0 in each scenario",
        "stages": {}}
    try:
        design = json.loads((out / "design.json").read_text(encoding="utf8"))
        production = json.loads((out / "run_manifest.json").read_text(encoding="utf8"))
        require(production["design_sha256"] == receipt["design_sha256"], "Production frozen design hash")
        if not args.numerics_only:
            require(production["status"] == "completed", "Production must finish before full audit")
            expected_cases = len(design["scenarios"]) * design["replicates"]
            require(len(production["completed"]) == expected_cases, "All planned scenario/replicate cases completed")
            science = ("revision/intervention_benchmark/simulation.py", "revision/intervention_benchmark/estimators.py",
                       "revision/intervention_benchmark/run.py", "revision/inference.py")
            repo = Path(__file__).resolve().parents[2]
            bound_sources = {name.replace("\\", "/"): digest for name, digest in production["source_sha256"].items()}
            for name in science:
                require(sha(repo / name) == bound_sources[name], "Production scientific code hash " + name)
            receipt["production_manifest_sha256"] = sha(out / "run_manifest.json")
        receipt["stages"]["analytic_operators"] = check_analytic_operators()
        numeric_records = []
        for scenario in design["scenarios"]:
            dataset_path = locate_input(out, scenario["name"], 0)
            trace_path = locate_input(out, scenario["name"], 0, trace=True)
            data, trace = load_npz(dataset_path), load_npz(trace_path)
            record = {"scenario": scenario["name"], "replicate": 0,
                "dataset_sha256": sha(dataset_path), "trace_sha256": sha(trace_path),
                "dataset": audit_dataset(data, scenario, design, 0),
                "trajectory": audit_trace(data, trace, scenario, design, 0)}
            numeric_records.append(record)
        receipt["stages"]["numerics"] = {"status": "passed", "records": numeric_records}
        if args.numerics_only:
            receipt["status"] = "PASS_NUMERICS_ONLY_MODELS_NOT_AUDITED"
        else:
            all_metrics, model_records = [], []
            with threadpool_limits(limits=2):
                for scenario in design["scenarios"]:
                    for replicate in range(design["replicates"]):
                        dataset_path = locate_input(out, scenario["name"], replicate)
                        data = load_npz(dataset_path)
                        data_check = audit_dataset(data, scenario, design, replicate)
                        records = [r for r in production["completed"] if r["scenario"] == scenario["name"] and r["replicate"] == replicate]
                        require(len(records) == 1, "Unique production completion record")
                        require(records[0]["episodes_sha256"] == sha(dataset_path), "Production source dataset hash")
                        numerics_path = dataset_path.parent / "numerics.json"
                        require(records[0]["numerics_sha256"] == sha(numerics_path), "Production numerical ledger hash")
                        ledgers = json.loads(numerics_path.read_text(encoding="utf8"))
                        for arm in ("arm0", "arm1"):
                            require(ledgers[arm]["cfl"] <= 1 and ledgers[arm]["water_residual"] < 1e-12,
                                    "Complete-production CFL and water ledger")
                            require(ledgers[arm]["min_rv"] >= -1e-14 and ledgers[arm]["min_rl"] >= -1e-14,
                                    "Complete-production moisture positivity")
                        if scenario["exact_null"]:
                            require(ledgers["arm0"]["all_episode_all_step_state_sha256"] ==
                                    ledgers["arm1"]["all_episode_all_step_state_sha256"],
                                    "Recorded full-ensemble/all-step paired null equality")
                        matches = list(out.glob(f"**/{scenario['name']}/rep_{replicate:03d}/test_predictions.csv.gz"))
                        require(len(matches) == 1, "Exactly one completed estimator prediction artifact")
                        record, metrics = audit_model_directory(data, matches[0].parent, scenario, replicate,
                                                               design, verify_hgb=True)
                        record.update(scenario=scenario["name"], replicate=replicate,
                                      dataset_sha256=sha(dataset_path), dataset_audit=data_check)
                        model_records.append(record); all_metrics.append(metrics)
            receipt["stages"]["models"] = {"status": "passed", "records": model_records,
                "total_metric_rows": sum(r["metric_rows"] for r in model_records),
                "total_prediction_rows": sum(r["prediction_rows"] for r in model_records),
                "total_ols_independent_refits": sum(r["ols_refits"] for r in model_records),
                "total_hgb_saved_model_prediction_checks": sum(r["hgb_saved_model_prediction_checks"] for r in model_records),
                "total_hgb_independent_factual_only_refits": sum(r["hgb_independent_factual_only_refits"] for r in model_records)}
            require(receipt["stages"]["models"]["total_hgb_independent_factual_only_refits"] == 8,
                    "Eight planned independent HGB refits")
            # Global summary audit is wired separately once the runner's final
            # summary contract exists. Do not grant full status without it.
            combined = pd.concat(all_metrics, ignore_index=True)
            receipt["stages"]["summaries"] = audit_summaries(out, combined, design)
            require(all(stage["status"] == "passed" for stage in receipt["stages"].values()),
                    "Every mandatory audit stage must pass")
            receipt["status"] = "PASS_NUMERICS_MODELS_METRICS_SUMMARIES"
    except Exception as error:
        receipt["status"] = "FAIL_OR_REQUIRED_EVIDENCE_MISSING"
        receipt["error"] = repr(error)
        receipt["traceback"] = traceback.format_exc()
        (out / "independent_audit.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf8")
        raise
    (out / "independent_audit.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(receipt["status"])


def audit_summaries(out, independently_recomputed, design):
    keys = ["scenario", "method", "horizon_hours", "outcome"]
    idkeys = keys + ["replicate"]
    combined = pd.read_csv(out / "all_metrics.csv").sort_values(idkeys).reset_index(drop=True)
    fresh = independently_recomputed.sort_values(idkeys).reset_index(drop=True)
    require(len(combined) == len(fresh), "Complete concatenated metric row count")
    require(combined[idkeys].equals(fresh[idkeys]), "Concatenated metric keys")
    numeric_fields = ["effect_estimate", "test_bank_truth_mean", "truth_mc_standard_error",
        "signed_error_vs_test_bank", "absolute_error_vs_test_bank", "episode_twin_effect_rmse",
        "factual_test_mse", "standard_error", "ci95_low", "ci95_high", "p_value_zero"]
    for field in numeric_fields:
        close(combined[field].to_numpy(), fresh[field].to_numpy(), "Global metric " + field, atol=3e-8)
    summary = pd.read_csv(out / "summary.csv")
    require(not summary.duplicated(keys).any(), "Unique aggregate summary keys")
    require(len(summary) == len(design["scenarios"]) * len(design["estimators"]) * 4,
            "Every frozen aggregate condition present")
    aggregated = ["effect_estimate", "test_bank_truth_mean", "signed_error_vs_test_bank",
                  "absolute_error_vs_test_bank", "factual_test_mse", "episode_twin_effect_rmse", "truth_mc_standard_error"]
    for key, group in fresh.groupby(keys, sort=True):
        mask = np.ones(len(summary), dtype=bool)
        for column, value in zip(keys, key):
            mask &= summary[column].to_numpy() == value
        require(mask.sum() == 1, "Matching aggregate row")
        row = summary.loc[mask].iloc[0]
        require(int(row.n_replicates) == design["replicates"] and len(group) == design["replicates"],
                "All independent replicates in aggregate")
        require(set(group.replicate) == set(range(design["replicates"])), "No duplicate/missing replicate")
        for field in aggregated:
            x = group[field].dropna().to_numpy()
            mean_column = "mean_" + field
            if not len(x):
                require(pd.isna(row[mean_column]), "No aggregate for absent diagnostic")
                continue
            average = float(np.sum(x) / len(x))
            mcse = float(np.sqrt(np.sum((x - average) ** 2) / (len(x) * (len(x) - 1))))
            half = float(stdtrit(len(x) - 1, .975) * mcse)
            for column, value in ((mean_column, average), ("mc95_low_" + field, average - half),
                                  ("mc95_high_" + field, average + half)):
                close(np.asarray(row[column]), np.asarray(value), "Aggregate Monte Carlo diagnostic " + column, atol=3e-8)
        rmse = float(np.sqrt(np.mean(group.signed_error_vs_test_bank.to_numpy() ** 2)))
        close(np.asarray(row.ate_rmse), np.asarray(rmse), "Aggregate ATE RMSE", atol=3e-8)
        if key[0] == "confounded_null" and not group.standard_error.isna().all():
            rejects = np.logical_or(group.ci95_low.to_numpy() > 0, group.ci95_high.to_numpy() < 0)
            n, k = len(rejects), int(rejects.sum())
            require(int(row.null_tests) == n and int(row.null_rejections) == k, "Exact-null rejection counts")
            rate = k / n
            radical = Z975 * math.sqrt(Z975 ** 2 + 4 * k * (1 - k / n))
            lower = (2 * k + Z975 ** 2 - radical) / (2 * (n + Z975 ** 2))
            upper = (2 * k + Z975 ** 2 + radical) / (2 * (n + Z975 ** 2))
            for column, value in (("null_rejection_rate", rate), ("null_wilson_low", lower), ("null_wilson_high", upper)):
                close(np.asarray(row[column]), np.asarray(value), "Null rejection/Wilson statistic " + column, atol=3e-8)
        else:
            require(pd.isna(row.null_tests), "No null calibration claims for other conditions")
    resolution = pd.read_csv(out / "resolution_sensitivity.csv")
    require(len(resolution) == 48, "All frozen resolution comparisons retained")
    for scenario in design["scenarios"]:
        effects = load_npz(out / f"resolution_effects_{scenario['name']}.npz")
        require(set(effects) == {"base", "half_dt", "refined_grid"}, "All effect-resolution arrays")
        for label, effect in effects.items():
            require(effect.shape == (128, 2, 2), "Resolution comparison fixed sample size")
            if scenario["exact_null"]:
                require(np.array_equal(effect, np.zeros_like(effect)), "Exact null at each numerical resolution")
            cells, dt = {"base": (48, 300), "half_dt": (48, 150), "refined_grid": (96, 150)}[label]
            for hi, horizon in enumerate((6, 12)):
                for oi, outcome in enumerate(("vapor_g_per_kg", "cloud_proxy_pp")):
                    match = resolution[(resolution.scenario == scenario["name"]) & (resolution.resolution == label) &
                                       (resolution.horizon == horizon) & (resolution.outcome == outcome)]
                    require(len(match) == 1, "Unique resolution comparison row")
                    row = match.iloc[0]
                    require(int(row.cells) == cells and int(row.dt_seconds) == dt and int(row.n) == 128,
                            "Resolution parameter provenance")
                    values = effect[:, hi, oi]
                    difference = values - effects["base"][:, hi, oi]
                    for name, value in (("mean_effect", values.mean()), ("mean_effect_change", difference.mean()),
                                        ("paired_effect_change_rmse", np.sqrt(np.mean(difference ** 2)))):
                        close(np.asarray(row[name]), np.asarray(value), "Resolution effect statistic " + name, atol=2e-10)
                    require(row.water_residual < 1e-12, "Resolution simulation water conservation ledger")
    return {"status": "passed", "aggregate_metric_rows": len(combined), "summary_rows": len(summary),
            "resolution_comparison_rows": len(resolution),
            "resolution_scope": "Metrics recomputed from paired effect arrays; independent full trajectory reconstruction covers production replicate 0 only",
            "all_metrics_sha256": sha(out / "all_metrics.csv"), "summary_sha256": sha(out / "summary.csv"),
            "resolution_sensitivity_sha256": sha(out / "resolution_sensitivity.csv")}


if __name__ == "__main__":
    main()
