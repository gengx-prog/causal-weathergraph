"""Fixed observational adapters evaluated against withheld simulator twins.

The original revision OLS coefficient is reused, but its pretreatment design is
an explicit new adapter. Predictive MSE, a coefficient ranking score, and an
intervention effect are never substituted for one another. Episodes are IID;
no calendar HAC covariance is applied to arbitrary episode ordering.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from revision.inference import fit_edge


HGB_PARAMETERS = dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=30,
                      learning_rate=.05, l2_regularization=1., max_bins=64,
                      early_stopping=False, random_state=20261002)
NORMAL_975 = 1.959963984540054


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf8")


def _normal_interval(estimate, standard_error):
    if not np.isfinite(estimate) or not np.isfinite(standard_error):
        return (float("nan"),) * 3
    low, high = (float(estimate - NORMAL_975 * standard_error),
                 float(estimate + NORMAL_975 * standard_error))
    p = (float(2 * ndtr(-abs(estimate) / standard_error))
         if standard_error > 0 else (1. if estimate == 0 else 0.))
    return low, high, p


def ols_adapter(y, action, controls, eval_controls):
    """Reuse fit_edge and independently derive IID HC3 covariance by SVD.

    Standardization is fitted on training controls only and does not alter the
    treatment coefficient. HC3 inference concerns the population projection
    coefficient, not automatically an ATE in a nonlinear/confounded system.
    """
    y, action = np.asarray(y, float), np.asarray(action, float)
    controls, eval_controls = np.asarray(controls, float), np.asarray(eval_controls, float)
    if controls.ndim != 2 or eval_controls.ndim != 2 or controls.shape[1] != eval_controls.shape[1]:
        raise ValueError("Training and evaluation controls must have equal column count")
    if len(y) != len(action) or len(y) != len(controls):
        raise ValueError("Training arrays are not aligned")
    mean = controls.mean(axis=0)
    scale = controls.std(axis=0)
    scale = np.where(scale > 1e-14, scale, 1.)
    c = (controls - mean) / scale
    ec = (eval_controls - mean) / scale
    original = fit_edge(y, c, action, bandwidths=())
    if not original.get("valid", False):
        raise ValueError("Original OLS implementation rejected design: " + original["status"])
    design = np.column_stack((np.ones(len(y)), c, action))
    left, singular, right = np.linalg.svd(design, full_matrices=False)
    tolerance = max(design.shape) * np.finfo(float).eps * singular.max()
    keep = singular > tolerance
    inverse = (right[keep].T / singular[keep]) @ left[:, keep].T
    coefficient = inverse @ y
    beta = float(original["effect_coefficient"])
    if not np.isclose(beta, coefficient[-1], rtol=2e-8, atol=2e-10):
        raise AssertionError("Independent SVD and original FWL treatment coefficients disagree")
    residual = y - design @ coefficient
    leverage = np.sum(left[:, keep] ** 2, axis=1)
    if np.any(leverage >= 1 - 1e-10):
        raise ValueError("HC3 is undefined or unstable for leverage approaching one")
    # Independent sandwich calculation; covariance includes intercept/controls.
    influence_matrix = inverse * (residual / (1 - leverage))[None, :]
    covariance = influence_matrix @ influence_matrix.T
    standard_error = float(np.sqrt(max(0., covariance[-1, -1])))
    low, high, p = _normal_interval(beta, standard_error)
    design0 = np.column_stack((np.ones(len(ec)), ec, np.zeros(len(ec))))
    design1 = design0.copy()
    design1[:, -1] = 1.
    return {
        "estimate": beta, "standard_error": standard_error,
        "ci95_low": low, "ci95_high": high, "p_value": p,
        "m0": design0 @ coefficient, "m1": design1 @ coefficient,
        "model": {"coefficient_standardized_controls": coefficient,
                  "control_mean": mean, "control_scale": scale,
                  "hc3_covariance": covariance, "singular_values": singular,
                  "numerical_rank": np.asarray(int(keep.sum())),
                  "max_leverage": np.asarray(float(leverage.max()))},
        "original_fit_edge": original,
    }


def _validate(data):
    split, action = np.asarray(data["split"]), np.asarray(data["A"])
    propensity, outcome = np.asarray(data["propensity"], float), np.asarray(data["Y"], float)
    if split.ndim != 1 or action.shape != split.shape or propensity.shape != split.shape:
        raise ValueError("split, A, and propensity must have the same vector shape")
    if not np.isin(split, [0, 1, 2]).all() or not np.isin(action, [0, 1]).all():
        raise ValueError("Unknown split or nonbinary treatment")
    if not np.isfinite(propensity).all() or not ((propensity > 0) & (propensity < 1)).all():
        raise ValueError("Known propensities must be finite and strictly between zero and one")
    if outcome.shape != (len(split), len(data["horizons"]), len(data["outcomes"])):
        raise ValueError("Factual outcome shape does not match horizon/outcome contract")
    if not np.isfinite(outcome).all():
        raise ValueError("Nonfinite factual outcomes")
    for name in ("X_local", "X_observed", "X_full"):
        x = np.asarray(data[name], float)
        if x.ndim != 2 or len(x) != len(split) or not np.isfinite(x).all():
            raise ValueError("Nonfinite or misaligned pretreatment design: " + name)
    train, test = np.flatnonzero(split == 0), np.flatnonzero(split == 2)
    if len(train) < 50 or len(test) < 3 or len(np.unique(action[train])) != 2:
        raise ValueError("Insufficient training/evaluation episodes or absent treatment arm")
    return train, test, action.astype(float), propensity, outcome


def analyze(data: dict, out_dir: Path, scenario: str, replicate: int) -> pd.DataFrame:
    """Fit the frozen adapters; evaluate twins ONLY on split==2 episodes.

    Outputs live in out_dir/scenario/rep_XXX. Reusing a populated directory is
    refused so an interrupted/final run cannot silently overwrite evidence.
    Validation rows are unused: there is no hyperparameter selection.
    """
    if not scenario or any(char in scenario for char in ("/", "\\", ":")) or scenario in (".", ".."):
        raise ValueError("scenario must be one safe path component")
    train, test, action, propensity, factual = _validate(data)
    output = Path(out_dir) / scenario / f"rep_{int(replicate):03d}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite estimator evidence: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    models = output / "models"
    models.mkdir()
    # These are the only reads of twins. Counterfactuals for train/validation
    # never enter transforms, nuisance fitting, parameter choice, or metrics.
    y0, y1 = np.asarray(data["Y0"][test], float), np.asarray(data["Y1"][test], float)
    expected = factual[test].shape
    if y0.shape != expected or y1.shape != expected or not np.isfinite(y0).all() or not np.isfinite(y1).all():
        raise ValueError("Invalid test-only potential outcomes")
    assigned_truth = np.where(action[test, None, None] == 1, y1, y0)
    if not np.allclose(assigned_truth, factual[test], rtol=1e-10, atol=1e-10):
        raise AssertionError("Factual outcome disagrees with its assigned simulator twin")
    a, p = action[test], propensity[test]
    exact_null = bool(data.get("exact_null", False))
    records, prediction_frames = [], []
    model_files = []
    with threadpool_limits(limits=2):
        for hi, horizon in enumerate(data["horizons"]):
            for oi, outcome in enumerate(data["outcomes"]):
                target_train, target_test = factual[train, hi, oi], factual[test, hi, oi]
                truth0, truth1 = y0[:, hi, oi], y1[:, hi, oi]
                truth = truth1 - truth0
                truth_mean = float(truth.mean())
                truth_mc_se = float(truth.std(ddof=1) / np.sqrt(len(test)))
                if exact_null and not np.allclose(truth, 0., rtol=0, atol=1e-12):
                    raise AssertionError("Exact-null flag contradicts test intervention twins")
                nuisance = {}
                fitted = []
                for method, design_name in (("unadjusted_ols", None), ("local_ols", "X_local"),
                                             ("observed_z_ols", "X_observed")):
                    x = np.empty((len(action), 0)) if design_name is None else np.asarray(data[design_name], float)
                    result = ols_adapter(target_train, action[train], x[train], x[test])
                    basename = f"h{int(horizon)}_y{oi}_{method}"
                    np.savez_compressed(models / (basename + ".npz"), **result["model"])
                    _json(models / (basename + ".json"), {
                        "method": method, "horizon": int(horizon), "outcome": str(outcome),
                        "design": design_name, "training_episodes": len(train),
                        "coefficient_source": "revision.inference.fit_edge; no HAC",
                        "independent_covariance": "IID HC3 sandwich; normal critical value",
                        "effect_coefficient": result["estimate"],
                        "hc3_standard_error": result["standard_error"],
                        "original_ols_standard_error": result["original_fit_edge"]["se_ols"],
                        "rank": int(result["model"]["numerical_rank"]),
                    })
                    model_files.extend([models / (basename + ".npz"), models / (basename + ".json")])
                    fitted.append((method, result, "constant_linear_contrast", "HC3 IID normal; linear projection coefficient"))
                for method, design_name in (("local_hgb_gcomp", "X_local"),
                                             ("observed_z_hgb_gcomp", "X_observed"),
                                             ("full_hgb_gcomp", "X_full")):
                    x = np.asarray(data[design_name], float)
                    estimator = HistGradientBoostingRegressor(**HGB_PARAMETERS)
                    estimator.fit(np.column_stack((action[train], x[train])), target_train)
                    m0 = estimator.predict(np.column_stack((np.zeros(len(test)), x[test])))
                    m1 = estimator.predict(np.column_stack((np.ones(len(test)), x[test])))
                    basename = f"h{int(horizon)}_y{oi}_{method}.joblib"
                    joblib.dump(estimator, models / basename)
                    model_files.append(models / basename)
                    nuisance[method] = (m0, m1)
                    fitted.append((method, {"estimate": float(np.mean(m1 - m0)), "m0": m0, "m1": m1},
                                   "conditional_mean_contrast", "none; replicate-level Monte Carlo summaries reported separately"))
                m0, m1 = nuisance["observed_z_hgb_gcomp"]
                pseudo_outcomes = {
                    "oracle_ipw": a * target_test / p - (1 - a) * target_test / (1 - p),
                    "oracle_aipw": m1 - m0 + a * (target_test - m1) / p
                                     - (1 - a) * (target_test - m0) / (1 - p),
                }
                for method, pseudo in pseudo_outcomes.items():
                    estimate, se = float(pseudo.mean()), float(pseudo.std(ddof=1) / np.sqrt(len(test)))
                    low, high, pv = _normal_interval(estimate, se)
                    result = {"estimate": estimate, "standard_error": se, "ci95_low": low,
                              "ci95_high": high, "p_value": pv, "pseudo_outcome": pseudo}
                    if method == "oracle_aipw":
                        result.update(m0=m0, m1=m1)
                    fitted.append((method, result, "marginal_population_effect",
                                   "IID test-episode influence-score normal; known true propensity"))
                for method, result, contrast_kind, interval in fitted:
                    estimate = float(result["estimate"])
                    error = estimate - truth_mean
                    prediction = None
                    if "m0" in result:
                        prediction = np.where(a == 1, result["m1"], result["m0"])
                    # Episode-twin RMSE is a benchmark diagnostic, not CATE risk:
                    # twins retain realized exogenous noise, rather than E[Y(a)|X].
                    effect_prediction = None
                    if method.endswith("ols") or method.endswith("gcomp"):
                        effect_prediction = result["m1"] - result["m0"]
                    low, high = result.get("ci95_low", np.nan), result.get("ci95_high", np.nan)
                    records.append({
                        "scenario": scenario, "replicate": int(replicate), "method": method,
                        "horizon_hours": int(horizon), "outcome": str(outcome),
                        "n_train": len(train), "n_test": len(test), "effect_estimate": estimate,
                        "test_bank_truth_mean": truth_mean, "truth_mc_standard_error": truth_mc_se,
                        "signed_error_vs_test_bank": error, "absolute_error_vs_test_bank": abs(error),
                        "episode_twin_effect_rmse": (float(np.sqrt(np.mean((effect_prediction - truth) ** 2)))
                                                     if effect_prediction is not None else np.nan),
                        "factual_test_mse": float(np.mean((prediction - target_test) ** 2)) if prediction is not None else np.nan,
                        "standard_error": result.get("standard_error", np.nan), "ci95_low": low, "ci95_high": high,
                        "p_value_zero": result.get("p_value", np.nan),
                        "exact_structural_null": exact_null,
                        "reject_zero_on_exact_null": (bool(low > 0 or high < 0) if exact_null and np.isfinite(low) else None),
                        "contrast_kind": contrast_kind, "interval_method": interval,
                        "coverage_test_bank_not_reported": True,
                        "effect_rmse_is_not_cate_risk": True,
                    })
                    frame = pd.DataFrame({"scenario": scenario, "replicate": int(replicate),
                        "method": method, "horizon_hours": int(horizon), "outcome": str(outcome),
                        "episode_index": test, "A": a.astype(int), "propensity": p,
                        "factual_y": target_test, "truth_y0": truth0, "truth_y1": truth1,
                        "paired_intervention_effect": truth,
                        "m0": result.get("m0", np.full(len(test), np.nan)),
                        "m1": result.get("m1", np.full(len(test), np.nan)),
                        "factual_prediction": prediction if prediction is not None else np.full(len(test), np.nan),
                        "effect_prediction": effect_prediction if effect_prediction is not None else np.full(len(test), np.nan),
                        "influence_pseudo_outcome": result.get("pseudo_outcome", np.full(len(test), np.nan))})
                    prediction_frames.append(frame)
    table = pd.DataFrame(records)
    table.to_csv(output / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "test_predictions.csv.gz", index=False, compression="gzip")
    _json(output / "manifest.json", {
        "status": "completed", "scenario": scenario, "replicate": int(replicate),
        "estimators_source_sha256": _sha(__file__),
        "inference_source_sha256": _sha(Path(__file__).parents[1] / "inference.py"),
        "hgb_parameters": HGB_PARAMETERS, "thread_limit": 2,
        "train_count": len(train), "test_count": len(test), "metric_rows": len(table),
        "validation_use": "none; no hyperparameter selection",
        "twins_use": "test rows only; absent from training, model selection, and preprocessing",
        "propensity_use": "true known test propensity used only by oracle IPW/AIPW; not an estimated real-data propensity",
        "interval_limits": "OLS HC3 targets a regression projection. Oracle intervals target population effects under IID episodes. No finite test-bank or selected-edge coverage claim. HGB g-computation has no interval.",
        "null_limits": "Zero rejection is reported only with an explicit exact structural null; confounded projection coefficients need not obey that null.",
        "effect_error_limits": "Paired test-bank signed/absolute errors include finite-bank variability. Episode-twin effect RMSE is not conditional-average-effect risk.",
        "factual_mse_limits": "Oracle AIPW uses observed-Z HGB nuisance predictions for factual MSE; this is not a predictive comparison of its pseudo-outcomes.",
        "artifacts": {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": _sha(path)}
                      for path in [output / "metrics.csv", output / "test_predictions.csv.gz", *model_files]},
    })
    return table
