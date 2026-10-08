"""Independent numerical checks for the intervention benchmark OLS adapter."""
import numpy as np
import pytest
import statsmodels.api as sm

from revision.intervention_benchmark.estimators import ols_adapter


def test_hc3_and_fwl_match_independent_statsmodels_under_heteroskedasticity():
    rng = np.random.default_rng(713)
    controls = rng.normal(size=(700, 3))
    propensity = .2 + .6 / (1 + np.exp(-controls[:, 0]))
    action = rng.binomial(1, propensity)
    outcome = (1.2 + 1.8 * action + controls @ np.array([.4, -.8, .2])
               + rng.normal(size=700) * (.25 + .8 * action + .3 * np.abs(controls[:, 1])))
    eval_controls = rng.normal(size=(50, 3))
    observed = ols_adapter(outcome, action, controls, eval_controls)
    # Independent library, raw controls, independently fit covariance.
    reference = sm.OLS(outcome, np.column_stack((np.ones(700), controls, action))).fit(cov_type="HC3")
    assert observed["estimate"] == pytest.approx(reference.params[-1], rel=1e-10)
    assert observed["standard_error"] == pytest.approx(reference.bse[-1], rel=1e-10)
    np.testing.assert_allclose([observed["ci95_low"], observed["ci95_high"]],
                               reference.conf_int()[-1], rtol=1e-10, atol=1e-12)
    reference0 = np.column_stack((np.ones(50), eval_controls, np.zeros(50))) @ reference.params
    reference1 = np.column_stack((np.ones(50), eval_controls, np.ones(50))) @ reference.params
    np.testing.assert_allclose(observed["m0"], reference0, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(observed["m1"], reference1, rtol=1e-10, atol=1e-12)


def test_redundant_controls_preserve_estimable_action_and_hc3():
    rng = np.random.default_rng(811)
    state = rng.normal(size=250)
    action = rng.binomial(1, .5, size=250)
    outcome = .6 * action + .9 * state + rng.normal(size=250)
    original = ols_adapter(outcome, action, state[:, None], state[:10, None])
    duplicate = np.column_stack((state, 3 * state, np.ones(len(state))))
    redundant = ols_adapter(outcome, action, duplicate, duplicate[:10])
    assert redundant["estimate"] == pytest.approx(original["estimate"], abs=1e-12)
    assert redundant["standard_error"] == pytest.approx(original["standard_error"], abs=1e-12)
    assert int(redundant["model"]["numerical_rank"]) == 3


def test_treatment_collinear_with_pretreatment_design_is_rejected():
    rng = np.random.default_rng(920)
    action = rng.binomial(1, .5, size=100)
    with pytest.raises(ValueError, match="source_collinear_with_controls"):
        ols_adapter(rng.normal(size=100), action, action[:, None], action[:4, None])
