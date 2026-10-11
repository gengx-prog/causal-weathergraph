"""Scientific alignment and conditional-regression checks for E5."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from revision.run_path_diagnostics import (
    initial_bearing, project_east_north, path_design, joint_lag_design,
    train_monthly_transform,
    annotate_distance_matches,
)
from revision.inference import fit_edge


def test_path_three_times_and_both_control_histories():
    names = ["temperature", "humidity", "wind", "cloud_cover"]
    data = np.arange(40)[:, None, None] * 10000 + np.arange(3)[None, :, None] * 100 + np.arange(4)[None, None, :]
    result = path_design(data, names, i=0, h=1, j=2, lag1=3, lag2=2)
    t = result["t"]
    np.testing.assert_array_equal(result["source"], data[t - 5, 0, 2])
    np.testing.assert_array_equal(result["mediator"], data[t - 2, 1, 1])
    np.testing.assert_array_equal(result["target"], data[t, 2, 3])
    np.testing.assert_array_equal(result["mediator_controls"][:, 0], data[t - 3, 1, 1])
    np.testing.assert_array_equal(result["mediator_controls"][:, 2], data[t - 5, 1, 1])
    np.testing.assert_array_equal(result["target_controls"][:, 0], data[t - 1, 2, 3])
    assert np.all(result["source_time"] < result["mediator_time"])
    assert np.all(result["mediator_time"] < result["t"])


def test_initial_bearing_and_vector_projection_cardinal_directions():
    np.testing.assert_allclose(initial_bearing(0, 0, 0, 90), np.pi / 2)
    np.testing.assert_allclose(initial_bearing(0, 0, 60, 0), 0.)
    np.testing.assert_allclose(initial_bearing(0, 0, 0, -90), -np.pi / 2)
    east, north = np.array([3., 4.]), np.array([8., 9.])
    np.testing.assert_allclose(project_east_north(east, north, np.pi / 2), east)
    np.testing.assert_allclose(project_east_north(east, north, 0.), north)


def test_joint_lag_coefficient_matches_single_full_statsmodels_model():
    rng = np.random.default_rng(19)
    n = 800
    data = rng.normal(size=(n, 2, 4))
    names = ["temperature", "humidity", "wind", "cloud_cover"]
    for t in range(12, n):
        data[t, 1, 1] = .6 * data[t - 1, 1, 1] + .5 * data[t - 2, 0, 2] - .3 * data[t - 9, 0, 2] + rng.normal()
    t, y, own, sources = joint_lag_design(data, names, 0, 1, "wind", "humidity")
    full = sm.OLS(y, sm.add_constant(np.column_stack((own, sources)))).fit(cov_type="HAC", cov_kwds={"maxlags": 64, "use_correction": True})
    for lag in (1, 2, 9, 12):
        controls = np.column_stack((own, np.delete(sources, lag - 1, axis=1)))
        result = fit_edge(y, controls, sources[:, lag - 1], bandwidths=(64,))
        np.testing.assert_allclose(result["effect"], full.params[3 + lag], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result["se_hac64"], full.bse[3 + lag], rtol=1e-10, atol=1e-10)


def test_projection_preprocessing_ignores_evaluation_changes():
    months = np.tile(np.arange(1, 13), 10)
    train = np.arange(120) < 72
    rng = np.random.default_rng(12)
    values = months.astype(float) + rng.normal(size=120)
    first, params = train_monthly_transform(values, months, train)
    changed = values.copy()
    changed[~train] += 500
    second, other_params = train_monthly_transform(changed, months, train)
    assert params == other_params
    np.testing.assert_array_equal(first[train], second[train])


def test_distance_calipers_require_both_legs_and_allow_no_match():
    distances = np.array([[0., 1000., 1900., 1100.], [1000., 0., 900., 1600.],
                          [1900., 900., 0., 950.], [1100., 1600., 950., 0.]])
    paths = pd.DataFrame([{"i": 0, "h": 1, "j": 2, "alternative_h": 3},
                          {"i": 0, "h": 2, "j": 2, "alternative_h": 1}])
    marked = annotate_distance_matches(paths, distances)
    assert marked.valid_distance_match.tolist() == [True, False]
    assert marked.distance_match_status.tolist() == ["valid_distance_match", "no_adequate_match"]
