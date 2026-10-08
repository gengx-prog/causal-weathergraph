"""Scientific input contracts and integration checks for the local estimator.

These checks use cardinal meteorological directions as external expectations,
not a second implementation of the production conversion formula.  Artifact
checks below are only run once the independently trained model is available.
"""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from revision.cloud_simulator import app


def valid_payload(**changes):
    value = {"wind_input": "uv", "u_mps": 2.0, "v_mps": -3.0,
             "humidity_gkg": 4.0, "mode": "wind_humidity", "horizon_hours": 0}
    value.update(changes)
    return value


@pytest.mark.parametrize("direction,expected", [
    (0, (0, -10)), (90, (-10, 0)), (180, (0, 10)),
    (270, (10, 0)), (360, (0, -10)),
])
def test_meteorological_from_direction_cardinal_vectors(direction, expected):
    result = app.normalize_payload(valid_payload(
        wind_input="speed_direction", speed_mps=10, direction_deg=direction))
    np.testing.assert_allclose([result["u_mps"], result["v_mps"]], expected,
                               atol=1e-12, rtol=0)
    assert result["humidity_gkg"] == 4.0  # conversion to kg/kg belongs to engine


def test_calm_wind_is_zero_at_every_direction():
    for direction in (0, 37, 90, 180, 273, 360):
        result = app.normalize_payload(valid_payload(
            wind_input="speed_direction", speed_mps=0, direction_deg=direction))
        assert result["u_mps"] == 0 and result["v_mps"] == 0


@pytest.mark.parametrize("changes", [
    {"u_mps": True}, {"v_mps": None}, {"humidity_gkg": "4"},
    {"u_mps": float("nan")}, {"v_mps": float("inf")},
    {"humidity_gkg": -0.001}, {"month": 1.5}, {"month": True},
    {"region_id": 2.5}, {"horizon_hours": 0.5},
    {"wind_input": "unknown"},
    {"wind_input": "speed_direction", "speed_mps": -1, "direction_deg": 0},
    {"wind_input": "speed_direction", "speed_mps": 3, "direction_deg": 361},
    {"wind_input": "speed_direction", "speed_mps": 3, "direction_deg": -1},
])
def test_invalid_numeric_inputs_are_rejected_before_prediction(changes):
    with pytest.raises(ValueError):
        app.normalize_payload(valid_payload(**changes))


def test_normalization_does_not_mutate_callers_input():
    value = valid_payload(wind_input="speed_direction", speed_mps=10, direction_deg=90)
    original = dict(value)
    converted = app.normalize_payload(value)
    assert value == original
    assert converted["u_mps"] == -10


def http_json(server, path, payload=None):
    address = f"http://127.0.0.1:{server.server_address[1]}{path}"
    if payload is None:
        request = Request(address)
    else:
        request = Request(address, data=json.dumps(payload).encode("utf8"),
                          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.fixture(scope="module")
def actual_estimator():
    artifacts = Path(app.DEFAULT_ARTIFACTS)
    if not (artifacts / "manifest.json").exists():
        pytest.skip("The separately trained simulator artifacts are not yet available")
    from revision.cloud_simulator.engine import CloudEstimator
    return CloudEstimator(artifacts)


@pytest.fixture(scope="module")
def actual_server(actual_estimator):
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(actual_estimator))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()


def test_real_http_prediction_matches_saved_model_engine(actual_server, actual_estimator):
    for payload in (valid_payload(),
                    valid_payload(mode="context", region_id=2, month=1),
                    valid_payload(wind_input="speed_direction", speed_mps=10,
                                  direction_deg=270, mode="context", region_id=63, month=7)):
        direct = actual_estimator.predict(app.normalize_payload(payload))
        status, response = http_json(actual_server, "/api/predict", payload)
        assert status == 200
        # Same genuine model, with only HTTP receipt fields appended by server.
        returned = {key: value for key, value in response.items()
                    if key not in ("input", "generated_utc")}
        assert returned == direct
    status, health = http_json(actual_server, "/api/health")
    assert status == 200 and health["status"] == "ready"


@pytest.mark.parametrize("changes", [
    {"humidity_gkg": -1}, {"humidity_gkg": None}, {"u_mps": float("nan")},
    {"mode": "context", "region_id": 999, "month": 1},
    {"mode": "context", "region_id": 2, "month": 13},
    {"mode": "context", "region_id": 2, "month": 0},
    {"horizon_hours": 24},
])
def test_real_http_rejects_invalid_or_untrained_scenarios(actual_server, changes):
    status, response = http_json(actual_server, "/api/predict", valid_payload(**changes))
    assert status == 400 and isinstance(response["error"], str)


def test_conformal_rank_uses_held_out_residual_order_statistic():
    from revision.cloud_simulator.engine import conformal_expansion
    # Nine residuals in deliberately unsorted order. For 80%, ceil(10*.8)=8,
    # so the correct one-based order statistic is .8, not a interpolated .74.
    observed = np.array([.9, .1, .5, .3, .8, .2, .7, .4, .6])
    correction, receipt = conformal_expansion(observed, np.zeros(9), np.zeros(9))
    assert correction == pytest.approx(.8)
    assert receipt["rank_one_based"] == 8
    assert receipt["n"] == 9


def test_crossed_quantiles_and_nonnegative_expansion_have_expected_meaning():
    from revision.cloud_simulator.engine import conformal_expansion
    # Every observation lies in [.2,.8], with both orientations represented.
    observed = np.array([.2, .3, .4, .5, .6, .7, .8, .4, .6])
    lo = np.array([.8, .2, .8, .2, .8, .2, .8, .2, .8])
    hi = 1 - lo
    correction, _ = conformal_expansion(observed, lo, hi)
    assert correction == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize("observed,lower,upper", [
    ([.5], [.2], [.8]), ([.5] * 2, [.2] * 2, [.8] * 2),
    ([.5] * 3, [.2] * 3, [.8] * 3),
    ([.5] * 5, [.2] * 4, [.8] * 5),
    ([.5] * 4 + [float("nan")], [.2] * 5, [.8] * 5),
    ([[.5]] * 5, [[.2]] * 5, [[.8]] * 5),
])
def test_conformal_rejects_insufficient_or_invalid_calibration(observed, lower, upper):
    from revision.cloud_simulator.engine import conformal_expansion
    # With fewer than four observations, an 80% finite order statistic does
    # not exist: silently capping its rank would change the method.
    with pytest.raises(ValueError):
        conformal_expansion(observed, lower, upper)


@pytest.fixture(scope="module")
def actual_artifacts(actual_estimator):
    directory = Path(app.DEFAULT_ARTIFACTS)
    split = np.load(directory / "split_indices.npz")
    predictions = np.load(directory / "predictions.npz")
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf8"))
    try:
        yield directory, split, predictions, protocol
    finally:
        split.close()
        predictions.close()


def test_saved_time_partitions_exclude_target_and_acquisition_boundaries(actual_artifacts):
    _, split, _, protocol = actual_artifacts
    times = split["timestamps"].astype("datetime64[ns]")
    bounds = {"train": ("1979-01-01", "2015-01-01"),
              "val": ("2015-01-01", "2019-01-01"),
              "test": ("2019-01-01", "2026-01-01")}
    for name, (start, end) in bounds.items():
        index = split[f"{name}_indices"]
        t, _ = index.T
        assert len(np.unique(index, axis=0)) == len(index)
        assert np.all(times[t] >= np.datetime64(start))
        assert np.all(times[t + 4] < np.datetime64(end))
        assert np.all(times[t].astype("datetime64[h]").astype(np.int64) % 24 == 0)
        assert t.min() >= 8
        for seam in ("2019-01-01", "2023-01-11"):
            crosses = (times[t - 8] < np.datetime64(seam)) & (times[t + 4] >= np.datetime64(seam))
            assert not crosses.any()
    assert protocol["horizons_hours"] == [0]
    assert protocol["fixed_hyperparameters"]["early_stopping"] is False


def test_saved_cqr_is_computed_from_validation_only(actual_artifacts, actual_estimator):
    _, split, predictions, _ = actual_artifacts
    metadata = actual_estimator.metadata()
    y = predictions["observed_val_h0"]
    assert len(y) == len(split["val_indices"])
    assert len(y) != len(split["test_indices"])
    for mode in ("wind_humidity", "context"):
        key = f"{mode}_h0"
        q10 = predictions[f"{key}_raw_q10_val"]
        q90 = predictions[f"{key}_raw_q90_val"]
        # Independent Python sort rather than production partition routine.
        scores = [max(min(a, b) - target, target - max(a, b), 0.)
                  for target, a, b in zip(y, q10, q90)]
        rank = int(np.ceil((len(scores) + 1) * .8))
        expected = sorted(scores)[rank - 1]
        receipt = metadata["calibration"][key]
        assert receipt["n"] == len(split["val_indices"])
        assert receipt["rank_one_based"] == rank
        assert receipt["nonnegative_expansion"] == expected
        assert receipt["score_sha256"] == hashlib.sha256(
            np.asarray(scores, dtype=np.float64).tobytes()).hexdigest()
        # The identical validation-derived correction is applied to the test
        # predictions; no test residual quantile is used here.
        low = np.minimum(predictions[f"{key}_raw_q10_test"],
                         predictions[f"{key}_raw_q90_test"])
        high = np.maximum(predictions[f"{key}_raw_q10_test"],
                          predictions[f"{key}_raw_q90_test"])
        np.testing.assert_allclose(predictions[f"{key}_lower_test"],
                                   np.clip(low - expected, 0, 1), atol=1e-14, rtol=0)
        np.testing.assert_allclose(predictions[f"{key}_upper_test"],
                                   np.clip(high + expected, 0, 1), atol=1e-14, rtol=0)


def test_saved_feature_receipts_exclude_cloud_and_match_source_units(actual_artifacts):
    from revision.weather_tokens.data import load_raw
    _, split, predictions, protocol = actual_artifacts
    # The source reader was separately audited. Here we independently assemble
    # the feature columns, dates, and labels, without calling engine helpers.
    source = load_raw()
    raw, times = source["raw"], source["timestamps"]
    nodes = source["node_ids"]
    np.testing.assert_array_equal(times, split["timestamps"])
    for name in ("train", "val", "test"):
        t, r = split[f"{name}_indices"].T
        u, v, q = (raw[t, r, column].astype(np.float64) for column in (0, 1, 2))
        month = times[t].astype("datetime64[M]").astype(np.int64) % 12 + 1
        angle = 2 * np.pi * (month - 1) / 12
        designs = {"wind_humidity": np.column_stack([u, v, q]),
                   "context": np.column_stack([u, v, q, nodes[r], np.sin(angle), np.cos(angle)])}
        receipt = protocol["fingerprints"][name]
        for mode, matrix in designs.items():
            assert receipt["features"][mode]["shape"] == list(matrix.shape)
            assert receipt["features"][mode]["sha256"] == hashlib.sha256(matrix.tobytes()).hexdigest()
        observed = raw[t, r, 5].astype(np.float64)
        assert receipt["targets"]["0"]["sha256"] == hashlib.sha256(observed.tobytes()).hexdigest()
        if name != "train":
            np.testing.assert_array_equal(observed, predictions[f"observed_{name}_h0"])


def test_humidity_conversion_occurs_once_in_real_prediction(actual_estimator):
    from revision.cloud_simulator.engine import _model_prediction
    # Directly pass kg/kg to the fitted 3-feature model, bypassing request
    # conversion and feature_matrix. Public inference receives g/kg instead.
    item = actual_estimator._bundle["estimators"]["wind_humidity_h0"]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        point, lower, upper, _ = _model_prediction(
            item["models"], np.array([[2., -3., .004]]), item["correction"])
    public = actual_estimator.predict(valid_payload())
    np.testing.assert_allclose([public["point"], public["lower"], public["upper"]],
                               100 * np.array([point[0], lower[0], upper[0]]), atol=1e-12, rtol=0)
    assert public["support"]["checks"]["humidity_gkg"]["value"] == 4.


def test_real_http_sweep_changes_only_humidity(actual_server, actual_estimator):
    payload = valid_payload(mode="context", region_id=24, month=8,
                            sweep_min_gkg=1., sweep_max_gkg=13.)
    status, response = http_json(actual_server, "/api/sweep", payload)
    assert status == 200 and len(response["points"]) == 25
    humidity = np.array([point["humidity_gkg"] for point in response["points"]])
    np.testing.assert_allclose(humidity, np.linspace(1, 13, 25), atol=0, rtol=0)
    for index in (0, 12, 24):
        direct = actual_estimator.predict(valid_payload(
            humidity_gkg=float(humidity[index]), mode="context", region_id=24, month=8))
        returned = {key: value for key, value in response["points"][index].items()
                    if key != "humidity_gkg"}
        assert returned == direct
