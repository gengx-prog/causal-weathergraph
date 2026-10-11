import numpy as np

from revision.run_hemisphere_contrasts import (
    fit_pair_batch, hac_statistics, interaction_influence, local_season_masks, mirrored_pairs,
)
from causal_weathergraph.candidate_edges import CandidateEdge
from revision.run_regime_contrasts import contrast_batch


def direct_fit_and_score(y, c, x, winter, mask):
    """Independent full-design inverse, not the production FWL path."""
    design = np.column_stack((np.ones(len(y)), c, winter, c * winter[:, None], x, x * winter))
    selected = design[mask]
    inverse = np.linalg.inv(selected.T @ selected)
    coef = inverse @ selected.T @ y[mask]
    residual = y[mask] - selected @ coef
    score = np.zeros(len(y))
    correction = mask.sum() / (mask.sum() - selected.shape[1])
    score[mask] = (selected @ inverse[:, -1]) * residual * np.sqrt(correction)
    return coef[-1], score


def direct_bartlett_variance(score, bandwidth):
    return score @ score + 2 * sum(
        (1 - h / (bandwidth + 1)) * (score[h:] @ score[:-h]) for h in range(1, bandwidth + 1)
    )


def test_mirrored_intersection_deterministic_and_reports_both_exclusions():
    lat = np.array([-45., -45., 45., 45.])
    lon = np.array([0., 90., 0., 90.])
    def edge(source, target):
        return CandidateEdge(source, target, "wind", "humidity", "wind_to_humidity", 0.)
    candidates = [edge(0, 0), edge(2, 2), edge(1, 0), edge(2, 3)]
    pairs, excluded, reflection = mirrored_pairs(candidates, lat, lon)
    assert reflection == {0: 2, 1: 3, 2: 0, 3: 1}
    assert pairs == [(candidates[1], candidates[0])]
    assert {row["candidate_id"] for row in excluded} == {
        "r1:wind->r0:humidity", "r2:wind->r3:humidity"}
    assert {row["expected_mirror_id"] for row in excluded} == {
        "r3:wind->r2:humidity", "r0:wind->r1:humidity"}
    reversed_pairs, _, _ = mirrored_pairs(list(reversed(candidates)), lat, lon)
    assert reversed_pairs == pairs


def test_local_season_encoding_and_contrast_sign():
    timestamps = np.array(["2020-01-01", "2020-04-01", "2020-07-01", "2020-12-01"], dtype="datetime64[D]")
    north_winter, south_winter, eligible = local_season_masks(timestamps)
    np.testing.assert_array_equal(north_winter, [True, False, False, True])
    np.testing.assert_array_equal(south_winter, [False, False, True, False])
    np.testing.assert_array_equal(eligible, [True, False, True, True])
    rng = np.random.default_rng(33)
    n = 800
    x = rng.normal(size=n)
    c = rng.normal(size=(n, 3))
    winter = (np.arange(n) % 80) < 40
    y_north = .2*x + .6*x*winter + c @ np.array([.1, .2, .3])
    y_south = .1*x + .25*x*(~winter) + c @ np.array([.1, .2, .3])
    _, _, effect, _ = fit_pair_batch(y_north, c, x, y_south, c, x, winter, ~winter, np.ones(n, bool))
    np.testing.assert_allclose(effect, [.35], atol=1e-12)


def test_calendar_covariance_matches_full_models_with_correlated_hemispheres():
    rng = np.random.default_rng(155)
    n = 1800
    innovations = rng.normal(size=n)
    shared = innovations.copy()
    for t in range(1, n):
        shared[t] += .65*shared[t-1]
    x = rng.normal(size=n)
    c = rng.normal(size=(n, 3))
    winter = (np.arange(n) % 160) < 80
    mask = (np.arange(n) % 160 < 60) | (np.arange(n) % 160 >= 100)
    y_north = .2*x + .3*x*winter + c @ np.array([.1, .2, .3]) + shared
    y_south = .1*x + .2*x*(~winter) + c @ np.array([.1, .2, .3]) + shared + .1*rng.normal(size=n)
    north, south, effect, score = fit_pair_batch(y_north, c, x, y_south, c, x, winter, ~winter, mask)
    bn, sn = direct_fit_and_score(y_north, c, x, winter, mask)
    bs, ss = direct_fit_and_score(y_south, c, x, ~winter, mask)
    np.testing.assert_allclose(effect, [bn-bs], rtol=1e-11)
    np.testing.assert_allclose(score[:, 0], sn-ss, atol=1e-13)
    assert np.all(score[~mask] == 0)
    for bandwidth in (32, 64, 128):
        expected = direct_bartlett_variance(sn-ss, bandwidth)
        got = hac_statistics(effect, score, [bandwidth])[0]
        np.testing.assert_allclose(got[f"se_hac{bandwidth}"]**2, expected, rtol=1e-11)
    independent_variance = direct_bartlett_variance(sn, 64) + direct_bartlett_variance(ss, 64)
    assert direct_bartlett_variance(sn-ss, 64) > 1.8 * independent_variance
    old = contrast_batch(y_north, c, x, winter, mask)[0]
    new = hac_statistics(north["delta"], north["influence"], [64])[0]
    np.testing.assert_allclose(old["se_hac64"], new["se_hac64"], rtol=1e-12)
    np.testing.assert_allclose(old["delta_beta"], north["delta"][0], rtol=1e-12)


def test_average_scores_preserve_cross_edge_covariance():
    rng = np.random.default_rng(654)
    score = rng.normal(size=600)
    scores = np.column_stack((score, score))
    got = hac_statistics(.1, scores.mean(axis=1), [32])[0]["se_hac32"]
    single = hac_statistics(.1, score, [32])[0]["se_hac32"]
    np.testing.assert_allclose(got, single, rtol=1e-12)
    assert not np.isclose(got, single / np.sqrt(2))
