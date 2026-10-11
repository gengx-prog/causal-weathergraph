import numpy as np

from revision.run_latitude_contrasts import latitude_weights, contrast_hac


def test_fixed_projection_matches_ols_with_unequal_ring_populations():
    latitude = np.array([15., 15., 45., 75., 75., 75.])
    values = np.array([.1, .4, -.2, .6, .9, 1.3])
    rows = {r["statistic"]: r for r in latitude_weights(latitude)}
    direct = np.linalg.lstsq(np.column_stack([np.ones(6), latitude / 10]), values, rcond=None)[0][1]
    assert np.isclose(rows["linear_trend_per_10deg"]["weights"] @ values, direct)
    assert np.isclose(rows["mean_0_30"]["weights"] @ values, .25)
    assert np.isclose(rows["difference_60_90_minus_0_30"]["weights"] @ values, values[3:].mean() - .25)
    assert abs(rows["linear_trend_per_10deg"]["weights"].sum()) < 1e-15


def test_empty_bands_zero_variation_and_exact_band_boundaries_are_explicit():
    rows = {r["statistic"]: r for r in latitude_weights(np.array([45., 45.]))}
    assert not rows["linear_trend_per_10deg"]["valid"]
    assert not rows["mean_0_30"]["valid"]
    assert not rows["difference_60_90_minus_30_60"]["valid"]
    assert rows["mean_30_60"]["valid"]
    rows = {r["statistic"]: r for r in latitude_weights(np.array([0., 30., 60., 90.]))}
    assert rows["mean_0_30"]["candidate_count"] == 1
    assert rows["mean_30_60"]["candidate_count"] == 1
    assert rows["mean_60_90"]["candidate_count"] == 2


def test_weighted_same_calendar_scores_keep_cross_edge_and_hemisphere_covariance():
    rng = np.random.default_rng(904)
    score = rng.normal(size=800)
    score[200:450] = 0  # Preserve a real calendar gap.
    # Perfectly shared within-band edge noise must not shrink as independent edges.
    repeated = np.column_stack([score, score])
    aggregate = repeated @ np.array([.5, .5])
    for b in [64, 128]:
        expected = aggregate @ aggregate + 2 * sum((1-h/(b+1)) * (aggregate[h:] @ aggregate[:-h]) for h in range(1, b+1))
        got = contrast_hac(.2, aggregate, b)[f"se_hac{b}"]**2
        assert np.isclose(got, expected, rtol=1e-12)
        same = contrast_hac(.2, score, b)[f"se_hac{b}"]
        assert np.isclose(np.sqrt(got), same)
        cancelled = contrast_hac(0., aggregate - score, b)[f"se_hac{b}"]
        assert cancelled == 0.
