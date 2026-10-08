from dataclasses import replace

import numpy as np

from revision.simulations import (
    SimulationConfig, bh, candidate_edges, lagged_design, pcmci_pvalues,
    score_discoveries, simulate_graph, simulate_independent, wilson_interval,
)


def test_full_calendar_lags_do_not_join_regime_gaps():
    data = np.arange(24).reshape(12, 2)
    lagged = lagged_design(data, 3)
    selected = np.array([3, 7, 10])
    np.testing.assert_array_equal(lagged[selected, 0], data[selected - 1, 0])
    np.testing.assert_array_equal(lagged[selected, 4], data[selected - 3, 0])
    assert np.isnan(lagged[:3, 4]).all()


def test_direction_and_lag_are_required_for_true_positives():
    edges = [(0, 1, 1), (1, 0, 1), (0, 1, 2)]
    score = score_discoveries([False, True, True], {(0, 1, 1)}, edges)
    assert score["tp"] == 0 and score["fp"] == 2 and score["fn"] == 1
    assert score["shd"] == 3 and score["fdp"] == 1


def test_bh_matches_hand_calculation_and_keeps_invalid_in_family():
    found, q = bh([0.001, 0.02, 0.04, np.nan], 0.05)
    np.testing.assert_array_equal(found, [True, True, False, False])
    np.testing.assert_allclose(q, [0.004, 0.04, 0.04 * 4 / 3, 1])
    lo, hi = wilson_interval(0, 200)
    assert abs(lo) < 1e-10 and 0.018 < hi < 0.02


def test_generator_records_stable_exact_truth_and_observed_driver():
    config = replace(SimulationConfig(), n_graph=1200)
    for scenario in ("weak_ar", "strong_ar", "common_driver_observed", "common_driver_omitted", "known_external_regime_switching"):
        data, masks, truth, radii = simulate_graph(config, scenario, np.random.default_rng(23))
        assert data.shape == (1200, 7 if scenario == "common_driver_observed" else 6)
        assert len(truth[0]) == 6 and max(radii) < 1
        assert np.isfinite(data).all()
        expected = np.ones(1200)
        expected[:2 * config.max_lag] = 0
        np.testing.assert_array_equal(np.sum(list(masks.values()), axis=0), expected)
        if len(truth) == 2:
            assert truth[0].isdisjoint(truth[1])


def test_null_has_independent_columns_and_selection_is_distinct():
    config = replace(SimulationConfig(), n_null=10000)
    data, masks = simulate_independent(config, "ar1_homoskedastic", np.random.default_rng(12))
    correlations = np.corrcoef(data.T) - np.eye(6)
    assert np.max(np.abs(correlations)) < 0.12
    assert abs(masks["outcome_selected"].mean() - 0.2) < 0.002
    assert not np.array_equal(masks["outcome_selected"], masks["external_calendar"])


def test_official_pcmci_recovers_obvious_lag_and_not_reverse():
    config = replace(SimulationConfig(), n_nodes=2, max_lag=2)
    rng = np.random.default_rng(99)
    data = rng.normal(size=(3000, 2))
    data[1:, 1] += 0.85 * data[:-1, 0]
    edges = candidate_edges(2, 2)
    p, diagnostics = pcmci_pvalues(data, np.ones(len(data), dtype=bool), edges, config)
    found, _ = bh(p)
    assert found[edges.index((0, 1, 1))]
    assert not found[edges.index((1, 0, 1))]
    assert diagnostics["invalid_tests"] == 0


def test_latent_regime_label_alignment_uses_only_state_agreement():
    from revision.run_rpcmci_benchmark import align_two_states
    true = np.array([0, 0, 1, 1, 0, 1])
    gamma = np.stack([true, 1 - true])
    mapping, accuracy, aligned = align_two_states(gamma, true, np.ones(6, dtype=bool))
    assert mapping == {0: 1, 1: 0} and accuracy == 1
    np.testing.assert_array_equal(aligned, true)


def test_official_iteration_tolerance_is_zero_indexed():
    from revision.run_rpcmci_benchmark import reached_official_tolerance
    assert reached_official_tolerance([400, 300, 200, 150, 100, 40], 2048)
    assert not reached_official_tolerance([400, 300, 200, 150, 40], 2048)
    assert reached_official_tolerance([400, 0], 2048)
