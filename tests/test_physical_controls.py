import numpy as np
from revision.run_physical_controls import analyze_sources, physical_history, evaluation_periods


def test_fwl_single_source_losses_match_direct_augmented_ols():
    rng = np.random.default_rng(20260930)
    c = rng.normal(size=(1200, 4))
    z = rng.normal(size=(1200, 3)) + c[:, :3] * .6
    y = .5 * c[:, 0] + .2 * z[:, 0] + rng.normal(size=1200)
    tr = np.arange(1200) < 800
    te = ~tr
    fits, loss = analyze_sources(y, c, z, tr, {'test': te}, te)
    base = np.column_stack((np.ones(1200), c))
    br = np.linalg.lstsq(base[tr], y[tr], rcond=None)[0]
    for k in range(3):
        full = np.column_stack((base, z[:, k]))
        bf = np.linalg.lstsq(full[tr], y[tr], rcond=None)[0]
        np.testing.assert_allclose(fits['training'][k]['effect'], bf[-1], atol=1e-13)
        expected = np.mean((y[te] - base[te] @ br)**2 - (y[te] - full[te] @ bf)**2)
        np.testing.assert_allclose(loss['test'][k]['mse_gain'], expected, atol=1e-13)
        np.testing.assert_allclose(loss['test'][k]['delta_r2_test_centered'], expected / np.var(y[te]), atol=1e-13)


def test_evaluation_perturbation_cannot_change_training_fit():
    rng = np.random.default_rng(4)
    c, z, y = rng.normal(size=(500, 3)), rng.normal(size=(500, 2)), rng.normal(size=500)
    tr = np.arange(500) < 350
    first, _ = analyze_sources(y, c, z, tr, {'test': ~tr}, ~tr)
    y[~tr] += 1000
    c[~tr] *= 17
    z[~tr] -= 100
    second, _ = analyze_sources(y, c, z, tr, {'test': ~tr}, ~tr)
    for left, right in zip(first['training'], second['training']):
        assert left['effect'] == right['effect']
        assert left['se_hac64'] == right['se_hac64']


def test_duplicate_control_columns_preserve_span_and_effects():
    rng = np.random.default_rng(7)
    c, z = rng.normal(size=(700, 3)), rng.normal(size=700)
    y = .4 * z + c[:, 0] + rng.normal(size=700)
    tr = np.arange(700) < 500
    f1, p1 = analyze_sources(y, c, z, tr, {'test': ~tr})
    f2, p2 = analyze_sources(y, np.column_stack((c, c[:, 0], c[:, 1] * 2)), z, tr, {'test': ~tr})
    np.testing.assert_allclose(f1['training'][0]['effect'], f2['training'][0]['effect'], atol=1e-13)
    assert f1['training'][0]['rank_full'] == f2['training'][0]['rank_full']
    np.testing.assert_allclose(p1['test'][0]['mse_gain'], p2['test'][0]['mse_gain'], atol=1e-13)


def test_control_indexing_and_unique_regions():
    physical = np.arange(20 * 4 * 6).reshape(20, 4, 6)
    t = np.arange(9, 15)
    actual = physical_history(physical, t, [2, 1, 2], (4, 5, 6))
    assert actual.shape == (6, 36)
    expected = np.column_stack([physical[t-k, r, f] for r in [1, 2] for f in range(6) for k in [4, 5, 6]])
    np.testing.assert_array_equal(actual, expected)
    assert np.all(t[:, None] - np.array([4, 5, 6]) < t[:, None] - 3)


def test_segment_masks_exclude_any_boundary_crossing_inputs():
    ts = np.arange(np.datetime64('2023-01-08T00'), np.datetime64('2023-01-14T00'), np.timedelta64(6, 'h'))
    t = np.arange(6, len(ts)-3)
    m = evaluation_periods(ts[t], ts[t-6], ts[t+3])
    assert np.all(ts[t+3][m['evaluation_wb2']] < np.datetime64('2023-01-11'))
    assert np.all(ts[t-6][m['evaluation_cds']] >= np.datetime64('2023-01-11'))
    assert not np.any(m['evaluation_wb2'] & m['evaluation_cds'])
    assert m['evaluation_all'].sum() > m['evaluation_wb2'].sum() + m['evaluation_cds'].sum()
