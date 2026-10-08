import numpy as np
from revision.prepare_inputs import fit_transform_aggregate, region_map


def test_transform_fit_does_not_use_holdout():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(480, 6))
    months = np.tile(np.arange(1,13),40)
    train = np.arange(480)<360
    mapping = np.array([0,0,0,1,1,1])
    first, stats = fit_transform_aggregate(x.copy(), months, train, mapping)
    before = x.copy()
    fit_transform_aggregate(x, months, train, mapping)
    np.testing.assert_array_equal(x, before)
    changed = x.copy(); changed[~train] += 10000
    second, stats2 = fit_transform_aggregate(changed, months, train, mapping)
    for k in stats:
        np.testing.assert_array_equal(stats[k],stats2[k])
    np.testing.assert_array_equal(first[train],second[train])
    assert np.max(np.abs(first[~train]-second[~train]))>1000


def test_original_regional_grid():
    mapping, lat, lon = region_map(np.linspace(-87.1875,87.1875,32),np.arange(64)*5.625)
    assert len(mapping)==2048
    assert len(lat)==len(lon)==66
    assert np.isfinite(lat).all() and np.isfinite(lon).all()
