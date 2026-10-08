import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from revision.run_regime_contrasts import contrast_batch,state_definitions


def test_interaction_matches_full_model():
    rng=np.random.default_rng(4);n=1000
    c=rng.normal(size=(n,3));x=rng.normal(size=n);s=(np.arange(n)%100)<40
    y=.3*x+.6*x*s+c@np.array([.1,.2,.4])+rng.normal(size=n)
    got=contrast_batch(y,c,x,s,np.ones(n,bool))[0]
    X=np.column_stack([np.ones(n),c,s,c*s[:,None],x,x*s])
    ref=sm.OLS(y,X).fit();se=np.sqrt(cov_hac(ref,nlags=64)[-1,-1])
    np.testing.assert_allclose(got['delta_beta'],ref.params[-1],rtol=1e-10)
    np.testing.assert_allclose(got['se_hac64'],se,rtol=1e-10)


def test_state_threshold_and_antecedence():
    rng=np.random.default_rng(14);n=500
    data=rng.normal(size=(n,2,4));names=['temperature','humidity','wind','cloud_cover']
    ts=np.datetime64('2018-11-01')+np.arange(n)*np.timedelta64(6,'h');train=np.arange(n)<350
    a,thresholds=state_definitions(data,names,ts,train)
    other=data.copy();other[~train]+=100
    _,thresholds2=state_definitions(other,names,ts,train)
    assert thresholds==thresholds2
    d={name:(high,eligible) for name,high,eligible,_ in a}
    for field in ['humidity','cloud']:
        assert not d[field+'_lag4'][1][:4].any()
        np.testing.assert_array_equal(d[field+'_lag4'][0][4:],d[field+'_lag0'][0][:-4])
