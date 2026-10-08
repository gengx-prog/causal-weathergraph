import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac,cov_cluster
from revision.run_full_var import shared_qr_fit,score_inference,frozen_ablation_gain


def test_shared_fit_scores_and_covariance_against_statsmodels():
    rng=np.random.default_rng(89)
    x=sm.add_constant(rng.normal(size=(350,8)))
    y=x@rng.normal(size=(9,3))+rng.normal(size=(350,3))
    cols=[1,4,8]
    fit=shared_qr_fit(x,y,cols)
    expected=np.linalg.lstsq(x,y,rcond=None)[0]
    np.testing.assert_allclose(fit['coefficients'],expected,atol=1e-12)
    np.testing.assert_allclose(fit['directions'],x@np.linalg.inv(x.T@x)[:,cols],atol=1e-12)
    target=2
    score=fit['directions']*fit['residual'][:,target,None]
    se,p,(cluster_se,cluster_p,g,corr)=score_inference(score,fit['coefficients'][cols,target],9,calendar_offset=3,bandwidth=12,block_length=32)
    independent=sm.OLS(y[:,target],x).fit()
    np.testing.assert_allclose(se,np.sqrt(np.diag(cov_hac(independent,nlags=12,use_correction=True)))[cols],rtol=1e-10)
    np.testing.assert_allclose(cluster_se,np.sqrt(np.diag(cov_cluster(independent,(np.arange(350)+3)//32,use_correction=True)))[cols],rtol=1e-10)


def test_frozen_drop_one_matches_explicit_restricted_training_refit():
    rng=np.random.default_rng(45)
    train=sm.add_constant(rng.normal(size=(200,6)))
    test=sm.add_constant(rng.normal(size=(70,6)))
    beta=rng.normal(size=(7,2))
    ytrain=train@beta+rng.normal(size=(200,2))
    ytest=test@beta+rng.normal(size=(70,2))
    fitted=shared_qr_fit(train,ytrain,[1,2,3])
    for j in (1,2,3):
        gain,full,restricted=frozen_ablation_gain(test,ytest,fitted['coefficients'],fitted['inverse_gram'],j,1)
        keep=np.arange(7)!=j
        restricted_beta=np.linalg.lstsq(train[:,keep],ytrain[:,1],rcond=None)[0]
        independent=ytest[:,1]-test[:,keep]@restricted_beta
        np.testing.assert_allclose(restricted,independent,atol=1e-12)
        np.testing.assert_allclose(gain,independent**2-full**2,atol=1e-12)


def test_rank_failure_is_explicit():
    rng=np.random.default_rng(4)
    z=rng.normal(size=100)
    x=np.column_stack([np.ones(100),z,z])
    with pytest.raises(ValueError,match='Rank-deficient'):
        shared_qr_fit(x,rng.normal(size=(100,2)),[1])


def test_source_lags_fixed_when_control_history_expands():
    from causal_weathergraph.candidate_edges import CandidateEdge
    from revision.run_full_var import make_design
    data=np.arange(30*2*2,dtype=float).reshape(30,2,2)
    candidates=[CandidateEdge(0,1,'wind','humidity','wind_to_humidity',1.)]
    x3,y3,_,s3=make_design(data,['wind','humidity'],candidates,3,3,12)
    x12,y12,_,s12=make_design(data,['wind','humidity'],candidates,12,3,12)
    assert x3.shape==(18,13)
    assert x12.shape==(18,49)
    np.testing.assert_array_equal(y3,y12)
    np.testing.assert_array_equal(x3,x12[:,:13])
    assert len(s3)==len(s12)==3
    np.testing.assert_array_equal(s3.column,s12.column)
    for row in s12.itertuples():
        np.testing.assert_array_equal(x12[:,row.column],data[12-row.lag:30-row.lag,0,0])
