import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy import stats

from revision.inference import fit_edge, score_autocovariances, sample_calendar_design_blocks, bootstrap_edge_effects


def sample(seed=2,n=400):
    rng=np.random.default_rng(seed)
    controls=rng.normal(size=(n,3))
    z=rng.normal(size=n)
    errors=np.empty(n)
    errors[0]=0
    for t in range(1,n):
        errors[t]=.7*errors[t-1]+rng.normal()
    return controls@np.array([.3,-.1,.7])+.22*z+errors,controls,z


def test_contiguous_hac_matches_statsmodels_and_ols_f():
    y,c,z=sample()
    answer=fit_edge(y,c,z,np.ones(len(y),dtype=bool),bandwidths=(0,4,12))
    fit=sm.OLS(y,sm.add_constant(np.column_stack([c,z]))).fit()
    assert answer['effect']==pytest.approx(fit.params[-1],rel=1e-11)
    assert answer['p_ols']==pytest.approx(fit.pvalues[-1],rel=1e-9)
    restricted=sm.OLS(y,sm.add_constant(c)).fit()
    assert answer['F_statistic']==pytest.approx(fit.compare_f_test(restricted)[0],rel=1e-10)
    for b in (0,4,12):
        se=np.sqrt(cov_hac(fit,nlags=b,use_correction=True)[-1,-1])
        assert answer[f'se_hac{b}']==pytest.approx(se,rel=1e-11)
        assert answer[f'p_hac{b}']==pytest.approx(2*stats.norm.sf(abs(fit.params[-1])/se),rel=1e-11)


def test_regime_mask_uses_actual_calendar_gaps_against_bruteforce():
    y,c,z=sample(n=500)
    mask=np.arange(len(y))%9<4
    result=fit_edge(y,c,z,mask,bandwidths=(12,))
    base=sm.add_constant(c[mask])
    zr=z[mask]-base@np.linalg.lstsq(base,z[mask],rcond=None)[0]
    fit=sm.OLS(y[mask],np.column_stack([base,z[mask]])).fit()
    scores=np.zeros(len(y)); scores[mask]=zr*fit.resid
    meat=scores@scores+2*sum((1-h/13)*(scores[h:]@scores[:-h]) for h in range(1,13))
    expected=np.sqrt(meat*mask.sum()/(mask.sum()-5))/(zr@zr)
    assert result['se_hac12']==pytest.approx(expected,rel=1e-10)
    compressed=np.sqrt(cov_hac(fit,nlags=12,use_correction=True)[-1,-1])
    assert abs(compressed-expected)>1e-4


def test_fft_products_match_direct_and_do_not_wrap():
    scores=np.zeros(30); scores[[0,7,29]]=[3,2,4]
    a=score_autocovariances(scores,12)[:,0]
    assert a[0]==pytest.approx(scores@scores)
    for lag in range(1,13):
        assert a[lag]==pytest.approx(scores[lag:]@scores[:-lag],abs=1e-12)


def test_singular_and_degenerate():
    y,c,z=sample()
    normal=fit_edge(y,c,z)
    singular=fit_edge(y,np.column_stack([c,c[:,0]]),z)
    assert singular['effect']==pytest.approx(normal['effect'])
    assert singular['se_hac64']==pytest.approx(normal['se_hac64'])
    assert not fit_edge(y,c,c[:,0])['valid']
    assert fit_edge(y,c,c[:,0])['p_hac64']==1
    assert not fit_edge(np.ones(len(y)),c,z)['valid']
    assert not fit_edge(y,c,z,np.zeros(len(y),dtype=bool))['valid']


def test_missing_rows_preserve_calendar_and_complete_cases():
    y,c,z=sample()
    mask=np.ones(len(y),dtype=bool); mask[[9,10,42]]=False
    expected=fit_edge(y,c,z,mask)
    y[[9,10]]=np.nan; c[42,0]=np.nan
    found=fit_edge(y,c,z)
    assert found['se_hac64']==pytest.approx(expected['se_hac64'])


def test_bootstrap_original_design_blocks_keep_lags_and_gaps():
    n=100; rng=np.random.default_rng(55)
    original=np.arange(n)
    design=np.column_stack([original[3:],original[2:-1],original[1:-2],original[:-3]])
    sampled=sample_calendar_design_blocks(len(design),11,rng)
    boot=design[sampled]
    assert np.all(boot[:,0]-boot[:,3]==3)
    for start in range(0,len(sampled),11):
        assert np.all(np.diff(sampled[start:start+11])==1)
    regime=original[3:]%5==0
    assert np.all(np.diff(boot[regime[sampled],:],axis=1)==-1)


def test_bootstrap_matches_independent_original_design_row_calculation():
    y,c,z=sample(n=200)
    mask=np.arange(len(y))%7<3
    answer=bootstrap_edge_effects(y,c,z,mask,block_length=17,n_bootstrap=5,seed=12)
    rng=np.random.default_rng(12)
    expected=[]
    for _ in range(5):
        starts=rng.integers(0,200-17+1,size=int(np.ceil(200/17)))
        rows=(starts[:,None]+np.arange(17)).ravel()[:200]
        rows=rows[mask[rows]]
        x=np.column_stack([np.ones(len(rows)),c[rows],z[rows]])
        expected.append(np.linalg.lstsq(x,y[rows],rcond=None)[0][-1])
    np.testing.assert_allclose(answer['effects'],expected,rtol=1e-12)
    assert answer['n_valid']==5


@pytest.mark.parametrize('all_vars',[False,True])
def test_graph_batch_matches_single_edge_and_legacy_ols(all_vars):
    from causal_weathergraph.candidate_edges import CandidateEdge
    from causal_weathergraph.causal.fast_granger import run_fast_granger_discovery
    from revision.inference import run_graph_discovery
    rng=np.random.default_rng(91)
    data=rng.normal(size=(350,2,2))
    data[1:,1,1]+=.5*data[:-1,0,0]
    names=['wind','humidity']
    candidates=[CandidateEdge(0,1,'wind','humidity','wind_to_humidity',10.),
                CandidateEdge(1,1,'wind','humidity','wind_to_humidity',0.)]
    regimes={'all':np.ones(350,dtype=bool),'state':np.arange(350)%7<4}
    controls={'include_target_own_lags':True,'include_target_region_all_vars':all_vars}
    answer=run_graph_discovery(data,names,candidates,regimes,3,controls,bandwidths=(4,),cluster_lengths=(16,))
    old=run_fast_granger_discovery(data,names,candidates,regimes,3,controls=controls)
    joined=answer.merge(old,on=['source_region','target_region','source_var','target_var','lag','regime'])
    np.testing.assert_allclose(joined.p_ols,joined.p_value,rtol=2e-10,atol=1e-12)
    np.testing.assert_allclose(joined.effect_coefficient_x,joined.effect_coefficient_y,rtol=2e-10,atol=1e-12)
    for row in answer.itertuples():
        keys=[(v,l) for v in (range(2) if all_vars else [1]) for l in (1,2,3)
              if not(row.source_region==1 and v==0 and l==row.lag)]
        c=np.column_stack([data[3-l:350-l,1,v] for v,l in keys])
        single=fit_edge(data[3:,1,1],c,data[3-row.lag:350-row.lag,row.source_region,0],regimes[row.regime][3:],bandwidths=(4,),cluster_lengths=(16,),calendar_offset=3)
        assert row.se_hac4==pytest.approx(single['se_hac4'],rel=1e-10)
        assert row.se_cluster16==pytest.approx(single['se_cluster16'],rel=1e-10)


@pytest.mark.parametrize('masked',[False,True])
def test_calendar_block_cluster_matches_statsmodels_cr1(masked):
    from statsmodels.stats.sandwich_covariance import cov_cluster
    from revision.inference import fit_edge_block_cluster
    y,c,z=sample(n=500)
    mask=np.arange(len(y))%9<4 if masked else np.ones(len(y),dtype=bool)
    answer=fit_edge_block_cluster(y,c,z,mask,block_lengths=(16,64,128))
    x=sm.add_constant(np.column_stack([c,z]))
    fit=sm.OLS(y[mask],x[mask]).fit()
    for b in (16,64,128):
        labels=(np.arange(len(y))//b)[mask]
        g=len(np.unique(labels))
        expected=np.sqrt(cov_cluster(fit,labels,use_correction=True)[-1,-1])
        assert answer[f'se_cluster{b}']==pytest.approx(expected,rel=1e-11)
        assert answer[f'p_cluster{b}']==pytest.approx(2*stats.t.sf(abs(fit.params[-1])/expected,g-1),rel=1e-11)
        assert answer[f'n_clusters{b}']==g


def test_cluster_empty_calendar_blocks_not_compressed_and_invalid_retained():
    from revision.inference import fit_edge_block_cluster
    y,c,z=sample(n=500)
    mask=(np.arange(500)<100)|(np.arange(500)>400)
    result=fit_edge_block_cluster(y,c,z,mask,block_lengths=(50,))
    assert result['n_clusters50']==4
    assert fit_edge_block_cluster(y,c,c[:,0],mask,block_lengths=(50,))['p_cluster50']==1
