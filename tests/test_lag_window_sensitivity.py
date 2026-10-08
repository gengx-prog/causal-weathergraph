import numpy as np
from revision.run_lag_window_sensitivity import fit_joint, loss_contrast, make_inputs


def test_deleted_joint_lag_prediction_equals_direct_restricted_refit():
    rng=np.random.default_rng(197)
    base=np.column_stack((np.ones(900),rng.normal(size=(900,4))))
    source=rng.normal(size=(900,12));source[:,1:]+=.8*source[:,[0]]
    y=base[:,1]*.3+source[:,0]*.4+source[:,8]*.2+rng.normal(size=900)
    train=np.arange(900)<650
    fit=fit_joint(y,base,source,train)
    x=np.column_stack((base,source))
    for k in range(12):
        reduced=np.delete(x,base.shape[1]+k,axis=1)
        beta=np.linalg.lstsq(reduced[train],y[train],rcond=None)[0]
        np.testing.assert_allclose(fit['deleted_prediction'][:,k],reduced@beta,atol=1e-12)


def test_hac_equals_direct_calendar_sandwich_with_gaps():
    rng=np.random.default_rng(96)
    base=np.column_stack((np.ones(1300),rng.normal(size=(1300,2))))
    source=rng.normal(size=(1300,3));y=.2*source[:,0]+rng.normal(size=1300)
    mask=(np.arange(1300)//77)%2==0
    fit=fit_joint(y,base,source,mask)
    x=np.column_stack((base,source));inverse=np.linalg.pinv(x[mask]);beta=inverse@y[mask]
    score=np.zeros((len(y),3));score[mask]=inverse[-3:].T*(y[mask]-x[mask]@beta)[:,None]
    v=(score*score).sum(0)
    for lag in range(1,65):v+=2*(1-lag/65)*(score[lag:]*score[:-lag]).sum(0)
    se=np.sqrt(v*mask.sum()/(mask.sum()-x.shape[1]))
    np.testing.assert_allclose(fit['se_hac64'],se,atol=1e-13)


def test_evaluation_values_do_not_change_training_coefficients():
    rng=np.random.default_rng(87)
    base=np.column_stack((np.ones(500),rng.normal(size=(500,2))))
    source=rng.normal(size=(500,6));y=rng.normal(size=500);mask=np.arange(500)<350
    first=fit_joint(y,base,source,mask)
    source[~mask]+=100; y[~mask]-=1000; base[~mask,1:]*=3
    second=fit_joint(y,base,source,mask)
    np.testing.assert_array_equal(first['beta'],second['beta'])
    np.testing.assert_array_equal(first['se_hac64'],second['se_hac64'])


def test_physical_history_strictly_precedes_all_twelve_source_lags():
    from types import SimpleNamespace
    data=np.arange(40*4*2).reshape(40,4,2)
    physical=np.arange(40*4*6).reshape(40,4,6)
    pair=SimpleNamespace(source_region=1,target_region=1,source_var='x',target_var='y')
    times=np.arange(15,40)
    y,b,s=make_inputs(data,physical,['x','y'],pair,times,'own3_physical_before12')
    assert b.shape==(25,22)
    np.testing.assert_array_equal(b[:,4],physical[times-13,1,0])
    np.testing.assert_array_equal(b[:,-1],physical[times-15,1,5])
    np.testing.assert_array_equal(s[:,-1],data[times-12,1,0])
    assert np.all(times-13<times-12)
