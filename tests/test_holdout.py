import numpy as np
from revision.run_holdout import predictive_losses,pcmci_all_candidates
from causal_weathergraph.candidate_edges import CandidateEdge


def test_frozen_prediction_has_no_coefficient_leakage():
    rng=np.random.default_rng(74)
    data=rng.normal(size=(600,1,2))
    data[1:,0,1]=.7*data[:-1,0,0]+rng.normal(scale=.4,size=599)
    train=np.arange(600)<400; test=~train
    edges=[CandidateEdge(0,0,'x','y','x_to_y',0)]
    a=predictive_losses(data,['x','y'],edges,train,test)
    changed=data.copy();changed[test,0,1]+=100
    b=predictive_losses(changed,['x','y'],edges,train,test)
    np.testing.assert_array_equal(a.frozen_source_beta,b.frozen_source_beta)
    assert a.loc[a.lag==1,'mse_gain'].iloc[0]>0
    assert len(a)==3


def test_pcmci_history_prefix_keeps_all_evaluation_responses():
    rng=np.random.default_rng(391)
    data=rng.normal(size=(150,1,2));mask=np.arange(150)>=100
    edge=CandidateEdge(0,0,'x','y','x_to_y',0)
    table=pcmci_all_candidates(data,['x','y'],[edge],mask)
    assert len(table)==3
    assert (table.pcmci_response_n==50).all()
    assert (table.pcmci_observed_history_prefix==6).all()
