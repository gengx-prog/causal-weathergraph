import numpy as np
from revision.run_castle_benchmark import generate_grid,neighborhoods,graph_metrics


def test_stencil_truth_and_layout():
    data,truth=generate_grid(2,size=6,length=64)
    assert data.shape==(6,6,64) and truth.shape==(16,8)
    assert truth.sum()==32
    patches=list(neighborhoods(data))
    np.testing.assert_array_equal(patches[0][:,4],data[1,1])
    np.testing.assert_array_equal(patches[0][:,3],data[1,0])
    np.testing.assert_array_equal(patches[0][:,1],data[0,1])


def test_metrics_common_indexed_family():
    truth=np.array([[True,False],[False,True]])
    m=graph_metrics(truth,np.where(truth,1e-8,1.))
    assert m['f1']==1 and m['shd']==0 and m['tested']==4
