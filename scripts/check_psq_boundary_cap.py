"""Regression checks for directional near-boundary CUDA caps."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.check_psq_step_cap import evaluate,evaluate_recorded


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before-dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    f0=np.array([[1.1-8*np.finfo(float).eps,0],[0,.2],[0,0.]])
    # The last direction initially contracts, then leaves the admissible set;
    # an endpoint-only check at the rank cap misses its useful interior segment.
    directions=np.stack([-.01*f0,.01*f0,-2*f0,np.zeros((3,2)),
                         np.array([[-.001,0],[0,0],[.1,0]])])
    f=np.repeat(f0[None],len(directions),axis=0)
    caps=evaluate(f,directions)
    assert caps[0,1]==1 and caps[0,0]<1e-10
    assert 0<caps[1,1]<1e-10
    assert 0<caps[2,1]<.5 and caps[3,1]==1
    assert .01<caps[4,1]<1
    max_stretch=0.;min_sigma=float('inf')
    for t in np.linspace(0,1,1001):
        s=np.linalg.svd(f+t*caps[:,1,None,None]*directions,compute_uv=False)
        assert np.all(s[:,0]<1.1) and np.all(s[:,1]>1e-12)
        max_stretch=max(max_stretch,float(s[:,0].max()));min_sigma=min(min_sigma,float(s[:,1].min()))
    # Away from the near-boundary margin the original directional formula
    # must be exactly unchanged for identical native SVD/norm inputs.
    rng=np.random.default_rng(20261116)
    n=10000
    u=np.linalg.qr(rng.normal(size=(n,3,2)))[0]
    v=np.linalg.qr(rng.normal(size=(n,2,2)))[0]
    high=1.1-10**rng.uniform(-10,0,n);low=high*10**rng.uniform(-7,0,n)
    ff=(u*np.stack([high,low],axis=1)[:,None,:])@v.transpose(0,2,1)
    dd=rng.normal(size=(n,3,2))*10**rng.uniform(-7,2,n)[:,None,None]
    norm=np.linalg.norm(dd,axis=(1,2));legacy=np.minimum(1,.9*np.minimum(1.1-high,low-1e-12)/norm)
    rows=np.column_stack((legacy,np.zeros(n),high,low,norm,
                          ff.transpose(0,2,1).reshape(n,6),dd.transpose(0,2,1).reshape(n,6)))
    old=evaluate_recorded(rows,a.before_dll.resolve());new=evaluate_recorded(rows)
    np.testing.assert_array_equal(old,new)
    invalid=np.array([[[1.1,0],[0,.5],[0,0]],[[1.,0],[0,0],[0,0]]])
    assert np.all(evaluate(invalid,np.zeros_like(invalid))==0)
    result=dict(special_cases=['inward','outward','rank_crossing','zero','inward_then_outward'],
        caps=caps.tolist(),samples_per_path=1001,maximum_stretch=max_stretch,minimum_sigma=min_sigma,
        non_boundary_exact_regressions=n,invalid_states_rejected=True,strain_limit=1.1)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
