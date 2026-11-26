"""Check the production step-cap helper on CUDA against NumPy path singular values."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def evaluate(f,d,dll_path=None):
    rows=np.ascontiguousarray(np.concatenate([f.transpose(0,2,1).reshape(-1,6),d.transpose(0,2,1).reshape(-1,6)],axis=1))
    out=np.empty((len(f),2),dtype=np.float64)
    dll=ctypes.CDLL(str(dll_path or ROOT/'native/libuipc/psq_step_cap_check.dll'))
    ptr=ctypes.POINTER(ctypes.c_double)
    dll.evaluate.argtypes=[ctypes.c_int,ptr,ptr];dll.evaluate.restype=ctypes.c_int
    assert dll.evaluate(len(f),rows.ctypes.data_as(ptr),out.ctypes.data_as(ptr))==0
    return out


def evaluate_recorded(rows,dll_path=None):
    rows=np.ascontiguousarray(rows,dtype=np.float64)
    assert rows.ndim==2 and rows.shape[1]==17 and len(rows)>0
    out=np.empty((len(rows),2),dtype=np.float64)
    dll=ctypes.CDLL(str(dll_path or ROOT/'native/libuipc/psq_step_cap_check.dll'))
    ptr=ctypes.POINTER(ctypes.c_double)
    dll.evaluate_recorded.argtypes=[ctypes.c_int,ptr,ptr];dll.evaluate_recorded.restype=ctypes.c_int
    assert dll.evaluate_recorded(len(rows),rows.ctypes.data_as(ptr),out.ctypes.data_as(ptr))==0
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rng=np.random.default_rng(20261116);n=10000
    u=np.linalg.qr(rng.normal(size=(n,3,2)))[0];v=np.linalg.qr(rng.normal(size=(n,2,2)))[0]
    high=1.1-10**rng.uniform(-11,0,n);low=high*10**rng.uniform(-8,0,n)
    f=(u*np.stack([high,low],axis=1)[:,None,:])@v.transpose(0,2,1)
    d=rng.normal(size=(n,3,2))*10**rng.uniform(-7,2,n)[:,None,None]
    # Near-limit contraction, extension, rank-crossing, zero update, rotation-like update.
    base=np.array([[1.099996,0],[0,.5],[0,0.]])
    special=np.stack([-.01*base,.01*base,-2*base,np.zeros((3,2)),np.array([[0,-.1],[.1,0],[0,0.]])])
    f=np.concatenate([np.repeat(base[None],5,axis=0),f]);d=np.concatenate([special,d])
    audit_count=0
    if a.audit:
        records=[json.loads(line) for line in a.audit.read_text().splitlines()]
        vals=np.array([face['values'] for r in records for face in r['faces']]);audit_count=len(vals)
        f=np.concatenate([f,vals[:,5:11].reshape(-1,2,3).transpose(0,2,1)])
        d=np.concatenate([d,vals[:,11:17].reshape(-1,2,3).transpose(0,2,1)])
    cap=evaluate(f,d);assert np.isfinite(cap).all() and (cap>0).all() and (cap<=1).all()
    if audit_count:
        np.testing.assert_allclose(cap[-audit_count:],vals[:,:2],atol=2e-10,rtol=2e-5)
    invalid=np.array([[[1.1,0],[0,.5],[0,0]],[[1.,0],[0,0],[0,0]]])
    assert np.all(evaluate(invalid,np.zeros_like(invalid))==0)
    assert cap[0,1]==1 and cap[0,0]<.001
    assert cap[2,1]<.5 and cap[3,1]==1
    min_sigma=np.inf;max_sigma=0.
    for t in np.linspace(0,1,201):
        s=np.linalg.svd(f+t*cap[:,1,None,None]*d,compute_uv=False)
        min_sigma=min(min_sigma,float(s[:,1].min()));max_sigma=max(max_sigma,float(s[:,0].max()))
        assert np.all(s[:,0]<1.1) and np.all(s[:,1]>1e-12)
    result=dict(cases=len(f),actual_direction_cases=audit_count,path_samples_per_case=201,maximum_stretch=max_sigma,
                minimum_sigma=min_sigma,contraction_caps=cap[0].tolist(),rank_crossing_caps=cap[2].tolist(),
                finite_positive_bounded=True,upper_stretch_and_rank_passed=True,invalid_states_rejected=True)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
