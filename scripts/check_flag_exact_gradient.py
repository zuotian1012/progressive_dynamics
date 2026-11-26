"""Compare the production CUDA offset VJP to independent NumPy SVD differences."""
from pathlib import Path
import ctypes,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/flag_L0_repair_20261118'

def gpu(x,f,anchors,inv,gamma,tangent,fg):
    dll=ctypes.CDLL(str(OUT/'exact_build/check_exact.dll'))
    arrays=[np.ascontiguousarray(a,dtype=np.int32 if i<2 else np.float64) for i,a in enumerate([f,anchors,inv,gamma,tangent,x,fg])]
    y=np.zeros_like(x);dll.evaluate.restype=ctypes.c_int
    dll.evaluate.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*8
    code=dll.evaluate(len(x),len(anchors),len(f),*[a.ctypes.data for a in arrays],y.ctypes.data)
    assert code==0,code
    return y

def offset(x,f,anchors,inv,gamma,tangent):
    tri=x[f];F=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=2)@inv
    u,_,v=np.linalg.svd(F,full_matrices=False);R=u@v
    n=np.cross(F[:,:,0],F[:,:,1]);n/=np.linalg.norm(n,axis=1)[:,None]
    return n[anchors]*gamma[:,None]+np.einsum('nij,nj->ni',R[anchors],tangent)

def main():
    rng=np.random.default_rng(183);nt=500
    f=np.arange(3*nt,dtype=np.int32).reshape(-1,3);anchors=np.arange(nt,dtype=np.int32)
    base=rng.normal(size=(nt,1,3));F=np.linalg.qr(rng.normal(size=(nt,3,2)))[0]*rng.uniform(.3,1.1,(nt,1,2))
    x=np.concatenate([base,base+F[:,:,0][:,None],base+F[:,:,1][:,None]],axis=1).reshape(-1,3)
    inv=np.tile(np.array([[1,.2],[.1,1.4]]),(nt,1,1));gamma=rng.normal(size=nt)*.02;tangent=rng.normal(size=(nt,2))*.03;fg=rng.normal(size=(nt,3))
    grad=gpu(x,f,anchors,inv,gamma,tangent,fg).reshape(nt,3,3);fd=np.empty_like(grad);eps=1e-6
    for j in range(3):
        for k in range(3):
            d=np.zeros_like(x).reshape(nt,3,3);d[:,j,k]=eps;d=d.reshape(-1,3)
            fd[:,j,k]=np.sum((offset(x+d,f,anchors,inv,gamma,tangent)-offset(x-d,f,anchors,inv,gamma,tangent))*fg,axis=1)/(2*eps)
    np.testing.assert_allclose(grad,fd,atol=2e-9,rtol=2e-5)
    assert np.max(abs(grad.sum(axis=1)))<1e-13
    d=np.load(ROOT/'output/flag_L0_shirt_aligned_20261118/operator/operator.npz')
    x=np.load(ROOT/'output/flag_L0_shirt_aligned_20261118/positions.npy')[206]
    fg=rng.normal(size=(len(d['fine_rest']),3));args=[d['coarse_faces'],d['anchors'],d['inv_dm'],d['gamma'],d['tangent']]
    grad=gpu(x,*args,fg);errors=[]
    for j in range(5):
        direction=rng.normal(size=x.shape);direction/=np.linalg.norm(direction)
        pred=float(np.sum(grad*direction));fd=float(np.sum((offset(x+eps*direction,*args)-offset(x-eps*direction,*args))*fg)/(2*eps))
        np.testing.assert_allclose(pred,fd,rtol=2e-5,atol=2e-8);errors.append(abs(pred-fd))
    result=dict(random_triangles=nt,checked_components=nt*9,actual_frozen_state_directions=5,
        frozen_state_max_directional_error=max(errors),passed=True,reference='Independent SVD central differences, including normal and tangential offsets')
    (OUT/'exact_gradient_check.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
