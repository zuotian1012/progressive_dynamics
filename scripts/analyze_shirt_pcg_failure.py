"""Read native failure dumps and independently check curvature and true residual."""
import argparse, json, math, time
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, save_npz

def read(path):
    with Path(path).open() as f:
        while True:
            line=f.readline()
            if not line.startswith('%'):break
        dims=tuple(map(int,line.split()));data=np.loadtxt(f)
    return dims,data

def main():
    arg=argparse.ArgumentParser();arg.add_argument('root',type=Path);a=arg.parse_args();root=a.root
    path=next(root.rglob('failure.*.meta.txt'));meta=dict(t.split('=') for t in path.read_text().split());prefix=path.name.removesuffix('meta.txt')
    dims,raw=read(next(root.rglob(f'A.{meta["frame"]}.{meta["newton"]}.mtx')))
    i=raw[:,0].astype(np.int32)-1;j=raw[:,1].astype(np.int32)-1;v=raw[:,2];off=i//3!=j//3
    A=coo_matrix((np.r_[v,v[off]],(np.r_[i,j[off]],np.r_[j,i[off]])),shape=dims[:2]).tocsr()
    vectors={n:read(path.with_name(prefix+n+'.mtx'))[1].ravel() for n in ['p','Ap','r','z','x','b']}
    p,Ap,r,z,x,b=[vectors[n] for n in ['p','Ap','r','z','x','b']]
    assert np.isfinite(A.data).all() and all(np.isfinite(t).all() for t in vectors.values())
    ps=np.max(abs(p));q=p/ps;aq=A@q;true_r=b-A@x;diag=A.diagonal();sym=A-A.T
    weights=v*q[i]*q[j]*np.where(off,2.,1.)
    B=A.multiply(1/np.sqrt(diag)[:,None]).multiply(1/np.sqrt(diag)[None,:]).tocsr()
    ids=np.argsort(diag)[-192:];patch=B[ids][:,ids].toarray();eigs=np.linalg.eigvalsh((patch+patch.T)*.5)
    result=dict(meta=meta,dofs=A.shape[0],nonzeros=A.nnz,diagonal_min=float(diag.min()),diagonal_max=float(diag.max()),diagonal_ratio=float(diag.max()/diag.min()),
        symmetry_max_abs=float(np.max(abs(sym.data),initial=0)),symmetry_relative_max=float(np.max(abs(sym.data),initial=0)/np.max(abs(A.data))),
        saved_pAp=float(p@Ap),cpu_pAp=float(p@(A@p)),saved_rz=float(r@z),search_direction_max_abs=float(ps),
        normalized_curvature_cpu=float(q@aq),normalized_curvature_fsum=math.fsum(weights),
        normalized_sum_abs_terms=math.fsum(abs(weights)),
        spmv_relative_error=float(np.linalg.norm(aq-Ap/ps)/np.linalg.norm(aq)),
        recursive_residual_norm=float(np.linalg.norm(r)),true_residual_norm=float(np.linalg.norm(true_r)),rhs_norm=float(np.linalg.norm(b)),
        recursive_relative_residual=float(np.linalg.norm(r)/np.linalg.norm(b)),true_relative_residual=float(np.linalg.norm(true_r)/np.linalg.norm(b)),
        residual_gap_relative_to_rhs=float(np.linalg.norm(r-true_r)/np.linalg.norm(b)),scaled_top_diagonal_patch_min_eigenvalue=float(eigs[0]))
    print(json.dumps(result,indent=2),flush=True)
    # Decimal.from_float preserves every binary64 input exactly; sum all stored
    # blocks with their implicit symmetric counterpart, not a fitted matrix.
    tick=time.perf_counter()
    with localcontext() as ctx:
        ctx.prec=80;qp=[Decimal.from_float(float(t)) for t in q];energy=Decimal(0)
        for ii,jj,vv,oo in zip(i,j,v,off):
            if vv:energy+=Decimal.from_float(float(vv))*qp[ii]*qp[jj]*(2 if oo else 1)
    result['normalized_curvature_decimal80']=str(energy);result['decimal_seconds']=time.perf_counter()-tick
    result['diagonal_relative_shift_for_this_direction']=max(0.,-float(energy)/float(q@(diag*q)))
    save_npz(root/'failed_A.npz',A);np.savez_compressed(root/'failed_vectors.npz',**vectors)
    (root/'matrix_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
