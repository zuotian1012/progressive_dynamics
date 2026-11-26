"""Replay the real failing matrix through the compensated CUDA arithmetic."""
import ctypes as ct,json,time
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from analyze_shirt_pcg_failure import read

base=Path('output/shirt_L3_solver_fix_20261119');root=base/'capture_from_546'
A=load_npz(root/'failed_A.npz');vec=np.load(root/'failed_vectors.npz');analysis=json.loads((root/'matrix_analysis.json').read_text())
dims,raw=read(next(root.rglob('A.*.mtx')));k=len(raw)//9
rows=np.ascontiguousarray((raw[::9,0].astype(np.int32)-1)//3);cols=np.ascontiguousarray((raw[::9,1].astype(np.int32)-1)//3)
blocks=np.ascontiguousarray(raw[:,2].reshape(k,3,3).transpose(0,2,1).reshape(-1))
dll=ct.CDLL(str((base/'stable_spmv_replay.dll').resolve()));fun=dll.stable_spmv_replay
i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
fun.argtypes=[ct.c_int,ct.c_int,i32,i32,f64,f64,f64,f64,ct.c_int,ct.c_int];fun.restype=ct.c_int
p=vec['p'];q=np.ascontiguousarray(p/abs(p).max());ref=float(analysis['normalized_curvature_decimal80']);reports=[]
for graph in [0,1]:
    y=np.zeros(len(q));dot=np.zeros(1);t=time.perf_counter();code=fun(len(q),k,rows,cols,blocks,q,y,dot,10,graph);seconds=time.perf_counter()-t
    assert code==0,code
    # Independently recompute the rows with the largest FP64/compensated difference.
    old=A@q;ids=np.argsort(abs(old-y))[-128:];errors=[];old_errors=[]
    with localcontext() as ctx:
        ctx.prec=80;qq=[Decimal.from_float(float(v)) for v in q]
        for row in ids:
            start,end=A.indptr[row:row+2];value=sum((Decimal.from_float(float(v))*qq[j] for j,v in zip(A.indices[start:end],A.data[start:end])),Decimal(0));r=float(value)
            errors.append(abs(y[row]-r));old_errors.append(abs(old[row]-r))
    row_error=max(errors);relative_dot=abs(dot[0]-ref)/abs(ref)
    assert dot[0]>0 and relative_dot<1e-10
    assert row_error<=max(1e-12,np.max(abs(y[ids]))*3e-16)
    reports.append(dict(graph=graph,repetitions=10,seconds=seconds,curvature=float(dot[0]),decimal80_curvature=ref,relative_curvature_error=float(relative_dot),
        selected_rows=128,maximum_row_absolute_error=row_error,original_cpu_maximum_row_error=max(old_errors)))
np.savez_compressed(base/'stable_spmv_verified.npz',q=q,Ap=y)
(base/'stable_spmv_verification.json').write_text(json.dumps(dict(passed=True,cases=reports,scope='Captured failing direction, both plain and CUDA Graph execution; 80-digit independent reference.'),indent=2));print(json.dumps(reports,indent=2))
