"""One CUDA operator comparison on an accepted L0 state; no simulation."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import csr_matrix
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from progressive.prolongation import ShellProlongation

root=Path('output/libuipc_psq_bridge_L0')
d=np.load(root/'operator.npz')
U=csr_matrix((d['U_data'],d['U_indices'],d['U_indptr']),shape=tuple(d['U_shape']))
p=ShellProlongation(d['coarse_rest'],d['fine_rest'],d['coarse_faces'],U,d['anchors'])
run=Path('output/libuipc_bunny_L0_contact_v2')
meta=json.loads((run/'run.json').read_text())
x=np.ascontiguousarray(np.load(run/'accepted_positions.npy')[meta['completed_steps']])
g=np.random.default_rng(527).normal(size=(U.shape[0],3))
dll=Path('native/libuipc/psq_operator_check.dll').resolve()
lib=ct.CDLL(str(dll))
I=ct.POINTER(ct.c_int);D=ct.POINTER(ct.c_double)
lib.psq_check.argtypes=[ct.c_int]*4+[I,I,D,I,I,D,D,D,D,D,D,D]
lib.psq_check.restype=ct.c_int
def pi(a):return a.ctypes.data_as(I)
def pd(a):return a.ctypes.data_as(D)
row=np.ascontiguousarray(d['U_indptr']);col=np.ascontiguousarray(d['U_indices']);v=np.ascontiguousarray(d['U_data'])
faces=np.ascontiguousarray(d['coarse_faces']);anchors=np.ascontiguousarray(d['anchors'])
inv=np.ascontiguousarray(d['inv_dm']);gamma=np.ascontiguousarray(d['gamma']);tangent=np.ascontiguousarray(d['tangent'])
y=np.empty((U.shape[0],3));gc=np.empty_like(x)
start=time.perf_counter()
code=lib.psq_check(U.shape[0],U.shape[1],len(faces),U.nnz,pi(row),pi(col),pd(v),pi(faces),pi(anchors),pd(inv),pd(gamma),pd(tangent),pd(x),pd(g),pd(y),pd(gc))
elapsed=time.perf_counter()-start
if code:raise RuntimeError(f'CUDA operator error {code}')
pe=float(np.max(np.abs(y-p.position(x))))
ge=float(np.max(np.abs(gc-U.T@g)))
assert np.isfinite(y).all() and np.isfinite(gc).all() and pe<1e-12 and ge<1e-10
result=dict(passed=True,fixture='accepted L0 native probe step 25',proxy_vertices=U.shape[0],coarse_vertices=U.shape[1],
    position_max_error_m=pe,frozen_gradient_max_error=ge,host_to_host_seconds_including_cuda_initialization=elapsed,
    gradient_fixture='seeded synthetic vector to verify U.T operator, not physical forces',
    backend='CUDA 12.6 sm_86 custom PSQ operators',libuipc_newton_connected=False,
    scope='Single operator comparison, not dynamics, strain or contact validation',
    hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [dll,Path('native/libuipc/psq_kernels.cuh'),root/'operator.npz',run/'accepted_positions.npy']})
(root/'cuda_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
