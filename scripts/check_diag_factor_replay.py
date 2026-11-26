"""Check the production diag factor on captured native blocks and edge cases."""
import argparse,ctypes,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path('output/shirt_L2_diag_fix_20261119'))
a=p.parse_args();dll=ctypes.CDLL(str((a.root/'production_factor_replay.dll').resolve()))
ptr=ctypes.POINTER(ctypes.c_double);dll.evaluate.argtypes=[ctypes.c_int,ptr,ptr];dll.evaluate.restype=ctypes.c_int
def evaluate(D,b):
    data=np.ascontiguousarray(np.c_[D.reshape(-1,9),b]);out=np.zeros((len(b),7))
    assert dll.evaluate(len(b),data.ctypes.data_as(ptr),out.ctypes.data_as(ptr))==0
    return out
results=[]
for k in [1900,2074]:
    d=np.load(f'output/shirt_L2_diag_20261119/blocks_{k}.npz');D,b=d['D'],d['r'];out=evaluate(D,b);z=out[:,3:6]
    residual=np.linalg.norm(np.einsum('nij,nj->ni',D,z)-b,axis=1)/np.maximum(np.linalg.norm(b,axis=1),1e-30)
    assert np.isfinite(z).all() and np.all(out[:,6]==0) and np.all((b*z).sum(1)>=0) and residual.max()<1e-7
    results.append(dict(step=k,blocks=len(b),max_local_relative_residual=float(residual.max()),negative_rz_blocks=0,fallbacks=0))
D=np.array([np.eye(3),[[1,2,0],[2,1,0],[0,0,1]],np.diag([0,1,1]),np.eye(3)*1e-200,np.eye(3)*1e200])
out=evaluate(D,np.ones((5,3)))
assert list(out[:,6])==[0,1,-1,0,0]
np.testing.assert_allclose(out[1,3:6],[1,1,1]);assert np.isfinite(out[[0,1,3,4],3:6]).all()
result=dict(captured_blocks=results,synthetic_factor_status=out[:,6].tolist(),passed=True)
(a.root/'block_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
