"""Replay saved triangle states through independently compiled production GPU predicates."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np

def evaluate(dll, rows):
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    out = np.zeros((len(rows), 16), dtype=np.float64)
    ptr = ctypes.POINTER(ctypes.c_double)
    fn = dll.evaluate
    fn.argtypes = [ctypes.c_int, ptr, ptr]
    fn.restype = ctypes.c_int
    error = fn(len(rows), rows.ctypes.data_as(ptr), out.ctypes.data_as(ptr))
    assert error == 0, error
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dll',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();dll=ctypes.CDLL(str(args.dll.resolve()));reports=[];cases=[]
    for name,steps in [('L2',[0,60,596,1076,1860,2028,2216,2400,2423]),('L2_ls8_failed',[1076,2028])]:
        run=Path('output/shirt_four_level_full_fold_20261118')/name/'run'
        x=np.load(run/'positions.npy',mmap_mode='r');faces=np.load(run/'faces.npy');rest=x[0][faces].reshape(-1,9)
        for step in steps:
            rows=np.concatenate([rest,x[step][faces].reshape(-1,9)],axis=1);out=evaluate(dll,rows)
            if step==0:
                assert np.max(np.abs(out[:,2:4]-1))<1e-10
                assert np.isfinite(out[:,:2]).all() and np.all(out[:,4]>0)
            finite=np.isfinite(out[:,0]);grad=np.isfinite(out[:,1]);cap=out[:,4]>0
            ids=np.flatnonzero((finite & ~cap)|(finite!=grad)|~finite)
            rec=dict(run=name,step=step,energy_nonfinite=int((~finite).sum()),
                gradient_energy_nonfinite=int((~grad).sum()),cap_zero=int((~cap).sum()),
                finite_energy_but_zero_cap=int((finite & ~cap).sum()),
                energy_derivative_domain_disagreement=int((finite!=grad).sum()),
                max_singular=float(out[:,2].max()),min_singular=float(out[:,3].min()),faces=ids.tolist())
            reports.append(rec);print({k:v for k,v in rec.items() if k!='faces'},flush=True)
            for face in ids[:30]:
                cases.append(dict(run=name,step=step,face=int(face),vertices=faces[face].tolist(),
                                  input=rows[face].tolist(),output=out[face].tolist()))
    args.output.write_text(json.dumps(dict(reports=reports,cases=cases,
        note='Production formulas in separately compiled kernels; kernel optimization/register allocation can differ from the original runtime.'),indent=2))

if __name__=='__main__':main()
