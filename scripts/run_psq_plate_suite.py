"""Run one complete load case using the same numerical settings for all models."""
import argparse
import hashlib
from pathlib import Path
import shutil
import numpy as np
from demo_psq_plate import ROOT, Model, load, run, write_json, numerical_threads


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=('compression','shear'),required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'output/psq_plate_boundary_fixed')
    a=ap.parse_args();root=a.output
    with numerical_threads(1):
        h,*_=load(root);model=Model(h,2,False)
        rng=np.random.default_rng(1603);x=h.rests[2].copy()
        x[:,:2]*=[.98,1.01];x[:,2]=.002*np.sin(x[:,0]*10)
        e,g,_=model.evaluate(x);legacy=model.fine.evaluate(x)[0]
        assert abs(e-legacy)<1e-8*max(1,abs(e))
        d=rng.normal(size=x.shape);d/=np.linalg.norm(d)
        epsilon=1e-7
        derivative=(model.evaluate(x+epsilon*d)[0]-model.evaluate(x-epsilon*d)[0])/(2*epsilon)
        analytic=float(np.sum(g*d))
        assert abs(derivative-analytic)<1e-5*max(1,abs(analytic))
        write_json(root/f'{a.case}_energy_check.json',dict(energy_J=e,legacy_energy_J=legacy,
            finite_difference_directional_force=derivative,analytic_directional_force=analytic))
        for name,k,proxy,parent in [('direct_L0',0,False,None),('psq_L0',0,True,None),
            ('psq_L1',1,True,'psq_L0_verified'),('psq_L2',2,True,'psq_L1_verified'),('direct_L2',2,False,None)]:
            run(root,a.case,name+'_verified',k,proxy,24,2000,parent)
        shutil.copy2(ROOT/'scripts/demo_psq_plate.py',root/a.case/'verified_source.py')
        write_json(root/a.case/'verified_sources.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'scripts/demo_psq_plate.py',ROOT/'progressive/shell.py',ROOT/'progressive/prolongation.py')})


if __name__=='__main__':main()
