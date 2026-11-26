"""Reconstructed PD++ flag hierarchy at the published endpoint scale.

Author mesh/forcing are unavailable: only documented scale and thickness are
matched. No animation or substitute success claim is produced here.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from progressive.prepare_intrinsic import prepare
from progressive.runtime import numerical_threads


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    # Exactly 35,000 vertices. Domain size is explicitly a reconstruction choice.
    nx,nz=249,139
    xx,zz=np.meshgrid(np.linspace(0,1.5,nx+1),np.linspace(0,1.,nz+1),indexing='xy')
    v=np.column_stack([xx.ravel(),np.zeros(xx.size),zz.ravel()+.25])
    ids=np.arange(len(v)).reshape(nz+1,nx+1)
    q=np.stack([ids[:-1,:-1].ravel(),ids[:-1,1:].ravel(),ids[1:,1:].ravel(),ids[1:,:-1].ravel()],axis=1)
    f=np.concatenate([q[:,[0,1,2]],q[:,[0,2,3]]])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    source=a.output.with_name(a.output.name+'_input.obj')
    with source.open('x',encoding='ascii') as fp:
        for row in v:fp.write('v '+' '.join(f'{c:.17g}' for c in row)+'\n')
        for row in f:fp.write('f '+' '.join(str(int(c)+1) for c in row)+'\n')
    targets=[44000,28000,17700,11100,6900,4300,2600,1600,1044]
    with numerical_threads(1):prepare(a.output,targets,source,contact_clearance=.0004)
    path=a.output/'manifest.json';m=json.loads(path.read_text())
    m['benchmark']={
        'name':'PD++ Waving Flag, Figures 5-6: paper-scale reconstructed assets',
        'paper':'https://pcs-sim.github.io/pd%2B%2B/PD%2B%2B_main.pdf',
        'confirmed':{'levels':10,'coarsest_vertices_approx':550,'finest_vertices_approx':35000,
                     'duration_seconds':20,'thickness_m':.0004,'maximum_stretch':1.1},
        'local_geometry':{'width_m':1.5,'height_m':1.,'initial_pose':'vertical flat sheet',
                          'finest_grid_vertices':[nx+1,nz+1]},
        'unknown_author_inputs':['exact rest meshes and hierarchy','flag dimensions','wind field/time profile',
            'material modulus, Poisson ratio, density','exact flag dt in PD++ (0.04 explicitly describes PD 2024)',
            'contact parameters','boundary pin selection','full solver settings'],
        'not_a_completed_reproduction':True}
    path.write_text(json.dumps(m,indent=2))
    print(json.dumps({'levels':m['levels'],'benchmark':m['benchmark']},indent=2))


if __name__=='__main__':main()
