"""Produce a versioned binary adapter bundle; no simulation and no old-file writes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from progressive.prolongation import frames

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',type=Path)
    p.add_argument('--identity',action='store_true')
    p.add_argument('--level',type=int,choices=[0,1,2],default=0,help='Bunny mesh level for an identity/reference bundle')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--young',type=float,default=30000.)
    p.add_argument('--poisson',type=float,default=.3)
    p.add_argument('--thickness',type=float,default=7e-5)
    p.add_argument('--strain-limit',type=float,default=1.068)
    p.add_argument('--strain-onset',type=float,default=1.034)
    a=p.parse_args()
    if a.identity == bool(a.operator): raise ValueError('Select exactly one operator/identity')
    if a.identity:
        mesh=np.load(f'output/bunny_paper_assets/mesh_{a.level}.npz'); rest=mesh['vertices'];faces=mesh['faces']
        anchors=np.full(len(rest),-1,dtype=np.int32)
        for i,t in enumerate(faces): anchors[t]=i
        if np.any(anchors<0):raise ValueError('Isolated identity vertex')
        d=dict(coarse_rest=rest,coarse_faces=faces,fine_rest=rest,fine_faces=faces,
               U_indptr=np.arange(len(rest)+1),U_indices=np.arange(len(rest)),U_data=np.ones(len(rest)),
               anchors=anchors,inv_dm=frames(rest,faces)[2],gamma=np.zeros(len(rest)),tangent=np.zeros((len(rest),2)))
    else:
        with np.load(a.operator) as f:d=dict(f)
    edges={}
    for t in d['fine_faces']:
        a0,b,c=sorted(map(int,t))
        for e,o in [((a0,b),c),((a0,c),b),((b,c),a0)]:edges.setdefault(e,[]).append(o)
    if any(len(v)>2 for v in edges.values()):raise ValueError('Nonmanifold fine shell')
    hinges=np.array([(min(v),e[0],e[1],max(v)) for e,v in sorted(edges.items()) if len(v)==2],dtype='<i4')
    nc,nct,nf,nt,nnz,nh=len(d['coarse_rest']),len(d['coarse_faces']),len(d['fine_rest']),len(d['fine_faces']),len(d['U_data']),len(hinges)
    E,nu,h=a.young,a.poisson,a.thickness
    params=[E*nu/(1-nu*nu),E/(2*(1+nu)),h,E*h**3/(12*(1-nu*nu)),a.strain_limit,a.strain_onset,1000.]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:
        f.write(b'PSQIPC01');f.write(np.array([nc,nct,nf,nt,nnz,nh],dtype='<i4').tobytes())
        f.write(np.array(params,dtype='<f8').tobytes())
        for key,dtype in [('coarse_rest','<f8'),('coarse_faces','<i4'),('fine_rest','<f8'),('fine_faces','<i4'),
            ('U_indptr','<i4'),('U_indices','<i4'),('U_data','<f8'),('anchors','<i4'),('inv_dm','<f8'),('gamma','<f8'),('tangent','<f8')]:
            f.write(np.asarray(d[key],dtype=dtype,order='C').tobytes())
        f.write(hinges.tobytes())
    identity=(nc==nf and np.array_equal(d['coarse_rest'],d['fine_rest']) and np.array_equal(d['coarse_faces'],d['fine_faces'])
        and np.array_equal(d['U_indptr'],np.arange(nc+1)) and np.array_equal(d['U_indices'],np.arange(nc))
        and np.array_equal(d['U_data'],np.ones(nc)) and not np.any(d['gamma']) and not np.any(d['tangent']))
    meta=dict(format='PSQIPC01',identity=identity,counts=dict(coarse_vertices=nc,coarse_faces=nct,proxy_vertices=nf,proxy_faces=nt,nnz=nnz,hinges=nh),
        parameters=dict(zip(['lambda','mu','full_thickness','bending','strain_limit','strain_onset','strain_stiffness'],params)),
        sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),source_operator=str(a.operator.resolve()) if a.operator else None)
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
