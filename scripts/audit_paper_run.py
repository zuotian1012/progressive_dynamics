"""Independent saved-state audit, including sampled intersections and VelPro readback."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import ipctk
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from progressive.prolongation import frames

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
    m=json.loads((root/'run.json').read_text());assets=Path(m['assets']);d=np.load(assets/f'mesh_{m["level"]}.npz')
    rest,f=d['vertices'],d['faces'];x=np.load(root/'positions.npy',mmap_mode='r')[:m['completed_steps']+1]
    records=json.loads((root/'steps.json').read_text());inv=frames(rest,f)[2]
    max_stretch=0.
    for state in x:
        tri=state[f];F=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=2)@inv
        max_stretch=max(max_stretch,float(np.linalg.svd(F,compute_uv=False).max()))
    rr=rest;ff=f
    if m['scene']=='bunny':
        b=np.load(assets/'bunny.npz');rr=np.vstack([rr,b['vertices']]);ff=np.vstack([ff,b['faces']+len(rest)])
    edges=np.unique(np.sort(ff[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    cm=ipctk.CollisionMesh(rr,edges,ff.astype(np.int32));samples=[]
    for step in np.unique(np.linspace(0,len(x)-1,min(len(x),9)).astype(int)):
        state=x[step] if m['scene']=='flag' else np.vstack([x[step],b['vertices']])
        samples.append(dict(step=int(step),intersections=bool(ipctk.has_intersections(cm,state))))
    pin_error=0.
    if m['scene']=='flag':
        ee,cc=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
        bd=np.unique(ee[cc==1]);pins=bd[rest[bd,0]<1e-4];pin_error=float(np.max(np.abs(x[:,pins]-rest[pins])))
    reasons={}
    for row in records:
        n=row['native'];r=n['termination_reason'];reasons[r]=reasons.get(r,0)+1
        assert n['completed'] and not n['hit_newton_limit']
        assert r in ('newton_converged','preview_non_descent','preview_small_step')
        if r!='newton_converged':
            assert not n['converged'] and n['last_line_search_alpha']==0
            assert m['level']<(9 if m['scene']=='flag' else 7)
        else:assert n['converged'] and not n['hit_line_search_limit']
    result=dict(completed_requested_run=m['completed'],accepted_steps=len(records),physical_seconds=m['completed_steps']*m['dt'],
        finite=bool(np.isfinite(x).all()),maximum_stretch=max_stretch,limit=m['material']['strain_limit'],pin_error_m=pin_error,
        velocity_readback_error=max((r['velocity_readback_error'] for r in records),default=0),
        position_injection_error=max((r['position_injection_error'] for r in records),default=0),
        termination_counts=reasons,intersection_samples=samples,
        limitation='Sampled static intersections only, not independent continuous or finite-thickness collision certification')
    displacements=np.array([np.linalg.norm(x[k]-x[k-1],axis=1).max() for k in range(1,len(x))])
    frozen=displacements<1e-12;longest=streak=0
    for z in frozen:
        streak=streak+1 if z else 0;longest=max(longest,streak)
    result.update(zero_motion_steps=int(frozen.sum()),longest_zero_motion_seconds=longest*m['dt'],
        maximum_vertex_speed_m_s=float(displacements.max()/m['dt']) if len(displacements) else 0,
        semantic_note='Numerical acceptance is not proof of a physically responsive or paper-matched trajectory')
    result['accepted_prefix_checks_passed']=bool(result['finite'] and max_stretch<m['material']['strain_limit']+1e-9
        and pin_error<1e-10 and result['velocity_readback_error']<1e-12 and result['position_injection_error']<1e-12
        and not any(s['intersections'] for s in samples) and len(records)==len(x)-1)
    (root/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['accepted_prefix_checks_passed'] else 1

if __name__=='__main__':sys.exit(main())
