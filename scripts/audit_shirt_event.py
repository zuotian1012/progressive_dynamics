"""Audit strict shirt PCG runs, including every saved cloth state."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import ipctk
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shirt_demo.solver_policy import converged_step, validate_parent
from progressive.prolongation import frames
from report_bunny_event_fallback_pilot import event_audit
from bunny_runtime import runtime_provenance


def audit(run):
    m=json.loads((run/'run.json').read_text());rows=json.loads((run/'steps.json').read_text())
    assert m['completed'] and m['status']=='completed' and m['event_fallback_enabled']
    assert m['requires_convergence'] and m['completed_steps']==m['steps']==len(rows)
    assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
    assert all(converged_step(r['native']) for r in rows)
    assert m['bunny_linear_solver']=='native_pcg' and m['config']['linear_system']['solver']=='fused_pcg'
    assert 0<m['config']['linear_system']['tol_rate']<=1e-10
    assert 0<m['config']['newton']['velocity_tol']<=.001
    assert not m['config']['newton']['semi_implicit']['enable']
    provenance=runtime_provenance(m['parameters']['runtime_package'])
    assert provenance['dll_sha256']==m['event_runtime_sha256']
    for path,h in m['runtime_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    for name,h in m['source_snapshots'].items():assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==h
    log=run.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
    assert 'LinearFusedPCG:' in log and 'BUNNY_DIRECT_SOLVE' not in log
    if m['level']:
        low=Path(m['lower_run']);validate_parent(json.loads((low/'run.json').read_text()),
            json.loads((low/'steps.json').read_text()),m['event_runtime_sha256'])
        subprocess.run([sys.executable,str(Path(__file__).with_name('audit_shirt_velocity_transfer.py')),str(run)],check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('audit_flag_fine_search.py')),str(run),'--check-converged-energy'],check=True)
    event=event_audit(run,m,rows)
    x=np.load(run/'positions.npy',mmap_mode='r');f=np.load(run/'faces.npy').astype(np.int32)
    assert x.shape==(m['steps']+1,m['vertices'],3) and np.isfinite(x).all()
    assert np.isfinite(np.load(run/'robot.npy',mmap_mode='r')).all()
    inv=frames(x[0],f)[2]
    edges=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    cm=ipctk.CollisionMesh(np.asarray(x[0]),edges,f)
    stretch=0.;intersections=[]
    for k,pos in enumerate(x):
        tri=pos[f];F=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)@inv
        stretch=max(stretch,float(np.linalg.svd(F,compute_uv=False).max()))
        if ipctk.has_intersections(cm,np.asarray(pos)):intersections.append(k)
    result=dict(passed=not intersections and stretch<=m['material_parameters']['strain_limit']+1e-9,
        steps=len(rows),converged_steps=len(rows),preview_steps=0,event_steps=len(event['event_frames']),
        saved_states_checked=len(x),maximum_principal_stretch=stretch,self_intersecting_states=intersections,
        finite=True,pcg_confirmed=True,runtime_and_source_hashes_verified=True,
        positions_sha256=hashlib.sha256((run/'positions.npy').read_bytes()).hexdigest(),
        scope='Every saved cloth state checked for strain and self-intersection; energy and event logs audited. Gripper-cloth intersections and between-frame collision geometry are not independently audited.')
    (run/'report').mkdir(exist_ok=True)
    (run/'report/strict_event_audit.json').write_text(json.dumps(result,indent=2))
    assert result['passed'],result
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);audit(p.parse_args().run)
