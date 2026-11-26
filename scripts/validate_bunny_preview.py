"""Verify native Bunny preview/strict/identity acceptance semantics and saved strain."""
import argparse
import json
from pathlib import Path
import re
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args();root=a.root
    read=lambda name:json.loads((root/name/'run.json').read_text())
    preview,strict,identity=map(read,['preview','strict','identity'])
    assert preview['completed'] and preview['completed_steps']==preview['requested_steps']
    assert not strict['completed'] and strict['completed_steps']==17 and strict['failed_step']==18
    assert identity['completed'] and identity['preview_stopped_steps']==0
    rows=json.loads((root/'preview/steps.json').read_text())
    stops=[]
    for row in rows:
        native=row['native'];stop=native['preview_non_descent']
        assert native['completed'] and bool(native['converged']) != bool(stop)
        if stop:
            assert native['termination_reason']=='preview_non_descent'
            assert native['last_line_search_alpha']==0
            stops.append(row['step'])
        else:assert native['termination_reason']=='newton_converged'
    assert stops and preview['preview_stopped_steps']==len(stops)
    log=(root/'preview.log').read_text(encoding='utf-8',errors='replace')
    assert 'PSQ_FORCE_ACCEPT' not in log
    probes=[]
    for line in log.splitlines():
        if 'PSQ_PREVIEW_STOP ' not in line:continue
        values={k:float(v) for k,v in re.findall(r'([a-zA-Z_]+[12]?)=([-+\d.eE]+)',line)}
        assert values['slope1']*values['h1']>values['energy_noise']
        assert values['slope2']*values['h2']>values['energy_noise']
        assert abs(values['slope1']-values['slope2'])<=.25*max(values['slope1'],values['slope2'])
        probes.append(values)
    assert [int(v['frame']) for v in probes]==stops
    states=np.load(root/'preview/accepted_positions.npy')
    assert len(states)==len(rows)+1 and np.isfinite(states).all()
    strict_states=np.load(root/'strict/accepted_positions.npy')
    prefix=min(stops[0],len(strict_states))
    prefix_error=float(np.max(np.abs(states[:prefix]-strict_states[:prefix])))
    assert prefix_error<1e-7
    mesh=np.load('output/bunny_paper_assets/mesh_0.npz');faces=mesh['faces'];rest=mesh['vertices'][faces]
    rest_edges=np.stack([rest[:,1]-rest[:,0],rest[:,2]-rest[:,0]],axis=-1)
    inv=np.linalg.inv(np.linalg.cholesky(rest_edges.transpose(0,2,1)@rest_edges).transpose(0,2,1))
    maximum=0.
    for x in states:
        tri=x[faces];F=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=-1)@inv
        maximum=max(maximum,float(np.linalg.svd(F,compute_uv=False)[:,0].max()))
    assert maximum<preview['strain_limit']
    result=dict(passed=True,completed_steps=len(rows),preview_stop_frames=stops,
        strict_converged_steps=preview['strictly_converged_steps'],forced_steps=0,
        maximum_principal_stretch=maximum,strain_limit=preview['strain_limit'],
        strict_control_failed_step=strict['failed_step'],identity_strict_steps=identity['completed_steps'],
        pre_preview_prefix_max_position_difference=prefix_error,
        independent_collision_audit=False,probe_evidence=probes)
    (root/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='probe_evidence'},indent=2))


if __name__=='__main__':main()
