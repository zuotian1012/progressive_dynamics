"""Verify the saved PSQ/VelPro predictor against its actual lower trajectory."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import load_npz
from progressive.prolongation import ShellProlongation
from progressive.integrators import predict
from shirt_demo.hierarchy import validate_level

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(root):
    m=json.loads((root/'run.json').read_text());assert m['status']!='running'
    level=m['level'];last=validate_level(m['parameters']['assets'],level)
    finest=level==last
    assert m.get('finest_level',last)==last and level>0 and m['method']==f'psq_velpro_L{level}'
    low=Path(m['lower_run']);lm=json.loads((low/'run.json').read_text())
    assert lm['completed'] and lm['level']==level-1 and not lm.get('force_updates')
    if m.get('event_fallback_enabled'):
        from shirt_demo.solver_policy import validate_parent, converged_step
        validate_parent(lm,json.loads((low/'steps.json').read_text()),m['event_runtime_sha256'])
        assert m.get('requires_convergence') and all(converged_step(r['native']) for r in json.loads((root/'steps.json').read_text()))
    assert digest(low/'run.json')==m['lower_run_sha256'] and digest(low/'positions.npy')==m['lower_sha256']
    expected_config=copy.deepcopy(lm['config'])
    budget=m['parameters'].get('line_search_max_iterations')
    if budget is not None:
        assert 8<=budget<=128 and m['solver_overrides']['line_search.max_iter']==budget
        expected_config['line_search']['max_iter']=budget
    preconditioner=m['parameters'].get('fem_preconditioner')
    if preconditioner is not None:
        assert preconditioner in ('diag','mas') and m['solver_overrides']['linear_system.fem_preconditioner']==preconditioner
        expected_config['linear_system']['fem_preconditioner']=preconditioner
    assert expected_config==m['config'] and lm['material_parameters']==m['material_parameters']
    assert lm['robot_q_sha256']==m['robot_q_sha256']
    # Isolated packages may have different absolute paths but identical DLLs.
    lower_libraries={Path(path).name:h for path,h in lm['runtime_hashes'].items()}
    current_libraries={Path(path).name:h for path,h in m['runtime_hashes'].items()}
    assert len(lower_libraries)==len(lm['runtime_hashes']) and len(current_libraries)==len(m['runtime_hashes'])
    assert lower_libraries.keys()==current_libraries.keys()
    changed_libraries=[path for path,h in m['runtime_hashes'].items() if lower_libraries[Path(path).name]!=h]
    if finest:
        assert not changed_libraries and m.get('finest_requires_convergence') and not m.get('coarse_preview')
    if changed_libraries:
        assert m.get('reject_inf_trials') and m.get('trial_energy_policy')
        assert all(Path(path).name=='uipc_backend_cuda.dll' for path in changed_libraries)
    assert lm['parameters'].get('robot_drive_scale',1)==m['parameters'].get('robot_drive_scale',1)
    assets=Path(m['parameters']['assets']);map_path=assets/f'map_{level-1}_{level}.npz'
    assert digest(map_path)==m['map_sha256']
    U=load_npz(map_path)
    coarse=np.load(assets/f'mesh_{level-1}.npz');fine=np.load(assets/f'mesh_{level}.npz')
    assert digest(assets/f'mesh_{level-1}.npz')==lm['mesh_sha256'] and digest(assets/f'mesh_{level}.npz')==m['mesh_sha256']
    P=ShellProlongation(coarse['vertices'],fine['vertices'],coarse['faces'],U,np.load(assets/f'anchors_{level-1}_{level}.npy'))
    x0=np.load(low/'positions.npy',mmap_mode='r');x1=np.load(root/'positions.npy',mmap_mode='r')
    np.testing.assert_allclose(x0[0],coarse['vertices'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(x1[0],fine['vertices'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(P.position(x0[0]),x1[0],atol=1e-12,rtol=0)
    n=m['completed_steps'];dt=m['dt'];assert lm['completed_steps']>=n and lm['dt']==dt
    injected=np.load(root/'injected_velocities.npy',mmap_mode='r')
    assert injected.shape==(m['steps'],len(fine['vertices']),3)
    errors=[];predict_errors=[]
    for k in range(n):
        expected=np.zeros_like(x1[0]) if k==0 else U@((x0[k]-x0[k-1])/dt)
        errors.append(float(np.max(np.abs(injected[k]-expected))))
        reference=predict('velpro',1,k,[x0,x1],[P],dt,[np.zeros_like(x0[0]),np.zeros_like(x1[0])])
        predict_errors.append(float(np.max(np.abs(x1[k]+dt*injected[k]-reference))))
    assert max(errors,default=0)<=1e-12 and max(predict_errors,default=0)<=1e-12
    records=json.loads((root/'steps.json').read_text());assert len(records)==n
    if finest:
        assert all(r['native']['converged'] and not r['native']['preview_non_descent'] and not r['native']['hit_newton_limit'] and not r['native']['hit_line_search_limit'] for r in records)
    assert all(r['lower_velocity_time_index']==i and r['velocity_readback_error']<=1e-12 and r['position_injection_error']<=1e-12 for i,r in enumerate(records))
    fine_operators=[]
    for meta in (lm,m):
        assert digest(meta['psq_bundle'])==meta['psq_bundle_sha256']
        bundle=json.loads(Path(meta['psq_bundle']).with_suffix('.json').read_text())
        with np.load(bundle['source_operator']) as op:fine_operators.append((op['fine_rest'].copy(),op['fine_faces'].copy()))
    for j in (0,1):np.testing.assert_array_equal(fine_operators[0][j],fine_operators[1][j])
    result=dict(steps=n,lower_run=str(low),all_injected_velocities_verified=True,
        solver_budget_override=budget,linear_preconditioner_override=preconditioner,
        maximum_velocity_formula_error=max(errors,default=0),maximum_predictor_error_vs_existing_algorithm=max(predict_errors,default=0),
        maximum_native_velocity_readback_error=max((r['velocity_readback_error'] for r in records),default=0),
        maximum_position_change_during_injection=max((r['position_injection_error'] for r in records),default=0),
        linear_rest_mismatch_m=float(np.max(np.abs(U@x0[0]-x1[0]))),psq_rest_matches=True,
        shared_finest_proxy_matches_lower=True,changed_runtime_libraries=changed_libraries,reject_inf_trials=m.get('reject_inf_trials',False),lower_newton_limit_steps=lm['newton_limit_steps'],
        note='The approved lower trajectory can include nonconverged preview steps; transferred velocities inherit those states. Fine cloth positions are never overwritten with prolongated coarse positions.')
    (root/'report').mkdir(exist_ok=True);(root/'report/velocity_transfer_audit.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();print(json.dumps(audit(a.run),indent=2))
