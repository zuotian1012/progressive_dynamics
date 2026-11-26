"""Check continuation provenance, isolated files, and every exact-gradient retry."""
from pathlib import Path
import argparse, hashlib, json, re, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]

from bunny_runtime import audit_bunny_contact, check_run_sources, runtime_provenance

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    m=json.loads((a.run/'run.json').read_text())
    assert m['completed'] and 0<m['completed_steps']==m['steps'] and m['scene']=='bunny' and 1<=m['level']<=7
    assert m['dt']==.01 and m['pins']==0 and m['proxy_vertices']==90000
    assert 0<m['config']['newton']['velocity_tol']==m.get('bunny_newton_velocity_tol',.05)<=.05
    assert 0<m['config']['linear_system']['tol_rate']==m.get('bunny_linear_tol_rate',1e-3)<=1e-3 and m['config']['line_search']['max_iter']==8
    assert m['ground']['height_m']==0 and m['ground']['cloth_friction']==0 and not m['ground']['bunny_ground_contact']
    verified_contact=audit_bunny_contact(a.run,m)
    assert m['adaptive_line_search'] and m['non_descent_probe_refinement']
    frozen=m.get('frozen_offset_only',False)
    assert sum((frozen,m['exact_offset_gradient'],m['initial_non_descent_rescue']))==1 and not m['tiny_probe']
    if frozen or (m['exact_offset_gradient'] and m['level']<7):
        log=a.run.with_suffix('.log').read_text(encoding='utf-8-sig',errors='replace')
        assert ('FLAG_PSQ_EXACT_OFFSET enabled='+('false' if frozen else 'true')) in log
        assert not m['unresolved_direction_rescue']
        if frozen:
            assert 'FLAG_PSQ_RESCUE frame=' not in log and 'FLAG_PSQ_UNRESOLVED_RESCUE frame=' not in log
    prov=runtime_provenance(m['native_package'])
    assert m.get('checked_search_exhaustion',False)==prov.get('checked_search_exhaustion',False)
    assert m.get('bunny_newton_norm','max_coordinate')==prov.get('bunny_newton_norm','max_coordinate')
    assert m.get('velocity_only_preserves_friction',False)==prov.get('velocity_only_preserves_friction',False)
    for source,h in prov['source_hashes'].items():assert digest(source)==h,source
    assert m['library_sha256']['uipc_backend_cuda.dll']==prov['dll_sha256']
    assert digest(Path(m['native_package'])/'uipc/_native/uipc_backend_cuda.dll')==prov['dll_sha256']
    lower=Path(m['lower_run']);old=json.loads((lower/'run.json').read_text())
    assert m.get('requested_contact_stiffness',1e9)==old.get('requested_contact_stiffness',1e9)
    assert old['completed'] and old['completed_steps']==old['steps'] and m['level']==old['level']+1 and m['method'] in ('velpro','semi','full')
    assert m['velpro_transfer_active']==(m['method']=='velpro')
    if m['method']=='velpro' and m['consistency_weight']==0:
        assert old.get('consistency_weight',0.)==0.
        assert old['level']==0 or (old['method']=='velpro' and old.get('velocity_map','intrinsic')==m.get('velocity_map','intrinsic'))
    if m.get('velocity_only_preserves_friction') and old['level']:
        assert old.get('velocity_only_preserves_friction')
        for library,previous in old['library_sha256'].items():
            if library=='uipc_backend_cuda.dll' and m['level']==7 and previous==prov.get('linear_solver_parent_sha256'):continue
            if library=='uipc_backend_cuda.dll' and m.get('representable_preview') and previous==prov.get('reference_dll_sha256'):continue
            assert m['library_sha256'][library]==previous
    if m['method'] in ('semi','full'):
        assert m['consistency_weight']==old.get('consistency_weight',0.)==0.
        assert old['level']==0 or old['method']==m['method'] or (old['level']==1 and {old['method'],m['method']}=={'semi','full'})
        if m['method']=='full' and m['level']>=2:
            assert m['grandparent_sha256']==old['lower_sha256']==digest(Path(m['grandparent_run'])/'positions.npy')
    assert json.loads((lower/'continuation_audit.json').read_text())['passed']
    assert m['lower_sha256']==digest(lower/'positions.npy')
    assert m['map_sha256']==digest(Path(m['assets'])/f'map_{old["level"]}_{m["level"]}.npz')
    assert m['lower_backend_sha256']==old['library_sha256']['uipc_backend_cuda.dll']
    old_cfg=json.loads(json.dumps(old['config']))
    budgets=m['newton_iteration_budgets']
    assert budgets['previous_level']==old_cfg['newton']['max_iter']
    assert budgets['current_level']==m['config']['newton']['max_iter']
    old_cfg['newton']['max_iter']=budgets['current_level']
    if 'bunny_newton_velocity_tol' in m:
        tol=m['bunny_newton_velocity_tol']
        assert 0<tol<=old_cfg['newton']['velocity_tol']
        assert budgets['tolerance_unchanged']==(tol==old_cfg['newton']['velocity_tol'])
        old_cfg['newton']['velocity_tol']=tol
    for key,record in [('fem_preconditioner','linear_preconditioners'),('use_cuda_graph','cuda_graph_modes')]:
        if record in m:
            assert m[record]['previous_level']==old_cfg['linear_system'][key]
            assert m[record]['current_level']==m['config']['linear_system'][key]
            old_cfg['linear_system'][key]=m[record]['current_level']
    assert old_cfg==m['config']
    assert m['steps']<=old['completed_steps']
    for k in ['assets','material','dt','ground','contact_friction']:
        assert m[k]==old[k],k
    check_run_sources(a.run,m['source_files_sha256'])
    protected=json.loads((a.run.parent/'protected_manifest.json').read_text())
    changed=[p for p,h in protected.items() if not Path(p).is_file() or digest(p)!=h]
    assert not changed,changed
    task_protected=protected;preexisting=[]
    triggers={};accepts={};restored_bases={};refined=False;search=False
    stabilized=bool(prov.get('linear_solver_parent_sha256'))
    final_residuals={};shifts=[];retries=0
    with a.run.with_suffix('.log').open(encoding='utf-8-sig',errors='replace') as log:
        for line in log:
            assert 'Line Search Exits with Max Iteration' not in line, line
            assert prov.get('checked_search_exhaustion') or 'PSQ_PREVIEW_REJECT frame=' not in line, line
            refined |= 'FLAG_PROBE_REFINE ' in line
            search |= 'FLAG_SEARCH_TRIAL ' in line
            if stabilized and 'PCG_' in line:
                values=dict(re.findall(r'(\w+)=([^\s;]+)',line))
                if 'PCG_TRUE_RESIDUAL ' in line:
                    assert float(values['tolerance'])==m['config']['linear_system']['tol_rate']
                    final_residuals[(int(values['frame']),int(values['newton']))]=float(values['ratio'])
                elif 'PCG_STABILIZED_SOLVE ' in line:shifts.append(float(values['relative_diagonal_shift']))
                elif 'PCG_ROUNDOFF_RETRY ' in line:retries+=1
            if 'FLAG_PSQ_' not in line:continue
            d=dict(re.findall(r'(\w+)=([^\s]+)',line))
            if 'FLAG_PSQ_UNRESOLVED_RESCUE ' in line:
                key=(int(d['frame']),int(d['retry']));assert key not in triggers
                assert int(d['retry'])==int(d['newton'])+1 and d['consistent']=='false'
                assert abs(float(d['E_retained'])-float(d['E_base']))<=float(d['noise'])
                triggers[key]='unresolved'
                restored_bases[key]=(float(d['E_base']),float(d['noise']))
            elif 'FLAG_PSQ_RESCUE frame=' in line:
                key=(int(d['frame']),1);assert key not in triggers;triggers[key]='first_iteration'
            elif 'FLAG_PSQ_RESCUE_ACCEPT ' in line:
                key=(int(d['frame']),int(d['newton']));assert key not in accepts
                assert float(d['E1'])<=float(d['E0']) and 0<float(d['alpha'])<=float(d['ccd'])<=1
                if key in restored_bases:
                    base,noise=restored_bases[key];assert abs(float(d['E0'])-base)<=noise
                accepts[key]=float(d['E1'])<float(d['E0'])
    assert triggers.keys()==accepts.keys()
    if stabilized:
        assert final_residuals and all(0<=v<=m['config']['linear_system']['tol_rate'] for v in final_residuals.values())
        (a.run/'linear_solver_audit.json').write_text(json.dumps(dict(
            checked_solves=len(final_residuals),maximum_final_residual_ratio=max(final_residuals.values()),
            roundoff_retries=retries,solves_with_diagonal_shift=len(shifts),max_relative_shift=max(shifts,default=0.)),indent=2))
    assert search or not prov.get('energy_checked_convergence')
    for script,needed in [('audit_flag_refined_probes.py',refined),('audit_flag_fine_search.py',search)]:
        if needed:
            extra=['--check-converged-energy'] if search and script=='audit_flag_fine_search.py' and prov.get('energy_checked_convergence') else []
            subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(a.run),*extra],check=True)
    audit=json.loads((a.run/'audit.json').read_text())
    assert audit['accepted_prefix_checks_passed'] and audit['completed_requested_run'] and not audit['zero_motion_steps']
    assert audit['velocity_readback_error']==0 and audit['position_injection_error']==0
    if m['level']<7:
        preview=json.loads((a.run/'report/non_descent_audit.json').read_text())
        assert preview['all_accepted_preview_evidence_verified'] and not preview['preview_at_first_newton_frames']
    else:assert m['native_steps']==m['steps'] and m['preview_steps']==0
    import numpy as np,ipctk
    x=np.load(a.run/'positions.npy',mmap_mode='r')
    assert len(x)==m['steps']+1
    rows=json.loads((a.run/'steps.json').read_text());assert max(r['obstacle_error'] for r in rows)==0
    assert not m.get('diagnostic_restart'), 'A completed comparison requires a continuous run from the initial state'
    minimum_z=float(x[:,:,2].min());assert minimum_z>=m['material']['thickness']/2-1e-9
    mesh=np.load(Path(m['assets'])/f'mesh_{m["level"]}.npz');b=np.load(Path(m['assets'])/'bunny.npz')
    rest=np.vstack([mesh['vertices'],b['vertices']]);faces=np.vstack([mesh['faces'],b['faces']+len(mesh['vertices'])]).astype(np.int32)
    edges=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    cm=ipctk.CollisionMesh(rest,edges,faces)
    intersections=[k for k,state in enumerate(x) if ipctk.has_intersections(cm,np.vstack([state,b['vertices']]))]
    assert not intersections,intersections
    contact_z=m['material']['thickness']/2+m['config']['contact']['d_hat']
    touching=np.flatnonzero(np.min(x[:,:,2],axis=1)<contact_z)
    ground_result=dict(all_saved_states_floor_clearance_passed=True,minimum_vertex_height_m=minimum_z,
        half_thickness_m=m['material']['thickness']/2,first_floor_activation_time=float(touching[0]*m['dt']) if len(touching) else None,
        final_vertices_within_contact_activation=int(np.sum(x[-1,:,2]<contact_z)),final_bounds=[x[-1].min(0).tolist(),x[-1].max(0).tolist()],
        static_intersection_states_checked=len(x),intersecting_states=intersections,fixed_bunny_error=0.,
        limitation='All saved states checked; not an independent continuous-time collision certificate')
    (a.run/'ground_audit.json').write_text(json.dumps(ground_result,indent=2))
    result=dict(passed=True,level=m['level'],protected_files_checked=len(protected),protected_files_changed_since_historical_baselines=changed,preexisting_differences=preexisting,task_start_files_checked=len(task_protected),task_start_files_changed=[],
        rescue_events=len(triggers),unresolved_rescues=sum(v=='unresolved' for v in triggers.values()),
        strict_energy_decreases=sum(accepts.values()),all_rescues_energy_nonincreasing=True,
        method=m["method"],velpro_transfer_active=m["velpro_transfer_active"],shirt_aligned_solver_tolerances=True,ground_added=True,no_unhandled_line_search_exhaustion=True)
    if 'bunny_newton_velocity_tol' in m:
        result.update(newton_velocity_tol=m['bunny_newton_velocity_tol'],linear_tol_rate=m['bunny_linear_tol_rate'],
                      shirt_aligned_solver_tolerances=m['bunny_newton_velocity_tol']==.05 and m['bunny_linear_tol_rate']==1e-3)
    result['native_contact']=verified_contact
    (a.run/'continuation_audit.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
