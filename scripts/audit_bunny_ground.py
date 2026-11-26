"""Check continuation provenance, isolated files, and every exact-gradient retry."""
from pathlib import Path
import argparse, hashlib, json, re, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]

from bunny_runtime import audit_bunny_contact, check_run_sources, runtime_provenance

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    p.add_argument('--identity-replay',action='store_true',help='Audit same-mesh velocity injection against its recorded reference')
    a=p.parse_args()
    m=json.loads((a.run/'run.json').read_text())
    assert m['completed'] and 0<m['completed_steps']==m['steps'] and m['scene']=='bunny' and m['level']==0
    assert m['dt']==.01 and m['pins']==0 and m['proxy_vertices']==90000
    assert m['velpro_transfer_active']==a.identity_replay
    if a.identity_replay:
        reference=Path(m['replay_reference'])
        assert m['method']=='velpro_identity_replay' and digest(reference/'positions.npy')==m['replay_sha256']
        assert json.loads((reference/'continuation_audit.json').read_text())['passed']
        parent=json.loads((reference/'run.json').read_text())
        assert parent['level']==0 and parent['bundle_sha256']==m['bundle_sha256']
        assert parent['config']==m['config'] and parent['material']==m['material'] and parent['ground']==m['ground']
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
    manifest=a.run.parent/'protected_manifest.json'
    if manifest.exists() or (a.run/'source_snapshot.py').is_file():
        protected=json.loads(manifest.read_text()) if manifest.exists() else {}
        changed=[p for p,h in protected.items() if not Path(p).is_file() or digest(p)!=h]
        assert not changed,changed
        check_run_sources(a.run,m['source_files_sha256'])
        preexisting=[]
        task_protected=protected
    else:
        protected={}
        for rel in ['flag_L0_repair_20261118/protected_manifest.json','flag_L1_fix_20261118/protected_additional.json','flag_L2_fix_20261118/protected_additional.json','flag_fine_20261118/protected_additional.json','bunny_ground_L0_20261118/protected_additional.json']:
            protected.update(json.loads((ROOT/'output'/rel).read_text()))
        changed=[p for p,h in protected.items() if not Path(p).is_file() or digest(p)!=h]
        # A shirt edit predates this task; preserve it and record the older-baseline difference.
        start=(ROOT/'output/bunny_ground_L0_20261118/protected_additional.json').stat().st_mtime
        known=ROOT/'shirt_demo/simulate.py'
        preexisting=[p for p in changed if Path(p)==known and Path(p).stat().st_mtime<start]
        assert set(changed)==set(preexisting),changed
        task_protected=json.loads((ROOT/'output/bunny_ground_L0_20261118/protected_additional.json').read_text())
        assert all(digest(p)==h for p,h in task_protected.items())
    triggers={};accepts={};restored_bases={};refined=False;search=False
    with a.run.with_suffix('.log').open(encoding='utf-8-sig',errors='replace') as log:
        for line in log:
            assert 'Line Search Exits with Max Iteration' not in line, line
            assert prov.get('checked_search_exhaustion') or 'PSQ_PREVIEW_REJECT frame=' not in line, line
            refined |= 'FLAG_PROBE_REFINE ' in line
            search |= 'FLAG_SEARCH_TRIAL ' in line
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
    assert search or not prov.get('energy_checked_convergence')
    for script,needed in [('audit_flag_refined_probes.py',refined),('audit_flag_fine_search.py',search)]:
        if needed:
            extra=['--check-converged-energy'] if search and script=='audit_flag_fine_search.py' and prov.get('energy_checked_convergence') else []
            subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(a.run),*extra],check=True)
    audit=json.loads((a.run/'audit.json').read_text())
    assert audit['accepted_prefix_checks_passed'] and audit['completed_requested_run'] and not audit['zero_motion_steps']
    assert audit['velocity_readback_error']==0 and audit['position_injection_error']==0
    preview=json.loads((a.run/'report/non_descent_audit.json').read_text())
    assert preview['all_accepted_preview_evidence_verified'] and not preview['preview_at_first_newton_frames']
    import numpy as np,ipctk
    x=np.load(a.run/'positions.npy',mmap_mode='r')
    assert len(x)==m['steps']+1
    rows=json.loads((a.run/'steps.json').read_text());assert max(r['obstacle_error'] for r in rows)==0
    minimum_z=float(x[:,:,2].min());assert minimum_z>=m['material']['thickness']/2-1e-9
    mesh=np.load(Path(m['assets'])/'mesh_0.npz');b=np.load(Path(m['assets'])/'bunny.npz')
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
        velpro_transfer_active=m['velpro_transfer_active'],shirt_aligned_solver_tolerances=True,ground_added=True,no_unhandled_line_search_exhaustion=True)
    if 'bunny_newton_velocity_tol' in m:
        result.update(newton_velocity_tol=m['bunny_newton_velocity_tol'],linear_tol_rate=m['bunny_linear_tol_rate'],
                      shirt_aligned_solver_tolerances=m['bunny_newton_velocity_tol']==.05 and m['bunny_linear_tol_rate']==1e-3)
    target='identity_replay_audit.json' if a.identity_replay else 'continuation_audit.json'
    result['native_contact']=verified_contact
    (a.run/target).write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
