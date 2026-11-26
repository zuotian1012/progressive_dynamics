"""Check continuation provenance, isolated files, and every exact-gradient retry."""
from pathlib import Path
import argparse, hashlib, json, re, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    m=json.loads((a.run/'run.json').read_text())
    assert m['completed'] and m['completed_steps']==400 and m['scene']=='bunny' and 1<=m['level']<=7
    assert m['dt']==.01 and m['pins']==0 and m['proxy_vertices']==90000 and m['velpro_transfer_active']
    assert m['config']['newton']['velocity_tol']==.05
    assert m['config']['linear_system']['tol_rate']==1e-3 and m['config']['line_search']['max_iter']==8
    assert m['ground']['height_m']==0 and m['ground']['cloth_friction']==0 and not m['ground']['bunny_ground_contact']
    assert m['contact_friction']==dict(cloth_bunny=.3,cloth_self=.3,cloth_ground=0.)
    assert m['adaptive_line_search'] and m['non_descent_probe_refinement'] and m['initial_non_descent_rescue']
    assert not m['exact_offset_gradient'] and not m['tiny_probe']
    prov=json.loads((ROOT/'output/flag_L3_fix_20261118/rescue_build/provenance.json').read_text())
    for source,h in prov['source_hashes'].items():assert digest(source)==h,source
    assert m['library_sha256']['uipc_backend_cuda.dll']==prov['dll_sha256']
    assert digest(Path(m['native_package'])/'uipc/_native/uipc_backend_cuda.dll')==prov['dll_sha256']
    lower=Path(m['lower_run']);old=json.loads((lower/'run.json').read_text())
    assert old['completed'] and old['completed_steps']==400 and m['level']==old['level']+1 and m['method']=='velpro'
    assert json.loads((lower/'continuation_audit.json').read_text())['passed']
    assert m['lower_sha256']==digest(lower/'positions.npy')
    assert m['map_sha256']==digest(Path(m['assets'])/f'map_{old["level"]}_{m["level"]}.npz')
    assert m['lower_backend_sha256']==old['library_sha256']['uipc_backend_cuda.dll']
    for k in ['assets','material','dt','steps','config','ground','contact_friction']:
        assert m[k]==old[k],k
    for source,h in m['source_files_sha256'].items():assert digest(source)==h,source
    protected=json.loads((a.run.parent/'protected_manifest.json').read_text())
    changed=[p for p,h in protected.items() if not Path(p).is_file() or digest(p)!=h]
    concurrent=json.loads((a.run.parent/'concurrent_changes.json').read_text())
    assert set(changed)==set(concurrent),changed
    for source,record in concurrent.items():
        assert Path(source)==ROOT/'shirt_demo/simulate.py'
        assert record['task_start_sha256']==protected[source] and record['observed_sha256']==digest(source)
    task_protected=protected;preexisting=[]
    triggers={};accepts={};restored_bases={};refined=False;search=False
    with a.run.with_suffix('.log').open(encoding='utf-8-sig',errors='replace') as log:
        for line in log:
            assert 'Line Search Exits with Max Iteration' not in line, line
            assert 'PSQ_PREVIEW_REJECT frame=' not in line, line
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
    for script,needed in [('audit_flag_refined_probes.py',refined),('audit_flag_fine_search.py',search)]:
        if needed:subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(a.run)],check=True)
    audit=json.loads((a.run/'audit.json').read_text())
    assert audit['accepted_prefix_checks_passed'] and audit['completed_requested_run'] and not audit['zero_motion_steps']
    assert audit['velocity_readback_error']==0 and audit['position_injection_error']==0
    if m['level']<7:
        preview=json.loads((a.run/'report/non_descent_audit.json').read_text())
        assert preview['all_accepted_preview_evidence_verified'] and not preview['preview_at_first_newton_frames']
    else:assert m['native_steps']==400 and m['preview_steps']==0
    import numpy as np,ipctk
    x=np.load(a.run/'positions.npy',mmap_mode='r')[:401]
    rows=json.loads((a.run/'steps.json').read_text());assert max(r['obstacle_error'] for r in rows)==0
    minimum_z=float(x[:,:,2].min());assert minimum_z>=m['material']['thickness']/2-1e-9
    mesh=np.load(Path(m['assets'])/f'mesh_{m["level"]}.npz');b=np.load(Path(m['assets'])/'bunny.npz')
    rest=np.vstack([mesh['vertices'],b['vertices']]);faces=np.vstack([mesh['faces'],b['faces']+len(mesh['vertices'])]).astype(np.int32)
    edges=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    cm=ipctk.CollisionMesh(rest,edges,faces)
    intersections=[k for k,state in enumerate(x) if ipctk.has_intersections(cm,np.vstack([state,b['vertices']]))]
    assert not intersections,intersections
    contact_z=m['material']['thickness']/2+m['config']['contact']['d_hat']
    touching=np.flatnonzero(np.min(x[:,:,2],axis=1)<contact_z)
    assert len(touching),'No ground contact occurred'
    ground_result=dict(all_saved_states_floor_clearance_passed=True,minimum_vertex_height_m=minimum_z,
        half_thickness_m=m['material']['thickness']/2,first_floor_activation_time=float(touching[0]*m['dt']),
        final_vertices_within_contact_activation=int(np.sum(x[-1,:,2]<contact_z)),final_bounds=[x[-1].min(0).tolist(),x[-1].max(0).tolist()],
        static_intersection_states_checked=len(x),intersecting_states=intersections,fixed_bunny_error=0.,
        limitation='All saved states checked; not an independent continuous-time collision certificate')
    (a.run/'ground_audit.json').write_text(json.dumps(ground_result,indent=2))
    result=dict(passed=True,level=m['level'],protected_files_checked=len(protected),protected_files_changed_since_historical_baselines=changed,preexisting_differences=preexisting,task_start_files_checked=len(task_protected),task_start_files_changed=changed,concurrent_changes_preserved=concurrent,file_protection_all_unchanged=False,
        rescue_events=len(triggers),unresolved_rescues=sum(v=='unresolved' for v in triggers.values()),
        strict_energy_decreases=sum(accepts.values()),all_rescues_energy_nonincreasing=True,
        velpro_transfer_active=True,shirt_aligned_solver_tolerances=True,ground_added=True,no_unhandled_line_search_exhaustion=True)
    (a.run/'continuation_audit.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
