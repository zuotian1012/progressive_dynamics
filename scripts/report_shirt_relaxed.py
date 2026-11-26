"""Check default-policy shirt motion before extending the first fold."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shirt_demo.report import load_run,draw,plt,Image
from progressive.prolongation import frames
from analyze_shirt_runs import analyze
from shirt_demo.assets import Robot,LINKS
from audit_shirt_preview import audit as audit_preview
from audit_shirt_velocity_transfer import audit as audit_transfer
from shirt_demo.hierarchy import validate_level

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--metrics-only',action='store_true')
p.add_argument('--rebuilt-hierarchy',action='store_true',help='Verify rebuilt assets against their audit and the unchanged finest reference')
p.add_argument('--revised-finest',action='store_true',help='Verify an explicitly recorded finest-mesh revision; requires rebuilt hierarchy')
a=p.parse_args();root=a.run
assert not a.revised_finest or a.rebuilt_hierarchy
m=json.loads((root/'run.json').read_text());n=m['completed_steps']
assert m['status']!='running'
last=validate_level(m['parameters']['assets'],m['level'])
finest=m['level']==last
assert m.get('finest_level',last)==last
data=load_run(root);x=data['x'][:n+1];poses=data['robot'][:n+1]
assert np.isfinite(x).all() and np.isfinite(poses).all()
old=json.loads(Path('output/shirt55k_L0_first_fold/run.json').read_text())
trajectory=m.get('trajectory_path')
if trajectory:
    assert hashlib.sha256(Path(trajectory).read_bytes()).hexdigest()==m['trajectory_sha256']
    assert hashlib.sha256(Robot(trajectory).q.tobytes()).hexdigest()==m['robot_q_sha256']
else:
    assert m['robot_q_sha256']==old['robot_q_sha256']
assert m['material_parameters']==old['material_parameters'] and m['config']['contact']==old['config']['contact']
transfer_audit=None
if a.rebuilt_hierarchy:
    assets=Path(m['parameters']['assets']);assert json.loads((assets/'audit.json').read_text())['passed']
    mesh_path=assets/f'mesh_{m["level"]}.npz'
    assert hashlib.sha256(mesh_path.read_bytes()).hexdigest()==m['mesh_sha256']
    assert hashlib.sha256(Path(m['psq_bundle']).read_bytes()).hexdigest()==m['psq_bundle_sha256']
    current_bundle=json.loads(Path(m['psq_bundle']).with_suffix('.json').read_text())
    prior_bundle=json.loads(Path(old['psq_bundle']).with_suffix('.json').read_text())
    with np.load(current_bundle['source_operator']) as op,np.load(prior_bundle['source_operator']) as prior,np.load(mesh_path) as mesh:
        np.testing.assert_array_equal(op['coarse_rest'],mesh['vertices']);np.testing.assert_array_equal(op['coarse_faces'],mesh['faces'])
        if a.revised_finest:
            revision=json.loads((assets/'geometry_revision.json').read_text());assert revision['passed']
            assert hashlib.sha256((assets/f'mesh_{last}.npz').read_bytes()).hexdigest()==revision['revised_finest_mesh_sha256']
            original_path=Path(revision['original_finest_mesh'])
            assert hashlib.sha256(original_path.read_bytes()).hexdigest()==revision['original_finest_mesh_sha256']
            with np.load(original_path) as original,np.load(assets/f'mesh_{last}.npz') as revised:
                np.testing.assert_array_equal(original['vertices'],prior['fine_rest']);np.testing.assert_array_equal(original['faces'],prior['fine_faces'])
                np.testing.assert_array_equal(revised['vertices'],op['fine_rest']);np.testing.assert_array_equal(revised['faces'],op['fine_faces'])
        else:
            np.testing.assert_array_equal(op['fine_rest'],prior['fine_rest']);np.testing.assert_array_equal(op['fine_faces'],prior['fine_faces'])
        np.testing.assert_array_equal(data['x'][0],mesh['vertices']);np.testing.assert_array_equal(data['f'],mesh['faces'])
    assert current_bundle['parameters']==prior_bundle['parameters']
if m['level']==0:
    if not a.rebuilt_hierarchy:
        assert m['mesh_sha256']==old['mesh_sha256'] and m['psq_bundle_sha256']==old['psq_bundle_sha256']
else:
    assert 0<m['level']<=last and m.get('lower_run')
    transfer_audit=audit_transfer(root)
defaults=json.loads((root/'native_default_config.json').read_text())
expected=copy.deepcopy(defaults)
disable_semi=m['parameters'].get('disable_semi_implicit',False)
expected_overrides={}
if disable_semi:
    expected_overrides['newton.semi_implicit.enable']=0
    expected['newton']['semi_implicit']['enable']=0
budget=m['parameters'].get('line_search_max_iterations')
if budget is not None:
    assert m.get('relaxed_solver') and m.get('psq_bundle') and 8<=budget<=128
    expected_overrides['line_search.max_iter']=budget
    expected['line_search']['max_iter']=budget
preconditioner=m['parameters'].get('fem_preconditioner')
if preconditioner is not None:
    assert m.get('relaxed_solver') and m.get('psq_bundle') and preconditioner in ('diag','mas')
    expected_overrides['linear_system.fem_preconditioner']=preconditioner
    expected['linear_system']['fem_preconditioner']=preconditioner
assert m['solver_overrides']==expected_overrides
for key in ['newton','line_search','linear_system','extras','cfl','integrator']:assert m['config'][key]==expected[key]
policy_label='Semi-implicit OFF; default tolerance' if disable_semi else 'Native default stop policy'
preview_enabled=bool(m.get('coarse_preview',False))
if preview_enabled:policy_label=f'L{m["level"]} non-descent preview; semi-implicit OFF'
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in m['runtime_hashes'].items())
records=json.loads((root/'steps.json').read_text());assert len(records)==n
assert all(r['native']['completed'] and r['native']['frame']==i for i,r in enumerate(records,1))
assert not m.get('force_updates')
if finest:
    assert m.get('finest_requires_convergence') and not preview_enabled and disable_semi
    assert all(r['native']['converged'] and not r['native']['preview_non_descent'] and not r['native']['hit_newton_limit'] and not r['native']['hit_line_search_limit'] for r in records)
log=root.with_suffix('.log').read_text(encoding='utf-8-sig',errors='replace')
assert 'PSQ_FORCE_ACCEPT frame=' not in log
if not preview_enabled:assert 'PSQ_PREVIEW_STOP frame=' not in log
preview_audit=audit_preview(root) if preview_enabled else None
inf_trials=[dict(frame=int(f),newton=int(it),search=int(ls),alpha=float(alpha)) for f,it,ls,alpha in re.findall(r'PSQ_REJECT_INF_TRIAL frame=(\d+) newton=(\d+) search=(\d+) alpha=([^\s]+)',log)]
if inf_trials:
    assert m.get('reject_inf_trials')
    assert all(0<t['alpha']<=1 and t['search']<m['config']['line_search']['max_iter'] and 1<=t['frame']<=n+(not m['completed']) for t in inf_trials)
checks={};key=None
for line in log.splitlines():
    match=re.search(r'Frame (\d+) Newton Iteration (\d+)',line)
    if match:key=tuple(map(int,match.groups()))
    match=re.search(r'MaxTranslationChecker> Residual/AbsTol: ([^/]+)/([^\s]+)',line)
    if key and match:checks[key]=tuple(map(float,match.groups()))
semi=[];tol=[];other=[];preview_frames=[]
for r in records:
    key=(r['step'],r['native']['newton_iterations']-1)
    if r['native'].get('preview_non_descent'):preview_frames.append(r['step'])
    elif r['native']['converged'] and key not in checks:semi.append(r['step'])
    elif r['native']['converged'] and key in checks and checks[key][0]<=checks[key][1]:tol.append(r['step'])
    else:other.append(r['step'])
inv=frames(x[0],data['f'])[2];stretch=[];minimum_sigma=[]
for z in x:
    tri=z[data['f']];F=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=-1)@inv
    singular=np.linalg.svd(F,compute_uv=False)
    stretch.append(float(singular[:,0].max()));minimum_sigma.append(float(singular[:,1].min()))
robot=Robot(trajectory);targets=[];tracking=[]
raw_vertices=[np.load(root/f'collider_{i}.npz')['vertices'] for i in range(len(LINKS))]
surface_tracking=[]
settle=m['parameters']['settle_steps'];substeps=m['parameters']['substeps']
for step in range(n):
    if step<settle:q=robot.q[0]
    else:
        f,s=divmod(step-settle,substeps);previous=robot.q[max(0,f-1)]
        q=previous+(s+1)/substeps*(robot.q[f]-previous)
    target=robot.poses(q);t=np.stack([target[name] for name in LINKS]);targets.append(t)
    tracking.append(np.linalg.norm(poses[step+1,:,:3,3]-t[:,:3,3],axis=1))
    surface_tracking.append([float(np.linalg.norm(v@(poses[step+1,j,:3,:3]-t[j,:3,:3]).T+poses[step+1,j,:3,3]-t[j,:3,3],axis=1).max()) for j,v in enumerate(raw_vertices)])
tracking=np.asarray(tracking)
surface_tracking=np.asarray(surface_tracking)
np.testing.assert_allclose(tracking,np.array([r['body_origin_tracking_errors_m'] for r in records]),atol=1e-12,rtol=0)
grip=analyze(root)['grip_region_tracking'] if m['completed'] and n>=786 and not trajectory else None
screen=dict(completed_requested_motion=m['completed'],strain_below_limit=max(stretch)<m['material_parameters']['strain_limit'],rank_above_floor=min(minimum_sigma)>1e-12,
            all_frames_native_converged=all(r['native']['converged'] for r in records),
            body_origin_tracking_within_10mm=bool(tracking.max()<=.01),
            body_surface_tracking_within_10mm=bool(surface_tracking.max()<=.01),
            both_grips_close_lift_and_hold=bool(grip and all(g['nearest_tcp_distance_at_close_m']<=.03 and g['peak_mean_region_lift_m']>=.02 and g['max_wrist_relative_drift_m']<=.02 for g in grip.values())))
if preview_enabled:
    screen.pop('all_frames_native_converged')
    screen['all_frames_reached_declared_stop']=not preview_audit['unclassified_nonconverged_frames']
    screen['all_preview_exits_verified']=preview_audit['all_preview_evidence_verified']
if trajectory:
    screen.pop('both_grips_close_lift_and_hold')
audit=dict(completed=m['completed'],completed_steps=n,requested_steps=m['steps'],physical_seconds=n*m['dt'],physics_seconds=m['physics_seconds'],
           stop_reason=m.get('stop_reason'),longest_newton_limit_streak=m.get('longest_newton_limit_streak'),
           unchanged_scene_material_and_bundle=m['level']==0 and not a.rebuilt_hierarchy and not trajectory and m['parameters'].get('robot_drive_scale',1.)==1.,
           unchanged_cloth_material_contact_and_bundle=m['level']==0 and not a.rebuilt_hierarchy,
           rebuilt_hierarchy_verified=a.rebuilt_hierarchy,revised_finest_verified=a.revised_finest,modified_robot_trajectory=bool(trajectory),velocity_transfer_audit=transfer_audit,robot_drive_strength=m.get('robot_drive_strength',[300.,1000.]),native_solver_matches_defaults=not disable_semi,
           native_solver_matches_declared_control=True,solver_overrides=m.get('solver_overrides',{}),
           native_converged_steps=sum(bool(r['native']['converged']) for r in records),
           line_search_limit_frames=[r['step'] for r in records if r['native']['hit_line_search_limit']],
           newton_limit_frames=[r['step'] for r in records if r['native']['hit_newton_limit']],
           semi_implicit_stop_frames_inferred=semi,default_tolerance_stop_frames=tol,other_stop_frames=other,
           preview_exit_frames=preview_frames,preview_stopped_steps=len(preview_frames),preview_audit=preview_audit,
           rejected_infinite_trials=inf_trials,rejected_infinite_trial_count=len(inf_trials),
           maximum_principal_stretch=max(stretch),strain_limit=m['material_parameters']['strain_limit'],
           minimum_singular_value=min(minimum_sigma),rank_threshold=1e-12,
           maximum_body_origin_tracking_error_m=float(tracking.max()),tracking_error_p95_m=float(np.percentile(tracking,95)),
           maximum_body_surface_tracking_error_m=float(surface_tracking.max()),
           final_body_origin_tracking_errors_m=tracking[-1].tolist(),grip_region_tracking=grip,original_grasp_schedule_applies=not bool(trajectory),
           screening=screen,screening_passed=all(screen.values()),
           screening_note='Local geometric screening: body origins and collider surfaces <=10mm; closure nearest vertex <=30mm; mean of nearest 8 vertices lifts >=20mm with wrist-relative drift <=20mm. Not paper criteria or contact-force proof.',
           independent_collision_audit=False,strict_newton_convergence_claim=False)
out=root/'report';out.mkdir(exist_ok=True)
(out/'audit.json').write_text(json.dumps(audit,indent=2))
if preview_audit:(out/'preview_audit.json').write_text(json.dumps(preview_audit,indent=2))
np.save(out/'maximum_stretch.npy',stretch);np.save(out/'body_origin_tracking_error.npy',tracking)
np.save(out/'body_surface_tracking_error.npy',surface_tracking)
if a.metrics_only:
    print(json.dumps({k:audit[k] for k in ['completed_steps','native_converged_steps','preview_stopped_steps','maximum_principal_stretch','minimum_singular_value','maximum_body_surface_tracking_error_m','grip_region_tracking','screening','screening_passed']},indent=2))
    sys.exit(0)
fig,ax=plt.subplots(2,1,figsize=(10,6),layout='constrained',sharex=True)
ax[0].plot(np.arange(n+1)*m['dt'],stretch);ax[0].axhline(audit['strain_limit'],color='r',ls='--');ax[0].set_ylabel('Maximum principal stretch')
ax[1].plot(np.arange(1,n+1)*m['dt'],tracking*1000);ax[1].axhline(10,color='r',ls='--');ax[1].set(xlabel='Physical time (s)',ylabel='Body origin target error (mm)')
fig.savefig(out/'metrics.png',dpi=140);plt.close(fig)
fig=plt.figure(figsize=(12,10))
for j,(name,k) in enumerate([('Initial',0),('Settled',60),('Closure',498),('Lift end',786)]):
    reached=k<=n;k=min(k,n)
    title=f'{name} | step {k} | t={k*m["dt"]:.2f}s' if reached else f'{name}: NOT REACHED\nLast saved state at step {n}'
    draw(fig.add_subplot(2,2,j+1,projection='3d'),data,k,title,True)
fig.suptitle(f'L{m["level"]} PSQ | {policy_label} | Actual robot and cloth states');fig.tight_layout();fig.savefig(out/'stages.png',dpi=140);plt.close(fig)
images=[]
for k in np.unique(np.linspace(0,n,min(80,n+1)).astype(int)):
    fig=plt.figure(figsize=(7,6));draw(fig.add_subplot(projection='3d'),data,k,f'L{m["level"]} PSQ | {policy_label} | t={k*m["dt"]:.2f}s\nNo custom forced acceptance',True)
    fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
images[-1].save(out/'final.png');images[0].save(out/'motion.gif',save_all=True,append_images=images[1:],duration=max(30,round(n*m['dt']*1000/len(images))),loop=0)
print(json.dumps({k:v for k,v in audit.items() if not k.endswith('_frames') and not k.endswith('_frames_inferred')},indent=2))
