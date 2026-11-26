"""Audit a stopped progressive first fold and render only its saved prefix."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from trimesh.triangles import closest_point
from progressive.prolongation import frames
from shirt_demo.report import load_run,draw,plt,Image
from scripts.check_psq_step_cap import evaluate

p=argparse.ArgumentParser();p.add_argument('lower',type=Path);p.add_argument('fine',type=Path);p.add_argument('--output',type=Path,required=True)
p.add_argument('--skip-animation',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
data=[load_run(a.lower),load_run(a.fine)];m=data[1]['meta'];n=m['completed_steps'];dt=m['dt']
assert m['status']=='stopped_for_diagnosis' and not m['completed'] and n<=data[0]['meta']['completed_steps']
records=json.loads((a.fine/'steps.json').read_text());assert len(records)==n
audit=json.loads((a.fine/'report/audit.json').read_text())
cap=[r for r in records if r['native']['hit_newton_limit']]
log=a.fine.with_suffix('.log').read_text(encoding='utf-8-sig',errors='replace')
residuals={};key=None;active_frame=None;contact_filters={};failed_checkers={}
for line in log.splitlines():
    start=re.search(r'>>> Begin Frame: (\d+)',line)
    if start:active_frame=int(start[1])
    if 'Feasible Step Filter:' in line:contact_filters[active_frame]=contact_filters.get(active_frame,0)+1
    match=re.search(r'Frame (\d+) Newton Iteration (\d+)',line)
    if match:key=tuple(map(int,match.groups()))
    if key and '[x]' in line and 'ABDToleranceChecker>' in line:failed_checkers[key]='ABDToleranceChecker'
    match=re.search(r'MaxTranslationChecker> Residual/AbsTol: ([^/]+)/([^\s]+)',line)
    if match and key:residuals[key]=tuple(map(float,match.groups()))
x=data[1]['x'][:n+1];faces=data[1]['f'];ib=frames(x[0],faces)[2]
last=x[-1][faces];F=np.stack((last[:,1]-last[:,0],last[:,2]-last[:,0]),axis=2)@ib
s=np.linalg.svd(F,compute_uv=False);face=int(s[:,0].argmax());ids=faces[face]
tri=x[:,ids];history=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)@ib[face]
gap=1.1-np.linalg.svd(history,compute_uv=False)[:,0]
target=1.1-64*np.finfo(float).eps*1.1
near=np.flatnonzero(gap<1.1-target)
motion=np.linalg.norm(np.diff(x,axis=0),axis=2).max(1)*1000
caprows=[]
for row in cap:
    k=row['step'];native=row['native'];res=residuals.get((k,native['newton_iterations']-1))
    caprows.append(dict(step=k,final_alpha=native['last_line_search_alpha'],ccd_toi=native['last_ccd_toi'],
        line_search_exhausted=native['hit_line_search_limit'],newton_iterations=native['newton_iterations'],
        line_search_trials=native['line_search_trials'],contact_feasible_filter_events=contact_filters.get(k,0),
        last_failed_checker=failed_checkers.get((k,native['newton_iterations']-1)),
        residual_over_tolerance=res[0]/res[1] if res else None,maximum_step_motion_mm=float(motion[k-1])))

# A controlled direction, NOT an extracted simulation Newton direction.
# This isolates the production helper's near-boundary fallback branch.
test_F=np.array([[[1.1-8*np.finfo(float).eps,0],[0,.2],[0,0.]]])
test_D=-.01*test_F
test_caps=evaluate(test_F,test_D)[0]
path_sigma=np.stack([np.linalg.svd(test_F[0]+t*test_D[0],compute_uv=False) for t in np.linspace(0,1,201)])
assert np.all(path_sigma[:,0]<1.1) and np.all(path_sigma[:,1]>1e-12)
assert 0<test_caps[1]<1e-8
synthetic=dict(description='Constructed inward contraction; not the recorded Newton direction',
    F=test_F[0].tolist(),D=test_D[0].tolist(),legacy_cap=float(test_caps[0]),directional_cap=float(test_caps[1]),
    full_alpha_one_feasible_on_201_samples=True,minimum_path_sigma=float(path_sigma[:,1].min()),
    endpoint_maximum_stretch=float(path_sigma[-1,0]),
    finding='The production directional helper falls back to its direction-independent bound when smax >= limit - 64*eps*max(1,limit), even for a feasible inward contraction.')
closure=[];grip=[]
for j,d in enumerate(data):
    for hand,link in [('left',1),('right',5)]:
        pose=d['robot'][500,link];tcp=pose[:3,:3]@np.array([.155,.001786,.014])+pose[:3,3]
        cp=closest_point(d['x'][500][d['f']],np.broadcast_to(tcp,(len(d['f']),3)))
        closure.append(dict(level=j,hand=hand,step=500,tcp_to_surface_mm=float(np.linalg.norm(cp-tcp,axis=1).min()*1000)))
        pose=d['robot'][498,link];tcp=pose[:3,:3]@np.array([.155,.001786,.014])+pose[:3,3]
        selected=np.argsort(np.linalg.norm(d['x'][498]-tcp,axis=1))[:8];ks=np.arange(498,n+1);poses=d['robot'][ks,link]
        center=d['x'][ks][:,selected].mean(1);local=np.einsum('nij,nj->ni',np.linalg.inv(poses[:,:3,:3]),center-poses[:,:3,3]);delta=np.linalg.norm(local-local[0],axis=1)
        grip.append(dict(level=j,hand=hand,first_step=498,last_step=n,vertex_ids=selected.tolist(),maximum_wrist_relative_drift_mm=float(delta.max()*1000)))
result=dict(completed=False,status=m['status'],completed_steps=n,requested_steps=m['steps'],last_time_s=n*dt,
    interrupted_uncommitted_step=m.get('interrupted_uncommitted_step'),native_converged_steps=m['native_converged_steps'],preview_steps=m['preview_stopped_steps'],
    newton_limit_steps=len(caprows),newton_limit_details=caprows,
    rejected_infinite_trial_count=audit['rejected_infinite_trial_count'],
    all_preview_exits_verified=audit['preview_audit']['all_preview_evidence_verified'],
    velocity_transfer_audit=audit['velocity_transfer_audit'],
    last_state_stretch_diagnostic=dict(face=face,vertex_ids=ids.tolist(),maximum_stretch=float(s[face,0]),
        gap_to_limit=float(gap[-1]),directional_margin_target=target,first_sample_in_margin=int(near[0]) if len(near) else None,
        minimum_sigma_all_faces=float(s[:,1].min()),
        caveat='CPU rest-frame/SVD reconstruction at saved states; not an exact native per-Newton cap capture.'),
    synthetic_production_cuda_cap_test=synthetic,closure_tcp_to_surface=closure,grip_until_stop=grip,
    inference='Repeated Newton caps coincide with a nearly saturated stretch face. No exhausted line-search budget or contact-feasibility filter is recorded during the stall, CCD=1 and there is one trial per Newton iteration; by the advance pipeline the tiny alpha comes from PSQ feasibility limiting. The source fallback is demonstrably over-conservative for synthetic inward motion. Exact binding face and direction still require native cap-audit capture; they are not proven by saved-state analysis.',
    limitations=['No full L1 fold, release or final settling was simulated.',
                 'No native per-Newton direction was saved in this run; synthetic test is not a replay.',
                 'No strain limit, material, trajectory or step-cap policy changed during the diagnostic.'])
(a.output/'diagnosis.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained');steps=np.arange(1,n+1)
ax[0,0].plot(steps,[r['native']['newton_iterations'] for r in records]);ax[0,0].axhline(1024,color='red',ls='--');ax[0,0].set(ylabel='Newton iterations',title='Native iteration limit retained')
alphas=[r['native']['last_line_search_alpha'] if not r['native']['preview_non_descent'] else np.nan for r in records]
ax[0,1].semilogy(steps,alphas);ax[0,1].set(ylabel='Last tested alpha',title='Preview exits excluded (alpha=0)')
ax[1,0].semilogy(np.arange(n+1),np.maximum(gap,np.finfo(float).eps));ax[1,0].axhline(1.1-target,color='red',ls='--');ax[1,0].set(ylabel='1.1 - maximum face stretch',title=f'Face {face}: remaining stretch gap (CPU)')
ax[1,1].plot(steps,motion);ax[1,1].set(ylabel='Maximum cloth step motion (mm)',title='Saved physical-step displacement')
for axis in ax.flat:axis.set(xlabel='Physical step',xlim=(max(0,n-160),n))
fig.savefig(a.output/'stall_metrics.png',dpi=150);plt.close(fig)
fig=plt.figure(figsize=(12,8))
for j,d in enumerate(data):
    for i,(label,k) in enumerate([('Closed',500),('Lift end',788),('Last saved prefix',n)]):
        draw(fig.add_subplot(2,3,j*3+i+1,projection='3d'),d,k,f'L{j} | {label} | {k*dt:.2f}s',True)
fig.suptitle(f'Incomplete L1 first fold: stopped at {n}/{m["steps"]} | Synchronized saved states')
fig.tight_layout();fig.savefig(a.output/'prefix_stages.png',dpi=140);plt.close(fig)
fig=plt.figure(figsize=(11,9))
for j,d in enumerate(data):
    for i in (0,1):
        axis=fig.add_subplot(2,2,i*2+j+1,projection='3d');draw(axis,d,n,f'L{j} at step {n} | {"Overview" if i==0 else "Overhead"}',True)
        if i:axis.view_init(elev=90,azim=-90);axis.set_proj_type('ortho');axis.set_zticks([]);axis.set_zlabel('')
fig.suptitle('Last saved common time; NOT a completed L1 fold')
fig.tight_layout();fig.savefig(a.output/'last_saved_comparison.png',dpi=140);plt.close(fig)
images=[]
for k in ([] if a.skip_animation else np.unique(np.linspace(0,n,100).astype(int))):
    fig=plt.figure(figsize=(11,5),dpi=85)
    for j,d in enumerate(data):draw(fig.add_subplot(1,2,j+1,projection='3d'),d,k,f'L{j} | {len(d["f"]):,} triangles',True)
    fig.suptitle(f'L0 -> L1 prefix | t={k*dt:.2f}s | Stopped at {n}/{m["steps"]}; release not reached')
    fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
if images:images[0].save(a.output/'prefix_comparison.gif',save_all=True,append_images=images[1:],duration=round(n*dt*1000/len(images)/10)*10,loop=0)
print(json.dumps(result,indent=2))
