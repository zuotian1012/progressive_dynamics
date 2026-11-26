"""Audit relaxed native PSQ/VelPro trajectories; never equate native stop with strict convergence."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

root=Path('output/bunny_pd_relaxed_20261116')
assets=Path('output/bunny_paper_assets')
bunny=np.load(assets/'bunny.npz')
rows=[]; states=[]; meshes=[]

def draw(ax,x,faces,title,bounds=None):
    c=(x.min(0)+x.max(0))/2;r=max(np.ptp(x,axis=0).max(),.3)*.65
    if bounds is not None:c,r=bounds
    triangles=[];colors=[]
    objects=[(x,faces,[.29,.62,.85])]
    if np.max(np.abs(bunny['vertices'].mean(0)-c))<r:
        objects.insert(0,(bunny['vertices'],bunny['faces'],[.6,.62,.65]))
    for v,f,col in objects:
        tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-20)
        triangles.append(tri);colors.append((.4+.6*np.abs(n@np.array([-.3,-.4,.866])))[:,None]*col)
    ax.add_collection3d(Poly3DCollection(np.concatenate(triangles),facecolors=np.concatenate(colors),edgecolors='none'))
    ax.set(xlim=(c[0]-r,c[0]+r),ylim=(c[1]-r,c[1]+r),zlim=(c[2]-r,c[2]+r));ax.view_init(18,-65);ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.set_title(title,fontsize=9)

for level in range(3):
    folder=root/f'L{level}'
    if not (folder/'run.json').exists():break
    m=json.loads((folder/'run.json').read_text());n=m['completed_steps']
    assert m['status']!='running','Wait until simulation exits before reporting'
    default=json.loads((folder/'installed_default_config.json').read_text())
    for k in ['newton','line_search','linear_system','extras','cfl','integrator']:assert m['config'][k]==default[k]
    assert m['psq_enabled'] and m['relaxed_pd'] and m['hard_strain_barrier']
    assert not m['coarse_preview'] and not m.get('force_updates')
    x=np.load(folder/'native_positions.npy');mesh=np.load(assets/f'mesh_{level}.npz');faces=mesh['faces']
    np.testing.assert_array_equal(x[0],mesh['vertices'])
    assert x.shape==(n+1,len(mesh['vertices']),3) and np.isfinite(x).all()
    records=json.loads((folder/'steps.json').read_text());assert len(records)==n
    assert all(r['native']['completed'] and r['native']['frame']==i and r['fixed_error']==0 for i,r in enumerate(records,1))
    injection_error=None
    if level:
        assert m['method']=='velpro'
        lower=states[-1];U=load_npz(assets/f'map_{level-1}_{level}.npz')
        expected=np.zeros((n,len(mesh['vertices']),3))
        for i in range(1,n):expected[i]=U@((lower[i]-lower[i-1])/.01)
        injected=np.load(folder/'injected_velocities.npy')
        injection_error=float(np.max(np.abs(expected-injected)))
        assert injection_error<1e-10
        assert all(r['velocity_readback_error']<1e-12 and r['position_injection_error']<1e-12 for r in records)
    tri=mesh['vertices'][faces];e=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=-1)
    inv=np.linalg.inv(np.linalg.cholesky(e.transpose(0,2,1)@e).transpose(0,2,1))
    stretch=[]
    for z in x:
        t=z[faces];F=np.stack([t[:,1]-t[:,0],t[:,2]-t[:,0]],axis=-1)@inv
        stretch.append(float(np.linalg.svd(F,compute_uv=False)[:,0].max()))
    log=folder.with_suffix('.log').read_text(errors='replace')
    assert 'PSQ_FORCE_ACCEPT' not in log and 'PSQ_PREVIEW_STOP frame=' not in log
    checks={};check_key=None
    for line in log.splitlines():
        match=re.search(r'Frame (\d+) Newton Iteration (\d+)',line)
        if match:check_key=tuple(map(int,match.groups()))
        match=re.search(r'MaxTranslationChecker> Residual/AbsTol: ([^/]+)/([^\s]+)',line)
        if match and check_key:checks[check_key]=tuple(map(float,match.groups()))
    tolerance_stops=[];semi_stops=[];unknown_stops=[]
    for r in records:
        key=(r['step'],r['native']['newton_iterations']-1)
        if key in checks and checks[key][0]<=checks[key][1] and r['native']['converged']:
            tolerance_stops.append(r['step'])
        elif key not in checks and r['native']['converged']:
            semi_stops.append(r['step'])
        else:unknown_stops.append(r['step'])
    ls=[r['step'] for r in records if r['native']['hit_line_search_limit']]
    nl=[r['step'] for r in records if r['native']['hit_newton_limit']]
    row=dict(level=level,vertices=len(mesh['vertices']),proxy_vertices=m['proxy_vertices'],steps=n,
             completed=m.get('completed',False),physical_seconds=n*.01,
             native_converged_steps=sum(bool(r['native']['converged']) for r in records),
             strict_newton_convergence_verified=False,line_search_limit_frames=ls,newton_limit_frames=nl,
             default_tolerance_stop_frames=tolerance_stops,
             semi_implicit_stop_frames_inferred_from_log=semi_stops,unclassified_stop_frames=unknown_stops,
             line_search_limit_events=len(re.findall('Line Search Exits with Max Iteration',log)),
             max_principal_stretch=max(stretch),strain_limit=m['strain_limit'],strain_bound_passed=max(stretch)<m['strain_limit'],
             velocity_prediction_max_error=injection_error,
             max_newton_iterations=max((r['native']['newton_iterations'] for r in records),default=0),
             physics_seconds=sum(r['seconds'] for r in records),wall_seconds=m.get('wall_seconds'),
             final_vertex_centroid_m=x[-1].mean(0).tolist(),
             termination_reason_counts={},independent_collision_audit=False,
             error=m.get('error'),failed_step=m.get('failed_step'))
    for r in records:
        reason=r['native'].get('termination_reason','not_reported')
        row['termination_reason_counts'][reason]=row['termination_reason_counts'].get(reason,0)+1
    (folder/'audit.json').write_text(json.dumps(row,indent=2))
    np.savez_compressed(folder/'trajectory.npz',positions=x,faces=faces,times=np.arange(n+1)*.01,bunny_vertices=bunny['vertices'],bunny_faces=bunny['faces'])
    with (folder/'final_cloth.obj').open('w') as f:
        for v in x[-1]:f.write('v '+' '.join(map(str,v))+'\n')
        for face in faces:f.write('f '+' '.join(map(str,face+1))+'\n')
    fig,ax=plt.subplots(1,2,figsize=(10,3),layout='constrained')
    ax[0].plot(np.arange(n+1)*.01,stretch);ax[0].axhline(m['strain_limit'],color='r',linestyle='--');ax[0].set_title('Max principal stretch; limit 1.068')
    ax[1].plot(np.arange(1,n+1)*.01,[r['native']['newton_iterations'] for r in records]);ax[1].set_title('Native Newton iterations')
    for a in ax:a.set_xlabel('Physical time (s)');a.grid(alpha=.3)
    fig.savefig(folder/'metrics.png',dpi=140);plt.close(fig)
    images=[]
    for frame in np.unique(np.linspace(0,n,min(n+1,41)).astype(int)):
        fig=plt.figure(figsize=(6,5));ax=fig.add_subplot(projection='3d')
        draw(ax,x[frame],faces,f'PD / VelPro L{level} -> 90K proxy | t={frame*.01:.2f}s\nRelaxed native stopping; {n}/400 steps\nCamera follows cloth; distant bunny omitted; mean z={x[frame,:,2].mean():.2f}m')
        fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    images[-1].save(folder/'final.png');images[0].save(folder/'simulation.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
    rows.append(row);states.append(x);meshes.append(mesh)
    print(json.dumps(row),flush=True)

fig=plt.figure(figsize=(14,8))
for row,desired in enumerate([50,400]):
    pts=np.concatenate([bunny['vertices']]+[x[min(desired,len(x)-1)] for x in states]);bounds=((pts.min(0)+pts.max(0))/2,np.ptp(pts,axis=0).max()*.55)
    for level,x in enumerate(states):
        frame=min(desired,len(x)-1);ax=fig.add_subplot(2,3,row*3+level+1,projection='3d')
        draw(ax,x[frame],meshes[level]['faces'],f'PD L{level} | t={frame*.01:.2f}s | {len(x)-1}/400 steps'+(f'\nCloth close-up; mean z={x[frame,:,2].mean():.2f}m' if row else ''),bounds if row==0 else None)
fig.suptitle('Native PD / VelPro | 90K elastic proxies | Default solver settings; strain barrier ON')
fig.tight_layout();fig.savefig(root/'comparison.png',dpi=140);plt.close(fig)
(root/'summary.json').write_text(json.dumps(rows,indent=2))
text=['Relaxed native PD / VelPro L0 -> L1 -> L2',
      'Existing native PSQ code and 90K elastic proxies, current-level contact/mass/strain, dt=0.01, target 400 steps.',
      'Defaults: velocity_tol=0.05, semi_implicit enabled, strict_mode disabled, line_search=8, PCG tol_rate=0.001, Newton cap=1024.',
      'Custom coarse preview and forced-update switches disabled. Material and hard strain barrier retained.',
      'Native converged and termination_reason=newton_converged may include semi-implicit early stopping. Strict convergence not verified.',
      'Trajectory finiteness, initial states, fixed bunny, default solver parameters, strain and inter-level velocity injections audited.',
      'Independent collision/intersection audit not performed. Native CCD remains enabled.',
      'Animations follow cloth; distant bunny omitted. Final comparison row uses cloth close-ups with mean world z labeled.',
      'This is a relaxation experiment on the existing PD implementation, not a proof that its descent direction is corrected.',
      'Direct controls have no hard strain barrier and no PSQ; do not interpret timing/shape differences as isolated method comparisons.',
      '', 'level,completed,steps,LS_flagged_frames,Newton_flagged_frames,max_stretch,physics_s']
for r in rows:text.append(f'L{r["level"]},{r["completed"]},{r["steps"]},{len(r["line_search_limit_frames"])},{len(r["newton_limit_frames"])},{r["max_principal_stretch"]:.9f},{r["physics_seconds"]:.3f}')
(root/'RESULTS.txt').write_text('\n'.join(text)+'\n')
