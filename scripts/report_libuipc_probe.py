"""Inspect saved short libuipc probe only; does not advance simulation."""
import json
import re
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

root = Path(sys.argv[1])
meta = json.loads((root/'run.json').read_text())
positions = np.load(root/'accepted_positions.npy')[:meta['completed_steps']+1]
level=meta.get('level',0)
cloth = np.load(f'output/bunny_paper_assets/mesh_{level}.npz')
bunny = np.load('output/bunny_paper_assets/bunny.npz')
log = root.with_suffix('.log').read_text(encoding='utf-8-sig')
frame = 0
counts, residuals, iterations, forced, last_residuals = {}, {}, {}, {}, {}
preview = {}
for line in log.splitlines():
    m = re.search(r'>>> Begin Frame: (\d+)',line)
    if m: frame=int(m[1])
    m = re.search(r'SimplexTrajectoryFilter PTs: (\d+), EEs: (\d+), PEs: (\d+), PPs: (\d+)',line)
    if m: counts[frame]=max(counts.get(frame,0),sum(map(int,m.groups())))
    m = re.search(r'\[\*\].*Residual/AbsTol: ([^/]+)/([^ ]+)',line)
    if m: residuals[frame]=[float(m[1]),float(m[2])]
    m = re.search(r'Residual/AbsTol: ([^/]+)/([^ ]+)',line)
    if m:last_residuals[frame]=[float(m[1]),float(m[2])]
    m = re.search(r'Newton Iteration Converged with Iteration Count: (\d+)',line)
    if m: iterations[frame]=int(m[1])
    m = re.search(r'PSQ_FORCE_ACCEPT frame=(\d+) reason=(\w+) newton=(\d+) alpha=(\S+)',line)
    if m:
        forced[int(m[1])]=dict(reason=m[2],newton_iterations=int(m[3])+1,alpha=float(m[4]))
        energies=re.search(r'E0=(\S+) E_applied=(\S+)',line)
        if energies:forced[int(m[1])].update(energy_before=float(energies[1]),energy_applied=float(energies[2]))
    m = re.search(r'PSQ_PREVIEW_STOP frame=(\d+) reason=preview_non_descent newton=(\d+)',line)
    if m:preview[int(m[1])]=dict(reason='preview_non_descent',newton_iterations=int(m[2])+1)
n = meta['completed_steps']
iterations={i:v for i,v in iterations.items() if i<=n}
residuals={i:v for i,v in residuals.items() if i<=n}
counts={i:v for i,v in counts.items() if i<=n}
forced={i:v for i,v in forced.items() if i<=n}
preview={i:v for i,v in preview.items() if i<=n}
assert not set(iterations)&set(forced)
assert not set(preview)&(set(iterations)|set(forced))
assert set(iterations)|set(forced)|set(preview)==set(range(1,n+1))
assert all(residuals[i][0]<=residuals[i][1] for i in iterations)
assert not forced or meta.get('force_updates'), 'Unexpected forced acceptance in strict run'
assert not preview or meta.get('coarse_preview'), 'Unexpected preview acceptance in strict run'
newton_counts={i:v+1 for i,v in iterations.items()}
newton_counts.update({i:v['newton_iterations'] for i,v in preview.items()})
newton_counts.update({i:v['newton_iterations'] for i,v in forced.items()})
assert np.isfinite(positions).all()
f=cloth['faces'];rest=cloth['vertices'];tri=rest[f];e=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0]
u=e/np.linalg.norm(e,axis=1)[:,None];norm=np.cross(e,b);norm/=np.linalg.norm(norm,axis=1)[:,None];v=np.cross(norm,u)
dm=np.zeros((len(f),2,2));dm[:,0,0]=np.linalg.norm(e,axis=1);dm[:,0,1]=np.sum(u*b,axis=1);dm[:,1,1]=np.sum(v*b,axis=1);inv=np.linalg.inv(dm)
stretch=[]
for x in positions:
    t=x[f];F=np.stack((t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=2)@inv
    stretch.append(float(np.linalg.svd(F,compute_uv=False)[:,0].max()))
summary=dict(accepted_steps=n,strict_native_converged_steps=len(iterations),preview_stops=len(preview),preview_frames=preview,
    forced_accepted_steps=len(forced),forced_frames=forced,strict_converged_frames=sorted(iterations),
    all_frames_strictly_converged=len(iterations)==n and meta['completed'],
    physical_time=n*.01,max_newton_iterations=max(newton_counts.values(),default=0),
    first_contact_candidate_step=min((i for i,c in counts.items() if c),default=None),
    max_contact_candidates=max(counts.values(),default=0),max_principal_stretch=max(stretch),
    max_accepted_translation_residual=max((v[0] for v in residuals.values()),default=None),
    last_translation_residuals={i:last_residuals[i] for i in range(1,n+1) if i in last_residuals},
    translation_threshold=1e-5,velocity_threshold=.001,convergence_norm='native maximum per-axis displacement / dt, not RMS',
    wall_seconds=meta['wall_seconds'],setup_seconds=meta['setup_seconds'],
    advance_retrieve_seconds=meta['physics_seconds'],psq_enabled=meta.get('psq_enabled',False),
    hard_strain_barrier=meta.get('hard_strain_barrier',False),completed=meta['completed'],
    process_wall_seconds=meta.get('process_wall_seconds'),
    error=meta.get('error'),failed_step=meta.get('failed_step'),
    warnings=[x for x in log.splitlines() if '[warning]' in x or '[error]' in x],
    limitations=meta['differences'])
(root/'audit.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(1,2,figsize=(9,3),layout='constrained')
barrier=meta.get('hard_strain_barrier',False)
axs[0].plot(np.arange(n+1)*.01,stretch);axs[0].axhline(1.068,color='r',ls='--',label='Hard limit' if barrier else 'Reference (not enforced)');axs[0].legend(fontsize=7);axs[0].set_title('Maximum principal stretch')
axs[1].plot(np.arange(1,n+1)*.01,[newton_counts[i] for i in range(1,n+1)])
if forced:
    axs[1].scatter(np.array(list(forced))*.01,[newton_counts[i] for i in forced],color='r',s=20,label='Forced, not converged');axs[1].legend(fontsize=7)
if preview:
    axs[1].scatter(np.array(list(preview))*.01,[newton_counts[i] for i in preview],color='orange',s=20,label='Preview stop, not converged');axs[1].legend(fontsize=7)
axs[1].set_title('Native Newton iterations')
for ax in axs: ax.set_xlabel('Physical time (s)');ax.grid(alpha=.3)
fig.savefig(root/'metrics.png',dpi=130);plt.close(fig)
pts=np.vstack([positions.min(axis=(0,1)),positions.max(axis=(0,1)),bunny['vertices']]);center=(pts.min(0)+pts.max(0))/2;radius=np.ptp(pts,axis=0).max()*.55
images=[]
for i in np.unique(np.linspace(0,n,min(n+1,41 if meta.get('force_updates') else 14)).astype(int)):
    fig=plt.figure(figsize=(6,5));ax=fig.add_subplot(projection='3d');tris=[];colors=[]
    for x,faces,col in [(positions[i],f,[.29,.62,.85]),(bunny['vertices'],bunny['faces'],[.6,.62,.65])]:
        t=x[faces];normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-20)
        tris.append(t);colors.append((.4+.6*np.abs(normal@np.array([-.3,-.4,.866])))[:,None]*col)
    ax.add_collection3d(Poly3DCollection(np.concatenate(tris),facecolors=np.concatenate(colors),edgecolors='none'))
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius));ax.view_init(18,-65);ax.set_box_aspect((1,1,1));ax.set_axis_off()
    mode=f"PSQ {'ON' if meta.get('psq_enabled') else 'OFF'}, strain barrier {'ON' if barrier else 'OFF'}"
    status=f'Completed {n} steps' if meta['completed'] else f'INCOMPLETE: accepted {n}/{meta["requested_steps"]} steps'
    if meta.get('coarse_preview'):
        state='Initial' if i==0 else ('PREVIEW STOP, NOT CONVERGED' if i in preview else 'Newton converged')
        status=f'{state} | preview stops: {sum(k<=i for k in preview)} | {n}/{meta["requested_steps"]} steps'
        if not meta['completed']:status='INCOMPLETE | '+status
    if meta.get('force_updates'):
        mode='EXPERIMENT: forced updates; collision/strain caps ON'
        state='Initial' if i==0 else ('FORCED, NOT CONVERGED' if i in forced else 'Newton converged')
        status=f'{state} | forced so far: {sum(k<=i for k in forced)} | {n}/{meta["requested_steps"]} steps'
    if meta.get('force_updates'):
        ax.set_title(f'libuipc PSQ | L0 1443 -> proxy 90000 | t={i*.01:.2f}s\n'
                     f'FORCED-UPDATE EXPERIMENT | collision/strain caps ON\n'
                     f'{state} | forced: {sum(k<=i for k in forced)} | saved: {n}/{meta["requested_steps"]}\n'
                     'Actual simulated trajectory (slowed playback)',fontsize=8)
    else:
        method='VelPro from L0' if meta.get('method')=='velpro' else 'independent'
        import textwrap
        status_lines='\n'.join(textwrap.wrap(status,width=75))
        ax.set_title(f'libuipc native | L{level} {len(cloth["vertices"])} vertices | {method} | t={i*.01:.2f}s\n{mode}\n{status_lines}\nActual accepted trajectory (slowed playback)',fontsize=8)
    fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
images[-1].save(root/'final.png');images[0].save(root/'contact.gif',save_all=True,append_images=images[1:],duration=140,loop=0)
print(json.dumps(summary,indent=2))
