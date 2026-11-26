"""Verify and visualize stock-solver controls against prior strict runs."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

root = Path(sys.argv[1]).resolve()
baseline = Path(sys.argv[2]).resolve()
assets = Path('output/bunny_paper_assets')
bunny = np.load(assets/'bunny.npz')
rows, trajectories, references, meshes = [], [], [], []

def differences(a,b,path=''):
    result={}
    for k in a.keys() | b.keys():
        p=path+'/'+k
        if isinstance(a.get(k),dict) and isinstance(b.get(k),dict):
            result.update(differences(a[k],b[k],p))
        elif a.get(k)!=b.get(k): result[p]={'before':a.get(k),'after':b.get(k)}
    return result

def draw(ax,x,faces,title,bounds=None):
    triangles, colors = [], []
    for v,f,color in [(bunny['vertices'],bunny['faces'],[.6,.62,.65]),(x,faces,[.29,.62,.85])]:
        tri=v[f]
        norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-20)
        triangles.append(tri)
        colors.append((.4+.6*np.abs(norm@np.array([-.3,-.4,.866])))[:,None]*color)
    ax.add_collection3d(Poly3DCollection(np.concatenate(triangles),facecolors=np.concatenate(colors),edgecolors='none'))
    pts=np.concatenate([bunny['vertices'],x])
    center=(pts.min(0)+pts.max(0))/2
    radius=np.ptp(pts,axis=0).max()*.55
    if bounds is not None: center,radius=bounds
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
    ax.view_init(18,-65); ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.set_title(title,fontsize=10)

for level in range(3):
    folder=root/f'L{level}'
    m=json.loads((folder/'run.json').read_text())
    default=json.loads((folder/'installed_default_config.json').read_text())
    old=json.loads((baseline/f'L{level}'/'run.json').read_text())
    records=json.loads((folder/'steps.json').read_text())
    old_records=json.loads((baseline/f'L{level}'/'steps.json').read_text())
    assert m['completed'] and m['completed_steps']==400 and len(records)==400
    assert m['default_solver'] and not m['psq_enabled'] and not m.get('isolated_source')
    assert m['assets']==old['assets'] and m['material']==old['material'] and m['package']==old['package']
    for key in ['newton','line_search','linear_system','extras','cfl','integrator']:
        assert m['config'][key]==default[key],key
    delta=differences(old['config'],m['config'])
    assert all(k.startswith(('/newton/','/line_search/','/linear_system/','/extras/strict_mode/')) for k in delta)
    (folder/'config_difference.json').write_text(json.dumps(delta,indent=2))
    mesh=np.load(assets/f'mesh_{level}.npz'); faces=mesh['faces']
    x=np.load(folder/'native_positions.npy')
    ref=np.load(baseline/f'L{level}'/'accepted_positions.npy')
    assert x.shape==(401,len(mesh['vertices']),3) and np.isfinite(x).all()
    np.testing.assert_array_equal(x[0],mesh['vertices'])
    assert all(r['native']['completed'] and r['native']['frame']==i and r['fixed_error']==0 for i,r in enumerate(records,1))
    log=folder.with_suffix('.log').read_text(encoding='utf-8-sig')
    warnings=[s for s in log.splitlines() if '[warning]' in s or '[error]' in s]
    ls_frames=[r['step'] for r in records if r['native']['hit_line_search_limit']]
    newton_frames=[r['step'] for r in records if r['native']['hit_newton_limit']]
    # Validate diagnostic counters independently against the native log.
    assert len(re.findall('Line Search Exits with Max Iteration',log))>=len(ls_frames)
    assert len(re.findall('Newton Iteration Exits with Max Iteration',log))==len(newton_frames)
    tri=mesh['vertices'][faces]; e=tri[:,1]-tri[:,0]; b=tri[:,2]-tri[:,0]
    u=e/np.linalg.norm(e,axis=1)[:,None]; normal=np.cross(e,b)
    normal/=np.linalg.norm(normal,axis=1)[:,None]; v=np.cross(normal,u)
    dm=np.zeros((len(faces),2,2)); dm[:,0,0]=np.linalg.norm(e,axis=1)
    dm[:,0,1]=np.sum(u*b,axis=1); dm[:,1,1]=np.sum(v*b,axis=1); inv=np.linalg.inv(dm)
    stretches=[]
    for state in x:
        t=state[faces]; F=np.stack((t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=2)@inv
        stretches.append(float(np.linalg.svd(F,compute_uv=False)[:,0].max()))
    n=old['completed_steps']
    rms=np.sqrt(np.mean(np.sum((x[:n+1]-ref)**2,axis=2),axis=1))
    row=dict(level=level,vertices=len(mesh['vertices']),completed_steps=400,physical_seconds=4.,
             native_converged_steps=sum(bool(r['native']['converged']) for r in records),
             convergence_meaning='Default native policy including semi-implicit termination, not strict Newton verification',
             line_search_limit_frames=ls_frames,newton_limit_frames=newton_frames,warnings=warnings,
             wall_seconds=m['wall_seconds'],physics_seconds=m['physics_seconds'],
             max_newton_iterations=max(r['native']['newton_iterations'] for r in records),
             max_principal_stretch=max(stretches),baseline_steps=n,baseline_failed_step=old['failed_step'],
             common_duration_seconds=n*.01,
             common_default_physics_seconds=sum(r['seconds'] for r in records[:n]),
             common_strict_physics_seconds=sum(r['seconds'] for r in old_records),
             common_final_vertex_rms_distance=float(rms[-1]),
             max_common_vertex_rms_distance=float(rms.max()))
    (folder/'audit.json').write_text(json.dumps(row,indent=2))
    np.savez_compressed(folder/'trajectory.npz',positions=x,faces=faces,times=np.arange(401)*.01,
                        bunny_vertices=bunny['vertices'],bunny_faces=bunny['faces'])
    with (folder/'final_cloth.obj').open('w') as f:
        for p in x[-1]: f.write('v '+' '.join(map(str,p))+'\n')
        for t in faces: f.write('f '+' '.join(map(str,t+1))+'\n')
    fig,ax=plt.subplots(1,3,figsize=(13,3),layout='constrained')
    ax[0].plot(np.arange(401)*.01,stretches); ax[0].set_title('Maximum principal stretch')
    ax[1].plot(np.arange(1,401)*.01,[r['native']['newton_iterations'] for r in records]); ax[1].set_title('Native Newton iterations')
    ax[2].plot(np.arange(n+1)*.01,rms); ax[2].set_title('Vertex RMS distance to strict run (m)')
    for a in ax: a.set_xlabel('Time (s)'); a.grid(alpha=.3)
    fig.savefig(folder/'metrics.png',dpi=140); plt.close(fig)
    images=[]
    for frame in range(0,401,10):
        fig=plt.figure(figsize=(6,5)); ax=fig.add_subplot(projection='3d')
        draw(ax,x[frame],faces,f'L{level} | Default solver | t={frame*.01:.2f} s\nNative termination policy; camera auto-fits cloth + bunny')
        fig.tight_layout(); fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())); plt.close(fig)
    images[-1].save(folder/'final.png')
    images[0].save(folder/'simulation.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
    rows.append(row); trajectories.append(x); references.append(ref); meshes.append(mesh)
    print(f'L{level}: verified, exported; line-search flags={len(ls_frames)}',flush=True)

fig=plt.figure(figsize=(14,9))
frame=50
pts=np.concatenate([bunny['vertices']]+[x[frame] for x in trajectories+references])
bounds=((pts.min(0)+pts.max(0))/2,np.ptp(pts,axis=0).max()*.55)
for row,(name,states) in enumerate([('Previous strict settings',references),('Installed solver defaults',trajectories)]):
    for level in range(3):
        ax=fig.add_subplot(2,3,row*3+level+1,projection='3d')
        draw(ax,states[level][frame],meshes[level]['faces'],f'{name} | L{level} | t=0.50 s',bounds)
fig.suptitle('Same scene and material; solver settings changed | Stock pyuipc 0.0.28')
fig.tight_layout(); fig.savefig(root/'comparison.png',dpi=140); plt.close(fig)
(root/'summary.json').write_text(json.dumps(rows,indent=2))
text=['Default solver control vs previous strict settings',
      'Same original meshes, material, contact settings, gravity and dt=0.01 s; 400 steps per level.',
      'Newton, line_search, linear_system, extras, CFL and integrator verified equal to installed Scene.default_config().',
      'Native converged includes semi-implicit early termination. No strict Newton convergence claim.',
      'No custom backend, PSQ, custom strain barrier, coarse injection, or custom forced update.',
      'Camera auto-fits each animation frame. Comparison PNG uses shared bounds at t=0.50 s.',
      'Per-level config_difference.json records every configuration change from the strict baseline.',
      'Timings are single runs, not repeated performance benchmarks.', '',
      'level,steps,native_converged,line_search_limit_frames,newton_limit_frames,physics_seconds,max_stretch']
for r in rows:
    text.append(f'L{r["level"]},400,{r["native_converged_steps"]},{len(r["line_search_limit_frames"])},{len(r["newton_limit_frames"])},{r["physics_seconds"]:.6f},{r["max_principal_stretch"]:.6f}')
text+=['','Reproduce from repository root (new output directories):',
       '.venv/Scripts/python.exe scripts/probe_libuipc_bunny.py --default-solver --level 0 --steps 400 --step-budget 400 --wall-budget 3600 --output <output>/L0',
       'Repeat levels 1 and 2; redirect logs to sibling L0.log/L1.log/L2.log.',
       '.venv/Scripts/python.exe scripts/report_bunny_default_control.py <output> <strict-baseline>']
(root/'RESULTS.txt').write_text('\n'.join(text)+'\n')
print(json.dumps(rows,indent=2))
