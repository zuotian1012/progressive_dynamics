"""Render the existing accepted L0 trajectory without rerunning simulation."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=Path('output/bunny_paper_scale/full/full_L0'))
parser.add_argument('--output',type=Path,default=Path('output/bunny_l0_review'))
parser.add_argument('--accepted-prefix',action='store_true')
args=parser.parse_args()
source, out = args.source, args.output
out.mkdir(exist_ok=True)
meta = json.loads((source/'run.json').read_text())
n = meta['completed_steps']
x = np.load(source/'positions.npy', mmap_mode='r')[:n+1]
faces = np.load(source/'faces.npy')
with np.load(source/'fixed_collider_0.npz') as b:
    bunny, bf = b['vertices'], b['faces']
solver = json.loads((source/'solver.json').read_text())[:n]
metrics = json.loads((source/'metrics.json').read_text())[:n]
if not meta['completed'] and not args.accepted_prefix:
    raise ValueError('Incomplete level requires --accepted-prefix')
assert n>0 and len(solver)==n and len(metrics)==n
assert all(r.get('accepted',False) for r in solver)
assert np.isfinite(x).all()
summary = dict(source=str(source.resolve()), reused_existing_trajectory=True,
    level=meta['level'],level_completed=meta['completed'], entire_experiment_completed=False, accepted_steps=n,
    newton_converged=sum(r['converged'] for r in solver),
    preview_stops=sum(r.get('preview_stop', False) for r in solver),
    physical_seconds=n*meta['dt'], physics_seconds=meta['physics_seconds'],
    max_stretch=max(m['max_stretch'] for m in metrics),
    final_cloth_bounds=[x[-1].min(0).tolist(), x[-1].max(0).tolist()],
    floor_present=False, timing_scope=meta['timing_scope'],
    scope='Selected-level shared-finest PSQ proxy preview; not strict Newton convergence or paper reproduction',
    accepted_physics_seconds=float(np.load(source/'timing.npz')['step_seconds'][:n].sum()),
    failure=meta.get('error'))
summary['sha256'] = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                     for name in ['faces.npy', 'run.json', 'solver.json']}
summary['sha256']['accepted_positions_float64_bytes'] = hashlib.sha256(x.tobytes()).hexdigest()
(out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

def render(frame, center, radius, label):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection='3d')
    # One collection sorts cloth and bunny together by depth.
    triangles = np.concatenate((x[frame][faces], bunny[bf]))
    normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True), 1e-20)
    light = np.array([-.5,-.7,1.]); light /= np.linalg.norm(light)
    base = np.vstack((np.tile([.38,.65,.88], (len(faces),1)), np.tile([.7,.7,.7], (len(bf),1))))
    colors = base*(.55+.4*np.abs(normals @ light))[:,None]
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors='none'))
    ax.set(xlim=(center[0]-radius,center[0]+radius), ylim=(center[1]-radius,center[1]+radius),
           zlim=(center[2]-radius,center[2]+radius))
    ax.view_init(elev=15,azim=-65); ax.set_box_aspect((1,1,1)); ax.set_axis_off()
    status='REST' if frame==0 else ('Newton converged' if solver[frame-1]['converged'] else 'PREVIEW EARLY STOP')
    ax.set_title(f'PSQ L{meta["level"]} | {meta["vertices"]:,} vertices | step {frame}/{meta["steps"]}\n'
                 f't = {frame*.01:.2f} s | {status}\n{label}', fontsize=11)
    fig.tight_layout(); fig.canvas.draw()
    result = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
    plt.close(fig)
    return result

for name, last, samples in [('contact',min(120,n),min(41,n+1)),('complete' if meta['completed'] else 'accepted_prefix',n,min(61,n+1))]:
    points = np.vstack((x[:last+1].reshape(-1,3),bunny))
    lo,hi = points.min(0),points.max(0);center=(lo+hi)/2;radius=(hi-lo).max()*.56
    label = (f'Excerpt (0-{last*.01:.2f} s)' if name=='contact' else 'Accepted trajectory; fixed world camera; no floor')
    if not meta['completed']:label='INCOMPLETE: accepted states only\n'+label
    images=[render(int(k),center,radius,label) for k in np.unique(np.linspace(0,last,samples).astype(int))]
    images[-1].save(out/f'{name}_final.png')
    images[0].save(out/f'{name}.gif',save_all=True,append_images=images[1:],duration=100,loop=0)

fig, axes = plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
t=np.arange(1,n+1)*.01
axes[0].plot(t,[m['max_stretch'] for m in metrics]);axes[0].axhline(1.068,color='r',ls='--');axes[0].set_title('Maximum principal stretch')
axes[1].plot(np.arange(n+1)*.01,x.mean(1)[:,2]);axes[1].set_title('Cloth mean height (m); no floor')
axes[2].plot(t,[r['iterations'] for r in solver]);axes[2].scatter(t,[r['iterations'] if r.get('preview_stop') else np.nan for r in solver],s=5,c='r',label='Preview stop');axes[2].legend();axes[2].set_title('Newton iterations / step')
for ax in axes:ax.set_xlabel('Physical time (s)');ax.grid(alpha=.3)
fig.savefig(out/'metrics.png',dpi=140);plt.close(fig)
print(json.dumps(summary,indent=2))
