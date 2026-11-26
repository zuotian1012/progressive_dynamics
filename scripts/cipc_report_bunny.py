"""Audit and visualize accepted native C-IPC frames, never unwritten states."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

def vertices(path):
    rows=[]
    with path.open() as stream:
        for line in stream:
            if line.startswith('v '): rows.append(line[2:])
            elif line.startswith('f '): break
    return np.loadtxt(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--assets',type=Path,default=Path('output/bunny_paper_assets'))
    p.add_argument('--accepted-prefix',action='store_true')
    p.add_argument('--frames',type=int,default=61)
    a=p.parse_args();root=a.root
    meta=json.loads((root/'run.json').read_text())
    if not meta['completed'] and not a.accepted_prefix:raise ValueError('Use --accepted-prefix for incomplete run')
    count=meta['completed_steps']
    if count<1:raise ValueError('No accepted steps to report')
    steps=[json.loads(s) for s in (root/'steps.jsonl').read_text().splitlines()][:count]
    if len(steps)!=count or any(s['step']!=i+1 or not s['converged'] or max(s['last_three_rms_velocity_decrements'])>1e-4 for i,s in enumerate(steps)):
        raise ValueError('Invalid accepted step log')
    cloth=np.load(a.assets/'mesh_7.npz');bunny=np.load(a.assets/'bunny.npz')
    rest,faces=cloth['vertices'],cloth['faces'];bv,bf=bunny['vertices'],bunny['faces']
    dm=np.stack((rest[faces[:,1],:2]-rest[faces[:,0],:2],rest[faces[:,2],:2]-rest[faces[:,0],:2]),axis=2)
    inv=np.linalg.inv(dm)
    selected=set(np.unique(np.linspace(0,count,min(a.frames,count+1)).astype(int)))
    saved={};stretch=[];fixed_errors=[];bounds=[]
    # OBJ uses upstream's %le output (7 significant digits), not full doubles.
    for i in range(count+1):
        world=vertices(root/f'raw/shell{i}.obj')
        if world.shape!=(len(rest)+len(bv),3) or not np.isfinite(world).all():raise ValueError(f'Bad frame {i}')
        x=world[:len(rest)]
        err=float(np.max(np.abs(world[len(rest):]-bv)));fixed_errors.append(err)
        if err>1e-6:raise ValueError('Fixed bunny moved')
        ds=np.stack((x[faces[:,1]]-x[faces[:,0]],x[faces[:,2]]-x[faces[:,0]]),axis=2)
        maximum=float(np.linalg.svd(ds@inv,compute_uv=False)[:,0].max());stretch.append(maximum)
        if i in selected:saved[i]=x;bounds.extend([x.min(axis=0),x.max(axis=0)])
    native_stretch=np.loadtxt(root/'raw/stretch.txt',ndmin=2)[:count,0]
    counters=np.loadtxt(root/'raw/counter.txt',ndmin=2)[:count]
    audit={'completed':meta['completed'],'accepted_steps':count,'physical_seconds':count*.01,
        'newton_converged_steps':count,'preview_stopped_steps':0,
        'native_max_principal_stretch':float(native_stretch.max()),
        'recomputed_max_principal_stretch_rounded_obj':max(stretch),
        'strain_limit':1.068,'native_strain_pass':bool(np.all(native_stretch<1.068)),
        'max_fixed_bunny_coordinate_error_m':max(fixed_errors),
        'physics_seconds':meta['physics_seconds'],'wall_seconds':meta['wall_seconds'],
        'setup_seconds':meta.get('setup_seconds'),
        'frames_with_native_active_contact':int(np.count_nonzero(counters[:,1])),
        'max_newton_iterations':max(s['newton_iterations'] for s in steps),
        'scope':'All accepted OBJ frames finite, fixed-obstacle and strain checks; native solver logs checked. OBJ rounded to 7 significant digits. No independent continuous CCD certification.',
        'limitations':meta['limitations']}
    out=root/('report' if meta['completed'] else 'report_prefix');out.mkdir(exist_ok=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    t=np.arange(1,count+1)*.01
    axes[0].plot(t,native_stretch);axes[0].axhline(1.068,color='r',ls='--');axes[0].set_title('Native maximum principal stretch')
    axes[1].plot(t,[s['newton_iterations'] for s in steps]);axes[1].set_title('Native Newton iterations')
    axes[2].plot(t,[s['seconds'] for s in steps]);axes[2].set_title('Solve wall time per step (s)')
    for ax in axes:ax.set_xlabel('Physical time (s)');ax.grid(alpha=.25)
    fig.savefig(out/'metrics.png',dpi=150);plt.close(fig)
    points=np.vstack(bounds+[bv.min(axis=0),bv.max(axis=0)])
    center=(points.min(axis=0)+points.max(axis=0))/2;radius=np.ptp(points,axis=0).max()*.55
    images=[]
    for i,x in sorted(saved.items()):
        fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(projection='3d')
        triangles=[];face_colors=[]
        for v,f,col in [(bv,bf,np.array([.65,.67,.7])),(x,faces,np.array([.29,.62,.85]))]:
            tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-20)
            light=np.array([-.3,-.5,1.]);light/=np.linalg.norm(light)
            colors=(.45+.55*np.abs(n@light))[:,None]*col
            triangles.append(tri);face_colors.append(colors)
        # A single collection depth-sorts cloth and bunny triangles together.
        ax.add_collection3d(Poly3DCollection(np.concatenate(triangles),facecolors=np.concatenate(face_colors),edgecolors='none'))
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.view_init(18,-65);ax.set_box_aspect((1,1,1));ax.set_axis_off()
        status='COMPLETE' if meta['completed'] else 'INCOMPLETE — accepted prefix only'
        if meta['steps']!=400:status=f'RUN CHECK ONLY ({count}/{meta["steps"]} steps)'
        ax.set_title(f'Native C-IPC | 90,000 cloth vertices\nt = {i*.01:.2f} s | {status}\nReconstructed bunny scene',fontsize=11)
        fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    images[-1].save(out/'final.png')
    images[0].save(out/'trajectory.gif',save_all=True,append_images=images[1:],duration=max(40,int(count*10/len(images))),loop=0)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
