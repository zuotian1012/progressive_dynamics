"""Render actual accepted L0 states, never substitute a PSQ fine reconstruction."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_hanging_shirt import acceleration


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--still-only',action='store_true')
    args=parser.parse_args();p=args.run
    m=json.loads((p/'run.json').read_text())
    mesh=np.load(p/'mesh.npz');rest=mesh['vertices'];faces=mesh['faces'];pins=mesh['pins']
    x=np.load(p/'positions.npy',mmap_mode='r')[:m['completed_steps']+1]
    records=json.loads((p/'steps.json').read_text())
    report=p/'report';report.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig=plt.figure(figsize=(12,7.2),dpi=100,facecolor='#f4f7fa')
    ax=fig.add_axes([.01,.17,.64,.68],projection='3d')
    side=fig.add_axes([.66,.20,.33,.61],projection='3d')
    meshes=[]
    fc=rest[faces].mean(1)
    stripe=(np.floor((fc[:,0]+.5)/.07).astype(int)+np.floor(fc[:,2]/.07).astype(int))%2
    base=np.where(stripe[:,None]==0,np.array([.07,.48,.70]),np.array([.12,.65,.77]))
    pinpoints=rest[pins]
    for a,azim in [(ax,-68),(side,8)]:
        a.set_facecolor('#f4f7fa')
        a.set_proj_type('ortho');a.view_init(elev=15,azim=azim)
        a.set_xlim(-.56,.56);a.set_ylim(-.5,.5);a.set_zlim(.25,1.18)
        a.set_box_aspect((1.12,1.,.93),zoom=1.50 if a is ax else 1.35);a.set_axis_off()
        poly=Poly3DCollection(x[0][faces],linewidths=.12,edgecolors=(.03,.18,.25,.22))
        a.add_collection3d(poly);meshes.append(poly)
        a.scatter(pinpoints[:,0],pinpoints[:,1],pinpoints[:,2],s=24,c='#e66a38',depthshade=False,zorder=10)
        # Visual support only: actual constraints are exactly the orange vertices.
        ztop=1.13
        a.plot([-.25,.25],[0,0],[ztop,ztop],c='#485c70',lw=3)
        for center in mesh['pin_centers']:
            q=rest[center];a.plot([q[0],q[0]],[0,q[1]],[ztop,q[2]],c='#485c70',lw=1.2)
    fig.text(.04,.94,'HANGING SHIRT',fontsize=24,fontweight='bold',color='#16364e')
    mode='L0 semi-implicit preview' if m.get('semi_implicit_preview') else 'L0 physical simulation'
    fig.text(.04,.895,mode+'  |  1,785 vertices / 3,500 triangles',color='#405b70',fontsize=12)
    fig.text(.69,.84,'SIDE VIEW',fontsize=11,fontweight='bold',color='#405b70')
    fig.text(.04,.12,'Orange: fixed shoulder patches   |   Fine PSQ mesh: elastic proxy only',fontsize=10,color='#405b70')
    extent=f'{m["completed_steps"]*m["dt"]:.2f} s / {m["steps"]*m["dt"]:.2f} s'
    fig.text(.04,.155,('Completed preview: ' if m['completed'] else 'INCOMPLETE accepted prefix: ')+extent+'  |  Gravity ramp: 0–2 s; forcing starts at 2 s',fontsize=9,color='#405b70')
    label=fig.text(.04,.065,'',fontsize=12,color='#16364e')
    status=fig.text(.68,.065,'',fontsize=10,color='#405b70')
    light=np.array([.25,-.8,.55]);light/=np.linalg.norm(light)
    def update(i):
        tri=x[i][faces];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
        colors=base*(.78+.22*np.abs(n@light))[:,None]
        for poly in meshes:poly.set_verts(tri);poly.set_facecolor(colors)
        t=i*m['dt'];wind=acceleration(t,m.get('wind_amplitude',3.))[1]
        label.set_text(f't = {t:5.2f} s     Smooth forcing: {wind:+.2f} m/s²')
        state='initial state' if i==0 else ('coarse preview stop' if records[i-1]['native']['preview_non_descent'] else ('native preview stop' if m.get('semi_implicit_preview') else 'Newton converged'))
        status.set_text(state)
        fig.canvas.draw()
    update(len(x)-1);fig.savefig(report/'preview.png',dpi=130,facecolor=fig.get_facecolor())
    if not args.still_only:
        indices=np.unique(np.r_[np.arange(0,len(x),5),len(x)-1])
        gif=[]
        with imageio.get_writer(report/'hanging_L0.mp4',fps=24,codec='libx264',quality=8,macro_block_size=1) as writer:
            for j,i in enumerate(indices):
                update(i)
                frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                writer.append_data(frame)
                # GIF at 12 fps; MP4 retains the full 24 fps rendering.
                if j%2==0:gif.append(np.asarray(Image.fromarray(frame).resize((960,576),Image.Resampling.LANCZOS)))
                if j%48==0:print(f'RENDER {j}/{len(indices)}',flush=True)
        imageio.mimsave(report/'hanging_L0.gif',gif,duration=1000/12,loop=0)
        for t in [0,1,2.5,4,5.5,7.5]:
            i=min(round(t/m['dt']),len(x)-1);update(i);fig.savefig(report/f'stage_{t:g}s.png',dpi=100,facecolor=fig.get_facecolor())
    plt.close(fig)
    if records:
        t=np.array([r['time'] for r in records]);center=np.array([r['center'] for r in records])
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True)
        axes[0].plot(t,[r['wind'][1] for r in records]);axes[0].set_ylabel('Forcing (m/s²)')
        axes[1].plot(t,center[:,1]*1000);axes[1].set_ylabel('COM y (mm)')
        axes[2].plot(t,[r['maximum_stretch'] for r in records]);axes[2].axhline(1.1,color='r',ls='--');axes[2].set_ylabel('Max stretch');axes[2].set_xlabel('Physical time (s)')
        for a in axes:a.grid(alpha=.2)
        fig.savefig(report/'metrics.png',dpi=150);plt.close(fig)
        summary=dict(completed=m['completed'],steps=m['completed_steps'],simulated_seconds=m['completed_steps']*m['dt'],
            native_steps=m['native_steps'],preview_steps=m['preview_steps'],
            semi_implicit_preview=m.get('semi_implicit_preview',False),
            max_pin_error_m=max(r['pin_error_m'] for r in records),maximum_stretch=max(r['maximum_stretch'] for r in records),
            center_y_range_mm=[float(center[:,1].min()*1000),float(center[:,1].max()*1000)],
            max_vertex_displacement_m=float(np.linalg.norm(x-rest,axis=2).max()),
            limitations=['L0 only; no inter-level VelPro validation','Prescribed smooth body force, not aerodynamic wind',
                         'Coarse preview stops are not Newton convergence','Native stop may include semi-implicit termination when enabled',
                         'No fine dynamics are rendered'])
        (report/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
