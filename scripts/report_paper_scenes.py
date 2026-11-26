"""Render actual hierarchy assets and accepted trajectories; label scope honestly."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

def assets_panel(assets,out,scene):
    m=json.loads((assets/'manifest.json').read_text());levels=m['levels'];cols=5 if scene=='flag' else 4
    fig,axes=plt.subplots(2,cols,figsize=(15,7),facecolor='#f5f7fa')
    for ax,row in zip(axes.flat,levels):
        d=np.load(assets/f'mesh_{row["level"]}.npz');v,f=d['vertices'],d['faces'];dims=[0,2] if scene=='flag' else [0,1]
        ax.add_collection(PolyCollection(v[f][:,:,dims],facecolors='#bbe5ef',edgecolors='#255365',linewidths=.16))
        ax.autoscale();ax.set_aspect('equal');ax.axis('off')
        ax.set_title(f'L{row["level"]}: {len(v):,} vertices\n{len(f):,} triangles',fontsize=11)
    fig.suptitle(f'PD++ {scene.upper()} | {len(levels)} levels at published endpoint scale',fontsize=20)
    fig.text(.5,.025,'Reconstructed rest meshes | geometry preparation only; no dynamics claim',ha='center',color='#536778')
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(out/f'{scene}_hierarchy.png',dpi=130);plt.close(fig)

def trajectory(run,end_seconds=None,zoom=None):
    m=json.loads((run/'run.json').read_text());assets=Path(m['assets']);d=np.load(assets/f'mesh_{m["level"]}.npz')
    rest,f=d['vertices'],d['faces'];xs=np.load(run/'positions.npy',mmap_mode='r')[:m['completed_steps']+1]
    rows=json.loads((run/'steps.json').read_text());flag=m['scene']=='flag';out=run/'report';out.mkdir(exist_ok=True)
    total_available=len(xs)
    if end_seconds is not None:xs=xs[:min(len(xs),int(round(end_seconds/m['dt']))+1)]
    fig=plt.figure(figsize=(10,7),dpi=100,facecolor='#f5f7fa');ax=fig.add_axes([.03,.12,.94,.76],projection='3d')
    ax.set_facecolor('#f5f7fa')
    ax.set_proj_type('ortho');ax.view_init(elev=18 if flag else 35,azim=-62 if flag else -52)
    lo=xs.min(axis=(0,1));hi=xs.max(axis=(0,1))
    if not flag:
        b=np.load(assets/'bunny.npz');bv,bf=b['vertices'],b['faces']
        ax.add_collection3d(Poly3DCollection(bv[bf],facecolors='#c7ccd1',edgecolors='none'))
        lo=np.minimum(lo,bv.min(0));hi=np.maximum(hi,bv.max(0))
    center=(lo+hi)/2;span=max(hi-lo)*1.12
    for setter,c in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],center):setter(c-span/2,c+span/2)
    ax.set_box_aspect((1,1,1),zoom=1.3 if zoom is None else zoom);ax.set_axis_off()
    if flag:
        ax.plot([0,0],[0,0],[rest[:,2].min()-.15,rest[:,2].max()+.15],color='#536778',lw=3)
    poly=Poly3DCollection(xs[0][f],edgecolors=(.05,.2,.3,.4),linewidths=.13);ax.add_collection3d(poly)
    fc=rest[f].mean(1);dims=[0,2] if flag else [0,1]
    checker=(np.floor(fc[:,dims[0]]/.08)+np.floor(fc[:,dims[1]]/.08)).astype(int)%2
    base=np.where(checker[:,None]==0,[.1,.54,.73],[.25,.73,.82])
    fig.text(.05,.94,f'{m["scene"].upper()} | L{m["level"]} | {m["vertices"]:,} dynamic vertices',fontsize=19,weight='bold')
    fig.text(.05,.895,f'{m["proxy_vertices"]:,} elastic proxy vertices | semi-implicit OFF | actual accepted states',fontsize=10)
    label=fig.text(.05,.065,'',fontsize=11)
    status='Requested window completed' if m['completed'] else 'INCOMPLETE: accepted prefix only'
    clip=f' | showing first {(len(xs)-1)*m["dt"]:g} s' if len(xs)<total_available else ''
    fig.text(.05,.025,status+clip+'; author inputs not fully recovered',fontsize=10,color='#914731' if not m['completed'] else '#536778')
    def draw(i):
        tri=xs[i][f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
        poly.set_verts(tri);poly.set_facecolor(base*(.72+.28*np.abs(n@np.array([.3,-.7,.64])))[:,None])
        reason='initial state' if i==0 else rows[i-1]['native']['termination_reason']
        label.set_text(f't = {i*m["dt"]:.2f} s / {m["steps"]*m["dt"]:.2f} s | {reason}')
        fig.canvas.draw()
    draw(len(xs)-1);fig.savefig(out/'last_accepted.png',dpi=140)
    indices=np.unique(np.r_[np.arange(0,len(xs),max(1,round(1/(24*m['dt'])))),len(xs)-1]).astype(int)
    if len(xs)>1:
        images=[]
        with imageio.get_writer(out/'accepted.mp4',fps=1/(max(1,round(1/(24*m['dt'])))*m['dt']),codec='libx264',macro_block_size=1) as writer:
            for i in indices:
                draw(i);frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();writer.append_data(frame)
                images.append(frame[::2,::2])
        imageio.mimsave(out/'accepted.gif',images,duration=max(1,round(1/(24*m['dt'])))*m['dt']*1000,loop=0)
    plt.close(fig)
    if rows:
        fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
        t=np.arange(1,len(rows)+1)*m['dt'];ax[0].plot(t,[r['maximum_stretch'] for r in rows]);ax[0].axhline(m['material']['strain_limit'],ls='--',color='r');ax[0].set_ylabel('Maximum stretch')
        ax[1].plot(t,[r['native']['newton_iterations'] for r in rows]);ax[1].set_ylabel('Newton iterations');ax[1].set_xlabel('Accepted physical time (s)')
        fig.savefig(out/'metrics.png',dpi=140);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--flag-assets',type=Path);p.add_argument('--bunny-assets',type=Path);p.add_argument('--output',type=Path);p.add_argument('--run',type=Path);p.add_argument('--end-seconds',type=float);p.add_argument('--zoom',type=float);a=p.parse_args()
    if a.zoom is not None and a.zoom<=0:p.error('Require positive zoom')
    if a.output:
        a.output.mkdir(parents=True,exist_ok=True)
        if a.flag_assets:assets_panel(a.flag_assets,a.output,'flag')
        if a.bunny_assets:assets_panel(a.bunny_assets,a.output,'bunny')
    if a.run:trajectory(a.run,a.end_seconds,a.zoom)

if __name__=='__main__':main()
