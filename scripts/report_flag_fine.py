"""Shared-camera plots and movie of the actual completed hierarchy, following provenance."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import imageio.v2 as imageio

def main():
    p=argparse.ArgumentParser();p.add_argument('last',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=[];path=a.last
    while True:
        m=json.loads((path/'run.json').read_text());assert m['completed'] and m['completed_steps']==500
        mesh=np.load(Path(m['assets'])/f'mesh_{m["level"]}.npz')
        x=np.load(path/'positions.npy',mmap_mode='r');f=mesh['faces'];rest=mesh['vertices']
        fc=rest[f].mean(1);checker=(np.floor(fc[:,0]/.08)+np.floor(fc[:,2]/.08)).astype(int)%2
        data.append(dict(m=m,x=x,f=f,path=path,colors=np.where(checker[:,None]==0,[.10,.54,.73],[.25,.73,.82])))
        if not m.get('lower_run'):break
        path=Path(m['lower_run'])
    data.reverse();a.output.mkdir(parents=True,exist_ok=True)
    summary=[]
    for d in data:
        m=d['m'];rows=json.loads((d['path']/'steps.json').read_text());audit=json.loads((d['path']/'audit.json').read_text())
        speed=[]
        for k in range(1,501):speed.append(float(np.linalg.norm(d['x'][k]-d['x'][k-1],axis=1).max()/.04))
        d['speed']=np.array(speed)
        summary.append(dict(level=m['level'],vertices=m['vertices'],faces=m['faces'],seconds=m['wall_seconds'],native=m['native_steps'],preview=m['preview_steps'],max_newton=max(r['native']['newton_iterations'] for r in rows),stretch=audit['maximum_stretch'],zero_motion=audit['zero_motion_steps'],run=str(d['path'].resolve()),backend=m['library_sha256']['uipc_backend_cuda.dll'],preconditioner=m['config']['linear_system']['fem_preconditioner']))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    selected=[d for d in data if d['m']['level'] in [0,3,6,9]]
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    ids=[s['level'] for s in summary]
    axes[0,0].bar(ids,[s['seconds'] for s in summary],color='#197d9c');axes[0,0].set_yscale('log');axes[0,0].set_ylabel('Wall time (s, log)');axes[0,0].set_title('Shared GPU: timings are not a benchmark')
    axes[0,1].bar(ids,[s['native'] for s in summary],label='Newton converged',color='#197d9c');axes[0,1].bar(ids,[s['preview'] for s in summary],bottom=[s['native'] for s in summary],label='Verified coarse non-descent',color='#e1a45a');axes[0,1].legend(fontsize=8);axes[0,1].set_ylabel('Accepted steps')
    for ax in axes[0]:ax.set_xticks(ids,[f'L{k}' for k in ids]);ax.grid(axis='y',alpha=.2)
    t=np.arange(1,501)*.04
    axes[1,0].plot(t,8+5*np.sin(2*np.pi*1.3*t),color='#62718a');axes[1,0].set_ylabel('Applied y acceleration (m/s²)')
    for d in selected:axes[1,1].plot(t,d['speed'],label=f'L{d["m"]["level"]}',lw=.8)
    axes[1,1].legend();axes[1,1].set_ylabel('Maximum vertex speed (m/s)')
    for ax in axes[1]:ax.set_xlabel('Physical time (s)');ax.grid(alpha=.2)
    fig.suptitle('Flag hierarchy | actual accepted trajectories, dt = 0.04 s');fig.savefig(a.output/'metrics.png',dpi=150);plt.close(fig)
    az,el=np.deg2rad([-62,18]);camera=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right=np.array([-np.sin(az),np.cos(az),0]);up=np.cross(camera,right);basis=np.stack([right,up,camera],axis=1)
    bounds=[]
    for d in data:
        for k in range(0,501,10):
            projected=d['x'][k]@basis;bounds.append([projected[:,:2].min(0),projected[:,:2].max(0)])
    b=np.array(bounds);lo=b[:,0].min(0);hi=b[:,1].max(0);center=(lo+hi)/2;span=max(hi-lo)*1.12
    def setup(ax):
        ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2);ax.set_aspect('equal');ax.axis('off');ax.set_facecolor('#f5f7fa')
        pole=np.array([[0,0,.1],[0,0,1.4]])@basis;ax.plot(pole[:,0],pole[:,1],color='#536778',lw=2,zorder=0)
    def polys(d,k):
        tri=d['x'][k][d['f']];projected=tri@basis;order=np.argsort(projected[:,:,2].mean(1));n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
        colors=d['colors']*(.72+.28*np.abs(n@np.array([.3,-.7,.64])))[:,None]
        return projected[order,:,:2],colors[order]
    fig,axs=plt.subplots(len(data),4,figsize=(13,2.1*len(data)),facecolor='#f5f7fa')
    for i,d in enumerate(data):
        for j,k in enumerate([0,167,333,500]):
            ax=axs[i,j];setup(ax);verts,colors=polys(d,k);ax.add_collection(PolyCollection(verts,facecolors=colors,edgecolors='none'))
            ax.set_title(f'L{d["m"]["level"]} | {d["m"]["vertices"]:,} vertices | {k*.04:.2f} s',fontsize=9)
    fig.suptitle('Actual flag motion across levels | shared camera, scale and physical time',fontsize=16,y=.99)
    fig.subplots_adjust(top=1-.65/(2.1*len(data)),bottom=.015,hspace=.18,wspace=.02);fig.savefig(a.output/'hierarchy.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(1,len(selected),figsize=(4*len(selected),5.3),dpi=100,facecolor='#f5f7fa');axs=np.atleast_1d(axs);collections=[]
    for ax,d in zip(axs,selected):
        setup(ax);poly=PolyCollection([],edgecolors='none');ax.add_collection(poly);collections.append(poly);ax.set_title(f'L{d["m"]["level"]} | {d["m"]["vertices"]:,} vertices',fontsize=12)
    fig.subplots_adjust(left=.02,right=.98,top=.80,bottom=.14,wspace=.05)
    fig.text(.025,.94,'FLAG | synchronized actual dynamics',fontsize=21,weight='bold');label=fig.text(.025,.86,'',fontsize=12)
    fig.text(.025,.05,'20 s / 500 steps per level | PSQ + VelPro with explicit repairs | author inputs not fully recovered',fontsize=10)
    with imageio.get_writer(a.output/'hierarchy.mp4',fps=12.5,codec='libx264',macro_block_size=1) as writer:
        for k in range(0,501,2):
            for d,poly in zip(selected,collections):
                verts,colors=polys(d,k);poly.set_verts(verts);poly.set_facecolor(colors)
            label.set_text(f'Physical time {k*.04:.2f} s | shared camera and scale | playback 1x');fig.canvas.draw();writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
            if k in [0,250,500]:fig.savefig(a.output/f'video_{k:03}.png',dpi=100)
    plt.close(fig)

if __name__=='__main__':main()
