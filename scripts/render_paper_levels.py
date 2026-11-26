"""Synchronized side-by-side movies with shared camera and actual level states."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

def shade(tri,base):
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
    return np.asarray(base)*(.65+.35*np.abs(n@np.array([.3,-.7,.64])))[:,None]

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--end-seconds',type=float,required=True);p.add_argument('--levels',type=int,nargs='+');p.add_argument('--playback-rate',type=float,default=1.)
    p.add_argument('--start-seconds',type=float,default=0.);p.add_argument('--tag',default='levels');p.add_argument('--azim',type=float);a=p.parse_args()
    if not 0<a.playback_rate<=2:p.error('Require 0 < playback rate <= 2')
    if not a.tag.replace('_','').isalnum():p.error('Output tag must be alphanumeric or underscore')
    paths=sorted(a.root.glob('L[0-9]'),key=lambda p:int(p.name[1:]));data=[]
    if (a.root/'chain.json').exists():
        chain=json.loads((a.root/'chain.json').read_text());paths=[Path(r['run']) for r in chain['chain']]
        if 'failed_level' in chain:paths.append(a.root/f'L{chain["failed_level"]}')
    for path in paths:
        if not (path/'run.json').exists():continue
        m=json.loads((path/'run.json').read_text())
        if a.levels and m['level'] not in a.levels:continue
        if m['status']=='running' or not m['completed_steps']:continue
        mesh=np.load(Path(m['assets'])/f'mesh_{m["level"]}.npz');rest,f=mesh['vertices'],mesh['faces'];x=np.load(path/'positions.npy',mmap_mode='r')[:m['completed_steps']+1]
        data.append(dict(m=m,rest=rest,f=f,x=x))
    if not data:raise ValueError('No accepted trajectories')
    dt=data[0]['m']['dt'];flag=data[0]['m']['scene']=='flag'
    end=min(int(round(a.end_seconds/dt)),min(len(d['x'])-1 for d in data));start=max(0,int(round(a.start_seconds/dt)))
    if start>=end:p.error('No accepted time window after the requested start')
    stride=max(1,round(a.playback_rate/(12.5*dt)));ids=np.unique(np.r_[np.arange(start,end+1,stride),end]).astype(int)
    lo=np.min([d['x'][ids].min(axis=(0,1)) for d in data],axis=0);hi=np.max([d['x'][ids].max(axis=(0,1)) for d in data],axis=0)
    if not flag:
        b=np.load(Path(data[0]['m']['assets'])/'bunny.npz');bv,bf=b['vertices'],b['faces'];lo=np.minimum(lo,bv.min(0));hi=np.maximum(hi,bv.max(0))
    center=(lo+hi)/2;span=max(hi-lo)*1.05
    fig=plt.figure(figsize=(3.6*len(data),5.8),dpi=100,facecolor='#f5f7fa');polys=[]
    for j,d in enumerate(data):
        ax=fig.add_axes([j/len(data),.16,1/len(data),.68],projection='3d');ax.set_facecolor('#f5f7fa');ax.set_proj_type('ortho');ax.view_init(elev=18 if flag else 30,azim=a.azim if a.azim is not None else (-62 if flag else -52))
        for setter,c in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],center):setter(c-span/2,c+span/2)
        ax.set_box_aspect((1,1,1),zoom=2.0 if flag else 1.6);ax.set_axis_off()
        if not flag:ax.add_collection3d(Poly3DCollection(bv[bf],facecolors=shade(bv[bf],[.77,.79,.81]),edgecolors='none'))
        else:ax.plot([0,0],[0,0],[.1,1.4],color='#536778',lw=2)
        poly=Poly3DCollection(d['x'][0][d['f']],edgecolors=(.05,.2,.3,.15),linewidths=.09);ax.add_collection3d(poly);polys.append(poly)
        fc=d['rest'][d['f']].mean(1);dims=[0,2] if flag else [0,1];checker=(np.floor(fc[:,dims[0]]/.08)+np.floor(fc[:,dims[1]]/.08)).astype(int)%2
        d['base']=np.where(checker[:,None]==0,[.1,.54,.73],[.25,.73,.82])
        ax.set_title(f'L{d["m"]["level"]} | {len(d["rest"]):,} vertices',fontsize=13,pad=3)
    scene='FLAG' if flag else 'BUNNY CONTACT AND SLIDE'
    fig.text(.035,.93,scene+' | synchronized actual dynamics',fontsize=21,weight='bold')
    fig.text(.035,.875,'Shared camera and physical time | PSQ elastic proxy + VelPro | semi-implicit OFF',fontsize=11,color='#536778')
    label=fig.text(.035,.10,'',fontsize=13)
    failures=[]
    for d in data:
        m=d['m']
        if not m['completed']:
            failures.append(f'L{m["level"]} rejected step {m["rejected_step"]["step"]}' if m.get('rejected_step')
                else f'L{m["level"]} interrupted after {m["completed_steps"]} accepted steps')
    footer='; '.join(failures) if failures else 'Requested windows completed; coarse preview stops are not Newton convergence'
    fig.text(.035,.055,footer+' | author inputs not fully recovered',fontsize=10,color='#904731')
    out=a.root/'comparison';out.mkdir(exist_ok=True);images=[]
    poster_id=ids[np.argmin(np.abs(ids*dt-(1.6 if flag else .4)))]
    with imageio.get_writer(out/f'{a.tag}.mp4',fps=a.playback_rate/(stride*dt),codec='libx264',macro_block_size=1) as writer:
        for i in ids:
            for d,poly in zip(data,polys):
                tri=d['x'][i][d['f']];poly.set_verts(tri);poly.set_facecolor(shade(tri,d['base']))
            label.set_text(f't = {i*dt:.2f} s | window {start*dt:g}-{end*dt:g} s | playback {a.playback_rate:g}x');fig.canvas.draw()
            frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();writer.append_data(frame);images.append(frame[::2,::2])
            if i==poster_id:fig.savefig(out/f'{a.tag}_poster.png',dpi=140)
    imageio.mimsave(out/f'{a.tag}.gif',images,duration=stride*dt/a.playback_rate*1000,loop=0)
    fig.savefig(out/f'{a.tag}_last.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
