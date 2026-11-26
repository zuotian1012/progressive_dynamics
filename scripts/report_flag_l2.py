"""Render actual L1 and L2 trajectories at equal physical time and camera scale."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/flag_L2_20261118'

def main():
    paths=[ROOT/'output/flag_L1_fix_20261118/full',OUT]
    metas=[json.loads((p/'run.json').read_text()) for p in paths]
    assert all(m['status']!='running' and m['dt']==.04 for m in metas)
    count=min(m['completed_steps'] for m in metas)
    xs=[np.load(p/'positions.npy',mmap_mode='r')[:count+1] for p in paths]
    meshes=[np.load(Path(m['assets'])/f'mesh_{m["level"]}.npz') for m in metas]
    out=OUT/'report';out.mkdir(exist_ok=True)
    speed=[np.linalg.norm(np.diff(x,axis=0),axis=2).max(axis=1)/.04 for x in xs];t=np.arange(1,count+1)*.04
    fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    ax[0].plot(t,8+5*np.sin(2*np.pi*1.3*t),color='#62718a');ax[0].set_ylabel('Applied acceleration y (m/s²)')
    for v,label,color in zip(speed,['L1 VelPro: 836 vertices','L2 VelPro: 1,345 vertices'],['#a95848','#167fa1']):ax[1].plot(t,v,label=label,color=color,lw=1.3)
    ax[1].set_ylabel('Maximum vertex speed (m/s)');ax[1].set_xlabel('Physical time (s)');ax[1].legend()
    for a in ax:a.grid(alpha=.2)
    fig.suptitle('Flag L1 / L2: accepted-prefix dynamic response');fig.savefig(out/'response_comparison.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(12,6.8),dpi=100,facecolor='#f5f7fa')
    lo=np.minimum(xs[0].min(axis=(0,1)),xs[1].min(axis=(0,1)));hi=np.maximum(xs[0].max(axis=(0,1)),xs[1].max(axis=(0,1)))
    center=(lo+hi)/2;span=max(hi-lo)*1.08
    polys=[];labels=[];colors=[]
    for i,title in enumerate(['L1 VelPro | 836 vertices / 1,600 triangles','L2 VelPro | 1,345 vertices / 2,600 triangles']):
        rest,f=meshes[i]['vertices'],meshes[i]['faces']
        fc=rest[f].mean(1);checker=(np.floor(fc[:,0]/.08)+np.floor(fc[:,2]/.08)).astype(int)%2
        colors.append(np.where(checker[:,None]==0,[.1,.54,.73],[.25,.73,.82]))
        a=fig.add_axes([.01+.5*i,.13,.48,.72],projection='3d');a.set_proj_type('ortho');a.view_init(18,-62);a.set_facecolor('#f5f7fa')
        for setter,c in zip([a.set_xlim,a.set_ylim,a.set_zlim],center):setter(c-span/2,c+span/2)
        a.set_box_aspect((1,1,1),zoom=2.5);a.set_axis_off()
        a.plot([0,0],[0,0],[.1,1.4],color='#536778',lw=3)
        poly=Poly3DCollection(xs[i][0][f],edgecolors=(.05,.2,.3,.35),linewidths=.12);a.add_collection3d(poly);polys.append(poly)
        fig.text(.03+.5*i,.88,title,fontsize=11,weight='bold');labels.append(fig.text(.03+.5*i,.09,'',fontsize=11))
    fig.text(.03,.95,'FLAG L1 / L2 | same force, material and dt = 0.04 s',fontsize=18,weight='bold')
    status='Completed' if metas[1]['completed'] else f'L2 step {count+1} rejected'
    fig.text(.03,.035,f'Accepted: {count*.04:.2f} / 20 s | {status} | shared 35,000-vertex proxy; explicit repair',fontsize=10)
    def draw(k):
        for i,poly in enumerate(polys):
            f=meshes[i]['faces'];tri=xs[i][k][f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
            poly.set_verts(tri);poly.set_facecolor(colors[i]*(.72+.28*np.abs(n@np.array([.3,-.7,.64])))[:,None])
            labels[i].set_text(f't = {k*.04:.2f} s | max vertex speed = {speed[i][k-1] if k else 0:.3f} m/s')
        fig.canvas.draw()
    with imageio.get_writer(out/'comparison.mp4',fps=25,codec='libx264',macro_block_size=1) as writer:
        for k in range(count+1):
            draw(k);writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
            if k in [0,min(205,count),count]:fig.savefig(out/f'comparison_{k:03}.png',dpi=120)
    plt.close(fig)

if __name__=='__main__':main()
