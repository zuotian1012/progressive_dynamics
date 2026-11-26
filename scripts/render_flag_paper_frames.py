"""Paper-style frame grid, retaining visibly missing/uncomputed hierarchy rows."""
import argparse
import json
from pathlib import Path
import numpy as np
from render_paper_levels import plt,Poly3DCollection,shade

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();paths=list(a.root.glob('L[0-9]'))
    if (a.root/'chain.json').exists():
        chain=json.loads((a.root/'chain.json').read_text());paths=[Path(v['run']) for v in chain['chain']]
        if 'failed_level' in chain:paths.append(a.root/f'L{chain["failed_level"]}')
    data={};indices=[166,167,168];bounds=[]
    for path in paths:
        if not (path/'run.json').exists():continue
        m=json.loads((path/'run.json').read_text())
        if m['scene']!='flag':raise ValueError('Flag only')
        if m['status']=='running':continue
        x=np.load(path/'positions.npy',mmap_mode='r');mesh=np.load(Path(m['assets'])/f'mesh_{m["level"]}.npz')
        rows=json.loads((path/'steps.json').read_text());data[m['level']]=(m,x,mesh['vertices'],mesh['faces'],rows)
        for i in indices:
            if i<=m['completed_steps']:bounds.append(np.array([x[i].min(0),x[i].max(0)]))
    if not bounds:raise ValueError('No accepted frames 166-168')
    lo=np.min(np.array(bounds)[:,0],axis=0);hi=np.max(np.array(bounds)[:,1],axis=0);center=(lo+hi)/2;span=max(hi-lo)*1.12
    fig=plt.figure(figsize=(10.5,22),dpi=100,facecolor='#f5f7fa')
    for k in range(10):
        for j,i in enumerate(indices):
            ax=fig.add_subplot(10,3,k*3+j+1,projection='3d');ax.set_facecolor('#f5f7fa');ax.set_axis_off()
            if k not in data or i>data[k][0]['completed_steps']:
                ax.text2D(.5,.5,f'L{k}: not computed',transform=ax.transAxes,ha='center',color='#aa5737');continue
            m,x,rest,f,records=data[k];tri=x[i][f];fc=rest[f].mean(1);checker=(np.floor(fc[:,0]/.08)+np.floor(fc[:,2]/.08)).astype(int)%2
            base=np.where(checker[:,None]==0,[.1,.54,.73],[.25,.73,.82])
            ax.set_proj_type('ortho');ax.view_init(elev=18,azim=-62)
            for setter,c in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],center):setter(c-span/2,c+span/2)
            ax.set_box_aspect((1,1,1),zoom=1.45)
            ax.add_collection3d(Poly3DCollection(tri,facecolors=shade(tri,base),edgecolors='none'))
            ax.plot([0,0],[0,0],[.1,1.4],color='#536778',lw=1)
            reason='Newton' if records[i-1]['native']['converged'] else 'coarse preview'
            ax.set_title(f'L{k} | {len(rest):,} vertices | {reason}',fontsize=9,pad=-7)
    fig.suptitle('VELPRO | Frames 166, 167, 168 across the flag hierarchy',fontsize=18,y=.99)
    fig.text(.5,.972,'Local dt = 0.04 s: 6.64, 6.68, 6.72 s | layout inspired by PD++ Fig. 5',ha='center',fontsize=11)
    fig.text(.5,.015,'Actual accepted states only | missing rows are not substituted by upsampled meshes | author inputs not fully recovered',ha='center',fontsize=10)
    fig.subplots_adjust(left=.035,right=.98,bottom=.035,top=.95,wspace=.01,hspace=.12)
    out=a.root/'comparison';out.mkdir(exist_ok=True);fig.savefig(out/'paper_frames_166_168.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
