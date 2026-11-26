"""Compare preserved-baseline and revised-mesh progressive settling states."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from progressive.prolongation import frames
from shirt_demo.report import load_run,draw,plt,Image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    oldroot=Path('output/shirt_L0_grasp_20261117')
    old=[load_run(oldroot/name/'run') for name in ['laydown_v2','laydown_L1','laydown_L2']]
    new=[load_run(root/f'validation_L{k}') for k in range(3)]
    assert all(d['meta']['status']!='running' for d in new)
    n=new[2]['meta']['completed_steps'];assert n>0
    result=[]
    for version,group in [('baseline',old),('revised',new)]:
        for level,d in enumerate(group):
            m=d['meta'];base=old[level]['meta'];assert m['material_parameters']==base['material_parameters']
            assert m['config']==base['config'] and m['runtime_hashes']==base['runtime_hashes']
            assert m['robot_q_sha256']==base['robot_q_sha256'] and m['parameters'].get('diagnostic_contact','full')=='full'
            x=d['x'];f=d['f'];inv=frames(x[0],f)[2];samples=[]
            for k in sorted({0,1,5,10,20,30,60,155,n}):
                if k>min(n,m['completed_steps']):continue
                t=x[k][f];sigma=np.linalg.svd(np.stack([t[:,1]-t[:,0],t[:,2]-t[:,0]],axis=2)@inv,compute_uv=False)
                samples.append(dict(step=k,maximum_stretch=float(sigma[:,0].max()),triangles_above_1_099=int((sigma[:,0]>1.099).sum()),
                    height_above_table_percentiles_mm=(np.percentile(x[k,:,2]-.8,[0,50,90,100])*1000).tolist()))
            result.append(dict(version=version,level=level,status=m['status'],completed_steps=m['completed_steps'],
                native_converged_steps=m['native_converged_steps'],preview_steps=m['preview_stopped_steps'],
                newton_limit_steps=m['newton_limit_steps'],physics_seconds=m['physics_seconds'],samples=samples))
    summary=dict(revised_L2_completed=new[2]['meta']['completed'],revised_L2_steps=n,
        material_contact_solver_runtime_unchanged=True,levels=result,
        limitations=['The finest geometry and intrinsic hierarchy were rebuilt; later-state differences combine these changes.',
            'At the first step all injected velocities are zero, isolating the finest-mesh change.',
            'This validates a short settling/approach prefix, not a complete fold.',
            'No independent continuous collision or universal artifact-free guarantee.'])
    (root/'dynamic_comparison.json').write_text(json.dumps(summary,indent=2))
    end=min(60,n);fig=plt.figure(figsize=(14,10))
    for row,group in enumerate([old,new]):
        for level,d in enumerate(group):
            ax=fig.add_subplot(2,3,row*3+level+1,projection='3d');draw(ax,d,end,f'{"Original" if row==0 else "Revised"} L{level} | {end/120:.3f} s',False)
    fig.suptitle('Actual progressive states | identical camera and shading | self-contact enabled')
    fig.tight_layout();fig.savefig(root/'settling_three_levels.png',dpi=140,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for row in result:
        if row['level']!=2:continue
        samples=row['samples'];ts=[v['step']/120 for v in samples]
        axes[0].plot(ts,[v['maximum_stretch'] for v in samples],marker='o',label=row['version'])
        axes[1].plot(ts,[v['triangles_above_1_099'] for v in samples],marker='o',label=row['version'])
    axes[0].set(ylabel='Maximum principal stretch');axes[1].set(ylabel='Faces with stretch > 1.099')
    for ax in axes:ax.set_xlabel('Physical time (s)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/'settling_stretch_comparison.png',dpi=150);plt.close(fig)
    images=[]
    for k in np.unique(np.linspace(0,end,41).astype(int)):
        fig=plt.figure(figsize=(11,5),dpi=95)
        for col,(label,d) in enumerate([('Original L2',old[2]),('Revised L2',new[2])]):
            ax=fig.add_subplot(1,2,col+1,projection='3d');draw(ax,d,k,f'{label} | {k/120:.3f} s',False)
        fig.suptitle('Self-contact ON | same physical time | slow playback')
        fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    images[0].save(root/'L2_settling_before_after.gif',save_all=True,append_images=images[1:],duration=120,loop=0)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
