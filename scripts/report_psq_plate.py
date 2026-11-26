"""Report actual equilibrium solves, without scaling out-of-plane geometry."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from demo_psq_plate import load, Model, boundary, SIZE, FORCE_TOL, STEP_TOL
from demo_psq_shallow_pleats import write_json
from demo_psq_intrinsic_pleats import collision_frames
from progressive.runtime import numerical_threads
from report_square_sphere_demo import rasterize

NAMES=('direct_L0','psq_L0','psq_L1','psq_L2','direct_L2')
LABELS=('Direct L0','PSQ L0','PSQ L1','PSQ L2','Direct L2')
COLORS=('#c97e49','#269ba7','#6387be','#245678','#6b647e')
BG='#f4f6f9';INK='#213147'


def draw(ax,x,faces,wire=False):
    view=np.array([.9,-2.5,1.5]);view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right)
    basis=np.column_stack((right,np.cross(view,right)))
    tri=x[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    light=np.array([-.8,-.6,1.2]);light/=np.linalg.norm(light)
    colors=np.array([.16,.65,.70])*(.3+.7*np.abs(normal@light))[:,None]
    # Fixed global camera for both cases and all methods; true physical z.
    projected=tri@basis*3.1+np.array([0,.045])
    ax.imshow(rasterize(projected,tri@view,colors,0 if wire else len(faces),700),extent=(-.79,.79,-.70,.79),origin='upper')
    # The two straight x-boundaries have prescribed positions; marking them
    # distinguishes boundary loading from an imposed interior deformation.
    for side in (-1,1):
        ids=np.flatnonzero(np.isclose(x[:,0],side*np.max(np.abs(x[:,0])),atol=1e-10))
        ids=ids[np.argsort(x[ids,1])]
        line=x[ids]@basis*3.1+np.array([0,.045])
        ax.plot(line[:,0],line[:,1],color='#d79842',lw=2.0)
    ax.set(xlim=(-.79,.79),ylim=(-.70,.79),aspect='equal');ax.axis('off')


def collect(root,case,suffix):
    h,*_=load(root);data={};audit={}
    fine=Model(h,2,False)
    for name in NAMES:
        out=root/case/(name+suffix)
        meta=json.loads((out/'run.json').read_text())
        rows=json.loads((out/'steps.json').read_text())
        traces=json.loads((out/'solver.json').read_text())
        x=np.load(out/'positions.npy');k=meta['level']
        assert meta['status']=='completed' and len(x)==len(rows)+1==25
        assert all(t['converged'] for t in traces)
        model=Model(h,k,meta['shared_fine_proxy'])
        free,_=boundary(h.rests[k],0,case)
        bound_error=0.;force_error=0.
        for i,y in enumerate(x):
            _,target=boundary(h.rests[k],meta['controls'][i],case)
            bound_error=max(bound_error,float(np.abs(y[free]-target[free]).max()))
            if i:
                _,g,_=model.evaluate(y)
                force_error=max(force_error,float(np.abs(g[~free]).max()))
        assert bound_error<1e-12 and force_error<FORCE_TOL
        assert max(r['step_inf_m'] for r in rows)<STEP_TOL
        assert not collision_frames(h.rests[k],h.faces[k],x)
        onfine=np.array([h.to_finest[k].position(y) for y in x])
        assert not collision_frames(h.rests[2],h.faces[2],onfine)
        # A stationary point need not be a stable equilibrium. Inspect the
        # exact (unprojected) constrained Hessian at the final load.
        _,_,H=model.evaluate(x[-1],True,False)
        ids=np.flatnonzero(np.repeat(~free,3));A=H[ids][:,ids].toarray()
        mineig=float(eigh((A+A.T)/2,subset_by_index=[0,0],eigvals_only=True)[0])
        assert mineig>-1e-5
        audit[name]=dict(max_boundary_error_m=bound_error,max_free_force_N=force_error,
            final_minimum_free_hessian_eigenvalue_N_per_m=mineig,
            sampled_self_intersections=0,all_steps_converged=True,seconds=sum(r['seconds'] for r in rows))
        data[name]=dict(x=x,level=k,rows=rows,meta=meta,onfine=onfine)
    weight=fine.shell.mass/fine.shell.mass.sum()
    ref=data['direct_L2']
    for name,d in data.items():
        diff=d['onfine']-ref['x']
        rms=np.sqrt(np.einsum('v,tvc,tvc->t',weight,diff,diff))*1000
        d['rms_mm']=rms
        audit[name].update(final_material_rms_vs_direct_fine_mm=float(rms[-1]),
            final_reaction_N=d['rows'][-1]['reaction_N'],final_energy_J=d['rows'][-1]['energy_J'],
            final_membrane_energy_J=d['rows'][-1]['membrane_J'],final_bending_energy_J=d['rows'][-1]['bending_J'],
            final_height_range_mm=[d['rows'][-1]['height_min_m']*1000,d['rows'][-1]['height_max_m']*1000])
    return h,data,audit


def render(root,case,h,data):
    report=root/'report';report.mkdir(exist_ok=True)
    columns=['method','index','amount','displacement_m','energy_J','membrane_J','bending_J',
             'reaction_N','height_max_m','height_min_m','force_inf_N','step_inf_m','max_stretch','iterations','seconds']
    with (report/f'{case}_curves.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=columns);writer.writeheader()
        for name,d in data.items():
            for row in d['rows']:
                writer.writerow(dict(method=name,**{key:row[key] for key in columns[1:]}))
    fig,axes=plt.subplots(2,3,figsize=(15,9),dpi=100,facecolor=BG)
    fig.subplots_adjust(left=.035,right=.975,bottom=.11,top=.83,hspace=.15,wspace=.20)
    fig.text(.035,.955,'PSQ / '+('PLATE BUCKLING' if case=='compression' else 'PLATE SHEAR'),fontsize=25,color=INK,weight='bold')
    subtitle=fig.text(.035,.907,'',fontsize=13,color='#526078')
    fig.text(.035,.035,'Computed equilibria | Gold: prescribed boundary positions | True height scale | No inertia or gravity',fontsize=11,color='#526078')
    frames=[];writer=imageio.get_writer(report/f'{case}.mp4',fps=8,codec='libx264',quality=8,macro_block_size=1)
    for i in range(25):
        displacement=data['direct_L2']['meta']['controls'][i]*SIZE*1000
        subtitle.set_text(f'30 cm metal plate | E = 70 GPa | thickness = 0.318 mm | relative edge displacement = {displacement:.2f} mm')
        for j,name in enumerate(NAMES):
            ax=axes.flat[j];ax.clear();d=data[name];k=d['level'];x=d['x'][i]
            draw(ax,x,h.faces[k],wire=k<2)
            ax.set_title(f'{LABELS[j]} / {len(x)} vertices',fontsize=14,color=COLORS[j],weight='bold',pad=2)
            reaction=d['rows'][i-1]['reaction_N'] if i else 0.
            mode='Own mesh energy' if name.startswith('direct') else 'Shared L2 elastic energy'
            ax.text(.5,-.015,f'{mode}\nReaction {reaction:.2f} N  |  max height {np.max(np.abs(x[:,2]))*1000:.2f} mm',
                transform=ax.transAxes,ha='center',fontsize=9.5,color='#526078')
        ax=axes.flat[5];ax.clear();ax.set_facecolor(BG)
        for j,name in enumerate(NAMES):
            d=data[name];xx=np.array([0]+[r['displacement_m']*1000 for r in d['rows']])
            yy=np.array([0]+[r['reaction_N'] for r in d['rows']])
            ax.plot(xx[:i+1],yy[:i+1],c=COLORS[j],lw=1.6,label=LABELS[j],ls='--' if name=='direct_L2' else '-')
        ymax=max(r['reaction_N'] for d in data.values() for r in d['rows'])
        ax.set(xlim=(0,data['direct_L2']['rows'][-1]['displacement_m']*1000),ylim=(0,ymax*1.08),
               xlabel='Relative edge displacement (mm)',ylabel='Work-conjugate reaction (N)')
        ax.set_title('Reaction vs. displacement',fontsize=13,color=INK);ax.grid(alpha=.17);ax.legend(fontsize=8,ncol=2,loc='upper left')
        fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
        frames.append(frame);writer.append_data(frame)
    imageio.imwrite(report/f'{case}_final.png',frames[-1]);writer.close();plt.close(fig)
    imageio.mimsave(report/f'{case}.gif',frames,duration=125,loop=0)

    fig,axes=plt.subplots(2,2,figsize=(12,8),facecolor=BG)
    for j,name in enumerate(NAMES):
        d=data[name];s=d['rows'];xx=[r['displacement_m']*1000 for r in s]
        axes[0,0].plot(xx,[r['reaction_N'] for r in s],c=COLORS[j],label=LABELS[j])
        axes[0,1].plot(xx,[r['energy_J'] for r in s],c=COLORS[j])
        axes[1,0].plot([0]+xx,d['rms_mm'],c=COLORS[j])
        axes[1,1].semilogy(xx,[r['force_inf_N'] for r in s],c=COLORS[j])
    for ax,title in zip(axes.flat,('Reaction (N)','Model elastic energy (J)','Material RMS vs Direct L2 (mm)','Free-force residual (N)')):
        ax.set_title(title);ax.set_xlabel('Relative edge displacement (mm)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9);axes[1,1].axhline(FORCE_TOL,c='black',ls='--',lw=1,label='Tolerance');axes[1,1].legend(fontsize=9)
    fig.suptitle(case.capitalize()+' / equilibrium validation',fontsize=19,color=INK)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(report/f'{case}_metrics.png',dpi=140);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'output/psq_plate_boundary_fixed')
    ap.add_argument('--suffix',default='_verified');args=ap.parse_args();root=args.output
    result={}
    with numerical_threads(1):
        for case in ('compression','shear'):
            h,data,audit=collect(root,case,args.suffix);result[case]=audit
            print(case,json.dumps(audit),flush=True);render(root,case,h,data)
        fig,axes=plt.subplots(1,3,figsize=(12,4.3),facecolor=BG)
        for k,ax in enumerate(axes):
            draw(ax,h.rests[k],h.faces[k],True)
            ax.set_title(f'L{k}: {len(h.rests[k])} vertices / {len(h.faces[k])} triangles',fontsize=12)
        fig.suptitle('Actual edge-collapse hierarchy / all material boundary vertices retained',fontsize=15)
        fig.tight_layout();fig.savefig(root/'report/hierarchy.png',dpi=130);plt.close(fig)
    result['scope']='Contact-free flat-domain PSQ energy/progressive equilibrium validation; not a nonzero-offset or safe-refinement test.'
    result['initialization']='PSQ L1/L2 initialized from lower equilibrium at the same load. Direct solves follow their own preceding load.'
    result['mapping']='Actual boundary-locked decimation plus exact shared-planar-material barycentric maps; independently agree with native successive-UV weights below 4e-14.'
    result['mapping_audit']=json.loads((root/'planar_maps/manifest.json').read_text())
    result['energy_checks']={case:json.loads((root/f'{case}_energy_check.json').read_text()) for case in ('compression','shear')}
    result['timing']='Seconds exclude preprocessing/reporting. No paper performance claim; projected finest proxy Hessian is used.'
    result['hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (ROOT/'scripts/demo_psq_plate.py',Path(__file__),ROOT/'progressive/shell.py',root/'scene.json')}
    write_json(root/'report/audit.json',result)


if __name__=='__main__':main()
