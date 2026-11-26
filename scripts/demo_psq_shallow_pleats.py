"""Prescribed-deformation PSQ illustration with a rigid-motion ground truth.

This is an operator demo, not a dynamics or contact simulation. The explicitly
constructed correspondence intentionally omits fine pleats on the coarse grid.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloth.mesh import grid, prolongation
from progressive.prolongation import ShellProlongation
from audit_psq_prolongation import reference
from report_square_sphere_demo import rasterize

AMPLITUDE = .018
COUNT = 81
METHODS = ('linear','fixed','psq')


def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def build(out):
    coarse, fine = grid(8), grid(64)
    cr=np.column_stack((coarse.uv-.5,np.zeros(len(coarse.uv))))
    fr=np.column_stack((fine.uv-.5,AMPLITUDE*np.sin(8*np.pi*fine.uv[:,0])*np.sin(np.pi*fine.uv[:,1])))
    fr[np.abs(fr)<1e-15]=0.
    U=prolongation(coarse,fine); U.eliminate_zeros()
    p=ShellProlongation(cr,fr,coarse.faces,U)
    assert not p.zero_offset
    np.testing.assert_allclose(p.position(cr),fr,atol=1e-14,rtol=0)
    for name,mesh,v in [('coarse',coarse,cr),('fine',fine,fr)]:
        np.savez(out/f'{name}_mesh.npz',vertices=v,faces=mesh.faces,uv=mesh.uv)
        with (out/f'{name}_mesh.obj').open('w') as fp:
            for a,b,c in v: fp.write(f'v {a:.12g} {b:.12g} {c:.12g}\n')
            for a,b,c in mesh.faces+1: fp.write(f'f {a} {b} {c}\n')
    save_npz(out/'map_coarse_fine.npz',U)
    np.save(out/'anchors.npy',p.anchors)
    np.savez(out/'offset_components.npz',gamma=p.gamma,tangent=p.tangent,delta=fr-U@cr)
    # Same rest-area weights for all methods and all poses.
    tri=fr[fine.faces]
    areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    weights=np.zeros(len(fr))
    np.add.at(weights,fine.faces.ravel(),np.repeat(areas/3,3)); weights/=weights.sum()
    return coarse,fine,p,weights


def create_poses(p,kind):
    phase=np.linspace(0,1,COUNT)
    poses=[]; truths=[]; controls=[]
    for t in phase:
        strength=np.sin(np.pi*t)**2
        if kind=='rotation':
            angle=np.pi/2*strength
            Q=rotation(angle)
            x=p.coarse_rest@Q.T
            truths.append(p.fine_rest@Q.T)
            controls.append(np.rad2deg(angle))
        else:
            # A prescribed bend plus a varying cross-section twist; no forces.
            u,v=p.coarse_rest[:,:2].T
            beta=2.0*strength
            yc=np.sin(beta*v)/beta if beta>1e-10 else v
            zc=(1-np.cos(beta*v))/beta if beta>1e-10 else np.zeros_like(v)
            twist=1.2*strength*v
            x=np.column_stack((u*np.cos(twist),yc,zc+u*np.sin(twist)))
            x=x@rotation(np.deg2rad(55)*strength).T
            controls.append(strength)
        poses.append(x)
    return np.array(poses),np.array(truths) if truths else None,np.array(controls)


def evaluate(p,poses,truth,weights):
    delta=p.fine_rest-p.U@p.coarse_rest
    outputs={m:[] for m in METHODS}
    independent_error=0.
    minimum_area=np.inf
    for x in poses:
        linear=p.U@x
        outputs['linear'].append(linear)
        outputs['fixed'].append(linear+delta)
        y=p.position(x)
        outputs['psq'].append(y)
        independent_error=max(independent_error,float(np.max(np.abs(y-reference(p,x)))))
        tri=x[p.faces]
        minimum_area=min(minimum_area,float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).min()/2))
    assert independent_error<1e-12 and minimum_area>1e-10
    outputs={m:np.asarray(x) for m,x in outputs.items()}
    errors={}
    if truth is not None:
        for m,y in outputs.items():
            diff=y-truth
            errors[m]=np.sqrt(np.einsum('v,tvc,tvc->t',weights,diff,diff))*1000
        assert float(np.max(np.abs(outputs['psq']-truth)))<1e-12
    return outputs,errors,dict(independent_polar_reference_max_error_m=independent_error,minimum_coarse_triangle_area_m2=minimum_area)


def render(out,kind,coarse,fine,poses,outputs,errors,controls,intrinsic=False,driver_level=None):
    direction=np.array([1.2,-2.4,1.5]); direction/=np.linalg.norm(direction)
    right=np.cross([0,0,1],direction); right/=np.linalg.norm(right)
    up=np.cross(direction,right); basis=np.column_stack((right,up))
    light=np.array([-.8,-.45,.7]); light/=np.linalg.norm(light)
    fig,axes=plt.subplots(2,2,figsize=(12,9),dpi=100,facecolor='#f4f6f9')
    fig.subplots_adjust(left=.03,right=.97,bottom=.105,top=.84,hspace=.20,wspace=.06)
    fig.text(.04,.954,'PSQ / '+('ACTUAL DECIMATION' if intrinsic else 'SHALLOW PLEATS'),fontsize=22,weight='bold',color='#213147')
    label=fig.text(.96,.954,'',fontsize=13,ha='right',color='#213147')
    fig.text(.04,.908,f'{len(poses[0]):,} coarse vertices -> {len(outputs["psq"][0]):,} original fine vertices | 1 m square | 18 mm pleats',fontsize=12,color='#526078')
    fig.text(.04,.055,'Prescribed coarse deformation; no dynamics. Exact mesh geometry, no displacement scaling.',fontsize=10,color='#526078')
    footer='Rigid rotation has an exact fine-mesh reference.' if kind=='rotation' else 'Bend + twist illustrates transport of rest detail; it is not a physical fine-cloth solution.'
    fig.text(.04,.025,footer,fontsize=10,color='#526078')
    labels={'coarse':f'COARSE DRIVER / '+(f'L{driver_level} / ' if intrinsic else '')+f'{len(poses[0])} vertices','linear':'LINEAR / U x','fixed':'FIXED REST OFFSET / U x + delta','psq':'PSQ / U x + rotated detail'}
    items=[]
    for ax,name in zip(axes.flat,('coarse',*METHODS)):
        ax.set_facecolor('#f4f6f9');ax.set(xlim=(-.79,.79),ylim=(-.70,.79),aspect='equal');ax.axis('off')
        ax.set_title(labels[name],fontsize=12,color='#34465c',pad=0)
        artist=ax.imshow(np.full((8,8,3),[244,246,249],dtype=np.uint8),extent=(-.79,.79,-.70,.79),origin='upper')
        annotation=ax.text(.5,-.02,'',transform=ax.transAxes,ha='center',fontsize=10,color='#526078')
        items.append((artist,annotation,name))
    gif=[]
    with imageio.get_writer(out/f'{kind}.mp4',fps=20,codec='libx264',quality=8,macro_block_size=1) as writer:
        for i in range(COUNT):
            for artist,annotation,name in items:
                vertices=poses[i] if name=='coarse' else outputs[name][i]
                faces=coarse.faces if name=='coarse' else fine.faces
                tri=vertices[faces]
                normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
                base=np.array([.32,.43,.55]) if name=='coarse' else np.array([.18,.65,.70])
                colors=base*(.40+.60*np.abs(normal@light))[:,None]
                # Fine surfaces have no artificial wire texture; coarse retains edges.
                artist.set_data(rasterize(tri@basis,tri@direction,colors,0 if name=='coarse' else len(faces),700))
                if errors and name!='coarse':
                    value=errors[name][i]
                    annotation.set_text('RMS to rotated fine rest: '+('< 1e-9 mm' if value<1e-9 else f'{value:.2f} mm'))
                elif name=='coarse':
                    annotation.set_text('Actual edge-collapse result' if intrinsic else 'Fine pleats are absent from this mesh')
                else:
                    annotation.set_text({'linear':'No rest-detail residual','fixed':'Detail stays in the original world direction','psq':'Detail follows the local surface frame'}[name])
            label.set_text(f'Rigid rotation: {controls[i]:.0f} deg' if kind=='rotation' else f'Bend + twist: {controls[i]*100:.0f}%')
            fig.canvas.draw()
            frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
            writer.append_data(frame)
            gif.append(frame[::2,::2])
            if i==COUNT//2:
                fig.savefig(out/f'{kind}_peak.png',dpi=140)
            if i%40==0:print(f'{kind}: rendered {i}/{COUNT-1}',flush=True)
    imageio.mimsave(out/f'{kind}.gif',gif,duration=50,loop=0)
    plt.close(fig)


def diagnostics(out,fine,p,outputs,errors):
    mid=COUNT//2
    indices=np.flatnonzero(np.isclose(fine.uv[:,1],.5))
    Q=rotation(np.pi/2)
    axis_x=Q@np.array([1.,0,0]);axis_n=Q@np.array([0.,0,1.])
    truth=p.fine_rest@Q.T
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharex=True,sharey=True,layout='constrained')
    for ax,m in zip(axes,METHODS):
        ax.plot(truth[indices]@axis_x*100,truth[indices]@axis_n*1000,color='#555f70',lw=2.5,ls='--',label='Exact rotated detail')
        y=outputs[m][mid]
        ax.plot(y[indices]@axis_x*100,y[indices]@axis_n*1000,color='#13858f',lw=1.8,label=m)
        ax.set_title({'linear':'Linear','fixed':'Fixed rest offset','psq':'PSQ'}[m])
        ax.set_xlabel('Material cross-section (cm)');ax.grid(alpha=.2);ax.legend(fontsize=9)
    axes[0].set_ylabel('Height along rotated normal (mm)')
    fig.suptitle('After a 90-degree rotation / actual 18 mm geometric pleats',fontsize=13)
    fig.savefig(out/'rotation_cross_section.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for m,c in zip(METHODS,('#a76735','#7757a4','#13858f')):
        angle=np.linspace(0,90,91);rms=[]
        for deg in angle:
            Q=rotation(np.deg2rad(deg));x=p.coarse_rest@Q.T
            linear=p.U@x
            y={'linear':linear,'fixed':linear+p.fine_rest-p.U@p.coarse_rest,'psq':p.position(x)}[m]
            # This companion curve uses uniform vertices and is explicitly labelled.
            rms.append(np.sqrt(np.mean(np.sum((y-p.fine_rest@Q.T)**2,axis=1)))*1000)
        ax.plot(angle,rms,color=c,label=m,lw=2)
    ax.set(xlabel='Rotation angle (degrees)',ylabel='Vertex RMS to exact rotation (mm)',title='Rigid-motion check')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'rotation_error.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/psq_shallow_pleats_demo')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'source_snapshot.py')
    coarse,fine,p,weights=build(out)
    summary=dict(scope='PSQ position prolongation only; no physical simulation or collision handling',
        correspondence='explicit nested UV barycentric map; synthetic hierarchy, not an intrinsic simplification benchmark',
        coarse_vertices=len(p.coarse_rest),fine_vertices=len(p.fine_rest),coarse_triangles=len(coarse.faces),fine_triangles=len(fine.faces),
        size_m=1.,pleat_amplitude_m=AMPLITUDE,pleat_cycles=4,max_normal_residual_m=float(np.abs(p.gamma).max()),
        max_tangent_residual_m=float(np.abs(p.tangent).max()),zero_offset=p.zero_offset,
        rest_reconstruction_error_m=float(np.max(np.abs(p.position(p.coarse_rest)-p.fine_rest))),
        caveat='This constructed cloth exercises normal-detail transport; tangent rotation is not a significant visual factor here.',clips={})
    for kind in ('rotation','bend_twist'):
        poses,truth,controls=create_poses(p,kind)
        outputs,errors,audit=evaluate(p,poses,truth,weights)
        np.savez_compressed(out/f'{kind}_geometry.npz',coarse=poses,controls=controls,**outputs,**({'exact_rigid':truth} if truth is not None else {}))
        if errors:
            audit['at_90_degrees_mass_weighted_rms_mm']={m:float(e[COUNT//2]) for m,e in errors.items()}
            diagnostics(out,fine,p,outputs,errors)
        summary['clips'][kind]=audit
        render(out,kind,coarse,fine,poses,outputs,errors,controls)
    summary['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (Path(__file__),ROOT/'progressive/prolongation.py',ROOT/'scripts/audit_psq_prolongation.py')}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
