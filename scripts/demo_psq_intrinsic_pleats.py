"""Render actual decimation hierarchy and original-topology PSQ reconstruction.

Preparation uses progressive.prepare_intrinsic on the original mesh. All maps
and anchors come from native successive self-parameterization, never grid maps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import ipctk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from progressive.prolongation import Hierarchy
from demo_psq_shallow_pleats import rotation, evaluate, render, write_json, COUNT
from report_square_sphere_demo import rasterize


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    source=ROOT/'output/psq_shallow_pleats_demo/fine_mesh.npz'
    shutil.copy2(source,root/'original_fine_mesh.npz')
    mesh=dict(np.load(source))
    with (root/'original_fine.obj').open('w') as fp:
        for v in mesh['vertices']: fp.write('v '+' '.join(format(float(a),'.17g') for a in v)+'\n')
        for face in mesh['faces']: fp.write('f '+' '.join(str(int(a)+1) for a in face)+'\n')
    subprocess.run([sys.executable,'-m','progressive.prepare_intrinsic','--input',str(root/'original_fine.obj'),
        '--targets','2048','512','128','32','--output',str(root/'assets'),'--quadric','probabilistic','--contact-clearance','.0002'],check=True,cwd=ROOT)


def load(root):
    assets=root/'assets'
    manifest=json.loads((assets/'manifest.json').read_text())
    n=len(manifest['levels'])
    meshes=[dict(np.load(assets/f'mesh_{k}.npz')) for k in range(n)]
    rests=[d['vertices'] for d in meshes];faces=[d['faces'] for d in meshes]
    maps=[load_npz(assets/f'map_{k}_{k+1}.npz') for k in range(n-1)]
    adjacent=[np.load(assets/f'anchors_{k}_{k+1}.npy') for k in range(n-1)]
    direct=[np.load(assets/f'anchors_{k}_{n-1}.npy') for k in range(n-1)]+[None]
    hierarchy=Hierarchy(rests,faces,maps,adjacent,direct)
    original=dict(np.load(root/'original_fine_mesh.npz'))
    np.testing.assert_array_equal(rests[-1],original['vertices'])
    np.testing.assert_array_equal(faces[-1],original['faces'])
    assert manifest['original_finest_topology_preserved'] and not manifest['nearest_point_mapping']
    assert manifest['nonintersecting_rest_hierarchy']
    tri=rests[-1][faces[-1]]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    weights=np.zeros(len(rests[-1]))
    np.add.at(weights,faces[-1].ravel(),np.repeat(area/3,3));weights/=weights.sum()
    return hierarchy,original,manifest,weights


def camera():
    view=np.array([1.2,-2.4,1.5]);view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right)
    up=np.cross(view,right)
    return view,np.column_stack((right,up))


def surface(ax,vertices,faces,wire=False):
    view,basis=camera()
    light=np.array([-.8,-.45,.7]);light/=np.linalg.norm(light)
    tri=vertices[faces]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-30)
    colors=np.array([.18,.65,.70])*(.40+.60*np.abs(normals@light))[:,None]
    ax.imshow(rasterize(tri@basis,tri@view,colors,0 if wire else len(faces),700),extent=(-.79,.79,-.70,.79),origin='upper')
    ax.set(xlim=(-.79,.79),ylim=(-.70,.79),aspect='equal');ax.axis('off')


def hierarchy_figures(root,h,original,metrics):
    n=len(h.rests)
    fig,axes=plt.subplots(2,n,figsize=(19,8),dpi=100,facecolor='#f4f6f9')
    fig.subplots_adjust(left=.05,right=.98,bottom=.11,top=.81,hspace=.36,wspace=.12)
    fig.text(.035,.952,'ORIGINAL FINE MESH -> ACTUAL EDGE COLLAPSES',fontsize=23,weight='bold',color='#213147')
    fig.text(.035,.900,'Top: actual simplified meshes  |  Bottom: original-material centerline mapped through each level',fontsize=12,color='#526078')
    ids=np.flatnonzero(np.isclose(original['uv'][:,1],.5))
    material_x=(original['uv'][ids,0]-.5)*100
    for col,k in enumerate(reversed(range(n))):
        x,f=h.rests[k],h.faces[k]
        surface(axes[0,col],x,f,wire=True)
        axes[0,col].set_title(f'L{k}'+(' / ORIGINAL' if k==n-1 else '')+f'\n{len(x):,} vertices | {len(f):,} triangles',fontsize=12,color='#34465c')
        p=h.to_finest[k]
        ax=axes[1,col]
        ax.plot(material_x,original['vertices'][ids,2]*1000,'--',c='#546273',lw=2,label='Original fine')
        ax.plot(material_x,(p.U@x)[ids,2]*1000,c='#c77b42',lw=1.7,label='Linear mapping')
        ax.plot(material_x,p.position(x)[ids,2]*1000,c='#148991',lw=1.4,label='PSQ recovery')
        ax.set(xlim=(-50,50),ylim=(-22,22),xlabel='Original material coordinate (cm)')
        ax.set_title('Rest-shape profile',fontsize=11)
        ax.grid(alpha=.2)
        if col==0:ax.set_ylabel('Height z (mm)');ax.legend(fontsize=8)
    fig.text(.035,.035,'Native probabilistic quadrics + successive self-parameterization; original finest vertices and faces preserved exactly.',fontsize=11,color='#526078')
    fig.savefig(root/'hierarchy.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,n,figsize=(19,4.7),dpi=100,facecolor='#f4f6f9')
    for ax,k in zip(axes,range(n)):
        surface(ax,h.to_finest[k].position(h.rests[k]),h.faces[-1])
        ax.set_title(f'From L{k} -> original L{n-1}\n{len(h.rests[k]):,} -> {len(h.rests[-1]):,} vertices',fontsize=11)
    fig.suptitle('PSQ recovers the original rest shape from every decimated level',fontsize=17)
    fig.subplots_adjust(top=.76,bottom=.06,left=.02,right=.98,wspace=.04)
    fig.savefig(root/'rest_reconstruction.png',dpi=130);plt.close(fig)


def driven_poses(rest,kind):
    poses=[];truth=[];controls=[]
    for t in np.linspace(0,1,COUNT):
        s=np.sin(np.pi*t)**2
        if kind=='rotation':
            Q=rotation(np.pi/2*s)
            x=rest@Q.T
            controls.append(90*s)
        else:
            # Smooth spatial map that is the identity at s=0; it preserves the
            # actual decimated rest z instead of replacing it with a flat grid.
            u,v,z=rest.T
            beta=1.0*s
            if beta>1e-10:
                y=np.sin(beta*v)/beta-z*np.sin(beta*v)
                zb=(1-np.cos(beta*v))/beta+z*np.cos(beta*v)
            else:y,zb=v,z
            theta=.6*s*v
            x=np.column_stack((u*np.cos(theta)+zb*np.sin(theta),y,-u*np.sin(theta)+zb*np.cos(theta)))
            x=x@rotation(np.deg2rad(55)*s).T
            controls.append(s)
        poses.append(x)
    return np.array(poses),np.array(controls)


def collision_frames(vertices,faces,states):
    edges=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    cm=ipctk.CollisionMesh(vertices,edges,faces.astype(np.int32))
    return [i for i,x in enumerate(states) if ipctk.has_intersections(cm,x)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/psq_intrinsic_pleats_demo')
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    if args.prepare:prepare(root)
    shutil.copy2(__file__,root/'source_snapshot.py')
    h,original,manifest,weights=load(root)
    rows=[]
    for k,p in enumerate(h.to_finest):
        x=h.rests[k]
        difference=p.U@x-h.rests[-1]
        restored=p.position(x)-h.rests[-1]
        row=dict(level=k,vertices=len(x),triangles=len(h.faces[k]),
            linear_material_rms_mm=float(np.sqrt(np.einsum('v,vc,vc',weights,difference,difference))*1000),
            normal_residual_rms_mm=float(np.sqrt(np.sum(weights*p.gamma**2))*1000),
            tangent_residual_rms_mm=float(np.sqrt(np.sum(weights*np.sum(p.tangent**2,axis=1)))*1000),
            psq_rest_max_error_m=float(np.abs(restored).max()))
        assert row['psq_rest_max_error_m']<1e-12
        rows.append(row)
    hierarchy_figures(root,h,original,rows)
    audits={}
    # Show the genuinely coarsest level. Record intersections rather than claim
    # the prolongation operator alone is a collision-safe refinement method.
    k=0;p=h.to_finest[k]
    coarse=SimpleNamespace(faces=h.faces[k]);fine=SimpleNamespace(faces=h.faces[-1])
    for kind in ('rotation','bend_twist'):
        poses,controls=driven_poses(h.rests[k],kind)
        truth=np.array([h.rests[-1]@rotation(np.deg2rad(a)).T for a in controls]) if kind=='rotation' else None
        outputs,errors,audit=evaluate(p,poses,truth,weights)
        audit['coarse_intersecting_frames']=collision_frames(h.rests[k],h.faces[k],poses)
        audit['psq_intersecting_frames']=collision_frames(h.rests[-1],h.faces[-1],outputs['psq'])
        if errors:audit['at_90_degrees_mass_weighted_rms_mm']={m:float(v[COUNT//2]) for m,v in errors.items()}
        audits[kind]=audit
        print(kind,audit,flush=True)
        np.savez_compressed(root/f'{kind}_geometry.npz',coarse=poses,controls=controls,**outputs)
        render(root,kind,coarse,fine,poses,outputs,errors,controls,intrinsic=True,driver_level=k)
        if kind=='bend_twist':
            # Same spatial driver applied separately to each actual coarse mesh;
            # each output uses its direct map to the ORIGINAL fine topology.
            fig,axes=plt.subplots(1,len(h.rests),figsize=(19,4.7),dpi=100,facecolor='#f4f6f9')
            all_level=[]
            for level,ax in enumerate(axes):
                coarse_poses,_=driven_poses(h.rests[level],kind)
                recovered=h.to_finest[level].position(coarse_poses[COUNT//2])
                surface(ax,recovered,h.faces[-1])
                ax.set_title(f'Driven L{level} -> original L4\n{len(h.rests[level]):,} driver vertices',fontsize=11)
                all_level.append(recovered)
            fig.suptitle('Prescribed bend + twist / PSQ reconstruction from each actual level',fontsize=17)
            fig.subplots_adjust(top=.76,bottom=.06,left=.02,right=.98,wspace=.04)
            fig.savefig(root/'deformed_all_levels.png',dpi=130);plt.close(fig)
            np.save(root/'deformed_all_levels.npy',all_level)
    result=dict(scope='Actual fine-to-coarse preprocessing plus prescribed deformation and PSQ position prolongation; no dynamics or safe-expansion solve',
        original_geometry_exact=True,original_topology_exact=True,nearest_point_mapping=False,
        source_mesh_sha256=hashlib.sha256((root/'original_fine_mesh.npz').read_bytes()).hexdigest(),
        preprocessing_manifest_sha256=hashlib.sha256((root/'assets/manifest.json').read_bytes()).hexdigest(),
        levels=rows,driver_level=k,clips=audits,
        metric_note='Linear material RMS includes tangent correspondence drift and is not a pure wrinkle-amplitude measure; normal and tangent residuals are separate.',
        simplification_parameters=manifest['quadric_parameters'],collapse_counts=[x['collapses'] for x in manifest['expansion_histories']],
        limitations='Nonlinear PSQ transport is not a collision or strain guarantee. The prescribed deformed fine mesh is not a converged physical solution.')
    result['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (Path(__file__),ROOT/'progressive/prolongation.py',ROOT/'progressive/prepare_intrinsic.py',ROOT/'scripts/demo_psq_shallow_pleats.py')}
    write_json(root/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
