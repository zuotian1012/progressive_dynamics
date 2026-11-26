"""Local tangent-plane optimization of short triangle altitudes, boundary fixed."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import ipctk
from improve_shirt_finest_mesh import SurfaceProjector,metrics
from flip_shirt_short_altitudes import altitude


def altitude_gradient(tri):
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area2=np.linalg.norm(cross,axis=1)
    normal=cross/np.maximum(area2,1e-20)[:,None]
    grad_area=np.stack([np.cross(tri[:,1]-tri[:,2],normal),np.cross(tri[:,2]-tri[:,0],normal),np.cross(tri[:,0]-tri[:,1],normal)],axis=1)
    edge=tri[:,[1,2,0]]-tri;lengths=np.linalg.norm(edge,axis=2);longest=lengths.argmax(1)
    rows=np.arange(len(tri));length=lengths[rows,longest];unit=edge[rows,longest]/length[:,None]
    grad_length=np.zeros_like(tri);grad_length[rows,longest]=-unit;grad_length[rows,(longest+1)%3]=unit
    h=area2/length
    return h,grad_area/length[:,None,None]-area2[:,None,None]/length[:,None,None]**2*grad_length


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.source) as m:v=m['vertices'].copy();f=m['faces'].astype(np.int32)
    with np.load(a.reference) as m:rv=m['vertices'];rf=m['faces'].astype(np.int32)
    edges,counts=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    boundary=np.unique(edges[counts==1]);project=SurfaceProjector(rv,rf);history=[]
    cm=ipctk.CollisionMesh(v,edges.astype(np.int32),f)
    for iteration in range(12):
        h=altitude(v[f]);bad=np.flatnonzero(h<.00185)
        contacts=ipctk.NormalCollisions();contacts.collision_set_type=ipctk.NormalCollisions.IPC
        contacts.use_area_weighting=False;contacts.build(cm,v,.00165,.0002)
        stencils=[(c,np.array([int(i) for i in c.vertex_ids(edges,f) if i>=0])) for c in contacts]
        if not len(bad) and not stencils:break
        ids=np.unique(np.concatenate([f[bad].ravel(),*[ids for c,ids in stencils]]))
        for _ in range(2):ids=np.unique(f[np.isin(f,ids).any(1)])
        ids=np.setdiff1d(ids,boundary);active_faces=np.flatnonzero(np.isin(f,ids).any(1));af=f[active_faces]
        start=v.copy();normals=np.zeros_like(v);tri=v[f];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        for j in range(3):np.add.at(normals,f[:,j],norm)
        normal=normals[ids]/np.linalg.norm(normals[ids],axis=1)[:,None]
        axis=np.eye(3)[np.abs(normal).argmin(1)];u=np.cross(normal,axis);u/=np.linalg.norm(u,axis=1)[:,None]
        basis=np.stack([u,np.cross(normal,u)],axis=2)
        def fun(uv):
            uv=uv.reshape(-1,2);current=start.copy();current[ids]+=np.einsum('nij,nj->ni',basis,uv)
            heights,dh=altitude_gradient(current[af]);deficit=np.minimum(heights/.00195-1,0)
            value=float(np.dot(deficit,deficit)+1e-4*np.sum((uv/.003)**2))
            grad=np.zeros_like(v);g=2*deficit[:,None,None]/.00195*dh
            for j in range(3):np.add.at(grad,af[:,j],g[:,j])
            for collision,vids in stencils:
                positions=current[vids].ravel();distance2=collision.compute_distance(positions)
                gap=min(distance2/.00195**2-1,0)
                if gap:
                    value+=gap*gap
                    np.add.at(grad,vids,2*gap/.00195**2*collision.compute_distance_gradient(positions).reshape(-1,3))
            guv=np.einsum('nji,nj->ni',basis,grad[ids])+2e-4*uv/.003**2
            return value,guv.ravel()
        z=np.zeros(len(ids)*2)
        if iteration==0:
            _,g=fun(z);checks=[]
            for i in np.argsort(np.abs(g))[-8:]:
                dz=np.zeros_like(z);dz[i]=1e-8
                fd=(fun(z+dz)[0]-fun(z-dz)[0])/2e-8
                checks.append(abs(fd-g[i])/max(1,abs(fd),abs(g[i])))
            assert max(checks)<1e-5,checks
        opt=minimize(fun,z,jac=True,method='L-BFGS-B',bounds=[(-.004,.004)]*len(z),options=dict(maxiter=200,ftol=1e-13,gtol=1e-7,maxls=30))
        proposed=start.copy();proposed[ids]+=np.einsum('nij,nj->ni',basis,opt.x.reshape(-1,2));proposed[ids]=project.project(proposed[ids])
        v=proposed;report=metrics(v,f,edges,cm,rv,rf);report.update(iteration=iteration,optimized_vertices=len(ids),objective=float(opt.fun),optimizer_message=str(opt.message))
        history.append(report);print(json.dumps(report),flush=True)
        np.savez_compressed(a.output/f'iteration_{iteration:02d}.npz',vertices=v,faces=f)
        (a.output/'metrics.json').write_text(json.dumps(history,indent=2))
        if report['self_intersections']:raise RuntimeError('Candidate self-intersects; do not use')
        if not report['active_self_contact_candidates']:break
    np.savez_compressed(a.output/'mesh.npz',vertices=v,faces=f)
    with (a.output/'mesh.obj').open('w') as stream:
        for xyz in v:stream.write('v '+' '.join(f'{x:.17g}' for x in xyz)+'\n')
        for face in f:stream.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
    np.testing.assert_array_equal(v[boundary],rv[boundary])


if __name__=='__main__':main()
