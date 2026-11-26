"""Redistribute finest-mesh vertices on the original surface, preserving topology/boundary.

This is geometry preprocessing only. No material, contact, solver, or PD change.
Projection searches nearby original triangles; it is not used as a PD transfer map.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point
import ipctk


class SurfaceProjector:
    def __init__(self, vertices, faces):
        self.triangles=vertices[faces]
        self.tree=cKDTree(self.triangles.mean(1))

    def project(self, points, neighbors=32):
        projected=[]
        for begin in range(0,len(points),2048):
            p=points[begin:begin+2048]
            _,ids=self.tree.query(p,k=min(neighbors,len(self.triangles)))
            targets=np.broadcast_to(p[:,None,:],(*ids.shape,3))
            q=closest_point(self.triangles[ids].reshape(-1,3,3),targets.reshape(-1,3)).reshape(*ids.shape,3)
            best=np.argmin(np.sum((q-targets)**2,axis=2),axis=1)
            projected.append(q[np.arange(len(p)),best])
        return np.concatenate(projected)


def metrics(v,f,edges,collision_mesh,source_v,source_f):
    tri=v[f];edge=tri[:,[1,2,0]]-tri
    lengths=np.linalg.norm(edge,axis=2)
    areas=np.linalg.norm(np.cross(edge[:,0],-edge[:,2]),axis=1)*.5
    altitudes=2*areas/lengths.max(1)
    quality=4*np.sqrt(3)*areas/(lengths**2).sum(1)
    contacts=ipctk.NormalCollisions();contacts.collision_set_type=ipctk.NormalCollisions.IPC
    contacts.use_area_weighting=False;contacts.build(collision_mesh,v,.0015,.0002)
    old_tri=source_v[source_f]
    old_normals=np.cross(old_tri[:,1]-old_tri[:,0],old_tri[:,2]-old_tri[:,0])
    new_normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    reverse=SurfaceProjector(v,f).project(source_v)
    return dict(vertices=len(v),faces=len(f),minimum_quality=float(quality.min()),median_quality=float(np.median(quality)),
        minimum_altitude_mm=float(altitudes.min()*1000),faces_altitude_below_1_7mm=int((altitudes<.0017).sum()),
        total_area_m2=float(areas.sum()),active_self_contact_candidates=len(contacts),
        minimum_active_distance_mm=None if contacts.empty() else float(np.sqrt(contacts.compute_minimum_distance(collision_mesh,v))*1000),
        self_intersections=bool(ipctk.has_intersections(collision_mesh,v)),
        reversed_face_normals=int((np.sum(old_normals*new_normals,axis=1)<=0).sum()),
        maximum_vertex_relocation_mm=float(np.linalg.norm(v-source_v,axis=1).max()*1000),
        original_vertices_to_new_surface_max_mm=float(np.linalg.norm(source_v-reverse,axis=1).max()*1000),
        original_vertices_to_new_surface_rms_mm=float(np.sqrt(np.mean(np.sum((source_v-reverse)**2,axis=1)))*1000))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=32);p.add_argument('--step',type=float,default=.5)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.source) as m:v0=m['vertices'];f=m['faces'].astype(np.int32)
    all_edges=np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    edges,counts=np.unique(all_edges,axis=0,return_counts=True)
    assert counts.max()==2
    boundary=np.unique(edges[counts==1]);n=len(v0)
    row=np.r_[edges[:,0],edges[:,1]];col=np.r_[edges[:,1],edges[:,0]]
    degrees=np.bincount(row,minlength=n)
    average=coo_matrix((1/degrees[row],(row,col)),shape=(n,n)).tocsr()
    project=SurfaceProjector(v0,f);v=v0.copy()
    cm=ipctk.CollisionMesh(v0,edges.astype(np.int32),f)
    rows=[];samples={0,1,4,8,16,32,a.iterations}
    for it in range(a.iterations+1):
        if it in samples:
            report=metrics(v,f,edges,cm,v0,f);report['iteration']=it
            report['boundary_exact']=bool(np.array_equal(v[boundary],v0[boundary]))
            report['new_vertices_to_original_surface_max_mm']=float(np.linalg.norm(v-project.project(v),axis=1).max()*1000)
            name=f'iteration_{it:03d}'
            np.savez_compressed(a.output/f'{name}.npz',vertices=v,faces=f)
            with (a.output/f'{name}.obj').open('w') as stream:
                for xyz in v:stream.write('v '+' '.join(f'{c:.17g}' for c in xyz)+'\n')
                for face in f:stream.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
            rows.append(report);(a.output/'metrics.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(report),flush=True)
        if it==a.iterations:break
        proposed=v+a.step*(average@v-v)
        proposed[boundary]=v0[boundary]
        v=project.project(proposed)
        v[boundary]=v0[boundary]
    (a.output/'provenance.json').write_text(json.dumps(dict(source=str(a.source.resolve()),
        source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),iterations=a.iterations,step=a.step,
        topology_preserved=True,boundary_vertices_fixed=len(boundary),projection_neighbors=32,
        projection_scope='Surface preprocessing, not a replacement for intrinsic PD correspondence.',
        limitations=['Vertex-to-surface samples are not a continuous Hausdorff bound.',
            'Projection uses a local triangle search; candidates require independent topology/intersection/contact checks.',
            'Static contact reduction does not prove dynamic artifact removal.']),indent=2))


if __name__=='__main__':main()
