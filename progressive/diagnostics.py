"""Offline contact reactions; these measurements do not modify a trajectory."""
import numpy as np
import trimesh


def nearest_material_point(vertices,faces,point):
    """Pick ONCE at closure; keep its face/barycentric coordinates thereafter."""
    triangles=vertices[faces]
    nearest=trimesh.triangles.closest_point(triangles,np.broadcast_to(point,(len(faces),3)))
    distance=np.linalg.norm(nearest-point,axis=1)
    face=int(np.argmin(distance))
    weights=trimesh.triangles.points_to_barycentric(triangles[face:face+1],nearest[face:face+1])[0]
    return face,weights,float(distance[face])


def body_contact_forces(contact,embedding,previous,current,dt):
    contact.begin_step(previous,dt)
    normal=contact.normal(current)
    ng=contact.barrier.gradient(normal,contact.mesh,current).reshape(-1,3)
    fg=contact.friction.gradient(contact.tangential,contact.mesh,current-previous).reshape(-1,3)
    cursor=embedding.n;result={}
    for name,v,_,_ in embedding.bodies:
        result[name]={'normal_force_N':(-ng[cursor:cursor+len(v)].sum(axis=0)).tolist(),
                      'friction_force_N':(-fg[cursor:cursor+len(v)].sum(axis=0)).tolist()}
        cursor+=len(v)
    return result
