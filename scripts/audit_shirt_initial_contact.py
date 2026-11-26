"""Read-only IPC activation preflight, beyond the earlier thickness-only asset check."""
import argparse
import json
from pathlib import Path
import numpy as np
import ipctk


def audit(assets,dhat=.0015,clearance=.0002):
    result=[]
    for path in sorted(assets.glob('mesh_*.npz')):
        level=int(path.stem.split('_')[-1]);d=np.load(path);v=d['vertices'];f=d['faces'].astype(np.int32)
        edges=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
        tri=v[f];ev=tri[:,[1,2,0]]-tri;area=.5*np.linalg.norm(np.cross(ev[:,0],-ev[:,2]),axis=1)
        quality=4*np.sqrt(3)*area/(ev*ev).sum((1,2))
        mesh=ipctk.CollisionMesh(v,edges,f);contacts=ipctk.NormalCollisions()
        contacts.collision_set_type=ipctk.NormalCollisions.IPC;contacts.use_area_weighting=False
        contacts.build(mesh,v,dhat,clearance)
        same_face=[]
        for c in contacts:
            ids=[int(i) for i in c.vertex_ids(edges,f) if i>=0]
            if len(ids)==3:
                face_ids=np.flatnonzero(np.all(np.isin(f,ids),axis=1))
                for i in face_ids:same_face.append(dict(face=int(i),vertices=ids,type=type(c).__name__,quality=float(quality[i])))
        result.append(dict(level=level,vertices=len(v),faces=len(f),dhat_m=dhat,clearance_m=clearance,
            initial_active_contact_count=len(contacts),minimum_active_distance_m=None if contacts.empty() else float(np.sqrt(contacts.compute_minimum_distance(mesh,v))),
            minimum_triangle_quality=float(quality.min()),median_triangle_quality=float(np.median(quality)),
            worst_face=int(quality.argmin()),same_face_active_stencils=same_face,
            note='Independent IPC Toolkit candidate check, not an exact dump of native libuipc constraints. Active contacts can be physically legitimate; count alone is not an invalidity test.'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('assets',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contact-distance',type=float,default=.0015,help='IPC activation distance in meters')
    p.add_argument('--clearance',type=float,default=.0002,help='Sum of the two surface radii in meters')
    a=p.parse_args();result=audit(a.assets,a.contact_distance,a.clearance);a.output.write_text(json.dumps(result,indent=2));print(json.dumps([{k:v for k,v in r.items() if k not in ('same_face_active_stencils','note')} for r in result],indent=2))
