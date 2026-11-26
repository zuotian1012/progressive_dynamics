"""Check saved cloth triangles for intersection with the solid Dynamic Drape gear."""
import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit

from tools.pd.run_velpro_drape import CELLS


EXTERIOR_INSET_M = 1e-7


@njit
def inside_gear_triangles(tris, planes, bounds, gear_bounds):
    ids = []
    polygon = np.empty((9, 3))
    other = np.empty_like(polygon)
    for index, triangle in enumerate(tris):
        low, high = np.empty(3), np.empty(3)
        for axis in range(3):
            low[axis] = min(triangle[0, axis], triangle[1, axis], triangle[2, axis])
            high[axis] = max(triangle[0, axis], triangle[1, axis], triangle[2, axis])
        if np.any(high < gear_bounds[0]) or np.any(low > gear_bounds[1]):
            continue
        for piece in range(len(planes)):
            if np.any(high < bounds[piece, 0]) or np.any(low > bounds[piece, 1]):
                continue
            polygon[:3] = triangle
            size = 3
            for plane in planes[piece]:
                count = 0
                for i in range(size):
                    p, q = polygon[i], polygon[(i + 1) % size]
                    dp = p[0]*plane[0] + p[1]*plane[1] + p[2]*plane[2] + plane[3]
                    dq = q[0]*plane[0] + q[1]*plane[1] + q[2]*plane[2] + plane[3]
                    if dp >= 0:
                        other[count] = p
                        count += 1
                    if (dp < 0 < dq) or (dq < 0 < dp):
                        other[count] = p + dp / (dp - dq) * (q - p)
                        count += 1
                size = count
                if size == 0:
                    break
                polygon[:size] = other[:size]
            if size:
                ids.append(index)
                break
    return ids


def gear_clip_geometry(path):
    """Decompose the closed extruded gear into wedges for contact verification."""
    with np.load(path, allow_pickle=False) as gear:
        verts, faces = gear["vertices"], gear["faces"]
    top, bottom = verts[:,2].max(), verts[:,2].min()
    top_faces = faces[np.all(verts[faces,2] == top,axis=1)]
    epsilon = EXTERIOR_INSET_M
    planes, bounds = [], []
    for face in top_faces:
        triangle = verts[face]
        xy = triangle[:,:2]
        assert np.array_equal(xy[0], [0,0])
        assert np.linalg.det(xy[1:]) > 0
        edge = np.roll(xy,-1,axis=0)-xy
        normals = np.column_stack((-edge[:,1],edge[:,0]))
        normals /= np.linalg.norm(normals,axis=1,keepdims=True)
        constants = -np.einsum('ij,ij->i',xy,normals)
        # Only the actual exterior of the gear is inset, shared radial edges stay flush.
        constants[1] -= epsilon
        planes.append(np.vstack((np.column_stack((normals,np.zeros(3),constants)),
                                 [0,0,1,-bottom-epsilon],[0,0,-1,top-epsilon])))
        bounds.append([[*xy.min(axis=0),bottom],[*xy.max(axis=0),top]])
    planes, bounds = np.asarray(planes), np.asarray(bounds)
    gear_bounds = np.stack((verts.min(axis=0),verts.max(axis=0)))
    assert len(top_faces) == 48
    p=verts[top_faces[3]].mean(axis=0)
    p[2]=(top+bottom)/2
    simple=np.asarray([p+[.001,0,0],p+[0,.001,0],p+[0,0,.001]])
    assert inside_gear_triangles(np.asarray([simple,simple+[0,0,1]]),planes,bounds,gear_bounds)==[0]
    return planes, bounds, gear_bounds


def intersecting_frames(positions, triangles, clip_geometry):
    bad = {}
    for frame, x in enumerate(positions):
        ids = inside_gear_triangles(x[triangles], *clip_geometry)
        if ids:
            bad[str(frame)] = dict(triangles=len(ids), first_triangle=ids[0])
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory
    clip_geometry = gear_clip_geometry(root / "gear.npz")
    last = len(CELLS) - 1
    cases = [("preview_L0", 0), ("direct_fine", last)]
    cases.extend((f"{mode}_L{level}", level) for mode in ("paper_eq5", "velpro")
                 for level in range(1, last + 1))
    result = {}
    for name, level in cases:
        with np.load(root / "hierarchy" / f"mesh_L{level}.npz", allow_pickle=False) as data:
            triangles = data["triangles"]
        with np.load(root / name / "state.npz", allow_pickle=False) as data:
            positions = data["x"]
        bad = intersecting_frames(positions, triangles, clip_geometry)
        print(name, 'bad frames:', len(bad), 't=0.46s:',bad.get('46'),flush=True)
        result[name]=dict(saved_frames=len(positions),intersections=bad)
    (root / "gear_contact_audit.json").write_text(json.dumps(dict(
        method='Full cloth triangle vs solid gear with convex clipping per wedge',
        exterior_inset_m=EXTERIOR_INSET_M, include_self_collision=False, runs=result),indent=2)+'\n')

if __name__=='__main__':
    main()
