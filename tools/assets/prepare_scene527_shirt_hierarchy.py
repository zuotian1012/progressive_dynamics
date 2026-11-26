"""Build a Scene527 shirt hierarchy while preserving all three handed-off OBJ meshes."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import trimesh

from tools.assets.prepare_psq_hierarchy import write_binding


INITIAL = np.array([.667, .015, .93])
LOW_FACE_COUNTS = (4950, 2780, 1560)
MID_VERTICES = 13974


def read_obj(path):
    mesh = trimesh.load(path, process=False)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)


def rotate(raw):
    return np.asarray(raw)[..., (0, 2, 1)] * [1., 1., -1.]


def shell_mesh(vertices, faces):
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def midpoint_subdivision(coarse, faces, finest, fine_faces):
    edges = np.unique(np.sort(faces[:, ((0, 1), (1, 2), (2, 0))], axis=-1).reshape(-1, 2), axis=0)
    centers = coarse[edges].mean(axis=1)
    if not np.array_equal(coarse, finest[:len(coarse)]) or len(edges) != len(finest) - len(coarse):
        raise ValueError('Expected original 13,767-face mesh and its midpoint-subdivided mesh')
    error, indices = cKDTree(finest[len(coarse):]).query(centers)
    if len(np.unique(indices)) != len(edges) or error.max() > .00001:
        raise ValueError(f'Finest mesh does not match midpoint coordinates: {error.max()} mm')
    midpoint = dict(zip(map(tuple, edges), indices + len(coarse), strict=True))
    native_triangles = []
    for a, b, c in faces:
        u, v, w = (midpoint[tuple(sorted(pair))] for pair in ((a,b),(b,c),(c,a)))
        native_triangles.extend(((a,u,w),(u,b,v),(w,v,c),(u,v,w)))
    if set(map(tuple, np.sort(native_triangles, axis=1))) != set(map(tuple, np.sort(fine_faces, axis=1))):
        raise ValueError('The original finest mesh has different subtriangle connectivity')
    lengths = np.linalg.norm(coarse[edges[:, 1]] - coarse[edges[:, 0]], axis=1)
    selected = np.sort(np.argsort(lengths, kind='stable')[-(MID_VERTICES - len(coarse)):])
    fine_ids = indices[selected] + len(coarse)
    middle = np.vstack((coarse, finest[fine_ids]))
    selected_edges = dict(zip(map(tuple, edges[selected]), range(len(coarse), len(middle)), strict=True))
    middle_faces = []
    for original in map(tuple, faces):
        pieces = [original]
        for a, b in zip(original, original[1:] + original[:1], strict=True):
            m = selected_edges.get(tuple(sorted((a, b))))
            if m is None:
                continue
            for i, triangle in enumerate(pieces):
                if a in triangle and b in triangle:
                    k = triangle.index(a)
                    if triangle[(k+1)%3] != b:
                        k = triangle.index(b)
                    u, v, w = (triangle[(k+j)%3] for j in range(3))
                    pieces[i:i+1] = [(u, m, w), (m, v, w)]
                    break
            else:
                raise AssertionError('Missing original triangle boundary edge')
        middle_faces.extend(pieces)
    return middle, np.asarray(middle_faces, dtype=np.int32), float(error.max())


def sparse_rows(cols, weights, n):
    return sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(len(cols)), cols.shape[1]), cols.ravel())),
                             shape=(len(cols), n))


def oriented_surface_map(coarse, faces, fine, fine_faces):
    """Match two preserved input assets using sided surface projection."""
    mesh = shell_mesh(coarse, faces)
    target = shell_mesh(fine, fine_faces)
    centers = mesh.triangles_center
    n_candidates = min(len(faces), 48)
    _, candidates = cKDTree(centers).query(fine, k=n_candidates)
    xyz = np.repeat(fine, n_candidates, axis=0)
    triangles = coarse[faces[candidates]].reshape(-1, 3, 3)
    points = trimesh.triangles.closest_point(triangles, xyz).reshape(len(fine), n_candidates, 3)
    dot = np.einsum('nki,ni->nk', mesh.face_normals[candidates], target.vertex_normals)
    distance = np.linalg.norm(points - fine[:, None, :], axis=2)
    score = np.where(dot > .25, distance + .0005 * (1 - dot), np.inf)
    closest = score.argmin(axis=1)
    rows = np.arange(len(fine))
    ids = candidates[rows, closest]
    world_points = points[rows, closest]
    if not np.all(np.isfinite(score[rows, closest])):
        raise ValueError('No consistently oriented material triangle for a target vertex')

    # Coincident vertices inherit their exact basis value; do not move them to
    # another triangle on the nearby opposing shirt layer.
    nearest_dist, nearest_vert = cKDTree(coarse).query(fine)
    exact = nearest_dist < 1e-10
    support = np.asarray(mesh.vertex_faces)[nearest_vert[exact]]
    dots = mesh.face_normals[support] @ target.vertex_normals[exact, :, None]
    dots = np.where(support >= 0, dots[..., 0], -np.inf)
    ids[exact] = support[np.arange(len(support)), dots.argmax(axis=1)]
    world_points[exact] = coarse[nearest_vert[exact]]
    weights = trimesh.triangles.points_to_barycentric(coarse[faces[ids]], world_points)
    if np.min(weights) < -1e-8:
        raise ValueError(f'Closest material point has invalid barycentric weight {np.min(weights)}')
    weights = np.maximum(weights, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    error = np.linalg.norm(world_points - fine, axis=1)
    stats = dict(exact_vertices=int(exact.sum()), oriented_normal_min=float(np.min(dot[rows, closest])),
                 projection_distance_p95_m=float(np.percentile(error, 95)), projection_distance_max_m=float(error.max()))
    return sparse_rows(faces[ids], weights, len(coarse)), ids, stats


def quality(vertices, faces):
    mesh = shell_mesh(vertices, faces)
    lengths = np.linalg.norm(vertices[faces] - vertices[faces[:, (1, 2, 0)]], axis=-1)
    if np.any(mesh.area_faces <= 1e-16):
        raise ValueError('Degenerate mesh face after simplification')
    counts = np.bincount(mesh.edges_unique_inverse)
    if np.any(counts > 2) or len(np.unique(faces)) != len(vertices):
        raise ValueError('Nonmanifold edge or unreferenced vertex')
    return dict(vertices=len(vertices), triangles=len(faces), boundary_edges=int(np.count_nonzero(counts == 1)),
                minimum_angle_deg=float(np.rad2deg(mesh.face_angles).min()),
                aspect_ratio_max=float((lengths.max(axis=1)**2/(2*mesh.area_faces)).max()),
                sliver_faces_below_5deg=int(np.sum(np.rad2deg(mesh.face_angles.min(axis=1)) < 5)))


def write_obj(path, vertices, faces):
    with path.open('w') as file:
        np.savetxt(file, vertices, fmt='v %.17g %.17g %.17g')
        np.savetxt(file, faces+1, fmt='f %d %d %d')


def prepare(handoff, ssp, output, repair=None):
    output.mkdir(parents=True, exist_ok=True)
    assets = handoff / 'assets/cloth'
    originals = {level: assets / f'short-shirt-{count}f.obj' for level, count in ((3,8000),(4,13767),(6,55068))}
    lower = output / 'ssp'
    subprocess.run([str(ssp.resolve()), str(originals[3]), str(lower), '4', *map(str, LOW_FACE_COUNTS)], check=True)
    raw = [np.loadtxt(lower / f'L{i}_vertices.txt', ndmin=2) for i in range(4)]
    faces = [np.loadtxt(lower / f'L{i}_triangles.txt', dtype=np.int32, ndmin=2) for i in range(4)]
    original_4090, faces_4090 = read_obj(originals[3])
    np.testing.assert_array_equal(raw[3], original_4090)
    np.testing.assert_array_equal(faces[3], faces_4090)
    original_mid, f_mid = read_obj(originals[4])
    finest, f_fine = read_obj(originals[6])
    intermediate, f_intermediate, midpoint_error = midpoint_subdivision(original_mid, f_mid, finest, f_fine)
    raw.extend((original_mid, intermediate, finest))
    faces.extend((f_mid, f_intermediate, f_fine))
    if repair:
        pivot = finest.mean(axis=0) * .001
        for level in (1, 2):
            with np.load(repair / f'L{level}.npz') as edits:
                ids = edits['ids']
                world = rotate(raw[level][ids] * .001) + INITIAL + pivot - rotate(pivot)
                np.testing.assert_allclose(world, edits['original'], rtol=0, atol=1e-12)
                raw[level][ids] += edits['delta'][:, (0, 2, 1)] * [1000., -1000., 1000.]
    # Genesis rotates the FEM mesh about its mean vertex. Compensate each
    # spawn position so all levels use the original finest mesh's pivot.
    centers = np.asarray([v.mean(axis=0) * .001 for v in raw])
    shifts = centers - rotate(centers)
    origins = INITIAL + shifts[-1] - shifts
    rest = [rotate(v * .001) + INITIAL + shifts[-1] for v in raw]
    levels = [quality(v, f) | {'obj': f'shirt_L{i}.obj', 'source_obj': originals[i].name if i in originals else None,
                               'scene_origin_m': origins[i].tolist()}
              for i, (v,f) in enumerate(zip(rest, faces, strict=True))]
    matrices, supports, mappings = [], [], []
    for level in range(6):
        if level < 3:
            stem = lower / f'L{level}'
            cols = np.loadtxt(f'{stem}_adjacent_vertices.txt', dtype=int, ndmin=2)
            w = np.loadtxt(f'{stem}_adjacent_weights.txt', ndmin=2)
            f = np.loadtxt(f'{stem}_adjacent_faces.txt', dtype=int, ndmin=1)
            m = sparse_rows(cols, w, len(rest[level]))
            info = {'method': 'SSP edge collapses'}
        else:
            m,f,info = oriented_surface_map(rest[level],faces[level],rest[level+1],faces[level+1])
            info['method'] = 'oriented barycentric surface bind'
        matrices.append(m)
        supports.append(f)
        mappings.append(info)
    operators = []
    target = len(rest) - 1
    for level, m in enumerate(matrices):
        operators.append(write_binding(output, 'adjacent', level, level+1, rest[level], rest[level+1],
                                       faces[level], m, supports[level]))
        chain = m
        descendants = np.arange(len(rest[-1]))
        for k in reversed(range(level+1, target)):
            descendants = np.asarray(matrices[k].argmax(axis=1)).ravel()[descendants]
        for child in matrices[level+1:]:
            chain = child @ chain
        operators.append(write_binding(output, 'direct', level, target, rest[level], rest[-1], faces[level],
                                       chain, supports[level][descendants], proxy=faces[-1]))
        np.savez_compressed(output / f'shirt_L{level}.vertex_map.npz', source_to_lowres=np.asarray(chain.argmax(axis=1)).ravel())
    for operator in operators:
        matrix = sparse.load_npz(output / operator['matrix'])
        operator['max_row_sum_error'] = float(np.max(np.abs(matrix.sum(axis=1) - 1)))
    for level, (vertices, topology) in enumerate(zip(raw,faces,strict=True)):
        obj = output / levels[level]['obj']
        if level in originals:
            shutil.copy2(originals[level],obj)
        else:
            write_obj(obj,vertices,topology)
        np.savez_compressed(output / f'mesh_L{level}.npz', rest=rest[level], triangles=topology)
    shutil.copy2(originals[6].with_suffix('.garment_atlas.npz'), (output/'shirt_L6.obj').with_suffix('.garment_atlas.npz'))
    (output/'manifest.json').write_text(json.dumps(dict(levels=levels,operators=operators,mappings=mappings,
        midpoint_error_mm=midpoint_error),indent=2)+'\n')
    print([(entry['vertices'],entry['triangles']) for entry in levels],flush=True)
    print(mappings,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('handoff', type=Path)
    p.add_argument('ssp', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--repair', type=Path)
    args=p.parse_args()
    prepare(args.handoff,args.ssp,args.output,args.repair)
