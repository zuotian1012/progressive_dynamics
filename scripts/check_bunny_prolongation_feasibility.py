"""Test the coarse solution's direct PSQ image against the finest mesh constraints."""
import argparse
import json
from pathlib import Path
import sys

import ipctk
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from progressive.prolongation import ShellProlongation, frames


def collision_mesh(vertices, faces):
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges = np.unique(np.sort(edges, axis=1), axis=0).astype(np.int32)
    return ipctk.CollisionMesh(vertices, edges, faces.astype(np.int32))


def measure(run, samples):
    meta = json.loads((run/'run.json').read_text())
    if meta['scene'] != 'bunny' or not json.loads((run/'continuation_audit.json').read_text())['passed']:
        raise ValueError('Expected an audited bunny trajectory')
    with np.load(run/'operator/operator.npz') as op:
        matrix = csr_matrix((op['U_data'], op['U_indices'], op['U_indptr']), shape=tuple(op['U_shape']))
        fine, faces = op['fine_rest'], op['fine_faces']
        transfer = ShellProlongation(op['coarse_rest'], fine, op['coarse_faces'], matrix, op['anchors'])
    inv, areas = frames(fine, faces)[2:]
    with np.load(Path(meta['assets'])/'bunny.npz') as bunny:
        obstacle, obstacle_faces = bunny['vertices'], bunny['faces']
    cloth_mesh = collision_mesh(fine, faces)
    combined = collision_mesh(np.vstack([fine, obstacle]), np.vstack([faces, obstacle_faces + len(fine)]))
    labels = np.r_[np.zeros(len(fine), np.int32), np.ones(len(obstacle), np.int32)]
    combined.can_collide = ipctk.make_vertex_patches_filter(labels)
    states = np.load(run/'positions.npy', mmap_mode='r')
    limit = meta['material']['strain_limit']
    floor = meta['ground']['height_m'] + meta['material']['thickness']/2
    indices = range(meta['completed_steps']+1) if samples is None else samples
    rows = []
    for k in indices:
        if not 0 <= k <= meta['completed_steps']:
            raise ValueError(f'Frame {k} outside accepted trajectory')
        x = transfer.position(states[k])
        tri = x[faces]
        F = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=2) @ inv
        stretch = np.linalg.svd(F, compute_uv=False)[:, 0]
        over = stretch > limit + 1e-9
        rows.append(dict(frame=k, self_intersection=bool(ipctk.has_intersections(cloth_mesh, x)),
                         bunny_intersection=bool(ipctk.has_intersections(combined, np.vstack([x, obstacle]))),
                         max_stretch=float(stretch.max()), faces_over_limit=int(over.sum()),
                         area_pct_over_limit=float(100*areas[over].sum()/areas.sum()),
                         floor_violation_mm=1000*max(0., floor-float(x[:, 2].min()))))
    return dict(run=str(run.resolve()), source_level=meta['level'], target_vertices=len(fine),
                target_faces=len(faces), strain_limit=limit, frames=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', help='Defaults to every accepted state')
    args = parser.parse_args()
    result = measure(args.run, args.frames)
    args.output.write_text(json.dumps(result, indent=2))
    rows = result['frames']
    print(f"{len(rows)} states, {sum(row['bunny_intersection'] for row in rows)} intersect the bunny, "
          f"{sum(row['faces_over_limit'] > 0 for row in rows)} exceed the finest strain limit")
