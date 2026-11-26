"""Compare cloth motion to BDF1 free fall before reaching the bunny or floor.

An initially flat mesh may already have active self-contact stencils. In
that case the analytic curve is a diagnostic for the resulting rest forces.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def measure(run):
    meta = json.loads((run / 'run.json').read_text())
    if meta['scene'] != 'bunny' or meta['pins'] or meta['config']['integrator']['type'] != 'bdf1':
        raise ValueError('Expected the unpinned BDF1 bunny drop')
    assets = Path(meta['assets'])
    mesh = np.load(assets / f"mesh_{meta['level']}.npz")
    rest, faces = mesh['vertices'], mesh['faces']
    states = np.load(run / 'positions.npy', mmap_mode='r')
    if not np.array_equal(states[0], rest) or np.ptp(rest[:, 2]) > 1e-12:
        raise ValueError('Expected a horizontal cloth at rest')
    gravity = np.asarray(meta['config']['gravity']).reshape(3)
    if np.any(gravity[:2]) or gravity[2] >= 0:
        raise ValueError('Expected vertical downward gravity')
    triangle = rest[faces]
    area = np.linalg.norm(np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0]), axis=1)/2
    mass = np.bincount(faces.ravel(), weights=np.repeat(area/3, 3)) / area.sum()
    top = max(float(np.load(assets/'bunny.npz')['vertices'][:, 2].max()), meta['ground']['height_m'])
    activation = meta['config']['contact']['d_hat'] + meta['material']['thickness']
    dt = meta['dt']
    rows = []
    for frame in range(1, meta['completed_steps']+1):
        displacement = gravity * (dt*dt*frame*(frame+1)/2)
        if rest[:, 2].min() + displacement[2] <= top + activation:
            break
        error = (states[frame] - rest) - displacement
        norm2 = np.einsum('ij,ij->i', error, error)
        rows.append(dict(frame=frame, time_s=frame*dt, rms_mm=float(np.sqrt(mass@norm2)*1000),
                         max_mm=float(np.sqrt(norm2.max())*1000), max_vertex=int(norm2.argmax())))
    return dict(run=str(run.resolve()), level=meta['level'], method=meta['method'], frames=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = [measure(run) for run in args.runs]
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2))
    for run in rows:
        if run['frames']:
            print(run['run'], run['frames'][-1])
