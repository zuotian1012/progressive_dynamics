"""Compare every accepted frame against its parent and the same L0 material points."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from progressive.prolongation import ShellProlongation, frames


def rms(delta, weights):
    return np.sqrt(np.einsum('vi,vi,v->', delta, delta, weights))


def measure(root, render_movie=False, accepted_prefix=False):
    chain = json.loads((root / 'chain.json').read_text())
    runs = [Path(chain['seed'])]
    parent = json.loads((runs[0]/'run.json').read_text())
    while parent['level']:
        runs.insert(0, Path(parent['lower_run']))
        parent = json.loads((runs[0]/'run.json').read_text())
    first_child_level = len(runs)
    runs.extend(Path(row['run']) for row in chain['levels'])
    end_frame = chain.get('steps')
    output = root
    if accepted_prefix:
        current = Path(chain['current_run'])
        partial = json.loads((current/'run.json').read_text())
        audit = json.loads((current/'audit.json').read_text())
        end_frame = partial['completed_steps']
        if (current in runs or partial['completed'] or not audit['accepted_prefix_checks_passed']
                or not 0 < end_frame == audit['accepted_steps'] < partial['steps']):
            raise ValueError('Expected an audited, incomplete final level')
        runs.append(current)
        output = root / f'accepted_prefix_{end_frame}'
        output.mkdir(exist_ok=True)
    states, transfers, masses, methods = [], [], [], []
    direct_to_finest = []
    coarse_solvers, total_mass_kg = [], []
    for level, run in enumerate(runs):
        meta = json.loads((run / 'run.json').read_text())
        if level >= first_child_level and 'steps' in chain and meta['steps'] != chain['steps']:
            raise ValueError(f'Child horizon differs from the requested window at {run}')
        incomplete = accepted_prefix and level == len(runs)-1
        if meta['scene'] != 'bunny' or meta['level'] != level or not (meta['completed'] or incomplete):
            raise ValueError(f'Expected a completed bunny L{level} run at {run}')
        if level:
            if meta['dt'] != dt or (end_frame is None and len(states[0]) != meta['completed_steps'] + 1):
                raise ValueError(f'Frame times differ at {run}')
            with (runs[level-1]/'positions.npy').open('rb') as source:
                if hashlib.file_digest(source, 'sha256').hexdigest() != meta['lower_sha256']:
                    raise ValueError(f'Trajectory at {run} has a different parent')
        else:
            dt = meta['dt']
        assert meta.get('consistency_weight', 0) == 0
        if not incomplete:
            assert json.loads((run / 'continuation_audit.json').read_text())['passed']
        methods.append(meta['method'])
        assets = Path(meta['assets'])
        mesh = np.load(assets / f'mesh_{level}.npz')
        rest, faces = mesh['vertices'], mesh['faces']
        # Reuse the direct PSQ operator packed for this solve. PSQ §4.3.1
        # composes the linear U matrices and transports the offset only once.
        with np.load(run/'operator/operator.npz') as op:
            if not level:
                fine_areas = frames(op['fine_rest'], op['fine_faces'])[3]
                fine_mass = np.bincount(op['fine_faces'].ravel(), weights=np.repeat(fine_areas/3, 3))/fine_areas.sum()
            if level == 7:
                direct_to_finest.append(None)
            else:
                matrix = csr_matrix((op['U_data'],op['U_indices'],op['U_indptr']), shape=tuple(op['U_shape']))
                direct_to_finest.append(ShellProlongation(rest, op['fine_rest'], faces, matrix, op['anchors']))
        areas = frames(rest, faces)[3]
        masses.append(np.bincount(faces.ravel(), weights=np.repeat(areas / 3, 3)) / areas.sum())
        total_mass_kg.append(areas.sum()*meta['material']['thickness']*meta['material']['density'])
        if end_frame is not None and meta['completed_steps'] < end_frame:
            raise ValueError(f'Missing required parent frames at {run}')
        track = np.load(run / 'positions.npy', mmap_mode='r')
        states.append(track[:(end_frame if end_frame is not None else meta['completed_steps'])+1])
        if level:
            pair = f'{level-1}_{level}'
            transfers.append(ShellProlongation(prev_rest, rest, prev_faces, load_npz(assets/f'map_{pair}.npz'),
                                               np.load(assets/f'anchors_{pair}.npy')))
            U = transfers[-1].U
            coarse_solvers.append(splu((U.T @ U).tocsc()))
        prev_rest, prev_faces = rest, faces

    fields = ['frame', 'level', 'parent_rms_mm', 'coarse_rms_mm', 'centroid_from_L0_mm', 'floor_pct', 'max_z_m',
              'projected_parent_rms_mm', 'projected_parent_mass_squared_error_kg_m2', 'max_vertex_speed_m_s',
              'common_finest_L0_rms_mm', 'common_finest_centroid_shift_mm', 'common_finest_centered_rms_mm']
    rows = []
    for frame in range(len(states[0])):
        reference = states[0][frame]
        finest_reference = direct_to_finest[0].position(reference)
        centroid = masses[0] @ reference
        for level, (track, mass) in enumerate(zip(states, masses)):
            x = track[frame]
            speed = np.linalg.norm(x-track[frame-1],axis=1).max()/dt if frame else 0.
            parent_error = projected_rms = projected_mass_error = 0.
            if level:
                transfer = transfers[level-1]
                parent_error = rms(x - transfer.position(states[level-1][frame]), mass)
                # PD++ Eq. (13) projects with the unweighted left inverse of U,
                # then evaluates the residual using the coarse mass matrix.
                projected = coarse_solvers[level-1].solve(transfer.U.T @ x)
                projected_rms = rms(projected - states[level-1][frame], masses[level-1])
                projected_mass_error = total_mass_kg[level-1] * projected_rms**2
                reference = transfer.position(reference)
            finest_error = shift_norm = centered_error = 0.
            if level:
                direct = direct_to_finest[level]
                fine_x = x if direct is None else direct.position(x)
                displacement = fine_x - finest_reference
                finest_error = rms(displacement, fine_mass)
                shift = np.einsum('i,ij->j', fine_mass, displacement)
                shift_norm = np.linalg.norm(shift)
                centered_error = rms(displacement - shift, fine_mass)
            rows.append(dict(zip(fields, (frame, level, 1000*parent_error, 1000*rms(x-reference, mass),
                                          1000*np.linalg.norm(mass @ x-centroid), 100*mass[x[:,2]<.01].sum(), x[:,2].max(),
                                          1000*projected_rms, projected_mass_error, speed, 1000*finest_error,
                                          1000*shift_norm, 1000*centered_error))))
    with (output/'level_errors.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for level in range(len(states)):
        sample = [r for r in rows if r['level'] == level]
        peak = max(sample, key=lambda r: r['coarse_rms_mm'])
        direct_peak = max(sample, key=lambda r: r['common_finest_L0_rms_mm'])
        fastest = max(sample, key=lambda r: r['max_vertex_speed_m_s'])
        above_floor = [r['frame'] for r in sample if r['max_z_m'] >= .01]
        floor_frame = max(above_floor, default=-1) + 1
        floor95 = 1 + max((r['frame'] for r in sample if r['floor_pct'] < 95), default=-1)
        summary.append(dict(level=level, method=methods[level], final_parent_mm=sample[-1]['parent_rms_mm'],
                            max_parent_mm=max(r['parent_rms_mm'] for r in sample),
                            max_projected_parent_mm=max(r['projected_parent_rms_mm'] for r in sample),
                            max_projected_parent_mass_squared_error_kg_m2=max(r['projected_parent_mass_squared_error_kg_m2'] for r in sample),
                            final_L0_mm=sample[-1]['coarse_rms_mm'], max_L0_mm=peak['coarse_rms_mm'],
                            max_L0_frame=peak['frame'], final_floor_pct=sample[-1]['floor_pct'],
                            max_common_finest_L0_mm=direct_peak['common_finest_L0_rms_mm'],
                            max_common_finest_L0_frame=direct_peak['frame'],
                            centroid_shift_at_peak_mm=direct_peak['common_finest_centroid_shift_mm'],
                            centered_rms_at_peak_mm=direct_peak['common_finest_centered_rms_mm'],
                            final_common_finest_L0_mm=sample[-1]['common_finest_L0_rms_mm'],
                            max_vertex_speed_m_s=fastest['max_vertex_speed_m_s'], max_speed_frame=fastest['frame'],
                            mass95_within_1cm_until_end_s=floor95*dt if floor95<len(sample) else None,
                            all_within_1cm_until_end_s=floor_frame*dt if floor_frame<len(sample) else None))
    (output/'level_errors.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if render_movie:
        from compare_bunny_consistency import movie
        # Use the same camera, scale, frame indices and material checkerboard.
        panels = []
        for level, (method, track) in enumerate(zip(methods, states)):
            faces = np.load(assets / f'mesh_{level}.npz')['faces']
            label = 'L0  preview' if not level else f'L{level}  {method}  w=0'
            label += f"  mu={meta['contact_friction']['cloth_bunny']:g}"
            if accepted_prefix:
                label += '   prefix'
            panels.append((label, track, faces))
        bunny = np.load(assets / 'bunny.npz')
        movie(output, panels, bunny['vertices'][bunny['faces']], meta['dt'], measure_silhouettes=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--movie', action='store_true')
    parser.add_argument('--accepted-prefix', action='store_true',help='Measure only the overlapping accepted frames of an interrupted chain')
    args = parser.parse_args()
    measure(args.root, args.movie, args.accepted_prefix)
