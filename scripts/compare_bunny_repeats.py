"""Measure variation between two runs of the same bunny problem on the same mesh."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def search_trials(run):
    simplex = plane = None
    with run.with_suffix('.log').open(encoding='utf-8-sig', errors='replace') as log:
        for line in log:
            if 'SimplexTrajectoryFilter PTs:' in line:
                simplex = list(map(int, re.findall(r'(?:PTs|EEs|PEs|PPs): (\d+)', line)))
            elif 'VertexHalfPlaneTrajectoryFilter PHs:' in line:
                plane = int(re.search(r'PHs: (\d+)', line)[1])
            elif 'FLAG_SEARCH_TRIAL ' in line:
                values = dict(re.findall(r'(\w+)=(\S+)', line))
                key = tuple(int(values[k]) for k in ('frame', 'newton', 'search'))
                yield key, dict(alpha=float(values['alpha']), energy_change=float(values['E'])-float(values['E0']),
                                simplex_counts_PT_EE_PE_PP=simplex, plane_count=plane)


def compare(first, repeat):
    metadata = [json.loads((path / 'run.json').read_text()) for path in (first, repeat)]
    reference, other = metadata
    if reference['scene'] != 'bunny':
        raise ValueError('Expected a bunny run')
    if reference.get('requested_contact_stiffness',1e9) != other.get('requested_contact_stiffness',1e9):
        raise ValueError('Different normal contact stiffness in the two runs')
    for key in ('scene', 'level', 'method', 'steps', 'dt', 'config', 'ground', 'material',
                'contact_friction', 'assets', 'bundle_sha256', 'library_sha256',
                'lower_sha256', 'velocity_map', 'consistency_weight', 'exact_offset_gradient',
                'adaptive_line_search', 'initial_non_descent_rescue', 'non_descent_probe_refinement',
                'replay_sha256', 'velocity_only_preserves_friction', 'diagnostic_restart'):
        if reference.get(key) != other.get(key):
            raise ValueError(f'Different settings for {key}')
    for meta, path in zip(metadata, (first, repeat)):
        audit = json.loads((path / 'audit.json').read_text())
        if not audit['accepted_prefix_checks_passed'] or audit['accepted_steps'] != meta['completed_steps']:
            raise ValueError(f'Unverified accepted trajectory at {path}')
    with np.load(Path(reference['assets']) / f'mesh_{reference["level"]}.npz') as mesh:
        faces = mesh['faces']
        tri = mesh['vertices'][faces]
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) / 2
    weights = np.bincount(faces.ravel(), weights=np.repeat(areas/3, 3)) / areas.sum()
    tracks = [np.load(path / 'positions.npy', mmap_mode='r') for path in (first, repeat)]
    if not np.array_equal(tracks[0][0], tracks[1][0]):
        raise ValueError('Initial vertex positions differ')
    end = min(meta['completed_steps'] for meta in metadata)
    rms, centroids, shape, worst_vertex = [], [], [], []
    for i in range(end+1):
        delta = 1000 * (tracks[0][i] - tracks[1][i])
        center = weights @ delta
        rms.append(float(np.sqrt(np.einsum('vi,vi,v->', delta, delta, weights))))
        worst_vertex.append(float(np.linalg.norm(delta, axis=1).max()))
        residual = delta - center
        shape.append(float(np.sqrt(np.einsum('vi,vi,v->', residual, residual, weights))))
        centroids.append(center.tolist())
    peak = int(np.argmax(rms))
    position_hashes = []
    for path in (first, repeat):
        with (path / 'positions.npy').open('rb') as stream:
            position_hashes.append(hashlib.file_digest(stream, 'sha256').hexdigest())
    step_logs = [json.loads((path / 'steps.json').read_text()) for path in (first, repeat)]
    first_count_change = None
    for frame in range(1, end+1):
        counts = [log[frame-1]['native']['newton_iterations'] for log in step_logs]
        if counts[0] != counts[1]:
            first_count_change = dict(frame=frame, iterations=counts)
            break
    first_decision = None
    if all(path.with_suffix('.log').is_file() for path in (first, repeat)):
        for (ka, ta), (kb, tb) in zip(search_trials(first), search_trials(repeat)):
            if ka != kb or ka[0] > end:
                break
            if (ta['energy_change'] <= 0) != (tb['energy_change'] <= 0):
                first_decision = dict(frame=ka[0], newton=ka[1], search=ka[2], first=ta, repeat=tb)
                break
    return dict(first=str(first.resolve()), repeat=str(repeat.resolve()), frames=end+1,
                positions_file_sha256=position_hashes,
                completed=[meta['completed'] for meta in metadata], unit='mm',
                first_newton_count_difference=first_count_change, first_trial_energy_decision_difference=first_decision,
                peak_mass_rms=rms[peak], peak_time_s=peak*reference['dt'], final_mass_rms=rms[-1],
                peak_vertex_mm=max(worst_vertex), final_vertex_mm=worst_vertex[-1],
                values_mm=rms, relative_centroid_mm=centroids, rms_after_removing_translation_mm=shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('repeat', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.first, args.repeat)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"{result['frames']} common states, peak RMS {result['peak_mass_rms']:.6g} mm "
          f"at {result['peak_time_s']:.2f} s")
