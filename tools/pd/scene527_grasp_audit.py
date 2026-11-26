"""Score fixed material triangles against the Scene527 gripper TCP at 8 s."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from trimesh.triangles import closest_point, points_to_barycentric


SAMPLE_SECONDS = 8.0
MAX_DISTANCE_M = 0.10
TCP_LOCAL = np.array([0.155, 0.001786, 0.014])
HANDS = {'left': 'left_link16', 'right': 'right_link26'}


def measure_triangle(close_positions, sample_positions, faces, close_tcp, sample_tcp):
    triangles = close_positions[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    valid = np.flatnonzero(np.linalg.norm(np.cross(edge_a, edge_b), axis=1) > 1e-14)
    if not len(valid):
        raise ValueError('No nondegenerate cloth triangle at grasp binding frame')
    candidates = triangles[valid]
    points = closest_point(candidates, np.broadcast_to(close_tcp, (len(candidates), 3)))
    nearest = np.argmin(np.sum((points - close_tcp)**2, axis=1))
    face = int(valid[nearest])
    weights = points_to_barycentric(triangles[face:face+1], points[nearest:nearest+1])[0]
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    sample_point = weights @ sample_positions[faces[face]]
    distance = float(np.linalg.norm(sample_point - sample_tcp))
    # Keep the inclusive boundary when reconstruction rounds upward by a few ulps.
    passes = distance <= MAX_DISTANCE_M or np.isclose(distance, MAX_DISTANCE_M, rtol=0, atol=1e-12)
    return {
        'verdict': 'PASS' if passes else 'FAIL',
        'triangle': face,
        'vertices': faces[face].tolist(),
        'barycentric_weights': weights.tolist(),
        'bind_distance_m': float(np.linalg.norm(points[nearest] - close_tcp)),
        'distance_m': distance,
        'point_world_m': sample_point.tolist(),
        'tcp_world_m': sample_tcp.tolist(),
    }


def evaluate(frames, positions, faces, link_names, transforms, openness, fps):
    expected_frame = SAMPLE_SECONDS * fps
    sample_index = np.flatnonzero(frames == round(expected_frame))
    result = {'time_s': SAMPLE_SECONDS, 'source_frame': round(expected_frame),
              'threshold_m': MAX_DISTANCE_M, 'hands': {}}
    if not np.isclose(expected_frame, round(expected_frame), rtol=0, atol=1e-8) or len(sample_index) != 1:
        result['hands'] = {hand: {'verdict': 'INCONCLUSIVE', 'reason': 'No recorded frame at 8 s'} for hand in HANDS}
        result['verdict'] = 'INCONCLUSIVE'
        return result

    sample_index = int(sample_index[0])
    for index, (hand, link) in enumerate(HANDS.items()):
        closed = np.flatnonzero((openness[frames, index] <= 0.5) & (frames <= expected_frame))
        if not len(closed):
            result['hands'][hand] = {'verdict': 'INCONCLUSIVE', 'reason': 'No closing frame by 8 s'}
            continue
        bind_index = int(closed[0])
        link_index = link_names.index(link)
        poses = transforms[[bind_index, sample_index], link_index]
        tcp = poses[:, :3, :3] @ TCP_LOCAL + poses[:, :3, 3]
        score = measure_triangle(positions[bind_index].astype(np.float64),
                                 positions[sample_index].astype(np.float64), faces, *tcp)
        score['bind_source_frame'] = int(frames[bind_index])
        score['sample_openness'] = float(openness[frames[sample_index], index])
        result['hands'][hand] = score
    outcomes = [score['verdict'] for score in result['hands'].values()]
    result['verdict'] = 'FAIL' if 'FAIL' in outcomes else ('PASS' if all(v == 'PASS' for v in outcomes) else 'INCONCLUSIVE')
    return result


def audit_case(output: Path, case: Path):
    replay = case/'replay.npz'
    if not replay.is_file():
        return None
    command = json.loads((case/'command.json').read_text())
    trajectory = Path(command[command.index('--trajectory')+1])
    fps = float(command[command.index('--action-fps')+1])
    with np.load(trajectory, allow_pickle=False) as control:
        openness = control['openness']
    with np.load(output/'hierarchy'/f'mesh_{case.name}.npz', allow_pickle=False) as mesh:
        faces = mesh['triangles']
    with np.load(replay, allow_pickle=False) as saved:
        result = evaluate(saved['source_frames'], saved['cloth_pos'], faces,
                          saved['ipc_link_names'].tolist(), saved['ipc_link_transforms'], openness, fps)
    with NamedTemporaryFile(mode='w', dir=case, prefix='.grasp_8s_', delete=False) as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(case/'grasp_8s.json')
    return result
