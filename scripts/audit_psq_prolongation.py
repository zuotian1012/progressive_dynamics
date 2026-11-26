"""Small, read-only PSQ Eqs. (4)-(9) audit; no time stepping or Newton solve.

The independent reference uses the Gram-matrix polar factor, not the
production SVD/frames helpers. Output is an audit, never a simulation result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import csr_matrix, load_npz
from scipy.spatial.transform import Rotation
from progressive.prolongation import ShellProlongation, Hierarchy
from progressive.runtime import numerical_threads


def reference(p, x):
    rest = p.coarse_rest[p.faces]
    a, b = rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]
    e = a / np.linalg.norm(a, axis=1)[:, None]
    n = np.cross(a, b)
    n /= np.linalg.norm(n, axis=1)[:, None]
    basis = np.stack((e, np.cross(n, e)), axis=2)
    rest_edges = np.stack((a, b), axis=2)
    B = basis.transpose(0, 2, 1) @ rest_edges
    tri = x[p.faces]
    edges = np.stack((tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=2)
    F = np.linalg.solve(B.transpose(0, 2, 1), edges.transpose(0, 2, 1)).transpose(0, 2, 1)
    values, vectors = np.linalg.eigh(F.transpose(0, 2, 1) @ F)
    assert np.min(values) > 0
    rotation = F @ ((vectors * (1 / np.sqrt(values))[:, None, :]) @ vectors.transpose(0, 2, 1))
    normal = np.cross(edges[:, :, 0], edges[:, :, 1])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    delta = p.fine_rest - p.U @ p.coarse_rest
    ids = p.anchors
    gamma = np.sum(n[ids] * delta, axis=1)
    tangent = np.einsum('nji,nj->ni', basis[ids], delta)
    return p.U @ x + normal[ids] * gamma[:, None] + np.einsum('nij,nj->ni', rotation[ids], tangent)


def error(a, b):
    result = float(np.max(np.abs(a - b), initial=0))
    assert np.isfinite(result) and result < 2e-12, result
    return result


def deform(x):
    y = x.copy()
    y[:, 0] += .15 * x[:, 1]
    y[:, 2] += .09 * np.sin(4 * x[:, 0]) + .04 * np.cos(3 * x[:, 1])
    return y


def audit(assets, diagnostic_state=None):
    # Two triangles bent about their common edge: both normal and tangent
    # offsets are nonzero. Shear additionally distinguishes polar rotation
    # from merely normalizing an edge or applying the entire deformation F.
    rest = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., -1, 0]])
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    U = csr_matrix([[.2, .3, .5, 0], [.4, .2, 0, .4]])
    fine = U @ rest + [[.03, -.02, .1], [-.01, .04, -.08]]
    p = ShellProlongation(rest, fine, faces, U, [0, 1])
    bent = rest.copy()
    bent[3] = [0, -np.cos(.8), np.sin(.8)]
    shear = bent.copy()
    shear[:, 0] += .4 * shear[:, 1]
    small = {'hinge_bend_error': error(p.position(bent), reference(p, bent)),
             'sheared_hinge_error': error(p.position(shear), reference(p, shear)),
             'stretch_offset_error': error(p.offset(rest * [1.5, 1.2, 1]), fine - U @ rest)}
    rests, topologies, maps, anchors, direct = [], [], [], [], []
    count = len(list(assets.glob('mesh_*.npz')))
    for k in range(count):
        with np.load(assets / f'mesh_{k}.npz') as data:
            rests.append(data['vertices']); topologies.append(data['faces'])
        if k < count - 1:
            maps.append(load_npz(assets / f'map_{k}_{k+1}.npz'))
            anchors.append(np.load(assets / f'anchors_{k}_{k+1}.npy'))
            direct.append(np.load(assets / f'anchors_{k}_{count-1}.npy'))
        else:
            direct.append(None)
    hierarchy = Hierarchy(rests, topologies, maps, anchors, direct)
    Q = Rotation.from_rotvec([.4, -.2, .8]).as_matrix()
    rows = []
    for kind, operators in [('adjacent', hierarchy.adjacent), ('direct', hierarchy.to_finest)]:
        for k, p in enumerate(operators):
            row_error = float(np.max(np.abs(np.asarray(p.U.sum(axis=1)).ravel() - 1)))
            assert row_error < 1e-12
            if kind == 'adjacent':
                row_ids = np.repeat(np.arange(p.U.shape[0]), np.diff(p.U.indptr))
                assert np.all(np.any(p.U.indices[:, None] == p.faces[p.anchors[row_ids]], axis=1))
            rows.append(dict(kind=kind, level=k, vertices=len(p.coarse_rest),
                row_sum_error=row_error,
                rest_error=error(p.position(p.coarse_rest), p.fine_rest),
                rigid_error=error(p.position(p.coarse_rest @ Q.T + [2, 3, 4]), p.fine_rest @ Q.T + [2, 3, 4]),
                deformed_reference_error=error(p.position(deform(p.coarse_rest)), reference(p, deform(p.coarse_rest))),
                nonzero_offset=not p.zero_offset))
    x = deform(rests[0])
    linear, sequential = x.copy(), x.copy()
    for p in hierarchy.adjacent:
        linear = p.U @ linear
        sequential = p.position(sequential)
    composition_error = error(hierarchy.to_finest[0].U @ x, linear)
    gap = float(np.max(np.abs(hierarchy.to_finest[0].position(x) - sequential)))
    assert gap > 1e-8, 'Fixture does not exercise direct versus sequential nonlinear prolongation'
    diagnostic_error = None
    if diagnostic_state is not None:
        state = np.load(diagnostic_state)
        p = hierarchy.to_finest[0]
        assert state.shape == p.coarse_rest.shape
        diagnostic_error = error(p.position(state), reference(p, state))
    return dict(scope='PSQ prolongation only; no simulation or convergence certification',
                synthetic=small, operators=rows, linear_composition_error=composition_error,
                diagnostic_iterate_reference_error=diagnostic_error,
                direct_vs_sequential_nonlinear_difference_m=gap, passed=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path('output/bunny_paper_assets'))
    parser.add_argument('--output', type=Path, default=Path('output/psq_prolongation_audit.json'))
    parser.add_argument('--diagnostic-state', type=Path, help='Optional L0 diagnostic iterate, not a trajectory frame')
    args = parser.parse_args()
    start = time.perf_counter()
    with numerical_threads(1):
        result = audit(args.assets, args.diagnostic_state)
    result['seconds'] = time.perf_counter() - start
    paths = [Path(__file__), Path('progressive/prolongation.py'), *sorted(args.assets.glob('mesh_*.npz')),
             *sorted(args.assets.glob('map_*.npz')), *sorted(args.assets.glob('anchors_*.npy'))]
    if args.diagnostic_state is not None:
        paths.append(args.diagnostic_state)
    result['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'sha256'}, indent=2))
