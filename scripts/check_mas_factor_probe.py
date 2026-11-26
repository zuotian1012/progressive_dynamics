"""Verify captured GPU MAS factors and the actual native preconditioner action."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.linalg import solve_triangular
from scipy.sparse import coo_matrix


def matrix(path):
    raw = mmread(path).tocoo()
    off = raw.row // 3 != raw.col // 3
    return (raw + coo_matrix((raw.data[off], (raw.col[off], raw.row[off])), shape=raw.shape)).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe', type=Path)
    args = parser.parse_args()
    root = args.probe
    meta = json.loads(next(root.rglob('mas_cluster_meta.f1.n0.json')).read_text())
    count = meta['num_cluster_blocks']
    factors = np.fromfile(next(root.rglob('mas_factors.f1.n0.bin')), dtype=np.float64).reshape(count, -1)
    assert factors.shape[1] == meta['factor_doubles_per_cluster'] == 2353
    L = factors[:, :2304].reshape(-1, 48, 48)
    scale = factors[:, 2304:2352]
    shifts = factors[:, -1]
    assert np.isfinite(factors).all() and (scale > 0).all()
    assert (np.diagonal(L, axis1=1, axis2=2) > 0).all()
    H = matrix(next(root.rglob('mas_cluster_hess.f1.n0.mtx')))
    errors = []
    for k in range(count):
        h = H[k*48:(k+1)*48, k*48:(k+1)*48].toarray()
        normalized = (h + h.T) * .5 / scale[k, :, None] / scale[k, None, :]
        empty = np.max(np.abs(h), axis=1) == 0
        normalized[np.diag_indices(48)] += shifts[k] + empty.astype(float)
        errors.append(np.linalg.norm(L[k] @ L[k].T - normalized, np.inf) / max(np.linalg.norm(normalized, np.inf), 1e-300))
    r2p = meta['real_to_part']
    going = meta['going_next']
    rows, cols = [], []
    for real, part in enumerate(r2p):
        if part < 0:
            continue
        nodes = [part]
        current = real
        for _ in range(meta['level_num'] - 1):
            current = going[current]
            nodes.append(current)
        for node in nodes:
            for d in range(3):
                rows.append(3*node+d)
                cols.append(3*real+d)
    R = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(count*48, len(r2p)*3)).tocsr()
    def vector(name):
        return np.asarray(mmread(next(root.rglob(name)))).ravel()
    r, z = vector('r.1.0.0.mtx'), vector('z.1.0.0.mtx')
    rf = r[108:]
    # Actual native residual plus independent random vectors test linear symmetry.
    random = np.random.default_rng(128).normal(size=(len(rf), 3))
    inputs = np.column_stack((rf, random))
    local = (R @ inputs).reshape(count, 48, -1)
    solved = np.empty_like(local)
    for k in range(count):
        y = solve_triangular(L[k], local[k] / scale[k, :, None], lower=True)
        solved[k] = solve_triangular(L[k].T, y, lower=False) / scale[k, :, None]
    result = R.T @ solved.reshape(count*48, -1)
    quadratic = np.sum(inputs * result, axis=0)
    native_error = np.linalg.norm(result[:, 0] - z[108:]) / np.linalg.norm(result[:, 0])
    gram = inputs.T @ result
    symmetry_error = np.linalg.norm(gram-gram.T) / np.linalg.norm(gram)
    passed = max(errors) < 1e-12 and native_error < 1e-7 and (quadratic > 0).all() and symmetry_error < 1e-10
    report = dict(passed=bool(passed), active_clusters=count, shifted_clusters=int(np.sum(shifts>0)),
                  maximum_shift=float(shifts.max()), maximum_factor_backward_error=float(max(errors)),
                  native_fem_action_relative_error=float(native_error), native_total_rz=float(r@z),
                  native_abd_rz=float(r[:108]@z[:108]), native_fem_rz=float(rf@z[108:]),
                  independent_quadratic_forms=quadratic.tolist(), independent_symmetry_error=float(symmetry_error),
                  limitation='Checks initial preconditioning only, not Newton or full-fold convergence')
    (root/'factor_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert passed, 'MAS factor or restriction/prolongation verification failed'


if __name__ == '__main__':
    main()
