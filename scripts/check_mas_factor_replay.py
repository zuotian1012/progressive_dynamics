"""GPU replay tests of production MAS factor/apply kernels on real and synthetic blocks."""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/shirt_L2_mas_fix_20261119'))
    root = parser.parse_args().root
    dll = ctypes.CDLL(str((root/'mas_factor_replay.dll').resolve()))
    ptr = ctypes.POINTER(ctypes.c_double)
    dll.evaluate.argtypes = [ctypes.c_int, ptr, ptr, ptr, ptr]
    dll.evaluate.restype = ctypes.c_int
    rng = np.random.default_rng(20261119)
    captured = np.load(root/'captured_clusters_2936.npz')['H']
    spd = rng.normal(size=(48, 48))
    spd = spd@spd.T + np.eye(48)
    indefinite = np.eye(48)
    indefinite[0, 1] = indefinite[1, 0] = 2
    negative = np.eye(48)
    negative[0, 0] = -1
    synthetic = np.stack([np.eye(48), np.zeros((48,48)), np.ones((48,48)),
                          indefinite, negative, spd, spd*1e-200, spd*1e200])
    H = np.ascontiguousarray(np.concatenate((captured, synthetic)))
    r = np.ascontiguousarray(rng.normal(size=(len(H),48)))
    out = np.zeros((len(H),2353))
    z = np.zeros_like(r)
    assert dll.evaluate(len(H), *(a.ctypes.data_as(ptr) for a in (H,r,out,z))) == 0
    assert np.isfinite(out).all() and np.isfinite(z).all()
    relative_error, backward = [], []
    for k, h in enumerate(H):
        L = out[k,:2304].reshape(48,48)
        s = out[k,2304:2352]
        shift = out[k,-1]
        assert np.all(np.diag(L)>0) and np.all(s>0)
        B = (h+h.T)*.5/s[:,None]/s[None,:]
        B[np.diag_indices(48)] += shift + (np.max(np.abs(h),axis=1)==0)
        backward.append(np.linalg.norm(L@L.T-B,np.inf)/np.linalg.norm(B,np.inf))
        ref = solve_triangular(L.T,solve_triangular(L,r[k]/s,lower=True),lower=False)/s
        # Scale first to keep tiny/huge synthetic cases from overflowing norms.
        denominator = np.max(np.abs(ref))
        relative_error.append(np.linalg.norm((z[k]-ref)/denominator)/np.linalg.norm(ref/denominator))
    q = np.sum(r*z,axis=1)
    assert (q>0).all() and max(backward)<1e-12 and max(relative_error)<1e-6
    np.testing.assert_allclose(z[-8],r[-8],rtol=1e-14)
    np.testing.assert_allclose(z[-7],r[-7],rtol=1e-14)
    assert out[-6,-1]>0 and out[-5,-1]>0 and out[-4,-1]>0
    report = dict(passed=True, captured_blocks=len(captured), synthetic_cases=len(synthetic),
                  maximum_factor_backward_error=float(max(backward)),
                  maximum_gpu_solve_relative_error=float(max(relative_error)),
                  negative_quadratic_forms=int(np.sum(q<=0)), synthetic_shifts=out[-8:,-1].tolist())
    (root/'replay_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
