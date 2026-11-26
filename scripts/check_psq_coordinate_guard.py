"""Compare GPU coordinate feasibility against the archived linear-path cap.

Uses real saved triangle positions with reproducible synthetic directions.
This is a numerical regression, not an exact replay of a failed Newton step.
"""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np


def steps(dll, rows):
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    out = np.empty((len(rows), 8), dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    fn = dll.evaluate_steps
    fn.argtypes = [ctypes.c_int, pointer, pointer]
    fn.restype = ctypes.c_int
    assert fn(len(rows), rows.ctypes.data_as(pointer), out.ctypes.data_as(pointer)) == 0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    source = Path('output/shirt_four_level_full_fold_20261118/L2/run')
    positions = np.load(source / 'positions.npy', mmap_mode='r')
    faces = np.load(source / 'faces.npy')
    ids = np.array([43, 446, 6009, 7440, 7564, 7600, 8085, 9795,
                    9796, 9918, 11943, 12475, 13141, 15146, 15971, 16451])
    rest = positions[0][faces[ids]].reshape(-1, 9)
    current = positions[2423][faces[ids]].reshape(-1, 9)
    base = np.tile(np.concatenate([rest, current], axis=1), (3000, 1))
    rng = np.random.default_rng(20261118)
    scale = 10 ** rng.uniform(-6, -2, (len(base), 1))
    rows = np.concatenate([base, rng.normal(size=(len(base), 9)) * scale], axis=1)
    records = []
    for variant, filename in [('baseline', 'baseline_check.dll'),
                              ('coordinate_guard', 'coordinate_check.dll')]:
        out = steps(ctypes.CDLL(str((args.root / 'probe' / filename).resolve())), rows)
        finite = np.isfinite(out[:, 1])
        positive = out[:, 0] > 0
        record = dict(variant=variant, cases=len(rows), initial_finite=int(finite.sum()),
                      positive_cap=int(positive.sum()),
                      positive_cap_from_invalid_initial=int((positive & ~finite).sum()),
                      finite_to_inf=int((finite & positive & ~np.isfinite(out[:, 2])).sum()),
                      finite_but_zero_cap=int((finite & ~positive).sum()),
                      zero_movement_positive_caps=int((positive & (out[:, 7] == 0)).sum()))
        records.append(record)
        print(record, flush=True)
    assert records[0]['finite_to_inf'] > 0, 'Regression must reproduce the baseline defect'
    assert records[1]['finite_to_inf'] == 0
    assert records[1]['positive_cap_from_invalid_initial'] == 0
    assert records[1]['finite_but_zero_cap'] == 0
    (args.root / 'coordinate_guard_replay.json').write_text(json.dumps(dict(
        variants=records, same_seed=20261118,
        scope='Local strain feasibility, actual saved positions with synthetic directions. Not a full simulation.'), indent=2))


if __name__ == '__main__':
    main()
