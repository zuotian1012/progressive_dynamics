"""Compare matching same-topology Scene527 IPC histories at identical substeps."""
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--row', type=int, default=0)
    args = parser.parse_args()
    with np.load(args.reference) as ref, np.load(args.candidate) as test:
        for key in ('dt', 'substeps', 'settle_steps'):
            np.testing.assert_equal(ref[key], test[key])
        for key, unit in [('x','m'),('v','m/s')]:
            a, b = ref[key], test[key]
            if a.shape != b.shape or not 0 <= args.row < len(a):
                raise ValueError(f'Incompatible {key} histories or invalid row')
            peak = 0.
            for i in range(0,len(a),16):
                distances = np.linalg.norm(a[i:i+16]-b[i:i+16],axis=-1)
                if not np.isfinite(distances).all():
                    raise ValueError(f'Nonfinite {key} near row {i}')
                peak = max(peak,float(distances.max()))
            for row in sorted({args.row, len(a)-1}):
                d = np.linalg.norm(a[row]-b[row],axis=-1)
                print(f'{key} row={row}: RMS={np.sqrt(np.mean(d*d)):.9g} max={d.max():.9g} {unit}')
            print(f'{key} whole-history maximum: {peak:.9g} {unit}')

if __name__ == '__main__':
    main()
