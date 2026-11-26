"""Compare saved settling states; mesh and lower trajectory both differ between L1 runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from progressive.prolongation import frames
from shirt_demo.report import load_run, draw, plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-l1', type=Path, required=True)
    parser.add_argument('--new-l0', type=Path, required=True)
    parser.add_argument('--new-l1', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.old_l1, args.new_l0, args.new_l1]
    labels = ['Previous L1', 'Rebuilt L0', 'Rebuilt L1']
    data = [load_run(path) for path in paths]
    results = []
    for label, path, item in zip(labels, paths, data):
        assert item['meta']['completed_steps'] >= 60
        x = item['x'][:61].copy()
        inv = frames(x[0], item['f'])[2]
        stretch, minimum = [], []
        for state in x:
            tri = state[item['f']]
            deformation = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=2) @ inv
            sigma = np.linalg.svd(deformation, compute_uv=False)
            stretch.append(float(sigma[:, 0].max()))
            minimum.append(float(sigma[:, 1].min()))
        rows = json.loads((path/'steps.json').read_text())[:60]
        results.append(dict(label=label, run=str(path.resolve()),
            vertices=len(x[0]), faces=len(item['f']),
            first_step_maximum_stretch=stretch[1],
            settling_maximum_stretch=max(stretch), settling_end_maximum_stretch=stretch[-1],
            settling_minimum_singular_value=min(minimum),
            settling_height_percentiles_mm=(np.percentile(x[-1, :, 2]-.8, [0, 50, 90, 100])*1000).tolist(),
            native_converged_steps=sum(r['native']['converged'] for r in rows),
            preview_steps=sum(r['native']['preview_non_descent'] for r in rows),
            newton_limit_steps=sum(r['native']['hit_newton_limit'] for r in rows)))
    result = dict(runs=results, scope='First 60 physical steps; saved current-level cloth, no interpolation.',
        limitations=['Old/new L1 differ in both hierarchy and lower trajectory; not a single-variable causal test.',
                     'Stretch and visual comparison do not certify artifact-free cloth or absence of intersections.'])
    (args.output/'settling_comparison.json').write_text(json.dumps(result, indent=2))
    fig = plt.figure(figsize=(15, 12))
    for row, step in enumerate((0, 1, 60)):
        for col, (label, item) in enumerate(zip(labels, data)):
            ax = fig.add_subplot(3, 3, row*3+col+1, projection='3d')
            draw(ax, item, step, f'{label} | step {step} | {step/120:.3f} s', False)
    fig.suptitle('Settling comparison | actual cloth states, same camera, no mesh edges')
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(args.output/'settling_comparison.png', dpi=130)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
