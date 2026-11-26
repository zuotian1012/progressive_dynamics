"""Short frozen/exact-gradient GPU PCG control, L0-L3 VelPro, 30 steps."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import ipctk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/bunny_preview_pilot_20261125'
ASSETS = ROOT / 'output/bunny_paper_assets'
RUNTIME = ROOT / 'output/flag_L3_fix_20261118/bunny_ordered_pcg_runtime/src'


def audit(run, expected_steps=30):
    for script, extra in [('audit_paper_run.py', []), ('audit_paper_non_descent.py', []),
                          ('audit_flag_fine_search.py', ['--check-converged-energy'])]:
        with (run / (script + '.log')).open('wb') as log:
            subprocess.run([sys.executable, str(ROOT / 'scripts' / script), str(run), *extra],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    meta = json.loads((run / 'run.json').read_text())
    assert meta['completed'] and meta['completed_steps'] == expected_steps and not meta['ground']['enabled']
    mesh = np.load(ASSETS / f'mesh_{meta["level"]}.npz')
    bunny = np.load(ASSETS / 'bunny.npz')
    x = np.load(run / 'positions.npy')
    assert x.shape == (expected_steps + 1, len(mesh['vertices']), 3) and np.isfinite(x).all()
    rest = np.vstack([mesh['vertices'], bunny['vertices']])
    faces = np.vstack([mesh['faces'], bunny['faces'] + len(mesh['vertices'])]).astype(np.int32)
    edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0).astype(np.int32)
    cm = ipctk.CollisionMesh(rest, edges, faces)
    intersections = [i for i, state in enumerate(x)
                     if ipctk.has_intersections(cm, np.vstack([state, bunny['vertices']]))]
    assert not intersections, (run, intersections)
    rows = json.loads((run / 'steps.json').read_text())
    assert max(r['obstacle_error'] for r in rows) == 0
    result = dict(passed=True, no_ground=True, saved_states_checked=len(x),
                  intersecting_states=intersections,
                  positions_sha256=hashlib.sha256((run / 'positions.npy').read_bytes()).hexdigest())
    (run / 'no_ground_audit.json').write_text(json.dumps(result, indent=2))


def main():
    OUT.mkdir(exist_ok=True)
    for level in range(4):
        for mode in ('frozen', 'exact'):
            run = OUT / (mode if level == 0 else f'L{level}_' + mode)
            command = [sys.executable, '-u', str(ROOT / 'scripts' / (
                'run_bunny_ground_l0.py' if level == 0 else 'run_bunny_ground_fine.py')),
                '--scene', 'bunny', '--assets', str(ASSETS), '--output', str(run),
                '--steps', '30', '--frozen-offset' if mode == 'frozen' else '--exact-offset',
                '--refined-probe', '--adaptive-search', '--max-newton', '1500',
                '--newton-velocity-tol', '.001', '--preconditioner', 'diag',
                '--native-package', str(RUNTIME)]
            if level == 0:
                command += ['--no-ground', '--friction', '0', '--contact-distance', '.0007',
                            '--linear-tol-rate', '1e-10']
            else:
                parent = OUT / (mode if level == 1 else f'L{level-1}_' + mode)
                command += ['--level', str(level), '--lower', str(parent), '--method', 'velpro']
            if run.exists():
                meta = json.loads((run / 'run.json').read_text())
                if not meta['completed'] or meta['completed_steps'] != 30:
                    raise RuntimeError(f'Existing incomplete pilot: {run}')
            else:
                (OUT / (run.name + '_command.json')).write_text(json.dumps(command, indent=2))
                print('Starting', run.name, flush=True)
                with run.with_suffix('.log').open('xb') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'Run failed: {run}; see adjacent .log')
            audit(run)
            print('Completed and audited', run.name, flush=True)


if __name__ == '__main__':
    main()
