"""Run and audit all remaining bunny levels L5-L7, in dependency order."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from report_bunny_pcg_no_ground_l1 import read_audited, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/bunny_pcg_no_ground_20261124'
ASSETS = ROOT / 'output/bunny_paper_assets'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    jobs = []
    for level in range(5, 8):
        for asset in (f'mesh_{level}.npz', f'map_{level-1}_{level}.npz',
                      f'anchors_{level-1}_{level}.npy'):
            if not (ASSETS / asset).is_file():
                raise FileNotFoundError(ASSETS / asset)
        for mu in ('mu0', 'mu03'):
            for method in ('semi', 'velpro'):
                name = f'{mu}_{method}'
                parent = BASE / f'L{level-1}' / name
                run = BASE / f'L{level}' / name
                if level == 5:
                    meta = json.loads((parent / 'run.json').read_text())
                    audit = json.loads((parent / 'no_ground_audit.json').read_text())
                    if not (meta['completed'] and meta['completed_steps'] == 100
                            and meta['level'] == 4 and meta['method'] == method
                            and audit['passed'] and audit['saved_states_checked'] == 101
                            and not audit['intersecting_states']
                            and audit['positions_sha256'] == digest(parent / 'positions.npy')):
                        raise RuntimeError(f'Invalid starting parent: {parent}')
                completed = False
                if run.exists():
                    meta = json.loads((run / 'run.json').read_text())
                    completed = (meta['completed'] and meta['status'] == 'completed'
                                 and meta['completed_steps'] == 100 and meta['dt'] == .01
                                 and meta['level'] == level and meta['method'] == method
                                 and Path(meta['lower_run']).resolve() == parent.resolve())
                    if not completed:
                        raise RuntimeError(f'Existing incomplete run; will not overwrite: {run}')
                elif run.with_suffix('.log').exists():
                    raise RuntimeError(f'Existing log; will not overwrite: {run.with_suffix(".log")}')
                jobs.append((level, name, method, parent, run, completed))
                print(f'L{level} {name}: {"audit existing simulation" if completed else "simulate and audit"}', flush=True)
    if args.dry_run:
        print('Preflight passed: 12 jobs. No simulations started.', flush=True)
        return

    for level, name, method, parent, run, completed in jobs:
        if not completed:
            run.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(ROOT / 'scripts/run_bunny_ground_fine.py'),
                       '--scene', 'bunny', '--assets', str(ASSETS), '--output', str(run),
                       '--level', str(level), '--lower', str(parent), '--steps', '100',
                       '--dt', '.01', '--method', method, '--frozen-offset',
                       '--refined-probe', '--adaptive-search', '--max-newton', '1500',
                       '--preconditioner', 'diag', '--newton-velocity-tol', '.001']
            if level == 7:
                command.append('--solver-trace')
            print(f'Starting L{level} {name}', flush=True)
            with run.with_suffix('.log').open('xb') as log:
                with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT) as process:
                    for line in process.stdout:
                        log.write(line)
                        if line.startswith(b'ACCEPTED '):
                            log.flush()
                            print(f'L{level} {name}: {line.decode("utf-8", errors="replace").strip()}', flush=True)
                    code = process.wait()
                if code:
                    raise RuntimeError(f'Simulation failed ({code}); see {run.with_suffix(".log")}')
        for script, extra in (
            ('audit_paper_run.py', []),
            ('audit_paper_non_descent.py', []),
            ('audit_flag_fine_search.py', ['--check-converged-energy']),
        ):
            audit_log = run / f'{script}.log'
            print(f'Auditing L{level} {name}: {script}', flush=True)
            with audit_log.open('wb') as log:
                result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), str(run), *extra],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Audit failed; see {audit_log}')
        # Generates the saved-state audit required to start the next level.
        print(f'Checking all 101 saved states: L{level} {name}', flush=True)
        read_audited(run)
        print(f'Completed and audited: L{level} {name}', flush=True)
    print('All L5-L7 simulations and audits completed.', flush=True)


if __name__ == '__main__':
    main()
