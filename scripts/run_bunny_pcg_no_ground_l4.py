"""Resume the four L4 bunny runs, preserving completed simulations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/bunny_pcg_no_ground_20261124'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    # Fail before starting expensive simulations if the audit dependency is absent.
    import ipctk  # noqa: F401

    jobs = []
    for mu in ('mu0', 'mu03'):
        for method in ('semi', 'velpro'):
            name = f'{mu}_{method}'
            parent = BASE / 'L3' / name
            run = BASE / 'L4' / name
            assert (parent / 'no_ground_audit.json').is_file(), parent
            completed = False
            if run.exists():
                meta = json.loads((run / 'run.json').read_text())
                completed = (meta['status'] == 'completed'
                             and meta['completed_steps'] == 100
                             and meta['level'] == 4 and meta['method'] == method)
                if not completed:
                    raise RuntimeError(f'Existing incomplete run; will not overwrite: {run}')
            jobs.append((name, method, parent, run, completed))
            print(f'{name}: {"audit existing simulation" if completed else "simulate and audit"}', flush=True)
    if args.dry_run:
        return

    for name, method, parent, run, completed in jobs:
        if not completed:
            run.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(ROOT / 'scripts/run_bunny_ground_fine.py'),
                       '--scene', 'bunny', '--assets', str(ROOT / 'output/bunny_paper_assets'),
                       '--output', str(run), '--level', '4', '--lower', str(parent),
                       '--steps', '100', '--dt', '.01', '--method', method,
                       '--frozen-offset', '--refined-probe', '--adaptive-search',
                       '--max-newton', '1500', '--preconditioner', 'diag',
                       '--newton-velocity-tol', '.001']
            print(f'Starting {name}', flush=True)
            # Write native bytes directly: PowerShell cannot change the log encoding.
            with run.with_suffix('.log').open('xb') as log:
                with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT) as process:
                    for line in process.stdout:
                        log.write(line)
                        if line.startswith(b'ACCEPTED '):
                            log.flush()
                            print(f'{name}: {line.decode("utf-8", errors="replace").strip()}', flush=True)
                    code = process.wait()
                if code:
                    raise RuntimeError(f'Simulation failed ({code}); see {run.with_suffix(".log")}')
        for script, extra in (
            ('audit_paper_run.py', []),
            ('audit_paper_non_descent.py', []),
            ('audit_flag_fine_search.py', ['--check-converged-energy']),
        ):
            print(f'Auditing {name}: {script}', flush=True)
            with (run / f'{script}.log').open('wb') as log:
                subprocess.run([sys.executable, str(ROOT / 'scripts' / script), str(run), *extra],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Completed and audited: {name}', flush=True)


if __name__ == '__main__':
    main()
