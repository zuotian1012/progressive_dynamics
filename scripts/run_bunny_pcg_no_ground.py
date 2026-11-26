"""Run the requested 1-second L0 friction pair, serially on the GPU."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/bunny_pcg_no_ground_20261124'

def main():
    OUT.mkdir(exist_ok=False)
    status = {'status': 'running', 'runs': {}}
    for label, friction in [('mu0', '0'), ('mu03', '.3')]:
        run = OUT / label
        cmd = [sys.executable, str(ROOT/'scripts/run_bunny_ground_l0.py'),
               '--scene', 'bunny', '--assets', str(ROOT/'output/bunny_paper_assets'),
               '--output', str(run), '--steps', '100', '--no-ground',
               '--frozen-offset', '--refined-probe', '--adaptive-search',
               '--friction', friction, '--contact-distance', '.0007',
               '--max-newton', '1500', '--newton-velocity-tol', '.001',
               '--linear-tol-rate', '1e-10', '--preconditioner', 'diag',
               '--native-package', str(ROOT/'output/flag_L3_fix_20261118/bunny_ordered_pcg_runtime/src')]
        print(f'Starting {label}', flush=True)
        with run.with_suffix('.log').open('w', encoding='utf-8') as log:
            code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        status['runs'][label] = {'returncode': code, 'command': cmd}
        (OUT/'suite.json').write_text(json.dumps(status, indent=2))
        print(f'Finished {label}: {code}', flush=True)
        if code:
            continue
        for script in ['audit_paper_run.py', 'audit_paper_non_descent.py']:
            with (run/(script+'.log')).open('w', encoding='utf-8') as log:
                audit = subprocess.run([sys.executable, str(ROOT/'scripts'/script), str(run)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status['runs'][label][script] = audit.returncode
            print(f'{label} {script}: {audit.returncode}', flush=True)
    status['status'] = 'finished'
    (OUT/'suite.json').write_text(json.dumps(status, indent=2))

if __name__ == '__main__':
    main()
