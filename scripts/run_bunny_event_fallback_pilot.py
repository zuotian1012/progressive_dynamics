"""Fixed-parent L3 A/B: preview exit versus event-triggered exact gradient."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from run_bunny_preview_pilot import audit, ROOT, ASSETS

OUT = ROOT / 'output/bunny_event_fallback_pilot_20261125'
PARENT = ROOT / 'output/bunny_preview_pilot_20261125/L2_frozen'
RUNTIME = ROOT / 'output/flag_L3_fix_20261118/bunny_event_fallback_runtime/src'


def main():
    OUT.mkdir(exist_ok=True)
    parent_hash = hashlib.sha256((PARENT / 'positions.npy').read_bytes()).hexdigest()
    for mode in ('baseline', 'fallback'):
        run = OUT / mode
        enabled = mode == 'fallback'
        command = [sys.executable, '-u', str(ROOT / 'scripts/run_bunny_ground_fine.py'),
            '--scene', 'bunny', '--assets', str(ASSETS), '--output', str(run),
            '--steps', '30', '--frozen-offset', '--refined-probe', '--adaptive-search',
            '--max-newton', '1500', '--newton-velocity-tol', '.001', '--preconditioner', 'diag',
            '--native-package', str(RUNTIME), '--level', '3', '--lower', str(PARENT), '--method', 'velpro']
        env = os.environ.copy()
        env['BUNNY_EVENT_FALLBACK'] = str(int(enabled))
        if not run.exists():
            (OUT / (mode + '_command.json')).write_text(json.dumps(dict(command=command,
                environment={'BUNNY_EVENT_FALLBACK': env['BUNNY_EVENT_FALLBACK']},
                parent_sha256=parent_hash), indent=2))
            print('Starting', mode, flush=True)
            with run.with_suffix('.log').open('xb') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Run failed: {run}; inspect adjacent .log')
        meta = json.loads((run / 'run.json').read_text())
        assert meta['completed'] and meta['completed_steps'] == 30
        assert meta['lower_sha256'] == parent_hash
        assert hashlib.sha256((PARENT / 'positions.npy').read_bytes()).hexdigest() == parent_hash
        meta['event_fallback_enabled'] = enabled
        meta['frozen_offset_only'] = not enabled
        meta['gradient_strategy'] = ('Frozen until preview exit condition; restore current iterate and use full PSQ gradient '
            'for remainder of time step; reset to frozen at next step.' if enabled else 'Original frozen-offset gradient')
        if enabled:
            meta['algorithm_note'] += '; event-triggered full-gradient continuation (see gradient_strategy)'
        (run / 'run.json').write_text(json.dumps(meta, indent=2))
        audit(run)
        print('Completed and audited', mode, flush=True)


if __name__ == '__main__':
    main()
