"""Run a no-ground GPU PCG VelPro hierarchy; optional event fallback requires convergence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from run_bunny_preview_pilot import audit, ROOT, ASSETS, RUNTIME
import run_bunny_preview_pilot as pilot
from bunny_runtime import runtime_provenance, check_run_sources
from report_bunny_event_fallback_pilot import event_audit

EVENT_RUNTIME = ROOT / 'build/runtime/src'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(run, level, args, dll_hash):
    m = json.loads((run / 'run.json').read_text())
    assert m['completed'] and m['status'] == 'completed', f'Incomplete run: {run}'
    assert m['steps'] == m['completed_steps'] == args.steps and m['dt'] == .01
    assert m['scene'] == 'bunny' and m['level'] == level and not m['ground']['enabled']
    assert m['contact_friction'] == dict(cloth_bunny=args.friction, cloth_self=args.friction, cloth_ground=0.)
    assert m['frozen_offset_only'] == (not args.event_fallback)
    assert not m['exact_offset_gradient'] and not m.get('initial_non_descent_rescue')
    assert m.get('event_fallback_enabled', False) == args.event_fallback
    if args.event_fallback:
        assert m.get('require_newton_convergence') and m['native_steps'] == args.steps and m['preview_steps'] == 0
        rows = json.loads((run / 'steps.json').read_text())
        assert len(rows) == args.steps
        assert all(r['native']['converged'] and r['native']['termination_reason'] == 'newton_converged'
                   and not r['native']['preview_non_descent'] and not r['native']['preview_small_step']
                   and not r['native']['hit_newton_limit'] and not r['native']['hit_line_search_limit'] for r in rows)
    assert m['bunny_linear_solver'] == 'native_pcg' and m['config']['linear_system']['solver'] == 'fused_pcg'
    assert m['config']['linear_system']['tol_rate'] == 1e-10
    assert m['config']['newton']['velocity_tol'] == .001
    assert m['library_sha256']['uipc_backend_cuda.dll'] == dll_hash
    check_run_sources(run, m['source_files_sha256'])
    if level:
        parent = args.output / f'L{level-1}'
        assert m['method'] == 'velpro' and m.get('velocity_map', 'intrinsic') == 'intrinsic'
        assert m.get('consistency_weight', 0.) == 0.
        assert Path(m['lower_run']).resolve() == parent
        assert m['lower_sha256'] == digest(parent / 'positions.npy')
    return m


def main():
    global ASSETS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=75)
    p.add_argument('--friction', type=float, choices=(0., .3), default=.3)
    p.add_argument('--start-level', type=int, choices=range(8), default=0,
                   help='Start from this level using its completed, audited predecessor')
    p.add_argument('--max-level', type=int, choices=range(8), default=5)
    p.add_argument('--assets', type=Path, default=ASSETS, help='Explicit prepared Bunny hierarchy directory')
    p.add_argument('--native-package', type=Path, help='Explicit runtime; event mode defaults to build/runtime/src')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--event-fallback', action='store_true', help='Use the improved backend and reject every unconverged step')
    args = p.parse_args()
    ASSETS = args.assets.resolve()
    pilot.ASSETS = ASSETS
    if args.steps < 1:
        p.error('--steps must be positive')
    if args.start_level > args.max_level:
        p.error('--start-level must not exceed --max-level')
    args.output = args.output.resolve()
    runtime = args.native_package or (EVENT_RUNTIME if args.event_fallback else RUNTIME)
    provenance = runtime_provenance(runtime)
    dll_hash = provenance['dll_sha256']
    if args.event_fallback:
        assert provenance.get('event_fallback_control')
    print('Solver policy: '+('event-triggered full gradient; EVERY step must converge'
                            if args.event_fallback else 'original frozen gradient; verified preview exits allowed'), flush=True)
    required = [ASSETS / 'bunny.npz', ASSETS / 'mesh_7.npz']
    required += [ASSETS / f'map_{i}_{i+1}.npz' for i in range(7)]
    for level in range(args.max_level + 1):
        required.append(ASSETS / f'mesh_{level}.npz')
        if level < 7:
            required.append(ASSETS / f'anchors_{level}_7.npy')
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.start_level:
        parent = args.output / f'L{args.start_level-1}'
        validate(parent, args.start_level-1, args, dll_hash)
        saved = json.loads((parent / 'no_ground_audit.json').read_text())
        assert saved['passed'] and saved['no_ground'] and not saved['intersecting_states']
        assert saved['saved_states_checked'] == args.steps+1
        assert saved['positions_sha256'] == digest(parent / 'positions.npy')
        assert json.loads((parent / 'audit.json').read_text())['accepted_prefix_checks_passed']
        assert json.loads((parent / 'line_search_audit.json').read_text())['passed']
        assert json.loads((parent / 'report/non_descent_audit.json').read_text())['all_accepted_preview_evidence_verified']
        if args.event_fallback:
            assert json.loads((parent / 'event_audit.json').read_text())['passed']
        print(f'Using completed and audited L{args.start_level-1}: {parent}', flush=True)
    commands = []
    for level in range(args.start_level, args.max_level + 1):
        run = args.output / f'L{level}'
        if run.exists():
            validate(run, level, args, dll_hash)
        elif run.with_suffix('.log').exists():
            raise RuntimeError(f'Existing log; will not overwrite: {run.with_suffix(".log")}')
        command = [sys.executable, '-u', str(ROOT / 'scripts' / (
            'run_bunny_ground_l0.py' if level == 0 else 'run_bunny_ground_fine.py')),
            '--scene', 'bunny', '--assets', str(ASSETS), '--output', str(run),
            '--steps', str(args.steps), '--dt', '.01', '--method', 'velpro',
            '--frozen-offset', '--refined-probe', '--adaptive-search', '--max-newton', '1500',
            '--newton-velocity-tol', '.001', '--preconditioner', 'diag', '--native-package', str(runtime)]
        if args.event_fallback:
            command.append('--event-fallback')
        if level == 7:
            command.append('--solver-trace')
        if level == 0:
            command += ['--no-ground', '--friction', str(args.friction), '--contact-distance', '.0007',
                        '--linear-tol-rate', '1e-10']
        else:
            command += ['--level', str(level), '--lower', str(args.output / f'L{level-1}'),
                        '--velocity-map', 'intrinsic']
        commands.append((level, run, command))
        print(f'L{level}: {args.steps} steps, mu={args.friction:g}, no ground, '
              f'{"L0 base" if level == 0 else "VelPro"}; '
              f'{"audit existing result" if run.exists() else "simulate and audit"}', flush=True)
    if args.dry_run:
        print('Preflight passed. No simulations started and no output folders created.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['BUNNY_EVENT_FALLBACK'] = str(int(args.event_fallback))
    for level, run, command in commands:
        if not run.exists():
            (args.output / f'L{level}_command.json').write_text(json.dumps(command, indent=2))
            print(f'Starting L{level}; log: {run.with_suffix(".log")}', flush=True)
            with run.with_suffix('.log').open('xb') as log:
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT) as process:
                    for line in process.stdout:
                        log.write(line)
                        if line.startswith(b'ACCEPTED '):
                            log.flush()
                            print(f'L{level}: {line.decode("utf-8", errors="replace").strip()}', flush=True)
                    code = process.wait()
            if code:
                raise RuntimeError(f'L{level} simulation failed ({code}); see {run.with_suffix(".log")}')
        validate(run, level, args, dll_hash)
        print(f'Auditing L{level}, including all {args.steps+1} saved states...', flush=True)
        audit(run, expected_steps=args.steps)
        if args.event_fallback:
            m = json.loads((run / 'run.json').read_text())
            event_audit(run, m, json.loads((run / 'steps.json').read_text()))
        print(f'Completed and audited L{level}', flush=True)
    print(f'All L{args.start_level}-L{args.max_level} simulations and audits completed.', flush=True)


if __name__ == '__main__':
    main()
