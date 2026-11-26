"""Audit a shirt run using the isolated compensated/stabilized PCG backend."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def audit(root):
    meta = json.loads((root / 'run.json').read_text())
    records = json.loads((root / 'steps.json').read_text())
    launch = json.loads((root.parent / 'launch.json').read_text())
    final_checks = {}
    checks = retries = restarts = 0
    stabilized = []
    iterations = []
    check_pattern = re.compile(r'PCG_TRUE_RESIDUAL frame=(\d+) newton=(\d+) iterations=(\d+) ratio=(\S+) tolerance=(\S+) restarts=(\d+)')
    shift_pattern = re.compile(r'PCG_STABILIZED_SOLVE frame=(\d+) newton=(\d+) relative_diagonal_shift=(\S+)')
    with root.with_suffix('.log').open(encoding='utf-8-sig', errors='replace') as stream:
        for line in stream:
            match = check_pattern.search(line)
            if match:
                frame, newton, count, ratio, tolerance, restart = match.groups()
                final_checks[(int(frame), int(newton))] = (float(ratio), float(tolerance))
                checks += 1
            match = shift_pattern.search(line)
            if match:
                frame, newton, shift = match.groups()
                stabilized.append(dict(frame=int(frame), newton=int(newton), shift=float(shift)))
            retries += 'PCG_ROUNDOFF_RETRY ' in line
            restarts += 'PCG_RELIABLE_RESTART ' in line
            match = re.search(r'Iterative linear solver iteration count: (\d+)', line)
            if match:
                iterations.append(int(match[1]))
    accepted = meta['completed_steps']
    accepted_checks = {k: v for k, v in final_checks.items() if k[0] <= accepted}
    failed_checks = [dict(frame=k[0], newton=k[1], ratio=v[0], tolerance=v[1])
                     for k, v in accepted_checks.items()
                     if not math.isfinite(v[0]) or not 0 <= v[0] <= v[1]]
    bad_steps = [i + 1 for i, record in enumerate(records[:accepted])
                 if (not record['native']['converged'] or
                     record['native'].get('preview_non_descent') or
                     record['native']['hit_line_search_limit'] or
                     record['native']['hit_newton_limit'])]
    runtime = Path(meta['backend_path']).parent
    dlls = list(runtime.parent.rglob('uipc_backend_cuda.dll'))
    assert len(dlls) == 1, dlls
    dll_hash = hashlib.sha256(dlls[0].read_bytes()).hexdigest()
    result = dict(
        completed=meta['completed'], completed_steps=accepted,
        expected_steps=meta['steps'], physics_seconds=meta['physics_seconds'],
        strict_acceptance_failed_steps=bad_steps,
        explicit_residual_checks=checks,
        accepted_final_residual_checks=len(accepted_checks),
        accepted_failed_final_residual_checks=failed_checks,
        maximum_accepted_final_residual_ratio=max((v[0] for v in accepted_checks.values()), default=None),
        reliable_residual_restarts=restarts, diagonal_retry_attempts=retries,
        stabilized_linear_solves=len(stabilized),
        maximum_relative_diagonal_shift=max((r['shift'] for r in stabilized), default=0),
        first_stabilized_solve=stabilized[0] if stabilized else None,
        maximum_successful_pcg_iterations=max(iterations, default=0),
        maximum_newton_iterations=max((r['native']['newton_iterations'] for r in records[:accepted]), default=0),
        all_velocity_injections_verified=all(r['velocity_readback_error'] <= 1e-12 and
            r['position_injection_error'] <= 1e-12 for r in records[:accepted]),
        runtime_sha256=dll_hash,
        runtime_matches_launch=dll_hash == launch['runtime_hashes']['uipc_backend_cuda.dll'],
        residual_operator='A_eff = A + tau*diag(A); tau starts at zero for every Newton solve',
        physical_energy_and_pd_transfers='Unchanged; nonzero tau modifies the Newton linear approximation only',
        iteration_accounting='Maximum successful PCG iteration count excludes work in rejected retry attempts; physics timing includes that work',
    )
    if meta['completed']:
        assert accepted == meta['steps'] == len(records)
        assert not bad_steps and not failed_checks
        assert result['runtime_matches_launch'] and result['all_velocity_injections_verified']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    out = args.run / 'report'
    out.mkdir(exist_ok=True)
    (out / 'solver_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
