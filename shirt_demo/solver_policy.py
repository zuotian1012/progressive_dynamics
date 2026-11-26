"""Strict event-fallback policy shared by shirt simulation and audits."""
import math


def converged_step(stats):
    return (stats.get('completed') is True and stats.get('converged') is True
            and stats.get('termination_reason') == 'newton_converged'
            and not any(stats.get(k) for k in ('preview_non_descent', 'preview_small_step',
                                              'hit_newton_limit', 'hit_line_search_limit')))


def validate_parent(meta, rows, runtime_hash):
    if (not meta.get('completed') or not meta.get('event_fallback_enabled')
            or not meta.get('requires_convergence')
            or meta.get('event_runtime_sha256') != runtime_hash
            or len(rows) != meta.get('completed_steps')
            or not all(converged_step(r['native']) for r in rows)):
        raise ValueError('Event-fallback parent must be complete, strictly converged and use the identical runtime')


def configure_event_solver(cfg, linear_tol, newton_tol, max_iterations):
    linear_tol = 1e-10 if linear_tol is None else linear_tol
    newton_tol = .001 if newton_tol is None else newton_tol
    if not math.isfinite(linear_tol) or not 0 < linear_tol <= 1e-10:
        raise ValueError('Event PCG squared residual ratio must be in (0, 1e-10]')
    if not math.isfinite(newton_tol) or not 0 < newton_tol <= .001:
        raise ValueError('Event Newton velocity tolerance must be in (0, .001]')
    if max_iterations < 1:
        raise ValueError('Newton iteration budget must be positive')
    cfg['linear_system'].update(solver='fused_pcg', tol_rate=linear_tol)
    cfg['newton'].update(velocity_tol=newton_tol, velocity_tol_relative=0.,
                         use_adaptive_tol=0, max_iter=max_iterations)
    cfg['newton']['semi_implicit']['enable'] = 0
