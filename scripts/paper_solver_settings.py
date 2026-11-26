"""Solver settings shared with the 2026-11-17 four-level shirt demo."""

NEWTON_VELOCITY_TOL = 0.05
LINEAR_TOL_RATE = 1e-3
# Fused PCG compares r^T M^-1 r, the square of the preconditioned norm.
BUNNY_LINEAR_TOL_RATE = 1e-10
LINE_SEARCH_MAX_ITER = 8


def validate_shirt_solver_history(meta):
    """Reject old solver/stopping histories before constructing a new level."""
    cfg = meta['config']
    velocity_tol = meta.get('bunny_newton_velocity_tol', NEWTON_VELOCITY_TOL) if meta.get('scene')=='bunny' else NEWTON_VELOCITY_TOL
    linear_ratio = meta.get('bunny_linear_tol_rate', LINEAR_TOL_RATE) if meta.get('scene')=='bunny' else LINEAR_TOL_RATE
    if (not 0 < velocity_tol <= NEWTON_VELOCITY_TOL or cfg['newton']['velocity_tol'] != velocity_tol
            or not 0 < linear_ratio <= LINEAR_TOL_RATE or cfg['linear_system']['tol_rate'] != linear_ratio
            or cfg['line_search']['max_iter'] != LINE_SEARCH_MAX_ITER
            or meta.get('preview_min_alpha', 0.) != 0.
            or meta.get('tiny_probe', False)
            or meta.get('small_step_preview_steps', 0) != 0):
        raise ValueError(
            'Lower trajectory uses different solver/stopping settings. '
            'The current settings require the recorded Newton tolerance, '
            'the recorded PCG ratio, line_search.max_iter=8, verified non-descent only '
            '(small-alpha stop and tiny-alpha probe disabled). '
            'Start a new hierarchy from L0 in a new output directory instead of mixing configurations.')
