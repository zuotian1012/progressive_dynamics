"""PD 2024 Eq. (7), PD++ 2025 Eqs. (4)-(5), Sections 5.4 and 8.2.3.

Positions use P(x); velocities use its intrinsic linear U, following the
lagged-offset approximation in PD 2024 Section 4.6. Never apply P to a velocity.
"""
import numpy as np

METHODS = ("direct", "semi", "full", "velpro")


def predict(method, level, step, tracks, transfers, dt, initial_velocities):
    if method not in METHODS:
        raise ValueError(f"Unknown integrator: {method}")
    x = tracks[level][step]
    def previous(k):
        return tracks[k][step-1] if step else tracks[k][0]-dt*initial_velocities[k]
    if level == 0 or method == "direct":
        return 2*x-previous(level)
    p = transfers[level-1]
    low = tracks[level-1][step]
    if method == "velpro":
        return x + p.U @ (low-previous(level-1))
    if method == "semi" or level == 1:
        return p.position(low) + p.U @ (low-previous(level-1))
    # Nonlinear extension of the paper's linear full-diagonal stencil:
    # lag the offsets at the current lower-level state, as for the other methods.
    return p.position(low) + p.U @ (low-transfers[level-2].position(previous(level-2)))


def consistency(x, target, mass, weight):
    """Explicit convention: w * (x-target)^T M (x-target), in energy units."""
    if weight < 0:
        raise ValueError("Consistency weight must be nonnegative")
    delta = x-target
    return float(weight*np.sum(mass[:, None]*delta**2)), 2*weight*mass[:, None]*delta
