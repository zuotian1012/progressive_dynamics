"""PD++ Eqs. (11)-(13), with explicit normalization and undefined ratios."""
import numpy as np
from scipy.sparse.linalg import splu


def squared_mass_norm(delta,mass):
    return float(np.sum(mass[:,None]*delta**2))


def consistency_metrics(coarse,fine,transfer,mass_coarse,mass_fine):
    U=transfer.U.tocsc()
    # Paper Eq. (13) uses the unweighted intrinsic least-squares inverse.
    factor=splu((U.T @ U).tocsc())
    restricted=factor.solve(U.T @ fine)
    residual=restricted-coarse
    detail=fine-U @ restricted
    squared=squared_mass_norm(residual,mass_coarse)
    return {"coarse_mass_squared":squared,"coarse_rms_m":float(np.sqrt(squared/mass_coarse.sum())),
            "fine_detail_rms_m":float(np.sqrt(squared_mass_norm(detail,mass_fine)/mass_fine.sum()))}


def continuity(previous,current,next_position,xhat,potential_gradient,mass,dt,solver_gradient=None,free=None):
    free=np.ones(len(mass),bool) if free is None else free
    midpoint=.5*(next_position-2*current+previous+dt**2*potential_gradient/mass[:,None])
    residual=next_position-xhat+dt**2*(potential_gradient if solver_gradient is None else solver_gradient)/mass[:,None]
    e=squared_mass_norm(midpoint[free],mass[free])
    denominator=squared_mass_norm(residual[free],mass[free])
    rms=np.sqrt(denominator/mass[free].sum())
    return {"midpoint_mass_squared":e,"solver_mass_squared":denominator,
            "midpoint_rms_m":float(np.sqrt(e/mass[free].sum())),"solver_rms_m":float(rms),
            # The printed equations contain a factor 1/2 only in the numerator:
            # exact same nonzero direct-solve residual gives ratio 1/4, not 1.
            "literal_squared_ratio":float(e/denominator) if rms > 1e-10 else None}
