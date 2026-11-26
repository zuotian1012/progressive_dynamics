"""
Cloth material for IPC-based cloth simulation.

This material is used with FEMEntity and IPCCoupler for shell/membrane simulation.
"""

from typing import Literal

from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base


class Cloth(Base):
    """
    Cloth material for thin shell/membrane simulation using IPC.

    This material is designed for cloth, fabric, and other thin flexible materials.
    It uses shell-based FEM formulation (NeoHookeanShell) in the IPC backend.

    When used with FEMEntity, it signals to IPCCoupler that this entity should be
    treated as a 2D shell (cloth) rather than a 3D volumetric FEM object.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa), controlling stiffness. Default is 1e4 (10 kPa).
    stretch_E, shear_E : float | None, optional
        Explicit libuipc 0.0.28 independent membrane inputs; supply both in
        native mode. Stretch stores E_stretch*2r/(1-nu²), while shear stores
        E_shear/[2(1+nu)] without an extra thickness factor. In this mode E is
        unused and bending_stiffness is the uniform native kappa (N m).
    nu : float, optional
        Poisson's ratio, describing volume change under stress.
        Default is 0.49 (nearly incompressible).
    rho : float, optional
        Material density (kg/m³). Default is 200 (typical fabric).
    thickness : float, optional
        One-sided collision radius r (m). Material full thickness is 2r.
        Default r is 0.001 (1mm).
    bending_stiffness : float, optional
        Raw bending coefficient, whose weighting depends on libuipc version.
        The optional legacy025 migration adapter preserves v0.0.25 weighting.
        If None, no bending resistance.
        Default is None.
    model : str, optional
        FEM material model (not used for cloth, kept for compatibility).
        Default is "stable_neohookean".
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    self_friction_mu : float | None, optional
        Optional friction coefficient for the cloth self-contact pair only.
        ``None`` keeps the regular ``friction_mu`` for self contact.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override. ``None`` uses the IPC coupler
        global default. Default is None.
    strain_rate : float, optional
        Native libuipc 0.0.28 cubic extension amplification, default 100.
        Dimensionless; this is not velocity damping or a hard strain limit.

    Notes
    -----
    - Only works with IPCCoupler enabled
    - Requires GPU backend
    - Only accepts surface mesh morphs (Mesh, etc.)
    - Uses FEMEntity infrastructure but simulated as 2D shell in IPC

    Examples
    --------
    >>> cloth = scene.add_entity(
    ...     morph=gs.morphs.Mesh(file="cloth.obj"),
    ...     material=gs.materials.FEM.Cloth(
    ...         E=10e3, nu=0.49, rho=200, thickness=0.001, bending_stiffness=10.0
    ...     ),
    ... )
    """

    E: PositiveFloat = 1e4
    # Explicit v0.0.28 independent membrane inputs. Both must be provided.
    stretch_E: PositiveFloat | None = None
    shear_E: PositiveFloat | None = None
    # Native 0.0.28 cubic extension amplification; dimensionless, not damping.
    strain_rate: NonNegativeFloat = 100.0
    nu: PositiveFloat = 0.49
    rho: PositiveFloat = 200.0
    thickness: PositiveFloat = 0.001
    bending_stiffness: NonNegativeFloat | None = None
    self_friction_mu: NonNegativeFloat | None = None
    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "stable_neohookean"
