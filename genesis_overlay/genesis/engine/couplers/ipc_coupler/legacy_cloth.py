"""Map v0.0.25 shell coefficients to v0.0.28's area-weighted model.

Explicit opt-in only. This preserves old constitutive coefficients, not solver
or contact behavior. Call after membrane and optional bending apply_to().
"""
import os
from importlib.metadata import version
import numpy as np
import uipc


def adapt_legacy_cloth(mesh, material):
    mode = os.environ.get('GENESIS_IPC_CLOTH_PARAMETERIZATION', 'native')
    if mode == 'native':
        return
    if mode != 'legacy025':
        raise ValueError(f'Unknown cloth parameterization: {mode}')
    if version('pyuipc') != '0.0.28':
        raise RuntimeError('legacy025 adapter is validated only for pyuipc 0.0.28')
    r = float(material.thickness)
    E, nu = float(material.E), float(material.nu)
    # v25 stores 2D Lame lambda/mu; backend multiplies both by area*2r.
    # v28 backend multiplies by area only. Write equivalent area coefficients.
    uipc.view(mesh.triangles().find('lambda'))[:] = E * nu / (1 - nu**2) * (2*r)
    uipc.view(mesh.triangles().find('mu'))[:] = E / (2*(1+nu)) * (2*r)
    uipc.view(mesh.triangles().find('strain_rate'))[:] = 100.0
    if material.bending_stiffness is None:
        return
    positions = np.asarray(mesh.positions().view()).reshape(-1, 3)
    triangles = np.asarray(mesh.triangles().topo().view()).reshape(-1, 3)
    edges = np.asarray(mesh.edges().topo().view()).reshape(-1, 2)
    area = .5*np.linalg.norm(np.cross(positions[triangles[:,1]]-positions[triangles[:,0]],
                                    positions[triangles[:,2]]-positions[triangles[:,0]]),axis=1)
    adjacent = {}
    for tri,a in zip(triangles,area):
        for i,j in ((0,1),(1,2),(2,0)):
            adjacent.setdefault(tuple(sorted((int(tri[i]),int(tri[j])))),[]).append(float(a))
    values = np.zeros(len(edges))
    for k,e in enumerate(edges):
        areas = adjacent.get(tuple(sorted(map(int,e))),[])
        if len(areas)>2:
            raise ValueError('Legacy bending mapping requires a manifold surface')
        if len(areas)==2:
            # v25 outer weight=(A1+A2)*mean(four vertex radii).
            # Genesis Cloth uses a uniform radius r.
            values[k] = float(material.bending_stiffness)*sum(areas)*r
    uipc.view(mesh.edges().find('bending_stiffness'))[:] = values
