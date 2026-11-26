"""Analytic first derivatives of triangular membrane and hinge bending energies.

SI units. Membrane: compressible surface Neo-Hookean model with plane-stress
Lame parameters. Bending: quadratic signed dihedral angle, rest-shape weighted.
No edge-spring approximation is used.
"""

import numpy as np


def dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def normalize(v):
    length = np.linalg.norm(v, axis=1)
    return v / np.maximum(length[:, None], 1e-30), length


class ShellEnergy:
    def __init__(self, mesh, young, poisson, thickness):
        self.faces = mesh.faces
        self.mu = young * thickness / (2 * (1 + poisson))
        self.lam = young * thickness * poisson / (1 - poisson**2)
        self.kb = young * thickness**3 / (12 * (1 - poisson**2))
        uv = mesh.uv[mesh.faces]
        dm = np.stack((uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]), axis=2)
        self.inv_dm = np.linalg.inv(dm)
        self.areas = np.abs(np.linalg.det(dm)) / 2
        edges, hinges, weights = {}, [], []
        for f, tri in enumerate(mesh.faces):
            for a, b, c in ((tri[0], tri[1], tri[2]), (tri[1], tri[2], tri[0]), (tri[2], tri[0], tri[1])):
                key = tuple(sorted((int(a), int(b))))
                if key in edges:
                    aa, bb, cc, other = edges.pop(key)
                    hinges.append((aa, bb, cc, c))
                    edge2 = np.sum((mesh.uv[bb] - mesh.uv[aa])**2)
                    weights.append(3 * edge2 / (self.areas[other] + self.areas[f]))
                else:
                    edges[key] = (a, b, c, f)
        self.hinges = np.asarray(hinges, dtype=int)
        self.weights = np.asarray(weights)
        self.rest_angles = self._hinge_geometry(mesh.rest)[0]

    def membrane(self, x):
        tri = x[self.faces]
        ds = np.stack((tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=2)
        f = ds @ self.inv_dm
        c = f.transpose(0, 2, 1) @ f
        det = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1]**2
        if np.any(det <= 1e-18):
            return float("inf"), np.zeros_like(x)
        logj = 0.5 * np.log(det)
        density = 0.5 * self.mu * (np.trace(c, axis1=1, axis2=2) - 2) - self.mu * logj + 0.5 * self.lam * logj**2
        invc = np.empty_like(c)
        invc[:, 0, 0], invc[:, 1, 1] = c[:, 1, 1] / det, c[:, 0, 0] / det
        invc[:, 0, 1] = invc[:, 1, 0] = -c[:, 0, 1] / det
        stress = self.mu * f + (self.lam * logj - self.mu)[:, None, None] * (f @ invc)
        gds = self.areas[:, None, None] * (stress @ self.inv_dm.transpose(0, 2, 1))
        g = np.zeros_like(x)
        np.add.at(g, self.faces[:, 1], gds[:, :, 0])
        np.add.at(g, self.faces[:, 2], gds[:, :, 1])
        np.add.at(g, self.faces[:, 0], -gds.sum(axis=2))
        return float(self.areas @ density), g

    def _hinge_geometry(self, x):
        a, b, c, d = (x[self.hinges[:, k]] for k in range(4))
        e, u, v = b - a, c - a, d - a
        eh, le = normalize(e)
        n1, l1 = normalize(np.cross(e, u))
        n2, l2 = normalize(np.cross(v, e))
        s, co = dot(eh, np.cross(n1, n2)), dot(n1, n2)
        return np.arctan2(s, co), (e, u, v, eh, le, n1, l1, n2, l2, s, co)

    def bending(self, x):
        theta, data = self._hinge_geometry(x)
        e, u, v, eh, le, n1, l1, n2, l2, s, co = data
        if np.any(np.minimum(l1, l2) < 1e-14):
            return float("inf"), np.zeros_like(x)
        delta = np.arctan2(np.sin(theta - self.rest_angles), np.cos(theta - self.rest_angles))
        coeff = self.kb * self.weights
        scale = coeff * delta / np.maximum(s*s + co*co, 1e-30)
        gs, gc = (scale * co)[:, None], (-scale * s)[:, None]
        geh = gs * np.cross(n1, n2)
        gn1 = gs * np.cross(n2, eh) + gc * n2
        gn2 = gs * np.cross(eh, n1) + gc * n1
        gr1 = (gn1 - n1 * dot(n1, gn1)[:, None]) / l1[:, None]
        gr2 = (gn2 - n2 * dot(n2, gn2)[:, None]) / l2[:, None]
        ge = (geh - eh * dot(eh, geh)[:, None]) / le[:, None]
        ge += np.cross(u, gr1) + np.cross(gr2, v)
        gu, gv = np.cross(gr1, e), np.cross(e, gr2)
        g = np.zeros_like(x)
        for k, values in enumerate((-ge-gu-gv, ge, gu, gv)):
            np.add.at(g, self.hinges[:, k], values)
        return float(0.5 * np.sum(coeff * delta**2)), g

    def value_grad(self, x):
        em, gm = self.membrane(x)
        eb, gb = self.bending(x)
        return em + eb, gm + gb


def floor_barrier(x, mass, height, distance, stiffness):
    """Frictionless vertex/plane log barrier; affine plane bounds keep steps feasible."""
    gap = x[:, 2] - height
    g = np.zeros_like(x)
    if np.any(gap <= 0):
        return float("inf"), g
    mask = gap < distance
    d = gap[mask]
    diff = d - distance
    log = np.log(d / distance)
    w = stiffness * mass[mask]
    energy = -np.sum(w * diff**2 * log)
    g[mask, 2] = -w * (2 * diff * log + diff**2 / d)
    return float(energy), g
