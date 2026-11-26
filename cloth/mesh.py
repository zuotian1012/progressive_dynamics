from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Mesh:
    uv: np.ndarray
    faces: np.ndarray
    segments: int

    @property
    def rest(self):
        return np.column_stack((self.uv[:, 0] - 0.5, self.uv[:, 1], np.zeros(len(self.uv))))


def grid(segments):
    if segments < 2:
        raise ValueError("A cloth grid needs at least two segments per side.")
    u, v = np.meshgrid(np.linspace(0, 1, segments + 1), np.linspace(0, 1, segments + 1))
    faces = []
    for j in range(segments):
        for i in range(segments):
            a = j * (segments + 1) + i
            b, c, d = a + 1, a + segments + 1, a + segments + 2
            faces.extend(((a, b, d), (a, d, c)))
    return Mesh(np.column_stack((u.ravel(), v.ravel())), np.array(faces), segments)


def prolongation(coarse, fine):
    """Piecewise linear barycentric interpolation on the coarse triangles."""
    n = coarse.segments
    q = fine.uv * n
    cell = np.minimum(np.floor(q).astype(int), n - 1)
    s, t = (q - cell).T
    a = cell[:, 1] * (n + 1) + cell[:, 0]
    b, c, d = a + 1, a + n + 1, a + n + 2
    lower = s >= t
    indices = np.column_stack((a, np.where(lower, b, c), d))
    weights = np.column_stack((1 - np.maximum(s, t), np.abs(s - t), np.minimum(s, t)))
    return csr_matrix((weights.ravel(), (np.repeat(np.arange(len(q)), 3), indices.ravel())),
                      shape=(len(fine.uv), len(coarse.uv)))


def lumped_mass(mesh, surface_density):
    p = mesh.rest[mesh.faces]
    areas = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2
    mass = np.zeros(len(mesh.uv))
    np.add.at(mass, mesh.faces.ravel(), np.repeat(areas * surface_density / 3, 3))
    return mass
