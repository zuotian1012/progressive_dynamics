"""PSQ 2023, Section 4.3, Eqs. (4)-(9): P(x)=U x+n(x) gamma+R(F(x)) t.

U and containing coarse triangles are INPUT hierarchy data. This module does
not pretend nearest-point projection is intrinsic self-parameterization.
Nested midpoint maps can supply exact containing triangles but have zero
rest offsets; a general intrinsic-decimation hierarchy must supply its maps.
Direct-to-finest operators are constructed from composed U, not composed P.
"""
import numpy as np
from scipy.sparse import csr_matrix, eye, kron


def frames(rest, faces):
    tri = rest[faces]
    a, b = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    lengths = np.linalg.norm(a, axis=1)
    cross = np.cross(a, b)
    areas2 = np.linalg.norm(cross, axis=1)
    if np.any(lengths <= 1e-14) or np.any(areas2 <= 1e-16):
        raise ValueError("Degenerate rest triangle")
    e, n = a/lengths[:, None], cross/areas2[:, None]
    basis = np.stack((e, np.cross(n, e)), axis=2)
    dm = np.zeros((len(faces), 2, 2))
    dm[:, 0, 0] = lengths
    dm[:, 0, 1] = np.einsum("ij,ij->i", e, b)
    dm[:, 1, 1] = areas2/lengths
    return basis, n, np.linalg.inv(dm), areas2/2


def nested_anchors(U, faces):
    """Exact containing-face search for nested, piecewise-linear hierarchy."""
    incidence = [set() for _ in range(U.shape[1])]
    for i, face in enumerate(faces):
        for v in face:
            incidence[v].add(i)
    anchors = []
    for row in range(U.shape[0]):
        ids = U.indices[U.indptr[row]:U.indptr[row+1]]
        candidates = set.intersection(*(incidence[i] for i in ids)) if len(ids) else set()
        if not candidates:
            raise ValueError("Map is not face-nested; supply intrinsic anchor triangles explicitly")
        anchors.append(min(candidates))
    return np.asarray(anchors, dtype=int)


class ShellProlongation:
    def __init__(self, coarse_rest, fine_rest, coarse_faces, U, anchors=None):
        self.U = csr_matrix(U, dtype=float)
        self.U.eliminate_zeros()
        self.faces = np.asarray(coarse_faces, dtype=int)
        self.coarse_rest, self.fine_rest = np.asarray(coarse_rest), np.asarray(fine_rest)
        if self.U.shape != (len(fine_rest), len(coarse_rest)):
            raise ValueError("Invalid prolongation dimensions")
        if not np.allclose(np.asarray(self.U.sum(axis=1)).ravel(), 1, atol=1e-12):
            raise ValueError("U must preserve translations (row sums equal one)")
        self.anchors = nested_anchors(self.U, self.faces) if anchors is None else np.asarray(anchors, int)
        if self.anchors.shape != (len(fine_rest),) or np.any(self.anchors < 0) or np.any(self.anchors >= len(self.faces)):
            raise ValueError("Invalid anchor triangles")
        basis, normal, self.inv_dm, _ = frames(self.coarse_rest, self.faces)
        delta = fine_rest-self.U @ coarse_rest
        self.gamma = np.einsum("ij,ij->i", normal[self.anchors], delta)
        self.tangent = np.einsum("nji,nj->ni", basis[self.anchors], delta)
        self.zero_offset = bool(np.max(np.abs(delta), initial=0) < 1e-13)
        self.U3 = kron(self.U, eye(3), format="csr")

    def offset(self, x):
        if self.zero_offset:
            return np.zeros((self.U.shape[0], 3))
        tri = x[self.faces]
        ds = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=2)
        F = ds @ self.inv_dm
        left, sigma, right = np.linalg.svd(F, full_matrices=False)
        if np.min(sigma) < 1e-12:
            raise ValueError("Degenerate triangle in PSQ prolongation")
        rotation = left @ right
        normal = np.cross(F[:, :, 0], F[:, :, 1])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        return (normal[self.anchors]*self.gamma[:, None]
                + np.einsum("nij,nj->ni", rotation[self.anchors], self.tangent))

    def velocity(self, x, v):
        """Exact directional derivative DP(x)[v] of shell position prolongation."""
        linear = self.U @ v
        if self.zero_offset:
            return linear
        tri, vel = x[self.faces], v[self.faces]
        F = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=2) @ self.inv_dm
        dF = np.stack((vel[:, 1]-vel[:, 0], vel[:, 2]-vel[:, 0]), axis=2) @ self.inv_dm
        left, sigma, vt = np.linalg.svd(F, full_matrices=False)
        if np.min(sigma) < 1e-12:
            raise ValueError("Degenerate triangle in PSQ velocity prolongation")
        rotation = left @ vt
        normal = np.cross(rotation[:, :, 0], rotation[:, :, 1])
        # Polar decomposition F = R S.  The in-plane part of dR is a
        # skew rotation, and its normal component is n n^T dF S^{-1}.
        body = rotation.transpose(0, 2, 1) @ dF
        spin = (body[:, 1, 0]-body[:, 0, 1]) / sigma.sum(axis=1)
        inv_stretch = (vt.transpose(0, 2, 1) / sigma[:, None, :]) @ vt
        tilt = np.einsum("fi,fij->fj", normal, dF)[:, None, :] @ inv_stretch
        d_rotation = normal[:, :, None] * tilt
        d_rotation[:, :, 0] += spin[:, None]*rotation[:, :, 1]
        d_rotation[:, :, 1] -= spin[:, None]*rotation[:, :, 0]
        d_normal = -(rotation @ tilt.transpose(0, 2, 1))[:, :, 0]
        return (linear + d_normal[self.anchors]*self.gamma[:, None]
                + np.einsum("nij,nj->ni", d_rotation[self.anchors], self.tangent))

    def position(self, x):
        return self.U @ x+self.offset(x)


class Hierarchy:
    def __init__(self, rests, faces, maps, adjacent_anchors=None, finest_anchors=None):
        if len(rests) != len(faces) or len(maps) != len(rests)-1:
            raise ValueError("Invalid hierarchy")
        self.rests, self.faces = rests, faces
        adjacent_anchors = adjacent_anchors or [None]*len(maps)
        finest_anchors = finest_anchors or [None]*len(rests)
        if len(adjacent_anchors) != len(maps) or len(finest_anchors) != len(rests):
            raise ValueError("Anchor lists must match hierarchy levels")
        self.adjacent = [ShellProlongation(a, b, f, u, anchor) for a, b, f, u, anchor in
                         zip(rests, rests[1:], faces, maps, adjacent_anchors)]
        self.to_finest = [None]*len(rests)
        U = eye(len(rests[-1]), format="csr")
        for k in reversed(range(len(rests))):
            if k < len(rests)-1:
                U = U @ maps[k]
            self.to_finest[k] = ShellProlongation(rests[k], rests[-1], faces[k], U, finest_anchors[k])
