"""Build PSQ shell-extension assets from successive self-parameterization."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy import sparse


def reference_frames(vertices, faces):
    a, b, c = np.moveaxis(vertices[faces], -2, 0)
    edge, side = b - a, c - a
    normal = np.cross(edge, side)
    length = np.linalg.norm(edge, axis=1)
    area2 = np.linalg.norm(normal, axis=1)
    if np.any(area2 <= 1e-16):
        raise ValueError("Degenerate PSQ support triangle")
    first, normal = edge / length[:, None], normal / area2[:, None]
    frame = np.stack((first, np.cross(normal, first), normal), axis=-1)
    reference = np.zeros((len(faces), 2, 2))
    reference[:, 0, 0] = length
    reference[:, 0, 1] = np.sum(first * side, axis=1)
    reference[:, 1, 1] = area2 / length
    return frame, np.linalg.inv(reference)


def shell_prolongation(coarse, intrinsic, support, local, inverse_rest):
    """PSQ Eq. (7)/(9). The stored intrinsic map acts on XYZ independently."""
    a, b, c = np.moveaxis(coarse[support], 1, 0)
    edge, side = b - a, c - a
    normal = np.cross(edge, side)
    lengths = np.linalg.norm(normal, axis=1)
    if np.any(lengths <= 1e-16):
        raise ValueError("Degenerate deformed PSQ support triangle")
    deformation = np.stack((edge, side), axis=-1) @ inverse_rest
    left, _, right = np.linalg.svd(deformation, full_matrices=False)
    rotation = left @ right
    return (intrinsic @ coarse + np.einsum("nij,nj->ni", rotation, local[:, :2])
            + normal * (local[:, 2] / lengths)[:, None])


def write_binding(output, kind, level, target, coarse, fine, coarse_faces, intrinsic, support_ids,
                  proxy=None, *, zero_offset=False):
    intrinsic = intrinsic.tocsr()
    intrinsic.eliminate_zeros()
    if (support_ids.shape != (len(fine),) or np.any(support_ids < 0)
            or np.any(support_ids >= len(coarse_faces))):
        raise ValueError("SSP produced invalid support face indices")
    if (not np.isfinite(intrinsic.data).all() or np.any(intrinsic.data < 0) or
            not np.allclose(np.asarray(intrinsic.sum(axis=1)), 1, atol=1e-12, rtol=0)):
        raise ValueError("SSP produced invalid intrinsic interpolation rows")
    support = coarse_faces[support_ids]
    frames, inverse_rest = reference_frames(coarse, support)
    anchor = intrinsic @ coarse
    local = np.einsum("nij,ni->nj", frames, fine - anchor)
    if zero_offset:
        if np.max(np.abs(fine - anchor)) > 1e-12:
            raise ValueError("Zero-offset PSQ requires an exact rest-pose interpolation")
        local.fill(0)
    recovered = shell_prolongation(coarse, intrinsic, support, local, inverse_rest)
    error = float(np.max(np.abs(recovered - fine)))
    if not error <= 1e-12:
        raise ValueError(f"PSQ rest reconstruction failed at {level}→{target}: {error}")
    matrix_path = output / f"P_{kind}_L{level}_to_L{target}.npz"
    sparse.save_npz(matrix_path, sparse.kron(intrinsic, sparse.eye(3), format="csr"))
    offset_path = output / f"PSQ_{kind}_L{level}_to_L{target}.npz"
    np.savez_compressed(offset_path, faces=support, local=local, inverse_rest=inverse_rest)
    if proxy is not None:
        columns, weights = [], []
        for begin, end in zip(intrinsic.indptr[:-1], intrinsic.indptr[1:], strict=True):
            columns.append(intrinsic.indices[begin:end].tolist())
            weights.append(intrinsic.data[begin:end].tolist())
        data = {"coarse_vertices": len(coarse), "rest": fine.tolist(), "triangles": proxy.tolist(),
                "columns": columns, "weights": weights,
                "psq": {"faces": support.tolist(), "offsets": local.tolist()}}
        (output / f"proxy_L{level}.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    return {"kind": kind, "source": level, "target": target, "matrix": matrix_path.name,
            "offsets": offset_path.name, "rest_error_m": error,
            "intrinsic_only_rest_error_m": float(np.max(np.linalg.norm(fine - anchor, axis=1))),
            "normal_offset_max_m": float(np.max(np.abs(local[:, 2]))),
            "tangent_offset_max_m": float(np.max(np.linalg.norm(local[:, :2], axis=1))),
            "max_intrinsic_weights_per_vertex": int(np.diff(intrinsic.indptr).max())}


def prepare(fine_mesh, face_targets, executable, output):
    """Decimate fine to coarse, then compose U and recompute offsets for direct transfers."""
    output.mkdir(parents=True, exist_ok=True)
    ssp = output / "ssp"
    last = len(face_targets)
    subprocess.run([str(executable.resolve()), str(fine_mesh.resolve()), str(ssp.resolve()),
                    str(last + 1), *map(str, face_targets)], check=True)
    rest, faces, matrices = [], [], []
    stats = []
    for level in range(last + 1):
        rest.append(np.loadtxt(ssp / f"L{level}_vertices.txt", ndmin=2))
        faces.append(np.loadtxt(ssp / f"L{level}_triangles.txt", dtype=np.int32, ndmin=2))
        np.savez_compressed(output / f"mesh_L{level}.npz", rest=rest[level], triangles=faces[level])
        if level == 0:
            continue
        base = ssp / f"L{level - 1}"
        weights = np.loadtxt(f"{base}_adjacent_weights.txt", ndmin=2)
        columns = np.loadtxt(f"{base}_adjacent_vertices.txt", dtype=np.int32, ndmin=2)
        support_ids = np.loadtxt(f"{base}_adjacent_faces.txt", dtype=np.int32, ndmin=1)
        matrix = sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(len(weights)), 3),
                                                       columns.ravel())),
                                   shape=(len(rest[level]), len(rest[level - 1])))
        matrices.append(matrix)
        stats.append(write_binding(output, "adjacent", level - 1, level, rest[level - 1],
                                   rest[level], faces[level - 1], matrix, support_ids))
    for level in range(last):
        direct = matrices[level]
        for matrix in matrices[level + 1:]:
            direct = matrix @ direct
        support = np.loadtxt(ssp / f"L{level}_direct_faces.txt", dtype=np.int32, ndmin=1)
        stats.append(write_binding(output, "direct", level, last, rest[level], rest[last],
                                   faces[level], direct, support, proxy=faces[last]))
    manifest = {"levels": [{"vertices": len(x), "triangles": len(f)} for x, f in zip(rest, faces)],
                "operators": stats, "input_mesh": str(fine_mesh.resolve()),
                "face_targets": list(face_targets)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--ssp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--face-targets", type=int, nargs="+", required=True,
                        help="Successive face counts from the fine mesh toward the coarsest")
    args = parser.parse_args()
    prepare(args.mesh, args.face_targets, args.ssp, args.output)


if __name__ == "__main__":
    main()
