"""Generate the nested square-cloth meshes and prolongations for Scene527."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


RESOLUTIONS = (8, 16, 32, 64)  # Cells per side
SIDE_LENGTH_M = 0.64
SCENE_CENTER_M = np.array([0.667, 0.015, 0.93], dtype=np.float64)
REST_TOLERANCE_M = 1e-12


def square_mesh(cells, side_length=SIDE_LENGTH_M, center=SCENE_CENTER_M):
    """Return the rest vertices in world meters and upward-wound triangles."""
    y, x = np.indices((cells + 1, cells + 1)).reshape(2, -1)
    rest = np.array(np.broadcast_to(center, (x.size, 3)), dtype=np.float64, order="C")
    rest[:, :2] += (np.column_stack((x, y)) / cells - 0.5) * side_length

    stride = cells + 1
    lower_left = np.arange(stride**2).reshape(stride, stride)[:-1, :-1].ravel()
    lower_right = lower_left + 1
    upper_left = lower_left + stride
    upper_right = upper_left + 1
    triangles = np.stack(
        (
            np.column_stack((lower_left, lower_right, upper_right)),
            np.column_stack((lower_left, upper_right, upper_left)),
        ),
        axis=1,
    ).reshape(-1, 3)
    return rest, triangles


def barycentric_stencil(source_cells, target_cells):
    """Return one coarse-cell triangle and its weights for every fine vertex."""
    if source_cells < 1 or target_cells < source_cells:
        raise ValueError("Target grid must be at least as fine as the source grid")

    # A fine-grid lattice point has source-cell coordinates (u, v). Use the
    # integer lattice to keep edge and diagonal weights exact on these grids.
    y, x = np.indices((target_cells + 1, target_cells + 1)).reshape(2, -1)
    x, y = x * source_cells, y * source_cells
    cell_x = np.minimum(x // target_cells, source_cells - 1)
    cell_y = np.minimum(y // target_cells, source_cells - 1)
    u = (x - cell_x * target_cells) / target_cells
    v = (y - cell_y * target_cells) / target_cells
    lower_triangle = v <= u

    stride = source_cells + 1
    lower_left = cell_y * stride + cell_x
    upper_right = lower_left + stride + 1
    third_vertex = lower_left + np.where(lower_triangle, 1, stride)
    columns = np.column_stack((lower_left, third_vertex, upper_right))
    weights = np.column_stack((1 - np.maximum(u, v), np.abs(u - v), np.minimum(u, v)))
    return columns, weights


def prolongation(source_cells, target_cells):
    """Compute the rest-plane barycentrics and expand them in vertex XYZ order."""
    columns, weights = barycentric_stencil(source_cells, target_cells)
    rows = np.repeat(np.arange(len(columns)), 3)
    scalar = sparse.csr_matrix(
        (weights.ravel(), (rows, columns.ravel())),
        shape=(len(columns), (source_cells + 1)**2),
    )
    scalar.eliminate_zeros()
    return sparse.kron(scalar, sparse.eye(3, format="csr"), format="csr")


def write_obj(path, rest, triangles, center=SCENE_CENTER_M):
    # Scene527 applies scale 0.001, Euler (-90, 0, 0), and the spawn position.
    x, y = ((rest[:, :2] - np.asarray(center)[:2]) * 1000).T
    with path.open("w", encoding="ascii") as obj:
        obj.write("# Scene527 square cloth in local XZ coordinates, millimeters\n")
        for px, py in zip(x, y):
            obj.write(f"v {px:.17g} 0 {py:.17g}\n")
        for a, b, c in triangles + 1:
            obj.write(f"f {a} {b} {c}\n")


def check_rest_prolongation(matrix, source_rest, target_rest):
    """Check the serialized operator against the rest positions and unity rows."""
    shape = (target_rest.size, source_rest.size)
    if matrix.shape != shape:
        raise AssertionError(f"Expected prolongation shape {shape}, got {matrix.shape}")
    projected = matrix @ source_rest.ravel()
    error = float(np.max(np.abs(projected - target_rest.ravel())))
    if not error < REST_TOLERANCE_M:
        raise AssertionError(f"Rest reconstruction error {error:.17g} m")
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, rtol=0, atol=2 * np.finfo(np.float64).eps):
        raise AssertionError("Every prolongation row must sum to 1.0 within floating-point rounding")
    return {
        "shape": list(shape),
        "nnz": matrix.nnz,
        "rest_error_inf_m": error,
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
    }


def generate(output_dir, resolutions=RESOLUTIONS, *, side_length=SIDE_LENGTH_M, center=SCENE_CENTER_M):
    if not resolutions or resolutions[0] < 1 or any(b <= a for a, b in zip(resolutions, resolutions[1:])):
        raise ValueError("Resolutions must form an increasing hierarchy")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "resolution_units": "cells per side",
        "side_length_m": side_length,
        "scene_center_m": list(center),
        "obj_scale": 0.001,
        "obj_euler_degrees": [-90.0, 0.0, 0.0],
        "rest_coordinates": "world meters",
        "dof_order": "vertex-major XYZ, X index varies fastest in the grid",
        "levels": [],
        "operators": [],
    }
    for level, cells in enumerate(resolutions):
        rest, triangles = square_mesh(cells, side_length, center)
        obj = f"square_L{level}.obj"
        mesh = f"mesh_L{level}.npz"
        write_obj(output_dir / obj, rest, triangles, center)
        np.savez_compressed(output_dir / mesh, rest=rest, triangles=triangles)
        manifest["levels"].append(
            {"level": level, "cells_per_side": cells, "vertices": len(rest),
             "triangles": len(triangles), "obj": obj, "mesh": mesh}
        )

    rest_by_level = []
    for level in manifest["levels"]:
        with np.load(output_dir / level["mesh"], allow_pickle=False) as mesh:
            rest_by_level.append(mesh["rest"])

    finest = len(resolutions) - 1
    pairs = [("direct", level, finest) for level in range(finest + 1)]
    pairs += [("adjacent", level - 1, level) for level in range(1, finest + 1)]
    for kind, source, target in pairs:
        filename = f"P_{kind}_L{source}_to_L{target}.npz"
        path = output_dir / filename
        sparse.save_npz(path, prolongation(resolutions[source], resolutions[target]))
        stats = check_rest_prolongation(
            sparse.load_npz(path), rest_by_level[source], rest_by_level[target]
        )
        manifest["operators"].append(
            {"kind": kind, "source_level": source, "target_level": target,
             "file": filename, **stats}
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.output)
    for level in manifest["levels"]:
        cells = level["cells_per_side"]
        print(f"L{level['level']} {cells}x{cells}: {level['vertices']} vertices, "
              f"{level['triangles']} triangles")
    for op in manifest["operators"]:
        rows, cols = op["shape"]
        print(f"{op['file']}: {rows}x{cols}, {op['nnz']} nonzeros, "
              f"rest error {op['rest_error_inf_m']:.3e} m, "
              f"row sums [{op['row_sum_min']:.17g}, {op['row_sum_max']:.17g}]")


if __name__ == "__main__":
    main()
