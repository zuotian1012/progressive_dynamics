"""Pack a Step 2 fine rest mesh and direct XYZ prolongation for libuipc."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def prepare(directory, level, output):
    manifest = json.loads((directory / "manifest.json").read_text())
    levels = manifest["levels"]
    finest = len(levels) - 1
    if not 0 <= level < finest:
        raise ValueError(f"Proxy level must be between 0 and {finest - 1}")
    operator = next(
        entry for entry in manifest["operators"]
        if entry["kind"] == "direct" and entry["source_level"] == level
        and entry["target_level"] == finest
    )
    with np.load(directory / levels[finest]["mesh"], allow_pickle=False) as mesh:
        rest, triangles = mesh["rest"], mesh["triangles"]
    with np.load(directory / levels[level]["mesh"], allow_pickle=False) as mesh:
        coarse_rest = mesh["rest"]
    matrix = sparse.load_npz(directory / operator["file"]).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    scalar = matrix[::3, ::3].tocsr()
    expected = sparse.kron(scalar, sparse.eye(3, format="csr"), format="csr")
    if matrix.shape != (rest.size, coarse_rest.size) or (matrix != expected).nnz:
        raise ValueError("Expected a vertex-major XYZ prolongation")
    if not np.all(np.isfinite(scalar.data) & (scalar.data > 0)):
        raise ValueError("Barycentric weights must be finite and positive")
    if not np.allclose(np.asarray(scalar.sum(axis=1)), 1.0, rtol=0, atol=2 * np.finfo(np.float64).eps):
        raise ValueError("Barycentric rows must sum to one within floating-point rounding")
    if np.max(np.abs(scalar @ coarse_rest - rest)) >= 1e-12:
        raise ValueError("Prolongation does not reconstruct the fine rest pose")

    row_lengths = np.diff(scalar.indptr)
    if np.any(row_lengths > 3):
        raise ValueError("Expected at most three nonzeros per vertex row")
    columns = np.zeros((len(rest), 3), dtype=np.int32)
    weights = np.zeros_like(columns, dtype=np.float64)
    for row, offset in enumerate(scalar.indptr[:-1]):
        length = row_lengths[row]
        columns[row, :length] = scalar.indices[offset:offset + length]
        weights[row, :length] = scalar.data[offset:offset + length]
    asset = {
        "coarse_vertices": len(coarse_rest),
        "rest": rest.tolist(),
        "triangles": triangles.tolist(),
        "columns": columns.tolist(),
        "weights": weights.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asset, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.hierarchy, args.level, args.output)


if __name__ == "__main__":
    main()
