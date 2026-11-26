from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from tools.assets.generate_pd_square_cloth import generate


class SquareClothTests(unittest.TestCase):
    def test_saved_meshes_and_prolongations(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = generate(output)
            meshes = []
            for level, cells in zip(manifest["levels"], (8, 16, 32, 64), strict=True):
                with np.load(output / level["mesh"], allow_pickle=False) as mesh:
                    rest, faces = mesh["rest"], mesh["triangles"]
                self.assertEqual(rest.shape, ((cells + 1)**2, 3))
                self.assertEqual(faces.shape, (2 * cells**2, 3))
                np.testing.assert_array_equal(rest[:, 2], 0.93)
                np.testing.assert_allclose(np.ptp(rest[:, :2], axis=0), 0.64)
                a, b, c = rest[faces].transpose(1, 0, 2)
                self.assertTrue(np.all(np.cross(b - a, c - a)[:, 2] > 0))
                meshes.append(rest)

                obj_lines = (output / level["obj"]).read_text().splitlines()
                obj_rest = np.array([line.split()[1:] for line in obj_lines
                                     if line.startswith("v ")], dtype=np.float64)
                obj_faces = np.array([line.split()[1:] for line in obj_lines
                                      if line.startswith("f ")], dtype=int) - 1
                world_rest = obj_rest[:, [0, 2, 1]] * [0.001, 0.001, -0.001]
                world_rest += [0.667, 0.015, 0.93]
                np.testing.assert_allclose(world_rest, rest, rtol=0, atol=1e-15)
                np.testing.assert_array_equal(obj_faces, faces)

            operators = {}
            for op in manifest["operators"]:
                source, target = op["source_level"], op["target_level"]
                with self.subTest(kind=op["kind"], source=source, target=target):
                    matrix = sparse.load_npz(output / op["file"])
                    self.assertTrue(sparse.isspmatrix_csr(matrix))
                    self.assertEqual(matrix.shape, (meshes[target].size, meshes[source].size))
                    self.assertTrue(np.all(matrix.data > 0))
                    projected = matrix @ meshes[source].ravel()
                    self.assertLess(np.max(np.abs(projected - meshes[target].ravel())), 1e-12)
                    np.testing.assert_array_equal(np.asarray(matrix.sum(axis=1)), 1.0)
                    blocks = matrix.tocoo()
                    np.testing.assert_array_equal(blocks.row % 3, blocks.col % 3)
                    scalar = matrix[::3, ::3]
                    for axis in (1, 2):
                        self.assertEqual((scalar != matrix[axis::3, axis::3]).nnz, 0)
                    operators[op["kind"], source, target] = matrix

            identity = sparse.eye(meshes[-1].size, format="csr")
            self.assertEqual((operators["direct", 3, 3] != identity).nnz, 0)
            composed = identity
            for source in (2, 1, 0):
                composed = composed @ operators["adjacent", source, source + 1]
                self.assertEqual((operators["direct", source, 3] != composed).nnz, 0)


if __name__ == "__main__":
    unittest.main()
