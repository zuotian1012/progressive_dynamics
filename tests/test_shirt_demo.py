import unittest
import importlib.util
import numpy as np
from scipy.spatial.transform import Rotation
from shirt_demo.assets import subdivide, Robot, BUNDLE


class ShirtInfrastructureTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('uipc'), 'Native IPC is optional')
    def test_native_mesh_preserves_fortran_layout_coordinates(self):
        from shirt_demo.simulate import trimesh
        v=np.asfortranarray([[.1,.2,.8],[.2,.2,.8],[.2,.3,.9]])
        f=np.array([[0,1,2]])
        mesh=trimesh(v,f)
        np.testing.assert_array_equal(np.asarray(mesh.positions().view()).reshape(-1,3),v)
        np.testing.assert_array_equal(np.asarray(mesh.triangles().topo().view()).reshape(-1,3),f)

    def test_subdivision_affine_precision_and_topology(self):
        v = np.array([[0.,0,0],[1,0,.2],[1,1,0],[0,1,-.1]])
        f = np.array([[0,1,2],[0,2,3]])
        fine, faces, p = subdivide(v,f)
        self.assertEqual(len(fine),9)
        self.assertEqual(len(faces),8)
        np.testing.assert_allclose(np.asarray(p.sum(axis=1)).ravel(),1)
        rotation=Rotation.from_rotvec([.2,.3,-.4]).as_matrix()
        np.testing.assert_allclose(p @ (v @ rotation.T + [1,2,3]), fine @ rotation.T + [1,2,3])
        self.assertGreaterEqual(p.data.min(),0)

    @unittest.skipUnless((BUNDLE / "controls/trajectory.npz").is_file(), "Reference assets missing")
    def test_fk_matches_reference_command_tcp(self):
        # Independent target from the source first-fold plan, not our FK code.
        robot=Robot()
        poses=robot.poses(robot.q[0])
        tcp=np.array([.155,.001786,.014])
        targets={"left_link16":[.4122331580453282,.2580795110032265,1.0549109116875564],
                 "right_link26":[.4078141188053146,-.24838986585543543,1.0721061951122544]}
        for name,expected in targets.items():
            p=poses[name]
            np.testing.assert_allclose(p[:3,:3]@tcp+p[:3,3], expected,atol=1e-10)


if __name__ == "__main__":
    unittest.main()
