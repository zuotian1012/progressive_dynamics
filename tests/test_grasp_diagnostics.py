import unittest
from types import SimpleNamespace
import numpy as np
from progressive.run import robot_targets
from progressive.affine import AffineObstacles
from progressive.contact import IPCContact
from progressive.diagnostics import body_contact_forces,nearest_material_point
from progressive.runtime import numerical_threads


class GraspDiagnosticsTests(unittest.TestCase):
    def test_material_point_follows_its_original_triangle(self):
        x=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        f=np.array([[0,1,2]])
        face,w,d=nearest_material_point(x,f,np.array([.2,.3,.01]))
        self.assertEqual(face,0);self.assertAlmostEqual(d,.01)
        np.testing.assert_allclose(w,[.5,.2,.3],atol=1e-14)
        deformed=x.copy();deformed[2,2]=.1
        np.testing.assert_allclose(w @ deformed[f[face]],[.2,.3,.03],atol=1e-14)

    def test_hold_starts_after_reaching_target_without_rewind(self):
        robot=SimpleNamespace(q=np.arange(6,dtype=float)[:,None],poses=lambda q:q.copy())
        target=robot_targets(robot,1,2,2)
        np.testing.assert_array_equal([target(k)[0] for k in range(10)],[0,0,0,.5,1,1.5,2,2,2,2])

    def test_body_reaction_opposes_cloth_contact_force(self):
        x=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        v=np.array([[.1,.1,.01],[.8,.1,.01],[.1,.8,.01]])
        f=np.array([[0,1,2]])
        embedding=AffineObstacles(x,f,[('jaw',v,f,np.eye(4))])
        before=embedding.positions(embedding.q0)
        now=before.copy();now[:3,0]+=.001
        with numerical_threads(1):
            contact=IPCContact(before,embedding.faces,3,dhat=.03,dmin=.002,stiffness=1e4,friction=.5)
            result=body_contact_forces(contact,embedding,before,now,.01)['jaw']
            n=contact.barrier.gradient(contact.normal(now),contact.mesh,now).reshape(-1,3)
            g=contact.friction.gradient(contact.tangential,contact.mesh,now-before).reshape(-1,3)
            np.testing.assert_allclose(result['normal_force_N'],n[:3].sum(axis=0),atol=1e-12)
            np.testing.assert_allclose(result['friction_force_N'],g[:3].sum(axis=0),atol=1e-12)
            self.assertGreater(result['normal_force_N'][2],0)
            self.assertGreater(result['friction_force_N'][0],0)


if __name__=='__main__':unittest.main()
