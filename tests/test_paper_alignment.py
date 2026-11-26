"""Equation-level checks independent of the long folding demonstration."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import ipctk
from scipy.sparse import eye
from cloth.mesh import grid
from progressive.shell import Shell,local_hessian
from progressive.solver import IncrementalProblem,NewtonOptions,solve
from progressive.prolongation import ShellProlongation


class PaperAlignmentTests(unittest.TestCase):
    def test_cipc_signed_hinge_difference_is_not_periodically_wrapped(self):
        def shape(angle):return np.array([[0.,0,0],[1.,0,0],[0.,1.,0],[0.,-np.cos(angle),-np.sin(angle)]])
        rest=shape(1.6);x=shape(-2.);faces=np.array([[0,1,2],[1,0,3]])
        s=Shell(rest,faces);legacy=Shell(rest,faces,hinge_difference='periodic')
        k=30000*.0004**3/(24*(1-.3**2));height=(.5+.5)/3
        expected=k*(-2.-1.6)**2/height
        self.assertAlmostEqual(s.bending_local(x[s.hinges])[0][0],expected,places=14)
        self.assertGreater(expected,legacy.bending_local(x[s.hinges])[0][0])
        np.testing.assert_allclose(s.bending_hessian(x[s.hinges]),local_hessian(s.bending_local,x[s.hinges],1e-6),atol=1e-10,rtol=1e-6)

    def test_all_hierarchy_levels_preflight_thickness_not_only_intersection(self):
        from progressive.run import preflight_contacts
        faces=np.array([[0,1,2],[3,4,5]])
        x=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[.1,.1,.001],[.8,.1,.001],[.1,.8,.001]])
        fine=x.copy();fine[3:,2]=.0001
        hierarchy=SimpleNamespace(rests=[x,fine],faces=[faces,faces])
        args=SimpleNamespace(dhat=.0015,dmin=.0002,contact_stiffness=1e5,friction=1.,ccd='additive')
        with self.assertRaisesRegex(ValueError,'L1.*thickness'):
            preflight_contacts(hierarchy,[],[],args)

    def test_libshell_membrane_energy_in_fundamental_forms(self):
        # libshell 855a07b, NeoHookeanMaterial::stretchingEnergy, expressed
        # independently using edge Gram matrices rather than F = Ds Dm^-1.
        m=grid(2);r=m.rest.copy();r[:,2]=.09*np.cos(4*r[:,0]);s=Shell(r,m.faces)
        x=r+np.random.default_rng(8).normal(scale=.01,size=r.shape)
        def form(y):
            p=y[m.faces];edges=np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=2)
            return edges.transpose(0,2,1) @ edges
        a,abar=form(x),form(r);lnj=.5*np.log(np.linalg.det(a)/np.linalg.det(abar))
        # s.mu and s.lam already include thickness.
        reference=(s.mu*(np.trace(np.linalg.inv(abar) @ a,axis1=1,axis2=2)-2-2*lnj)+s.lam*lnj**2)*np.sqrt(np.linalg.det(abar))/4
        np.testing.assert_allclose(s.membrane_local(x[m.faces])[0],reference,atol=1e-14,rtol=1e-12)

    def test_cipc_thickness_shifted_barrier(self):
        mesh=ipctk.CollisionMesh(np.array([[0.,0,0],[0.,0,.01]]),np.empty((0,2),dtype=np.int32))
        c=ipctk.NormalCollisions();vv=ipctk.VertexVertexNormalCollision(0,1);vv.dmin=.002
        c.vv_collisions=[vv];barrier=ipctk.BarrierPotential(.03,1e4)
        for distance in (.003,.01,.025):
            x=np.array([[0.,0,0],[0.,0,distance]])
            shifted=distance**2-.002**2;activation=.03**2+2*.03*.002
            expected=-1e4*(shifted-activation)**2*np.log(shifted/activation)
            self.assertAlmostEqual(barrier(c,mesh,x),expected,places=12)

    def test_cipc_equation_5_and_volume_scaling(self):
        m=grid(2);kappa=1000.;thickness=.0004
        # Equal normalized strain gives equal energy density across limits.
        for limit,activation in ((1.1,1.05),(1.068,1.),(1.02,1.01)):
            s=Shell(m.rest,m.faces,strain_limit=limit,strain_activation=activation,strain_stiffness=kappa)
            stretch=activation+.4*(limit-activation)
            x=m.rest*np.array([stretch,1.,1.])
            expected=kappa*thickness*s.areas*(-.4**2*np.log(.6))
            np.testing.assert_allclose(s.strain_local(x[m.faces])[0],expected,rtol=1e-11)
            _,g,H=s.evaluate(x,True,elastic=False,strain=True,project=False)
            d=m.rest*np.array([1.,0.,0.]);eps=1e-7
            numerical=(s.evaluate(x+eps*d,elastic=False,strain=True)[1]-s.evaluate(x-eps*d,elastic=False,strain=True)[1])/(2*eps)
            np.testing.assert_allclose(H @ d.ravel(),numerical.ravel(),rtol=2e-7,atol=1e-6)

    def test_cipc_stiffness_requires_same_element_two_iterations(self):
        m=grid(2);s=Shell(m.rest,m.faces)
        self.assertFalse(s.adapt_strain_stiffness(None,np.array([1e-8,.01])))
        self.assertFalse(s.adapt_strain_stiffness(np.array([1e-8,.01]),np.array([.01,1e-8])))
        self.assertTrue(s.adapt_strain_stiffness(np.array([1e-8,.01]),np.array([1e-8,.01])))
        self.assertEqual(s.strain_stiffness,2000.)
        for _ in range(20):s.adapt_strain_stiffness(np.zeros(2),np.zeros(2))
        self.assertEqual(s.strain_stiffness,1e5)
        self.assertFalse(s.adapt_strain_stiffness(np.zeros(2),np.zeros(2)))
        legacy=Shell(m.rest,m.faces,strain_model='legacy',strain_stiffness=1e4)
        self.assertFalse(legacy.adapt_strain_stiffness(np.zeros(2),np.zeros(2)))

    def test_exact_elastic_hessians_without_numerical_perturbations(self):
        m=grid(3);rest=m.rest.copy();rest[:,2]=.08*np.sin(3*rest[:,0])
        s=Shell(rest,m.faces,thickness=.004)
        rng=np.random.default_rng(502);x=rest+rng.normal(scale=.015,size=rest.shape)
        for indices,gradient,exact in ((s.faces,s.membrane_local,s.membrane_hessian),
                                        (s.hinges,s.bending_local,s.bending_hessian)):
            p=x[indices];H=exact(p)
            numerical=local_hessian(gradient,p,2e-7)
            np.testing.assert_allclose(H,numerical,rtol=2e-5,atol=1e-6)
            np.testing.assert_allclose(H,H.transpose(0,2,1),rtol=1e-10,atol=1e-10)
            translation=np.tile([.3,-.2,.4],len(indices[0]))
            np.testing.assert_allclose(H @ translation,0,atol=1e-10)
        with patch('progressive.shell.local_hessian',side_effect=AssertionError('Finite differences entered production')):
            s.evaluate(x,True)

    def test_rms_decrement_records_paper_norm_and_max_control(self):
        m=grid(2);s=Shell(m.rest,m.faces,strain_limit=None)
        P=ShellProlongation(m.rest,m.rest,m.faces,eye(len(m.rest)))
        dt=.01;p=IncrementalProblem(s,s,P,dt,m.rest,m.rest,gravity=(1.,2.,-3.))
        _,report=solve(p,m.rest,NewtonOptions())
        r=report['trace'][0]
        self.assertAlmostEqual(r['velocity_decrement_rms'],dt*np.sqrt(14),places=10)
        self.assertAlmostEqual(r['velocity_decrement_max'],dt*3,places=10)
        self.assertEqual(r['velocity_decrement'],r['velocity_decrement_rms'])
        self.assertLess(r['linear_relative_residual'],1e-10)


if __name__=='__main__':unittest.main()
