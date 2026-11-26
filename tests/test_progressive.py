import unittest
import numpy as np
from scipy.sparse import csr_matrix,eye
from scipy.spatial.transform import Rotation
from cloth.mesh import grid,prolongation
from progressive.prolongation import ShellProlongation,Hierarchy
from progressive.shell import Shell
from progressive.integrators import predict
from progressive.solver import IncrementalProblem,NewtonOptions,solve
from progressive.contact import IPCContact
from progressive.affine import AffineObstacles
from progressive.metrics import consistency_metrics,continuity


class ProgressiveTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(83)

    def check_gradient(self,fun,x,tol=2e-5,epsilon=1e-6):
        e,g,*_=fun(x)
        for _ in range(4):
            d=self.rng.normal(size=x.shape);d/=np.linalg.norm(d)
            num=(fun(x+epsilon*d)[0]-fun(x-epsilon*d)[0])/(2*epsilon)
            self.assertLess(abs(num-np.sum(g*d))/max(1,abs(num)),tol)

    def test_psq_rest_rigid_and_tangent_rotation(self):
        coarse=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        U=csr_matrix([[.2,.3,.5],[.4,.3,.3]])
        fine=U @ coarse+[[.02,-.01,.1],[-.03,.04,-.08]]
        P=ShellProlongation(coarse,fine,np.array([[0,1,2]]),U)
        np.testing.assert_allclose(P.position(coarse),fine,atol=1e-14)
        Q=Rotation.from_rotvec([.4,-.2,.8]).as_matrix()
        np.testing.assert_allclose(P.position(coarse @ Q.T+[2,3,4]),fine @ Q.T+[2,3,4],atol=1e-13)
        # In-plane offset rotates but does not scale with membrane stretching.
        stretched=coarse*np.array([1.5,1.2,1.])
        np.testing.assert_allclose(P.position(stretched)-U @ stretched,fine-U @ coarse,atol=1e-13)

    def test_nonlinear_offset_is_not_added_to_velocity(self):
        coarse=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        U=csr_matrix([[.2,.3,.5],[.4,.3,.3]])
        fine=U @ coarse+[[.02,-.01,.1],[-.03,.04,-.08]]
        P=ShellProlongation(coarse,fine,np.array([[0,1,2]]),U)
        velocity=np.tile([.3,.2,-.1],(3,1));dt=.02
        tracks=[np.stack((coarse,coarse+dt*velocity)),np.stack((fine,fine+dt*(U @ velocity)))]
        result=predict('velpro',1,1,tracks,[P],dt,[velocity,U @ velocity])
        np.testing.assert_allclose(result,fine+2*dt*(U @ velocity),atol=1e-14)

    def test_psq_velocity_is_position_directional_derivative(self):
        a=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        U=csr_matrix([[.2,.3,.5],[.4,.3,.3]])
        p=ShellProlongation(a,U@a+[[.02,-.01,.1],[-.03,.04,-.08]],np.array([[0,1,2]]),U)
        v=self.rng.normal(size=a.shape)
        for x in [a,a+self.rng.normal(scale=.15,size=a.shape)]:
            h=1.e-6
            numerical=(p.position(x+h*v)-p.position(x-h*v))/(2*h)
            np.testing.assert_allclose(p.velocity(x,v),numerical,rtol=1e-8,atol=1e-9)
            angular=np.array([.2,-.5,.8])
            np.testing.assert_allclose(p.velocity(x,np.cross(angular,x)),
                                       np.cross(angular,p.position(x)),atol=1e-12)

    def test_frozen_nonlinear_proxy_gradient(self):
        a,b=grid(2),grid(4)
        coarse=a.rest.copy();fine=b.rest.copy()
        coarse[:,2]=.04*np.sin(coarse[:,0]*np.pi)
        fine[:,2]=.04*np.sin(fine[:,0]*np.pi)
        P=ShellProlongation(coarse,fine,a.faces,prolongation(a,b))
        s0,s1=Shell(coarse,a.faces),Shell(fine,b.faces)
        x=coarse+self.rng.normal(scale=.002,size=coarse.shape)
        p=IncrementalProblem(s0,s1,P,.02,x,x)
        frozen=P.offset(x)
        self.assertFalse(P.zero_offset)
        self.check_gradient(lambda y:p.evaluate(y,frozen),x)

    def test_nonlinear_psq_search_decreases_actual_energy(self):
        a,b=grid(2),grid(4)
        coarse=a.rest.copy();fine=b.rest.copy()
        coarse[:,2]=.04*np.sin(coarse[:,0]*np.pi)
        fine[:,2]=.04*np.sin(fine[:,0]*np.pi)
        P=ShellProlongation(coarse,fine,a.faces,prolongation(a,b))
        p=IncrementalProblem(Shell(coarse,a.faces),Shell(fine,b.faces),P,.02,coarse,coarse)
        fixed=np.broadcast_to(np.isclose(coarse[:,0],-.5)[:,None],coarse.shape)
        x,r=solve(p,coarse,NewtonOptions(velocity_tolerance=1e-5),fixed)
        self.assertTrue(r['converged'])
        self.assertGreater(np.linalg.norm(P.offset(x)-P.offset(coarse)),1e-8)
        self.assertLess(r['energy'],r['trace'][0]['energy'])
        for before,after in zip(r['trace'],r['trace'][1:]):
            self.assertLessEqual(after['energy'],before['energy']+1e-12)
            self.assertAlmostEqual(before['candidate_energy'],after['energy'],places=12)

    def test_collision_hessian_and_affine_restriction(self):
        x=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[.1,.1,.01],[.8,.1,.01],[.1,.8,.01]])
        faces=np.array([[0,1,2],[3,4,5]])
        contact=IPCContact(x,faces,3,dhat=.03,dmin=.002,stiffness=1e4,friction=0)
        value,gradient,H=contact.evaluate(x,True)
        self.assertTrue(np.isfinite(H.data).all())
        self.assertGreater(np.linalg.eigvalsh(H.toarray()).min(),-1e-7)
        scene=AffineObstacles(x[:3],faces[:1],[('jaw',x[3:],np.array([[0,1,2]]),np.eye(4))])
        def composed(q):
            e,g,_=contact.evaluate(scene.positions(q))
            return e,scene.B.T @ g
        self.check_gradient(composed,scene.q0,tol=2e-4,epsilon=1e-7)

    def test_failed_newton_is_marked_unconverged(self):
        m=grid(2);s=Shell(m.rest,m.faces)
        P=ShellProlongation(m.rest,m.rest,m.faces,eye(len(m.rest)))
        p=IncrementalProblem(s,s,P,.02,m.rest,m.rest)
        _,r=solve(p,m.rest,NewtonOptions(max_iterations=1))
        self.assertFalse(r['converged']);self.assertEqual(r['reason'],'iteration_limit')
        self.assertFalse(r['accepted'])

    def test_preview_stop_is_not_newton_convergence(self):
        meshes=[grid(2),grid(4)]
        rests=[np.column_stack((m.rest[:,0]*.3,.04*np.sin(np.pi*m.rest[:,0]),.25+m.rest[:,1]*.3)) for m in meshes]
        P=ShellProlongation(rests[0],rests[1],meshes[0].faces,prolongation(*meshes))
        # Retain the historical fixture that actually reaches a non-descent
        # preview. New barrier scaling need not stall at the same physical time.
        shells=[Shell(x,m.faces,strain_model='legacy',strain_stiffness=1e4,elastic_hessian='finite_difference') for x,m in zip(rests,meshes)]
        fixed=np.broadcast_to(np.isclose(rests[0][:,0],-.15)[:,None],rests[0].shape)
        previous=x=rests[0].copy()
        stopped=False
        for step in range(8):
            p=IncrementalProblem(shells[0],shells[1],P,.01,2*x-previous,x,
                                 gravity=(0.,8*np.sin((step+1)*.01*5),-9.81))
            next_x,r=solve(p,x,NewtonOptions(allow_preview_stall=True,convergence_norm='max'),fixed)
            self.assertTrue(r['accepted'])
            if r['preview_stop']:
                self.assertFalse(r['converged'])
                # The same input cannot be mislabeled accepted by a strict solve.
                _,strict=solve(p,x,NewtonOptions(convergence_norm='max'),fixed)
                self.assertFalse(strict['accepted']);self.assertFalse(strict['converged'])
                stopped=True
                break
            previous,x=x,next_x
        self.assertTrue(stopped)

    def test_3d_shell_gradient_and_hessian(self):
        m=grid(3);rest=m.rest.copy();rest[:,2]=.05*np.sin(rest[:,0]*3)
        s=Shell(rest,m.faces,thickness=.003)
        x=rest+self.rng.normal(scale=.004,size=rest.shape)
        self.check_gradient(lambda y:s.evaluate(y),x)
        _,g,H=s.evaluate(x,True,project=False)
        d=self.rng.normal(size=x.shape);d/=np.linalg.norm(d)
        eps=2e-6
        numerical=(s.evaluate(x+eps*d)[1]-s.evaluate(x-eps*d)[1])/(2*eps)
        np.testing.assert_allclose(H @ d.ravel(),numerical.ravel(),atol=3e-4,rtol=1e-4)
        _,_,projected=s.evaluate(x,True)
        self.assertGreater(np.linalg.eigvalsh(projected.toarray()).min(),-1e-7)

    def test_strain_barrier(self):
        m=grid(2);s=Shell(m.rest,m.faces)
        x=m.rest*1.075
        self.check_gradient(lambda y:s.evaluate(y,elastic=False,strain=True),x,epsilon=1e-7)
        self.assertTrue(s.feasible(x))
        self.assertFalse(s.feasible(m.rest*1.101))
        self.assertFalse(np.isfinite(s.evaluate(m.rest*1.101,elastic=False,strain=True)[0]))

    def test_analytic_strain_hessian_near_limit(self):
        m=grid(2);s=Shell(m.rest,m.faces)
        for scale in ([1.075,1.065,1.],[1.09999,1.08,1.],[1.075,1.075,1.]):
            x=m.rest*np.asarray(scale)
            _,_,H=s.evaluate(x,True,elastic=False,strain=True)
            d=self.rng.normal(size=x.shape);d/=np.linalg.norm(d)
            eps=1e-9
            numeric=(s.evaluate(x+eps*d,elastic=False,strain=True)[1]-s.evaluate(x-eps*d,elastic=False,strain=True)[1])/(2*eps)
            np.testing.assert_allclose(H @ d.ravel(),numeric.ravel(),rtol=3e-5,atol=1e-5)

    def test_shared_proxy_chain_gradient(self):
        meshes=[grid(2),grid(4)]
        hierarchy=Hierarchy([m.rest for m in meshes],[m.faces for m in meshes],[prolongation(*meshes)])
        shells=[Shell(m.rest,m.faces) for m in meshes]
        x=meshes[0].rest+self.rng.normal(scale=.003,size=meshes[0].rest.shape)
        p=IncrementalProblem(shells[0],shells[1],hierarchy.to_finest[0],.02,x,x,gravity=(0,0,0))
        offset=hierarchy.to_finest[0].offset(x)
        self.check_gradient(lambda y:p.evaluate(y,offset),x)
        _,_,H=p.evaluate(x,offset,True,"proxy")
        self.assertGreater(np.linalg.eigvalsh(H.toarray()).min(),0)

    def test_newton_free_fall_and_consistency_closed_form(self):
        m=grid(2);s=Shell(m.rest,m.faces)
        P=ShellProlongation(m.rest,m.rest,m.faces,eye(len(m.rest)))
        x=m.rest+[0,0,1];dt=.02;w=30
        target=x+[.1,0,.2]
        p=IncrementalProblem(s,s,P,dt,x,x,target=target,weight=w)
        result,report=solve(p,x,NewtonOptions(velocity_tolerance=1e-8))
        expected=(x+dt**2*np.array([0,0,-9.81])+2*w*dt**2*target)/(1+2*w*dt**2)
        np.testing.assert_allclose(result,expected,atol=1e-9)
        self.assertTrue(report['converged'])

    def test_integrators_constant_velocity_and_nonlinear_offsets(self):
        a=grid(2);b=grid(4);c=grid(8)
        h=Hierarchy([m.rest for m in (a,b,c)],[m.faces for m in (a,b,c)],
                    [prolongation(a,b),prolongation(b,c)])
        v=[np.tile([.2,-.1,.3],(len(m.rest),1)) for m in (a,b,c)]
        tracks=[np.stack([m.rest+i*.02*vk for i in range(4)]) for m,vk in zip((a,b,c),v)]
        for method in ('direct','semi','full','velpro'):
            for l in range(3):
                for t in range(3):
                    np.testing.assert_allclose(predict(method,l,t,tracks,h.adjacent,.02,v),tracks[l][t+1],atol=1e-13)

    def test_ipc_ccd_friction_and_derivatives(self):
        x=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[.1,.1,.01],[.8,.1,.01],[.1,.8,.01]])
        f=np.array([[0,1,2],[3,4,5]])
        contact=IPCContact(x,f,3,dhat=.03,dmin=.002,stiffness=1e4,friction=.5)
        contact.validate(x);contact.begin_step(x,.01)
        trial=x.copy();trial[:3,0]+=.001
        self.check_gradient(lambda y:contact.evaluate(y),trial,tol=1e-4,epsilon=1e-7)
        self.assertGreater(contact.friction(contact.tangential,contact.mesh,trial-x),0)
        end=x.copy();end[:3,2]=.03
        alpha=contact.max_step(x,end)
        self.assertGreater(alpha,0);self.assertLess(alpha,1)
        contact.validate(x+alpha*(end-x))

    def test_affine_body_gradients_and_no_cloth_targets(self):
        m=grid(2);v=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        scene=AffineObstacles(m.rest,m.faces,[('jaw',v,np.array([[0,1,2]]),np.eye(4))])
        x=scene.q0+self.rng.normal(scale=.001,size=scene.q0.shape)
        self.check_gradient(lambda y:scene.evaluate(y),x,tol=1e-4)
        self.assertEqual(np.max(np.abs(scene.evaluate(x)[1][:len(m.rest)])),0)

    def test_restriction_and_undefined_continuity(self):
        a,b=grid(2),grid(4);U=prolongation(a,b)
        p=ShellProlongation(a.rest,b.rest,a.faces,U)
        metric=consistency_metrics(a.rest,U @ a.rest,p,np.ones(len(a.rest)),np.ones(len(b.rest)))
        self.assertLess(metric['coarse_rms_m'],1e-14)
        c=continuity(a.rest,a.rest,a.rest,a.rest,np.zeros_like(a.rest),np.ones(len(a.rest)),.01)
        self.assertIsNone(c['literal_squared_ratio'])


if __name__=='__main__': unittest.main()
