"""IPC Toolkit contact energies, lagged friction, and conservative linear CCD.

Contact lives on the CURRENT level, never on the elastic proxy surface.
Friction potential takes displacements with eps_v*dt, equivalent to dt*D(v).
"""
import numpy as np
import ipctk


class IPCContact:
    def __init__(self, rest, faces, cloth_vertices, dhat=.0015, dmin=.0002,
                 stiffness=1e7, friction=.5, eps_velocity=.001,ccd='additive'):
        if dmin < 0 or dhat <= 0 or stiffness <= 0 or friction < 0:
            raise ValueError("Invalid contact parameters")
        self.dhat, self.dmin, self.stiffness = dhat, dmin, stiffness
        self.mu, self.eps_velocity = friction, eps_velocity
        if ccd not in ('additive','tight'):raise ValueError('Unknown CCD method')
        self.ccd=ipctk.AdditiveCCD() if ccd=='additive' else ipctk.TightInclusionCCD()
        edges = np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
        self.mesh = ipctk.CollisionMesh(rest,edges.astype(np.int32),faces.astype(np.int32))
        # Cloth self-contact and cloth/obstacle contact. Ignore obstacle/obstacle
        # pairs, including tessellations within the same driven rigid body.
        self.mesh.can_collide = ipctk.make_static_obstacle_filter(cloth_vertices)
        self.barrier = ipctk.BarrierPotential(dhat,stiffness)
        self.friction = None
        self.tangential = ipctk.TangentialCollisions()
        self.previous = rest.copy()

    def normal(self,x):
        collisions = ipctk.NormalCollisions()
        # C-IPC uses the original IPC collision-set formulation. Make this
        # explicit rather than depending on a library-version default.
        collisions.collision_set_type=ipctk.NormalCollisions.IPC
        collisions.use_area_weighting=False
        collisions.build(self.mesh,x,self.dhat,self.dmin)
        return collisions

    def begin_step(self,previous,dt):
        self.previous = previous.copy()
        self.friction = ipctk.FrictionPotential(self.eps_velocity*dt)
        self.tangential = ipctk.TangentialCollisions()
        self.tangential.build(self.mesh,previous,self.normal(previous),self.barrier,self.mu)

    def evaluate(self,x,hessian=False):
        collisions = self.normal(x)
        value = self.barrier(collisions,self.mesh,x)
        gradient = self.barrier.gradient(collisions,self.mesh,x).reshape(-1,3)
        H = self.barrier.hessian(collisions,self.mesh,x,ipctk.PSDProjectionMethod.CLAMP) if hessian else None
        if self.friction is not None and self.mu:
            delta = x-self.previous
            value += self.friction(self.tangential,self.mesh,delta)
            gradient += self.friction.gradient(self.tangential,self.mesh,delta).reshape(-1,3)
            if hessian:
                H += self.friction.hessian(self.tangential,self.mesh,delta,ipctk.PSDProjectionMethod.CLAMP)
        return float(value),gradient,H

    def max_step(self,start,end):
        return float(ipctk.compute_collision_free_stepsize(self.mesh,start,end,self.dmin,narrow_phase_ccd=self.ccd))

    def validate(self,x):
        if ipctk.has_intersections(self.mesh,x):
            raise ValueError("Initial collision mesh has intersections")
        c = self.normal(x)
        if not np.isfinite(self.barrier(c,self.mesh,x)):
            raise ValueError("Initial contact state violates the thickness barrier")
