"""3D reference shells with analytic membrane/strain and exact hinge Hessians.

Hinge Hessians use forward chain-rule differentiation of the analytic gradient.
Local Hessians are projected per element for the Newton preconditioner.
Material is surface Neo-Hookean plus rest-dihedral bending; no edge springs.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from .prolongation import frames


def assemble_hessian(indices, blocks, n, project=True):
    blocks = (blocks+blocks.transpose(0, 2, 1))/2
    if project and len(blocks):
        w, q = np.linalg.eigh(blocks)
        blocks = (q*np.maximum(w, 0)[:, None, :]) @ q.transpose(0, 2, 1)
    dofs = (3*indices[:, :, None]+np.arange(3)).reshape(len(indices), -1)
    m = dofs.shape[1]
    return coo_matrix((blocks.ravel(), (np.repeat(dofs, m, axis=1).ravel(),
                     np.tile(dofs, (1, m)).ravel())), shape=(3*n, 3*n)).tocsr()


def local_hessian(fun, positions, epsilon):
    count, nodes, _ = positions.shape
    result = np.empty((count, nodes*3, nodes*3))
    for i in range(nodes*3):
        d = np.zeros_like(positions)
        d[:, i//3, i%3] = epsilon
        gp, gm = fun(positions+d)[1], fun(positions-d)[1]
        result[:, :, i] = ((gp-gm)/(2*epsilon)).reshape(count, -1)
    return result


class Shell:
    def __init__(self, rest, faces, young=30000., poisson=.3, thickness=.0004, density=200., bending=None,
                 strain_limit=1.1, strain_activation=1.05, strain_stiffness=1e3,
                 strain_model='cipc',elastic_hessian='exact',hinge_difference='principal'):
        self.rest, self.faces = np.asarray(rest, float), np.asarray(faces, int)
        if young <= 0 or thickness <= 0 or density <= 0 or not -1 < poisson < .5:
            raise ValueError("Invalid shell material")
        if strain_limit is not None and not 1 <= strain_activation < strain_limit:
            raise ValueError("Require 1 <= strain activation < strain limit")
        _, _, self.inv_dm, self.areas = frames(self.rest, self.faces)
        self.mu = young*thickness/(2*(1+poisson))
        self.lam = young*thickness*poisson/(1-poisson**2)
        self.kb = young*thickness**3/(12*(1-poisson**2)) if bending is None else bending
        self.mass = np.zeros(len(rest))
        np.add.at(self.mass, self.faces.ravel(), np.repeat(self.areas*thickness*density/3, 3))
        if np.any(self.mass <= 0):
            raise ValueError("Isolated/zero-mass vertices")
        edges, hinges, weights = {}, [], []
        for i, (a, b, c) in enumerate(self.faces):
            for a, b, c in ((a,b,c), (b,c,a), (c,a,b)):
                key = tuple(sorted((a,b)))
                if key in edges:
                    aa, bb, cc, j = edges.pop(key)
                    hinges.append((aa,bb,cc,c))
                    weights.append(3*np.sum((rest[aa]-rest[bb])**2)/(self.areas[i]+self.areas[j]))
                else:
                    edges[key] = (a,b,c,i)
        self.hinges = np.asarray(hinges, int).reshape(-1, 4)
        self.weights = np.asarray(weights)
        self.rest_angles = self.hinge_geometry(rest[self.hinges])[0]
        self.strain_limit, self.strain_activation = strain_limit, strain_activation
        if strain_model not in ('cipc','legacy'):raise ValueError('Unknown strain model')
        if elastic_hessian not in ('exact','finite_difference'):raise ValueError('Unknown elastic Hessian')
        if strain_limit is not None and not 1<=strain_activation<strain_limit:raise ValueError('Invalid strain thresholds')
        self.strain_model,self.elastic_hessian=strain_model,elastic_hessian
        if hinge_difference not in ('principal','periodic'):raise ValueError('Unknown hinge angle difference')
        self.hinge_difference=hinge_difference
        self.strain_stiffness=float(strain_stiffness)
        self.strain_volume=thickness*self.areas
        self.set_strain_stiffness(strain_stiffness)
        self.epsilon = max(np.median(np.sqrt(self.areas))*1e-5, 1e-9)

    def set_strain_stiffness(self,value):
        if value<=0:raise ValueError('Strain stiffness must be positive')
        self.strain_stiffness=float(value)
        normalization=(self.strain_limit-self.strain_activation)**2 if self.strain_model=='cipc' and self.strain_limit is not None else 1.
        self.strain_scale=self.strain_stiffness*self.strain_volume/normalization

    def adapt_strain_stiffness(self,previous_gaps,current_gaps):
        """C-IPC Sec. 4.1: same element close to limit in consecutive iterates."""
        if self.strain_model!='cipc' or self.strain_limit is None or previous_gaps is None:return False
        threshold=1e-4*(self.strain_limit-self.strain_activation)
        if self.strain_stiffness<1e5 and np.any((previous_gaps<threshold)&(current_gaps<threshold)):
            self.set_strain_stiffness(min(2*self.strain_stiffness,1e5))
            return True
        return False

    def F(self, tri):
        return np.stack((tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]),axis=2) @ self.inv_dm

    def scatter_membrane(self, stress):
        gd = stress @ self.inv_dm.transpose(0,2,1)
        return np.stack((-gd.sum(axis=2), gd[:,:,0], gd[:,:,1]),axis=1)

    def membrane_local(self, tri):
        F = self.F(tri)
        C = F.transpose(0,2,1) @ F
        det = np.linalg.det(C)
        if np.any(det <= 1e-20):
            return np.full(len(tri), np.inf), np.zeros_like(tri)
        logj = .5*np.log(det)
        e = self.areas*(.5*self.mu*(np.trace(C,axis1=1,axis2=2)-2)-self.mu*logj+.5*self.lam*logj**2)
        stress = self.mu*F+(self.lam*logj-self.mu)[:,None,None]*(F @ np.linalg.inv(C))
        return e, self.scatter_membrane(stress*self.areas[:,None,None])

    def membrane_hessian(self,tri):
        F=self.F(tri);C=F.transpose(0,2,1) @ F;inv=np.linalg.inv(C)
        logj=.5*np.log(np.linalg.det(C));Q=F @ inv
        coeff=self.lam*logj-self.mu
        H=np.empty((len(tri),9,9))
        for i in range(9):
            delta=np.zeros_like(tri);delta[:,i//3,i%3]=1.
            dF=self.F(delta)
            dC=dF.transpose(0,2,1) @ F+F.transpose(0,2,1) @ dF
            dlog=np.einsum('nij,nij->n',Q,dF)
            dQ=dF @ inv-Q @ dC @ inv
            response=self.mu*dF+self.lam*dlog[:,None,None]*Q+coeff[:,None,None]*dQ
            H[:,:,i]=self.scatter_membrane(response*self.areas[:,None,None]).reshape(-1,9)
        return H

    def bending_hessian(self,p):
        from .differentials import hinge_gradient_hessian
        return hinge_gradient_hessian(p,self.rest_angles,self.kb*self.weights,self.hinge_difference=='periodic')

    @staticmethod
    def hinge_geometry(p):
        a,b,c,d = (p[:,i] for i in range(4))
        e,u,v = b-a,c-a,d-a
        le = np.linalg.norm(e,axis=1)
        eh = e/np.maximum(le[:,None],1e-30)
        raw1, raw2 = np.cross(e,u),np.cross(v,e)
        l1,l2 = np.linalg.norm(raw1,axis=1),np.linalg.norm(raw2,axis=1)
        n1,n2 = raw1/np.maximum(l1[:,None],1e-30),raw2/np.maximum(l2[:,None],1e-30)
        dot = lambda a,b: np.einsum('ij,ij->i',a,b)
        s,co = dot(eh,np.cross(n1,n2)),dot(n1,n2)
        return np.arctan2(s,co),(e,u,v,eh,le,n1,l1,n2,l2,s,co)

    def bending_local(self, p):
        theta,data = self.hinge_geometry(p)
        e,u,v,eh,le,n1,l1,n2,l2,s,co = data
        if np.any(np.minimum(l1,l2) < 1e-18):
            return np.full(len(p),np.inf),np.zeros_like(p)
        delta = theta-self.rest_angles
        if self.hinge_difference=='periodic':delta=np.arctan2(np.sin(delta),np.cos(delta))
        coeff = self.kb*self.weights
        scale = coeff*delta/np.maximum(s*s+co*co,1e-30)
        gs,gc = (scale*co)[:,None],(-scale*s)[:,None]
        dot = lambda a,b: np.einsum('ij,ij->i',a,b)
        geh = gs*np.cross(n1,n2)
        gn1,gn2 = gs*np.cross(n2,eh)+gc*n2,gs*np.cross(eh,n1)+gc*n1
        gr1 = (gn1-n1*dot(n1,gn1)[:,None])/l1[:,None]
        gr2 = (gn2-n2*dot(n2,gn2)[:,None])/l2[:,None]
        ge = (geh-eh*dot(eh,geh)[:,None])/le[:,None]+np.cross(u,gr1)+np.cross(gr2,v)
        gu,gv = np.cross(gr1,e),np.cross(e,gr2)
        return .5*coeff*delta**2, np.stack((-ge-gu-gv,ge,gu,gv),axis=1)

    def strain_local(self, tri):
        if self.strain_limit is None:
            return np.zeros(len(tri)),np.zeros_like(tri)
        left,s,right = np.linalg.svd(self.F(tri),full_matrices=False)
        gap = self.strain_limit-s
        if np.any(gap <= 0):
            return np.full(len(tri),np.inf),np.zeros_like(tri)
        width = self.strain_limit-self.strain_activation
        active = s > self.strain_activation
        diff = np.where(active,gap-width,0)
        log = np.where(active,np.log(gap/width),0)
        e = -self.strain_scale*np.sum(diff**2*log,axis=1)
        derivative = self.strain_scale[:,None]*(2*diff*log+diff**2/gap)
        stress = (left*derivative[:,None,:]) @ right
        return e,self.scatter_membrane(stress)

    def strain_hessian(self,tri):
        """Analytic spectral Hessian; finite differences fail near the hard limit.

        For W(F)=sum f(sigma_i), diagonal modes use f'', symmetric modes
        (f'_i-f'_j)/(sigma_i-sigma_j), skew modes (f'_i+f'_j)/(sigma_i+sigma_j),
        and out-of-plane modes f'_i/sigma_i. Assemble through the rest map.
        """
        F=self.F(tri)
        left,s,right=np.linalg.svd(F,full_matrices=True)
        gap=self.strain_limit-s
        if np.any(gap<=0): raise ValueError('Strain Hessian outside feasible domain')
        width=self.strain_limit-self.strain_activation
        active=s>self.strain_activation
        diff=np.where(active,gap-width,0.)
        log=np.where(active,np.log(gap/width),0.)
        first=self.strain_scale[:,None]*(2*diff*log+diff**2/gap)
        second=self.strain_scale[:,None]*(-2*log-4*diff/gap+diff**2/gap**2)
        difference=s[:,0]-s[:,1]
        symmetric=np.mean(second,axis=1)
        separated=np.abs(difference)>1e-10
        symmetric[separated]=(first[separated,0]-first[separated,1])/difference[separated]
        skew=first.sum(axis=1)/s.sum(axis=1)
        H=np.empty((len(tri),9,9))
        for i in range(9):
            delta=np.zeros_like(tri);delta[:,i//3,i%3]=1.
            D=left.transpose(0,2,1) @ self.F(delta) @ right.transpose(0,2,1)
            response=np.zeros_like(D)
            response[:,0,0]=second[:,0]*D[:,0,0]
            response[:,1,1]=second[:,1]*D[:,1,1]
            sym=.5*symmetric*(D[:,0,1]+D[:,1,0])
            anti=.5*skew*(D[:,0,1]-D[:,1,0])
            response[:,0,1]=sym+anti;response[:,1,0]=sym-anti
            response[:,2,:]=first/s*D[:,2,:]
            d_stress=left @ response @ right
            H[:,:,i]=self.scatter_membrane(d_stress).reshape(-1,9)
        return H

    def evaluate(self, x, hessian=False, elastic=True, strain=False, project=True):
        value,gradient = 0.,np.zeros_like(x)
        H = csr_matrix((x.size,x.size)) if hessian else None
        parts = []
        if elastic:
            parts.extend(((self.faces,self.membrane_local),(self.hinges,self.bending_local)))
        if strain and self.strain_limit is not None:
            parts.append((self.faces,self.strain_local))
        for indices, fun in parts:
            if not len(indices):
                continue
            p = x[indices]
            e,g = fun(p)
            value += float(e.sum())
            np.add.at(gradient,indices.ravel(),g.reshape(-1,3))
            if hessian and np.isfinite(value):
                if fun == self.strain_local:
                    # Convex singular-value barrier: analytic local Hessian is PSD.
                    H += assemble_hessian(indices,self.strain_hessian(p),len(x),False)
                else:
                    if self.elastic_hessian=='finite_difference':blocks=local_hessian(fun,p,self.epsilon)
                    else:blocks=self.membrane_hessian(p) if fun==self.membrane_local else self.bending_hessian(p)
                    H += assemble_hessian(indices,blocks,len(x),project)
        return value,gradient,H

    def feasible(self, x):
        s = np.linalg.svd(self.F(x[self.faces]),compute_uv=False)
        return bool(np.isfinite(s).all() and np.min(s) > 1e-8
                    and (self.strain_limit is None or np.max(s) < self.strain_limit))

    def max_stretch(self, x):
        return float(np.linalg.svd(self.F(x[self.faces]),compute_uv=False).max())
