"""Barrier-filtered inexact projected Newton, with horizontal warm starts.

PSQ offsets are refreshed each Newton iteration and held fixed in that
iteration's derivatives. Line search evaluates the full nonlinear potential
by default; frozen-offset model search is an explicitly labeled alternative.
Default preconditioner uses CURRENT
level elastic Hessian, as PSQ 4.5; optional proxy Hessian is an ablation.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.sparse import diags, eye, kron, csr_matrix
from scipy.sparse.linalg import spsolve
from .integrators import consistency


@dataclass
class NewtonOptions:
    max_iterations: int = 100
    velocity_tolerance: float = 1e-3
    max_linesearch: int = 40
    armijo: float = 1e-4
    hessian: str = "coarse"
    offset_linesearch: str = "nonlinear"
    allow_preview_stall: bool = False
    preview_min_alpha: float = 1e-8
    convergence_norm: str = 'rms'


class IncrementalProblem:
    def __init__(self, shell, finest, transfer, dt, xhat, previous, gravity=(0,0,-9.81),
                 target=None, weight=0., contact=None, embedding=None, auxiliary=None):
        self.shell,self.finest,self.transfer = shell,finest,transfer
        self.dt,self.xhat,self.previous = dt,xhat,previous
        self.target,self.weight = target,weight
        self.gravity = np.asarray(gravity,float)
        self.contact,self.auxiliary = contact,auxiliary
        self.n = len(shell.rest)
        self.embedding = embedding
        if contact:
            contact.begin_step(self.world(previous),dt)

    def world(self,x):
        return x if self.embedding is None else self.embedding.positions(x)

    def evaluate(self,x,offset,hessian=False,hessian_mode="coarse"):
        cloth = x[:self.n]
        y = self.transfer.U @ cloth+offset
        e,gfine,Hfine = self.finest.evaluate(y,hessian and hessian_mode == "proxy")
        g = np.zeros_like(x)
        g[:self.n] = self.transfer.U.T @ gfine
        H = csr_matrix((x.size,x.size)) if hessian else None
        if hessian:
            if hessian_mode == "proxy":
                Hcloth = self.transfer.U3.T @ Hfine @ self.transfer.U3
            else:
                Hcloth = self.shell.evaluate(cloth,True)[2]
            # Zero padding includes optional driven affine bodies.
            H = _pad(Hcloth,x.size)
        es,gs,Hs = self.shell.evaluate(cloth,hessian,elastic=False,strain=True)
        e += es
        g[:self.n] += gs
        if hessian: H += _pad(Hs,x.size)
        mass = self.shell.mass[:,None]
        dx = cloth-self.xhat
        e += float(np.sum(mass*dx**2)/(2*self.dt**2)-np.sum(mass*cloth*self.gravity))
        g[:self.n] += mass*(dx/self.dt**2-self.gravity)
        diag = np.zeros_like(x)
        diag[:self.n] = mass/self.dt**2
        if self.target is not None and self.weight:
            ec,gc = consistency(cloth,self.target,self.shell.mass,self.weight)
            e += ec
            g[:self.n] += gc
            diag[:self.n] += 2*self.weight*mass
        if hessian: H += diags(diag.ravel())
        if self.contact:
            ec,gc,Hc = self.contact.evaluate(self.world(x),hessian)
            e += ec
            if self.embedding is None:
                g += gc
                if hessian: H += Hc
            else:
                g += self.embedding.B.T @ gc
                if hessian: H += self.embedding.B3.T @ Hc @ self.embedding.B3
        if self.auxiliary:
            ea,ga,Ha = self.auxiliary.evaluate(x,hessian)
            e += ea
            g += ga
            if hessian: H += Ha
        return float(e),g,H

    def feasible(self,x,offset=None):
        if not self.shell.feasible(x[:self.n]):
            return False
        if offset is not None:
            # Proxy elastics require non-degenerate triangles, but strain bounds
            # belong only to the current level, NOT the prolonged surface.
            F = self.finest.F((self.transfer.U @ x[:self.n]+offset)[self.finest.faces])
            if np.min(np.linalg.svd(F,compute_uv=False)) < 1e-8:
                return False
        return True

    def safe_step(self,x,direction):
        alpha = 1.
        if self.contact:
            alpha = self.contact.max_step(self.world(x),self.world(x+direction))
        # Weyl bound keeps membrane gradients full rank along the entire
        # linear search segment, rather than only checking its endpoint.
        for shell,start,delta in ((self.shell,x[:self.n],direction[:self.n]),
                                  (self.finest,self.transfer.position(x[:self.n]),
                                   self.transfer.U @ direction[:self.n])):
            F=shell.F(start[shell.faces])
            dF=shell.F(delta[shell.faces])
            singular=np.linalg.svd(F,compute_uv=False)[:,-1]
            speed=np.linalg.norm(dF,axis=(1,2))
            moving=speed > 1e-14
            if np.any(moving):
                alpha=min(alpha,float(np.min(.9*singular[moving]/speed[moving])))
        return alpha


def _pad(H,size):
    if H.shape[0] == size: return H
    c = H.tocoo()
    from scipy.sparse import coo_matrix
    return coo_matrix((c.data,(c.row,c.col)),shape=(size,size)).tocsr()


def solve(problem,initial,options=None,fixed=None):
    opt = options or NewtonOptions()
    if opt.hessian not in ("coarse","proxy"):
        raise ValueError("Unknown Hessian mode")
    if opt.offset_linesearch not in ("nonlinear","frozen"):
        raise ValueError("Unknown offset line-search mode")
    if opt.convergence_norm not in ('rms','max'):raise ValueError('Unknown convergence norm')
    x = initial.copy()
    fixed = np.zeros(x.shape,bool) if fixed is None else np.broadcast_to(fixed,x.shape)
    free = np.flatnonzero(~fixed.ravel())
    if not len(free): raise ValueError("No free DOFs")
    if not problem.feasible(x): raise ValueError("Infeasible horizontal warm start")
    start = time.perf_counter()
    records=[]
    converged=False
    reason="iteration_limit"
    previous_strain_gaps=None
    for iteration in range(opt.max_iterations):
        offset = problem.transfer.offset(x[:problem.n])
        value,g,H = problem.evaluate(x,offset,True,opt.hessian)
        if not np.isfinite(value) or not np.isfinite(g).all():
            raise FloatingPointError("Non-finite incremental potential or gradient")
        A = H[free][:,free].tocsc()
        direction=np.zeros(x.size)
        direction[free] = spsolve(A,-g.ravel()[free])
        direction=direction.reshape(x.shape)
        if not np.isfinite(direction).all():
            raise FloatingPointError("Non-finite Newton solve")
        maximum = float(np.max(np.abs(direction[:problem.n]))/problem.dt)
        active_nodes=np.count_nonzero(np.any(~fixed[:problem.n],axis=1))
        rms=float(np.linalg.norm(direction[:problem.n])/np.sqrt(max(active_nodes,1))/problem.dt)
        decrement = rms if opt.convergence_norm=='rms' else maximum
        # Auxiliary bodies must also converge; otherwise a stationary cloth
        # could terminate before its grippers have reached their driven pose.
        auxiliary_decrement = float(np.max(np.abs(direction[problem.n:]),initial=0)/problem.dt)
        record={"iteration":iteration,"energy":value,"velocity_decrement":decrement,
                "velocity_decrement_rms":rms,"velocity_decrement_max":maximum,
                "strain_stiffness":problem.shell.strain_stiffness,"strain_stiffness_updated":False,
                "linear_relative_residual":float(np.linalg.norm(A @ direction.ravel()[free]+g.ravel()[free])/max(np.linalg.norm(g.ravel()[free]),1e-30)),
                "auxiliary_decrement":auxiliary_decrement,"gradient_inf":float(np.max(np.abs(g.ravel()[free])))}
        if max(decrement,auxiliary_decrement) <= opt.velocity_tolerance:
            records.append(record)
            converged=True
            reason="newton_decrement"
            break
        slope=float(np.sum(g*direction))
        if slope >= 0:
            reason="non_descent_direction"
            records.append(record)
            break
        alpha=problem.safe_step(x,direction)
        accepted=False
        for backtrack in range(opt.max_linesearch):
            trial=x+alpha*direction
            trial_offset=offset
            if opt.offset_linesearch=="nonlinear" and problem.feasible(trial):
                trial_offset=problem.transfer.offset(trial[:problem.n])
            if alpha > 1e-14 and problem.feasible(trial,trial_offset):
                candidate=problem.evaluate(trial,trial_offset)[0]
                if np.isfinite(candidate) and candidate <= value+opt.armijo*alpha*slope:
                    accepted=True
                    break
            alpha*=.5
        record.update(alpha=float(alpha),backtracks=backtrack,accepted=accepted)
        if accepted:record['candidate_energy']=float(candidate)
        records.append(record)
        if not accepted:
            reason="line_search_failed"
            break
        if opt.allow_preview_stall and alpha < opt.preview_min_alpha:
            # PSQ Algorithm 1, lines 14-15: stop a proxy preview before applying
            # an ineffectively small step. This is NOT finest-level convergence.
            record['accepted']=False
            reason="preview_step_too_small"
            break
        x=trial
        # C-IPC Shell/IMPLICIT_EULER.h updates after accepted Newton steps.
        # Compare two successive accepted iterates, not an arbitrary predictor.
        if problem.shell.strain_limit is not None:
            gaps=problem.shell.strain_limit-np.linalg.svd(problem.shell.F(x[:problem.n][problem.shell.faces]),compute_uv=False)[:,0]
            record['strain_stiffness_updated']=problem.shell.adapt_strain_stiffness(previous_strain_gaps,gaps)
            record['strain_stiffness_after_step']=problem.shell.strain_stiffness
            previous_strain_gaps=gaps
    offset=problem.transfer.offset(x[:problem.n])
    value,g,_=problem.evaluate(x,offset)
    # PD 2024 Sec. 4.6 permits a preview level to stop when the approximate
    # decrement no longer gives descent. Finest/direct solves remain strict.
    preview_stop=(not converged and opt.allow_preview_stall and
                  reason in ("non_descent_direction","line_search_failed","preview_step_too_small"))
    accepted=converged or preview_stop
    report={"converged":converged,"accepted":accepted,"preview_stop":preview_stop,
            "reason":reason,"iterations":len(records),
            "convergence_norm":opt.convergence_norm,"strain_stiffness":problem.shell.strain_stiffness,
            "strain_model":problem.shell.strain_model,
            "seconds":time.perf_counter()-start,"energy":value,
            "gradient_inf":float(np.max(np.abs(g.ravel()[free]))),
            "max_stretch":problem.shell.max_stretch(x[:problem.n]),"trace":records}
    return x,report
