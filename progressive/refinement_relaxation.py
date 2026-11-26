"""PSQ patch relaxation and PCS 2022 Eq. (9) strain-feasibility correction.

Budgets, stiffness schedules and tolerances are explicit project parameters.
An exhausted budget raises an error; it never silently accepts an initializer.
"""
from copy import copy
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from .shell import assemble_hessian
from .refinement_contact import RefinementFailure,minimum_gap,rank_step,full_rank,edges_of


def subset_shell(shell, mask):
    result=copy(shell)
    for name in ('faces','inv_dm','areas','strain_scale','strain_volume'):
        setattr(result,name,getattr(shell,name)[mask])
    return result


def arap_local(shell,tri,stiffness,hessian=False):
    """Sum kappa_a*(sigma_i-1)^2, without an unmentioned area weight."""
    F=shell.F(tri);left,s,right=np.linalg.svd(F,full_matrices=True)
    if np.min(s,initial=1)<1e-12:raise RefinementFailure('Degenerate ARAP element')
    first=2*stiffness*(s-1)
    energy=stiffness*np.sum((s-1)**2,axis=1)
    gradient=shell.scatter_membrane((left[:,:,:2]*first[:,None,:]) @ right)
    if not hessian:return energy,gradient,None
    blocks=np.empty((len(tri),9,9))
    skew=first.sum(axis=1)/s.sum(axis=1)
    for i in range(9):
        d=np.zeros_like(tri);d[:,i//3,i%3]=1
        D=left.transpose(0,2,1) @ shell.F(d) @ right.transpose(0,2,1)
        R=np.zeros_like(D);R[:,0,0]=2*stiffness*D[:,0,0];R[:,1,1]=2*stiffness*D[:,1,1]
        symmetric=stiffness*(D[:,0,1]+D[:,1,0]);antisymmetric=.5*skew*(D[:,0,1]-D[:,1,0])
        R[:,0,1]=symmetric+antisymmetric;R[:,1,0]=symmetric-antisymmetric
        R[:,2,:]=first/s*D[:,2,:]
        blocks[:,:,i]=shell.scatter_membrane(left @ R @ right).reshape(-1,9)
    return energy,gradient,blocks


def correction_energy(shell,x,target,feasible,contact,dt,kappa,hessian=False):
    """K + B + S[feasible] + ARAP[remaining], PCS Eq. (9)."""
    n=len(shell.rest);mass=shell.mass[:,None]/dt**2;d=x[:n]-target[:n]
    value=float(.5*np.sum(mass*d*d));g=np.zeros_like(x);g[:n]=mass*d
    diagonal=np.zeros_like(x);diagonal[:n]=mass
    H=diags(diagonal.ravel()).tocsr() if hessian else None
    for mask,strain in ((feasible,True),(~feasible,False)):
        if not np.any(mask):continue
        part=subset_shell(shell,mask);tri=x[part.faces]
        if strain:
            e,local_g=part.strain_local(tri)
            local_H=part.strain_hessian(tri) if hessian and np.isfinite(e).all() else None
        else:e,local_g,local_H=arap_local(part,tri,kappa,hessian)
        value+=float(e.sum());np.add.at(g,part.faces.ravel(),local_g.reshape(-1,3))
        if hessian and local_H is not None:H+=assemble_hessian(part.faces,local_H,len(x),True)
    if contact:
        e,cg,cH=contact.evaluate(x,hessian);value+=e;g+=cg
        if hessian:H+=cH
    return value,g,H


def strain_relaxation(shell,x,target,fixed,contact,options):
    """Fix inherited coarse vertices; monotonically grow the feasible face set."""
    x=x.copy();n=len(shell.rest);free=np.flatnonzero(~np.broadcast_to(fixed[:,None],x.shape).ravel())
    feasible=np.zeros(len(shell.faces),bool);records=[]
    if shell.strain_limit is None:return x,records
    # A spectral stretch bound implies the same bound on every material edge.
    # Two fixed endpoints cannot change their separation: this is a certificate
    # of infeasibility, not a heuristic Newton failure or permission to unpin.
    edges=edges_of(shell.faces);edges=edges[fixed[edges].all(axis=1)]
    ratios=np.linalg.norm(x[edges[:,1]]-x[edges[:,0]],axis=1)/np.linalg.norm(
        shell.rest[edges[:,1]]-shell.rest[edges[:,0]],axis=1)
    bad=ratios>=shell.strain_limit
    if np.any(bad):
        error=RefinementFailure('Fixed inherited endpoints violate the strain limit; correction is infeasible')
        error.diagnostics={'phase':'strain_correction_preflight','strain_limit':shell.strain_limit,
                           'fixed_edge_vertices':edges[bad].tolist(),'fixed_edge_stretches':ratios[bad].tolist(),
                           'fixed_vertices':np.flatnonzero(fixed).tolist(),
                           'max_stretch':shell.max_stretch(x[:n])}
        error.failed_state=x.copy()
        raise error
    for iteration in range(options.relaxation_iterations):
        singular=np.linalg.svd(shell.F(x[:n][shell.faces]),compute_uv=False)
        feasible |= np.all(singular<shell.strain_limit,axis=1)
        if feasible.all():
            # PCS final target step, filtered by BOTH contact and strain limits.
            d=target-x;d[fixed]=0
            alpha=min(rank_step(x,d,shell.faces),contact.max_step(x,x+d) if contact else 1.)
            while alpha>1e-14:
                trial=x+alpha*d
                if shell.feasible(trial[:n]):x=trial;break
                alpha*=.5
            if contact:contact.validate(x)
            return x,records
        if not len(free):raise RefinementFailure('Strain constraints infeasible with all vertices fixed')
        value,g,H=correction_energy(shell,x,target,feasible,contact,options.dt,options.arap_stiffness,True)
        direction=np.zeros(x.size);direction[free]=spsolve(H[free][:,free].tocsc(),-g.ravel()[free]);direction=direction.reshape(x.shape)
        if not np.isfinite(direction).all():raise RefinementFailure('Nonfinite strain-correction direction')
        slope=float(np.sum(g*direction))
        if slope>=0 or np.linalg.norm(direction)<1e-14:raise RefinementFailure('Strain correction stalled before feasibility')
        alpha=min(rank_step(x,direction,shell.faces),contact.max_step(x,x+direction) if contact else 1.)
        accepted=False
        for _ in range(60):
            trial=x+alpha*direction
            sigma=np.linalg.svd(shell.F(trial[:n][shell.faces]),compute_uv=False)
            if np.min(sigma)>1e-8 and np.all(sigma[feasible]<shell.strain_limit):
                e=correction_energy(shell,trial,target,feasible,contact,options.dt,options.arap_stiffness)[0]
                if np.isfinite(e) and e<=value+1e-4*alpha*slope:accepted=True;break
            alpha*=.5
        if not accepted:raise RefinementFailure('Strain correction line search failed')
        x=trial;records.append({'iteration':iteration,'infeasible_faces':int((~feasible).sum()),
                                'alpha':alpha,'energy_before':value,'energy_after':e,
                                'max_stretch':shell.max_stretch(x[:n])})
    error=RefinementFailure('Strain correction budget exhausted')
    error.diagnostics={'phase':'strain_correction','iterations':records,
                       'fixed_vertices':np.flatnonzero(fixed).tolist(),
                       'max_stretch':shell.max_stretch(x[:n])}
    error.failed_state=x.copy()
    raise error


def patch_relaxation(shell,x,target,movable,contact,gap,options,transfer=None,finest=None):
    """Local incremental shell energy + IPC; outside-patch coordinates fixed.

    Safe initialization supplies the dynamic-to-finest intrinsic PSQ proxy.
    Standalone callers may explicitly use the identity/current-shell problem.
    """
    from .solver import IncrementalProblem,NewtonOptions,solve
    from .prolongation import ShellProlongation
    from scipy.sparse import eye
    class LocalProblem(IncrementalProblem):
        def safe_step(self,positions,direction):
            return min(self.contact.max_step(positions,positions+direction),
                       rank_step(positions,direction,self.shell.faces))
    n=len(shell.rest);x=x.copy();records=[]
    if transfer is None:transfer=ShellProlongation(shell.rest,shell.rest,shell.faces,eye(n))
    if finest is None:finest=shell
    fixed=np.ones(x.shape,bool);fixed[np.asarray(movable,int)]=False
    original_stiffness=contact.stiffness
    try:
        for stage in range(options.barrier_stages):
            current_gap=minimum_gap(contact.mesh,x,gap)
            if current_gap>=gap*(1-1e-8) and shell.feasible(x[:n]):return x,records
            contact.stiffness=original_stiffness*options.barrier_multiplier**stage
            import ipctk
            contact.barrier=ipctk.BarrierPotential(contact.dhat,contact.stiffness)
            # No strain barrier yet: the newly expanded topology may be over
            # the strain limit. PCS correction is the separate final pass.
            problem=LocalProblem(shell,finest,transfer,options.dt,target[:n],x,
                                 gravity=(0,0,0),contact=contact)
            next_x,report=solve(problem,x,NewtonOptions(max_iterations=options.relaxation_iterations,allow_preview_stall=True),fixed)
            contact.validate(next_x)
            if not full_rank(next_x,shell.faces):raise RefinementFailure('Patch relaxation produced degeneracy')
            x=next_x;records.append({'stage':stage,'barrier_stiffness':contact.stiffness,
                                    'minimum_gap':minimum_gap(contact.mesh,x,gap),'solver':report})
        if minimum_gap(contact.mesh,x,gap)<gap*(1-1e-8):
            error=RefinementFailure('Patch relaxation could not restore gap')
            error.diagnostics={'required_gap':gap,'final_gap':minimum_gap(contact.mesh,x,gap),'stages':records}
            error.failed_state=x.copy()
            raise error
        return x,records
    finally:
        contact.stiffness=original_stiffness
        import ipctk
        contact.barrier=ipctk.BarrierPotential(contact.dhat,original_stiffness)
