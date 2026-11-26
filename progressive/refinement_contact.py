"""PSQ Algorithms 3--5: perturbation, pair-wise ACCD and stencil freezing."""
from dataclasses import dataclass, asdict
import numpy as np
import ipctk


class RefinementFailure(RuntimeError):
    """A failed initializer never becomes an accepted physical state."""


@dataclass
class ExpansionOptions:
    gap_cap: float = .001
    minimum_gap: float | None = None  # PSQ Algorithm 3 epsilon; None restores g0
    fixed_policy: str = 'unsplit-lineage'
    perturbation_fraction: float = .01
    small_gap_ratio: float = .1  # explicit interpretation of paper's d0 << g
    sampling_budget: int = 2048
    seed: int = 527
    max_advances: int = 256
    relaxation_iterations: int = 100
    barrier_stages: int = 7
    barrier_multiplier: float = 10.
    dt: float = .01
    arap_stiffness: float = 1000.

    def validate(self):
        if self.fixed_policy not in ('unsplit-lineage','coarse-indices'):
            raise ValueError('Unknown inherited vertex policy')
        if not (self.gap_cap>0 and 0<self.perturbation_fraction<.1 and 0<self.small_gap_ratio<1):
            raise ValueError('Invalid gap or perturbation settings')
        if min(self.sampling_budget,self.max_advances,self.relaxation_iterations,self.barrier_stages)<1:
            raise ValueError('Refinement budgets must be positive')
        if self.dt<=0 or self.arap_stiffness<=0 or self.barrier_multiplier<=1:
            raise ValueError('Invalid relaxation settings')
        if self.minimum_gap is not None and not 0<self.minimum_gap<=self.gap_cap:
            raise ValueError('Minimum expansion gap must lie in (0, gap_cap]')
        return asdict(self)


def edges_of(faces):
    return np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)


def collision_mesh(rest, faces, cloth_vertices=None):
    faces=np.asarray(faces,np.int32)
    mesh=ipctk.CollisionMesh(np.asarray(rest,float),edges_of(faces),faces)
    if cloth_vertices is not None:mesh.can_collide=ipctk.make_static_obstacle_filter(cloth_vertices)
    return mesh


def minimum_gap(mesh, x, cap):
    """A conservative lower bound, capped when no nearby pair exists."""
    collisions=ipctk.NormalCollisions();collisions.build(mesh,x,cap)
    if collisions.empty():return float(cap)
    return min(float(cap),float(np.sqrt(max(0.,collisions.compute_minimum_distance(mesh,x)))))


def full_rank(x, faces):
    p=x[faces];a=p[:,1]-p[:,0];b=p[:,2]-p[:,0]
    lengths=np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-300)
    return bool(np.isfinite(p).all() and np.all(np.linalg.norm(np.cross(a,b),axis=1)>1e-12*lengths))


def rank_step(x,d,faces):
    """Check triangle area over the ENTIRE linear segment, not only endpoints.

    Cross(e0+t*d0,e1+t*d1) is quadratic; its squared norm is quartic.
    Evaluate its cubic derivative's real roots to detect interior collapse.
    A Weyl certificate skips triangles already known safe. Near a collapse,
    bisect the first unsafe interval and back off. This avoids the extremely
    restrictive norm bound on thin but expanding triangles during refinement.
    """
    p=x[faces];q=d[faces]
    A=np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=2)
    D=np.stack((q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=2)
    speed=np.linalg.norm(D,axis=(1,2));moving=speed>1e-15
    if not moving.any():return 1.
    sigma=np.linalg.svd(A,compute_uv=False)[:,-1]
    uncertain=np.flatnonzero(moving & (.9*sigma<speed))
    alpha=1.
    for i in uncertain:
        a,b=A[i,:,0],A[i,:,1];da,db=D[i,:,0],D[i,:,1]
        c=np.array([np.cross(a,b),np.cross(da,b)+np.cross(a,db),np.cross(da,db)])
        scale=max(float(np.linalg.norm(A[i])**2),1e-30)
        floor=1e-10*scale
        def area(t):return float(np.linalg.norm(c[0]+t*(c[1]+t*c[2])))
        if area(0)<=floor:raise RefinementFailure('Triangle starts below refinement area tolerance')
        polynomial=sum((np.convolve(c[:,j],c[:,j]) for j in range(3)),np.zeros(5))
        derivative=np.polynomial.polynomial.polyder(polynomial)
        if not np.isfinite(derivative).all():raise RefinementFailure('Nonfinite triangle area polynomial')
        roots=np.polynomial.polynomial.polyroots(derivative) if np.any(derivative) else []
        times=sorted([float(z.real) for z in roots if abs(z.imag)<1e-7 and 0<z.real<alpha]+[alpha])
        previous=0.
        for t in times:
            if area(t)<=floor:
                lo,hi=previous,t
                for _ in range(55):
                    mid=(lo+hi)/2
                    if area(mid)>floor:lo=mid
                    else:hi=mid
                alpha=min(alpha,.9*lo)
                break
            previous=t
    return alpha


def stratified_directions(rng, count):
    """Equal-area latitude strata with random longitude and within-cell jitter."""
    z=1-2*(np.arange(count)+rng.random(count))/count
    phi=2*np.pi*rng.random(count);radius=np.sqrt(np.maximum(0.,1-z*z))
    values=np.column_stack((radius*np.cos(phi),radius*np.sin(phi),z))
    return values[rng.permutation(count)]


def safe_perturb(x, split, target, patch_faces, epsilon, rng, budget):
    """Keep the first endpoint direction, sample the second when necessary.

    Local checks include degeneracy, intersection and zero-distance pairs.
    The caller bounds displacement by the pre-split global gap and checks the
    entire batch afterwards. No CCD query starts at the coincident topology.
    """
    split=np.asarray(split,int);base=x[split].copy()
    ids=np.unique(patch_faces);lookup={int(v):i for i,v in enumerate(ids)}
    faces=np.array([[lookup[int(v)] for v in f] for f in patch_faces],np.int32)
    local_split=np.array([lookup[int(v)] for v in split])
    direction=np.asarray(target)-base;length=np.linalg.norm(direction,axis=1)
    for j in range(2):
        if length[j]<1e-15:
            direction[j]=target[j]-target[1-j];length[j]=np.linalg.norm(direction[j])
        if length[j]<1e-15:raise RefinementFailure('Coincident expansion targets')
    direction/=length[:,None]
    local=x[ids].copy()
    # CollisionMesh needs only topology here. Use a finite noncollapsed target
    # where possible; intersection checks do not evaluate rest mollifiers.
    mesh=collision_mesh(local,faces)
    candidates=np.vstack((direction[1],stratified_directions(rng,budget)))
    for attempt,d in enumerate(candidates):
        local[local_split[0]]=base[0]+epsilon*direction[0]
        local[local_split[1]]=base[1]+epsilon*d
        if full_rank(local,faces) and not ipctk.has_intersections(mesh,local) and minimum_gap(mesh,local,epsilon)>epsilon*1e-8:
            x[split]=local[local_split]
            return {'attempts':attempt+1,'random_fallback':bool(attempt),'epsilon':float(epsilon)}
    raise RefinementFailure(f'No safe local perturbation within {budget+1} attempts')


def ratio_advance(mesh,x,target,gap,options):
    """Additive advancement with pair-specific ratios and limiting-stencil stops.

    IPC Toolkit's conservative_rescaling is 1-r, NOT r. Its v1.6.1 source
    defines the retained gap as (1-conservative_rescaling)*initial_distance.
    Candidate sets are rebuilt after stopping stencils (conservative CPU path).
    """
    x=x.copy();target=target.copy();records=[]
    faces=np.asarray(mesh.faces);edges=np.asarray(mesh.edges)
    for iteration in range(options.max_advances):
        d=target-x
        if np.max(np.linalg.norm(d,axis=1),initial=0)<1e-14:return x,records
        candidates=ipctk.Candidates();candidates.build(mesh,x,target,gap)
        alpha=1.;limiting=[];small=0
        for pair in (*candidates.fv_candidates,*candidates.ee_candidates,*candidates.ev_candidates,*candidates.vv_candidates):
            ids=np.array(pair.vertex_ids(edges,faces),int);ids=ids[ids>=0]
            if not np.any(d[ids]):continue
            start=pair.dof(x,edges,faces);end=pair.dof(target,edges,faces)
            d0=float(np.sqrt(max(0.,pair.compute_distance(start))))
            if d0<=1e-14:raise RefinementFailure('ACCD requires separated primitives')
            r=.999 if d0<options.small_gap_ratio*gap else .1
            small+=int(r==.999)
            hit,toi=pair.ccd(start,end,0.,1.,ipctk.AdditiveCCD(conservative_rescaling=1-r))
            if hit and toi<alpha-1e-12:alpha=float(toi);limiting=[ids]
            elif hit and abs(toi-alpha)<=1e-12:limiting.append(ids)
        rank_alpha=rank_step(x,d,faces)
        used=min(alpha,rank_alpha)
        if not np.isfinite(used) or used<=0:raise RefinementFailure('No positive expansion step')
        trial=x+used*d
        if not full_rank(trial,faces) or ipctk.has_intersections(mesh,trial):
            raise RefinementFailure('Expansion postcheck rejected a CCD step')
        x=trial;stopped=[]
        if alpha<1 and alpha<=rank_alpha:
            stopped=np.unique(np.concatenate(limiting)).tolist()
            target[stopped]=x[stopped]
        records.append({'alpha':used,'ccd_alpha':alpha,'rank_alpha':rank_alpha,
                        'stopped_vertices':stopped,'small_gap_pairs':small})
        if used==1:return x,records
    raise RefinementFailure('Expansion advancement budget exhausted')
