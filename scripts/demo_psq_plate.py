"""Quasistatic plate compression/shear with actual decimation and PSQ proxies.

Displacement boundary loading, no mass/inertia/gravity or prescribed interior
motion. Material parameters follow PSQ supplement Fig.16; geometry, loading,
mesh counts and numerical tolerances are local reproducible choices.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import eye
from scipy.sparse import coo_matrix, save_npz
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from progressive.shell import Shell
from progressive.prolongation import ShellProlongation, Hierarchy
from progressive.runtime import numerical_threads
from demo_psq_intrinsic_pleats import collision_frames
from demo_psq_shallow_pleats import write_json

SIZE = .3
MATERIAL = dict(young=7e10, poisson=.33, thickness=3.18e-4, density=472.6,
                strain_limit=None)
FORCE_TOL = 1e-3
STEP_TOL = 3e-8


def prepare(root, resolution=24):
    root.mkdir(parents=True, exist_ok=False)
    u, v = np.meshgrid(np.linspace(0, 1, resolution+1), np.linspace(0, 1, resolution+1))
    uv = np.column_stack((u.ravel(), v.ravel()))
    rng = np.random.default_rng(1603)
    interior = (uv>0).all(axis=1)&(uv<1).all(axis=1)
    uv[interior] += rng.uniform(-.12/resolution, .12/resolution, (interior.sum(), 2))
    x = np.column_stack((SIZE*(uv-.5), np.zeros(len(uv))))
    f = []
    for j in range(resolution):
        for i in range(resolution):
            a = j*(resolution+1)+i; b=a+1; c=a+resolution+1; d=c+1
            f.extend(((a,b,d),(a,d,c)) if (i+j)%2==0 else ((a,b,c),(b,d,c)))
    f = np.asarray(f, dtype=np.int32)
    np.savez(root/'original_fine_mesh.npz', vertices=x, faces=f, uv=uv)
    with (root/'original_fine.obj').open('w') as fp:
        for p in x: fp.write('v '+' '.join(format(float(a), '.17g') for a in p)+'\n')
        for face in f: fp.write('f '+' '.join(str(int(a)+1) for a in face)+'\n')
    subprocess.run([sys.executable, '-m', 'progressive.prepare_intrinsic', '--input', str(root/'original_fine.obj'),
                    '--targets', '384', '192', '--output', str(root/'assets'), '--quadric', 'probabilistic', '--contact-clearance', '.0002',
                    '--lock-boundary','--executable',str(ROOT/'output/intrinsic_plate_build/Release/intrinsic_builder.exe')],
                   check=True, cwd=ROOT)
    prepare_planar_maps(root)
    write_json(root/'scene.json', dict(size_m=SIZE, material=MATERIAL,
        boundary='Both x edges prescribed in xyz; y edges free. Edge positions are constrained, edge slopes are not clamped.',
        compression_fraction=.06, shear_fraction=.12,
        geometry='Jittered interior 24x24 flat triangulation; actual native edge collapses; original finest topology retained.',
        scope='Contact-free quasistatic plate experiment, not dynamics or a complete safe-refinement benchmark.',
        force_tolerance_N=FORCE_TOL, step_tolerance_m=STEP_TOL,
        seed='Initial guess only: 0.0001*size sine perturbation; rest shape and external forces unchanged.',
        paper='https://pcs-sim.github.io/psq/psq-combined.pdf; Fig.16 and supplementary Table 1',
        local_choices='Size, boundary implementation, loading amounts, seed, resolution, solver tolerances and load schedule.'))


def planar_map(coarse, faces, fine):
    """Exact intrinsic coordinates in the shared XY material domain.

    This flat-domain specialization is NOT the native successive-UV map, and
    uses no 3D nearest-point approximation. All points must lie in a triangle.
    """
    tri = coarse[faces,:2]
    B = np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)
    inv = np.linalg.inv(B)
    indices = []; weights = []; anchors = []
    for point in fine[:,:2]:
        bc = np.einsum('fij,fj->fi',inv,point-tri[:,0])
        bc = np.column_stack((1-bc.sum(axis=1),bc))
        quality = bc.min(axis=1)
        j = int(np.argmax(quality))
        assert quality[j]>-1e-10
        w=bc[j].copy();w[np.abs(w)<1e-12]=0;w/=w.sum()
        indices.extend(faces[j]);weights.extend(w);anchors.append(j)
    U=coo_matrix((weights,(np.repeat(np.arange(len(fine)),3),indices)),shape=(len(fine),len(coarse))).tocsr()
    U.eliminate_zeros()
    np.testing.assert_allclose(U@coarse,fine,atol=1e-13,rtol=0)
    return U,np.asarray(anchors)


def prepare_planar_maps(root):
    from scipy.sparse import load_npz
    meshes=[np.load(root/'assets'/f'mesh_{k}.npz') for k in range(3)]
    target=root/'planar_maps';target.mkdir()
    native_errors=[]
    for k in range(2):
        U,a=planar_map(meshes[k]['vertices'],meshes[k]['faces'],meshes[k+1]['vertices'])
        difference=U-load_npz(root/'assets'/f'map_{k}_{k+1}.npz')
        error=float(np.max(np.abs(difference.data),initial=0))
        assert error<1e-11
        native_errors.append(error)
        save_npz(target/f'map_{k}_{k+1}.npz',U);np.save(target/f'anchors_{k}_{k+1}.npy',a)
        _,a=planar_map(meshes[k]['vertices'],meshes[k]['faces'],meshes[2]['vertices'])
        np.save(target/f'direct_anchors_{k}.npy',a)
    write_json(target/'manifest.json',dict(method='Exact barycentric containment in shared planar XY material domain',
        native_maps_retained_in='../assets',native_successive_uv_used=False,nearest_point_mapping=False,
        native_successive_uv_max_weight_differences=native_errors,
        purpose='Flat-domain PSQ specialization; affine-exact rest reproduction and identical straight boundary constraints.'))


def load(root):
    from scipy.sparse import load_npz
    meshes=[np.load(root/'assets'/f'mesh_{k}.npz') for k in range(3)]
    a=root/'planar_maps'
    h=Hierarchy([d['vertices'] for d in meshes],[d['faces'] for d in meshes],
        [load_npz(a/f'map_{k}_{k+1}.npz') for k in range(2)],
        [np.load(a/f'anchors_{k}_{k+1}.npy') for k in range(2)],
        [np.load(a/f'direct_anchors_{k}.npy') for k in range(2)]+[None])
    original=np.load(root/'original_fine_mesh.npz')
    np.testing.assert_array_equal(h.rests[-1],original['vertices'])
    np.testing.assert_array_equal(h.faces[-1],original['faces'])
    return h,None,None,None


def boundary(rest, amount, case):
    mask = np.isclose(np.abs(rest[:,0]), SIZE/2, atol=1e-12)
    target = rest.copy()
    if case=='compression': target[:,0] *= 1-amount
    else: target[:,1] += amount*rest[:,0]
    return mask, target


class Model:
    def __init__(self, h, k, proxy):
        self.shell = Shell(h.rests[k], h.faces[k], **MATERIAL)
        self.fine = Shell(h.rests[-1], h.faces[-1], **MATERIAL) if proxy else self.shell
        self.p = h.to_finest[k] if proxy else ShellProlongation(h.rests[k], h.rests[k], h.faces[k], eye(len(h.rests[k]), format='csr'))
        self.n = len(h.rests[k])
        # Flat input should be affine-exact. Otherwise the force used here is
        # only a frozen-offset approximation and must not be called exact.
        assert self.p.zero_offset

    def evaluate(self, x, hessian=False, projected=True):
        y = self.p.position(x)
        _, gf, hf = self.fine.evaluate(y, hessian, project=projected)
        # Same Neo-Hookean energy, evaluated without tr(C)-2 / log(det C)
        # cancellation near stress-free configurations of very stiff plates.
        em,eb = self.energy_parts(y);e=em+eb
        g = self.p.U.T@gf
        H = self.p.U3.T@hf@self.p.U3 if hessian else None
        return e, g, H

    def energy_parts(self, y):
        F=self.fine.F(y[self.fine.faces]);C=F.transpose(0,2,1)@F
        D=C-np.eye(2)
        detD=D[:,0,0]*D[:,1,1]-D[:,0,1]*D[:,1,0]
        q=D[:,0,0]+D[:,1,1]+detD
        if np.any(q<=-1): return np.inf,np.inf
        logdet=np.log1p(q)
        remainder=q-logdet
        small=np.abs(q)<1e-3
        t=q[small]
        remainder[small]=t*t*(.5+t*(-1/3+t*(.25+t*(-.2+t*(1/6-t/7)))))
        em=float(np.sum(self.fine.areas*(.5*self.fine.mu*(remainder-detD)+.125*self.fine.lam*logdet**2)))
        eb=float(self.fine.bending_local(y[self.fine.hinges])[0].sum())
        return em,eb

    def safe_step(self, x, d):
        y, dy = self.p.position(x), self.p.U@d
        F, dF = self.fine.F(y[self.fine.faces]), self.fine.F(dy[self.fine.faces])
        small = np.linalg.svd(F, compute_uv=False)[:,-1]
        rate = np.linalg.norm(dF, axis=(1,2))
        moving = rate>1e-14
        return min(1., float(np.min(.9*small[moving]/rate[moving]))) if moving.any() else 1.


def solve(model, initial, fixed, max_iterations=400):
    x = initial.copy(); free = np.flatnonzero(np.repeat(~fixed, 3))
    traces = []; start = time.perf_counter()
    for it in range(max_iterations):
        e, g, H = model.evaluate(x, True)
        A = H[free][:,free].tocsc()
        d = np.zeros(x.size); d[free] = spsolve(A, -g.ravel()[free]); d = d.reshape(x.shape)
        assert np.isfinite(d).all()
        force = float(np.max(np.abs(g.ravel()[free])))
        step = float(np.max(np.abs(d)))
        row = dict(iteration=it, energy_J=e, force_inf_N=force, step_inf_m=step)
        traces.append(row)
        if force<FORCE_TOL and step<STEP_TOL:
            return x, dict(converged=True, reason='force_and_step', seconds=time.perf_counter()-start, trace=traces)
        slope = float(np.sum(g*d)); assert slope<0
        alpha = model.safe_step(x, d)
        for bt in range(45):
            trial = x+alpha*d
            en = model.evaluate(trial)[0]
            if np.isfinite(en) and en<=e+1e-4*alpha*slope: break
            alpha *= .5
        else:
            return x, dict(converged=False, reason='linesearch_failed', seconds=time.perf_counter()-start, trace=traces)
        row.update(alpha=alpha, candidate_energy_J=en, backtracks=bt)
        x = trial
        if it%25==0: print('    iteration', it, 'force', force, 'step', step, flush=True)
    return x, dict(converged=False, reason='iteration_limit', seconds=time.perf_counter()-start, trace=traces)


def run(root, case, name, k, proxy, steps, max_iterations=400, parent=None):
    h, _, _, _ = load(root)
    out = root/case/name
    out.mkdir(parents=True, exist_ok=False)
    model = Model(h, k, proxy)
    rest = model.shell.rest
    fixed, _ = boundary(rest, 0, case)
    # Every original fine boundary point must remain constrained under U.
    if proxy:
        finest_fixed, _ = boundary(h.rests[-1], 0, case)
        assert np.max(np.abs(model.p.U[finest_fixed][:,~fixed].data), initial=0)<1e-10
    x = rest.copy(); states = [x.copy()]; reports = []; rows = []
    maximum = .06 if case=='compression' else .12
    controls = np.linspace(0, maximum, steps+1)
    uv = rest[:,:2]/SIZE+.5
    seed = .0001*SIZE*np.sin(np.pi*uv[:,0])*(1 if case=='compression' else np.sin(2*np.pi*uv[:,1]))
    seed[fixed] = 0
    parent_states=None
    if parent:
        assert proxy and k>0
        parent_dir=root/case/parent
        meta=json.loads((parent_dir/'run.json').read_text())
        assert meta['status']=='completed' and meta['level']==k-1 and meta['steps']==steps
        parent_states=np.load(parent_dir/'positions.npy')
    for i, amount in enumerate(controls[1:], 1):
        before = controls[i-1]
        if case=='compression': x[:,0] *= (1-amount)/(1-before)
        else: x[:,1] += (amount-before)*rest[:,0]
        if i==1: x[:,2] += seed
        if parent_states is not None:
            x=h.adjacent[k-1].position(parent_states[i])
        _, target = boundary(rest, amount, case); x[fixed] = target[fixed]
        assert not collision_frames(rest,model.shell.faces,[x])
        print(case, name, f'{i}/{steps}', 'load', amount, flush=True)
        x, report = solve(model, x, fixed, max_iterations)
        report.update(index=i, amount=float(amount))
        write_json(out/'latest_solve.json', report)
        if not report['converged']:
            np.save(out/'failed_iterate.npy', x)
            write_json(out/'failure.json', report)
            raise RuntimeError(f'{case}/{name} failed: {report["reason"]}')
        assert np.max(np.abs(x[fixed]-target[fixed]))<1e-14
        assert not collision_frames(rest, model.shell.faces, [x])
        y = model.p.position(x)
        assert not collision_frames(model.fine.rest, model.fine.faces, [y])
        e, g, _ = model.evaluate(x)
        right = fixed&(rest[:,0]>0)
        rx, ry, rz = g[right].sum(axis=0)
        # Work conjugate to the total relative edge displacement.
        direction = np.zeros_like(rest)
        direction[fixed, 0 if case=='compression' else 1] = (-1 if case=='compression' else 1)*rest[fixed,0]/SIZE
        reaction = float(np.sum(g*direction))
        em,eb = model.energy_parts(y)
        rows.append(dict(index=i, amount=float(amount), displacement_m=float(amount*SIZE), energy_J=e,
            membrane_J=em, bending_J=eb, reaction_N=reaction, right_reaction_N=[float(rx),float(ry),float(rz)],
            height_max_m=float(x[:,2].max()), height_min_m=float(x[:,2].min()),
            force_inf_N=report['trace'][-1]['force_inf_N'], step_inf_m=report['trace'][-1]['step_inf_m'],
            max_stretch=model.shell.max_stretch(x), iterations=len(report['trace']), seconds=report['seconds']))
        states.append(x.copy()); reports.append(report)
        np.save(out/'positions.npy', states)
        write_json(out/'steps.json', rows)
        write_json(out/'solver.json', reports)
    write_json(out/'run.json', dict(status='completed', level=k, shared_fine_proxy=proxy, steps=steps,
        material=MATERIAL, controls=controls.tolist(), force_tol_N=FORCE_TOL, step_tol_m=STEP_TOL,
        method='Projected Newton using U^T H_f U, Armijo, exact elastic gradients; no inertia or gravity.',
        parent=parent,initialization='Parent equilibrium at the same load' if parent else 'Previous equilibrium at the preceding load',
        seed_max_m=float(np.max(np.abs(seed))) if not parent else 0., seed_only_initial_guess=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'output/psq_plate_boundary_fixed')
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--case', choices=('compression','shear'), default='compression')
    ap.add_argument('--name', default='psq_L0')
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--direct', action='store_true')
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--max-iterations', type=int, default=400)
    ap.add_argument('--parent')
    args = ap.parse_args(); root=args.output.resolve()
    with numerical_threads(1):
        if args.prepare:
            prepare(root)
            shutil.copy2(__file__, root/'source_snapshot.py')
        else: run(root,args.case,args.name,args.level,not args.direct,args.steps,args.max_iterations,args.parent)


if __name__=='__main__': main()
