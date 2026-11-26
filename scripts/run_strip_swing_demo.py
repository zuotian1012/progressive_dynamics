"""Three nested strip meshes, two PD predictors, and a historical native control.

Run from the project root with .venv/Scripts/python scripts/run_strip_swing_demo.py.
Pass --output to choose a fresh output directory.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bunny_runtime import BUILD_ROOT, ORDERED_PACKAGE, runtime_provenance
from progressive.integrators import predict
from progressive.prolongation import ShellProlongation, frames, nested_anchors

DT, STEPS = .01, 60
LENGTH, WIDTH, THICKNESS = .40, .08, .0004
NX, NY = 6, 2
PACKAGES = {'corrected': ORDERED_PACKAGE, 'historical': BUILD_ROOT/'bunny_control_runtime/src'}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def prepare(assets):
    assets.mkdir(parents=True)
    maps = []
    for level in range(3):
        nx, ny = NX * 2**level, NY * 2**level
        i, j = np.meshgrid(np.arange(nx+1), np.arange(ny+1))
        vertices = np.column_stack((LENGTH*i.ravel()/nx, WIDTH*(j.ravel()/ny-.5), np.zeros(i.size)))
        corner = (j[:-1, :-1]*(nx+1) + i[:-1, :-1]).ravel()
        faces = np.stack((corner, corner+1, corner+nx+2, corner, corner+nx+2, corner+nx+1), axis=1).reshape(-1,3)
        np.savez(assets/f'mesh_{level}.npz', vertices=vertices, faces=faces)
        if level:
            cx, cy = nx//2, ny//2
            u, v = i.ravel()/2, j.ravel()/2
            ix, iy = np.minimum(u.astype(int), cx-1), np.minimum(v.astype(int), cy-1)
            s, t = u-ix, v-iy
            a = iy*(cx+1)+ix
            indices = np.column_stack((a, a+np.where(s>=t, 1, cx+1), a+cx+2))
            weights = np.column_stack((1-np.maximum(s,t), np.abs(s-t), np.minimum(s,t)))
            U = csr_matrix((weights.ravel(), (np.repeat(np.arange(len(a)),3), indices.ravel())),
                           shape=(len(a), (cx+1)*(cy+1)))
            U.eliminate_zeros()
            maps.append(U)
            save_npz(assets/f'map_{level-1}_{level}.npz', U)
    for lo, hi, U in [(0,1,maps[0]), (1,2,maps[1]), (0,2,maps[1]@maps[0])]:
        coarse = np.load(assets/f'mesh_{lo}.npz')
        fine = np.load(assets/f'mesh_{hi}.npz')
        np.testing.assert_allclose(U@coarse['vertices'], fine['vertices'], atol=1e-14, rtol=0)
        np.save(assets/f'anchors_{lo}_{hi}.npy', nested_anchors(U, coarse['faces']))
    for level in range(3):
        operator = assets/f'operator_{level}'
        with (assets/f'prepare_{level}.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/export_psq_libuipc.py'), '--assets', str(assets),
                            '--level', str(level), '--fine-level', '2', '--output', str(operator)], stdout=log, check=True)
            subprocess.run([sys.executable, str(ROOT/'scripts/pack_psq_native.py'), '--operator', str(operator/'operator.npz'),
                            '--output', str(assets/f'L{level}.bin'), '--thickness', str(THICKNESS)], stdout=log, check=True)


def run_path(root, build, case, method, level):
    return root/build/case/('shared_L0' if level == 0 else f'{method}_L{level}')


def solve(root, build, case, method, level):
    assets = root/'assets'
    out = run_path(root, build, case, method, level)
    for key in list(os.environ):
        if key.startswith('UIPC_PSQ_'):
            del os.environ[key]
    os.environ.update(UIPC_PSQ_BUNDLE=str(assets/f'L{level}.bin'), UIPC_PSQ_MODE='proxy',
                      UIPC_PSQ_STRAIN_CAP='directional', UIPC_PSQ_PREVIEW_STOP=str(int(level<2)),
                      UIPC_PSQ_REJECT_INF_TRIAL='1', UIPC_PSQ_ADAPTIVE_SEARCH='1',
                      UIPC_PSQ_REFINE_PROBES='1', UIPC_PSQ_FROZEN_ONLY='1')
    from libuipc_native_runtime import load, loaded_libraries
    uipc = load(PACKAGES[build])
    from uipc import builtin, view
    from uipc.core import Engine, Scene, World, FiniteElementStateAccessorFeature
    from uipc.geometry import trimesh, label_surface
    from uipc.constitution import NeoHookeanShell, DiscreteShellBending, ElasticModuli2D

    mesh_data = np.load(assets/f'mesh_{level}.npz')
    rest, faces = mesh_data['vertices'], mesh_data['faces'].astype(np.int32)
    inv = frames(rest, faces)[2]
    pins = (rest[:,0] == 0) & (case == 'swing')
    tracks, transfers, v0 = {}, {}, {level: np.zeros_like(rest)}
    target = STEPS
    if level:
        low_mesh = np.load(assets/f'mesh_{level-1}.npz')
        low = np.load(run_path(root, build, case, method, level-1)/'positions.npy')
        target = min(target, len(low)-1)
        tracks[level-1] = low
        v0[level-1] = np.zeros_like(low[0])
        transfers[level-1] = ShellProlongation(low_mesh['vertices'], rest, low_mesh['faces'],
                                                load_npz(assets/f'map_{level-1}_{level}.npz'))

    cfg = Scene.default_config()
    cfg['dt'] = DT
    cfg['gravity'] = [[0.],[0.],[-9.81]]
    cfg['newton'].update(velocity_tol=.001, velocity_tol_relative=0., use_adaptive_tol=0, max_iter=1500)
    cfg['newton']['semi_implicit']['enable'] = 0
    cfg['line_search'].update(max_iter=8, report_energy=0)
    cfg['linear_system'].update(tol_rate=1e-10, fem_preconditioner='diag')
    cfg['contact']['enable'] = 1
    cfg['contact']['d_hat'] = .0007
    cfg['contact']['friction']['enable'] = 0
    cfg['extras']['strict_mode']['enable'] = 0
    uipc.Timer.disable_all()
    uipc.Logger.set_level(uipc.Logger.Level.Warn)
    engine = Engine('cuda', str(out/'backend'))
    world, scene = World(engine), Scene(cfg)
    # Initialize the native IPC subsystem, with the sole material pair disabled.
    scene.contact_tabular().default_model(0., 1e9, False)
    mesh = trimesh(np.ascontiguousarray(rest), faces)
    label_surface(mesh)
    NeoHookeanShell().apply_to(mesh, ElasticModuli2D.youngs_poisson(30000., .3), 200., THICKNESS/2)
    DiscreteShellBending().apply_to(mesh, 30000., .3)
    if pins.any():
        fixed = mesh.vertices().find(builtin.is_fixed)
        if fixed is None:
            fixed = mesh.vertices().create(builtin.is_fixed, 0)
        view(fixed).reshape(-1)[:] = pins
    slot, _ = scene.objects().create('strip').geometries().create(mesh)
    states = [rest.copy()]
    tracks[level] = states
    records = []
    meta = dict(build=build, case=case, method=method, level=level, steps_requested=STEPS, dt=DT,
                pinned_vertices=int(pins.sum()), consistency_weight=0., friction=0., contact_pair_enabled=False, config=cfg)
    try:
        world.init(scene)
        if not world.is_valid():
            raise RuntimeError('Invalid initial scene')
        meta['libraries'] = loaded_libraries()
        meta['dll_sha256'] = runtime_provenance(PACKAGES[build])['dll_sha256']
        accessor = world.features().find(FiniteElementStateAccessorFeature)
        offset = int(view(slot.geometry().meta().find('backend_fem_vertex_offset')).ravel()[0])
        state = accessor.create_geometry(offset, len(rest))
        state.vertices().create(builtin.position, np.zeros(3))
        if level:
            state.vertices().create(builtin.velocity, np.zeros(3))
            injection = accessor.create_geometry(offset, len(rest))
            injection.vertices().create(builtin.velocity, np.zeros(3))
        for n in range(target):
            if level:
                xhat = predict(method, level, n, tracks, transfers, DT, v0)
                velocity = (xhat-states[-1])/DT
                velocity[pins] = 0.
                view(injection.vertices().find(builtin.velocity))[:] = velocity.reshape(-1,3,1)
                accessor.copy_from(injection)
                accessor.copy_to(state)
                np.testing.assert_allclose(view(state.positions()).reshape(-1,3), states[-1], rtol=0, atol=1e-12)
                np.testing.assert_allclose(view(state.vertices().find(builtin.velocity)).reshape(-1,3), velocity, rtol=0, atol=1e-12)
            world.advance()
            stats = engine.frame_stats()
            preview = (level<2 and stats.get('preview_non_descent') and not stats.get('preview_small_step')
                       and stats.get('termination_reason')=='preview_non_descent' and stats.get('last_line_search_alpha')==0)
            if (not world.is_valid() or not stats.get('completed') or stats.get('hit_newton_limit') or
                stats.get('preview_small_step') or (stats.get('hit_line_search_limit') and not preview)
                or not (stats.get('converged') or preview)):
                meta['rejected'] = stats
                raise RuntimeError(f'Native solver rejected frame {n+1}')
            world.retrieve()
            accessor.copy_to(state)
            x = np.array(view(state.positions())).reshape(-1,3)
            tri = x[faces]
            F = np.stack((tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=2)@inv
            stretch = float(np.linalg.svd(F, compute_uv=False).max())
            pin_error = float(np.max(np.abs(x[pins]-rest[pins]), initial=0))
            if not np.isfinite(x).all() or stretch > 1.068000001 or pin_error > 1e-10:
                raise RuntimeError(f'Invalid frame {n+1}: stretch={stretch}, pin error={pin_error}')
            states.append(x)
            records.append(dict(frame=n+1, stretch=stretch, pin_error_m=pin_error, native=stats))
        meta['status'] = 'completed' if target==STEPS else 'parent_stopped'
    except Exception as exc:
        meta.update(status='failed', error=str(exc))
        raise
    finally:
        np.save(out/'positions.npy', states)
        meta['accepted_frames'] = len(states)-1
        write_json(out/'run.json', meta)
        write_json(out/'steps.json', records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--worker', nargs=4, metavar=('BUILD','CASE','METHOD','LEVEL'))
    args = p.parse_args()
    root = (args.output or ROOT/'output'/datetime.now().strftime('strip_swing_demo_%Y%m%d_%H%M%S')).resolve()
    if args.worker:
        build, case, method, level = args.worker
        solve(root, build, case, method, int(level))
        return
    root.mkdir(parents=True, exist_ok=False)
    print(root, flush=True)
    prepare(root/'assets')
    failures = []
    for build in PACKAGES:
        for case in ('fall','swing'):
            for method, level in [('direct',0),('semi',1),('semi',2),('velpro',1),('velpro',2)]:
                out = run_path(root, build, case, method, level)
                out.mkdir(parents=True)
                if level and not (run_path(root, build, case, method, level-1)/'positions.npy').exists():
                    write_json(out/'run.json', dict(status='parent_unavailable', accepted_frames=0))
                    failures.append(str(out.relative_to(root)))
                    continue
                with (out/'native.log').open('w') as log:
                    result = subprocess.run([sys.executable, __file__, '--output', str(root), '--worker',
                                             build, case, method, str(level)], stdout=log, stderr=subprocess.STDOUT)
                print(out.relative_to(root), result.returncode, flush=True)
                if result.returncode:
                    if not (out/'run.json').exists():
                        write_json(out/'run.json', dict(status='process_exited', returncode=result.returncode))
                    failures.append(str(out.relative_to(root)))
    write_json(root/'failed_runs.json', failures)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
