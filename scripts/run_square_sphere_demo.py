"""Four-level square cloth drop onto a fixed triangulated sphere using the audited native PSQ runtime.

Only accepted physical frames are saved. Each worker has its own CUDA context.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import load_npz, save_npz, eye

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bunny_runtime import ORDERED_PACKAGE, runtime_provenance
from progressive.integrators import predict
from progressive.prolongation import ShellProlongation, frames, nested_anchors

DT, STEPS, THICKNESS = .01, 200, .0004
FRICTION, CONTACT_STIFFNESS = .3, 1e8
FREEFALL_STEPS = 22


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def run_path(root, method, level):
    return root / ('shared_L0' if level == 0 else f'{method}_L{level}')


def prepare(root):
    assets = root / 'assets'
    assets.mkdir()
    source = ROOT / 'output/hanging_square_mesh_preview'
    import trimesh
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=.30)
    sphere.vertices += [0., 0., .6]
    np.savez(assets/'sphere.npz', vertices=np.asarray(sphere.vertices), faces=np.asarray(sphere.faces))
    sphere.export(assets/'sphere.obj')
    for level in range(4):
        d = dict(np.load(source / f'mesh_{level}.npz'))
        d['vertices'] = np.column_stack((d['uv'][:, 0]-.5, d['uv'][:, 1]-.5, np.full(len(d['uv']), 1.15)))
        d['pin_indices'] = np.array([], dtype=int)
        np.savez(assets / f'mesh_{level}.npz', **d)
        if level:
            shutil.copy2(source / f'map_{level-1}_{level}.npz', assets)
    for lo in range(3):
        U = eye(len(np.load(assets / f'mesh_{lo}.npz')['vertices']), format='csr')
        for hi in range(lo+1, 4):
            U = load_npz(assets / f'map_{hi-1}_{hi}.npz') @ U
            coarse = np.load(assets / f'mesh_{lo}.npz')
            fine = np.load(assets / f'mesh_{hi}.npz')
            np.testing.assert_allclose(U @ coarse['vertices'], fine['vertices'], atol=1e-14)
            U.eliminate_zeros()
            np.save(assets / f'anchors_{lo}_{hi}.npy', nested_anchors(U, coarse['faces']))
    for level in range(4):
        operator = assets / f'operator_{level}'
        with (assets / f'prepare_{level}.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/export_psq_libuipc.py'), '--assets', str(assets),
                            '--level', str(level), '--fine-level', '3', '--output', str(operator)], stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(ROOT/'scripts/pack_psq_native.py'), '--operator', str(operator/'operator.npz'),
                            '--output', str(assets/f'L{level}.bin'), '--thickness', str(THICKNESS)], stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(root/'scene.json', dict(dt=DT, steps=STEPS, duration_s=DT*STEPS, size_m=1., initial_height_m=1.15, sphere_center=[0.,0.,.6], sphere_radius_m=.3,
        initial_velocity=0., fixed='sphere only; all cloth vertices free', young_Pa=30000., poisson=.3, density_kg_m3=200.,
        full_thickness_m=THICKNESS, contact=True, friction=FRICTION, requested_contact_stiffness=CONTACT_STIFFNESS, wind=False, consistency_weight=0.,
        psq='L3 elastic proxy on each level; zero rest residual for this nested planar mesh',
        initialization='horizontal square 0.25 m above sphere apex; zero velocity, no initial strain',
        analytic_freefall_steps=FREEFALL_STEPS,
        analytic_freefall='Exact discrete implicit-Euler uniform gravity motion until just before contact; native solve starts at step 23',
        solver='existing ordered native CUDA build, coarse preview stops recorded separately',
        source_meshes=str(source)))


def solve(root, method, level):
    assets, out = root/'assets', run_path(root, method, level)
    progressive_update = level > 0 and method != 'direct'
    for key in list(os.environ):
        if key.startswith('UIPC_PSQ_'):
            del os.environ[key]
    os.environ.update(UIPC_PSQ_BUNDLE=str(assets/f'L{level}.bin'), UIPC_PSQ_MODE='proxy',
        UIPC_PSQ_STRAIN_CAP='directional', UIPC_PSQ_PREVIEW_STOP=str(int(level < 3)),
        UIPC_PSQ_REJECT_INF_TRIAL='1', UIPC_PSQ_ADAPTIVE_SEARCH='1', UIPC_PSQ_REFINE_PROBES='1', UIPC_PSQ_FROZEN_ONLY='1')
    from libuipc_native_runtime import load, loaded_libraries
    uipc = load(ORDERED_PACKAGE)
    from uipc import builtin, view
    from uipc.core import Engine, Scene, World, FiniteElementStateAccessorFeature
    from uipc.geometry import trimesh, label_surface
    from uipc.constitution import NeoHookeanShell, DiscreteShellBending, ElasticModuli2D, Empty
    data = np.load(assets/f'mesh_{level}.npz')
    rest, faces = data['vertices'], data['faces'].astype(np.int32)
    inv = frames(rest, faces)[2]
    pins = np.zeros(len(rest), dtype=bool)
    pins[data['pin_indices']] = True
    tracks, transfers, v0 = {}, {}, {level: np.zeros_like(rest)}
    if progressive_update:
        lower = run_path(root, method, level-1)
        if json.loads((lower/'run.json').read_text())['status'] != 'completed':
            raise RuntimeError('Parent trajectory is incomplete')
        low_mesh = np.load(assets/f'mesh_{level-1}.npz')
        tracks[level-1] = np.load(lower/'positions.npy')
        v0[level-1] = np.zeros_like(tracks[level-1][0])
        transfers[level-1] = ShellProlongation(low_mesh['vertices'], rest, low_mesh['faces'], load_npz(assets/f'map_{level-1}_{level}.npz'))
    cfg = Scene.default_config()
    cfg['dt'], cfg['gravity'] = DT, [[0.], [0.], [-9.81]]
    cfg['newton'].update(velocity_tol=.001, velocity_tol_relative=0., use_adaptive_tol=0, max_iter=1500)
    cfg['newton']['semi_implicit']['enable'] = 0
    cfg['line_search'].update(max_iter=8, report_energy=0)
    cfg['linear_system'].update(tol_rate=1e-10, fem_preconditioner='diag')
    cfg['contact']['enable'] = 1
    cfg['contact']['d_hat'] = .0007
    cfg['contact']['friction']['enable'] = int(FRICTION > 0)
    cfg['contact']['eps_velocity'] = .001
    cfg['extras']['strict_mode']['enable'] = 0
    uipc.Timer.disable_all()
    uipc.Logger.set_level(uipc.Logger.Level.Warn)
    engine = Engine('cuda', str(out/'backend'))
    world, scene = World(engine), Scene(cfg)
    scene.contact_tabular().default_model(FRICTION, CONTACT_STIFFNESS, True)
    mesh = trimesh(np.ascontiguousarray(rest), faces)
    label_surface(mesh)
    NeoHookeanShell().apply_to(mesh, ElasticModuli2D.youngs_poisson(30000., .3), 200., THICKNESS/2)
    DiscreteShellBending().apply_to(mesh, 30000., .3)
    fixed = mesh.vertices().find(builtin.is_fixed)
    if fixed is None:
        fixed = mesh.vertices().create(builtin.is_fixed, 0)
    view(fixed).reshape(-1)[:] = pins
    slot, _ = scene.objects().create('square_cloth').geometries().create(mesh)
    sphere = np.load(assets/'sphere.npz')
    sphere_rest = sphere['vertices']
    ob = trimesh(np.ascontiguousarray(sphere_rest), np.ascontiguousarray(sphere['faces'], dtype=np.int32))
    label_surface(ob)
    Empty().apply_to(ob, 200., THICKNESS/2)
    fixed_sphere = ob.vertices().find(builtin.is_fixed)
    if fixed_sphere is None:
        fixed_sphere = ob.vertices().create(builtin.is_fixed, 1)
    view(fixed_sphere)[:] = 1
    tab = scene.contact_tabular()
    obstacle_contact = tab.create('fixed_sphere')
    tab.insert(obstacle_contact, obstacle_contact, FRICTION, CONTACT_STIFFNESS, False)
    obstacle_contact.apply_to(ob)
    sphere_slot, _ = scene.objects().create('fixed_sphere').geometries().create(ob)
    states, records = [rest.copy()], []
    for n in range(1, min(FREEFALL_STEPS, STEPS)+1):
        x = rest.copy()
        x[:, 2] -= .5*9.81*DT**2*n*(n+1)
        # This plane lies above the entire obstacle and has zero elastic strain.
        if x[:, 2].min()-sphere_rest[:, 2].max() <= THICKNESS+cfg['contact']['d_hat']:
            raise ValueError('Analytic prefix entered the contact activation band')
        states.append(x)
        records.append(dict(frame=n, stretch=1., pin_error_m=0., sphere_error_m=0., elapsed_s=0.,
            integration='analytic_discrete_freefall', native=None))
    tracks[level] = states
    meta = dict(method=method, level=level, steps_requested=STEPS, dt=DT, pinned_vertices=int(pins.sum()),
        coarse_trajectory_used=progressive_update, velocity_injection_each_step=progressive_update,
        consistency_weight=0., friction=FRICTION, contact_pair_enabled=True, requested_contact_stiffness=CONTACT_STIFFNESS, config=cfg, native_package=str(ORDERED_PACKAGE))
    started = time.perf_counter()
    try:
        world.init(scene)
        if not world.is_valid():
            raise RuntimeError('Invalid initial scene')
        meta['libraries'] = loaded_libraries()
        meta['dll_sha256'] = runtime_provenance(ORDERED_PACKAGE)['dll_sha256']
        accessor = world.features().find(FiniteElementStateAccessorFeature)
        offset = int(view(slot.geometry().meta().find('backend_fem_vertex_offset')).ravel()[0])
        sphere_offset = int(view(sphere_slot.geometry().meta().find('backend_fem_vertex_offset')).ravel()[0])
        sphere_state = accessor.create_geometry(sphere_offset, len(sphere_rest))
        sphere_state.vertices().create(builtin.position, np.zeros(3))
        state = accessor.create_geometry(offset, len(rest))
        state.vertices().create(builtin.position, np.zeros(3))
        initial_state = accessor.create_geometry(offset, len(rest))
        initial_state.vertices().create(builtin.position, np.zeros(3))
        initial_state.vertices().create(builtin.velocity, np.zeros(3))
        view(initial_state.positions())[:] = states[-1].reshape(-1, 3, 1)
        initial_velocity = np.zeros_like(rest)
        initial_velocity[:, 2] = -9.81*DT*(len(states)-1)
        view(initial_state.vertices().find(builtin.velocity))[:] = initial_velocity.reshape(-1, 3, 1)
        accessor.copy_from(initial_state)
        accessor.copy_to(state)
        np.testing.assert_allclose(np.asarray(view(state.positions())).reshape(-1,3), states[-1], rtol=0, atol=1e-12)
        if progressive_update:
            state.vertices().create(builtin.velocity, np.zeros(3))
            injection = accessor.create_geometry(offset, len(rest))
            injection.vertices().create(builtin.velocity, np.zeros(3))
        for n in range(len(states)-1, STEPS):
            frame_start = time.perf_counter()
            if progressive_update:
                velocity = (predict(method, level, n, tracks, transfers, DT, v0)-states[-1])/DT
                velocity[pins] = 0.
                view(injection.vertices().find(builtin.velocity))[:] = velocity.reshape(-1, 3, 1)
                accessor.copy_from(injection)
                accessor.copy_to(state)
                np.testing.assert_allclose(view(state.positions()).reshape(-1, 3), states[-1], rtol=0, atol=1e-12)
                np.testing.assert_allclose(view(state.vertices().find(builtin.velocity)).reshape(-1, 3), velocity, rtol=0, atol=1e-12)
            world.advance()
            stats = engine.frame_stats()
            preview = (level < 3 and stats.get('preview_non_descent') and not stats.get('preview_small_step')
                and stats.get('termination_reason') == 'preview_non_descent' and stats.get('last_line_search_alpha') == 0)
            if (not world.is_valid() or not stats.get('completed') or stats.get('hit_newton_limit')
                or stats.get('preview_small_step') or (stats.get('hit_line_search_limit') and not preview)
                or not (stats.get('converged') or preview)):
                meta['rejected'] = stats
                raise RuntimeError(f'Native solver rejected frame {n+1}')
            world.retrieve()
            accessor.copy_to(state)
            x = np.array(view(state.positions())).reshape(-1, 3)
            tri = x[faces]
            F = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=2) @ inv
            stretch = float(np.linalg.svd(F, compute_uv=False).max())
            pin_error = float(np.max(np.abs(x[pins]-rest[pins]), initial=0))
            if not np.isfinite(x).all() or stretch > 1.068000001 or pin_error > 1e-10:
                raise RuntimeError(f'Invalid frame {n+1}: stretch={stretch}, pin_error={pin_error}')
            accessor.copy_to(sphere_state)
            sphere_error = float(np.max(np.abs(np.asarray(view(sphere_state.positions())).reshape(-1,3)-sphere_rest)))
            if sphere_error > 1e-12:
                raise RuntimeError('Fixed sphere moved')
            states.append(x)
            records.append(dict(frame=n+1, stretch=stretch, pin_error_m=pin_error, sphere_error_m=sphere_error, elapsed_s=time.perf_counter()-frame_start, native=stats))
            if (n+1) % 10 == 0:
                write_json(out/'progress.json', dict(accepted_frames=n+1, requested=STEPS, elapsed_s=time.perf_counter()-started))
                np.save(out/'positions.npy', states)
        meta['status'] = 'completed'
    except Exception as exc:
        meta.update(status='failed', error=str(exc))
        raise
    finally:
        np.save(out/'positions.npy', states)
        meta.update(accepted_frames=len(states)-1, elapsed_s=time.perf_counter()-started)
        write_json(out/'run.json', meta)
        write_json(out/'steps.json', records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'output/square_sphere_demo')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--direct-l3', action='store_true', help='Add independent L3 to an existing completed sphere experiment')
    p.add_argument('--worker', nargs=2, metavar=('METHOD', 'LEVEL'))
    a = p.parse_args()
    global STEPS
    STEPS = a.steps
    root = a.output.resolve()
    if a.worker:
        solve(root, a.worker[0], int(a.worker[1]))
        return
    if a.direct_l3:
        scene = json.loads((root/'scene.json').read_text())
        expected = dict(dt=DT, steps=STEPS, full_thickness_m=THICKNESS, friction=FRICTION,
            requested_contact_stiffness=CONTACT_STIFFNESS, analytic_freefall_steps=FREEFALL_STEPS)
        if any(scene[key] != value for key,value in expected.items()):
            raise ValueError('Direct settings must match the existing experiment')
        if not json.loads((root/'assets/L3.json').read_text())['identity']:
            raise ValueError('Independent L3 requires an identity elastic proxy')
        out = run_path(root, 'direct', 3)
        out.mkdir(exist_ok=False)
        shutil.copy2(__file__, out/'source_snapshot.py')
        write_json(out/'input_hashes.json', {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (root/'scene.json', root/'assets/mesh_3.npz', root/'assets/sphere.npz', root/'assets/L3.bin')})
        with (out/'native.log').open('w') as log:
            result = subprocess.run([sys.executable, __file__, '--output', str(root), '--steps', str(STEPS),
                '--worker', 'direct', '3'], stdout=log, stderr=subprocess.STDOUT)
        print(f'Direct L3 finished: {result.returncode}', flush=True)
        raise SystemExit(result.returncode)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, root/'source_snapshot.py')
    write_json(root/'source_hashes.json', {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (Path(__file__), ROOT/'progressive/integrators.py', ROOT/'progressive/prolongation.py')})
    prepare(root)
    if a.prepare_only:
        return
    failures = []
    for method, level in [('direct', 0)] + [(m, k) for m in ('semi', 'velpro') for k in range(1, 4)]:
        out = run_path(root, method, level)
        out.mkdir()
        print(f'Starting {out.name}', flush=True)
        with (out/'native.log').open('w') as log:
            result = subprocess.run([sys.executable, __file__, '--output', str(root), '--steps', str(STEPS), '--worker', method, str(level)], stdout=log, stderr=subprocess.STDOUT)
        print(f'Finished {out.name}: {result.returncode}', flush=True)
        if result.returncode:
            failures.append(out.name)
            if not (out/'run.json').exists():
                write_json(out/'run.json', dict(status='process_exited', returncode=result.returncode))
    write_json(root/'failed_runs.json', failures)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
