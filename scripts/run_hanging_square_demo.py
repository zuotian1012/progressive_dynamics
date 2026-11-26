"""Four-level, two-corner square swing using the audited native PSQ runtime.

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

DT, STEPS, THICKNESS = .01, 300, .0004


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def run_path(root, method, level):
    return root / ('shared_L0' if level == 0 else f'{method}_L{level}')


def prepare(root):
    assets = root / 'assets'
    assets.mkdir()
    source = ROOT / 'output/hanging_square_mesh_preview'
    angle = np.deg2rad(35.)
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    pivot = np.array([0., 0., 1.5])
    for level in range(4):
        d = dict(np.load(source / f'mesh_{level}.npz'))
        d['vertices'] = (d['vertices'] - pivot) @ rotation.T + pivot
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
    write_json(root/'scene.json', dict(dt=DT, steps=STEPS, duration_s=DT*STEPS, size_m=1., initial_tilt_deg=35.,
        initial_velocity=0., fixed='two upper corners only', young_Pa=30000., poisson=.3, density_kg_m3=200.,
        full_thickness_m=THICKNESS, contact=False, wind=False, consistency_weight=0.,
        psq='L3 elastic proxy on each level; zero rest residual for this nested planar mesh',
        initialization='rigid rotation about upper edge; no initial strain',
        solver='existing ordered native CUDA build, coarse preview stops recorded separately',
        source_meshes=str(source)))


def solve(root, method, level):
    assets, out = root/'assets', run_path(root, method, level)
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
    from uipc.constitution import NeoHookeanShell, DiscreteShellBending, ElasticModuli2D
    data = np.load(assets/f'mesh_{level}.npz')
    rest, faces = data['vertices'], data['faces'].astype(np.int32)
    inv = frames(rest, faces)[2]
    pins = np.zeros(len(rest), dtype=bool)
    pins[data['pin_indices']] = True
    tracks, transfers, v0 = {}, {}, {level: np.zeros_like(rest)}
    if level:
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
    cfg['contact']['friction']['enable'] = 0
    cfg['extras']['strict_mode']['enable'] = 0
    uipc.Timer.disable_all()
    uipc.Logger.set_level(uipc.Logger.Level.Warn)
    engine = Engine('cuda', str(out/'backend'))
    world, scene = World(engine), Scene(cfg)
    scene.contact_tabular().default_model(0., 1e9, False)
    mesh = trimesh(np.ascontiguousarray(rest), faces)
    label_surface(mesh)
    NeoHookeanShell().apply_to(mesh, ElasticModuli2D.youngs_poisson(30000., .3), 200., THICKNESS/2)
    DiscreteShellBending().apply_to(mesh, 30000., .3)
    fixed = mesh.vertices().find(builtin.is_fixed)
    if fixed is None:
        fixed = mesh.vertices().create(builtin.is_fixed, 0)
    view(fixed).reshape(-1)[:] = pins
    slot, _ = scene.objects().create('square_cloth').geometries().create(mesh)
    states, records = [rest.copy()], []
    tracks[level] = states
    meta = dict(method=method, level=level, steps_requested=STEPS, dt=DT, pinned_vertices=int(pins.sum()),
        consistency_weight=0., friction=0., contact_pair_enabled=False, config=cfg, native_package=str(ORDERED_PACKAGE))
    started = time.perf_counter()
    try:
        world.init(scene)
        if not world.is_valid():
            raise RuntimeError('Invalid initial scene')
        meta['libraries'] = loaded_libraries()
        meta['dll_sha256'] = runtime_provenance(ORDERED_PACKAGE)['dll_sha256']
        accessor = world.features().find(FiniteElementStateAccessorFeature)
        offset = int(view(slot.geometry().meta().find('backend_fem_vertex_offset')).ravel()[0])
        state = accessor.create_geometry(offset, len(rest))
        state.vertices().create(builtin.position, np.zeros(3))
        if level:
            state.vertices().create(builtin.velocity, np.zeros(3))
            injection = accessor.create_geometry(offset, len(rest))
            injection.vertices().create(builtin.velocity, np.zeros(3))
        for n in range(STEPS):
            frame_start = time.perf_counter()
            if level:
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
            states.append(x)
            records.append(dict(frame=n+1, stretch=stretch, pin_error_m=pin_error, elapsed_s=time.perf_counter()-frame_start, native=stats))
            if (n+1) % 25 == 0:
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
    p.add_argument('--output', type=Path, default=ROOT/'output/hanging_square_demo')
    p.add_argument('--worker', nargs=2, metavar=('METHOD', 'LEVEL'))
    a = p.parse_args()
    root = a.output.resolve()
    if a.worker:
        solve(root, a.worker[0], int(a.worker[1]))
        return
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, root/'source_snapshot.py')
    write_json(root/'source_hashes.json', {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (Path(__file__), ROOT/'progressive/integrators.py', ROOT/'progressive/prolongation.py')})
    prepare(root)
    failures = []
    for method, level in [('direct', 0)] + [(m, k) for m in ('semi', 'velpro') for k in range(1, 4)]:
        out = run_path(root, method, level)
        out.mkdir()
        print(f'Starting {out.name}', flush=True)
        with (out/'native.log').open('w') as log:
            result = subprocess.run([sys.executable, __file__, '--output', str(root), '--worker', method, str(level)], stdout=log, stderr=subprocess.STDOUT)
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
