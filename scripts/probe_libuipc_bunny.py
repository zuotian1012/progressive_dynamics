"""Bounded native libuipc contact probe; NOT PSQ or matched C-IPC physics.

Uses the actual Bunny L0 assets, unchanged rest geometry. Never overwrites runs.
All Newton/PCG/CCD/line search are owned by the installed CUDA backend.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import os
import sys
import numpy as np
if '--native-source' in sys.argv:
    from libuipc_native_runtime import load, loaded_libraries
    uipc = load()
else:
    import uipc
from uipc import view
from uipc.core import Engine, Scene, World, ContactSystemFeature, FiniteElementStateAccessorFeature
from uipc.constitution import NeoHookeanShell, DiscreteShellBending, ElasticModuli2D, Empty
from uipc.geometry import trimesh, label_surface
from uipc.stats import SimulationStats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--step-budget',type=int,default=40)
    p.add_argument('--wall-budget',type=float,default=180.,help='Wall time limit checked between accepted steps')
    p.add_argument('--max-newton',type=int,default=100)
    p.add_argument('--default-solver',action='store_true',help='Use installed Scene solver defaults; retain experiment physics')
    p.add_argument('--relaxed-pd',action='store_true',help='Use native solver defaults with existing PSQ/VelPro backend')
    p.add_argument('--native-source', action='store_true')
    p.add_argument('--bundle',type=Path)
    p.add_argument('--psq-mode',choices=['proxy','reference'],default='proxy')
    p.add_argument('--initial-stretch',type=float,default=1.)
    p.add_argument('--force-updates',action='store_true')
    p.add_argument('--coarse-preview',action='store_true')
    p.add_argument('--level',type=int,choices=[0,1,2],default=0)
    p.add_argument('--lower',type=Path)
    a = p.parse_args()
    relaxed = a.default_solver or a.relaxed_pd
    if a.relaxed_pd and (a.default_solver or not a.native_source or not a.bundle or a.psq_mode!='proxy' or a.force_updates or a.coarse_preview or a.initial_stretch!=1):
        raise ValueError('Relaxed PD requires native PSQ, without custom stop or forced update')
    if a.default_solver and (a.native_source or a.bundle or a.lower or a.force_updates or a.coarse_preview or a.initial_stretch != 1):
        raise ValueError('Default solver control requires unmodified wheel and independent rest geometry')
    if not np.isfinite(a.wall_budget) or a.wall_budget <= 0:
        raise ValueError('Wall budget must be finite and positive')
    if a.max_newton < 1:
        raise ValueError('Newton iteration limit must be positive')
    if a.coarse_preview:
        if not a.native_source or not a.bundle or a.psq_mode!='proxy' or a.force_updates:
            raise ValueError('Coarse preview requires isolated PSQ proxy and forbids forced updates')
        counts=json.loads(a.bundle.with_suffix('.json').read_text())['counts']
        if counts['proxy_vertices']<=counts['coarse_vertices']:
            raise ValueError('Finest/identity solve must remain strict')
        os.environ['UIPC_PSQ_PREVIEW_STOP']='1'
    elif 'UIPC_PSQ_PREVIEW_STOP' in os.environ:
        raise ValueError('Unexpected inherited preview policy: use explicit --coarse-preview')
    if a.bundle:
        if not a.native_source:raise ValueError('PSQ requires isolated source backend')
        os.environ['UIPC_PSQ_BUNDLE']=str(a.bundle.resolve())
        os.environ['UIPC_PSQ_MODE']=a.psq_mode
    elif 'UIPC_PSQ_BUNDLE' in os.environ:
        raise ValueError('Unexpected inherited PSQ bundle: use explicit --bundle')
    if a.force_updates:
        if not a.bundle or a.psq_mode!='proxy':raise ValueError('Force updates require native proxy bundle')
        os.environ['UIPC_PSQ_FORCE_UPDATE']='1'
    elif 'UIPC_PSQ_FORCE_UPDATE' in os.environ:
        raise ValueError('Unexpected inherited forced-update switch')
    if not 1<=a.initial_stretch<1.068:raise ValueError('Initial stretch must be strain-feasible')
    if not 1 <= a.steps <= a.step_budget <= 2000:
        raise ValueError('Require 1 <= steps <= step-budget <= 2000')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assets = Path('output/bunny_paper_assets')
    mesh_name=f'mesh_{a.level}.npz'
    cloth = np.load(assets/mesh_name)
    bunny = np.load(assets/'bunny.npz')
    lower=None
    if a.lower:
        from scipy.sparse import load_npz
        if a.level not in (1,2) or not a.bundle or a.psq_mode!='proxy' or a.initial_stretch!=1 or a.force_updates:
            raise ValueError('Lower trajectory requires unstretched L1/L2 PSQ without forced acceptance')
        lower_meta=json.loads((a.lower/'run.json').read_text())
        if not lower_meta.get('completed') or lower_meta['completed_steps']<a.steps or lower_meta['dt']!=.01 or lower_meta.get('level',0)!=a.level-1 or lower_meta.get('force_updates'):
            raise ValueError('Previous level must provide a completed compatible trajectory')
        for name in [f'mesh_{a.level-1}.npz','bunny.npz']:
            if lower_meta['assets'][name]!=hashlib.sha256((assets/name).read_bytes()).hexdigest():
                raise ValueError('L0 asset mismatch')
        lower_file=a.lower/lower_meta.get('positions_file','accepted_positions.npy')
        lower=np.load(lower_file)
        map_file=assets/f'map_{a.level-1}_{a.level}.npz'
        U=load_npz(map_file)
        if U.shape!=(len(cloth['vertices']),lower.shape[1]) or len(lower)<a.steps+1 or not np.isfinite(lower).all():
            raise ValueError('Invalid L0 trajectory or intrinsic map')
        np.testing.assert_allclose(np.asarray(U.sum(axis=1)).ravel(),1,atol=1e-10,rtol=0)
        np.testing.assert_allclose(lower[0],np.load(assets/f'mesh_{a.level-1}.npz')['vertices'],atol=1e-12,rtol=0)
        injected=[]
    cfg = Scene.default_config()
    cfg['dt'] = .01
    cfg['gravity'] = [[0.], [0.], [-9.81]]
    if not relaxed:
        cfg['extras']['strict_mode']['enable'] = 1
        cfg['newton'].update(velocity_tol=.001, velocity_tol_relative=0., use_adaptive_tol=0, max_iter=a.max_newton)
        cfg['newton']['semi_implicit']['enable'] = 0
        cfg['line_search'].update(max_iter=32, report_energy=1)
        cfg['linear_system']['tol_rate'] = 1e-4
    cfg['contact']['d_hat'] = .001
    cfg['contact']['eps_velocity'] = .001
    meta = dict(status='initializing', completed_steps=0, requested_steps=a.steps,
        package=importlib.metadata.version('pyuipc'), package_path=uipc.__file__,
        config=cfg, cloth_vertices=len(cloth['vertices']), dt=.01,
        level=a.level,assets={n: hashlib.sha256((assets/n).read_bytes()).hexdigest() for n in [mesh_name,'bunny.npz']},
        material=dict(young=30000., poisson=.3, density=200., full_thickness=.00007, stored_radius=.000035),
        psq_enabled=False, scope='Native single-level contact integration probe only',
        differences=['No PSQ proxy or multilevel dynamics', 'No C-IPC hard strain barrier; native NeoHookeanShell + DiscreteShellBending',
                     'Native max-axis Newton tolerance, not RMS', 'Native adaptive contact stiffness, not old fixed kappa'],
        budget=dict(max_steps=a.step_budget, max_newton_iterations_per_step=cfg['newton']['max_iter'], wall_budget_between_steps_seconds=a.wall_budget))
    state_file = 'native_positions.npy' if relaxed else 'accepted_positions.npy'
    meta.update(default_solver=a.default_solver, relaxed_pd=a.relaxed_pd, positions_file=state_file,
                native_converged_steps=0, line_search_limit_steps=0, newton_limit_steps=0)
    if relaxed:
        meta.update(scope='Independent stock libuipc default-solver control',
                    convergence_interpretation='Native convergence includes default semi-implicit early termination; not strict Newton convergence')
        (out/'installed_default_config.json').write_text(json.dumps(Scene.default_config(),indent=2))
    def save(): (out/'run.json').write_text(json.dumps(meta,indent=2))
    if a.native_source:
        meta['package']=uipc.__version__
        meta['isolated_source']=True
    if a.bundle:
        bundle_meta=json.loads(a.bundle.with_suffix('.json').read_text())
        meta.update(psq_enabled=a.psq_mode=='proxy',hard_strain_barrier=True,
            strain_limit=1.068,strain_activation=1.034,strain_stiffness=1000.,
            proxy_vertices=bundle_meta['counts']['proxy_vertices'],psq_mode=a.psq_mode,
            bundle=str(a.bundle.resolve()),bundle_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(),
            initial_stretch=a.initial_stretch,scope='Bounded native PSQ integration acceptance',
            differences=['Native max-axis tolerance, not RMS','Native adaptive contact stiffness',
                         'Fixed initial strain stiffness 1000; adaptive strain stiffness not implemented',
                         'Single cloth with verified FEM offset; native NH/DSB proxy elasticity'])
        if a.psq_mode=='reference':
            meta.update(method='independent',scope='Independent native current-level NH/DSB elasticity with custom strain barrier')
            meta['differences'][-1]='Native current-level elasticity; no PSQ elastic evaluation or coarse-trajectory injection'
    if a.force_updates:
        meta.update(force_updates=True,scope='User-requested unconverged update experiment',
            force_policy='After 32 failed backtracks apply original collision/strain-capped direction and end frame; at 100 Newton iterations retain final update',
            convergence_claim=False)
        meta['differences'].append('Explicit nonconverged forced acceptance; not paper preview stopping or strict validation')
    meta.update(coarse_preview=a.coarse_preview,strictly_converged_steps=None if relaxed else 0,preview_stopped_steps=0)
    if a.coarse_preview:
        meta['scope']='Coarse PSQ preview with verified non-descent stopping; not finest convergence'
        meta['differences'].append('Local two-scale non-descent detector; not recovered author stopping code')
    if lower is not None:
        if lower_meta['config']!=cfg or lower_meta['material']!=meta['material']:
            raise ValueError('L0 physical/solver configuration mismatch')
        meta.update(method='velpro',scope=f'L{a.level} VelPro from L{a.level-1} trajectory, horizontal warm start',
            lower_run=str(a.lower.resolve()),lower_sha256=hashlib.sha256(lower_file.read_bytes()).hexdigest(),
            lower_run_sha256=hashlib.sha256((a.lower/'run.json').read_bytes()).hexdigest(),
            map_sha256=hashlib.sha256(map_file.read_bytes()).hexdigest(),
            predictor='x_level[n] + U @ (x_lower[n]-x_lower[n-1]); initial velocity zero; native BDF1 adds external forces')
    save()
    uipc.Logger.set_level(uipc.Logger.Level.Info)
    engine = Engine('cuda', str(out/'engine'))
    world = World(engine)
    scene = Scene(cfg)
    scene.contact_tabular().default_model(.3, 1e9)
    obstacle_contact = scene.contact_tabular().create('fixed_obstacle')
    # Match the original IPC filter: keep cloth self/obstacle contact,
    # exclude obstacle-obstacle pairs within the fixed triangulated bunny.
    scene.contact_tabular().insert(obstacle_contact, obstacle_contact, .3, 1e9, False)
    mesh = trimesh(cloth['vertices'], cloth['faces'].astype(np.int32))
    label_surface(mesh)
    NeoHookeanShell().apply_to(mesh, ElasticModuli2D.youngs_poisson(30000., .3), 200., .000035)
    DiscreteShellBending().apply_to(mesh, 30000., .3)
    slot, _ = scene.objects().create(f'cloth_L{a.level}').geometries().create(mesh)
    initial=cloth['vertices'].copy()
    if a.initial_stretch!=1:
        initial[:,:2]*=a.initial_stretch
        view(slot.geometry().positions()).reshape(-1,3)[:]=initial
    obstacle = trimesh(bunny['vertices'], bunny['faces'].astype(np.int32))
    label_surface(obstacle)
    obstacle_contact.apply_to(obstacle)
    Empty().apply_to(obstacle, 200., .000035)
    fixed = obstacle.vertices().find(uipc.builtin.is_fixed)
    if fixed is None: fixed = obstacle.vertices().create(uipc.builtin.is_fixed, 1)
    view(fixed)[:] = 1
    bunny_slot, _ = scene.objects().create('fixed_bunny').geometries().create(obstacle)
    states = [initial]
    records = []
    stats = SimulationStats()
    start = time.perf_counter()
    try:
        world.init(scene)
        if not world.is_valid(): raise RuntimeError('Native init rejected scene')
        if lower is not None:
            feature=world.features().find(FiniteElementStateAccessorFeature)
            offset=int(view(slot.geometry().meta().find('backend_fem_vertex_offset')).ravel()[0])
            velocity_state=feature.create_geometry(offset,len(initial))
            velocity_state.vertices().create(uipc.builtin.velocity,np.zeros(3))
            check_state=feature.create_geometry(offset,len(initial))
            check_state.vertices().create(uipc.builtin.position,np.zeros(3))
            check_state.vertices().create(uipc.builtin.velocity,np.zeros(3))
            feature.copy_to(check_state)
            np.testing.assert_allclose(view(check_state.positions()).reshape(-1,3),initial,atol=1e-12,rtol=0)
            meta['cloth_state_offset']=offset
        if a.native_source:meta['loaded_libraries']=loaded_libraries()
        meta['setup_seconds'] = time.perf_counter()-start
        contact = world.features().find(ContactSystemFeature)
        np.save(out/state_file,np.asarray(states))
        meta['status'] = 'running';save()
        for i in range(1, a.steps+1):
            if time.perf_counter()-start > a.wall_budget:
                meta['status'] = 'budget_stopped'; break
            tick = time.perf_counter()
            injection_error=position_error=0.
            if lower is not None:
                n=i-1
                velocity=np.zeros_like(initial) if n==0 else U@((lower[n]-lower[n-1])/.01)
                view(velocity_state.vertices().find(uipc.builtin.velocity))[:]=velocity.reshape(-1,3,1)
                feature.copy_from(velocity_state)
                feature.copy_to(check_state)
                injection_error=float(np.max(np.abs(view(check_state.vertices().find(uipc.builtin.velocity)).reshape(-1,3)-velocity)))
                position_error=float(np.max(np.abs(view(check_state.positions()).reshape(-1,3)-states[-1])))
                if injection_error>1e-12 or position_error>1e-12:raise RuntimeError('Velocity injection readback/position preservation failed')
            world.advance(); world.retrieve()
            if not world.is_valid() or world.frame() != i: raise RuntimeError('Native advance invalid/frame mismatch')
            x = view(slot.geometry().positions()).reshape(-1,3).copy()
            bv = view(bunny_slot.geometry().positions()).reshape(-1,3)
            if not np.isfinite(x).all(): raise RuntimeError('Nonfinite positions')
            fixed_error = float(np.max(np.abs(bv-bunny['vertices'])))
            if fixed_error > 1e-10: raise RuntimeError('Fixed bunny moved')
            frame_stats=engine.frame_stats()
            if relaxed:
                if not frame_stats.get('completed'):
                    raise RuntimeError('Default native solver did not complete frame')
            elif not a.force_updates:
                preview=frame_stats.get('preview_non_descent',False)
                if not frame_stats.get('completed') or not (frame_stats.get('converged') or (a.coarse_preview and preview)):
                    raise RuntimeError('Native frame lacks an authorized acceptance status')
                if preview and frame_stats.get('converged'):
                    raise RuntimeError('Preview stop incorrectly labeled Newton converged')
            meta['native_converged_steps']+=int(frame_stats.get('converged',False))
            meta['line_search_limit_steps']+=int(frame_stats.get('hit_line_search_limit',False))
            meta['newton_limit_steps']+=int(frame_stats.get('hit_newton_limit',False))
            if not relaxed:
                meta['strictly_converged_steps']+=int(frame_stats.get('converged',False))
            meta['preview_stopped_steps']+=int(frame_stats.get('preview_non_descent',False))
            records.append(dict(step=i, seconds=time.perf_counter()-tick, fixed_error=fixed_error,native=frame_stats))
            if lower is not None:
                records[-1].update(lower_velocity_time_index=i-1,velocity_readback_error=injection_error,position_injection_error=position_error)
                injected.append(velocity.copy())
                np.save(out/'injected_velocities.npy',np.asarray(injected))
            states.append(x);stats.collect()
            # Checkpoint returned states and their native status even on a later native abort.
            np.save(out/state_file,np.asarray(states))
            (out/'steps.json').write_text(json.dumps(records,indent=2))
            meta['completed_steps'] = i;save()
            print(f'PROBE_ACCEPTED {i}/{a.steps}', flush=True)
        else: meta['status'] = 'completed'
    except Exception as exc:
        meta['status'] = 'failed';meta['error'] = repr(exc)
        raise
    finally:
        meta['wall_seconds'] = time.perf_counter()-start
        meta['completed'] = meta['completed_steps'] == a.steps and meta['status'] == 'completed'
        meta['physics_seconds'] = sum(r['seconds'] for r in records)
        meta['contact_evidence'] = 'See native solver log; contact count is not inferred from geometry alone'
        np.save(out/state_file, np.asarray(states))
        (out/'steps.json').write_text(json.dumps(records,indent=2))
        stats.save_timer_frames_json(out/'timer_frames.json')
        save()
    print(json.dumps(meta,indent=2))


if __name__ == '__main__': main()
