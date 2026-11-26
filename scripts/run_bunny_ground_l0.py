"""Bunny L0 with an explicit frictionless floor; isolated from flag/shirt runs.

--rescue retries a verified first-iteration non-descent with the exact PSQ
gradient once. This is an explicit extension of the frozen-offset method.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from progressive.prolongation import frames
from bunny_runtime import BUNNY_CONTACT_DISTANCE, DEFAULT_PACKAGE, FROZEN_PACKAGE, ORDERED_PACKAGE, runtime_provenance
from paper_solver_settings import (
    NEWTON_VELOCITY_TOL, LINEAR_TOL_RATE, BUNNY_LINEAR_TOL_RATE, LINE_SEARCH_MAX_ITER,
    validate_shirt_solver_history,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restart',type=Path,help='Replay one zero-friction L0 step from a verified trajectory')
    p.add_argument('--dump-linear-system',action='store_true',help='Export Newton matrices, right-hand sides and directions in a one-step replay')
    p.add_argument('--restart-frame',type=int,default=205)
    p.add_argument('--direction-audit',action='store_true')
    offset = p.add_mutually_exclusive_group(required=True)
    offset.add_argument('--frozen-offset', action='store_true', help='Compute every Newton direction with a frozen PSQ offset (PD 2024, section 4.6)')
    offset.add_argument('--exact-offset', action='store_true')
    offset.add_argument('--rescue', action='store_true')
    p.add_argument('--refined-probe',action='store_true')
    p.add_argument('--adaptive-search',action='store_true')
    p.add_argument('--event-fallback',action='store_true',help='Switch to full gradient on preview exit; require Newton convergence for every accepted step')
    p.add_argument('--dt',type=float,default=.01)
    p.add_argument('--scene',choices=['flag','bunny'],required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int)
    p.add_argument('--level',type=int,default=0)
    p.add_argument('--lower',type=Path)
    p.add_argument('--replay-velocity-from',type=Path,
                   help='Test the velocity accessor against a completed same-mesh L0 trajectory')
    p.add_argument('--method',choices=['direct','velpro'],default='velpro')
    p.add_argument('--tiny-probe',action='store_true',help='Legacy option; rejected by shirt-aligned stopping')
    p.add_argument('--preview-min-alpha',type=float,default=0.,help='Legacy compatibility option; must be 0 with shirt-aligned stopping')
    p.add_argument('--max-newton',type=int,default=300,help='Iteration budget only; does not relax acceptance tolerance')
    p.add_argument('--preconditioner',choices=['diag','mas'],default='diag')
    p.add_argument('--no-ground',action='store_true',help='Omit the ground geometry and its clearance check')
    p.add_argument('--native-package',type=Path,help='Native runtime override. Zero-friction frozen-offset runs default to ordered assembly and sparse LDLT.')
    p.add_argument('--linear-tol-rate',type=float,help='Squared preconditioned residual ratio; defaults to 1e-10, or the reference setting for a velocity replay')
    p.add_argument('--newton-velocity-tol',type=float,help='Velocity decrement tolerance (m/s), using the norm supplied by the native build')
    p.add_argument('--friction',type=float,choices=(0.,.3),help='Cloth self-contact and cloth–bunny friction. Defaults to 0.3 or the replay reference. Ground friction is zero.')
    p.add_argument('--contact-stiffness',type=float,help='Requested IPC normal stiffness for all contact pairs. Fine levels inherit it. The native scene bounds still apply.')
    p.add_argument('--contact-distance', type=float, help='IPC barrier activation distance in meters. Defaults to 0.0007 for the reconstructed bunny, or the saved reference value for a replay. Fine levels inherit it.')
    a=p.parse_args();out=a.output.resolve()
    reference_path=a.replay_velocity_from or a.restart
    reference_meta=json.loads((reference_path/'run.json').read_text()) if reference_path else None
    if a.native_package is None:
        if reference_meta and reference_meta.get('native_package'):
            a.native_package=Path(reference_meta['native_package'])
        elif a.frozen_offset and a.friction == 0:
            a.native_package=ORDERED_PACKAGE
        else:
            a.native_package=FROZEN_PACKAGE if a.frozen_offset else DEFAULT_PACKAGE
    provenance=runtime_provenance(a.native_package)
    a.event_fallback = a.event_fallback or os.environ.get('BUNNY_EVENT_FALLBACK') == '1'
    if a.event_fallback:
        if not a.frozen_offset or a.direction_audit or not provenance.get('event_fallback_control'):
            p.error('--event-fallback requires frozen-offset mode and the event-fallback CUDA runtime')
        os.environ['BUNNY_EVENT_FALLBACK']='1'
    newton_norm=provenance.get('bunny_newton_norm','max_coordinate')
    reference_cfg=reference_meta['config'] if reference_meta else None
    if a.friction is None:
        a.friction=reference_meta['contact_friction']['cloth_bunny'] if reference_meta else .3
    if reference_meta and a.friction!=reference_meta['contact_friction']['cloth_bunny']:
        p.error('A velocity replay must use the friction of its reference')
    reference_kappa=reference_meta.get('requested_contact_stiffness',1e9) if reference_meta else 1e9
    if a.contact_stiffness is None:
        a.contact_stiffness=reference_kappa
    if not np.isfinite(a.contact_stiffness) or a.contact_stiffness<=0:
        p.error('Normal contact stiffness must be positive and finite')
    if reference_meta and a.contact_stiffness!=reference_kappa:
        p.error('A replay must retain the reference normal contact stiffness')
    reference_distance = reference_cfg['contact']['d_hat'] if reference_cfg else BUNNY_CONTACT_DISTANCE
    if a.contact_distance is None:
        a.contact_distance = reference_distance
    if not np.isfinite(a.contact_distance) or a.contact_distance <= 0:
        p.error('Contact activation distance must be positive and finite')
    if reference_meta and a.contact_distance != reference_distance:
        p.error('A replay must retain the reference contact activation distance')
    kappa=a.contact_stiffness
    if a.linear_tol_rate is None:
        a.linear_tol_rate=reference_cfg['linear_system']['tol_rate'] if reference_cfg else BUNNY_LINEAR_TOL_RATE
    if a.newton_velocity_tol is None:
        default_tol=.001 if newton_norm=='free_vertex_rms_3' else NEWTON_VELOCITY_TOL
        a.newton_velocity_tol=reference_cfg['newton']['velocity_tol'] if reference_cfg else default_tol
    if not 0<a.newton_velocity_tol<=NEWTON_VELOCITY_TOL:p.error('Require a positive Newton tolerance at most 0.05 m/s')
    if not 0<a.linear_tol_rate<=LINEAR_TOL_RATE:p.error('Require a positive PCG tolerance ratio at most 1e-3')
    if a.scene!='bunny' or a.level!=0 or a.lower:
        p.error('This entry supports bunny L0 only')
    if (a.level>0 and not a.lower) or not a.refined_probe or not a.adaptive_search:
        p.error('This run requires --refined-probe and --adaptive-search')
    if a.restart and (a.friction!=0 or a.steps!=1 or a.replay_velocity_from or a.direction_audit
                      or reference_meta.get('diagnostic_restart') or reference_meta['velpro_transfer_active']
                      or not 1<=a.restart_frame<reference_meta['completed_steps']):
        p.error('A restart control needs one zero-friction step from an interior frame of an independent L0')
    if a.dump_linear_system and not a.restart:
        p.error('Linear-system export is supported in a one-step restart control')
    if a.rescue and a.direction_audit:p.error('Direction-only diagnosis disables preview; incompatible with rescue')
    if a.dt<=0 or (a.steps is not None and a.steps<1):p.error('Require positive dt and step count')
    expected=provenance['dll_sha256']
    if a.tiny_probe or a.preview_min_alpha!=0:
        p.error('Shirt-aligned stopping requires --preview-min-alpha 0 and no --tiny-probe')
    if a.max_newton<1:p.error('Require positive Newton iteration budget')
    flag=a.scene=='flag';last=9 if flag else 7;thickness=.0004 if flag else .00007
    if not 0<=a.level<=last:p.error('Level outside the paper hierarchy')
    if (a.level>0 and a.method=='velpro') != (a.lower is not None):p.error('VelPro requires exactly the preceding level')
    if a.lower:
        lm=json.loads((a.lower/'run.json').read_text())
        validate_shirt_solver_history(lm)
        if (lm.get('adaptive_line_search')!=a.adaptive_search or lm.get('non_descent_probe_refinement')!=a.refined_probe or lm.get('initial_non_descent_rescue')!=a.rescue or lm.get('exact_offset_gradient')!=a.exact_offset
            or lm['library_sha256']['uipc_backend_cuda.dll'] not in ({expected} if a.level>3 else {expected,'f2e6eed73c7e5192ca2747ec6dfadcccbc6017dc94d351ac11166b4b21d3c826'})):
            p.error('Lower level must use the identical repair mode and native DLL')
    out.mkdir(parents=True,exist_ok=False)
    limit=1.1 if flag else 1.068;onset=1+(limit-1)/2
    dt=a.dt if flag else .01;steps=a.steps or (500 if flag else 400)
    subprocess.run([sys.executable,str(ROOT/'scripts/export_psq_libuipc.py'),'--assets',str(a.assets),
        '--level',str(a.level),'--fine-level',str(last),'--output',str(out/'operator')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/pack_psq_native.py'),'--operator',str(out/'operator/operator.npz'),
        '--output',str(out/'bundle.bin'),'--thickness',str(thickness),'--strain-limit',str(limit),'--strain-onset',str(onset)],check=True)
    for k in list(os.environ):
        if k.startswith('UIPC_PSQ_'):os.environ.pop(k)
    os.environ.update(UIPC_PSQ_BUNDLE=str(out/'bundle.bin'),UIPC_PSQ_MODE='proxy',UIPC_PSQ_STRAIN_CAP='directional',
        UIPC_PSQ_PREVIEW_STOP=str(int(a.level<last)),UIPC_PSQ_REJECT_INF_TRIAL='1')
    if a.adaptive_search:os.environ['UIPC_PSQ_ADAPTIVE_SEARCH']='1'
    if a.refined_probe:os.environ['UIPC_PSQ_REFINE_PROBES']='1'
    if a.rescue and a.level<last:os.environ['UIPC_PSQ_UNRESOLVED_RESCUE']='1'
    if a.rescue:os.environ['UIPC_PSQ_RESCUE']='1'
    if a.exact_offset:os.environ['UIPC_PSQ_EXACT_OFFSET']='1'
    if a.frozen_offset:os.environ['UIPC_PSQ_FROZEN_ONLY']='1'
    if a.direction_audit:
        os.environ['UIPC_PSQ_PREVIEW_STOP']='0'
        os.environ['UIPC_PSQ_DIRECTION_AUDIT']='1'
    from libuipc_native_runtime import load,loaded_libraries
    uipc=load(a.native_package)
    from uipc import view,builtin
    from uipc.core import Engine,Scene,World,FiniteElementStateAccessorFeature
    from uipc.geometry import trimesh,label_surface,ground
    from uipc.constitution import NeoHookeanShell,DiscreteShellBending,ElasticModuli2D,Empty,FiniteElementExternalForce
    d=np.load(a.assets/f'mesh_{a.level}.npz');rest,faces=d['vertices'],d['faces']
    lower=U=None
    if a.lower:
        from scipy.sparse import load_npz
        if (not lm['completed'] or lm['completed_steps']<steps or lm['dt']!=dt or
            lm['level']!=a.level-1 or lm['scene']!=a.scene or Path(lm['assets'])!=a.assets.resolve()):
            raise ValueError('Previous level must be completed and use the same scene, assets, dt and time window')
        lower=np.load(a.lower/'positions.npy',mmap_mode='r')
        U=load_npz(a.assets/f'map_{a.level-1}_{a.level}.npz')
        if U.shape!=(len(rest),lower.shape[1]):raise ValueError('Transfer shape mismatch')
    cfg=Scene.default_config();cfg['dt']=dt;cfg['gravity']=[[0.],[0.],[-9.81]]
    cfg['newton'].update(velocity_tol=a.newton_velocity_tol,velocity_tol_relative=0.,use_adaptive_tol=0,max_iter=a.max_newton)
    cfg['newton']['semi_implicit']['enable']=0
    cfg['line_search'].update(max_iter=LINE_SEARCH_MAX_ITER,report_energy=0)
    cfg['linear_system']['tol_rate']=a.linear_tol_rate
    cfg['linear_system']['fem_preconditioner']=a.preconditioner
    cfg['contact']['d_hat']=a.contact_distance;cfg['contact']['eps_velocity']=.001
    cfg['contact']['friction']['enable']=int(a.friction>0)
    cfg['extras']['strict_mode']['enable']=0
    if a.dump_linear_system:cfg['extras']['debug']['dump_linear_system']=1
    # Retain the three-scale non-descent evidence in coarse-level logs, as in the shirt demo.
    uipc.Timer.disable_all();uipc.Logger.set_level(uipc.Logger.Level.Info if a.level<last else uipc.Logger.Level.Warn)
    engine=Engine('cuda',str(out/'backend'));world=World(engine);scene=Scene(cfg)
    scene.contact_tabular().default_model(a.friction,kappa)
    mesh=trimesh(np.ascontiguousarray(rest),np.ascontiguousarray(faces,dtype=np.int32));label_surface(mesh)
    NeoHookeanShell().apply_to(mesh,ElasticModuli2D.youngs_poisson(30000.,.3),200.,thickness/2)
    DiscreteShellBending().apply_to(mesh,30000.,.3)
    # Probabilistic simplification perturbs the nominal straight edge by ~15 um.
    # Select the boundary chain, never a lone numerical x-minimum vertex.
    pins=np.zeros(len(rest),bool)
    if flag:
        edges=np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
        unique,count=np.unique(edges,axis=0,return_counts=True)
        boundary=np.unique(unique[count==1])
        pins[boundary]=rest[boundary,0]<1e-4
        if pins.sum()<3 or np.ptp(rest[pins,2])<.95:raise ValueError('Incomplete flagpole boundary')
    if flag:
        fixed=mesh.vertices().find(builtin.is_fixed)
        if fixed is None:fixed=mesh.vertices().create(builtin.is_fixed,0)
        view(fixed).reshape(-1)[:]=pins
        FiniteElementExternalForce().apply_to(mesh,np.zeros(3))
    obj=scene.objects().create(f'cloth_L{a.level}');slot,_=obj.geometries().create(mesh)
    inv=frames(rest,faces)[2];area=frames(rest,faces)[3];mass=np.zeros(len(rest))
    np.add.at(mass,faces.ravel(),np.repeat(area*thickness*200/3,3))
    if flag:
        def animate(info):
            geo=info.geo_slots()[0].geometry();force=np.zeros_like(rest)
            force[:,1]=mass*(8+5*np.sin(2*np.pi*1.3*(info.frame()+(a.restart_frame if a.restart else 0))*dt));force[pins]=0
            view(geo.vertices().find('external_force'))[:]=force.reshape(-1,3,1)
            view(geo.vertices().find(builtin.is_constrained)).reshape(-1)[:]=~pins
        scene.animator().insert(obj,animate)
    else:
        b=np.load(a.assets/'bunny.npz');ob=trimesh(np.ascontiguousarray(b['vertices']),np.ascontiguousarray(b['faces'],dtype=np.int32))
        label_surface(ob);Empty().apply_to(ob,200.,thickness/2)
        fixed=ob.vertices().find(builtin.is_fixed)
        if fixed is None:fixed=ob.vertices().create(builtin.is_fixed,1)
        view(fixed)[:]=1
        tab=scene.contact_tabular();oc=tab.create('fixed_bunny');tab.insert(oc,oc,a.friction,kappa,False);oc.apply_to(ob)
        bunny_slot,_=scene.objects().create('fixed_bunny').geometries().create(ob)
        if not a.no_ground:
            floor=ground(0.0,np.array([0.,0.,1.]))
            floor_contact=tab.create('frictionless_floor')
            tab.insert(tab.default_element(),floor_contact,0.0,kappa,True)
            tab.insert(oc,floor_contact,0.0,kappa,False)
            tab.insert(floor_contact,floor_contact,0.0,kappa,False)
            floor_contact.apply_to(floor)
            scene.objects().create('ground_z0').geometries().create(floor)

    meta=dict(scene=a.scene,assets=str(a.assets.resolve()),level=a.level,method='psq_coarse' if lower is None else 'velpro',completed=False,status='initializing',steps=steps,completed_steps=0,
        dt=dt,vertices=len(rest),faces=len(faces),proxy_vertices=35000 if flag else 90000,config=cfg,pins=int(pins.sum()),
        native_steps=0,preview_steps=0,material=dict(young=30000.,poisson=.3,density=200.,thickness=thickness,strain_limit=limit),
        scope=f'Single-level L{a.level} trajectory at paper endpoint scale; no finer-level trajectory implied',
        paper_matched=['10 levels, ~550 to 35000 vertices, thickness .4mm, strain limit 10%, final duration target 20s'] if flag else ['8 levels, ~1500 to 90000 vertices, thickness .07mm, strain limit 6.8%, dt .01s'],
        reconstruction_choices=['E=30000, nu=.3, rho=200; source assets not authenticated author meshes',
            f'Newton velocity decrement norm {newton_norm}, tolerance {a.newton_velocity_tol:g} m/s; native adaptive contact, fixed strain barrier stiffness',
            'flag: 1.5m x 1m, left edge fixed, body acceleration y=8+5*sin(2*pi*1.3*t), dt=.04 inferred from PD2024' if flag else f'bunny: previously recorded local geometry and {steps*dt:g}s local observation window; author duration not recovered'],
        stop_policy='Shirt-aligned: no semi-implicit or forced updates; coarse verified non-descent only; no small-alpha stop or tiny-alpha probe; finest requires native convergence; reject budget exhaustion without verified preview',
        source_files_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.assets/f'mesh_{a.level}.npz',a.assets/f'mesh_{last}.npz',Path(__file__),Path(__file__).with_name('paper_solver_settings.py')]})
    meta.update(preview_min_alpha=a.preview_min_alpha,small_step_preview_steps=0,
        native_package=str(a.native_package.resolve()) if a.native_package else None)
    meta['ground']=dict(enabled=not a.no_ground,height_m=None if a.no_ground else 0.,normal=[0.,0.,1.],cloth_friction=0.,bunny_ground_contact=False,geometry='none' if a.no_ground else 'infinite halfplane')
    meta['requested_contact_stiffness']=kappa
    meta['contact_friction']=dict(cloth_bunny=a.friction,cloth_self=a.friction,cloth_ground=0.)
    meta['velpro_transfer_active']=False
    meta['scene_change']='Ground omitted; fixed bunny and cloth self-contact only' if a.no_ground else 'Fixed frictionless ground added at bunny base; explicit local modification, not the original bunny scene'
    meta['bundle_sha256']=hashlib.sha256((out/'bundle.bin').read_bytes()).hexdigest()
    if not flag:meta['source_files_sha256'][str(a.assets/'bunny.npz')]=hashlib.sha256((a.assets/'bunny.npz').read_bytes()).hexdigest()
    if lower is not None:
        lower_cfg=json.loads(json.dumps(lm['config']))
        previous_budget=lower_cfg['newton']['max_iter'];lower_cfg['newton']['max_iter']=a.max_newton
        previous_preconditioner=lower_cfg['linear_system']['fem_preconditioner']
        lower_cfg['linear_system']['fem_preconditioner']=a.preconditioner
        if lower_cfg!=cfg or lm['material']!=meta['material']:raise ValueError('Physical/solver configuration differs from lower level')
        meta['newton_iteration_budgets']=dict(previous_level=previous_budget,current_level=a.max_newton,
            tolerance_unchanged=True)
        meta['linear_preconditioners']=dict(previous_level=previous_preconditioner,current_level=a.preconditioner,
            linear_tolerance_unchanged=True)
        meta.update(lower_run=str(a.lower.resolve()),lower_sha256=hashlib.sha256((a.lower/'positions.npy').read_bytes()).hexdigest(),
            map_sha256=hashlib.sha256((a.assets/f'map_{a.level-1}_{a.level}.npz').read_bytes()).hexdigest(),
            predictor='x_level[n] + U @ (x_lower[n]-x_lower[n-1]); initial velocity zero; fixed vertices zeroed')
    def save():(out/'run.json').write_text(json.dumps(meta,indent=2))
    save();(out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'solver_settings_snapshot.py').write_bytes(Path(__file__).with_name('paper_solver_settings.py').read_bytes())
    positions=np.lib.format.open_memmap(out/'positions.npy',mode='w+',dtype='float64',shape=(steps+1,len(rest),3));positions[0]=rest
    meta['diagnostic_restart']=dict(run=str(a.restart),frame=a.restart_frame) if a.restart else None
    meta['direction_audit']=a.direction_audit
    meta['frozen_offset_only']=a.frozen_offset and not a.event_fallback
    meta['event_fallback_enabled']=a.event_fallback
    meta['require_newton_convergence']=a.event_fallback
    meta['frozen_direction_derivative']=a.frozen_offset and provenance.get('frozen_direction_derivative', False)
    meta['representable_preview']=a.frozen_offset and provenance.get('representable_preview', False)
    meta['exact_offset_gradient']=a.exact_offset
    meta['energy_checked_convergence']=provenance.get('energy_checked_convergence',False)
    meta['checked_search_exhaustion']=provenance.get('checked_search_exhaustion',False)
    if meta['checked_search_exhaustion']:
        meta['exhausted_search_acceptance']=provenance.get('exhausted_search_acceptance','resolved_decrease')
    meta['bunny_linear_tol_rate']=a.linear_tol_rate
    meta['bunny_newton_velocity_tol']=a.newton_velocity_tol
    meta['bunny_newton_norm']=newton_norm
    meta['bunny_linear_solver']=provenance.get('bunny_linear_solver','native_pcg')
    meta['ordered_fem_assembly']=provenance.get('ordered_fem_assembly', False)
    meta['velocity_only_preserves_friction']=provenance.get('velocity_only_preserves_friction',False)
    meta['initial_non_descent_rescue']=a.rescue
    meta['non_descent_probe_refinement']=a.refined_probe
    meta['adaptive_line_search']=a.adaptive_search
    meta['unresolved_direction_rescue']=a.rescue and a.level<last
    meta['per_iteration_acceptance_required']=True
    meta['line_search_policy']='Safeguarded quadratic interpolation of two failed trials; next alpha clamped to [0.1,0.5] times last; unchanged max_iter'
    meta['probe_policy']='Up to three decade refinements of resolved positive slopes; unchanged noise and 25% consistency limits'
    meta['lower_backend_sha256']=lm['library_sha256']['uipc_backend_cuda.dll'] if a.lower else None
    meta['algorithm_note']=('Exact PSQ gradient retry at first-iteration non-descent or unresolved coarse backtracking; all retries restore x_k and require energy acceptance; explicit repair extension'
        if a.rescue else 'Full nonlinear PSQ Jacobian gradient; separate repair extension, not frozen-offset paper approximation'
        if a.exact_offset else 'Original frozen-offset gradient')
    meta['time_discretization_note']=f'Actual integration dt={dt:.17g}; bunny dt=0.01. No hidden substeps.'
    if a.event_fallback:
        meta['gradient_strategy']='Frozen until preview exit; restore current iterate, then use full PSQ gradient for remainder of step; reset at next step'
        meta['algorithm_note']=meta['gradient_strategy']+'; every accepted step must converge'
    if a.replay_velocity_from:
        from scipy.sparse import eye
        source=a.replay_velocity_from.resolve()
        reference=json.loads((source/'run.json').read_text())
        if (not reference['completed'] or reference['level']!=0 or reference['completed_steps']!=steps
            or reference['velpro_transfer_active'] or reference['bundle_sha256']!=meta['bundle_sha256']
            or reference['config']!=cfg or reference['exact_offset_gradient']!=a.exact_offset
            or reference['assets']!=meta['assets'] or reference['ground']!=meta['ground']):
            raise ValueError('Identity replay requires the same completed L0 problem')
        lower=np.load(source/'positions.npy',mmap_mode='r')
        np.testing.assert_array_equal(lower[0],rest)
        U=eye(len(rest),format='csr')
        meta.update(method='velpro_identity_replay',velpro_transfer_active=True,replay_reference=str(source),
                    replay_sha256=hashlib.sha256((source/'positions.npy').read_bytes()).hexdigest())
    records=[];start=time.perf_counter()
    try:
        world.init(scene)
        if not world.is_valid():raise RuntimeError('Invalid initial scene')
        feat=world.features().find(FiniteElementStateAccessorFeature)
        offset=int(view(slot.geometry().meta().find('backend_fem_vertex_offset')).ravel()[0])
        state=feat.create_geometry(offset,len(rest));state.vertices().create(builtin.position,np.zeros(3))
        if lower is not None:
            velocity_state=feat.create_geometry(offset,len(rest));velocity_state.vertices().create(builtin.velocity,np.zeros(3))
            check_state=feat.create_geometry(offset,len(rest));check_state.vertices().create(builtin.velocity,np.zeros(3));check_state.vertices().create(builtin.position,np.zeros(3))
        if a.restart:
            audit=json.loads((a.restart/'continuation_audit.json').read_text())
            reference_solver_config=json.loads(json.dumps(reference_meta['config']))
            source_linear_tol=reference_solver_config['linear_system']['tol_rate']
            source_newton_tol=reference_solver_config['newton']['velocity_tol']
            if a.linear_tol_rate>source_linear_tol or a.newton_velocity_tol>source_newton_tol:
                raise ValueError('A one-step tolerance control can only tighten the source settings')
            reference_solver_config['linear_system']['tol_rate']=a.linear_tol_rate
            reference_solver_config['newton']['velocity_tol']=a.newton_velocity_tol
            reference_solver_config['extras']['debug']['dump_linear_system']=cfg['extras']['debug']['dump_linear_system']
            if (not audit['passed'] or reference_meta['level']!=0 or reference_meta['scene']!='bunny'
                    or reference_solver_config!=cfg or reference_meta['material']!=meta['material']
                    or reference_meta['bundle_sha256']!=meta['bundle_sha256'] or reference_meta['assets']!=meta['assets']
                    or reference_meta['contact_friction']!=meta['contact_friction'] or reference_meta['ground']!=meta['ground']
                    or reference_meta['library_sha256']['uipc_backend_cuda.dll'] not in (expected,provenance.get('reference_dll_sha256'))
                    or any(reference_meta[key]!=meta[key] for key in ('exact_offset_gradient','adaptive_line_search',
                        'initial_non_descent_rescue','non_descent_probe_refinement'))):
                raise ValueError('One-step control requires the same audited zero-friction L0 problem')
            history=np.load(a.restart/'positions.npy',mmap_mode='r');n=a.restart_frame
            start_velocity=(history[n]-history[n-1])/dt
            restart_state=feat.create_geometry(offset,len(rest))
            restart_state.vertices().create(builtin.position,np.zeros(3));restart_state.vertices().create(builtin.velocity,np.zeros(3))
            view(restart_state.positions())[:]=history[n].reshape(-1,3,1)
            view(restart_state.vertices().find(builtin.velocity))[:]=start_velocity.reshape(-1,3,1)
            feat.copy_from(restart_state);feat.copy_to(restart_state)
            np.testing.assert_array_equal(view(restart_state.positions()).reshape(-1,3),history[n])
            np.testing.assert_array_equal(view(restart_state.vertices().find(builtin.velocity)).reshape(-1,3),start_velocity)
            positions[0]=history[n]
            with (a.restart/'positions.npy').open('rb') as src:
                meta['diagnostic_restart'].update(reference_sha256=hashlib.file_digest(src,'sha256').hexdigest(),
                                                  initial_state_readback_verified=True,source_linear_tol_rate=source_linear_tol,
                                                  source_newton_velocity_tol=source_newton_tol,
                                                  source_newton_norm=reference_meta.get('bunny_newton_norm','max_coordinate'))
        libs=loaded_libraries()
        meta.update(status='running',loaded_libraries=libs,
            tiny_probe=a.tiny_probe,library_sha256={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in libs.items()});save()
        for step in range(1,steps+1):
            tick=time.perf_counter();injection_error=position_error=0.
            if lower is not None:
                n=step-1;velocity=np.zeros_like(rest) if n==0 else U@((lower[n]-lower[n-1])/dt)
                velocity[pins]=0
                view(velocity_state.vertices().find(builtin.velocity))[:]=velocity.reshape(-1,3,1)
                feat.copy_from(velocity_state);feat.copy_to(check_state)
                injection_error=float(np.max(np.abs(view(check_state.vertices().find(builtin.velocity)).reshape(-1,3)-velocity)))
                position_error=float(np.max(np.abs(view(check_state.positions()).reshape(-1,3)-positions[n])))
                if max(injection_error,position_error)>1e-12:raise RuntimeError('VelPro velocity injection or position preservation failed')
            world.advance()
            if not world.is_valid():
                meta['rejected_step']=dict(step=step,native=engine.frame_stats())
                raise RuntimeError(f'Native solver rejected step {step}. See the native log for the failed search.')
            world.retrieve();feat.copy_to(state)
            x=np.array(view(state.positions())).reshape(-1,3)
            stats=engine.frame_stats();small=bool(stats.get('preview_small_step'));preview=bool(stats.get('preview_non_descent')) or small
            valid_preview=(not a.event_fallback and a.level<last and bool(stats.get('preview_non_descent')) and not small and not stats.get('converged') and
                stats.get('termination_reason')=='preview_non_descent' and stats.get('last_line_search_alpha')==0)
            if (not world.is_valid() or not stats.get('completed') or (preview and not valid_preview) or stats.get('hit_newton_limit') or
                (stats.get('hit_line_search_limit') and not valid_preview) or not(stats.get('converged') or valid_preview)):
                meta['rejected_step']=dict(step=step,native=stats)
                np.save(out/'rejected_positions.npy',x)
                raise RuntimeError(f'Rejected step {step}')
            tri=x[faces];F=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=2)@inv
            stretch=float(np.linalg.svd(F,compute_uv=False).max())
            if not np.isfinite(x).all() or stretch>limit+1e-9:raise RuntimeError('Nonfinite state or strain violation')
            pin_error=0. if not flag else float(np.max(np.abs(x[pins]-rest[pins])))
            if pin_error>1e-10:
                meta['rejected_step']=dict(step=step,native=stats,pin_error_m=pin_error)
                np.save(out/'rejected_positions.npy',x)
                raise RuntimeError('Pinned boundary moved')
            obstacle_error=0. if flag else float(np.max(np.abs(view(bunny_slot.geometry().positions()).reshape(-1,3)-b['vertices'])))
            if obstacle_error>1e-10:
                meta['rejected_step']=dict(step=step,native=stats,obstacle_error_m=obstacle_error)
                np.save(out/'rejected_positions.npy',x)
                raise RuntimeError('Fixed bunny moved')
            minimum_z=float(x[:,2].min())
            if not a.no_ground and minimum_z < thickness/2-1e-9:
                np.save(out/'rejected_positions.npy',x)
                raise RuntimeError(f'Cloth crossed floor thickness clearance: {minimum_z}')
            positions[step]=x;positions.flush()
            meta['completed_steps']=step;meta['native_steps']+=int(bool(stats.get('converged')));meta['preview_steps']+=int(preview)
            meta['small_step_preview_steps']+=int(small)
            records.append(dict(step=step,native=stats,maximum_stretch=stretch,seconds=time.perf_counter()-tick,
                velocity_readback_error=injection_error,position_injection_error=position_error,obstacle_error=obstacle_error,minimum_z=minimum_z,maximum_z=float(x[:,2].max())));save()
            (out/'steps.json').write_text(json.dumps(records,indent=2))
            if step%10==0:print(f'ACCEPTED {step}/{steps}, native={meta["native_steps"]}, preview={meta["preview_steps"]}',flush=True)
        meta.update(status='completed',completed=True)
    except Exception as e:meta.update(status='failed',error=str(e));raise
    finally:
        positions.flush();meta['wall_seconds']=time.perf_counter()-start;save();(out/'steps.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
