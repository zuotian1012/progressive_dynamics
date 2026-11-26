"""Bunny finer levels with an explicit frictionless floor; isolated from flag/shirt runs.

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
from bunny_runtime import continuation_package, runtime_provenance
from paper_solver_settings import (
    LINE_SEARCH_MAX_ITER,
    validate_shirt_solver_history,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restart',type=Path)
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
    p.add_argument('--consistency-weight',type=float,default=0.,help='PD++ potential coefficient w in w ||x-P(x_lower[next])||_M^2 (s^-2)')
    p.add_argument('--method',choices=['semi','full','velpro'],default='velpro')
    p.add_argument('--tiny-probe',action='store_true',help='Legacy option; rejected by shirt-aligned stopping')
    p.add_argument('--preview-min-alpha',type=float,default=0.,help='Legacy compatibility option; must be 0 with shirt-aligned stopping')
    p.add_argument('--max-newton',type=int,default=300,help='Iteration budget only; does not relax acceptance tolerance')
    p.add_argument('--newton-velocity-tol',type=float,help='Override the inherited Newton decrement tolerance for a tighter paired control (m/s)')
    p.add_argument('--preconditioner',choices=['diag','mas'],default='diag')
    p.add_argument('--no-cuda-graph',action='store_true',help='Use plain launches for the same fused PCG iterations')
    p.add_argument('--solver-trace',action='store_true',help='Record per-Newton linear system and line search logs on the finest level')
    p.add_argument('--native-package',type=Path,help='CUDA runtime override. Defaults to the corrected parent build.')
    p.add_argument('--velocity-map',choices=['intrinsic','psq-jacobian'],default='intrinsic',
                   help='VelPro velocity transport: frozen-offset U or exact DP(x) applied to coarse velocity')
    a=p.parse_args();out=a.output.resolve()
    exact_velocity=a.velocity_map=='psq-jacobian'
    if exact_velocity and (a.method!='velpro' or a.consistency_weight):
        p.error('The velocity-map control requires unpenalized VelPro')
    if not np.isfinite(a.consistency_weight) or a.consistency_weight<0:
        p.error('Require finite nonnegative consistency weight')
    if a.scene!='bunny' or not 1<=a.level<=7 or not a.lower:
        p.error('This entry supports bunny L1-L7 with a completed preceding ground run')
    if (a.level>0 and not a.lower) or not a.refined_probe or not a.adaptive_search or a.restart:
        p.error('This run requires --refined-probe and --adaptive-search')
    if a.rescue and a.direction_audit:p.error('Direction-only diagnosis disables preview; incompatible with rescue')
    if a.dt<=0 or (a.steps is not None and a.steps<1):p.error('Require positive dt and step count')
    lm=json.loads((a.lower/'run.json').read_text())
    ground_enabled=bool(lm.get('ground',{}).get('enabled'))
    a.native_package=a.native_package or continuation_package(lm)
    provenance=runtime_provenance(a.native_package)
    a.event_fallback = a.event_fallback or os.environ.get('BUNNY_EVENT_FALLBACK') == '1'
    if a.event_fallback:
        if not a.frozen_offset or a.direction_audit or not provenance.get('event_fallback_control'):
            p.error('--event-fallback requires frozen-offset mode and the event-fallback CUDA runtime')
        os.environ['BUNNY_EVENT_FALLBACK']='1'
    expected=provenance['dll_sha256']
    newton_norm=provenance.get('bunny_newton_norm','max_coordinate')
    if a.tiny_probe or a.preview_min_alpha!=0:
        p.error('Shirt-aligned stopping requires --preview-min-alpha 0 and no --tiny-probe')
    if a.max_newton<1:p.error('Require positive Newton iteration budget')
    flag=a.scene=='flag';last=9 if flag else 7;thickness=.0004 if flag else .00007
    if not 0<=a.level<=last:p.error('Level outside the paper hierarchy')
    diagonal=a.method in ('semi','full')
    if diagonal and a.consistency_weight:
        p.error('The diagonal PD experiment requires zero consistency weight')
    if a.lower:
        friction=lm['contact_friction']['cloth_bunny']
        kappa=lm.get('requested_contact_stiffness',1e9)
        if friction not in (0.,.3) or lm['contact_friction']!=dict(cloth_bunny=friction,cloth_self=friction,cloth_ground=0.):
            p.error('Expected a supported bunny friction configuration on the preceding level')
        parent_tol=lm['config']['newton']['velocity_tol']
        if a.newton_velocity_tol is None:
            a.newton_velocity_tol=min(.001,parent_tol) if newton_norm=='free_vertex_rms_3' else parent_tol
        if not 0<a.newton_velocity_tol<=parent_tol:p.error('A fine-level tolerance control must be at least as strict as its parent')
        validate_shirt_solver_history(lm)
        # Eq. (8) makes full and semi identical at L1.
        valid_parent = (lm['level']==0 or lm['method']==a.method or
                        (lm['level']==1 and {a.method,lm['method']}=={'semi','full'}))
        if a.method=='velpro' and not a.consistency_weight and (
            lm.get('consistency_weight',0.) or not valid_parent or
            (lm['level'] and lm.get('velocity_map','intrinsic')!=a.velocity_map)):
            p.error('Unpenalized VelPro requires a matching unpenalized parent trajectory')
        if diagonal and (lm.get('consistency_weight',0.) or not valid_parent):
            p.error('Expected an unpenalized parent using the same diagonal stencil')
        if ground_enabled:
            if not json.loads((a.lower/'continuation_audit.json').read_text())['passed']:
                p.error('Lower ground run must have passed its full audit')
        else:
            parent_audit=json.loads((a.lower/'no_ground_audit.json').read_text())
            if (not parent_audit['no_ground'] or parent_audit['intersecting_states']
                or parent_audit['saved_states_checked']!=lm['completed_steps']+1
                or parent_audit['positions_sha256']!=hashlib.sha256((a.lower/'positions.npy').read_bytes()).hexdigest()
                or not json.loads((a.lower/'audit.json').read_text())['accepted_prefix_checks_passed']
                or not json.loads((a.lower/'report/non_descent_audit.json').read_text())['all_accepted_preview_evidence_verified']):
                p.error('Lower no-ground run must have passed the saved-state and preview audits')
        parent_builds={expected}
        if a.frozen_offset and provenance.get('representable_preview'):
            parent_builds.add(provenance['reference_dll_sha256'])
        if provenance.get('linear_solver_parent_sha256') and a.level==last:
            parent_builds.add(provenance['linear_solver_parent_sha256'])
        if provenance.get('velocity_only_preserves_friction') or provenance.get('paired_corrected_dll_sha256'):
            # L0 never writes velocity. Reuse it for a controlled accessor A/B.
            if lm['level']==0 and not lm['velpro_transfer_active']:
                parent_builds.add(runtime_provenance(lm['native_package'])['dll_sha256'])
        elif a.level<=3:
            parent_builds.add('f2e6eed73c7e5192ca2747ec6dfadcccbc6017dc94d351ac11166b4b21d3c826')
        if (lm.get('adaptive_line_search')!=a.adaptive_search or lm.get('non_descent_probe_refinement')!=a.refined_probe or lm.get('initial_non_descent_rescue')!=a.rescue or lm.get('exact_offset_gradient')!=a.exact_offset
            or lm['library_sha256']['uipc_backend_cuda.dll'] not in parent_builds):
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
    uses_grandparent=a.method=='full' and a.level>=2
    if uses_grandparent:
        from progressive.prolongation import ShellProlongation
        ancestor_path=Path(lm['lower_run'])/'positions.npy'
        with ancestor_path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=lm['lower_sha256']:
                raise ValueError('Grandparent trajectory differs from the one used by the parent')
        ancestor=np.load(ancestor_path,mmap_mode='r')
        pair=f'{a.level-2}_{a.level-1}'
        ancestor_mesh=np.load(a.assets/f'mesh_{a.level-2}.npz')
        parent_rest=np.load(a.assets/f'mesh_{a.level-1}.npz')['vertices']
        ancestor_transfer=ShellProlongation(ancestor_mesh['vertices'],parent_rest,ancestor_mesh['faces'],
            load_npz(a.assets/f'map_{pair}.npz'),np.load(a.assets/f'anchors_{pair}.npy'))
    cfg=Scene.default_config();cfg['dt']=dt;cfg['gravity']=[[0.],[0.],[-9.81]]
    cfg['newton'].update(velocity_tol=a.newton_velocity_tol,velocity_tol_relative=0.,use_adaptive_tol=0,max_iter=a.max_newton)
    cfg['newton']['semi_implicit']['enable']=0
    cfg['line_search'].update(max_iter=LINE_SEARCH_MAX_ITER,report_energy=0)
    cfg['linear_system']['tol_rate']=lm['config']['linear_system']['tol_rate']
    cfg['linear_system']['fem_preconditioner']=a.preconditioner
    if a.no_cuda_graph:cfg['linear_system']['use_cuda_graph']=0
    cfg['contact']['d_hat']=lm['config']['contact']['d_hat'];cfg['contact']['eps_velocity']=.001
    cfg['contact']['friction']['enable']=int(friction>0)
    cfg['extras']['strict_mode']['enable']=0
    # Retain the three-scale non-descent evidence in coarse-level logs, as in the shirt demo.
    uipc.Timer.disable_all();uipc.Logger.set_level(uipc.Logger.Level.Info if a.level<last or a.solver_trace or provenance.get('energy_checked_convergence') else uipc.Logger.Level.Warn)
    engine=Engine('cuda',str(out/'backend'));world=World(engine);scene=Scene(cfg)
    scene.contact_tabular().default_model(friction,kappa)
    mesh=trimesh(np.ascontiguousarray(rest),np.ascontiguousarray(faces,dtype=np.int32));label_surface(mesh)
    NeoHookeanShell().apply_to(mesh,ElasticModuli2D.youngs_poisson(30000.,.3),200.,thickness/2)
    DiscreteShellBending().apply_to(mesh,30000.,.3)
    if a.consistency_weight or diagonal or exact_velocity:
        from progressive.prolongation import ShellProlongation
        coarse=np.load(a.assets/f'mesh_{a.level-1}.npz')
        anchor_file=a.assets/f'anchors_{a.level-1}_{a.level}.npy'
        prolong=ShellProlongation(coarse['vertices'],rest,coarse['faces'],U,np.load(anchor_file))
    if a.consistency_weight:
        from uipc.constitution import SoftPositionConstraint
        strength_ratio=2*dt**2*a.consistency_weight
        SoftPositionConstraint().apply_to(mesh,strength_ratio)
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
    target_frame=0
    if a.consistency_weight:
        scene.animator().substep(1)
        def update_target(info):
            nonlocal target_frame
            target_frame=info.frame()
            if target_frame!=step:raise RuntimeError('Consistency target frame mismatch')
            geo=info.geo_slots()[0].geometry()
            view(geo.vertices().find(builtin.is_constrained))[:]=1
            view(geo.vertices().find(builtin.aim_position))[:]=prolong.position(lower[target_frame]).reshape(-1,3,1)
        scene.animator().insert(obj,update_target)
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
        tab=scene.contact_tabular();oc=tab.create('fixed_bunny');tab.insert(oc,oc,friction,kappa,False);oc.apply_to(ob)
        bunny_slot,_=scene.objects().create('fixed_bunny').geometries().create(ob)
        if ground_enabled:
            floor=ground(0.0,np.array([0.,0.,1.]))
            floor_contact=tab.create('frictionless_floor')
            tab.insert(tab.default_element(),floor_contact,0.0,kappa,True)
            tab.insert(oc,floor_contact,0.0,kappa,False)
            tab.insert(floor_contact,floor_contact,0.0,kappa,False)
            floor_contact.apply_to(floor)
            scene.objects().create('ground_z0').geometries().create(floor)

    meta=dict(scene=a.scene,assets=str(a.assets.resolve()),level=a.level,method='psq_coarse' if lower is None else a.method,completed=False,status='initializing',steps=steps,completed_steps=0,
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
    meta['ground']=dict(lm['ground'])
    meta['requested_contact_stiffness']=kappa
    meta['bunny_linear_solver']=provenance.get('bunny_linear_solver','native_pcg')
    meta['ordered_fem_assembly']=provenance.get('ordered_fem_assembly', False)
    meta['contact_friction']=dict(cloth_bunny=friction,cloth_self=friction,cloth_ground=0.)
    meta['velpro_transfer_active']=a.method=='velpro'
    meta['velocity_map']=a.velocity_map
    meta['consistency_weight']=a.consistency_weight
    if a.consistency_weight:
        meta['consistency']=dict(target='PSQ P_adjacent(x_lower[step])',strength_ratio=strength_ratio,animation_substeps=1)
    if a.consistency_weight or diagonal or exact_velocity:
        for path in [anchor_file,a.assets/f'mesh_{a.level-1}.npz',ROOT/'progressive/prolongation.py']:
            meta['source_files_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    meta['scene_change']='Fixed frictionless ground added at bunny base; explicit local modification, not the original bunny scene' if ground_enabled else 'Ground omitted; fixed bunny and cloth self-contact only'
    meta['bundle_sha256']=hashlib.sha256((out/'bundle.bin').read_bytes()).hexdigest()
    if not flag:meta['source_files_sha256'][str(a.assets/'bunny.npz')]=hashlib.sha256((a.assets/'bunny.npz').read_bytes()).hexdigest()
    if lower is not None:
        lower_cfg=json.loads(json.dumps(lm['config']))
        previous_budget=lower_cfg['newton']['max_iter'];lower_cfg['newton']['max_iter']=a.max_newton
        lower_cfg['newton']['velocity_tol']=a.newton_velocity_tol
        previous_preconditioner=lower_cfg['linear_system']['fem_preconditioner']
        lower_cfg['linear_system']['fem_preconditioner']=a.preconditioner
        previous_graph=lower_cfg['linear_system']['use_cuda_graph']
        lower_cfg['linear_system']['use_cuda_graph']=cfg['linear_system']['use_cuda_graph']
        if lower_cfg!=cfg or lm['material']!=meta['material'] or lm['ground']!=meta['ground'] or lm['contact_friction']!=meta['contact_friction']:raise ValueError('Physical/solver configuration differs from lower level')
        meta['newton_iteration_budgets']=dict(previous_level=previous_budget,current_level=a.max_newton,
            tolerance_unchanged=a.newton_velocity_tol==parent_tol,
            norm_unchanged=lm.get('bunny_newton_norm','max_coordinate')==newton_norm)
        meta['bunny_newton_velocity_tol']=a.newton_velocity_tol
        meta['bunny_newton_norm']=newton_norm
        meta['linear_preconditioners']=dict(previous_level=previous_preconditioner,current_level=a.preconditioner,
            linear_tolerance_unchanged=True)
        meta['cuda_graph_modes']=dict(previous_level=previous_graph,current_level=cfg['linear_system']['use_cuda_graph'])
        meta.update(lower_run=str(a.lower.resolve()),lower_sha256=hashlib.sha256((a.lower/'positions.npy').read_bytes()).hexdigest(),
            map_sha256=hashlib.sha256((a.assets/f'map_{a.level-1}_{a.level}.npz').read_bytes()).hexdigest(),
            predictor=('P(x_lower[n])' if diagonal else 'x_level[n]') + ' + U @ (x_lower[n]-x_lower[n-1]); initial velocity zero; fixed vertices zeroed')
    if exact_velocity:
        meta['predictor']='x_level[n] + DP(x_lower[n]) @ (x_lower[n]-x_lower[n-1]); zero initial velocity'
    if uses_grandparent:
        meta.update(predictor='P(x_lower[n]) + U @ (x_lower[n] - P_ancestor(x_ancestor[n-1])); zero initial velocity',
                    grandparent_run=str(ancestor_path.parent),grandparent_sha256=lm['lower_sha256'])
        for path in (a.assets/f'map_{pair}.npz', a.assets/f'anchors_{pair}.npy', a.assets/f'mesh_{a.level-2}.npz'):
            meta['source_files_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
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
    meta['bunny_linear_tol_rate']=cfg['linear_system']['tol_rate']
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
            history=np.load(a.restart/'positions.npy',mmap_mode='r');n=a.restart_frame
            restart_state=feat.create_geometry(offset,len(rest))
            restart_state.vertices().create(builtin.position,np.zeros(3));restart_state.vertices().create(builtin.velocity,np.zeros(3))
            view(restart_state.positions())[:]=history[n].reshape(-1,3,1)
            view(restart_state.vertices().find(builtin.velocity))[:]=((history[n]-history[n-1])/dt).reshape(-1,3,1)
            feat.copy_from(restart_state);positions[0]=history[n]
        libs=loaded_libraries()
        meta.update(status='running',loaded_libraries=libs,
            tiny_probe=a.tiny_probe,library_sha256={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in libs.items()});save()
        for step in range(1,steps+1):
            tick=time.perf_counter();injection_error=position_error=0.
            if lower is not None:
                n=step-1;velocity=np.zeros_like(rest) if n==0 else U@((lower[n]-lower[n-1])/dt)
                if exact_velocity and n:
                    velocity=prolong.velocity(lower[n],(lower[n]-lower[n-1])/dt)
                if uses_grandparent and n:
                    velocity=U@((lower[n]-ancestor_transfer.position(ancestor[n-1]))/dt)
                if diagonal:
                    # Native BDF1 builds x_hat from the current fine position and
                    # injected velocity. Shift its inertial center to Eq. (5)
                    # while keeping the feasible fine position for contact/CCD.
                    velocity += (prolong.position(lower[n])-positions[n])/dt
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
            if a.consistency_weight and target_frame!=step:raise RuntimeError('Consistency target was not updated')
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
            if ground_enabled and minimum_z < thickness/2-1e-9:
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
