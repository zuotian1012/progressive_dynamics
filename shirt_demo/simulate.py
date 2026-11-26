"""Run one direct or VelPro-inspired level with physical IPC gripper contact.

With a PSQ bundle, elasticity uses the shared finest proxy. Otherwise each
level has its own shell energy. VelPro changes the inertial predictor only;
cloth positions are never projected, attached, or overwritten by the robot.
"""
import argparse
import copy
import importlib.metadata
import hashlib
import json
from pathlib import Path
import time
import os

import numpy as np
from scipy.sparse import load_npz
import trimesh as tm
import uipc
from uipc import view, builtin
from uipc.constitution import (AffineBodyConstitution, SoftTransformConstraint,
    StrainLimitingBaraffWitkinShell, DiscreteShellBending, ElasticModuli2D, NeoHookeanShell)
from uipc.core import Engine, Scene, World, FiniteElementStateAccessorFeature
from uipc.geometry import trimesh as native_trimesh, label_surface

from .assets import Robot, LINKS
from .hierarchy import validate_level
from .solver_policy import converged_step, validate_parent, configure_event_solver


def trimesh(vertices, faces):
    # Native binding assumes row-major buffers; Fortran arrays otherwise scramble geometry.
    return native_trimesh(np.ascontiguousarray(vertices, dtype=np.float64),
                          np.ascontiguousarray(faces, dtype=np.int32))


def run(args):
    psq_bundle=getattr(args,'psq_bundle',None)
    last_level=validate_level(args.assets,args.level) if psq_bundle else None
    finest=bool(psq_bundle and args.level==last_level)
    relaxed=getattr(args,'relaxed_solver',False)
    disable_semi=getattr(args,'disable_semi_implicit',False)
    preview=getattr(args,'coarse_preview',False)
    event=getattr(args,'event_fallback',False)
    provenance=None
    if event:
        if (not psq_bundle or not relaxed or not disable_semi or preview
                or getattr(args,'force_updates',False) or getattr(args,'consistency_audit',False)
                or getattr(args,'strain_cap',None)!='directional' or args.runtime_package is None):
            raise ValueError('Event fallback requires explicit event runtime, relaxed PSQ, directional cap, semi-implicit OFF and no preview acceptance')
        from scripts.bunny_runtime import runtime_provenance
        provenance=runtime_provenance(args.runtime_package)
        if not (provenance.get('event_fallback_control') and provenance.get('bunny_linear_solver')=='native_pcg'
                and provenance.get('ordered_fem_assembly') and provenance.get('bunny_newton_norm')=='free_vertex_rms_3'):
            raise ValueError('Expected verified ordered PCG event-fallback runtime')
        for key in ('UIPC_PSQ_EXACT_OFFSET','UIPC_PSQ_EXACT_ONCE','UIPC_PSQ_RESCUE','UIPC_PSQ_UNRESOLVED_RESCUE'):
            if os.environ.get(key) not in (None,'0'):
                raise ValueError(f'Conflicting inherited switch: {key}')
    elif os.environ.get('BUNNY_EVENT_FALLBACK') not in (None,'0'):
        raise ValueError('Inherited event fallback requires --event-fallback')
    if not event and any(getattr(args,k,None) is not None for k in ('linear_tol_rate','newton_velocity_tol')):
        raise ValueError('Explicit solver tolerances require --event-fallback')
    reject_inf=getattr(args,'reject_inf_trials',False)
    cap_stop=int(getattr(args,'stop_after_consecutive_newton_limits',0))
    if cap_stop<0 or (cap_stop and not relaxed):
        raise ValueError('Consecutive Newton-limit guard requires relaxed solver and a nonnegative count')
    if os.environ.get('UIPC_PSQ_REJECT_INF_TRIAL'):raise ValueError('Unexpected inherited infinite-trial policy')
    if reject_inf:
        if not psq_bundle or not relaxed or getattr(args,'force_updates',False):raise ValueError('Infinite-trial rejection requires relaxed PSQ without forced updates')
        os.environ['UIPC_PSQ_REJECT_INF_TRIAL']='1'
    drive_scale=float(getattr(args,'robot_drive_scale',1.))
    if not np.isfinite(drive_scale) or drive_scale<=0:
        raise ValueError('Robot drive scale must be positive and finite')
    if preview and (not psq_bundle or finest or not relaxed or not disable_semi or (args.level>0 and args.lower is None) or getattr(args,'strain_cap',None)!='directional' or getattr(args,'consistency_audit',False)):
        raise ValueError('Preview requires a non-finest level with its immediate lower input (except L0), relaxed defaults, semi-implicit OFF, directional cap, and no consistency probes')
    if disable_semi and not relaxed:
        raise ValueError('The semi-implicit control requires --relaxed-solver')
    if relaxed and (not psq_bundle or getattr(args,'force_updates',False)):
        raise ValueError('Relaxed PSQ control requires a bundle and forbids custom forced acceptance')
    if relaxed and any(os.environ.get(k) for k in ['UIPC_PSQ_FORCE_UPDATE','UIPC_PSQ_PREVIEW_STOP','UIPC_PSQ_DIRECTION_AUDIT']):
        raise ValueError('Unexpected inherited PSQ experimental switches')
    if preview:
        if os.environ.get('UIPC_PSQ_CONSISTENCY_AUDIT'):
            raise ValueError('Coarse-preview control forbids inherited consistency probes')
        os.environ['UIPC_PSQ_PREVIEW_STOP']='1'
    if psq_bundle:
        if args.lower is not None and (args.level<1 or not relaxed or getattr(args,'force_updates',False)):
            raise ValueError('PSQ velocity transfer requires an immediate lower level without forced updates')
        os.environ['UIPC_PSQ_BUNDLE']=str(psq_bundle.resolve())
        os.environ['UIPC_PSQ_MODE']='proxy'
        if getattr(args,'force_updates',False):os.environ['UIPC_PSQ_FORCE_UPDATE']='1'
        elif os.environ.get('UIPC_PSQ_FORCE_UPDATE'):raise ValueError('Strict run forbids forced updates')
        bundle_meta=json.loads(psq_bundle.with_suffix('.json').read_text())
        validate_level(args.assets,args.level,bundle_meta.get('identity',False))
        material=bundle_meta['parameters']
        if hashlib.sha256(psq_bundle.read_bytes()).hexdigest()!=bundle_meta['sha256']:
            raise ValueError('PSQ bundle checksum mismatch')
        if preview and bundle_meta['counts']['proxy_vertices']<=bundle_meta['counts']['coarse_vertices']:
            raise ValueError('Finest/identity solve cannot use non-descent preview')
        if finest and (not bundle_meta.get('identity') or preview or not relaxed or not disable_semi or args.lower is None):
            raise ValueError('Finest level requires an identity bundle, immediate lower input, semi-implicit OFF and no preview exit')
    diagnostic_contact=getattr(args,'diagnostic_contact','full')
    diagnostic_zero_gravity=getattr(args,'diagnostic_zero_gravity',False)
    if (diagnostic_contact!='full' or diagnostic_zero_gravity) and (not psq_bundle or args.level!=0 or args.lower is not None):
        raise ValueError('Contact/gravity ablation is restricted to independent PSQ L0')
    require_convergence=event or finest or getattr(args,'require_convergence',False)
    if require_convergence and (not relaxed or not disable_semi or preview or getattr(args,'force_updates',False)):
        raise ValueError('Convergence guard requires relaxed PSQ, semi-implicit OFF, no preview and no forced acceptance')
    if event:
        # Internal preview detection triggers a retry; it never authorizes an accepted preview.
        os.environ.update(BUNNY_EVENT_FALLBACK='1',UIPC_PSQ_FROZEN_ONLY='1',
            UIPC_PSQ_PREVIEW_STOP=str(int(not finest)),UIPC_PSQ_ADAPTIVE_SEARCH='1',UIPC_PSQ_REFINE_PROBES='1')
    line_budget=getattr(args,'line_search_max_iterations',None)
    preconditioner=getattr(args,'fem_preconditioner',None)
    if preconditioner is not None and (not psq_bundle or not relaxed or preconditioner not in ('diag','mas')):
        raise ValueError('FEM preconditioner override requires a relaxed PSQ run')
    if line_budget is not None and (not psq_bundle or not relaxed or not 8<=line_budget<=128):
        raise ValueError('Explicit backtracking budget requires relaxed PSQ and a budget between 8 and 128')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    uipc.Logger.set_level(uipc.Logger.Level.Warn)
    uipc.Timer.disable_all()
    trajectory=getattr(args,'trajectory',None)
    robot = Robot(trajectory)
    if args.frames>len(robot.q):raise ValueError('Requested frames exceed robot trajectory')
    with np.load(args.assets / f"mesh_{args.level}.npz") as mesh:
        vertices, faces = mesh["vertices"], mesh["faces"]
    total = args.settle_steps + args.frames * args.substeps
    dt = 1 / (60 * args.substeps)
    lower, p = None, None
    if args.lower is not None:
        if args.level < 1:
            raise ValueError("Coarsest level cannot use a lower trajectory")
        lower_meta = json.loads((args.lower / "run.json").read_text())
        if event:
            validate_parent(lower_meta,json.loads((args.lower/'steps.json').read_text()),provenance['dll_sha256'])
        if not lower_meta["completed"] or lower_meta['completed_steps'] < total or lower_meta["dt"] != dt:
            raise ValueError("Lower trajectory is incomplete or has a different time grid")
        if lower_meta["level"] != args.level-1:
            raise ValueError("Expected the immediate lower level")
        for key in ("substeps", "settle_steps", "young", "shear_young", "bending"):
            if lower_meta["parameters"][key] != getattr(args, key):
                raise ValueError(f"Lower trajectory has different {key}")
        lower_file=args.lower/'positions.npy'
        map_file=args.assets/f'map_{args.level-1}_{args.level}.npz'
        lower = np.load(lower_file, mmap_mode="r")
        p = load_npz(map_file)
        if p.shape != (len(vertices), lower.shape[1]):
            raise ValueError("Prolongation dimensions do not match")
        if len(lower)<total+1 or not np.isfinite(lower).all():raise ValueError('Invalid lower trajectory')
        coarse_file=args.assets/f'mesh_{args.level-1}.npz'
        coarse=np.load(coarse_file)
        if lower_meta['mesh_sha256']!=hashlib.sha256(coarse_file.read_bytes()).hexdigest():raise ValueError('Lower mesh checksum mismatch')
        np.testing.assert_allclose(lower[0],coarse['vertices'],atol=1e-12,rtol=0)
        if psq_bundle:
            from progressive.prolongation import ShellProlongation
            transfer=ShellProlongation(coarse['vertices'],vertices,coarse['faces'],p,
                np.load(args.assets/f'anchors_{args.level-1}_{args.level}.npy'))
            np.testing.assert_allclose(transfer.position(lower[0]),vertices,atol=1e-12,rtol=0)
            if lower_meta.get('force_updates'):raise ValueError('Forced lower trajectory is not supported')
            with np.load(bundle_meta['source_operator']) as operator:
                np.testing.assert_array_equal(operator['coarse_faces'],faces)
                np.testing.assert_allclose(operator['coarse_rest'],vertices,atol=1e-12,rtol=0)
                if finest:
                    np.testing.assert_array_equal(operator['fine_faces'],faces)
                    np.testing.assert_array_equal(operator['fine_rest'],vertices)
                    np.testing.assert_array_equal(operator['U_indptr'],np.arange(len(vertices)+1))
                    np.testing.assert_array_equal(operator['U_indices'],np.arange(len(vertices)))
                    np.testing.assert_array_equal(operator['U_data'],np.ones(len(vertices)))
                    assert not np.any(operator['gamma']) and not np.any(operator['tangent'])
            lower_bundle=json.loads(Path(lower_meta['psq_bundle']).with_suffix('.json').read_text())
            with np.load(lower_bundle['source_operator']) as lo,np.load(bundle_meta['source_operator']) as hi:
                np.testing.assert_array_equal(lo['fine_faces'],hi['fine_faces'])
                np.testing.assert_array_equal(lo['fine_rest'],hi['fine_rest'])
        elif not np.allclose(p @ lower[0], vertices, atol=1e-10, rtol=0):
            raise ValueError('Lower initial geometry and fine mesh do not match')
    cfg = Scene.default_config()
    cfg["dt"] = dt
    cfg["gravity"] = [[0.0], [0.0], [-9.81]]
    if diagnostic_zero_gravity:cfg['gravity']=[[0.],[0.],[0.]]
    if diagnostic_contact=='off':cfg['contact']['enable']=0
    cfg["contact"]["d_hat"] = .0015
    cfg["extras"]["strict_mode"]["enable"] = 1
    cfg["newton"]["semi_implicit"]["enable"] = 0
    cfg["newton"]["velocity_tol"] = .01
    cfg["newton"]["velocity_tol_relative"] = 0.0
    cfg["newton"]["max_iter"] = 100
    cfg["line_search"]["max_iter"] = 32
    cfg["linear_system"]["tol_rate"] = .001
    if psq_bundle:
        cfg['newton'].update(velocity_tol=.001,use_adaptive_tol=0,max_iter=getattr(args,'max_iterations',100))
        cfg['linear_system']['tol_rate']=1e-4
        cfg['line_search']['report_energy']=0
        uipc.Logger.set_level(uipc.Logger.Level.Info)
    if relaxed:
        defaults=Scene.default_config()
        for key in ['newton','line_search','linear_system','extras','cfl','integrator']:
            cfg[key]=copy.deepcopy(defaults[key])
        (output/'native_default_config.json').write_text(json.dumps(defaults,indent=2))
        if disable_semi:
            cfg['newton']['semi_implicit']['enable']=0
        if line_budget is not None:
            cfg['line_search']['max_iter']=line_budget
        if preconditioner is not None:
            cfg['linear_system']['fem_preconditioner']=preconditioner
    if event:
        configure_event_solver(cfg,getattr(args,'linear_tol_rate',None),
                               getattr(args,'newton_velocity_tol',None),args.max_iterations)
    if psq_bundle and lower is not None:
        expected_lower_cfg=copy.deepcopy(cfg)
        if line_budget is not None:
            expected_lower_cfg['line_search']['max_iter']=lower_meta['config']['line_search']['max_iter']
        if preconditioner is not None:
            expected_lower_cfg['linear_system']['fem_preconditioner']=lower_meta['config']['linear_system']['fem_preconditioner']
        if lower_meta['config']!=expected_lower_cfg or lower_meta['material_parameters']!=dict(material,density=200.,poisson=.3,young=args.young):
            raise ValueError('Lower physical/solver configuration mismatch')
        if lower_meta['robot_q_sha256']!=hashlib.sha256(robot.q.tobytes()).hexdigest() or lower_meta['parameters'].get('robot_drive_scale',1.)!=drive_scale:
            raise ValueError('Lower robot trajectory/drive mismatch')
    start_build = time.perf_counter()
    engine = Engine("cuda", str(output / "backend"))
    world = World(engine)
    scene = Scene(cfg)
    contact = scene.contact_tabular()
    contact.default_model(1.0, 1e7)
    cloth_contact = contact.create("cloth")
    robot_contact = contact.create("robot")
    table_contact = contact.default_element()
    contact.insert(cloth_contact, cloth_contact, 2.0, 1e7, diagnostic_contact!='no-self')
    contact.insert(cloth_contact, table_contact, 1.0, 1e7, True)
    contact.insert(cloth_contact, robot_contact, np.sqrt(2), 1e7, True)
    contact.insert(robot_contact, table_contact, np.sqrt(2), 1e7, True)
    contact.insert(robot_contact, robot_contact, 0.0, 1e7, False)
    cloth = trimesh(vertices, faces)
    label_surface(cloth)
    # Explicit native-0.0.28 parameters, not a claim of exact 0.0.25 equivalence.
    if psq_bundle:
        NeoHookeanShell().apply_to(cloth,ElasticModuli2D.youngs_poisson(args.young,.3),200.,material['full_thickness']/2)
        DiscreteShellBending().apply_to(cloth,args.young,.3)
    else:
        StrainLimitingBaraffWitkinShell().apply_to(cloth,
        ElasticModuli2D.youngs_poisson(args.young, .49),
        ElasticModuli2D.youngs_poisson(args.shear_young, .49),
        mass_density=800, thickness=.0001, strain_rate=100.0)
        DiscreteShellBending().apply_to(cloth, args.bending)
    cloth_contact.apply_to(cloth)
    cloth_obj = scene.objects().create("shirt")
    cloth_obj.geometries().create(cloth)

    abd, drive = AffineBodyConstitution(), SoftTransformConstraint()
    box = tm.creation.box(extents=[1, 2, .8])
    table = trimesh(box.vertices + [.65, 0, .4], np.asarray(box.faces, dtype=np.int32))
    label_surface(table)
    abd.apply_to(table, 1e8, 1000)
    view(table.instances().find(builtin.is_fixed))[:] = 1
    table_contact.apply_to(table)
    scene.objects().create("table").geometries().create(table)
    targets = robot.poses(robot.q[0])
    objects, colliders = [], robot.collisions()
    robot_geometries=[]
    for name, v, f in colliders:
        geo = trimesh(v, f)
        label_surface(geo)
        abd.apply_to(geo, 1e8, 1000)
        view(geo.instances().find(builtin.is_dynamic))[:] = 0
        view(geo.transforms())[0] = targets[name]
        drive.apply_to(geo, drive_scale*np.array([300.0, 1000.0]))
        view(geo.instances().find(builtin.is_constrained))[:] = 1
        robot_contact.apply_to(geo)
        obj = scene.objects().create(name)
        robot_slot,_=obj.geometries().create(geo)
        robot_geometries.append((name,robot_slot.geometry()))
        objects.append(obj)

        def animate(info, name=name):
            mesh = info.geo_slots()[0].geometry()
            view(mesh.instances().find(builtin.aim_transform))[0] = targets[name]
        scene.animator().insert(obj, animate)
    meta = {"completed": False, "completed_steps": 0, "steps": total, "dt": dt,
            "robot_drive_strength": [300.*drive_scale,1000.*drive_scale],
            "level": args.level, "vertices": len(vertices), "faces": len(faces),
            "method": "velpro_inspired" if lower is not None else "direct",
            "grasp": "IPC contact and friction only; no cloth attachments",
            "energy": "per-level native Baraff-Witkin and discrete shell; not PSQ proxy energy",
            "pyuipc": str(uipc.__version__), "backend_path": uipc.__file__,
            "mesh_sha256": hashlib.sha256((args.assets / f"mesh_{args.level}.npz").read_bytes()).hexdigest(),
            "robot_q_sha256": hashlib.sha256(robot.q.tobytes()).hexdigest(),
            "material_parameters": {"stretch_young_pa": args.young, "shear_young_effective": args.shear_young,
                                    "poisson": .49, "density_kg_m3": 800, "collision_radius_m": .0001,
                                    "bending_raw": args.bending, "cloth_self_friction": 2,
                                    "cloth_table_friction": 1, "cloth_robot_friction": float(np.sqrt(2))},
            "config": cfg, "parameters": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
    def save_meta():
        (output / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if trajectory is not None:
        meta.update(trajectory_path=str(trajectory.resolve()),trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                    boundary_condition_experiment='Modified robot trajectory; cloth physics and PD core unchanged')
    if psq_bundle:
        meta.update(method=f'psq_proxy_L{args.level}',energy='55k shared finest PSQ NH/DSB elasticity; current-level contact/mass/strain',
                    integration='independent BDF1 at selected resolution; no lower-level velocity injection',
                    pyuipc=uipc.__version__,psq_bundle=str(psq_bundle),
                    strain_step_cap=os.environ.get('UIPC_PSQ_STRAIN_CAP','legacy'),
                    cap_audit=os.environ.get('UIPC_PSQ_CAP_AUDIT'),
                    psq_bundle_sha256=hashlib.sha256(psq_bundle.read_bytes()).hexdigest(),
                    material_parameters=dict(material,density=200.,poisson=.3,young=args.young),
                    convergence_policy='strict native max-axis tolerance; no forced updates or preview early stop')
    if diagnostic_contact!='full' or diagnostic_zero_gravity:
        meta.update(diagnostic_only=True,diagnostic_contact=diagnostic_contact,diagnostic_zero_gravity=diagnostic_zero_gravity,
                    diagnostic_caveat='Contact/gravity causal probe only; not a valid replacement for self-colliding cloth dynamics')
        if getattr(args,'force_updates',False):
            meta.update(force_updates=True,convergence_claim=False,
                        convergence_policy='Exploratory: accept capped feasible update on failed backtracking or current state at Newton limit; not strict convergence')
    if relaxed:
        meta.update(relaxed_solver=True,status='initializing',native_converged_steps=0,
                    line_search_limit_steps=0,newton_limit_steps=0,
                    convergence_policy='Native defaults including semi-implicit termination; no strict Newton convergence claim')
        meta['solver_overrides']={'newton.semi_implicit.enable':0} if disable_semi else {}
        if line_budget is not None:
            meta['solver_overrides']['line_search.max_iter']=line_budget
        if preconditioner is not None:
            meta['solver_overrides']['linear_system.fem_preconditioner']=preconditioner
        if disable_semi:
            meta['convergence_policy']='Native default displacement tolerance with semi-implicit termination disabled; no other solver override or forced acceptance'
        if preview:
            meta.update(coarse_preview=True,preview_stopped_steps=0,
                convergence_policy=f'L{args.level}: default displacement tolerance OR verified non-descent preview exit retaining current feasible iterate; preview is not Newton convergence')
    if psq_bundle and lower is not None:
        meta.update(method=f'psq_velpro_L{args.level}',
            integration='VelPro horizontal warm start: native BDF1 with lower-level velocity; preserve current-level positions',
            predictor=f'x_L{args.level}[n] + U @ (x_L{args.level-1}[n]-x_L{args.level-1}[n-1]); first velocity zero; native BDF1 adds external forces',
            lower_run=str(args.lower.resolve()),lower_sha256=hashlib.sha256(lower_file.read_bytes()).hexdigest(),
            lower_run_sha256=hashlib.sha256((args.lower/'run.json').read_bytes()).hexdigest(),
            map_sha256=hashlib.sha256(map_file.read_bytes()).hexdigest(),
            lower_newton_limit_steps=lower_meta.get('newton_limit_steps'),lower_preview_stopped_steps=lower_meta.get('preview_stopped_steps'))
    if finest:
        meta.update(finest_requires_convergence=True,preview_stopped_steps=0,
            convergence_policy=f'Finest L{args.level}: native Newton convergence required; no semi-implicit/preview/iteration-limit acceptance; reject any line-search budget exhaustion before saving a step')
    if require_convergence:
        meta['requires_convergence']=True
        if not finest:
            meta['convergence_policy']='Native convergence required; stop before saving any nonconverged, preview or budget-exhausted step; semi-implicit OFF'
    if psq_bundle:meta.update(finest_level=last_level,hierarchy_level_count=last_level+1)
    if event:
        meta.update(event_fallback_enabled=True,require_newton_convergence=True,preview_stopped_steps=0,
            event_runtime_sha256=provenance['dll_sha256'],
            gradient_strategy='Frozen until non-descent event; restore iterate and use full PSQ gradient for remainder of step; reset at next step',
            convergence_policy='Every level requires native Newton convergence; no preview, small-step or budget-exhaustion acceptance')
        for key in ('bunny_linear_solver','bunny_newton_norm','frozen_direction_derivative','representable_preview'):
            meta[key]=provenance[key]
        meta['solver_overrides'].update({'linear_system.tol_rate':cfg['linear_system']['tol_rate'],
            'newton.velocity_tol':cfg['newton']['velocity_tol'],'newton.max_iter':cfg['newton']['max_iter']})
        source_dir=output/'source';source_dir.mkdir(exist_ok=True)
        meta['source_snapshots']={}
        for source in (Path(__file__),Path(__file__).with_name('solver_policy.py'),
                       Path(__file__).parents[1]/'scripts/run_shirt_psq.py'):
            data=source.read_bytes();(source_dir/source.name).write_bytes(data)
            meta['source_snapshots'][source.name]=hashlib.sha256(data).hexdigest()
        (output/'runtime_provenance.json').write_text(json.dumps(provenance,indent=2))
    meta['reject_inf_trials']=reject_inf
    if reject_inf:meta['trial_energy_policy']='PSQ FEM +inf trial -> reject and halve alpha; initial/NaN/negative-inf/non-FEM errors remain fatal; never accept a nonfinite final trial'
    save_meta()
    world.init(scene)
    if psq_bundle:
        meta['abd_body_mapping']={name:int(view(geo.meta().find('backend_abd_body_offset'))[0]) for name,geo in robot_geometries}
    if not world.is_valid():
        raise RuntimeError("IPC scene initialization failed; inspect backend sanity diagnostics")
    feature = world.features().find(FiniteElementStateAccessorFeature)
    state = feature.create_geometry()
    state.vertices().create(builtin.position, np.zeros(3))
    state.vertices().create(builtin.velocity, np.zeros(3))
    feature.copy_to(state)
    position = lambda: view(state.positions()).reshape(-1, 3)
    if len(position()) != len(vertices):
        raise RuntimeError("Unexpected FEM vertex ordering/count")
    if lower is not None:
        velocity_state=feature.create_geometry()
        velocity_state.vertices().create(builtin.velocity,np.zeros(3))
        injected=np.lib.format.open_memmap(output/'injected_velocities.npy',mode='w+',dtype='float64',shape=(total,len(vertices),3))
    input_velocities=None
    if getattr(args,'record_input_velocities',False):
        input_velocities=np.lib.format.open_memmap(output/'input_velocities.npy',mode='w+',dtype='float64',shape=(total,len(vertices),3))
    x = np.lib.format.open_memmap(output / "positions.npy", mode="w+", dtype="float64", shape=(total+1, len(vertices), 3))
    x[0] = position()
    poses = np.lib.format.open_memmap(output / "robot.npy", mode="w+", dtype="float64", shape=(total+1, len(LINKS), 4, 4))
    np.save(output / "faces.npy", faces)
    for i, (name, v, f) in enumerate(colliders):
        np.savez_compressed(output / f"collider_{i}.npz", vertices=v, faces=f)
    times, predictor_times = [], []

    def current_poses():
        return np.stack([np.asarray(scene.geometries().find(o.geometries().ids()[0])[0].geometry().transforms().view())[0] for o in objects])

    poses[0] = current_poses()
    meta["build_seconds"] = time.perf_counter()-start_build
    if relaxed:
        from scripts.libuipc_native_runtime import loaded_libraries
        meta['loaded_libraries']=loaded_libraries()
        meta['runtime_hashes']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in meta['loaded_libraries'].values()}
        if event and meta['runtime_hashes'].get(meta['loaded_libraries']['uipc_backend_cuda.dll'])!=provenance['dll_sha256']:
            raise ValueError('Loaded CUDA DLL differs from the verified event runtime')
    meta['status']='running';save_meta()
    records=[]
    cap_streak=0
    previous_q = robot.q[0]
    try:
        for step in range(total):
            if step < args.settle_steps:
                q = robot.q[0]
            else:
                k = step-args.settle_steps
                frame, sub = divmod(k, args.substeps)
                previous_q = robot.q[max(0, frame-1)]
                q = previous_q + (sub+1)/args.substeps*(robot.q[frame]-previous_q)
            targets = robot.poses(q)
            tick = time.perf_counter()
            injection_error=position_error=0.
            if lower is not None:
                velocity = np.zeros((len(vertices), 3)) if step == 0 else p @ ((lower[step]-lower[step-1])/dt)
                # State accessor writes velocity only; current positions remain untouched.
                view(velocity_state.vertices().find(builtin.velocity))[:] = velocity.reshape(-1, 3, 1)
                feature.copy_from(velocity_state)
                feature.copy_to(state)
                injection_error=float(np.max(np.abs(view(state.vertices().find(builtin.velocity)).reshape(-1,3)-velocity)))
                position_error=float(np.max(np.abs(position()-x[step])))
                if injection_error>1e-12 or position_error>1e-12:raise RuntimeError('Velocity readback or position preservation failed')
                injected[step]=velocity;injected.flush()
            predictor_times.append(time.perf_counter()-tick)
            if input_velocities is not None:
                feature.copy_to(state)
                input_velocities[step]=view(state.vertices().find(builtin.velocity)).reshape(-1,3)
                input_velocities.flush()
            world.advance()
            world.retrieve()
            feature.copy_to(state)
            times.append(time.perf_counter()-tick)
            if not world.is_valid() or not np.isfinite(position()).all():
                raise RuntimeError(f"Invalid IPC solve at step {step}")
            if relaxed:
                native=engine.frame_stats()
                if not native.get('completed') or world.frame()!=step+1:
                    raise RuntimeError('Native frame not completed or frame mismatch')
                if require_convergence and not converged_step(native):
                    meta['failed_step']=step+1
                    (output/'failed_step.json').write_text(json.dumps(dict(step=step+1,native=native,accepted=False),indent=2))
                    raise RuntimeError(f'L{args.level} step {step+1} did not meet convergence/line-search requirements; state not saved as accepted')
                actual=current_poses()
                if not np.isfinite(actual).all():raise RuntimeError('Nonfinite robot pose')
                tracking=[float(np.linalg.norm(actual[j,:3,3]-targets[name][:3,3])) for j,name in enumerate(LINKS)]
                records.append(dict(step=step+1,native=native,body_origin_tracking_errors_m=tracking))
                if lower is not None:records[-1].update(lower_velocity_time_index=step,velocity_readback_error=injection_error,position_injection_error=position_error)
                meta['native_converged_steps']+=int(native.get('converged',False))
                meta['line_search_limit_steps']+=int(native.get('hit_line_search_limit',False))
                meta['newton_limit_steps']+=int(native.get('hit_newton_limit',False))
                if preview:
                    stopped=bool(native.get('preview_non_descent',False))
                    if stopped and (native.get('converged') or native.get('termination_reason')!='preview_non_descent' or native.get('last_line_search_alpha')!=0):
                        raise RuntimeError('Invalid preview exit status')
                    meta['preview_stopped_steps']+=int(stopped)
                (output/'steps.json').write_text(json.dumps(records,indent=2))
            x[step+1] = position()
            poses[step+1] = current_poses()
            meta["completed_steps"] = step+1
            if psq_bundle:
                x.flush();poses.flush()
                meta['physics_seconds']=sum(times)
                np.savez(output/'timing.npz',step_seconds=times,predictor_seconds=predictor_times)
                save_meta()
            if (step+1) % 60 == 0:
                print(f"{meta['method']} L{args.level}: {step+1}/{total}, solve {sum(times):.1f}s, z=[{x[step+1,:,2].min():.3f},{x[step+1,:,2].max():.3f}]", flush=True)
                save_meta()
            if cap_stop:
                cap_streak=cap_streak+1 if native.get('hit_newton_limit',False) else 0
                if cap_streak>=cap_stop:
                    meta['status']='stopped_for_diagnosis'
                    meta['stop_reason']=f'Diagnostic guard: {cap_streak} consecutive Newton-limit steps'
                    meta['consecutive_newton_limits_at_stop']=cap_streak
                    break
        else:
            meta["completed"] = True
            meta['status']='completed'
    except Exception as error:
        meta['status']='failed'
        meta["error"] = repr(error)
        raise
    finally:
        x.flush()
        poses.flush()
        np.savez(output / "timing.npz", step_seconds=times, predictor_seconds=predictor_times)
        meta["physics_seconds"] = sum(times)
        meta["physics_steps_per_second"] = len(times)/sum(times) if sum(times) else None
        meta["simulated_seconds_per_wall_second"] = len(times)*dt/sum(times) if sum(times) else None
        meta["timing_scope"] = ("velocity transfer + native readback audit + injected-velocity flush + advance + retrieve; excludes FK, other file logging, initialization and rendering"
                                if lower is not None else "predictor + state transfer + advance + retrieve; excludes FK, file logging, initialization and rendering")
        save_meta()
    print(json.dumps({k: meta[k] for k in ("completed", "level", "vertices", "faces", "physics_seconds", "physics_steps_per_second")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("output/shirt_assets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--lower", type=Path)
    parser.add_argument("--frames", type=int, default=740)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--young", type=float, default=20000)
    parser.add_argument("--shear-young", type=float, default=4)
    parser.add_argument("--bending", type=float, default=.00001)
    args = parser.parse_args()
    if not 1 <= args.frames <= 4373 or args.substeps < 1 or args.settle_steps < 0:
        parser.error("Invalid time range")
    run(args)


if __name__ == "__main__":
    main()
