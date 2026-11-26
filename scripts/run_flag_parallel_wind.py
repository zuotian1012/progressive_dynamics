"""L0 flag under prescribed +x (initially tangential) body wind load.

Strict native convergence, boundary-preserving actual decimation, no imposed
interior motion. Optional initial displacement leaves material rest unchanged.
Fine PSQ mesh is an elastic proxy only.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from progressive.prolongation import frames
from bunny_runtime import ORDERED_PACKAGE,runtime_provenance

DT=.04
STEPS=200
THICKNESS=.0004
LINEAR_TOL=1e-10
CACHED_PACKAGE=ROOT/'output/flag_L3_fix_20261118/bunny_ordered_cached_runtime/src'


def write_json(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for attempt in range(20):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    source=ROOT/'output/pdpp_flag_paper_assets_20261117_v2/mesh_9.npz'
    original=np.load(source);rest,faces=original['vertices'],original['faces']
    shutil.copy2(source,root/'original_fine.npz')
    with (root/'original_fine.obj').open('w') as fp:
        for v in rest:fp.write('v '+' '.join(format(float(x),'.17g') for x in v)+'\n')
        for f in faces:fp.write('f '+' '.join(str(int(x)+1) for x in f)+'\n')
    command=[sys.executable,'-m','progressive.prepare_intrinsic','--input',str(root/'original_fine.obj'),
        '--targets','44000','28000','17700','11100','6900','4300','2600','1600','1044',
        '--output',str(root/'assets'),'--lock-boundary','--contact-clearance',str(THICKNESS),
        '--executable',str(ROOT/'output/intrinsic_plate_build/Release/intrinsic_builder.exe')]
    with (root/'prepare.log').open('w') as fp:subprocess.run(command,cwd=ROOT,stdout=fp,stderr=subprocess.STDOUT,check=True)
    with (root/'export.log').open('w') as fp:
        subprocess.run([sys.executable,'scripts/export_psq_libuipc.py','--assets',str(root/'assets'),
            '--level','0','--fine-level','9','--output',str(root/'operator')],cwd=ROOT,stdout=fp,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,'scripts/pack_psq_native.py','--operator',str(root/'operator/operator.npz'),
            '--output',str(root/'bundle.bin'),'--thickness',str(THICKNESS),'--strain-limit','1.1','--strain-onset','1.05'],
            cwd=ROOT,stdout=fp,stderr=subprocess.STDOUT,check=True)
    from scipy.sparse import csr_matrix
    d=np.load(root/'operator/operator.npz')
    U=csr_matrix((d['U_data'],d['U_indices'],d['U_indptr']),shape=tuple(d['U_shape']))
    np.testing.assert_array_equal(d['fine_rest'],rest);np.testing.assert_array_equal(d['fine_faces'],faces)
    fixed=np.isclose(d['coarse_rest'][:,0],0,atol=1e-12)
    fine_fixed=np.isclose(rest[:,0],0,atol=1e-12)
    assert fixed.sum()==fine_fixed.sum()==140
    assert np.max(np.abs(U[fine_fixed][:,~fixed].data),initial=0)<1e-12
    residual=rest-U@d['coarse_rest']
    assert np.max(np.abs(residual))<1e-12
    write_json(root/'scene.json',dict(scope='L0 dynamics with shared 35000-vertex elastic proxy; finer levels not simulated',
        width_m=1.5,height_m=1.,initial_plane='xz',initial_velocity=0.,out_of_plane_seed=0.,
        full_thickness_m=THICKNESS,young_Pa=30000.,poisson=.3,density_kg_m3=200.,strain_limit=1.1,
        dt_s=DT,steps=STEPS,duration_s=DT*STEPS,
        wind=dict(type='Prescribed mass-proportional body load, not aerodynamic pressure',axis=[1,0,0],
            mean_acceleration_m_s2=8.,amplitude_m_s2=5.,frequency_Hz=1.3,phase_rad=0.,ramp=False,
            formula='f_i(t) = m_i * [8 + 5*sin(2*pi*1.3*t), 0, 0]; fixed-node explicit wind is zero'),
        gravity_m_s2=[0,0,-9.81],fixed='Entire original left boundary (140 vertices)',
        coarse_vertices=len(d['coarse_rest']),fine_vertices=len(rest),max_rest_residual_m=float(np.abs(residual).max()),
        source_fine_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        rest_topology_exact=True,boundary_preprocessing='Native probabilistic edge collapse with all boundary vertices retained'))
    shutil.copy2(__file__,root/'source_snapshot.py')


def simulate(root,seed=0.,wind_scale=1.,pin_mode='edge',level=0,method='direct',lower=None,elastic='proxy'):
    progressive_update=level>0 and method!='direct'
    out=root/('L0' if level==0 else f'L{level}_{method}'+('_local' if elastic=='local' else ''));out.mkdir(exist_ok=False)
    for k in list(os.environ):
        if k.startswith('UIPC_PSQ_'):del os.environ[k]
    os.environ.update(UIPC_PSQ_BUNDLE=str(root/('local_bundle.bin' if elastic=='local' else 'bundle.bin')),UIPC_PSQ_MODE='reference' if elastic=='local' else 'proxy',
        UIPC_PSQ_STRAIN_CAP='directional',UIPC_PSQ_PREVIEW_STOP='0',UIPC_PSQ_REJECT_INF_TRIAL='1',
        UIPC_PSQ_ADAPTIVE_SEARCH='1',UIPC_PSQ_REFINE_PROBES='1',UIPC_PSQ_FROZEN_ONLY='1')
    from libuipc_native_runtime import load,loaded_libraries
    native_package=CACHED_PACKAGE if level>=5 else ORDERED_PACKAGE
    native_provenance=runtime_provenance(native_package)
    uipc=load(native_package)
    from uipc import view,builtin
    from uipc.core import Engine,Scene,World,FiniteElementStateAccessorFeature
    from uipc.geometry import trimesh,label_surface
    from uipc.constitution import NeoHookeanShell,DiscreteShellBending,ElasticModuli2D,FiniteElementExternalForce
    d=np.load(root/f'assets/mesh_{level}.npz');rest,faces=d['vertices'],d['faces'].astype(np.int32)
    pins=np.isclose(rest[:,0],0,atol=1e-12)
    if pin_mode=='corners':
        pins &= np.isclose(rest[:,2],.25,atol=1e-12)|np.isclose(rest[:,2],1.25,atol=1e-12)
    assert pins.sum()==(2 if pin_mode=='corners' else 140)
    initial=rest.copy()
    initial[:,1]+=seed*(rest[:,0]/1.5)**2*np.sin(2*np.pi*(rest[:,2]-.25))
    initial[pins]=rest[pins]
    parent=transfer=None
    if progressive_update:
        from scipy.sparse import load_npz
        from progressive.prolongation import ShellProlongation
        parent_meta=json.loads((lower/'run.json').read_text())
        assert parent_meta['status']=='completed' and parent_meta['accepted_frames']>=STEPS
        assert parent_meta['level']==level-1 and parent_meta['dt']==DT
        assert parent_meta['wind_scale']==wind_scale and parent_meta['pin_mode']==pin_mode
        assert parent_meta['initial_out_of_plane_seed_m']==seed
        if level>1:assert parent_meta['method']==method
        parent=np.load(lower/'positions.npy');low=np.load(root/f'assets/mesh_{level-1}.npz')
        U=load_npz(root/f'assets/map_{level-1}_{level}.npz')
        transfer=ShellProlongation(low['vertices'],rest,low['faces'],U,np.load(root/f'assets/anchors_{level-1}_{level}.npy'))
        np.testing.assert_allclose(transfer.position(low['vertices']),rest,rtol=0,atol=1e-12)
        assert np.max(np.abs(rest-U@low['vertices']))<1e-12
        parent_root=lower.parent
        assert (root/f'assets/mesh_{level-1}.npz').read_bytes()==(parent_root/f'assets/mesh_{level-1}.npz').read_bytes()
    inv,area=frames(rest,faces)[2:]
    mass=np.zeros(len(rest));np.add.at(mass,faces.ravel(),np.repeat(area*THICKNESS*200/3,3))
    cfg=Scene.default_config();cfg['dt']=DT;cfg['gravity']=[[0.],[0.],[-9.81]]
    cfg['newton'].update(velocity_tol=.001,velocity_tol_relative=0.,use_adaptive_tol=0,max_iter=1500)
    cfg['newton']['semi_implicit']['enable']=0
    cfg['line_search'].update(max_iter=8,report_energy=0)
    cfg['linear_system'].update(tol_rate=LINEAR_TOL,fem_preconditioner='diag')
    cfg['contact']['enable']=1;cfg['contact']['d_hat']=.001;cfg['contact']['eps_velocity']=.001
    cfg['contact']['friction']['enable']=1;cfg['extras']['strict_mode']['enable']=0
    uipc.Timer.disable_all();uipc.Logger.set_level(uipc.Logger.Level.Info if elastic=='local' else uipc.Logger.Level.Warn)
    engine=Engine('cuda',str(out/'backend'));world=World(engine);scene=Scene(cfg)
    scene.contact_tabular().default_model(.3,1e9,True)
    mesh=trimesh(np.ascontiguousarray(rest),faces);label_surface(mesh)
    NeoHookeanShell().apply_to(mesh,ElasticModuli2D.youngs_poisson(30000.,.3),200.,THICKNESS/2)
    DiscreteShellBending().apply_to(mesh,30000.,.3)
    attr=mesh.vertices().find(builtin.is_fixed)
    if attr is None:attr=mesh.vertices().create(builtin.is_fixed,0)
    view(attr).reshape(-1)[:]=pins
    np.testing.assert_array_equal(view(attr).reshape(-1),pins.astype(int))
    FiniteElementExternalForce().apply_to(mesh,np.zeros(3))
    obj=scene.objects().create('flag_L0_parallel_wind');slot,_=obj.geometries().create(mesh)
    applied=[]
    def animate(info):
        frame=int(info.frame());t=frame*DT;accel=wind_scale*(8+5*np.sin(2*np.pi*1.3*t))
        force=np.zeros_like(rest);force[:,0]=mass*accel;force[pins]=0
        geo=info.geo_slots()[0].geometry()
        view(geo.vertices().find('external_force'))[:]=force.reshape(-1,3,1)
        view(geo.vertices().find(builtin.is_constrained)).reshape(-1)[:]=~pins
        applied.append(dict(frame=frame,t_s=t,acceleration_x_m_s2=float(accel),
            total_force_N=force.sum(axis=0).tolist(),max_off_axis_force_N=float(np.abs(force[:,1:]).max())))
    scene.animator().insert(obj,animate)
    states=[initial.copy()];records=[];injections=[];own_velocities=[];start=time.perf_counter()
    meta=dict(status='initializing',process_id=os.getpid(),level=level,method=method,steps_requested=STEPS,dt=DT,config=cfg,
        wind_scale=wind_scale,pin_mode=pin_mode,pin_indices=np.flatnonzero(pins).tolist(),
        coarse_trajectory_used=progressive_update,velocity_injection_each_step=progressive_update,
        pin_positions_m=rest[pins].tolist(),pole_geometry='visual only; no pole collider',
        native_package=str(native_package),runtime_sha256=native_provenance['dll_sha256'],
        linear_solver=native_provenance['bunny_linear_solver'],
        symbolic_analysis_cache=native_provenance['bunny_linear_solver']=='eigen_sparse_ldlt_cached_symbolic',
        stopping='Strict convergence required; no preview, small-step acceptance, or exact-gradient rescue',
        initial_out_of_plane_seed_m=seed,initial_max_abs_y_m=float(np.abs(initial[:,1]).max()),
        initial_seed_formula='y0=A*(x/1.5)^2*sin(2*pi*(z-0.25)/1.0)',
        material_rest_shape='original flat geometry',initial_velocity_m_s=0.,frozen_offset_gradient=True,
        dynamic_vertices=len(rest),elastic_proxy_vertices=0 if elastic=='local' else 35000,pins=int(pins.sum()),
        elastic_model='native_current_level' if elastic=='local' else 'PSQ_original_fine_proxy',
        native_psq_mode='reference' if elastic=='local' else 'proxy')
    if progressive_update:
        expected_cfg=json.loads(json.dumps(parent_meta['config']))
        expected_cfg['linear_system']['tol_rate']=LINEAR_TOL
        assert cfg==expected_cfg
        meta['parent_linear_tol_rate']=parent_meta['config']['linear_system']['tol_rate']
        meta.update(lower_run=str(lower),lower_positions_sha256=hashlib.sha256((lower/'positions.npy').read_bytes()).hexdigest(),
            transfer_sha256=hashlib.sha256((root/f'assets/map_{level-1}_{level}.npz').read_bytes()).hexdigest(),
            predictor=(f'x_L{level}[n] + U*(x_L{level-1}[n]-x_L{level-1}[n-1])' if method=='velpro' else f'P(x_L{level-1}[n]) + U*(x_L{level-1}[n]-x_L{level-1}[n-1])'),
            first_lower_velocity='zero',consistency_weight=0.,initialization='same analytic 1 mm field sampled on each level')
    shutil.copy2(__file__,out/'source_snapshot.py')
    write_json(out/'run.json',meta)
    try:
        world.init(scene)
        if not world.is_valid():raise RuntimeError('Invalid initial native scene')
        uipc.Logger.set_level(uipc.Logger.Level.Warn)
        meta['libraries']=loaded_libraries()
        accessor=world.features().find(FiniteElementStateAccessorFeature)
        offset=int(view(slot.geometry().meta().find('backend_fem_vertex_offset')).ravel()[0])
        state=accessor.create_geometry(offset,len(rest));state.vertices().create(builtin.position,np.zeros(3))
        if level:
            state.vertices().create(builtin.velocity,np.zeros(3))
        if progressive_update:
            injection=accessor.create_geometry(offset,len(rest));injection.vertices().create(builtin.velocity,np.zeros(3))
        if seed:
            initial_state=accessor.create_geometry(offset,len(rest))
            initial_state.vertices().create(builtin.position,np.zeros(3))
            initial_state.vertices().create(builtin.velocity,np.zeros(3))
            view(initial_state.positions())[:]=initial.reshape(-1,3,1)
            accessor.copy_from(initial_state)
            accessor.copy_to(initial_state)
            np.testing.assert_allclose(view(initial_state.positions()).reshape(-1,3),initial,rtol=0,atol=1e-12)
            assert np.max(np.abs(view(initial_state.vertices().find(builtin.velocity))))==0.
            meta['initial_state_readback_verified']=True
        meta['status']='running'
        write_json(out/'run.json',meta)
        for n in range(1,STEPS+1):
            if progressive_update:
                k=n-1;delta=np.zeros_like(parent[0]) if k==0 else parent[k]-parent[k-1]
                target=(states[-1] if method=='velpro' else transfer.position(parent[k]))+transfer.U@delta
                velocity=(target-states[-1])/DT;velocity[pins]=0.
                view(injection.vertices().find(builtin.velocity))[:]=velocity.reshape(-1,3,1)
                accessor.copy_from(injection);accessor.copy_to(state)
                np.testing.assert_allclose(view(state.positions()).reshape(-1,3),states[-1],rtol=0,atol=1e-12)
                np.testing.assert_allclose(view(state.vertices().find(builtin.velocity)).reshape(-1,3),velocity,rtol=0,atol=1e-12)
                injections.append(velocity.copy())
            elif level:
                accessor.copy_to(state)
                actual=np.array(view(state.vertices().find(builtin.velocity))).reshape(-1,3)
                expected=np.zeros_like(initial) if n==1 else (states[-1]-states[-2])/DT
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
                own_velocities.append(actual.copy())
            tick=time.perf_counter();world.advance();stats=engine.frame_stats()
            if (not world.is_valid() or not stats.get('completed') or not stats.get('converged')
                or stats.get('preview_non_descent') or stats.get('preview_small_step') or stats.get('hit_newton_limit')):
                meta['rejected']=dict(frame=n,native=stats);raise RuntimeError(f'Rejected native frame {n}')
            world.retrieve();accessor.copy_to(state);x=np.array(view(state.positions())).reshape(-1,3)
            tri=x[faces];F=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)@inv
            stretch=float(np.linalg.svd(F,compute_uv=False).max());pin_error=float(np.abs(x[pins]-rest[pins]).max())
            assert np.isfinite(x).all() and stretch<1.100000001 and pin_error<1e-12
            assert applied[-1]['frame']==n and applied[-1]['max_off_axis_force_N']==0.
            states.append(x.copy());records.append(dict(frame=n,t_s=n*DT,stretch=stretch,pin_error_m=pin_error,
                max_out_of_plane_m=float(np.abs(x[:,1]).max()),seconds=time.perf_counter()-tick,native=stats,wind=applied[-1]))
            write_json(out/'progress.json',dict(accepted_frames=n,elapsed_s=time.perf_counter()-start,
                step_seconds=records[-1]['seconds'],newton_iterations=stats['newton_iterations']))
            if n%10==0:
                print(f'CONVERGED {n}/{STEPS} | strain {stretch:.6f} | max |y| {np.abs(x[:,1]).max():.3e} m',flush=True)
                np.save(out/'positions.npy',states);write_json(out/'steps.json',records)
                meta.update(accepted_frames=n,elapsed_s=time.perf_counter()-start);write_json(out/'run.json',meta)
        meta['status']='completed'
    except Exception as exc:
        meta.update(status='failed',error=str(exc));raise
    finally:
        if progressive_update:np.save(out/'injected_velocities.npy',injections)
        elif level:np.save(out/'pre_step_velocities.npy',own_velocities)
        np.save(out/'positions.npy',states);write_json(out/'steps.json',records);write_json(out/'applied_wind.json',applied)
        meta.update(accepted_frames=len(states)-1,elapsed_s=time.perf_counter()-start);write_json(out/'run.json',meta)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'output/flag_L0_parallel_wind_20261121')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--reuse-assets',type=Path)
    parser.add_argument('--seed',type=float,default=0.,help='Initial out-of-plane amplitude in meters')
    parser.add_argument('--wind-scale',type=float,default=1.,help='Scale relative to baseline 8+5*sin(2*pi*1.3*t)')
    parser.add_argument('--pin-mode',choices=['edge','corners'],default='edge')
    parser.add_argument('--level',type=int,choices=range(10),default=0)
    parser.add_argument('--method',choices=['direct','semi','velpro'],default='direct')
    parser.add_argument('--lower',type=Path)
    parser.add_argument('--elastic',choices=['proxy','local'],default=None)
    parser.add_argument('--export-only',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    if args.elastic is None:args.elastic='local' if args.level and args.method=='direct' else 'proxy'
    if not np.isfinite(args.wind_scale) or args.wind_scale<=0:parser.error('--wind-scale must be finite and positive')
    if args.prepare and args.reuse_assets:parser.error('--prepare and --reuse-assets are mutually exclusive')
    if args.level and args.method!='direct' and not args.lower:parser.error('Progressive level requires --lower')
    if args.method=='direct' and args.lower:parser.error('Direct must not receive --lower')
    if args.elastic=='local' and (args.level==0 or args.method!='direct'):parser.error('Local elastic control requires Direct at L1 or above')
    if args.reuse_assets:
        source=args.reuse_assets.resolve();root.mkdir(parents=True,exist_ok=False)
        for name in (['assets'] if args.level else ['assets','operator']):shutil.copytree(source/name,root/name)
        for name in (['original_fine.npz','scene.json'] if args.level else ['bundle.bin','original_fine.npz','scene.json']):shutil.copy2(source/name,root/name)
        if args.level:
            with (root/'export.log').open('w') as log:
                subprocess.run([sys.executable,'scripts/export_psq_libuipc.py','--assets',str(root/'assets'),'--level',str(args.level),'--fine-level','9','--output',str(root/'operator')],stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,'scripts/pack_psq_native.py','--operator',str(root/'operator/operator.npz'),'--output',str(root/'bundle.bin'),'--thickness',str(THICKNESS),'--strain-limit','1.1','--strain-onset','1.05'],stdout=log,stderr=subprocess.STDOUT,check=True)
        scene=json.loads((root/'scene.json').read_text())
        scene.update(out_of_plane_seed=args.seed,initial_plane='xz reference plane with optional y perturbation',
            initial_seed_formula='y0=A*(x/1.5)^2*sin(2*pi*(z-0.25)/1.0)',
            assets_reused_from=str(source),material_rest_shape='unchanged flat original')
        write_json(root/'scene.json',scene)
        shutil.copy2(__file__,root/'source_snapshot.py')
    if args.prepare:prepare(root)
    else:
        if args.elastic=='local':
            with (root/'export_local.log').open('w') as log:
                subprocess.run([sys.executable,'scripts/export_psq_libuipc.py','--assets',str(root/'assets'),'--level',str(args.level),'--fine-level',str(args.level),'--output',str(root/'operator_local')],stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,'scripts/pack_psq_native.py','--operator',str(root/'operator_local/operator.npz'),'--output',str(root/'local_bundle.bin'),'--thickness',str(THICKNESS),'--strain-limit','1.1','--strain-onset','1.05'],stdout=log,stderr=subprocess.STDOUT,check=True)
            assert json.loads((root/'local_bundle.json').read_text())['identity']
        scene=json.loads((root/'scene.json').read_text())
        scene.update(pin_mode=args.pin_mode,fixed=('Only the two left-edge endpoints' if args.pin_mode=='corners' else 'Entire original left boundary (140 vertices)'),
            pole_geometry='visual only; no pole collider')
        scene.update(scope=f'L{args.level} dynamics with shared 35000-vertex elastic proxy',dynamic_level=args.level,
            coarse_vertices=len(np.load(root/f'assets/mesh_{args.level}.npz')['vertices']))
        scene['wind'].update(scale=args.wind_scale,mean_acceleration_m_s2=8*args.wind_scale,
            amplitude_m_s2=5*args.wind_scale,
            formula=f'f_i(t) = m_i * {args.wind_scale:g} * [8 + 5*sin(2*pi*1.3*t), 0, 0]; fixed-node explicit wind is zero')
        write_json(root/'scene.json',scene)
        if not args.export_only:simulate(root,args.seed,args.wind_scale,args.pin_mode,args.level,args.method,args.lower.resolve() if args.lower else None,args.elastic)


if __name__=='__main__':main()
