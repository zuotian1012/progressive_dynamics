"""Independent single-level shirt settling in the unmodified installed pyuipc wheel.

No PSQ runtime loader, bundle, hierarchy map, coarse trajectory or velocity write.
"""
import argparse
import base64
import copy
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import trimesh as tm
import uipc
from uipc import view,builtin
from uipc.core import Engine,Scene,World,FiniteElementStateAccessorFeature
from uipc.constitution import NeoHookeanShell,DiscreteShellBending,ElasticModuli2D,AffineBodyConstitution,SoftTransformConstraint
from uipc.geometry import trimesh as native_trimesh,label_surface
from shirt_demo.assets import Robot,LINKS


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mesh(v,f):return native_trimesh(np.ascontiguousarray(v,dtype=np.float64),np.ascontiguousarray(f,dtype=np.int32))


def verify_wheel():
    distribution=importlib.metadata.distribution('pyuipc');package=Path(distribution.locate_file('uipc')).resolve()
    assert Path(uipc.__file__).resolve()==package/'__init__.py'
    assert Path(uipc.pyuipc.__file__).resolve().is_relative_to(package/'_native')
    records={}
    for f in distribution.files:
        if str(f).startswith('uipc/') and (str(f).endswith(('.dll','.pyd')) or str(f)=='uipc/__init__.py'):
            p=Path(distribution.locate_file(f)).resolve();raw=hashlib.sha256(p.read_bytes()).digest()
            ok=f.hash is not None and f.hash.mode=='sha256' and f.hash.value==base64.urlsafe_b64encode(raw).decode().rstrip('=')
            if not ok:raise RuntimeError(f'Installed wheel RECORD hash mismatch: {p}')
            records[str(p)]=raw.hex()
    return package,dict(distribution_version=distribution.version,module_version=uipc.__version__,package_path=str(package),all_record_hashes_match=True,files=records)


def loaded_libraries(package):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetModuleHandleW.argtypes=[ctypes.c_wchar_p];kernel.GetModuleHandleW.restype=ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint]
    result={}
    for name in ['uipc_core.dll','uipc_geometry.dll','uipc_constitution.dll','uipc_backend_cuda.dll']:
        handle=kernel.GetModuleHandleW(name)
        if not handle:raise RuntimeError(f'Not loaded: {name}')
        buf=ctypes.create_unicode_buffer(32768)
        if not kernel.GetModuleFileNameW(handle,buf,len(buf)):raise ctypes.WinError(ctypes.get_last_error())
        path=Path(buf.value).resolve()
        if not path.is_relative_to(package/'_native'):raise RuntimeError(f'Non-stock DLL: {path}')
        result[name]=str(path)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path('output/shirt55k_psq_ready'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--level',type=int,choices=[0,1,2],default=1)
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--solver',choices=['matched-semi-off','native-default'],default='matched-semi-off')
    a=p.parse_args()
    if not 1<=a.steps<=60:raise ValueError('This experiment covers the original 60-step settling segment only')
    inherited={k:v for k,v in os.environ.items() if k.startswith('UIPC_PSQ')}
    if inherited:raise RuntimeError(f'PSQ environment is forbidden: {list(inherited)}')
    if any(n.startswith('progressive') or n.endswith('libuipc_native_runtime') for n in sys.modules):
        raise RuntimeError('A progressive/isolated runtime module was imported')
    package,wheel=verify_wheel();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'wheel_integrity.json').write_text(json.dumps(wheel,indent=2))
    uipc.Logger.set_level(uipc.Logger.Level.Info);uipc.Timer.disable_all()
    source=a.assets/f'mesh_{a.level}.npz'
    with np.load(source) as z:vertices=z['vertices'].copy();faces=z['faces'].copy()
    robot=Robot();dt=1/120;defaults=Scene.default_config();cfg=copy.deepcopy(defaults)
    cfg['dt']=dt;cfg['gravity']=[[0.],[0.],[-9.81]];cfg['contact']['d_hat']=.0015
    overrides={}
    if a.solver=='matched-semi-off':cfg['newton']['semi_implicit']['enable']=0;overrides['newton.semi_implicit.enable']=0
    (out/'native_default_config.json').write_text(json.dumps(defaults,indent=2))
    engine=Engine('cuda',str(out/'backend'));world=World(engine);scene=Scene(cfg)
    contact=scene.contact_tabular();contact.default_model(1.,1e7)
    cloth_contact=contact.create('cloth');robot_contact=contact.create('robot');table_contact=contact.default_element()
    contact.insert(cloth_contact,cloth_contact,2.,1e7,True);contact.insert(cloth_contact,table_contact,1.,1e7,True)
    contact.insert(cloth_contact,robot_contact,np.sqrt(2),1e7,True);contact.insert(robot_contact,table_contact,np.sqrt(2),1e7,True)
    contact.insert(robot_contact,robot_contact,0.,1e7,False)
    cloth=mesh(vertices,faces);label_surface(cloth)
    NeoHookeanShell().apply_to(cloth,ElasticModuli2D.youngs_poisson(30000.,.3),200.,.0001)
    DiscreteShellBending().apply_to(cloth,30000.,.3);cloth_contact.apply_to(cloth)
    scene.objects().create(f'independent_cloth_L{a.level}').geometries().create(cloth)
    abd=AffineBodyConstitution();drive=SoftTransformConstraint();box=tm.creation.box(extents=[1,2,.8])
    table=mesh(box.vertices+[.65,0,.4],box.faces);label_surface(table);abd.apply_to(table,1e8,1000)
    view(table.instances().find(builtin.is_fixed))[:]=1;table_contact.apply_to(table);scene.objects().create('table').geometries().create(table)
    targets=robot.poses(robot.q[0]);objects=[];colliders=robot.collisions()
    for i,(name,v,f) in enumerate(colliders):
        geo=mesh(v,f);label_surface(geo);abd.apply_to(geo,1e8,1000)
        view(geo.instances().find(builtin.is_dynamic))[:]=0;view(geo.transforms())[0]=targets[name]
        drive.apply_to(geo,np.array([300.,1000.]));view(geo.instances().find(builtin.is_constrained))[:]=1
        robot_contact.apply_to(geo);obj=scene.objects().create(name);obj.geometries().create(geo);objects.append(obj)
        def animate(info,name=name):view(info.geo_slots()[0].geometry().instances().find(builtin.aim_transform))[:]=targets[name]
        scene.animator().insert(obj,animate)
        np.savez_compressed(out/f'collider_{i}.npz',vertices=v,faces=f)
    np.save(out/'faces.npy',faces)
    meta=dict(status='initializing',completed=False,completed_steps=0,steps=a.steps,level=a.level,vertices=len(vertices),faces=len(faces),dt=dt,
        method='stock_libuipc_independent',package_version=uipc.__version__,backend_path=uipc.__file__,stock_wheel=True,
        psq_enabled=False,pd_enabled=False,velocity_injection=False,hard_strain_barrier=False,solver_mode=a.solver,solver_overrides=overrides,
        config=cfg,parameters=dict(assets=str(a.assets),level=a.level,settle_steps=60,frames=0,substeps=2),
        material_parameters=dict(young=30000.,poisson=.3,density=200.,full_thickness=.0002,stored_radius=.0001,
            bending=30000.*.0002**3/(12*(1-.3**2)),membrane='native current-level NeoHookeanShell',bending_model='native current-level DiscreteShellBending'),
        mesh_sha256=digest(source),robot_q_sha256=hashlib.sha256(robot.q.tobytes()).hexdigest(),robot_drive_strength=[300.,1000.],
        native_converged_steps=0,line_search_limit_steps=0,newton_limit_steps=0,preview_stopped_steps=0,
        convergence_interpretation='Native default acceptance; all frame flags retained. Default semi-implicit may terminate early only in native-default control.',
        model_differences_vs_pd=['Current-level native elasticity instead of finest PSQ proxy.',
            'No custom C-IPC strain barrier or PSQ directional cap; original installed backend only.',
            'Installed pyuipc 0.0.28 rather than isolated modified source runtime.'],
        loaded_modules_forbidden_checked=True,physics_seconds=0.)
    def save(): (out/'run.json').write_text(json.dumps(meta,indent=2))
    save();start=time.perf_counter();rows=[];states=[];poses=[];velocities=[];timings=[]
    try:
        world.init(scene)
        if not world.is_valid():raise RuntimeError('Stock scene init invalid')
        loaded=loaded_libraries(package);meta['loaded_libraries']=loaded;meta['runtime_hashes']={path:digest(path) for path in loaded.values()}
        assert all(wheel['files'][path]==h for path,h in meta['runtime_hashes'].items())
        feature=world.features().find(FiniteElementStateAccessorFeature);state=feature.create_geometry()
        state.vertices().create(builtin.position,np.zeros(3));state.vertices().create(builtin.velocity,np.zeros(3));feature.copy_to(state)
        def position():return view(state.positions()).reshape(-1,3).copy()
        def body_poses():return np.stack([np.asarray(scene.geometries().find(o.geometries().ids()[0])[0].geometry().transforms().view())[0] for o in objects])
        np.testing.assert_array_equal(position(),vertices);states.append(position());poses.append(body_poses())
        meta['build_seconds']=time.perf_counter()-start;meta['status']='running';save()
        np.save(out/'positions.npy',np.asarray(states));np.save(out/'robot.npy',np.asarray(poses))
        for step in range(a.steps):
            # Read only: direct native BDF1 keeps its own velocity. No copy_from calls.
            feature.copy_to(state);v=view(state.vertices().find(builtin.velocity)).reshape(-1,3).copy()
            expected=np.zeros_like(vertices) if step==0 else (states[-1]-states[-2])/dt
            velocity_error=float(np.max(np.abs(v-expected)))
            if velocity_error>1e-12:raise RuntimeError('Native velocity disagrees with direct BDF1 history')
            velocities.append(v);tick=time.perf_counter();world.advance();world.retrieve();feature.copy_to(state);timings.append(time.perf_counter()-tick)
            native=engine.frame_stats();x=position();rp=body_poses()
            if not world.is_valid() or world.frame()!=step+1 or not native.get('completed') or not np.isfinite(x).all() or not np.isfinite(rp).all():
                raise RuntimeError('Invalid stock frame')
            if native.get('preview_non_descent'):raise RuntimeError('Unexpected preview in stock runtime')
            states.append(x);poses.append(rp)
            rows.append(dict(step=step+1,native=native,input_velocity_formula_error=velocity_error,
                body_origin_tracking_errors_m=[float(np.linalg.norm(rp[j,:3,3]-targets[name][:3,3])) for j,name in enumerate(LINKS)]))
            for target,field in [('native_converged_steps','converged'),('line_search_limit_steps','hit_line_search_limit'),('newton_limit_steps','hit_newton_limit')]:meta[target]+=int(native.get(field,False))
            meta['completed_steps']=step+1;meta['physics_seconds']=sum(timings)
            np.save(out/'positions.npy',np.asarray(states));np.save(out/'robot.npy',np.asarray(poses));np.save(out/'input_velocities.npy',np.asarray(velocities))
            (out/'steps.json').write_text(json.dumps(rows,indent=2));save()
            if (step+1)%10==0:print(f'Stock L{a.level} {a.solver}: {step+1}/{a.steps}',flush=True)
        meta.update(status='completed',completed=True)
    except Exception as error:
        meta.update(status='failed',error=repr(error),failed_step=meta['completed_steps']+1);raise
    finally:
        meta['physics_seconds']=sum(timings);save();np.savez(out/'timing.npz',step_seconds=timings)
    print(json.dumps({k:meta[k] for k in ['status','completed_steps','native_converged_steps','line_search_limit_steps','newton_limit_steps','physics_seconds']},indent=2))


if __name__=='__main__':main()
