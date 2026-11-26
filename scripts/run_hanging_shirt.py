"""L0 hanging preview from the existing four-level shirt; no fine dynamics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from progressive.prolongation import ShellProlongation, frames


def acceleration(t, amplitude=3.):
    # C1 ramp, then a sinusoidal body acceleration (not an aerodynamic model).
    s = np.clip((t - 2.0) / 1.0, 0., 1.)
    ramp = s*s*(3-2*s)
    return np.array([0., amplitude*ramp*np.sin(2*np.pi*(t-2)/3.2), 0.])


def gravity_acceleration(t):
    s=np.clip(t/2.,0.,1.)
    return np.array([0.,0.,-9.81*s*s*(3-2*s)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=8.)
    parser.add_argument('--velocity-tolerance', type=float, default=.05,
                        help='Native Newton velocity tolerance; .05 matches the existing four-level demo')
    parser.add_argument('--pin-radius', type=float, default=.06)
    parser.add_argument('--wind-amplitude', type=float, default=3.)
    parser.add_argument('--semi-implicit-preview',action='store_true',help='Explicit native approximate preview, not strict Newton validation')
    parser.add_argument('--line-search-budget',type=int,default=32)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'output/shirt_four_level_pickup_20261117/operator_L0/operator.npz'
    with np.load(source) as data:
        d = dict(data)
    # Proper rigid rotation: original x -> x, original y -> z, original z -> -y.
    R = np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    origin = np.array([.667, 0., .93])
    shift = np.array([0., 0., .8])
    original = d['coarse_rest'].copy()
    for key in ('coarse_rest','fine_rest'):
        d[key] = np.ascontiguousarray((d[key]-origin)@R.T+shift)
    U = csr_matrix((d['U_data'],d['U_indices'],d['U_indptr']), shape=tuple(d['U_shape']))
    P = ShellProlongation(d['coarse_rest'],d['fine_rest'],d['coarse_faces'],U,d['anchors'])
    d.update(gamma=P.gamma,tangent=P.tangent,inv_dm=P.inv_dm)
    np.testing.assert_allclose(P.position(d['coarse_rest']),d['fine_rest'],atol=1e-12,rtol=0)
    np.savez(out/'operator.npz', **d)
    subprocess.run([sys.executable,str(ROOT/'scripts/pack_psq_native.py'),'--operator',str(out/'operator.npz'),
                    '--output',str(out/'L0.bin'),'--thickness','.0002','--strain-limit','1.1','--strain-onset','1.05'], check=True)
    rest, faces = d['coarse_rest'], d['coarse_faces']
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,count=np.unique(edges,axis=0,return_counts=True)
    boundary=np.unique(unique[count==1])
    centers=[]
    for xy in ([.48,.235],[.85,.235]):
        centers.append(boundary[np.argmin(np.linalg.norm(original[boundary,:2]-xy,axis=1))])
    pins=np.any(np.linalg.norm(rest[:,None]-rest[centers][None],axis=2)<args.pin_radius,axis=1)
    if pins.sum()<2 or pins.sum()>150:raise ValueError('Unexpected shoulder pin region')
    np.savez(out/'mesh.npz',vertices=rest,faces=faces,pins=pins,pin_centers=centers)
    os.environ['UIPC_PSQ_BUNDLE']=str(out/'L0.bin')
    os.environ['UIPC_PSQ_MODE']='proxy'
    os.environ['UIPC_PSQ_STRAIN_CAP']='directional'
    os.environ['UIPC_PSQ_PREVIEW_STOP']='1'
    os.environ['UIPC_PSQ_REJECT_INF_TRIAL']='1'
    os.environ.pop('UIPC_PSQ_FORCE_UPDATE',None)
    from libuipc_native_runtime import load, loaded_libraries
    uipc=load()
    from uipc import view,builtin
    from uipc.core import Engine,Scene,World,FiniteElementStateAccessorFeature
    from uipc.constitution import NeoHookeanShell,DiscreteShellBending,ElasticModuli2D,FiniteElementExternalForce
    from uipc.geometry import trimesh,label_surface
    uipc.Logger.set_level(uipc.Logger.Level.Info)
    uipc.Timer.disable_all()
    dt=1/120
    steps=round(args.seconds/dt)
    cfg=Scene.default_config()
    cfg['dt']=dt
    cfg['gravity']=[[0.],[0.],[0.]]  # Gravity is smoothly loaded via the force reporter.
    cfg['newton']['semi_implicit']['enable']=int(args.semi_implicit_preview)
    cfg['newton']['velocity_tol']=args.velocity_tolerance
    cfg['newton']['velocity_tol_relative']=0.
    cfg['newton']['use_adaptive_tol']=0
    cfg['newton']['max_iter']=1024
    cfg['line_search']['max_iter']=args.line_search_budget
    cfg['contact']['d_hat']=.0015
    engine=Engine('cuda',str(out/'backend'))
    world=World(engine)
    scene=Scene(cfg)
    scene.contact_tabular().default_model(2.,1e7)
    mesh=trimesh(np.ascontiguousarray(rest),np.ascontiguousarray(faces,dtype=np.int32))
    label_surface(mesh)
    NeoHookeanShell().apply_to(mesh,ElasticModuli2D.youngs_poisson(30000.,.3),200.,.0001)
    DiscreteShellBending().apply_to(mesh,30000.,.3)
    fixed=mesh.vertices().find(builtin.is_fixed)
    if fixed is None:fixed=mesh.vertices().create(builtin.is_fixed,0)
    view(fixed).reshape(-1)[:]=pins.astype(int)
    FiniteElementExternalForce().apply_to(mesh,np.zeros(3))
    area=frames(rest,faces)[3]
    mass=np.zeros(len(rest))
    np.add.at(mass,faces.ravel(),np.repeat(area*.0002*200/3,3))
    obj=scene.objects().create('hanging_shirt_L0')
    slot,_=obj.geometries().create(mesh)
    def animate(info):
        geo=info.geo_slots()[0].geometry()
        force=mass[:,None]*(acceleration(info.frame()*dt,args.wind_amplitude)+gravity_acceleration(info.frame()*dt))
        force[pins]=0
        view(geo.vertices().find('external_force'))[:]=force.reshape(-1,3,1)
        view(geo.vertices().find(builtin.is_constrained)).reshape(-1)[:]=(~pins).astype(int)
    scene.animator().insert(obj,animate)
    meta=dict(status='initializing',completed=False,completed_steps=0,steps=steps,dt=dt,
        source_operator=str(source),source_operator_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        level=0,vertices=len(rest),faces=len(faces),proxy_vertices=len(d['fine_rest']),proxy_faces=len(d['fine_faces']),
        rotation=R.tolist(),origin=origin.tolist(),translation=shift.tolist(),pin_ids=np.flatnonzero(pins).tolist(),
        pin_radius_m=args.pin_radius,
        semi_implicit_preview=args.semi_implicit_preview,
        config=cfg,wind='mass-weighted body force; acceleration y=A*smoothstep(t-2)*sin(2*pi*(t-2)/3.2), ramp t=2..3 s',
        wind_amplitude=args.wind_amplitude,
        gravity='smoothly loaded from 0 to -9.81 m/s² during t=0..2 s via external force; constant thereafter',
        method='L0 BDF1 with shared finest PSQ elastic energy; no inter-level VelPro at L0',
        preview_policy='Verified coarse non-descent stops separately counted; no iteration-limit or unverified search-budget acceptance',
        rendering_policy='L0 simulated positions only; any fine PSQ reconstruction is not fine dynamics',
        psq_rest_error_m=float(np.max(np.abs(P.position(rest)-d['fine_rest']))),
        psq_offset_max_m=float(np.max(np.linalg.norm(d['fine_rest']-U@rest,axis=1))),
        native_steps=0,preview_steps=0)
    def save(): (out/'run.json').write_text(json.dumps(meta,indent=2))
    save()
    (out/'run_hanging_shirt_snapshot.py').write_bytes(Path(__file__).read_bytes())
    records=[]
    positions=np.lib.format.open_memmap(out/'positions.npy',mode='w+',dtype='float64',shape=(steps+1,len(rest),3))
    positions[0]=rest;positions.flush()
    inv=frames(rest,faces)[2]
    tick=time.perf_counter()
    try:
        world.init(scene)
        if not world.is_valid():raise RuntimeError('Invalid initial scene')
        feat=world.features().find(FiniteElementStateAccessorFeature)
        state=feat.create_geometry()
        state.vertices().create(builtin.position,np.zeros(3))
        meta['loaded_libraries']=loaded_libraries()
        meta['runtime_sha256']={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in meta['loaded_libraries'].items()}
        meta['status']='running';save()
        uipc.Logger.set_level(uipc.Logger.Level.Warn)
        for step in range(1,steps+1):
            begin=time.perf_counter()
            world.advance();world.retrieve();feat.copy_to(state)
            stats=engine.frame_stats()
            x=np.array(view(state.positions())).reshape(-1,3)
            preview=bool(stats.get('preview_non_descent'))
            if preview and (stats.get('converged') or stats.get('termination_reason')!='preview_non_descent' or stats.get('last_line_search_alpha')!=0):
                raise RuntimeError('Invalid preview exit metadata')
            if (not world.is_valid() or not stats.get('completed') or stats.get('hit_newton_limit') or (stats.get('hit_line_search_limit') and not preview)
                or not (stats.get('converged') or preview) or not np.isfinite(x).all()):
                meta['rejected_step']=dict(step=step,stats=stats)
                raise RuntimeError(f'Unaccepted solve at step {step}')
            pin_error=float(np.max(np.abs(x[pins]-rest[pins])))
            tri=x[faces]
            F=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)@inv
            stretch=float(np.linalg.svd(F,compute_uv=False).max())
            if pin_error>1e-10 or stretch>1.1+1e-9:raise RuntimeError('Pin or strain violation')
            positions[step]=x
            meta['completed_steps']=step
            meta['native_steps']+=int(bool(stats.get('converged')))
            meta['preview_steps']+=int(preview)
            records.append(dict(step=step,time=step*dt,seconds=time.perf_counter()-begin,
                maximum_stretch=stretch,pin_error_m=pin_error,wind=acceleration(step*dt,args.wind_amplitude).tolist(),
                center=(mass@x/mass.sum()).tolist(),native=stats))
            if step%30==0 or step==steps:
                positions.flush();save();(out/'steps.json').write_text(json.dumps(records,indent=2))
                print(f'PROGRESS {step}/{steps} native={meta["native_steps"]} preview={meta["preview_steps"]} stretch={stretch:.6f}',flush=True)
        meta.update(completed=True,status='completed')
    except Exception as e:
        meta.update(status='failed',error=str(e));raise
    finally:
        positions.flush()
        meta['wall_seconds']=time.perf_counter()-tick
        save();(out/'steps.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
