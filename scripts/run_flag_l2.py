"""Isolated flag L2 continuation; never selects a new shirt backend.

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
from paper_solver_settings import (
    NEWTON_VELOCITY_TOL, LINEAR_TOL_RATE, LINE_SEARCH_MAX_ITER,
    validate_shirt_solver_history,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restart',type=Path)
    p.add_argument('--restart-frame',type=int,default=205)
    p.add_argument('--direction-audit',action='store_true')
    p.add_argument('--exact-offset',action='store_true')
    p.add_argument('--rescue',action='store_true')
    p.add_argument('--refined-probe',action='store_true')
    p.add_argument('--dt',type=float,default=.04)
    p.add_argument('--scene',choices=['flag','bunny'],required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int)
    p.add_argument('--level',type=int,default=0)
    p.add_argument('--lower',type=Path)
    p.add_argument('--method',choices=['direct','velpro'],default='velpro')
    p.add_argument('--tiny-probe',action='store_true',help='Legacy option; rejected by shirt-aligned stopping')
    p.add_argument('--preview-min-alpha',type=float,default=0.,help='Legacy compatibility option; must be 0 with shirt-aligned stopping')
    p.add_argument('--max-newton',type=int,default=300,help='Iteration budget only; does not relax acceptance tolerance')
    p.add_argument('--preconditioner',choices=['diag','mas'],default='diag')
    p.add_argument('--native-package',type=Path,help='Explicit isolated runtime package; default remains the pinned baseline build')
    a=p.parse_args();out=a.output.resolve()
    if a.scene!='flag' or a.level!=2 or a.preconditioner!='diag':
        p.error('This isolated entry supports flag L2 with diagonal PCG')
    if not a.lower or not a.rescue or not a.refined_probe or a.restart:
        p.error('L2 requires completed L1, --rescue, --refined-probe, and no restart')
    if a.exact_offset and a.rescue:p.error('Select either full exact-gradient control or first-iteration rescue')
    if a.rescue and a.direction_audit:p.error('Direction-only diagnosis disables preview; incompatible with rescue')
    if a.dt<=0 or (a.steps is not None and a.steps<1):p.error('Require positive dt and step count')
    if a.exact_offset or a.rescue:
        variant='rescue' if a.rescue else 'exact'
        provenance=ROOT/'output/flag_L1_fix_20261118'/f'{variant}_build/provenance.json'
        if not a.native_package or not provenance.is_file():p.error('Repair requires an explicitly built isolated runtime')
        expected=json.loads(provenance.read_text())['dll_sha256']
        dll=a.native_package/'uipc/_native/uipc_backend_cuda.dll'
        if not dll.is_file() or hashlib.sha256(dll.read_bytes()).hexdigest()!=expected:
            p.error('Explicit runtime does not match this repair build provenance')
    if a.tiny_probe or a.preview_min_alpha!=0:
        p.error('Shirt-aligned stopping requires --preview-min-alpha 0 and no --tiny-probe')
    if a.max_newton<1:p.error('Require positive Newton iteration budget')
    flag=a.scene=='flag';last=9 if flag else 7;thickness=.0004 if flag else .00007
    if not 0<=a.level<=last:p.error('Level outside the paper hierarchy')
    if (a.level>0 and a.method=='velpro') != (a.lower is not None):p.error('VelPro requires exactly the preceding level')
    if a.lower:
        lm=json.loads((a.lower/'run.json').read_text())
        validate_shirt_solver_history(lm)
        if (lm.get('non_descent_probe_refinement')!=a.refined_probe or lm.get('initial_non_descent_rescue')!=a.rescue or lm.get('exact_offset_gradient')!=a.exact_offset
            or lm['library_sha256']['uipc_backend_cuda.dll']!=expected):
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
    if a.refined_probe:os.environ['UIPC_PSQ_REFINE_PROBES']='1'
    if a.rescue:os.environ['UIPC_PSQ_RESCUE']='1'
    if a.exact_offset:os.environ['UIPC_PSQ_EXACT_OFFSET']='1'
    if a.direction_audit:
        os.environ['UIPC_PSQ_PREVIEW_STOP']='0'
        os.environ['UIPC_PSQ_DIRECTION_AUDIT']='1'
    from libuipc_native_runtime import load,loaded_libraries
    uipc=load(a.native_package)
    from uipc import view,builtin
    from uipc.core import Engine,Scene,World,FiniteElementStateAccessorFeature
    from uipc.geometry import trimesh,label_surface
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
    cfg['newton'].update(velocity_tol=NEWTON_VELOCITY_TOL,velocity_tol_relative=0.,use_adaptive_tol=0,max_iter=a.max_newton)
    cfg['newton']['semi_implicit']['enable']=0
    cfg['line_search'].update(max_iter=LINE_SEARCH_MAX_ITER,report_energy=0)
    cfg['linear_system']['tol_rate']=LINEAR_TOL_RATE
    cfg['linear_system']['fem_preconditioner']=a.preconditioner
    cfg['contact']['d_hat']=.001;cfg['contact']['eps_velocity']=.001
    cfg['extras']['strict_mode']['enable']=0
    # Retain the three-scale non-descent evidence in coarse-level logs, as in the shirt demo.
    uipc.Timer.disable_all();uipc.Logger.set_level(uipc.Logger.Level.Info if a.level<last else uipc.Logger.Level.Warn)
    engine=Engine('cuda',str(out/'backend'));world=World(engine);scene=Scene(cfg)
    scene.contact_tabular().default_model(.3,1e9)
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
        tab=scene.contact_tabular();oc=tab.create('fixed_bunny');tab.insert(oc,oc,.3,1e9,False);oc.apply_to(ob)
        bunny_slot,_=scene.objects().create('fixed_bunny').geometries().create(ob)
    meta=dict(scene=a.scene,assets=str(a.assets.resolve()),level=a.level,method='direct' if lower is None else 'velpro',completed=False,status='initializing',steps=steps,completed_steps=0,
        dt=dt,vertices=len(rest),faces=len(faces),proxy_vertices=35000 if flag else 90000,config=cfg,pins=int(pins.sum()),
        native_steps=0,preview_steps=0,material=dict(young=30000.,poisson=.3,density=200.,thickness=thickness,strain_limit=limit),
        scope=f'Single-level L{a.level} trajectory at paper endpoint scale; no finer-level trajectory implied',
        paper_matched=['10 levels, ~550 to 35000 vertices, thickness .4mm, strain limit 10%, final duration target 20s'] if flag else ['8 levels, ~1500 to 90000 vertices, thickness .07mm, strain limit 6.8%, friction .3, dt .01s'],
        reconstruction_choices=['E=30000, nu=.3, rho=200; source assets not authenticated author meshes',
            'native max-axis velocity increment .05 matches shirt demo, not author RMS; native adaptive contact, fixed strain barrier stiffness',
            'flag: 1.5m x 1m, left edge fixed, body acceleration y=8+5*sin(2*pi*1.3*t), dt=.04 inferred from PD2024' if flag else f'bunny: previously recorded local geometry and {steps*dt:g}s local observation window; author duration not recovered'],
        stop_policy='Shirt-aligned: no semi-implicit or forced updates; coarse verified non-descent only; no small-alpha stop or tiny-alpha probe; finest requires native convergence; reject budget exhaustion without verified preview',
        source_files_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.assets/f'mesh_{a.level}.npz',a.assets/f'mesh_{last}.npz',Path(__file__),Path(__file__).with_name('paper_solver_settings.py')]})
    meta.update(preview_min_alpha=a.preview_min_alpha,small_step_preview_steps=0,
        native_package=str(a.native_package.resolve()) if a.native_package else None)
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
    meta['exact_offset_gradient']=a.exact_offset
    meta['initial_non_descent_rescue']=a.rescue
    meta['non_descent_probe_refinement']=a.refined_probe
    meta['probe_policy']='Up to three decade refinements of resolved positive slopes; unchanged noise and 25% consistency limits'
    meta['lower_backend_sha256']=lm['library_sha256']['uipc_backend_cuda.dll'] if a.lower else None
    meta['algorithm_note']=('One exact PSQ gradient retry after verified non-descent at first Newton iteration; explicit repair extension'
        if a.rescue else 'Full nonlinear PSQ Jacobian gradient; separate repair extension, not frozen-offset paper approximation'
        if a.exact_offset else 'Original frozen-offset gradient')
    meta['time_discretization_note']=f'Actual integration dt={dt:.17g}; original flag dt=0.04. No hidden substeps.'
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
                velocity[pins]=0
                view(velocity_state.vertices().find(builtin.velocity))[:]=velocity.reshape(-1,3,1)
                feat.copy_from(velocity_state);feat.copy_to(check_state)
                injection_error=float(np.max(np.abs(view(check_state.vertices().find(builtin.velocity)).reshape(-1,3)-velocity)))
                position_error=float(np.max(np.abs(view(check_state.positions()).reshape(-1,3)-positions[n])))
                if max(injection_error,position_error)>1e-12:raise RuntimeError('VelPro velocity injection or position preservation failed')
            world.advance();world.retrieve();feat.copy_to(state)
            x=np.array(view(state.positions())).reshape(-1,3)
            stats=engine.frame_stats();small=bool(stats.get('preview_small_step'));preview=bool(stats.get('preview_non_descent')) or small
            valid_preview=(a.level<last and bool(stats.get('preview_non_descent')) and not small and not stats.get('converged') and
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
            positions[step]=x;positions.flush()
            meta['completed_steps']=step;meta['native_steps']+=int(bool(stats.get('converged')));meta['preview_steps']+=int(preview)
            meta['small_step_preview_steps']+=int(small)
            records.append(dict(step=step,native=stats,maximum_stretch=stretch,seconds=time.perf_counter()-tick,
                velocity_readback_error=injection_error,position_injection_error=position_error,obstacle_error=obstacle_error));save()
            (out/'steps.json').write_text(json.dumps(records,indent=2))
            if step%10==0:print(f'ACCEPTED {step}/{steps}, native={meta["native_steps"]}, preview={meta["preview_steps"]}',flush=True)
        meta.update(status='completed',completed=True)
    except Exception as e:meta.update(status='failed',error=str(e));raise
    finally:
        positions.flush();meta['wall_seconds']=time.perf_counter()-start;save();(out/'steps.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
