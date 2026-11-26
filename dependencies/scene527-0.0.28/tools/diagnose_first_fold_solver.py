"""Run an isolated uploaded 55k first-fold solver comparison."""
import argparse
import csv
import datetime
import difflib
import fcntl
import hashlib
import inspect
import json
import os
import re
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"reproduction/scene527_55k_73s"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def merge_timer_reports(target, source):
    """Sum consumed native Timer trees without double-counting parent/child scopes."""
    if target is None:
        target = dict(name=source['name'], parent=source.get('parent'), count=0,
                      duration=0., children=[])
    target['count'] += source.get('count', 0)
    target['duration'] += source.get('duration', 0.)
    children = {child['name']: child for child in target['children']}
    for child in source.get('children', []):
        merged = merge_timer_reports(children.get(child['name']), child)
        if child['name'] not in children:
            target['children'].append(merged)
    return target

def summarize_timers(out):
    path=out/'timers.jsonl'
    if not path.exists():
        return
    total=None
    with path.open() as stream:
        for line in stream:
            total=merge_timer_reports(total,json.loads(line)['timer'])
    if total is not None:
        total['aggregation']='sum of per-coupling native reports; report_as_json consumes each report'
        write(out/'timer_final.json',total)


def contact_model_snapshot(coupler, view):
    tabular=coupler._ipc_contact_tabular
    elements={}
    for elem in [tabular.default_element(),coupler._ipc_no_collision_contact,
                 *coupler._ipc_cloth_contacts.values(),*coupler._ipc_fem_contacts.values(),
                 *coupler._ipc_abd_contacts.values(),*coupler._ipc_ground_contacts.values()]:
        elements[int(elem.id())]=str(elem.name())
    models=tabular.contact_models()
    topology=np.asarray(view(models.find('topo'))).reshape(-1,2)
    enabled=np.asarray(view(models.find('is_enabled'))).reshape(-1)
    resistance=np.asarray(view(models.find('resistance'))).reshape(-1)
    friction=np.asarray(view(models.find('friction_rate'))).reshape(-1)
    return dict(elements=[dict(id=i,name=name) for i,name in sorted(elements.items())],
                models=[dict(ids=[int(i),int(j)],names=[elements.get(int(i)),elements.get(int(j))],
                    enabled=bool(enabled[k]),input_resistance=float(resistance[k]),friction_mu=float(friction[k]))
                    for k,(i,j) in enumerate(topology)],
                evidence='Readback of registered CPU ContactTabular after scene build; backend CUDA resistance clamping is separately recorded in logs.')

def assemble_action_replay(out):
    paths=sorted((out/'action_steps').glob('source_*.npz'))
    if not paths:
        return None
    records=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            records.append({k:data[k].copy() for k in data.files})
    frames=np.asarray([int(r['source_frame']) for r in records],dtype=np.int32)
    if not np.array_equal(frames,np.arange(frames[0],frames[0]+len(frames))):
        raise ValueError('Saved complete action frames have gaps')
    target=out/'partial_action.replay_states.npz'
    np.savez_compressed(target,source_frames=frames,
        native_frames=np.asarray([int(r['native_frame']) for r in records]),
        cloth_pos=np.stack([r['cloth_pos'] for r in records]),robot_q=np.stack([r['robot_q'] for r in records]),
        ipc_link_transforms=np.stack([r['ipc_link_transforms'] for r in records]),
        ipc_link_names=records[0]['ipc_link_names'],phase=np.asarray('trajectory'))
    return target

def assemble_settle_replay(out, physical_dt):
    """Recover durable settling samples even when native abort skips finally."""
    target=out/'settle_native.replay_states.npz'
    if target.exists():
        return target
    paths=sorted((out/'settle_steps').glob('step_*.npz'))
    action=out/'partial_action.replay_states.npz'
    if not paths:
        return None
    records=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            records.append({k:data[k].copy() for k in data.files})
    if action.exists():
        with np.load(action,allow_pickle=False) as data:
            names=data['ipc_link_names'].copy()
    elif 'ipc_link_names' in records[0]:
        names=records[0]['ipc_link_names']
    else:
        return None
    np.savez_compressed(target,source_frames=np.arange(len(records),dtype=np.int32),
        native_frames=np.asarray([int(r['native_frame']) for r in records]),
        cloth_pos=np.stack([r['cloth_pos'] for r in records]),robot_q=np.stack([r['robot_q'] for r in records]),
        ipc_link_transforms=np.stack([r['ipc_link_transforms'] for r in records]),
        ipc_link_names=names,phase=np.asarray('initial_settle'),physical_dt=float(physical_dt))
    return target

def worker(out, argv, geometry_guard_interval=0, al_mu_scale_mode=None, settle_only_steps=0,
           capture_settle_replay=False, capture_action_replay=False, solver_overrides=None):
    worker_source=Path(__file__).read_bytes()
    (out/'solver_worker_source_snapshot.py').write_bytes(worker_source)
    write(out/'solver_worker_version.json',dict(source_path=str(Path(__file__).resolve()),
          sha256=hashlib.sha256(worker_source).hexdigest()))
    requested_substeps = [int(argv[i + 1]) for i, token in enumerate(argv[:-1]) if token == '--substeps']
    requested_settle_steps = [int(argv[i + 1]) for i, token in enumerate(argv[:-1]) if token == '--settle-frames']
    settling_target_steps = settle_only_steps or (requested_settle_steps[-1] if requested_settle_steps else 0)
    last_action_substep = (requested_substeps[-1] if requested_substeps else 2) - 1
    import genesis as gs
    import uipc
    import genesis.engine.couplers.ipc_coupler.coupler as cm
    original_config = cm.build_ipc_scene_config
    physical_dt = None
    def config(*args, **kwargs):
        nonlocal physical_dt
        result = original_config(*args, **kwargs)
        physical_dt = float(result['dt'])
        if al_mu_scale_mode is not None:
            if result['contact']['constitution'] != 'al-ipc':
                raise ValueError('AL penalty initialization override requires al-ipc')
            result['contact']['al-ipc']['mu_scale_mode'] = al_mu_scale_mode
            print('AL_MU_INITIALIZATION_OVERRIDE '+al_mu_scale_mode, flush=True)
        for family,key,value in solver_overrides or []:
            target=result
            for part in family.split('/'):
                target=target[part]
            target[key]=value
            print('SOLVER_CONFIG_OVERRIDE '+json.dumps([family,key,value]),flush=True)
        write(out/'effective_ipc_config.json', result)
        return result
    cm.build_ipc_scene_config = config
    original_build=gs.Scene.build
    def build(scene,*args,**kwargs):
        result=original_build(scene,*args,**kwargs)
        records=getattr(scene.sim.coupler,'native_cloth_material_records',[])
        write(out/'contact_models.json',contact_model_snapshot(scene.sim.coupler,uipc.view))
        mapped=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                       if 'libuipc_backend_cuda.so' in line})
        write(out/'native_material.json',dict(parameterization=os.environ.get('GENESIS_IPC_CLOTH_PARAMETERIZATION','native'),
            pyuipc_version=str(uipc.__version__),backend_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in mapped},
            meshes=records))
        return result
    gs.Scene.build=build
    context = {}
    settling_records=[]
    settling_links=[]
    class SettlingComplete(Exception):
        pass
    def capture_settling(scene, local, symbols):
        nonlocal settling_links
        cloth, robot=local['cloth'],local['robot']
        coupler=cloth.sim.coupler
        frame=int(coupler._ipc_world.frame())
        if settling_records and frame==settling_records[-1]['native_frame']:
            return
        as_numpy=symbols['as_numpy']
        settling_links=list(symbols['IPC_ROBOT_LINKS'])
        merge=symbols['find_target_link_for_fixed_merge']
        row=dict(native_frame=frame,
            cloth_pos=as_numpy(cloth.get_state().pos).astype(np.float32).reshape(-1,3).copy(),
            robot_q=as_numpy(robot.get_qpos()).astype(np.float32).reshape(-1).copy(),
            ipc_link_transforms=np.stack([np.asarray(coupler._abd_data_by_link[
                merge(robot.get_link(name=name))][0].transform,dtype=np.float32).reshape(4,4)
                for name in settling_links]))
        settling_records.append(row)
        samples=out/'settle_steps';samples.mkdir(exist_ok=True)
        np.savez_compressed(samples/f'step_{len(settling_records):06d}.npz',**row,
            ipc_link_names=np.asarray(settling_links),physical_dt=physical_dt)
        write(out/'settle_progress.json',dict(completed_steps=len(settling_records),target_steps=settling_target_steps,
            native_frame=frame,trajectory_started=False,physical_seconds=len(settling_records)*physical_dt))
        if settle_only_steps and len(settling_records) in (30,settle_only_steps):
            folder=out/'settle_checkpoints';folder.mkdir(exist_ok=True)
            target=folder/f'step_{len(settling_records):06d}.pkl'
            scene.save_checkpoint(target)
            np.savez_compressed(target.with_suffix('.ipc_state.npz'),**symbols['snapshot_ipc_state'](coupler))
            coupler._ipc_world.dump()
            write(target.with_suffix('.meta.json'),dict(kind='initial_settle_only',native_frame=frame,
                settle_steps=len(settling_records),scene_t=int(scene._t),
                sim_substep_global=int(scene._sim._cur_substep_global),trajectory_started=False,
                native_workspace='/tmp/genesis_ipc_'+scene.uid.full(),
                note='Settling checkpoint; not tagged as a folding trajectory source frame'))
    original_step = gs.Scene.step
    def step(scene, *args, **kwargs):
        caller=inspect.currentframe().f_back
        local = caller.f_locals
        if 'settle_frame' in local and 'frame' not in local:
            context.update(phase='settle', source_frame=None,
                           settle_step=int(local['settle_frame']), substep=0)
        else:
            i = int(local.get('frame', -1))
            frames = local.get('source_frames')
            context.update(phase='trajectory', source_frame=int(frames[i]) if frames is not None and i >= 0 else i,
                           settle_step=None, substep=int(local.get('substep', 0)))
        print('SOLVER_STEP_BEGIN '+json.dumps(context), flush=True)
        if settle_only_steps and context['phase']!='settle':
            raise RuntimeError('Settling-only run reached a non-settling action; refusing to advance')
        try:
            result = original_step(scene, *args, **kwargs)
        except BaseException:
            if (settle_only_steps or capture_settle_replay) and context['phase']=='settle':
                capture_settling(scene,local,caller.f_globals)
            raise
        if (settle_only_steps or capture_settle_replay) and context['phase']=='settle':
            capture_settling(scene,local,caller.f_globals)
            if settle_only_steps and len(settling_records)>=settle_only_steps:
                raise SettlingComplete()
        if capture_action_replay and context['phase']=='trajectory' and context.get('source_frame',-1)>=0 and context['substep']==last_action_substep:
            symbols=caller.f_globals; as_numpy=symbols['as_numpy']; robot=local['robot']; cloth=local['cloth']
            coupler=cloth.sim.coupler; links=list(symbols['IPC_ROBOT_LINKS']); merge=symbols['find_target_link_for_fixed_merge']
            folder=out/'action_steps';folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'source_{context["source_frame"]:06d}.npz',
                source_frame=context['source_frame'],native_frame=int(coupler._ipc_world.frame()),
                cloth_pos=as_numpy(cloth.get_state().pos).astype(np.float32).reshape(-1,3).copy(),
                robot_q=as_numpy(robot.get_qpos()).astype(np.float32).reshape(-1).copy(),
                ipc_link_names=np.asarray(links),ipc_link_transforms=np.stack([np.asarray(coupler._abd_data_by_link[
                    merge(robot.get_link(name=name))][0].transform,dtype=np.float32).reshape(4,4) for name in links]))
        source_frame = context.get('source_frame')
        if geometry_guard_interval and source_frame is not None and context['substep']==last_action_substep and (
                source_frame % geometry_guard_interval == 0 or source_frame in (740,3292)):
            positions = local['cloth'].get_state().pos
            positions = positions.detach().cpu().numpy() if hasattr(positions,'detach') else np.asarray(positions)
            positions = positions.reshape(-1,3)
            if not np.isfinite(positions).all() or np.any(np.abs(positions)>5) or np.any(positions[:,2]<0):
                write(out/'geometry_guard.json',dict(source_frame=source_frame,
                      reason='nonfinite positions or coordinates beyond 5m or below world z=0',
                      finite=bool(np.isfinite(positions).all())))
                raise RuntimeError('Geometry invalid: stop this run and preserve diagnostics')
        return result
    gs.Scene.step = step
    original_couple = cm.IPCCoupler.couple
    events = (out/'step_mapping.jsonl').open('x', buffering=1)
    timers_enabled = os.environ.get('GENESIS_IPC_DETAILED_TIMERS','0') == '1'
    timers = (out/'timers.jsonl').open('x', buffering=1) if timers_enabled else None
    def couple(coupler, *args, **kwargs):
        begin = time.perf_counter()
        result = original_couple(coupler, *args, **kwargs)
        record = dict(context, native_frame=int(coupler._ipc_world.frame()),
                      coupling_wall_seconds=time.perf_counter()-begin)
        events.write(json.dumps(record)+'\n')
        if timers is not None:
            timers.write(json.dumps(dict(record, timer=uipc.Timer.report_as_json()))+'\n')
        print('SOLVER_STEP_END '+json.dumps(record), flush=True)
        return result
    cm.IPCCoupler.couple = couple
    sys.argv = [str(REPO/'src/run_genesis_ipc.py'), *argv]
    try:
        # Preserve the original __file__ so relative diagnostic helpers resolve.
        source = (out/'physics_source_snapshot.py').read_text()
        exec(compile(source, sys.argv[0], 'exec'),
             dict(__name__='__main__', __file__=sys.argv[0], __package__=None))
    except SettlingComplete:
        print('SETTLE_ONLY_COMPLETE '+str(len(settling_records)),flush=True)
    finally:
        if settling_records:
            np.savez_compressed(out/'settle_native.replay_states.npz',
                source_frames=np.arange(len(settling_records),dtype=np.int32),
                native_frames=np.asarray([r['native_frame'] for r in settling_records]),
                cloth_pos=np.stack([r['cloth_pos'] for r in settling_records]),
                robot_q=np.stack([r['robot_q'] for r in settling_records]),
                ipc_link_transforms=np.stack([r['ipc_link_transforms'] for r in settling_records]),
                ipc_link_names=np.asarray(settling_links),physical_dt=physical_dt,
                phase=np.asarray('initial_settle'))
        events.close()
        if timers is not None:
            timers.close()
            write(out/'timer_final.json', uipc.Timer.report_as_json())


def summarize(out):
    summarize_timers(out)
    rows = []
    current = None
    ansi = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
    for raw in (out/'physics.log').open(errors='replace'):
        line = ansi.sub('', raw)
        if 'SOLVER_STEP_BEGIN ' in line:
            current = dict(json.loads(line.split('SOLVER_STEP_BEGIN ', 1)[1]),
                           newton_cap=False, line_search_caps=0, iterations=None,
                           checker_calls=0, ccd_filters=0, last_residual=None, last_abs_tol=None)
        if current is None: continue
        if 'Newton Tolerance Checker' in line: current['checker_calls'] += 1
        m = re.search(r'Residual/AbsTol:\s*([\deE+.-]+)/([\deE+.-]+)', line)
        if m: current.update(last_residual=float(m[1]), last_abs_tol=float(m[2]))
        if 'CCD Filter:' in line: current['ccd_filters'] += 1
        m = re.search(r'Newton Iteration Exits with Max Iteration:\s*(\d+)', line)
        if m: current.update(newton_cap=True, iterations=int(m[1]))
        if 'Line Search Exits with Max Iteration' in line: current['line_search_caps'] += 1
        m = re.search(r'Newton Iteration Converged with Iteration Count:\s*(\d+)', line)
        if m: current['iterations'] = int(m[1])
        if 'SOLVER_STEP_END ' in line:
            current.update(json.loads(line.split('SOLVER_STEP_END ', 1)[1])); rows.append(current); current=None
    write(out/'solver_steps.json', rows)
    if rows:
        with (out/'solver_steps.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def group(records):
        values=[r['iterations'] for r in records if r['iterations'] is not None]
        return dict(steps=len(records), steps_with_final_newton_report=len(values),
                    newton_cap_steps=sum(r['newton_cap'] for r in records),
                    newton_cap_rate=sum(r['newton_cap'] for r in records)/len(records) if records else None,
                    line_search_cap_events=sum(r['line_search_caps'] for r in records),
                    iteration_index_p50_p95_max=np.percentile(values,[50,95,100]).tolist() if values else None,
                    checker_calls_p50_p95_max=np.percentile([r['checker_calls'] for r in records],[50,95,100]).tolist() if records else None,
                    coupling_wall_seconds=sum(r.get('coupling_wall_seconds',0) for r in records))
    report=dict(all=group(rows),settle=group([r for r in rows if r['phase']=='settle']),
                trajectory=group([r for r in rows if r['phase']=='trajectory']),
                newton_cap_locations=[r for r in rows if r['newton_cap']],
                line_search_cap_locations=[r for r in rows if r['line_search_caps']],
                note='Upstream convergence count is a zero-based iteration index; checker_calls gives observed solve/check attempts. Timer/debug overhead included.')
    write(out/'solver_summary.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--newton-max-iterations',type=int,default=50)
    p.add_argument('--compare-run',type=Path)
    p.add_argument('--geometry-guard-interval',type=int,default=0)
    p.add_argument('--al-mu-scale-mode',choices=('per_vertex','diag_norm'))
    p.add_argument('--settle-only-steps',type=int,default=0)
    p.add_argument('--capture-settle-replay',action='store_true')
    p.add_argument('--capture-action-replay',action='store_true')
    p.add_argument('--pcg-tol-rate',type=float)
    p.add_argument('--line-search-max',type=int)
    p.add_argument('--velocity-tol',type=float)
    p.add_argument('--al-toi-threshold',type=float)
    args, extra=p.parse_known_args();out=args.output.resolve()
    if extra and extra[0]=='--':extra=extra[1:]
    overrides=[]
    if args.velocity_tol is not None:
        if not np.isfinite(args.velocity_tol) or args.velocity_tol<=0:p.error('velocity-tol must be positive')
        overrides.append(('newton','velocity_tol',args.velocity_tol))
    if args.al_toi_threshold is not None:
        if not np.isfinite(args.al_toi_threshold) or not 0<args.al_toi_threshold<1:p.error('AL threshold must be in (0,1)')
        overrides.append(('contact/al-ipc','toi_threshold',args.al_toi_threshold))
    if args.pcg_tol_rate is not None:
        if not np.isfinite(args.pcg_tol_rate) or not 0<args.pcg_tol_rate<1:p.error('pcg tolerance must be in (0,1)')
        overrides.append(('linear_system','tol_rate',args.pcg_tol_rate))
    if args.line_search_max is not None:
        if args.line_search_max<1:p.error('line-search max must be positive')
        overrides.append(('line_search','max_iter',args.line_search_max))
    if args.worker:return worker(out,extra,args.geometry_guard_interval,args.al_mu_scale_mode,args.settle_only_steps,
                                 args.capture_settle_replay,args.capture_action_replay,overrides)
    if args.summarize_only:print(json.dumps(summarize(out)['all']));return
    assert args.newton_max_iterations > 0
    out.mkdir(parents=True,exist_ok=False)
    original_source=(REPO/'src/run_genesis_ipc.py').read_text()
    old='        newton_max_iterations = 50\n'
    assert original_source.count(old)==1
    new=f'        newton_max_iterations = {args.newton_max_iterations}\n'
    source=original_source.replace(old,new)
    (out/'physics_source_snapshot.py').write_text(source)
    (out/'newton_override.patch').write_text(''.join(difflib.unified_diff(original_source.splitlines(True),source.splitlines(True),
        fromfile='src/run_genesis_ipc.py',tofile='src/run_genesis_ipc.py')))
    profile=json.loads((BUNDLE/'bundle.json').read_text());stage=profile['stages'][0]
    preset=[s.strip() for s in (BUNDLE/profile['preset']).read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    base=['--sim1-root',str(BUNDLE),'--trajectory',str(BUNDLE/profile['trajectory']),
          '--shirt-obj',str(BUNDLE/profile['cloth']),'--robot-urdf',str(BUNDLE/profile['robot_urdf']),
          *preset,*stage['args'],'--no-record']
    physical=base+['--log-level','debug','--viewer','--output',str(out/'stage_physics.mp4'),
                   '--dump-replay-states',str(out/'stage_physics.replay_states.npz'),
                   '--save-checkpoint-source-frame','739','--save-third-fold-checkpoint',str(out/'checkpoint.pkl'),
                   '--keyframe-diagnostics-dir',str(out/'keyframes')]
    command=[sys.executable,str(Path(__file__).resolve()),'--output',str(out),'--worker','--',*physical]
    state=dict(status='preparing',source_commit='05b1d312150c4cc4c1025a2745fda0b7a5410109',
               frames=[0,739],initial_settle_steps=60,physics_parameters_changed=args.newton_max_iterations!=50,viewer=True,
               newton_max_iterations=args.newton_max_iterations, live_analysis=False,
               changed_parameters={'newton/max_iter':[50,args.newton_max_iterations]} if args.newton_max_iterations!=50 else {})
    def save():
        write(out/'state.json',state)
        links=[('收敛统计','solver_summary.json'),('逐物理步表','solver_steps.csv'),('详细物理日志','physics.log'),
               ('与50次版本对比','comparison_with_50.json'),('迭代与耗时图','solver_diagnostic.png'),('分阶段计时','timing_summary.json'),
               ('实际执行源码快照','physics_source_snapshot.py'),('单参数补丁','newton_override.patch'),
               ('实际IPC配置','effective_ipc_config.json'),('逐步计时','timers.jsonl'),('帧映射','step_mapping.jsonl'),
               ('Genesis六合一','stage_multiview.mp4'),('Isaac hero','isaac_hero/deliverables/hero.mp4'),
               ('第一折checkpoint','checkpoint.pkl'),('checkpoint元数据','checkpoint.meta.json'),
               ('配置/命令/版本','manifest.json'),('物理指标','stage_physics.json'),('TCP','stage_physics_tcp_trajectory.csv'),
               ('回放','stage_physics.replay_states.npz'),('状态','state.json'),('Genesis渲染日志','genesis_render.log'),('Isaac渲染日志','isaac_render.log')]
        lines=['# 55k第一折详细求解日志','',f'状态：`{state["status"]}`。初态静置60步，再运行0–739；Newton上限{args.newton_max_iterations}、线搜索8及原容差保持。','',
               '保留DEBUG日志、实际配置/计时与帧映射，结束后统一分析，不实时解析。计时含详细日志开销，不作为正式性能基准。','']
        lines += [f'- [{label}]({path})'+('' if (out/path).exists() else '（待生成）') for label,path in links]
        if (out/'solver_summary.json').exists():
            summary=json.loads((out/'solver_summary.json').read_text())
            lines += ['', '## 观测结果','', '```json',json.dumps({k:summary[k] for k in ('all','settle','trajectory')},indent=2),'```']
        (out/'INDEX.md').write_text('\n'.join(lines)+'\n')
    env=dict(os.environ);purelib=Path(sys.executable).parent.parent/'lib/python3.12/site-packages'
    libs=[str(purelib/'nvidia'/x) for x in ('cu13/lib','cuda_runtime/lib','cuda_nvrtc/lib','cublas/lib','cusparse/lib','cusolver/lib','nvjitlink/lib') if (purelib/'nvidia'/x).is_dir()]
    env['LD_LIBRARY_PATH']=':'.join(libs+[env.get('LD_LIBRARY_PATH','')]);env['PYTHONUNBUFFERED']='1'
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    tracked=[Path(__file__),out/'physics_source_snapshot.py',REPO/'src/run_genesis_ipc.py',ROOT/'src/genesis-world/genesis/engine/couplers/ipc_coupler/coupler.py',
             BUNDLE/'bundle.json',BUNDLE/profile['trajectory'],BUNDLE/profile['cloth'],BUNDLE/profile['robot_urdf']]
    write(out/'manifest.json',dict(state,command=command,cwd=str(REPO),input_sha256={str(x):digest(x) for x in tracked},
                                  source_patch=str(ROOT/'handoff/20261006_uploaded_only_repro_1019/run/local_changes_final.patch')))
    save()
    def execute(command,log):
        with (out/log).open('x') as f:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
            state.update(current_pid=child.pid,current_log=log);save()
            code=child.wait()
            if code:raise subprocess.CalledProcessError(code,command)
    try:
        with open(f'/tmp/genesis_ipc_shirt_demo_{os.getuid()}.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state['status']='physics_running';save();execute(command,'physics.log')
        state['status']='summarizing';save();summary=summarize(out)
        metrics=json.loads((out/'stage_physics.json').read_text())
        assert metrics['executed_frames']==740 and metrics['persistent_checkpoint_loaded_source_frame'] is None
        assert summary['all']['steps']==1540 and summary['all']['steps_with_final_newton_report']==1540,summary['all']
        assert metrics['solver']['newton_max_iterations']==args.newton_max_iterations
        assert json.loads((out/'effective_ipc_config.json').read_text())['newton']['max_iter']==args.newton_max_iterations
        plan=ROOT/'outputs/workbench/scene527_55068/first_fold/20260928_tidy_plus10_run_v2/action_plans/20260928T105938120409Z/plan.json'
        subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_solver_diagnostic_timers.py')),
                        '--run',str(out),'--plan',str(plan)],check=True)
        if args.compare_run:
            baseline=args.compare_run.resolve()
            previous=json.loads((baseline/'effective_ipc_config.json').read_text())
            actual=json.loads((out/'effective_ipc_config.json').read_text())
            old_max=previous['newton'].pop('max_iter');new_max=actual['newton'].pop('max_iter')
            assert previous==actual,'Unexpected IPC configuration differences beyond max_iter'
            def positions(a,b):
                d=np.linalg.norm(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64),axis=-1)
                return dict(p50_p95_max_mm=(np.percentile(d,[50,95,100])*1000).tolist(),rms_mm=float(np.sqrt(np.mean(d*d))*1000))
            with np.load(baseline/'stage_physics_settled_cloth.npz') as a,np.load(out/'stage_physics_settled_cloth.npz') as b:
                settled_delta=positions(a['cloth_pos'],b['cloth_pos'])
            with np.load(baseline/'stage_physics.replay_states.npz') as a,np.load(out/'stage_physics.replay_states.npz') as b:
                np.testing.assert_array_equal(a['source_frames'],b['source_frames'])
                final_delta=positions(a['cloth_pos'][-1],b['cloth_pos'][-1])
            write(out/'comparison_with_50.json',dict(baseline=str(baseline),changed_config={'newton/max_iter':[old_max,new_max]},
                baseline_solver=json.loads((baseline/'solver_summary.json').read_text()),current_solver=summary,
                settled_cloth_delta=settled_delta,final_cloth_delta=final_delta,
                limitation='One run at each limit; cloth differences also contain initial-state numerical variability, not an isolated causal estimate.'))
        print('SOLVER_SUMMARY '+json.dumps({k:summary[k] for k in ('all','settle','trajectory')}),flush=True)
        state['status']='genesis_rendering';save()
        render=base+['--replay-states',str(out/'stage_physics.replay_states.npz'),
                     '--record-multi-view','--replay-six-view','--replay-closeup-hand','left',
                     '--output',str(out/'stage.mp4'),'--dump-robot-visuals',str(out/'robot_visuals.npz')]
        execute([sys.executable,str(REPO/'src/run_genesis_ipc.py'),*render],'genesis_render.log')
        from prepare_workbench_run_render import prepare
        cfg_path=prepare(out/'stage_physics.replay_states.npz',out/'isaac_hero',camera='hero',cloth_obj=BUNDLE/profile['cloth'])
        cfg=json.loads(cfg_path.read_text());state['status']='isaac_rendering';save()
        execute([sys.executable,cfg['controller'],'run','--job',str(out/'isaac_hero')],'isaac_render.log')
        from concat_continuous_stage_videos import inspect as video_info
        assert video_info(out/'isaac_hero/deliverables/hero.mp4')==(1920,1080,60.,740)
        assert video_info(out/'stage_multiview.mp4')[3]==740
        state.update(status='complete',visual_review='pending',physics_complete=True,solver_reports_complete=True);save()
    except BaseException as e:
        state.update(status='failed_stopped',error=repr(e));save()
        if (out/'physics.log').exists():summarize(out);save()
        raise


if __name__=='__main__':main()
