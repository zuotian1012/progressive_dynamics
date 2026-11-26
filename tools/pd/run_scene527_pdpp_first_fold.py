"""Run matched first-fold experiments on a shared seven-level Scene527 shirt."""

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

from tools.pd.run_integrator_benchmark import predictor_input, read_mesh, read_p
from tools.pd.report_scene527_pdpp_first_fold import generate
from tools.pd.scene527_grasp_audit import audit_case
from tools.pd.run_square_fold import ROOT, runtime_environment
from tools.pd.run_velpro_drape import BACKEND


HANDOFF = ROOT / 'reproduction/scene527_55k_73s'
DEMO = ROOT / 'dependencies/scene527-0.0.28/src/run_genesis_ipc.py'
WORKER = DEMO.parents[1] / 'tools/diagnose_first_fold_solver.py'
ROBOT = ROOT / 'reproduction/scene527_55k_73s/assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf'
DT = 1/120
SUBSTEPS = 2


def command_for(args, case, manifest, level, frames, settle, proxy, previous, trajectory):
    template = json.loads((args.handoff/'baseline/01_first_fold/command.json').read_text())
    baseline = template[template.index('--')+1:]
    topology = manifest['levels'][level]
    override = {
        '--sim1-root': str(ROOT/'reproduction/scene527_55k_73s'),
        '--trajectory': str(trajectory), '--robot-urdf': str(ROBOT),
        '--shirt-obj': str(args.output/'hierarchy'/topology['obj']),
        '--frames': str(frames), '--settle-frames': str(settle),
        '--output': str(case/'fold.mp4'),
        '--expected-cloth-vertices': str(topology['vertices']), '--expected-cloth-faces': str(topology['triangles']),
    }
    override.update({f'--initial-shirt-{axis}': str(position)
                     for axis,position in zip('xyz',topology['scene_origin_m'],strict=True)})
    drop_with_values = {'--save-checkpoint-source-frame','--save-third-fold-checkpoint','--dump-replay-states'}
    physics = []
    tokens = iter(baseline)
    for token in tokens:
        if token in override or token in drop_with_values:
            next(tokens)
        elif token != '--viewer':
            physics.append(token)
    for key, value in override.items():
        physics.extend((key,value))
    physics.extend(('--ipc-newton-max-iterations','100','--pd-cloth-backend-dir',str(args.backend),
                    '--pd-cloth-state-history',str(case/'state.npz')))
    if proxy:
        physics.extend(('--pd-cloth-proxy',str(args.output/'hierarchy'/f'proxy_L{level}.json')))
    if previous:
        physics.extend(('--pd-cloth-integrator','velpro','--pd-cloth-velpro-velocity',str(case/'velpro.f64')))
    if frames > 15:
        physics.extend(('--dump-replay-states',str(case/'replay.npz')))
    return [sys.executable,'-B',str(WORKER),'--output',str(case),'--worker','--geometry-guard-interval','60',
            '--',*physics]


def validate(args, case, level, frames, settle, proxy, previous):
    baseline = args.handoff/'baseline/01_first_fold'
    filename = 'effective_ipc_config.json'
    expected = json.loads((baseline/filename).read_text())
    actual = json.loads((case/filename).read_text())
    assert actual == expected, f'Physical solver configuration mismatch in {case}'
    filename = 'contact_models.json'
    contact_reference = ROOT/'outputs/scene527_55k_uipc028_native_20261113_chain/01_first_fold'
    expected = json.loads((contact_reference/filename).read_text())
    actual = json.loads((case/filename).read_text())
    assert actual['models'] == expected['models'], 'Scene527 contact table changed'
    original = json.loads((baseline/'native_material.json').read_text())['meshes'][0]
    materials = json.loads((case/'native_material.json').read_text())
    actual = materials['meshes'][0]
    for key in ('stretch_E','shear_E','nu','rho','radius_m','uniform_kappa','strain_rate_input'):
        assert actual[key] == original[key], key
    for key in ('lambda','mu','strain_rate'):
        assert (actual[key]['min'],actual[key]['max']) == (original[key]['min'],original[key]['max']), (case,key)
    assert bool(actual.get('progressive_cloth_proxy_asset')) == proxy
    log = (case/'physics.log').read_text()
    if proxy:
        assert 'PD FEM elastic routing: proxies=1 native_energy_slots=0 native_gradient_blocks=0 native_hessian_blocks=0' in log
        assert 'PD proxy compute_energy: membrane_elements=55068' in log
        assert 'PD proxy material: stretch=2.5792867482563495 shear=1.342281879194631 strain_rate=100 kappa=1e-07 thickness=0.0001' in log
    else:
        assert (actual['kappa']['min'],actual['kappa']['max']) == (original['kappa']['min'],original['kappa']['max'])
    if previous:
        assert 'PD VelPro velocity input:' in log, f'No VelPro path evidence in {case}'
    rest, _ = read_mesh(args.output/'hierarchy',level)
    with np.load(case/'state.npz', allow_pickle=False) as state:
        expected = (settle + frames * SUBSTEPS + 1, len(rest), 3)
        x,v = state['x'],state['v']
        assert x.shape == v.shape == expected and state['dt'] == DT
        assert np.isfinite(x).all() and np.isfinite(v).all()
        np.testing.assert_allclose(x[0],rest,rtol=0,atol=2e-6)
    if previous:
        matrix = read_p(args.output/'hierarchy','adjacent',level-1,level)
        with np.load(previous/'state.npz',allow_pickle=False) as state:
            indices = [0, min(3, expected[0]-2), (expected[0]-1)//2, expected[0]-2]
            samples = state['v'][indices]
        saved = np.memmap(case/'velpro.f64',mode='r',dtype='<f8',shape=(expected[0]-1,len(rest)*3))
        np.testing.assert_array_equal(saved[indices], (matrix @ samples.reshape(len(indices),-1).T).T)
    summary = json.loads((case/'fold.json').read_text())
    assert summary['frames'] == frames and not summary['virtual_grasp']
    steps=json.loads((case/'solver_steps.json').read_text())
    assert len(steps) == expected[0]-1 and all(step['iterations'] is not None for step in steps)
    return dict(vertices=len(rest), frames=frames, states=expected[0], mode='velpro' if previous else 'bdf1',
                proxy=proxy, action_wall_s=summary['wall_seconds'], contact_grasp=summary['contact_grasp_summary'],
                grasp_8s=audit_case(args.output, case),
                solver=json.loads((case/'solver_summary.json').read_text())['all'])


def run(args, manifest, group, level, frames, settle, proxy, previous=None, trajectory=None):
    case = args.output/group/f'L{level}'
    accepted = case/'verified.json'
    if accepted.is_file():
        return case
    # The diagnostic worker creates its instrumentation logs exclusively.
    # Preserve an interrupted case and write the repeated launch to a fresh directory.
    if case.exists():
        case.rename(case.with_name(f'{case.name}_attempt_{time.time_ns()}'))
    case.mkdir(parents=True)
    trajectory = trajectory or args.output/'motion/trajectory.npz'
    if previous:
        matrix = read_p(args.output/'hierarchy','adjacent',level-1,level)
        predictor_input(previous/'state.npz',matrix,'velpro',settle+frames*SUBSTEPS,case/'velpro.f64',dt=DT)
    cmd = command_for(args,case,manifest,level,frames,settle,proxy,previous,trajectory)
    (case/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    shutil.copy2(DEMO,case/'physics_source_snapshot.py')
    env = runtime_environment()
    env['GENESIS_IPC_DETAILED_TIMERS'] = '0'
    for key in ('UIPC_BACKEND_PATH','UIPC_BACKEND_DIR','LD_PRELOAD','UIPC_NORMAL_LOSS_RATE_S',
                'UIPC_NORMAL_LOSS_SCOPE','UIPC_NORMAL_LOSS_FEM_CONTACT_ID','UIPC_NORMAL_LOSS_OTHER_CONTACT_ID'):
        env.pop(key,None)
    start = time.perf_counter()
    print(f'Start {group}/L{level}: {frames} controls, proxy={proxy}, velpro={previous is not None}',flush=True)
    with (case/'physics.log').open('w') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'-B',str(WORKER),'--output',str(case),'--summarize-only'],
                   env=env,stdout=subprocess.DEVNULL,check=True)
    result=validate(args,case,level,frames,settle,proxy,previous)
    result['wall_s']=time.perf_counter()-start
    accepted.write_text(json.dumps(result,indent=2)+'\n')
    print(accepted,flush=True)
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('stage',choices=('smoke','pilot','direct','scan','chain4090','chain7021'))
    parser.add_argument('--handoff',type=Path,default=HANDOFF)
    parser.add_argument('--backend',type=Path,default=BACKEND)
    args=parser.parse_args()
    args.output,args.handoff,args.backend=(p.resolve() for p in (args.output,args.handoff,args.backend))
    manifest=json.loads((args.output/'hierarchy/manifest.json').read_text())
    if args.stage == 'smoke':
        reference = args.handoff/'controls/trajectory.npz'
        coarse = run(args,manifest,'smoke',3,2,2,True,trajectory=reference)
        run(args,manifest,'smoke',4,2,2,True,coarse,reference)
        return
    motion=json.loads((args.output/'motion/trajectory_check.json').read_text())
    frames=int(motion['actions'])
    if args.stage in ('chain4090', 'chain7021'):
        start = 3 if args.stage == 'chain4090' else 4
        previous = None
        for level in range(start, 7):
            previous = run(args, manifest, f'L{start}_start', level, frames, 60, level < 6, previous)
        return
    if args.stage == 'pilot':
        assert (args.output/'smoke/L4/verified.json').is_file()
        run(args,manifest,'L4_start',4,frames,60,True)
    elif args.stage == 'direct':
        run(args,manifest,'direct',6,frames,60,False)
    else:
        assert (args.output/'smoke/L4/verified.json').is_file()
        assert (args.output/'L4_start/L4/verified.json').is_file()
        gate=hashlib.sha256((args.output/'motion/trajectory.npz').read_bytes()).hexdigest()
        (args.output/'scan_schedule.json').write_text(json.dumps(dict(trajectory_sha256=gate,
            branches=[list(range(start,7)) for start in range(5)],frames=frames),indent=2)+'\n')
        failures = []
        for start in reversed(range(5)):
            group = f'L{start}_start'
            previous=None
            try:
                for level in range(start,7):
                    previous=run(args,manifest,group,level,frames,60,level<6,previous)
            except (subprocess.CalledProcessError, AssertionError, ValueError) as error:
                failures.append(f'{group}/L{level}: {error}')
                print(failures[-1],flush=True)
            if start == 4:
                try:
                    run(args,manifest,'direct',6,frames,60,False)
                except (subprocess.CalledProcessError, AssertionError, ValueError) as error:
                    failures.append(f'direct/L6: {error}')
                    print(failures[-1],flush=True)
            generate(args.output)
        if failures:
            raise SystemExit('\n'.join(failures))


if __name__ == '__main__':
    main()
