"""Run the four-level Scene527 trajectory with five stage checkpoints per level."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from tools.pd.run_scene527_pdpp_first_fold import DEMO, DT, HANDOFF, ROOT, WORKER, command_for
from tools.pd.run_square_fold import runtime_environment
from tools.pd.run_integrator_benchmark import read_p

FIRST = ROOT/'outputs/pd_scene527_shirt_first_fold'
OUT = ROOT/'outputs/pd_scene527_shirt_five_stages'
BACKEND = FIRST/'proxy_diagnosis/native'
TRAJECTORY = OUT/'trajectory.npz'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def prepare(plan):
    for layer in plan['levels']:
        for key in ('mesh','elastic_target','proxy','velocity_operator'):
            ref = layer[key]
            if ref and digest(ref['path']) != ref['sha256']:
                raise ValueError(f'Input changed: {ref["path"]}')
    if not TRAJECTORY.exists():
        with np.load(HANDOFF/'controls/trajectory.npz') as ref, np.load(FIRST/'motion/trajectory.npz') as tuned:
            q=ref['joint_q'][:plan['frames']].copy()
            q[:621]=tuned['joint_q'][:621]
            np.testing.assert_array_equal(q[620],ref['joint_q'][620])
            np.savez_compressed(TRAJECTORY,joint_q=q,openness=ref['openness'][:len(q)],joint_names=ref['joint_names'])
    reference=json.loads((FIRST/'proxy_4090_chainrule/L3/native_material.json').read_text())
    binary=BACKEND/'libuipc_backend_cuda.so'
    if digest(binary)!=reference['backend_sha256'][str(binary)]:
        raise ValueError('Chain-rule backend hash mismatch')


def simulate(layer, folder, frames, settle, saves, parent=None):
    i=layer['level']
    case=folder/f'L{i}'
    if (case/'verified.json').exists():
        return case
    case.mkdir(parents=True, exist_ok=True)
    if (case/'command.json').exists():
        raise FileExistsError(f'Inspect previous attempt before reusing {case}')
    manifest=json.loads((FIRST/'hierarchy/manifest.json').read_text())
    args=SimpleNamespace(handoff=HANDOFF,output=FIRST,backend=BACKEND,stage='full')
    steps=settle+2*frames
    if parent:
        matrix=read_p(FIRST/'hierarchy','adjacent',i-1,i)
        with np.load(parent/'state.npz') as history, (case/'velpro.f64').open('wb') as f:
            v=history['v']
            assert len(v)==steps+1
            for offset in range(0,steps,16):
                block=v[offset:min(offset+16,steps)]
                projected=matrix@block.reshape(len(block),-1).T
                projected.T.astype('<f8').tofile(f)
    cmd=command_for(args,case,manifest,i,frames,settle,i<6,parent,TRAJECTORY)
    cmd.extend(['--pd-checkpoint-context',str(case/'pd_context.json')])
    ck=case/'checkpoints'
    cmd.extend(['--save-checkpoint-source-frames',*map(str,saves),'--checkpoint-output-dir',str(ck)])
    context=dict(layer=layer,trajectory_sha256=digest(TRAJECTORY),backend_sha256=digest(BACKEND/'libuipc_backend_cuda.so'),
                 parent_states=str(parent/'state.npz') if parent else None,
                 parent_velocity_sha256=digest(case/'velpro.f64') if parent else None,
                 energy_target_faces=55068,gradient='PSQ Jacobian transpose' if i<6 else 'native')
    write(case/'pd_context.json',context)
    write(case/'command.json',cmd)
    (case/'physics_source_snapshot.py').write_bytes(DEMO.read_bytes())
    env=runtime_environment()
    env['GENESIS_IPC_DETAILED_TIMERS']='0'
    with (case/'physics.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    subprocess.run([sys.executable,str(WORKER),'--output',str(case),'--summarize-only'],env=env,check=True)
    stats=json.loads((case/'solver_summary.json').read_text())['all']
    assert stats['steps']==steps
    actual=json.loads((case/'native_material.json').read_text())
    assert actual['backend_sha256'][str(BACKEND/'libuipc_backend_cuda.so')]==context['backend_sha256']
    with np.load(case/'state.npz') as states:
        for key,field in [('x','fem_position'),('v','fem_velocity')]:
            data=states[key]
            assert len(data)==steps+1 and all(np.isfinite(block).all() for block in np.array_split(data,32))
            for frame in saves:
                row=settle+2*(frame+1)
                with np.load(ck/f'source_{frame:06}.ipc_state.npz') as ipc:
                    np.testing.assert_allclose(data[row],ipc[field].reshape(-1,3),rtol=0,atol=1e-12)
            del data
    for frame in saves:
        stem=f'source_{frame:06}'
        meta=json.loads((ck/f'{stem}.meta.json').read_text())
        assert meta['source_frame']==frame and (ck/f'{stem}.pkl').stat().st_size>0
        write(ck/f'{stem}.pd.json',dict(context='../pd_context.json',context_sha256=digest(case/'pd_context.json'),
              source_frame=frame,next_source_frame=frame+1,completed_substeps=settle+2*(frame+1),
              next_parent_velocity_row=settle+2*(frame+1) if parent else None,
              parent_checkpoint=str(parent/'checkpoints'/f'{stem}.pkl') if parent else None))
    write(case/'verified.json',stats)
    return case


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','debug','run'])
    mode=parser.parse_args().mode
    plan=json.loads((OUT/'checkpoint_plan.json').read_text())
    prepare(plan)
    if mode!='prepare':
        short=mode=='debug'
        parent=None
        for layer in (plan['levels'][:2] if short else plan['levels']):
            print(mode,layer['level'],flush=True)
            parent=simulate(layer,OUT/'debug' if short else OUT,3 if short else plan['frames'],
                            2 if short else plan['settle_steps'],[0,2] if short else [s['end'] for s in plan['stages']],parent)
