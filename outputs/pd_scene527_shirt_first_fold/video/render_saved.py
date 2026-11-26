import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from tools.pd.run_square_fold import runtime_environment

root = Path.cwd()
output = (root/'outputs/pd_scene527_shirt_first_fold').resolve()
video = output/'video'
source = root/'dependencies/scene527-0.0.28/src/run_genesis_ipc.py'
for level in (sys.argv[1:] or ['L4', 'L5']):
    case = output/level if '/' in level else output/'L4_start'/level
    level = level.replace('/', '_')
    with np.load(case/'replay.npz', allow_pickle=False) as raw:
        frames = raw['source_frames']
        if not np.array_equal(frames, np.arange(801)):
            raise ValueError(f'Unexpected source frames at {case}')
        subset = {key: raw[key][::2] for key in ('source_frames','cloth_pos','robot_q','ipc_link_transforms')}
        subset['ipc_link_names'] = raw['ipc_link_names']
    sample = video/f'{level}_replay_30fps.npz'
    np.savez(sample, **subset)
    previous = json.loads((case/'command.json').read_text())
    flags = iter(previous[previous.index('--')+1:])
    replaced = {'--output','--settle-frames','--camera-view','--dump-replay-states',
                '--pd-cloth-backend-dir','--pd-cloth-proxy','--pd-cloth-state-history',
                '--pd-cloth-integrator','--pd-cloth-velpro-velocity'}
    forwarded = []
    for flag in flags:
        if flag in replaced:
            next(flags)
        elif flag not in {'--no-record','--contact-grasp-test'}:
            forwarded.append(flag)
    movie = video/f'{level}_oblique.mp4'
    command = [sys.executable,'-B',str(source),*forwarded,'--replay-states',str(sample),
               '--output',str(movie),'--camera-view','oblique','--settle-frames','0','--record-fps','30']
    env = runtime_environment()
    env['GENESIS_IPC_DETAILED_TIMERS'] = '0'
    env.setdefault('PYOPENGL_PLATFORM','egl')
    env['PYTHONUNBUFFERED'] = '1'
    with (video/f'{level}_render.log').open('w') as stream:
        stream.write(json.dumps(command)+'\n')
        stream.flush()
        subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(movie, flush=True)
